"""Host-side prefix-reuse scoping, resets and accounting with synthetic workers."""

import http.client
import io
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from stark_convert.chat import prefix_cache_enabled
from stark_convert.desktop import Desktop, valid_reuse
from stark_convert.generation_eta import GenerationEstimates, progress_frame
from stark_convert.local_api import APIError, LocalAPI

SESSION = "a" * 32
READY = {
    "ready": True,
    "mode": "text-generation-v1",
    "operations": ["tokenize", "decode", "generate", "reset_cache"],
    "state_policy": "consecutive-prefix-v1",
    "prefix_cache": {"supported": True, "enabled": True, "chunk_tokens": 1024, "idle_seconds": 600},
    "max_context_tokens": 8192,
    "max_new_tokens": 4096,
    "streaming": "text-delta-v1",
}


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "local model"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "decoder_config": {},
                "token_codec_config": {
                    "type": "ChatCodecConfig",
                    "prompt_template": (
                        "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
                        "{% if add_generation_prompt %}assistant:{% endif %}"
                    ),
                },
            }
        )
    )
    (directory / "tokenizer.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"synthetic model fixture")
    return directory


@pytest.fixture
def log(tmp_path):
    return tmp_path / "worker requests.jsonl"


@pytest.fixture
def executable(tmp_path, log):
    """Reuse 1024 tokens when a scoped request repeats the previous scope."""
    path = tmp_path / "fake worker"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"log = open({str(log)!r}, 'a')\n"
        "log.write(json.dumps({'argv': sys.argv[1:]}) + '\\n'); log.flush()\n"
        f"print(json.dumps({READY!r}), flush=True)\n"
        "previous = None\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    log.write(json.dumps({'operation': request['operation'], 'scope': request.get('cache_scope')}) + '\\n')\n"
        "    log.flush()\n"
        "    if request['operation'] == 'tokenize':\n"
        "        prompt = request['text']\n"
        "        print(json.dumps({'token_count': 2048, 'token_ids': [2] * 2048}), flush=True)\n"
        "        continue\n"
        "    if request['operation'] == 'reset_cache':\n"
        "        previous = None\n"
        "        print(json.dumps({'cache_reset': True}), flush=True)\n"
        "        continue\n"
        "    scope = request.get('cache_scope')\n"
        "    reused = 1024 if scope is not None and scope == previous else 0\n"
        "    if 'bad reuse fixture' in prompt: reused = 100\n"
        "    previous = scope\n"
        "    if request.get('stream'):\n"
        "        print(json.dumps({'event': 'delta', 'text': 'synthetic answer'}), flush=True)\n"
        "    print(json.dumps({'text': 'synthetic answer', 'generated_token_ids': [3], 'input_tokens': 2048,\n"
        "        'output_tokens': 1, 'sampled_tokens': 2, 'finish_reason': 'eos', 'reused_input_tokens': reused,\n"
        "        'prefill_seconds': 0.2, 'decode_seconds': 0.01}), flush=True)\n"
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def desktop(tmp_path, executable):
    instance = Desktop(executable, tmp_path / "library", io.StringIO(), timeout=2)
    yield instance
    instance.stop(quiet=True)


def finish(desktop):
    thread = desktop.job
    if thread is not None:
        thread.join(timeout=5)
        assert not thread.is_alive()


def running(desktop, model):
    desktop.dispatch({"command": "install", "kind": "local", "source": str(model)})
    finish(desktop)
    identifier = desktop.models[0]["id"]
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.active_model == identifier
    return identifier


def requests(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def generated_scopes(log):
    return [entry["scope"] for entry in requests(log) if entry.get("operation") == "generate"]


def principal(session, client=None):
    return SimpleNamespace(is_set=lambda: False, cache_principal=(session, client))


def test_readiness_contract_requires_every_field():
    assert prefix_cache_enabled(READY)
    for change in (
        {"state_policy": "fresh-per-request"},
        {"prefix_cache": {**READY["prefix_cache"], "enabled": False}},
        {"prefix_cache": {**READY["prefix_cache"], "chunk_tokens": 512}},
        {"operations": ["tokenize", "decode", "generate"]},
        {"prefix_cache": None},
    ):
        assert not prefix_cache_enabled({**READY, **change})


def test_backend_chunk_sets_the_reuse_grid():
    hybrid = {**READY, "prefix_cache": {**READY["prefix_cache"], "chunk_tokens": 4096}}
    assert prefix_cache_enabled(hybrid)
    for chunk in (0, 512, 1536, 2048.0, True, "4096", 131072, None):
        assert not prefix_cache_enabled({**READY, "prefix_cache": {**READY["prefix_cache"], "chunk_tokens": chunk}})
    assert valid_reuse({"reused_input_tokens": 8192}, 9000, 4096)
    assert not valid_reuse({"reused_input_tokens": 1024}, 9000, 4096)
    assert valid_reuse({"reused_input_tokens": 1024}, 9000)


def test_text_workers_request_reuse_and_playground_scope_follows_the_conversation(desktop, model, log):
    running(desktop, model)
    assert "--prefix-cache" in requests(log)[0]["argv"]
    assert desktop.worker.prefix_cache is True
    for text in ("first", "second"):
        desktop.dispatch({"command": "generate", "text": text})
        finish(desktop)
    desktop.dispatch({"command": "reset"})
    desktop.dispatch({"command": "generate", "text": "third"})
    finish(desktop)
    first, second, third = generated_scopes(log)
    assert first == second and first != third
    assert all(len(scope) == 32 and int(scope, 16) >= 0 for scope in (first, third))


def test_api_scopes_are_separate_per_principal_and_report_cached_tokens(desktop, model, log):
    running(desktop, model)
    desktop.api = SimpleNamespace(api_session=SESSION)
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    master = desktop.api_complete(payload, cancelled=principal(SESSION))
    assert "prompt_tokens_details" in master["usage"]
    repeated = desktop.api_complete(payload, cancelled=principal(SESSION))
    assert repeated["usage"]["prompt_tokens_details"] == {"cached_tokens": 1024}
    assert repeated["usage"]["prompt_tokens"] == 2048
    client = desktop.api_complete(payload, cancelled=principal(SESSION, "openwork-peer"))
    assert client["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
    desktop.api_complete(payload, cancelled=principal(SESSION, "other-client"))
    stale = desktop.api_complete(payload, cancelled=principal("b" * 32))
    unscoped = desktop.api_complete(payload)
    scopes = generated_scopes(log)
    assert scopes[0] == scopes[1]
    assert len({scopes[0], scopes[2], scopes[3]}) == 3
    # A principal from an earlier API session, or none at all, never gets a scope.
    assert scopes[4] is None and scopes[5] is None
    assert stale["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
    assert unscoped["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}


def test_revocation_and_session_changes_reset_the_worker_and_rotate_scopes(desktop, model, log):
    running(desktop, model)
    revoked = []
    desktop.api = SimpleNamespace(
        api_session=SESSION,
        revoke_clients=lambda session, client=None, prefix=None: revoked.append((session, client)),
    )
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    desktop.api_complete(payload, cancelled=principal(SESSION, "openwork-peer"))
    desktop.api_complete(payload, cancelled=principal(SESSION))
    desktop.api_client_command(
        {
            "command": "revoke_api_clients",
            "request_id": "12345678-1234-4234-8234-123456789012",
            "api_session": SESSION,
            "client_id": "openwork-peer",
        }
    )
    assert revoked == [(SESSION, "openwork-peer")]
    assert [entry["operation"] for entry in requests(log)[1:]][-1] == "reset_cache"
    desktop.api_complete(payload, cancelled=principal(SESSION, "openwork-peer"))
    desktop.api_complete(payload, cancelled=principal(SESSION))
    scopes = generated_scopes(log)
    assert scopes[0] != scopes[2], "a revoked client's later grant starts a new scope"
    assert scopes[1] == scopes[3], "other principals keep theirs"
    desktop.api.api_session = "c" * 32
    desktop.api_complete(payload, cancelled=principal("c" * 32))
    assert generated_scopes(log)[4] not in scopes


def test_prefix_revocation_rotates_only_that_managed_clients_scopes(desktop, model, log):
    running(desktop, model)
    revoked = []
    desktop.api = SimpleNamespace(
        api_session=SESSION,
        revoke_clients=lambda session, client=None, prefix=None: revoked.append((session, client, prefix)),
    )
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    peers = (principal(SESSION, "opencode.1.2.3"), principal(SESSION, "openwork.4.5.6"), principal(SESSION))
    for peer in peers:
        desktop.api_complete(payload, cancelled=peer)
    desktop.api_client_command(
        {
            "command": "revoke_api_clients",
            "request_id": "12345678-1234-4234-8234-123456789012",
            "api_session": SESSION,
            "client_prefix": "opencode.",
        }
    )
    assert revoked == [(SESSION, None, "opencode.")]
    assert [entry["operation"] for entry in requests(log)[1:]][-1] == "reset_cache"
    for peer in peers:
        desktop.api_complete(payload, cancelled=peer)
    scopes = generated_scopes(log)
    assert scopes[0] != scopes[3], "every scope under the revoked prefix rotates"
    assert scopes[1] == scopes[4] and scopes[2] == scopes[5], "the other client and the master key keep theirs"


def test_reset_waits_for_a_running_request(desktop, model, log):
    running(desktop, model)
    desktop.job = threading.current_thread()
    try:
        desktop.reset_prefix_cache()
        assert "reset_cache" not in [entry.get("operation") for entry in requests(log)]
        assert desktop.prefix_reset_pending
    finally:
        desktop.job = None
    desktop.api = SimpleNamespace(api_session=SESSION)
    desktop.api_complete({"messages": [{"role": "user", "content": "hello"}]}, cancelled=principal(SESSION))
    operations = [entry.get("operation") for entry in requests(log)[1:]]
    assert operations == ["reset_cache", "tokenize", "generate"]
    assert not desktop.prefix_reset_pending


def test_invalid_reuse_counts_fail_the_request(desktop, model):
    running(desktop, model)
    desktop.api = SimpleNamespace(api_session=SESSION)
    with pytest.raises(APIError):
        desktop.api_complete(
            {"messages": [{"role": "user", "content": "bad reuse fixture"}]}, cancelled=principal(SESSION)
        )


def test_progress_and_estimates_count_processed_prompt_tokens():
    frame = {
        "event": "progress",
        "input_tokens": 4096,
        "reused_input_tokens": 3072,
        "sampled_tokens": 0,
        "output_token_budget": 16,
        "elapsed_seconds": 0.0,
        "prefill_seconds": None,
    }
    assert progress_frame(frame) == frame
    for invalid in (4096, -1, 1.5, None):
        assert progress_frame({**frame, "reused_input_tokens": invalid}) is None
    later = {**frame, "sampled_tokens": 1, "elapsed_seconds": 0.1, "prefill_seconds": 0.1}
    assert progress_frame(later, frame) is not None
    assert progress_frame({**later, "reused_input_tokens": 0}, frame) is None
    estimates = GenerationEstimates()
    # One cold 4096-token prefill. A 6144-token prompt that reuses 2048 tokens
    # processes as many tokens as that sample; an unreused one processes more.
    estimates.record("key", {"input_tokens": 4096, "sampled_tokens": 2, "prefill_seconds": 4.0, "decode_seconds": 0.1})
    longer = {**frame, "input_tokens": 6144}
    warm = estimates.estimate("key", {**longer, "reused_input_tokens": 2048})
    cold = estimates.estimate("key", {**longer, "reused_input_tokens": 0})
    assert warm["ttft_lower_seconds"] < cold["ttft_lower_seconds"]
    assert warm["ttft_upper_seconds"] < cold["ttft_upper_seconds"]
    # Recorded replies are scaled by processed tokens as well.
    estimates.record(
        "warm",
        {
            "input_tokens": 8192,
            "reused_input_tokens": 4096,
            "sampled_tokens": 2,
            "prefill_seconds": 4.0,
            "decode_seconds": 0.1,
        },
    )
    assert estimates.estimate("warm", {**longer, "reused_input_tokens": 2048})["ttft_lower_seconds"] is not None


def test_local_api_tags_each_request_with_its_principal():
    seen = []

    def completion(payload, on_delta, cancelled):
        seen.append(cancelled.cache_principal)
        return {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]}

    api = LocalAPI(lambda: [], completion, api_key="a" * 64, port=0, cancellable=True)
    api.start()
    try:
        api.grant_client("openwork-peer", "b" * 64, 120, api.api_session)
        for key in ("a" * 64, "b" * 64):
            connection = http.client.HTTPConnection("127.0.0.1", api._server.server_port, timeout=4)
            body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            assert connection.getresponse().status == 200
            connection.close()
        assert seen == [(api.api_session, None), (api.api_session, "openwork-peer")]
    finally:
        api.close()
