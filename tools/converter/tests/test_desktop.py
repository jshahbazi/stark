"""Model library and lifecycle checks using private synthetic workers."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from stark_convert.desktop import Desktop, DesktopError, converter_progress, metric


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
                    "default_system_prompt": "synthetic system",
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
def executable(tmp_path):
    path = tmp_path / "fake worker"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "print(json.dumps({'ready':True,'mode':'text-generation-v1',"
        "'max_context_tokens':8192,'max_new_tokens':4096,'streaming':'text-delta-v1'}),flush=True)\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if request['operation']=='tokenize': last_prompt=request['text']\n"
        "    if request['operation']=='generate' and 'wait fixture' in last_prompt: time.sleep(30)\n"
        "    if request['operation']=='generate' and 'failure fixture' in last_prompt: raise SystemExit(2)\n"
        "    if request.get('stream'):\n"
        "        p={'event':'progress','input_tokens':20,'sampled_tokens':0,"
        "'output_token_budget':request['max_new_tokens'],'elapsed_seconds':0.0,'prefill_seconds':None}\n"
        "        print(json.dumps(p),flush=True)\n"
        "        print(json.dumps({**p,'sampled_tokens':1,'elapsed_seconds':0.2,'prefill_seconds':0.2}),flush=True)\n"
        "        for chunk in ['synthetic ', 'answer']: print(json.dumps({'event':'delta','text':chunk}),flush=True)\n"
        "        print(json.dumps({**p,'sampled_tokens':5,'elapsed_seconds':0.28,'prefill_seconds':0.2}),flush=True)\n"
        "    reply={'token_count':20,'token_ids':[2]*20} if request['operation']=='tokenize' else {\n"
        "        'text':'synthetic answer','input_tokens':20,'output_tokens':4,'sampled_tokens':5,\n"
        "        'finish_reason':'eos',\n"
        "        'input_tokens_per_second':100.0,'output_tokens_per_second':50.0,\n"
        "        'prefill_seconds':0.2,'decode_seconds':0.08}\n"
        "    print(json.dumps(reply),flush=True)\n"
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def desktop(tmp_path, executable):
    instance = Desktop(executable, tmp_path / "library", io.StringIO(), timeout=2)
    yield instance
    instance.stop(quiet=True)


def conditional_support_report(command, _environment, _epoch):
    """Explicit metadata result for converter lifecycle fixtures; no source IO."""
    source = command[command.index("stark_convert.cli") + 1]
    local = "--offline" in command
    return {
        "schema_version": 1,
        "status": "conditional",
        "mode": command[command.index("--support-mode") + 1],
        "family_id": "qwen3_5",
        "family_name": "Synthetic Qwen3.5 family",
        "architecture": "qwen3_5",
        "model_kind": "language",
        "message": "Synthetic metadata admission; converter checks remain required.",
        "reasons": ["synthetic_metadata_fixture"],
        "warnings": [],
        "pending_checks": ["tensor_headers", "native_loading"],
        "modes": dict.fromkeys(("text", "decision", "vision-language"), "conditional"),
        "alternatives": [],
        "qualification": {"status": "not_assessed", "scope": "checkpoint_and_build", "calibration": "not_assessed"},
        "source": {
            "repo": None if local else source,
            "revision": None if local else "a" * 40,
            "metadata_sha256": {},
            "metadata_sources": [],
        },
    }


@pytest.fixture
def converter_support(desktop, monkeypatch):
    checks = []

    def preflight(command, environment, epoch):
        assert "--preflight" in command and "--output" not in command
        checks.append(command)
        return conditional_support_report(command, environment, epoch)

    monkeypatch.setattr(desktop, "run_support_check", preflight)
    return checks


def events(desktop):
    return [json.loads(line) for line in desktop.output.getvalue().splitlines()]


def finish(desktop):
    thread = desktop.job
    if thread is not None:
        thread.join(timeout=5)
        assert not thread.is_alive()


def installed(desktop, model):
    desktop.dispatch({"command": "install", "kind": "local", "source": str(model)})
    finish(desktop)
    assert len(desktop.models) == 1
    return desktop.models[0]["id"]


def running(desktop, model):
    identifier = installed(desktop, model)
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.active_model == identifier
    return identifier


def test_api_completions_are_stateless_and_share_loaded_worker(desktop, model):
    from stark_convert.local_api import APIError

    with pytest.raises(APIError) as unavailable:
        desktop.api_complete({"model": "missing", "messages": [{"role": "user", "content": "hello"}]})
    assert unavailable.value.status == 503
    identifier = running(desktop, model)
    process = desktop.worker.process
    history = list(desktop.messages)
    payload = {"model": identifier, "messages": [{"role": "user", "content": "synthetic API request"}]}
    response = desktop.api_complete(payload)
    assert response["model"] == model.name
    assert response["choices"][0]["message"]["content"] == "synthetic answer"
    assert response["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert desktop.messages == history
    assert desktop.worker.process is process

    assert desktop.api_models()[0]["id"] == model.name
    assert desktop.job is None
    assert all(e["event"] != "reply" for e in events(desktop))
    metrics_events = [e for e in events(desktop) if e["event"] == "metrics"]
    assert len(metrics_events) == 1
    assert metrics_events[0]["input_tokens"] == 20
    assert metrics_events[0]["output_tokens"] == 4
    assert metrics_events[0]["input_tokens_per_second"] == 100.0
    assert metrics_events[0]["output_tokens_per_second"] == 50.0
    assert "text" not in metrics_events[0]
    desktop.dispatch({"command": "generate", "text": "synthetic desktop request"})
    finish(desktop)
    assert len(desktop.messages) == 3
    assert desktop.worker.process is process


def test_playground_streams_content_free_eta_and_calibrates_the_next_request(desktop, model):
    identifier = running(desktop, model)
    request_id = "11111111-2222-4333-8444-555555555555"
    desktop.dispatch({"command": "generate", "text": "timing fixture", "request_id": request_id})
    finish(desktop)
    first = [event for event in events(desktop) if event["event"] == "generation_progress"]
    assert [event["sampled_tokens"] for event in first] == [0, 1, 5]
    assert all(event["id"] == request_id and event["model"] == identifier for event in first)
    assert first[0]["ttft_seconds"] is None and first[0]["ttft_lower_seconds"] is None
    assert first[1]["ttft_seconds"] == 0.2
    assert first[-1]["remaining_upper_seconds"] > 0
    assert all("text" not in event and "token_ids" not in event for event in first)
    assert [event for event in events(desktop) if event["event"] == "reply"][-1]["request_id"] == request_id
    desktop.output.seek(0)
    desktop.output.truncate()
    desktop.dispatch({"command": "generate", "text": "next timing fixture", "request_id": "second-request"})
    finish(desktop)
    second = [event for event in events(desktop) if event["event"] == "generation_progress"]
    assert second[0]["calibrated"]
    assert second[0]["ttft_lower_seconds"] <= 0.2 <= second[0]["ttft_upper_seconds"]
    assert second[0]["remaining_upper_seconds"] > second[0]["remaining_lower_seconds"] > 0
    assert second[-1]["remaining_upper_seconds"] != second[0]["remaining_upper_seconds"]


def test_progress_from_a_replaced_model_is_fenced_and_never_trains(desktop, model, monkeypatch):
    from stark_convert.local_api import APIError

    running(desktop, model)
    previous = desktop.worker.request

    def request(payload, **options):
        if payload["operation"] == "tokenize":
            return previous(payload, **options)
        desktop.epoch += 1
        with pytest.raises(OSError, match="no longer belongs"):
            options["on_progress"](
                {
                    "event": "progress",
                    "input_tokens": 20,
                    "sampled_tokens": 0,
                    "output_token_budget": 512,
                    "elapsed_seconds": 0.0,
                    "prefill_seconds": None,
                }
            )
        raise APIError(503, "Request stopped.", "request_stopped")

    monkeypatch.setattr(desktop.worker, "request", request)
    desktop.dispatch({"command": "generate", "text": "replacement timing fixture", "request_id": "replacement"})
    finish(desktop)
    assert not any(event["event"] == "generation_progress" for event in events(desktop))
    assert not desktop.generation_estimates.history


@pytest.mark.parametrize("request_id", [None, "", "bad id", 3])
def test_playground_rejects_invalid_request_identifiers(desktop, request_id):
    with pytest.raises(DesktopError, match="Invalid app command"):
        desktop.dispatch({"command": "generate", "text": "timing fixture", "request_id": request_id})


@pytest.fixture(params=["qwen3-hermes-v1", "qwen38-xml-v1", "qwen35-xml-v1"])
def tool_desktop(desktop, model, monkeypatch, request):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    running(desktop, model)
    seen = []
    parser = request.param
    reply = {
        "text": (
            '<tool_call>{"name":"lookup","arguments":{"value":3}}</tool_call>'
            if parser == "qwen3-hermes-v1"
            else "<tool_call>\n<function=lookup>\n<parameter=value>\n3\n</parameter>\n</function>\n</tool_call>"
        ),
        "input_tokens": 24,
        "output_tokens": 17,
        "finish_reason": "eos",
    }

    def request(payload, **options):
        seen.append(payload)
        if payload["operation"] == "tokenize":
            return {"token_count": 24, "token_ids": [2] * 24}
        if on_delta := options.get("on_delta"):
            try:
                on_delta(reply["text"])
            except OSError:
                from stark_convert.chat import ChatCancelled

                raise ChatCancelled("Request cancelled.") from None
        return dict(reply)

    monkeypatch.setattr(desktop.worker, "request", request)
    desktop.worker.streaming = True
    monkeypatch.setattr("stark_convert.desktop.tool_parser_for", lambda codec: parser)
    desktop.template = ImmutableSandboxedEnvironment().from_string("{{ tools | tojson }};{{ messages | tojson }}")
    payload = {
        "messages": [{"role": "user", "content": "synthetic fixture"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}},
                },
            }
        ],
    }
    return desktop, payload, seen, reply


def test_tool_calls_render_and_replay_complete_history(tool_desktop):
    desktop, payload, seen, _ = tool_desktop
    response = desktop.api_complete(payload)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None
    call = message["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"value": 3}
    assert response["usage"] == {"prompt_tokens": 24, "completion_tokens": 17, "total_tokens": 41}
    assert json.loads(seen[0]["text"].split(";", 1)[0]) == payload["tools"]
    payload["messages"] += [message, {"role": "tool", "tool_call_id": call["id"], "content": "6"}]
    desktop.api_complete(payload)
    rendered = json.loads(seen[2]["text"].split(";", 1)[1])
    assert rendered[-2]["tool_calls"][0]["function"]["arguments"] == {"value": 3}
    assert rendered[-1]["tool_call_id"] == call["id"] and rendered[-1]["content"] == "6"
    assert desktop.job is None and desktop.api_queue == []


def test_xml_schema_rejection_precedes_queue_and_worker_dispatch(tool_desktop):
    from stark_convert.local_api import APIError

    desktop, payload, seen, _ = tool_desktop
    if desktop.settings_summary()["tool_parser"] == "qwen3-hermes-v1":
        pytest.skip("Hermes arguments carry their JSON types directly")
    payload["tools"][0]["function"]["parameters"]["properties"]["value"] = {"type": None}
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload)
    assert error.value.code == "tool_schema_unsupported"
    assert not seen and desktop.job is None and desktop.api_queue == []


def test_unused_untyped_hint_preserves_normal_tool_requests(tool_desktop):
    desktop, payload, seen, _ = tool_desktop
    payload["tools"][0]["function"]["parameters"]["properties"]["unused"] = {
        "description": "Synthetic unconstrained optional value"
    }
    result = desktop.api_complete(payload)
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(seen[0]["text"].split(";", 1)[0]) == payload["tools"]
    assert desktop.job is None and desktop.api_queue == []


def test_unused_schemas_do_not_block_none_or_named_tool_choice(tool_desktop):
    desktop, payload, seen, reply = tool_desktop
    payload["tools"].append(
        {"type": "function", "function": {"name": "unused", "parameters": {"properties": {"x": {}}}}}
    )
    payload["tool_choice"] = {"type": "function", "function": {"name": "lookup"}}
    assert desktop.api_complete(payload)["choices"][0]["finish_reason"] == "tool_calls"
    assert [tool["function"]["name"] for tool in json.loads(seen[0]["text"].split(";", 1)[0])] == ["lookup"]
    reply["text"] = "Synthetic text without tool calls."
    payload["tool_choice"] = "none"
    assert desktop.api_complete(payload)["choices"][0]["message"]["content"] == reply["text"]
    assert json.loads(seen[2]["text"].split(";", 1)[0]) is None
    assert desktop.job is None and desktop.api_queue == []


def test_tool_capabilities_follow_parser_and_chat_mode(tool_desktop, monkeypatch):
    from stark_convert.local_api import APIError

    desktop, payload, seen, reply = tool_desktop
    parser = desktop.settings_summary()["tool_parser"]
    assert desktop.api_models()[0]["tool_calling"] is True
    assert desktop.api_status()["capabilities"]["tool_parser"] == parser
    desktop.active_mode = "decision"
    assert desktop.api_models() == []
    assert desktop.settings_summary()["tools"] is False
    assert desktop.api_status()["capabilities"]["tool_parser"] is None
    desktop.active_mode = "chat"
    monkeypatch.setattr("stark_convert.desktop.tool_parser_for", lambda codec: None)
    assert desktop.api_models()[0]["tool_calling"] is False
    assert desktop.api_status()["capabilities"]["tool_choices"] == []
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload)
    assert error.value.code == "tools_unsupported"
    assert not seen and desktop.job is None and desktop.api_queue == []
    reply["text"] = "Ordinary chat remains available."
    payload.pop("tools")
    assert desktop.api_complete(payload)["choices"][0]["message"]["content"] == reply["text"]


@pytest.mark.parametrize(
    "failure", ["malformed", "unknown", "truncated", "context", "cancel", "replacement", "worker"]
)
def test_tool_failure_releases_request_and_never_delivers_partial_calls(tool_desktop, monkeypatch, failure):
    import threading

    from stark_convert.local_api import APIError

    desktop, payload, _, reply = tool_desktop
    worker = desktop.worker
    control = threading.Event()
    if failure == "malformed":
        reply["text"] = '<tool_call>{"name":"lookup","arguments":'
    elif failure == "unknown":
        reply["text"] = reply["text"].replace("lookup", "unavailable")
    elif failure == "truncated":
        reply["finish_reason"] = "length"
    elif failure == "context":
        payload["max_tokens"] = desktop.api_maximum
        monkeypatch.setattr(worker, "request", lambda *args, **kwargs: {"token_count": 8192})
    else:
        original = desktop.infer

        def infer(*args, **kwargs):
            result = original(*args, **kwargs)
            if failure == "cancel":
                control.set()
            elif failure == "replacement":
                desktop.epoch += 1
            else:
                raise ValueError
            return result

        monkeypatch.setattr(desktop, "infer", infer)
    delivered = []
    with pytest.raises(APIError):
        desktop.api_complete(payload, on_delta=delivered.append, cancelled=control)
    assert delivered == [] and desktop.job is None and desktop.api_queue == []
    if failure in {"malformed", "unknown", "truncated", "cancel"}:
        assert desktop.worker is worker
        reply["text"], reply["finish_reason"] = "synthetic retry", "eos"
        if failure != "cancel":
            assert desktop.api_complete(payload)["choices"][0]["message"]["content"] == "synthetic retry"
    if failure == "worker":
        assert desktop.worker is None


@pytest.mark.parametrize("terminal", ["worker_fault", "cancelled", "reply"])
def test_active_model_replacement_is_classified_without_closing_new_worker(tool_desktop, monkeypatch, terminal):
    from unittest.mock import Mock

    from stark_convert.chat import ChatCancelled
    from stark_convert.local_api import APIError

    desktop, payload, _, reply = tool_desktop
    old_worker, old_model = desktop.worker, desktop.active_model
    replacement = Mock(spec=["close"])

    def infer(*args, **kwargs):
        with desktop.lock:
            desktop.epoch += 1
            desktop.worker = replacement
            desktop.active_model = "synthetic-replacement"
        if terminal == "worker_fault":
            raise ValueError("Synthetic old-worker failure")
        if terminal == "cancelled":
            raise ChatCancelled("Synthetic old-worker cancellation")
        return dict(reply)

    monkeypatch.setattr(desktop, "infer", infer)
    delivered = []
    try:
        with pytest.raises(APIError) as error:
            desktop.api_complete(payload, on_delta=delivered.append)
        assert error.value.status == 503 and error.value.code == "model_changed"
        assert desktop.worker is replacement and desktop.active_model == "synthetic-replacement"
        replacement.close.assert_not_called()
        assert desktop.job is None and desktop.api_queue == [] and delivered == []
    finally:
        # The synthetic replacement owns no process. Restore the fixture's real
        # private worker so its normal teardown can reap that process.
        desktop.worker, desktop.active_model = old_worker, old_model


@pytest.mark.parametrize(
    "reason,expected_status,expected_code",
    [
        (None, 499, "request_cancelled"),
        ("deadline", 504, "request_timeout"),
    ],
)
def test_same_model_worker_fault_after_cancellation_keeps_cancellation_category(
    tool_desktop, monkeypatch, reason, expected_status, expected_code
):
    import threading

    from stark_convert.local_api import APIError

    desktop, payload, _, _ = tool_desktop
    control = threading.Event()
    control.reason = reason
    process = desktop.worker.process

    def infer(*args, **kwargs):
        control.set()
        raise ValueError("Synthetic interrupted worker failure")

    monkeypatch.setattr(desktop, "infer", infer)
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload, cancelled=control)
    assert error.value.status == expected_status and error.value.code == expected_code
    assert desktop.worker is None and process.poll() is not None
    assert desktop.job is None and desktop.api_queue == []


def test_tool_http_stream_matches_buffered_calls_and_never_duplicates(tool_desktop):
    import urllib.request

    from stark_convert.local_api import LocalAPI

    desktop, payload, _, _ = tool_desktop
    buffered = desktop.api_complete(payload)["choices"][0]
    api = LocalAPI(desktop.api_models, desktop.api_complete, api_key="b" * 64, port=0, cancellable=True)
    info = api.start()
    try:
        request = urllib.request.Request(
            info["base_url"] + "/chat/completions",
            data=json.dumps({**payload, "stream": True, "stream_options": {"include_usage": True}}).encode(),
            headers={"Authorization": "Bearer " + "b" * 64, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        calls = [
            call for chunk in chunks for choice in chunk["choices"] for call in choice["delta"].get("tool_calls", [])
        ]
        assert len(calls) == 1 and calls[0]["index"] == 0
        assert calls[0]["function"] == buffered["message"]["tool_calls"][0]["function"]
        assert any(c.get("finish_reason") == "tool_calls" for chunk in chunks for c in chunk["choices"])
        assert chunks[-1]["usage"]["total_tokens"] == 41
        assert body.count(b"data: [DONE]") == 1
    finally:
        api.close()


def test_tool_buffered_stream_disconnect_keeps_worker_aligned(tool_desktop):
    from stark_convert.local_api import APIError

    desktop, payload, _, reply = tool_desktop
    worker = desktop.worker
    reply["text"] = "synthetic preamble\n" + reply["text"]

    def disconnected(_text):
        raise BrokenPipeError

    with pytest.raises(APIError) as error:
        desktop.api_complete(payload, on_delta=disconnected)
    assert error.value.code == "request_cancelled"
    assert desktop.worker is worker and desktop.job is None
    assert desktop.api_complete(payload)["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_enabled_http_delivers_text_before_generation_finishes(tool_desktop, monkeypatch):
    import threading
    import urllib.request

    from stark_convert.local_api import LocalAPI

    desktop, payload, _, reply = tool_desktop
    original = desktop.worker.request
    release = threading.Event()
    terminal = threading.Event()

    def generate(request, **options):
        if request["operation"] != "generate":
            return original(request, **options)
        assert request["stream"] is True
        options["on_delta"]("Early text")
        assert release.wait(5)
        options["on_delta"](" then final.")
        terminal.set()
        return {**reply, "text": "Early text then final."}

    monkeypatch.setattr(desktop.worker, "request", generate)
    api = LocalAPI(desktop.api_models, desktop.api_complete, api_key="b" * 64, port=0, cancellable=True)
    info = api.start()
    try:
        request = urllib.request.Request(
            info["base_url"] + "/chat/completions",
            data=json.dumps({**payload, "stream": True}).encode(),
            headers={"Authorization": "Bearer " + "b" * 64, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            chunks = []
            while True:
                line = response.readline()
                assert line
                if line.startswith(b"data: {"):
                    item = json.loads(line[6:])
                    chunks.append(item)
                    if item["choices"][0]["delta"].get("content"):
                        break
            assert not terminal.is_set()  # The HTTP client can render while native work is still active.
            release.set()
            rest = response.read()
        chunks += [json.loads(line[6:]) for line in rest.splitlines() if line.startswith(b"data: {")]
        text = "".join(choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"])
        assert text == "Early text then final."
        assert rest.count(b"data: [DONE]") == 1
        assert desktop.job is None and desktop.api_queue == []
    finally:
        release.set()
        api.close()


@pytest.mark.parametrize("failure", ["malformed", "unknown", "truncated"])
def test_tool_stream_failure_after_provisional_text_recovers(tool_desktop, failure):
    from stark_convert.local_api import APIError

    desktop, payload, _, reply = tool_desktop
    worker = desktop.worker
    if failure == "malformed":
        reply["text"] = "<tool_call>{"
    elif failure == "unknown":
        reply["text"] = reply["text"].replace("lookup", "unknown")
    else:
        reply["finish_reason"] = "length"
    reply["text"] = "Provisional text \n" + reply["text"]
    delivered = []
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload, on_delta=delivered.append)
    assert error.value.code == "invalid_tool_call"
    assert "".join(delivered) == "Provisional text"
    assert desktop.worker is worker and desktop.job is None and desktop.api_queue == []
    reply.update(text="Fresh response.", finish_reason="eos")
    delivered.clear()
    result = desktop.api_complete(payload, on_delta=delivered.append)
    assert "".join(delivered) == result["choices"][0]["message"]["content"] == "Fresh response."


def test_private_api_configuration_preserves_listener_on_port_conflict(desktop):
    import socket
    import urllib.error
    import urllib.request

    # Production command validation never accepts an ephemeral port.
    with pytest.raises(DesktopError):
        desktop.configure_api({"command": "configure_api", "api_key": "a" * 64, "port": 0})
    desktop.start_api("a" * 64, port=0)
    old = desktop.api
    base = events(desktop)[-1]["base_url"]
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    try:
        desktop.configure_api({"command": "configure_api", "api_key": "b" * 64, "port": occupied.getsockname()[1]})
        event = events(desktop)[-1]
        assert event["event"] == "connection" and "in use" in event["message"]
        assert event["base_url"] == base
        assert desktop.api is old and "api_key" not in event
        request = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + "b" * 64})
        with urllib.request.urlopen(request, timeout=2) as result:
            assert result.status == 200
    finally:
        occupied.close()
        desktop.api.close()


def test_api_assistant_reasoning_history_reaches_model_template(desktop, model):
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["token_codec_config"]["prompt_template"] = (
        "{% for m in messages %}{{ m.role }}:{{ m.content }}"
        "{% if m.reasoning_content is defined %}[{{ m.reasoning_content }}]{% endif %};{% endfor %}"
    )
    config_path.write_text(json.dumps(config))
    running(desktop, model)
    process = desktop.worker.process
    history = list(desktop.messages)
    request = desktop.worker.request
    prompts = []

    def capture(payload, **options):
        if payload["operation"] == "tokenize":
            prompts.append(payload["text"])
        return request(payload, **options)

    desktop.worker.request = capture
    response = desktop.api_complete(
        {
            "messages": [
                {"role": "user", "content": "first synthetic turn"},
                {"role": "assistant", "content": "synthetic reply", "reasoning_content": "synthetic reasoning"},
                {"role": "user", "content": "second synthetic turn"},
            ]
        }
    )
    assert response["choices"][0]["message"]["content"] == "synthetic answer"
    assert "assistant:synthetic reply[synthetic reasoning];user:second synthetic turn;" in prompts[0]
    assert desktop.worker.process is process and desktop.messages == history
    assert desktop.job is None


@pytest.mark.parametrize(
    "source, expected",
    [
        ("huggingface:Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
        ("local:/synthetic/JevK5", "JevK5"),
    ],
)
def test_public_model_ids_preserve_private_library_identity(desktop, model, source, expected):
    identifier = installed(desktop, model)
    desktop.models[0].update(source=source, name="JevK5")
    desktop.persist()
    desktop.library()
    metadata = events(desktop)[-1]["models"][0]
    assert metadata["id"] == identifier
    assert metadata["api_id"] == expected
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.api_models()[0]["id"] == expected
    for alias in (expected, identifier, "JevK5", None):
        response = desktop.api_complete({"model": alias, "messages": [{"role": "user", "content": "fixture"}]})
        assert response["model"] == expected
    reloaded = Desktop(desktop.executable, desktop.data_dir, io.StringIO())
    assert reloaded.models[0]["id"] == identifier
    reloaded.library()
    assert events(reloaded)[-1]["models"][0]["api_id"] == expected


@pytest.mark.parametrize("stream", [False, True])
def test_http_rejects_unknown_model_with_public_ids(desktop, model, stream):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from stark_convert.local_api import LocalAPI

    running(desktop, model)
    api = LocalAPI(desktop.api_models, desktop.api_complete, api_key="a" * 64, port=0, cancellable=True)
    info = api.start()
    try:
        request = Request(
            info["base_url"] + "/chat/completions",
            data=json.dumps(
                {"model": "unknown", "stream": stream, "messages": [{"role": "user", "content": "fixture"}]}
            ).encode(),
            headers={"Authorization": "Bearer " + info["api_key"], "Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 404
        assert json.loads(error.value.read())["error"]["code"] == "model_not_found"
    finally:
        api.close()


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 3},
        {"temperature": False},
        {"n": 2},
        {"tools": [{"type": "function", "function": {"name": "lookup"}}]},
        {"max_tokens": True},
        {"max_tokens": 4097},
        {"max_tokens": 1, "max_completion_tokens": 1},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "unsupported"}]}]},
        {"model": "unknown"},
    ],
)
def test_api_rejects_unsupported_requests_without_affecting_worker(desktop, model, extra):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process
    with pytest.raises(APIError) as error:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "hello"}], **extra})
    assert error.value.status in (400, 404)
    assert desktop.worker.process is process
    assert desktop.job is None


def test_api_busy_and_stop_fence_inflight_request(desktop, model):
    import threading

    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    errors = []
    payload = {"model": identifier, "messages": [{"role": "user", "content": "wait fixture"}]}

    def request():
        try:
            desktop.api_complete(payload)
        except APIError as error:
            errors.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    deadline = time.monotonic() + 2
    while desktop.job is None and time.monotonic() < deadline:
        time.sleep(0.01)
    desktop.api_queue_timeout = 0.05
    with pytest.raises(APIError) as busy:
        desktop.api_complete(payload)
    assert busy.value.status == 429
    desktop.dispatch({"command": "generate", "text": "competing request"})
    assert any("Wait for the current operation" in e.get("message", "") for e in events(desktop))
    desktop.stop()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(errors) == 1 and errors[0].status == 503
    assert desktop.worker is None
    assert desktop.api_models() == []


def test_api_live_http_uses_desktop_model(desktop, model):
    import urllib.request

    identifier = running(desktop, model)
    desktop.start_api("a" * 64, port=0)
    details = next(e for e in events(desktop) if e["event"] == "connection")
    try:
        request = urllib.request.Request(
            details["base_url"] + "/chat/completions",
            data=json.dumps(
                {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True}
            ).encode(),
            headers={"Authorization": "Bearer " + "a" * 64, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        assert text == "synthetic answer"
        assert b"data: [DONE]" in body
        assert "a" * 64 not in desktop.registry.read_text()
    finally:
        desktop.api.close()


def test_api_worker_failure_clears_job_and_can_restart(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    with pytest.raises(APIError) as failed:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "failure fixture"}]})
    assert failed.value.status == 503 and failed.value.code == "generation_failed"
    assert desktop.job is None
    assert desktop.worker is None
    assert not any(e["event"] == "metrics" for e in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.active_model == identifier


def test_api_streams_native_deltas_and_retains_worker(desktop, model):
    identifier = running(desktop, model)
    process = desktop.worker.process
    chunks = []
    response = desktop.api_complete(
        {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True},
        on_delta=chunks.append,
    )
    assert chunks == ["synthetic ", "answer"]
    assert "".join(chunks) == response["choices"][0]["message"]["content"]
    metrics_events = [e for e in events(desktop) if e["event"] == "metrics"]
    assert len(metrics_events) == 1
    assert metrics_events[0]["output_tokens"] == 4
    assert metrics_events[0]["input_tokens_per_second"] == 100.0
    assert metrics_events[0]["output_tokens_per_second"] == 50.0
    assert "text" not in metrics_events[0]
    assert desktop.worker.process is process
    assert desktop.job is None


def test_api_stream_disconnect_fences_worker_and_clears_job(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process

    def disconnected(_):
        raise BrokenPipeError

    with pytest.raises(APIError) as failed:
        desktop.api_complete(
            {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True},
            on_delta=disconnected,
        )
    assert failed.value.status == 499 and failed.value.code == "request_cancelled"
    assert desktop.worker is None
    assert desktop.job is None
    assert process.poll() is not None
    assert not any(e["event"] == "metrics" for e in events(desktop))


def test_old_worker_streaming_is_explicitly_rejected(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    desktop.worker.streaming = False
    with pytest.raises(APIError) as failed:
        desktop.api_complete(
            {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True},
            on_delta=lambda _: None,
        )
    assert failed.value.code == "streaming_unavailable"
    assert desktop.worker is not None
    assert desktop.job is None


def test_api_context_rejection_keeps_loaded_worker(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process
    desktop.api_context = 21
    with pytest.raises(APIError) as oversized:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "hello"}]})
    assert oversized.value.code == "context_length_exceeded"
    assert desktop.job is None
    assert desktop.worker.process is process


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("overflow", ["ascii", "utf8", "template"])
def test_http_worker_request_overflow_preserves_model_and_allows_retry(desktop, model, stream, overflow):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from stark_convert.local_api import LocalAPI

    if overflow == "template":
        config_path = model / "config.json"
        config = json.loads(config_path.read_text())
        config["token_codec_config"]["prompt_template"] = (
            "{% for m in messages %}{{ m.content * 2 }}{% endfor %}assistant:"
        )
        config_path.write_text(json.dumps(config))
    running(desktop, model)
    worker = desktop.worker
    history = list(desktop.messages)
    payload = {"stream": stream, "messages": [{"role": "user", "content": ""}]}
    limit = 1024 * 1024
    if overflow == "template":
        content = "x" * (limit // 2)
    else:
        budget = limit - len(json.dumps(payload, ensure_ascii=False).encode())
        unit = "界" if overflow == "utf8" else "x"
        count, padding = divmod(budget, len(unit.encode()))
        content = unit * count + "x" * padding
    payload["messages"][0]["content"] = content
    body = json.dumps(payload, ensure_ascii=False).encode()
    assert len(body) <= limit
    if overflow != "template":
        assert len(body) == limit
    api = LocalAPI(desktop.api_models, desktop.api_complete, api_key="a" * 64, port=0, cancellable=True)
    details = api.start()
    headers = {"Authorization": "Bearer " + "a" * 64, "Content-Type": "application/json"}
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(details["base_url"] + "/chat/completions", data=body, headers=headers), timeout=5)
        assert error.value.code == 400
        assert json.loads(error.value.read())["error"]["code"] == "request_too_large"
        assert desktop.worker is worker and worker.process.poll() is None
        assert desktop.messages == history
        assert desktop.job is None and desktop.api_queue == []

        payload["messages"][0]["content"] = "small fixture"
        retry = Request(details["base_url"] + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        with urlopen(retry, timeout=5) as response:
            result = response.read()
            assert response.status == 200
        if stream:
            assert b"data: [DONE]" in result
        else:
            assert json.loads(result)["choices"][0]["message"]["content"] == "synthetic answer"
        assert desktop.worker is worker and worker.process.poll() is None
    finally:
        api.close()


def test_oversized_desktop_turn_preserves_model_and_allows_retry(desktop, model):
    running(desktop, model)
    worker = desktop.worker
    history = list(desktop.messages)
    desktop.dispatch({"command": "generate", "text": "x" * 1024 * 1024})
    finish(desktop)
    assert desktop.worker is worker and worker.process.poll() is None
    assert desktop.messages == history and desktop.job is None
    assert any(e["event"] == "error" and "request limit" in e["message"] for e in events(desktop))
    desktop.dispatch({"command": "generate", "text": "small fixture"})
    finish(desktop)
    assert desktop.worker is worker and worker.process.poll() is None
    assert len(desktop.messages) == len(history) + 2


def test_api_sampling_limits_aliases_and_message_parts(desktop, model):
    running(desktop, model)
    seen = []
    infer = desktop.infer

    def capture(messages, maximum, **options):
        seen.append((messages, maximum, options))
        return infer(messages, maximum, **options)

    desktop.infer = capture
    result = desktop.api_complete(
        {
            "model": desktop.models[0]["name"],
            "temperature": 0.7,
            "top_p": 0.8,
            "seed": 42,
            "max_tokens": 1000,
            "messages": [
                {"role": "developer", "content": "system"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
        }
    )
    assert result["model"] == model.name
    assert seen[0][0] == [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]
    assert seen[0][1] == 1000
    assert seen[0][2]["sampling"] == {"mode": "stochastic", "temperature": 0.7, "top_p": 0.8}
    assert seen[0][2]["seed"] == 42


def test_stale_epoch_and_expired_deadline_preserve_worker(desktop, model):
    import threading

    from stark_convert.local_api import APIError

    running(desktop, model)
    process = desktop.worker.process
    with patch.object(desktop.worker, "request") as request:
        with pytest.raises(APIError) as stale:
            desktop.infer([{"role": "user", "content": "hello"}], 16, epoch=desktop.epoch - 1)
        assert stale.value.code == "model_changed"
        request.assert_not_called()
    control = threading.Event()
    control.reason = "deadline"
    control.set()
    with pytest.raises(APIError) as expired:
        desktop.api_complete({"messages": [{"role": "user", "content": "hello"}]}, cancelled=control)
    assert expired.value.status == 504
    assert desktop.worker.process is process and process.poll() is None


def test_api_fifo_queue_and_waiter_cancellation(desktop, model):
    import threading

    from stark_convert.local_api import APIError

    running(desktop, model)
    entered, release = threading.Event(), threading.Event()
    order, errors = [], []

    def infer(messages, maximum, **options):
        text = messages[-1]["content"]
        order.append(text)
        if text == "first":
            entered.set()
            assert release.wait(2)
        return {"text": "answer", "input_tokens": 1, "output_tokens": 1, "finish_reason": "eos"}

    desktop.infer = infer

    def call(text, control=None):
        try:
            desktop.api_complete({"messages": [{"role": "user", "content": text}]}, cancelled=control)
        except APIError as error:
            errors.append(error.code)

    def queued(count):
        deadline = time.monotonic() + 2
        while len(desktop.api_queue) != count and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(desktop.api_queue) == count

    first = threading.Thread(target=call, args=("first",))
    cancelled = threading.Event()
    second = threading.Thread(target=call, args=("cancelled", cancelled))
    third = threading.Thread(target=call, args=("third",))
    first.start()
    assert entered.wait(2)
    second.start()
    queued(1)
    third.start()
    queued(2)
    desktop.api_queue_limit = 2
    with pytest.raises(APIError) as full:
        desktop.api_complete({"messages": [{"role": "user", "content": "over limit"}]})
    assert full.value.status == 429 and full.value.code == "queue_full"
    cancelled.set()
    second.join(timeout=2)
    release.set()
    first.join(timeout=2)
    third.join(timeout=2)
    assert not any(t.is_alive() for t in (first, second, third))
    assert order == ["first", "third"]
    assert errors == ["request_cancelled"]
    assert desktop.api_queue == [] and desktop.job is None


@pytest.fixture
def score_executable(tmp_path):
    path = tmp_path / "fake score worker"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "assert sys.argv[1]=='--score-model-path'\n"
        "print(json.dumps({'ready':True,'mode':'exact-token-scoring-cache-v1','vocab_size':1000}),flush=True)\n"
        "prompt=''\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if request['operation']=='tokenize':\n"
        "        text=request['text']\n"
        "        if len(text)>1: prompt=text\n"
        "        ids=[ord(text)] if len(text)==1 else [1,2,3]\n"
        "        reply={'token_ids':ids,'token_count':len(ids)}\n"
        "    elif request['operation']=='reset_cache': reply={'cache_reset':True}\n"
        "    else:\n"
        "        if 'wait fixture' in prompt: time.sleep(30)\n"
        "        if 'failure fixture' in prompt:\n"
        "            print('private exception fixture',file=sys.stderr,flush=True)\n"
        "            raise SystemExit(2)\n"
        "        candidates=request['candidate_token_ids']\n"
        "        p=[0.1/(len(candidates)-1)]*(len(candidates)-1)+[0.9]\n"
        "        if 'bad readout fixture' in prompt: p[0]=float('nan')\n"
        "        reply={'candidate_token_ids':candidates,'candidate_probabilities':p,\n"
        "               'input_tokens':len(request['input_token_ids']),\n"
        "               'temperature':request['temperature'],'truncated':False}\n"
        "        if 'context fixture' in prompt: reply={'error':'context_overflow'}\n"
        "    print(json.dumps(reply),flush=True)\n"
    )
    path.chmod(0o755)
    return path


def running_decision(desktop, model, score_executable):
    identifier = installed(desktop, model)
    desktop.score_executable = score_executable
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "decision"})
    finish(desktop)
    assert desktop.active_model == identifier and desktop.active_mode == "decision"
    return identifier


def test_api_rejects_decision_worker_and_does_not_touch_protocol(desktop, model, score_executable):
    from stark_convert.local_api import APIError

    identifier = running_decision(desktop, model, score_executable)
    assert desktop.api_models() == []
    status = desktop.api_status()
    assert status["model_loaded"] is False
    assert status["capabilities"]["stop_sequences"] is False
    assert status["capabilities"]["cancellation"] is False
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        with pytest.raises(APIError) as error:
            desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "fixture"}]})
        assert error.value.code == "model_not_loaded"
        request.assert_not_called()


def decision_command(**overrides):
    return {
        "command": "decide",
        "text": "synthetic decision text",
        "question": "synthetic decision question",
        "answers": ["First", "Second", "Third"],
        **overrides,
    }


def test_decision_private_pipe_roundtrip_uses_input_order_and_keeps_library_content_free(
    desktop, model, score_executable
):
    running_decision(desktop, model, score_executable)
    worker = desktop.worker
    desktop.dispatch(decision_command(answers=["Third", "First", "Second"]))
    finish(desktop)
    result = next(event for event in events(desktop) if event["event"] == "decision")
    assert result == {
        "event": "decision",
        "choice_index": 2,
        "answers": ["Third", "First", "Second"],
        "probabilities": [0.05, 0.05, 0.9],
        "input_tokens": 3,
    }
    assert desktop.worker is worker
    assert desktop.messages == []
    assert "synthetic decision" not in desktop.registry.read_text()
    assert set(json.loads(desktop.registry.read_text())[0]) == {"id", "name", "path", "source"}
    assert not any(key in result for key in ("logits", "token_ids", "candidate_token_ids"))
    ready = next(event for event in events(desktop) if event["event"] == "ready")
    assert ready["mode"] == "decision"


def test_manual_mode_switch_reaps_old_worker_and_reloads_selected_model(desktop, model, score_executable):
    identifier = running_decision(desktop, model, score_executable)
    score_process = desktop.worker.process
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "chat"})
    finish(desktop)
    assert score_process.poll() is not None
    assert desktop.active_mode == "chat"
    chat_process = desktop.worker.process
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "decision"})
    finish(desktop)
    assert chat_process.poll() is not None
    assert desktop.active_mode == "decision"
    assert desktop.worker.process.pid != score_process.pid


@pytest.mark.parametrize("count", [2, 16])
def test_decision_minimum_and_maximum_answer_sets_roundtrip(desktop, model, score_executable, count):
    running_decision(desktop, model, score_executable)
    labels = [f"Answer {index}" for index in range(count)]
    desktop.dispatch(decision_command(answers=labels))
    finish(desktop)
    result = next(event for event in events(desktop) if event["event"] == "decision")
    assert result["answers"] == labels
    assert len(result["probabilities"]) == count
    assert result["choice_index"] == count - 1


def test_mode_commands_are_guarded_and_invalid_mode_preserves_loaded_worker(desktop, model, score_executable):
    identifier = running_decision(desktop, model, score_executable)
    process = desktop.worker.process
    with pytest.raises(DesktopError, match="Choose Chat or Decision"):
        desktop.dispatch({"command": "run", "model_id": identifier, "mode": "unknown"})
    assert desktop.worker.process is process
    with pytest.raises(DesktopError, match="Chat mode"):
        desktop.dispatch(advanced_options())
    desktop.dispatch({"command": "generate", "text": "synthetic text"})
    finish(desktop)
    assert desktop.worker.process is process
    assert not any(event["event"] == "reply" for event in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    desktop.dispatch(decision_command())
    finish(desktop)
    assert not any(event["event"] == "decision" for event in events(desktop))


@pytest.mark.parametrize("answers", [["single"], ["duplicate", " Duplicate "], ["first", ""], None])
def test_bad_decision_answers_reject_before_starting_or_writing_to_worker(desktop, model, score_executable, answers):
    running_decision(desktop, model, score_executable)
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        with pytest.raises(DesktopError):
            desktop.dispatch(decision_command(answers=answers))
        request.assert_not_called()
    assert desktop.job is None


@pytest.mark.parametrize("text", ["failure fixture", "bad readout fixture"])
def test_decision_native_or_protocol_failure_fences_worker_without_diagnostic_payloads(
    desktop, model, score_executable, text
):
    running_decision(desktop, model, score_executable)
    process = desktop.worker.process
    desktop.dispatch(decision_command(text=text))
    finish(desktop)
    assert desktop.worker is None and process.poll() is not None
    assert desktop.active_mode is None
    assert not any(event["event"] == "decision" for event in events(desktop))
    assert text not in desktop.output.getvalue()
    assert "private exception fixture" not in desktop.output.getvalue()


def test_native_model_context_rejection_preserves_worker_and_allows_retry(desktop, model, score_executable):
    running_decision(desktop, model, score_executable)
    worker = desktop.worker
    desktop.dispatch(decision_command(text="context fixture"))
    finish(desktop)
    assert desktop.worker is worker
    assert not any(event["event"] == "decision" for event in events(desktop))
    assert any("context limit" in event.get("message", "") for event in events(desktop))
    desktop.dispatch(decision_command())
    finish(desktop)
    assert len([event for event in events(desktop) if event["event"] == "decision"]) == 1


def test_open_reasoning_boundary_is_recoverable_and_does_not_write_to_worker(desktop, model, score_executable):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    running_decision(desktop, model, score_executable)
    worker, original_template, original_codec = desktop.worker, desktop.template, desktop.codec
    path = Path(__file__).parent / "fixtures" / "rendering_templates.json"
    desktop.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads(path.read_text())["templates"]["qwen3-thinking"]
    )
    desktop.codec = {"end_of_thinking_tag": "\n</think>"}
    with patch.object(worker, "request", wraps=worker.request) as request:
        desktop.dispatch(decision_command(text="literal </think> in supplied evidence"))
        finish(desktop)
        request.assert_not_called()
    assert desktop.worker is worker and worker.process.poll() is None
    assert events(desktop)[-1]["state"] == "ready"
    assert any("starts in reasoning mode" in event.get("message", "") for event in events(desktop))
    assert not any(event["event"] == "decision" for event in events(desktop))
    desktop.template, desktop.codec = original_template, original_codec
    desktop.dispatch(decision_command(text="literal <think> without a closing tag"))
    finish(desktop)
    assert len([event for event in events(desktop) if event["event"] == "decision"]) == 1


def test_stop_cancels_complete_decision_and_fences_late_results(desktop, model, score_executable):
    import threading

    identifier = running_decision(desktop, model, score_executable)
    process = desktop.worker.process
    entered = threading.Event()
    request = desktop.worker.request

    def blocking_request(payload):
        if payload["operation"] == "score":
            entered.set()
        return request(payload)

    with patch.object(desktop.worker, "request", side_effect=blocking_request):
        desktop.dispatch(decision_command(text="wait fixture"))
        assert entered.wait(timeout=2)
        desktop.dispatch(decision_command())
        assert any("Wait for the current operation" in event.get("message", "") for event in events(desktop))
        started = time.monotonic()
        desktop.stop()
        assert time.monotonic() - started < 5
    assert process.poll() is not None
    assert not any(event["event"] == "decision" for event in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "decision"})
    finish(desktop)
    assert desktop.worker.process.pid != process.pid


def advanced_options(**overrides):
    return {
        "command": "configure",
        "system_prompt": "custom system fixture",
        "context_tokens": 2048,
        "max_new_tokens": 128,
        "thinking": True,
        **overrides,
    }


def test_advanced_options_render_reset_and_generate_without_reloading(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "previous turn fixture"})
    finish(desktop)
    desktop.dispatch(advanced_options())
    assert desktop.messages == [{"role": "system", "content": "custom system fixture"}]
    assert desktop.worker.process is process
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    desktop.template = ImmutableSandboxedEnvironment().from_string(
        "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}thinking={{ enable_thinking }}"
    )
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.dispatch({"command": "generate", "text": "new turn fixture"})
        finish(desktop)
        payload = request.call_args_list[-1].args[0]
        tokenization = request.call_args_list[0].args[0]
    assert payload["context_tokens"] == 2048
    assert payload["max_new_tokens"] == 128
    assert tokenization["text"] == "system:custom system fixture;user:new turn fixture;thinking=True"
    assert payload["input_token_ids"] == [2] * 20
    desktop.dispatch({"command": "reset"})
    assert desktop.messages == [{"role": "system", "content": "custom system fixture"}]
    assert "custom system fixture" not in desktop.registry.read_text()
    assert "custom system fixture" not in desktop.output.getvalue()


def test_advanced_options_model_default_empty_override_and_restart(desktop, model):
    identifier = running(desktop, model)
    desktop.dispatch(advanced_options(system_prompt=""))
    assert desktop.messages == []
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.messages == []
    assert (desktop.context, desktop.maximum, desktop.thinking) == (2048, 128, True)
    desktop.dispatch(advanced_options(system_prompt=None, thinking=False))
    assert desktop.messages == [{"role": "system", "content": "synthetic system"}]
    assert desktop.thinking is False
    replacement = Desktop(desktop.executable, desktop.data_dir, io.StringIO())
    assert replacement.system_prompt is None
    assert (replacement.context, replacement.maximum, replacement.thinking) == (None, 512, False)
    assert replacement.context_override is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_tokens": True},
        {"context_tokens": 1},
        {"context_tokens": 8193},
        {"context_tokens": 100.5},
        {"context_tokens": "2048"},
        {"context_tokens": None},
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"max_new_tokens": 4097},
        {"max_new_tokens": 2048},
        {"max_new_tokens": False},
        {"max_new_tokens": float("nan")},
        {"thinking": "true"},
        {"thinking": 1},
        {"system_prompt": []},
        {"unknown": "private fixture"},
    ],
)
def test_advanced_options_invalid_changes_are_atomic(desktop, model, overrides):
    running(desktop, model)
    before = list(desktop.messages)
    with pytest.raises(DesktopError, match="Invalid advanced options"):
        desktop.dispatch(advanced_options(**overrides))
    assert desktop.messages == before
    assert (desktop.context, desktop.maximum, desktop.thinking, desktop.system_prompt) == (8192, 512, False, None)
    assert not any(e["event"] == "configured" for e in events(desktop))


def test_advanced_options_reject_missing_fields_no_model_and_busy(desktop, model):
    with pytest.raises(DesktopError, match="Run a model"):
        desktop.dispatch(advanced_options())
    running(desktop, model)
    with pytest.raises(DesktopError, match="Invalid advanced options"):
        desktop.dispatch({"command": "configure"})
    desktop.dispatch({"command": "generate", "text": "wait fixture"})
    with pytest.raises(DesktopError, match="Wait for the current operation"):
        desktop.dispatch(advanced_options())
    desktop.dispatch({"command": "stop"})


@pytest.mark.parametrize("context_limit", [2, 512, 32768, 262144, 1048576])
def test_model_context_capacity_reaches_settings_and_api(desktop, model, context_limit):
    identifier = installed(desktop, model)
    with patch(
        "stark_convert.desktop.Worker.read",
        return_value={
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": context_limit,
            "max_new_tokens": 4096,
        },
    ):
        desktop.dispatch({"command": "run", "model_id": identifier})
        finish(desktop)
    ready = [event for event in events(desktop) if event["event"] == "ready"][-1]
    assert ready["max_context_tokens"] == context_limit
    assert ready["context_tokens"] == context_limit
    assert desktop.context == context_limit
    assert desktop.api_models()[0]["context_length"] == context_limit
    desktop.dispatch(advanced_options(context_tokens=context_limit, max_new_tokens=min(128, context_limit - 1)))
    assert desktop.context == context_limit
    with pytest.raises(DesktopError, match="Invalid advanced options"):
        desktop.dispatch(advanced_options(context_tokens=context_limit + 1))


def test_default_context_follows_model_changes_and_reaches_generation(desktop, model):
    from stark_convert.chat import Worker

    identifier = running(desktop, model)
    read = Worker.read
    for context_limit in [512, 32768, 262144, 1048576, 32768]:

        def ready(worker, limit=context_limit):
            reply = read(worker)
            reply["max_context_tokens"] = limit
            return reply

        with patch(
            "stark_convert.desktop.Worker.read",
            autospec=True,
            side_effect=ready,
        ):
            desktop.dispatch({"command": "run", "model_id": identifier})
            finish(desktop)
        assert desktop.context == context_limit
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.dispatch({"command": "generate", "text": "model default fixture"})
        finish(desktop)
    assert request.call_args_list[-1].args[0]["context_tokens"] == 32768
    assert any(event["event"] == "reply" for event in events(desktop))


def test_custom_context_survives_clamps_and_restore_defaults_tracks_models(desktop, model):
    identifier = running(desktop, model)
    desktop.dispatch(advanced_options(context_tokens=2048))

    def load(context_limit):
        with patch(
            "stark_convert.desktop.Worker.read",
            return_value={
                "ready": True,
                "mode": "text-generation-v1",
                "max_context_tokens": context_limit,
                "max_new_tokens": 4096,
            },
        ):
            desktop.dispatch({"command": "run", "model_id": identifier})
            finish(desktop)

    load(512)
    assert desktop.context == 512
    load(262144)
    assert desktop.context == 2048
    desktop.dispatch(advanced_options(context_tokens=262144))
    load(1048576)
    assert desktop.context == 1048576
    desktop.dispatch({"command": "reset"})
    assert desktop.context == 1048576
    desktop.dispatch({"command": "stop"})
    load(32768)
    assert desktop.context == 32768


def test_model_change_clamps_options_to_advertised_limits(desktop, model):
    identifier = running(desktop, model)
    desktop.dispatch(advanced_options())
    with patch(
        "stark_convert.desktop.Worker.read",
        return_value={
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": 512,
            "max_new_tokens": 64,
        },
    ):
        desktop.dispatch({"command": "run", "model_id": identifier})
        finish(desktop)
    assert (desktop.context, desktop.maximum) == (512, 64)
    ready = [e for e in events(desktop) if e["event"] == "ready"][-1]
    assert ready["context_tokens"] == 512 and ready["max_new_tokens"] == 64
    assert ready["max_context_tokens"] == 512 and ready["max_output_tokens"] == 64


def test_output_limit_metadata_reserves_prompt_capacity(desktop, model):
    identifier = installed(desktop, model)
    with patch(
        "stark_convert.desktop.Worker.read",
        return_value={
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": 512,
            "max_new_tokens": 4096,
        },
    ):
        desktop.dispatch({"command": "run", "model_id": identifier})
        finish(desktop)
    ready = [event for event in events(desktop) if event["event"] == "ready"][-1]
    assert ready["max_output_tokens"] == 511
    assert desktop.api_models()[0]["max_output_tokens"] == 511


def test_advanced_context_budget_rejection_leaves_history_unchanged(desktop, model):
    running(desktop, model)
    desktop.dispatch(advanced_options(context_tokens=21, max_new_tokens=2))
    before = list(desktop.messages)
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.dispatch({"command": "generate", "text": "budget fixture"})
        finish(desktop)
        assert [c.args[0]["operation"] for c in request.call_args_list] == ["tokenize"]
    assert desktop.messages == before


@pytest.fixture
def mtp_executable(tmp_path):
    path = tmp_path / "MTP fixture worker"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "model=Path(sys.argv[sys.argv.index('--model-path')+1])\n"
        "config=json.loads((model/'config.json').read_text())\n"
        "mtp=config.get('fixture_mtp',{'supported':True,'max_drafts':3,'recommended_drafts':2,"
        "'default_drafts':0,'sampling':'exact-rejection-v1','verification':'fixture-exact-v1',"
        "'operations':['generate']})\n"
        "print(json.dumps({'ready':True,'mode':'text-generation-v1','vocab_size':32000,"
        "'max_context_tokens':8192,'max_new_tokens':4096,'streaming':'text-delta-v1',"
        "'stop_sequences':True,'mtp':mtp}),flush=True)\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if request['operation']=='tokenize':\n"
        "        reply={'token_count':20,'token_ids':[2]*20}\n"
        "    else:\n"
        "        drafts=request.get('mtp_drafts',0)\n"
        "        reply={'text':'synthetic answer','input_tokens':20,'output_tokens':4,"
        "'finish_reason':'length','mtp':{'enabled':drafts>0,'drafts':drafts,"
        "'sampling':'exact-rejection-v1','verification':mtp.get('verification')}}\n"
        "        if request.get('stream'): print(json.dumps({'event':'delta','text':'synthetic answer'}),flush=True)\n"
        "    print(json.dumps(reply),flush=True)\n"
    )
    path.chmod(0o755)
    return path


def set_mtp(desktop, identifier, enabled):
    desktop.dispatch({"command": "configure_mtp", "model_id": identifier, "enabled": enabled})


@pytest.fixture
def timed_mtp_requests(desktop, model, mtp_executable, monkeypatch):
    """Give the real worker bridge distinct, deterministic MTP stage timings."""
    desktop.executable = mtp_executable
    identifier = running(desktop, model)
    request = desktop.worker.request
    drafts_seen = []

    def timed_request(payload, **options):
        reply = request(payload, **options)
        if payload["operation"] == "generate":
            drafts = payload.get("mtp_drafts", 0)
            drafts_seen.append(drafts)
            prefill, decode = (0.6, 0.03) if drafts else (0.2, 0.06)
            reply.update(sampled_tokens=4, prefill_seconds=prefill, decode_seconds=decode)
            if options.get("on_progress") is not None:
                frame = {
                    "event": "progress",
                    "input_tokens": 20,
                    "sampled_tokens": 0,
                    "output_token_budget": payload["max_new_tokens"],
                    "elapsed_seconds": 0.0,
                    "prefill_seconds": None,
                }
                for value in (
                    frame,
                    {**frame, "sampled_tokens": 1, "elapsed_seconds": prefill, "prefill_seconds": prefill},
                    {**frame, "sampled_tokens": 4, "elapsed_seconds": prefill + decode, "prefill_seconds": prefill},
                ):
                    options["on_progress"](value)
        return reply

    monkeypatch.setattr(desktop.worker, "request", timed_request)
    return identifier, drafts_seen


def test_playground_eta_profiles_follow_mtp_toggle_and_infer_selection(desktop, timed_mtp_requests):
    identifier, drafts_seen = timed_mtp_requests

    def generate(request_id):
        desktop.output.seek(0)
        desktop.output.truncate()
        desktop.dispatch({"command": "generate", "text": "MTP timing fixture", "request_id": request_id})
        finish(desktop)
        progress = [event for event in events(desktop) if event["event"] == "generation_progress"]
        assert [event["sampled_tokens"] for event in progress] == [0, 1, 4]
        return progress[0]

    assert not generate("off-first")["calibrated"]
    assert generate("off-second")["calibrated"]
    set_mtp(desktop, identifier, True)
    assert not generate("on-first")["calibrated"]
    on = generate("on-second")
    assert on["calibrated"]
    assert on["ttft_lower_seconds"] <= 0.6 <= on["ttft_upper_seconds"]
    set_mtp(desktop, identifier, False)
    off = generate("off-again")
    assert off["calibrated"]
    assert off["ttft_lower_seconds"] <= 0.2 <= off["ttft_upper_seconds"] < 0.6
    assert drafts_seen == [0, 0, 2, 2, 0]
    assert len(desktop.generation_estimates.history) == 2

    progress = []
    desktop.infer(
        [{"role": "user", "content": "Direct MTP timing fixture"}],
        16,
        mtp=True,
        on_progress=progress.append,
    )
    assert progress[0]["calibrated"]
    assert progress[0]["ttft_lower_seconds"] <= 0.6 <= progress[0]["ttft_upper_seconds"]
    assert drafts_seen[-1] == 2


@pytest.mark.parametrize("stream", [False, True])
def test_api_eta_profiles_follow_request_local_mtp_override(desktop, timed_mtp_requests, stream):
    identifier, drafts_seen = timed_mtp_requests
    set_mtp(desktop, identifier, True)
    payload = {"messages": [{"role": "user", "content": "API MTP timing fixture"}], "temperature": 0}
    for override in (False, True, False, None, False):
        request = {**payload, **({"mtp": override} if override is not None else {})}
        desktop.api_complete(request, on_delta=(lambda _: None) if stream else None)
    profiles = list(desktop.generation_estimates.history.values())
    assert len(profiles) == 2
    assert sorted(len(profile) for profile in profiles) == [2, 3]
    assert {tuple(sample.prefill_seconds for sample in profile) for profile in profiles} == {
        (0.2, 0.2, 0.2),
        (0.6, 0.6),
    }
    assert drafts_seen == [0, 2, 0, 2, 0]
    assert desktop.settings_summary()["mtp_enabled"] is True


@pytest.mark.parametrize("override, expected", [(None, 2), (True, 2), (False, 0)])
@pytest.mark.parametrize("stream", [False, True])
def test_tool_requests_preserve_mtp_selection_and_tool_rendering(tool_desktop, override, expected, stream):
    desktop, payload, seen, reply = tool_desktop
    capability = {
        "supported": True,
        "max_drafts": 3,
        "recommended_drafts": 2,
        "default_drafts": 0,
        "sampling": "exact-rejection-v1",
        "verification": "fixture-exact-v1",
        "operations": ["generate"],
    }
    desktop.worker.mtp = capability
    set_mtp(desktop, desktop.active_model, True)
    reply["mtp"] = {
        "enabled": expected > 0,
        "drafts": expected,
        "sampling": capability["sampling"],
        "verification": capability["verification"],
    }
    payload["stream"] = stream
    if override is not None:
        payload["mtp"] = override
    response = desktop.api_complete(payload, on_delta=(lambda _: None) if stream else None)
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(seen[0]["text"].split(";", 1)[0]) == payload["tools"]
    assert seen[-1]["mtp_drafts"] == expected
    assert desktop.api_models()[0]["tool_calling"] is True
    assert desktop.api_models()[0]["defaults"]["mtp"] is True
    assert desktop.worker is not None
    assert desktop.job is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_unknown_context_mtp_requests_reserve_only_request_capacity(desktop, model, mtp_executable, enabled, stream):
    from stark_convert.chat import Worker

    desktop.executable = mtp_executable
    read = Worker.read

    def ready(worker):
        result = read(worker)
        result["max_context_tokens"] = 0xFFFFFFFF
        return result

    with patch("stark_convert.desktop.Worker.read", autospec=True, side_effect=ready):
        identifier = running(desktop, model)
    set_mtp(desktop, identifier, enabled)
    worker = desktop.worker
    with patch.object(worker, "request", wraps=worker.request) as request:
        desktop.api_complete(
            {"messages": [{"role": "user", "content": "Synthetic MTP context request"}], "max_tokens": 32},
            on_delta=(lambda _: None) if stream else None,
        )
    generated = request.call_args_list[-1].args[0]
    assert generated["context_tokens"] == 20 + 32
    assert generated["mtp_drafts"] == (2 if enabled else 0)
    assert desktop.api_models()[0]["context_length"] is None
    assert desktop.api_models()[0]["defaults"]["mtp"] is enabled
    assert desktop.worker is worker
    assert desktop.job is None


def test_mtp_preference_is_per_model_persistent_and_does_not_reset_chat(desktop, model, mtp_executable, tmp_path):
    import shutil

    desktop.executable = mtp_executable
    first = running(desktop, model)
    process = desktop.worker.process
    before = list(desktop.messages)
    assert desktop.settings_summary()["mtp_enabled"] is False
    set_mtp(desktop, first, True)
    assert desktop.worker.process is process and desktop.messages == before
    assert desktop.settings_summary()["mtp_enabled"] is True
    assert json.loads(desktop.registry.read_text())[0]["mtp_enabled"] is True
    assert desktop.registry.stat().st_mode & 0o777 == 0o600
    assert events(desktop)[-1]["event"] == "mtp_configured"

    second_model = tmp_path / "second model"
    shutil.copytree(model, second_model)
    config = json.loads((second_model / "config.json").read_text())
    config["fixture_mtp"] = {
        "supported": True,
        "max_drafts": 6,
        "recommended_drafts": 4,
        "default_drafts": 0,
        "sampling": "exact-rejection-v1",
        "verification": "second-exact-v1",
        "operations": ["generate"],
    }
    (second_model / "config.json").write_text(json.dumps(config))
    desktop.dispatch({"command": "install", "kind": "local", "source": str(second_model)})
    finish(desktop)
    second = next(m["id"] for m in desktop.models if m["id"] != first)
    desktop.dispatch({"command": "run", "model_id": second})
    finish(desktop)
    assert desktop.settings_summary()["mtp_enabled"] is False
    assert desktop.settings_summary()["mtp"]["recommended_drafts"] == 4
    set_mtp(desktop, second, True)
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.api_complete({"messages": [{"role": "user", "content": "fixture"}]})
        assert request.call_args_list[-1].args[0]["mtp_drafts"] == 4
    set_mtp(desktop, second, False)
    desktop.stop(quiet=True)
    assert desktop.api_status()["capabilities"]["mtp"]["supported"] is False

    replacement = Desktop(mtp_executable, desktop.data_dir, io.StringIO(), timeout=2)
    try:
        for identifier, expected in ((second, False), (first, True)):
            replacement.dispatch({"command": "run", "model_id": identifier})
            finish(replacement)
            assert replacement.settings_summary()["mtp_enabled"] is expected
        assert replacement.api_models()[0]["defaults"]["mtp"] is True
        assert replacement.api_models()[0]["capabilities"]["mtp"]["recommended_drafts"] == 2
    finally:
        replacement.stop(quiet=True)


def test_mtp_api_overrides_are_request_local_and_streaming_uses_same_selection(desktop, model, mtp_executable):
    desktop.executable = mtp_executable
    identifier = running(desktop, model)
    set_mtp(desktop, identifier, True)
    payload = {"messages": [{"role": "user", "content": "fixture"}]}
    for override, expected in ((None, 2), (False, 0), (True, 2), (None, 2)):
        request_body = {**payload, **({"mtp": override} if override is not None else {})}
        with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
            desktop.api_complete(request_body, on_delta=lambda _: None)
            assert request.call_args_list[-1].args[0]["mtp_drafts"] == expected
    assert desktop.models[0]["mtp_enabled"] is True
    assert desktop.api_status()["defaults"]["mtp"] is True
    set_mtp(desktop, identifier, False)
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.dispatch({"command": "generate", "text": "fixture"})
        finish(desktop)
        assert request.call_args_list[-1].args[0]["mtp_drafts"] == 0


def test_unsupported_mtp_is_rejected_without_touching_worker_and_false_is_compatible(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process
    before = desktop.registry.read_bytes()
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        with pytest.raises(DesktopError, match="not supported"):
            set_mtp(desktop, identifier, True)
        with pytest.raises(APIError) as error:
            desktop.api_complete({"messages": [{"role": "user", "content": "fixture"}], "mtp": True})
        assert error.value.code == "mtp_not_supported"
        assert request.call_count == 0
    assert desktop.registry.read_bytes() == before
    assert desktop.worker.process is process
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.api_complete({"messages": [{"role": "user", "content": "fixture"}], "mtp": False})
        assert "mtp_drafts" not in request.call_args_list[-1].args[0]


def test_mtp_preference_write_failure_stale_model_and_busy_changes_are_atomic(desktop, model, mtp_executable):
    desktop.executable = mtp_executable
    identifier = running(desktop, model)
    before = desktop.registry.read_bytes()
    with patch.object(desktop, "persist", side_effect=OSError("synthetic private failure")):
        with pytest.raises(DesktopError, match="Cannot save"):
            set_mtp(desktop, identifier, True)
    assert "mtp_enabled" not in desktop.models[0]
    assert desktop.registry.read_bytes() == before
    with pytest.raises(DesktopError, match="Model changed"):
        set_mtp(desktop, "stale-model", True)
    desktop.job = object()
    try:
        with pytest.raises(DesktopError, match="Wait for"):
            set_mtp(desktop, identifier, True)
    finally:
        desktop.job = None
    desktop.api_queue.append(object())
    try:
        with pytest.raises(DesktopError, match="Wait for"):
            set_mtp(desktop, identifier, True)
    finally:
        desktop.api_queue.clear()
    assert desktop.settings_summary()["mtp_enabled"] is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_mtp_configuration_requires_a_boolean(desktop, model, mtp_executable, invalid):
    desktop.executable = mtp_executable
    identifier = running(desktop, model)
    with pytest.raises(DesktopError, match="Invalid MTP"):
        set_mtp(desktop, identifier, invalid)
    assert "mtp_enabled" not in desktop.models[0]


def test_saved_mtp_preference_is_inactive_when_replacement_worker_lacks_support(desktop, model, mtp_executable):
    desktop.executable = mtp_executable
    identifier = running(desktop, model)
    set_mtp(desktop, identifier, True)
    config = json.loads((model / "config.json").read_text())
    config["fixture_mtp"] = {"supported": False}
    (model / "config.json").write_text(json.dumps(config))
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.models[0]["mtp_enabled"] is True
    assert desktop.settings_summary()["mtp_enabled"] is False
    assert desktop.api_status()["defaults"]["mtp"] is False
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.api_complete({"messages": [{"role": "user", "content": "fixture"}]})
        assert "mtp_drafts" not in request.call_args_list[-1].args[0]


def test_image_requests_keep_text_preference_and_reject_explicit_mtp_before_dispatch(desktop, model, mtp_executable):
    import base64

    from stark_convert.local_api import APIError

    desktop.executable = mtp_executable
    identifier = running(desktop, model)
    set_mtp(desktop, identifier, True)
    desktop.worker.images = True
    image_url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nsynthetic").decode()
    payload = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}]}
    with patch.object(
        desktop.worker, "request_image", return_value={"text": "fixture", "finish_reason": "length"}
    ) as image:
        desktop.api_complete(payload)
        assert "mtp_drafts" not in image.call_args.args[2]
        with pytest.raises(APIError) as error:
            desktop.api_complete({**payload, "mtp": True})
        assert error.value.code == "mtp_not_supported"
        assert image.call_count == 1
    assert desktop.models[0]["mtp_enabled"] is True
    assert desktop.api_status()["defaults"]["mtp"] is True


@pytest.mark.parametrize("enabled", [True, False])
def test_mtp_acknowledgement_mismatch_fences_worker(desktop, model, mtp_executable, enabled):
    from stark_convert.local_api import APIError

    desktop.executable = mtp_executable
    running(desktop, model)
    original = desktop.worker.request
    process = desktop.worker.process

    def mismatch(payload, **options):
        reply = original(payload, **options)
        if payload["operation"] == "generate":
            reply["mtp"]["enabled"] = not enabled
        return reply

    with patch.object(desktop.worker, "request", side_effect=mismatch):
        with pytest.raises(APIError) as error:
            desktop.api_complete({"messages": [{"role": "user", "content": "fixture"}], "mtp": enabled})
    assert error.value.code == "generation_failed"
    assert process.poll() is not None
    assert desktop.api_models() == []
    assert desktop.api_status()["capabilities"]["mtp"]["supported"] is False


def test_local_registration_is_private_atomic_persistent_and_deduplicated(desktop, model):
    identifier = installed(desktop, model)
    assert model.is_dir()
    assert desktop.models[0]["path"] == str(model)
    assert desktop.registry.stat().st_mode & 0o777 == 0o600
    assert desktop.data_dir.stat().st_mode & 0o777 == 0o700
    assert desktop.model_dir.stat().st_mode & 0o777 == 0o700
    assert not list(desktop.data_dir.glob(".models-*"))
    assert installed(desktop, model) == identifier
    replacement = Desktop(desktop.executable, desktop.data_dir, io.StringIO())
    assert replacement.models == desktop.models
    assert set(json.loads(desktop.registry.read_text())[0]) == {"id", "name", "path", "source"}


def test_generation_metrics_history_reset_and_stop(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "synthetic request"})
    finish(desktop)
    reply = next(e for e in events(desktop) if e["event"] == "reply")
    assert reply["text"] == "synthetic answer"
    assert reply["input_tokens_per_second"] == 100.0
    assert reply["output_tokens_per_second"] == 50.0
    metrics_event = next(e for e in events(desktop) if e["event"] == "metrics")
    assert metrics_event["input_tokens"] == 20
    assert metrics_event["output_tokens"] == 4
    assert "text" not in metrics_event
    assert len(desktop.messages) == 3
    assert "synthetic request" not in desktop.registry.read_text()
    desktop.dispatch({"command": "reset"})
    assert desktop.messages == [{"role": "system", "content": "synthetic system"}]
    desktop.dispatch({"command": "stop"})
    assert process.poll() is not None
    assert desktop.worker is None
    assert desktop.messages == []


def test_stop_interrupts_generation_then_restart_uses_a_new_worker(desktop, model):
    identifier = running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "wait fixture"})
    started = time.monotonic()
    desktop.dispatch({"command": "stop"})
    assert time.monotonic() - started < 5
    assert process.poll() is not None
    assert not any(e["event"] == "reply" for e in events(desktop))
    assert not any(e["event"] == "metrics" for e in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.worker.process.pid != process.pid
    assert desktop.active_model == identifier


def test_worker_failure_fences_process_without_private_diagnostics(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "failure fixture"})
    finish(desktop)
    assert process.poll() is not None
    assert desktop.worker is None
    assert "failure fixture" not in desktop.output.getvalue()
    assert any(e["event"] == "error" for e in events(desktop))


def test_idle_worker_exit_is_reported_and_reaped(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    process.kill()
    process.wait()
    desktop.monitor()
    assert desktop.worker is None
    assert events(desktop)[-1]["state"] == "error"


def test_raw_conversion_is_offline_and_cancel_reaps_converter(desktop, tmp_path, converter_support):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    commands = []
    original = subprocess.Popen

    def spawn(command, **kwargs):
        commands.append(command)
        return original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    with patch("stark_convert.desktop.subprocess.Popen", side_effect=spawn):
        desktop.dispatch({"command": "install", "kind": "local", "source": str(checkpoint)})
        deadline = time.monotonic() + 3
        while desktop.converter is None and time.monotonic() < deadline:
            time.sleep(0.01)
        process = desktop.converter
        assert process is not None
        desktop.stop()
    assert process.poll() is not None
    assert len(converter_support) == len(commands) == 1 and "--preflight" not in commands[0]
    assert "--offline" in commands[0]
    assert "hf-generic" in commands[0]
    assert desktop.models == []
    assert not list(desktop.model_dir.iterdir())


@pytest.mark.parametrize(
    "repo, generic", [("example/model", True), ("synthetic-model", True), ("Qwen/Qwen3-0.6B", False)]
)
@pytest.mark.parametrize(
    "source", ["{repo}", "https://huggingface.co/{repo}", " https://huggingface.co/{repo}/?download=true#model-card "]
)
def test_huggingface_conversion_publishes_complete_model(desktop, model, repo, generic, source, converter_support):
    original = subprocess.Popen
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        target = command[command.index("--output") + 1]
        return original(
            [sys.executable, "-c", "import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2])", str(model), target],
            **kwargs,
        )

    with patch("stark_convert.desktop.subprocess.Popen", side_effect=spawn):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": source.format(repo=repo)})
        finish(desktop)
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": repo})
        finish(desktop)
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": f"https://huggingface.co/{repo}"})
        finish(desktop)
    assert len(commands) == 1
    assert len(converter_support) == 1
    assert commands[0][commands[0].index("--revision") + 1] == "a" * 40
    assert commands[0][commands[0].index("stark_convert.cli") + 1] == repo
    assert "--offline" not in commands[0]
    assert ("--metadata-profile" in commands[0]) is generic
    assert ("hf-generic" in commands[0]) is generic
    assert len(desktop.models) == 1
    assert desktop.models[0]["source"] == "huggingface:" + repo
    assert desktop.models[0]["name"] == repo.split("/")[-1]
    assert Path(desktop.models[0]["path"]).parent == desktop.model_dir
    assert not list(desktop.model_dir.glob(".install-*"))


def progress_events(desktop, phase=None):
    return [e for e in events(desktop) if e["event"] == "progress" and (phase is None or e["phase"] == phase)]


def progress_spawn(model, lines, *, exit_code=0, pause=False):
    original = subprocess.Popen

    def spawn(command, **kwargs):
        descriptor = command[command.index("--progress-fd") + 1]
        target = command[command.index("--output") + 1]
        script = (
            "import os,shutil,sys,time\n"
            "stream=os.fdopen(int(sys.argv[1]),'wb',buffering=0)\n"
            "for line in " + repr(lines) + ": stream.write(line+b'\\n')\n"
            "print('private diagnostic fixture',flush=True)\n"
        )
        if pause:
            script += "time.sleep(30)\n"
        if exit_code == 0:
            script += "shutil.copytree(sys.argv[2],sys.argv[3])\n"
        script += f"raise SystemExit({exit_code})\n"
        return original([sys.executable, "-c", script, descriptor, str(model), target], **kwargs)

    return spawn


def progress_line(phase, state, completed, total):
    return json.dumps({"phase": phase, "state": state, "completed_bytes": completed, "total_bytes": total}).encode()


def test_converter_progress_is_numeric_private_bounded_and_completed_after_publication(
    desktop, model, converter_support
):
    lines = [
        b"{malformed",
        b"private fixture" * 1000,
        json.dumps({"phase": "download", "state": ["private fixture"]}).encode(),
        json.dumps(
            {
                "phase": "download",
                "state": "running",
                "completed_bytes": 1,
                "total_bytes": 2,
                "private": "private fixture",
            }
        ).encode(),
        progress_line("download", "running", 10, 100),
        progress_line("download", "running", 50, 100),
        progress_line("download", "running", 20, 100),
        progress_line("download", "complete", 100, 100),
        progress_line("conversion", "running", 100, 200),
        progress_line("conversion", "complete", 200, 200),
    ]
    original_persist = desktop.persist

    def persist():
        assert not any(e["state"] == "complete" for e in progress_events(desktop, "conversion"))
        original_persist()

    with (
        patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines)),
        patch.object(desktop, "persist", side_effect=persist),
    ):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    assert len(desktop.models) == 1
    assert "private fixture" not in desktop.output.getvalue()
    assert "private diagnostic" not in desktop.output.getvalue()
    download = progress_events(desktop, "download")
    assert [e["completed_bytes"] for e in download] == [0, 10, 50, 100]
    conversion = progress_events(desktop, "conversion")
    assert [e["state"] for e in conversion] == ["waiting", "running", "running", "complete"]
    assert conversion[-1]["completed_bytes"] == conversion[-1]["total_bytes"] == 200
    for event in progress_events(desktop):
        assert set(event) == {"event", "phase", "state", "completed_bytes", "total_bytes"}
    all_events = events(desktop)
    assert all_events.index(conversion[-1]) < next(i for i, e in enumerate(all_events) if e["event"] == "installed")


def test_progress_skips_local_converted_and_duplicate_models(desktop, model):
    installed(desktop, model)
    assert [e["state"] for e in progress_events(desktop, "download")] == ["skipped"]
    assert progress_events(desktop, "conversion")[-1]["state"] == "skipped"
    desktop.output = io.StringIO()
    installed(desktop, model)
    assert progress_events(desktop, "download")[-1]["state"] == "skipped"
    assert progress_events(desktop, "conversion")[-1]["state"] == "skipped"


def test_failed_conversion_does_not_complete_progress_or_publish_model(desktop, model, converter_support):
    lines = [progress_line("download", "complete", 100, 100), progress_line("conversion", "running", 5, 100)]
    with patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines, exit_code=2)):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    assert not desktop.models
    assert progress_events(desktop, "conversion")[-1]["state"] == "failed"
    assert not any(e["state"] == "complete" for e in progress_events(desktop, "conversion"))
    assert not list(desktop.model_dir.iterdir())


def test_registry_failure_keeps_conversion_progress_failed(desktop, model, converter_support):
    lines = [progress_line("download", "complete", 100, 100), progress_line("conversion", "complete", 200, 200)]
    with (
        patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines)),
        patch.object(desktop, "persist", side_effect=OSError("private fixture")),
    ):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    assert not desktop.models
    assert progress_events(desktop, "conversion")[-1]["state"] == "failed"
    assert not any(e["event"] == "installed" for e in events(desktop))
    assert "private fixture" not in desktop.output.getvalue()
    assert not list(desktop.model_dir.iterdir())


def test_cancelled_progress_reaps_converter_and_cannot_emit_late_completion(desktop, model, converter_support):
    lines = [progress_line("download", "running", 5, 100)]
    with patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines, pause=True)):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        deadline = time.monotonic() + 3
        while not any(e["completed_bytes"] == 5 for e in progress_events(desktop)) and time.monotonic() < deadline:
            time.sleep(0.01)
        process = desktop.converter
        old_epoch = desktop.epoch
        assert process is not None
        desktop.stop()
        desktop.progress(old_epoch, "download", "complete", 100, 100)
    assert process.poll() is not None
    assert desktop.converter is None
    assert not desktop.models
    assert progress_events(desktop, "download")[-1]["state"] == "cancelled"
    assert progress_events(desktop, "conversion")[-1]["state"] == "cancelled"
    assert not any(e["state"] == "complete" for e in progress_events(desktop))
    assert not list(desktop.model_dir.iterdir())


def test_progress_pipe_descriptors_close_when_converter_spawn_fails(desktop, model, converter_support):
    original_pipe = os.pipe
    descriptors = []

    def pipe():
        result = original_pipe()
        descriptors.extend(result)
        return result

    with (
        patch("stark_convert.desktop.os.pipe", side_effect=pipe),
        patch("stark_convert.desktop.subprocess.Popen", side_effect=OSError("private fixture")),
    ):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert progress_events(desktop, "download")[-1]["state"] == "failed"
    assert "private fixture" not in desktop.output.getvalue()
    assert not list(desktop.model_dir.iterdir())


def test_exited_converter_cannot_leave_a_descendant_holding_progress_pipe(desktop, model, tmp_path, converter_support):
    original = subprocess.Popen
    child_pid_file = tmp_path / "descendant.pid"
    child_pid = None

    def spawn(command, **kwargs):
        descriptor = command[command.index("--progress-fd") + 1]
        target = command[command.index("--output") + 1]
        script = (
            "import os,shutil,subprocess,sys\n"
            "from pathlib import Path\n"
            "fd=int(sys.argv[1])\n"
            "code='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'\n"
            "child=subprocess.Popen([sys.executable,'-c',code],pass_fds=(fd,))\n"
            "Path(sys.argv[4]).write_text(str(child.pid))\n"
            "shutil.copytree(sys.argv[2],sys.argv[3])\n"
        )
        return original([sys.executable, "-c", script, descriptor, str(model), target, str(child_pid_file)], **kwargs)

    try:
        with patch("stark_convert.desktop.subprocess.Popen", side_effect=spawn):
            desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
            finish(desktop)
        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 3
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        assert len(desktop.models) == 1
        assert desktop.converter is None
    finally:
        if child_pid is None and child_pid_file.exists():
            child_pid = int(child_pid_file.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    "patch_value",
    [
        {"phase": []},
        {"phase": "other"},
        {"state": []},
        {"state": "other"},
        {"completed_bytes": True},
        {"completed_bytes": -1},
        {"completed_bytes": 3.5},
        {"completed_bytes": 11},
        {"total_bytes": False},
        {"total_bytes": float("nan")},
        {"total_bytes": -1},
        {"state": "complete"},
        {"private": "private fixture"},
    ],
)
def test_converter_progress_rejects_invalid_schema(patch_value):
    value = {"phase": "download", "state": "running", "completed_bytes": 5, "total_bytes": 10, **patch_value}
    assert converter_progress(value) is None


@pytest.mark.parametrize("payload", [None, [], {}, {"command": "wat"}, {"command": "generate", "text": 3}])
def test_malformed_commands_are_rejected(desktop, payload):
    with pytest.raises(DesktopError, match="Invalid app command"):
        desktop.dispatch(payload)


def test_missing_local_and_invalid_hf_sources_report_fixed_errors(desktop):
    for kind, source in [("local", "/missing synthetic fixture"), ("huggingface", "https://invalid/example")]:
        desktop.dispatch({"command": "install", "kind": kind, "source": source})
        finish(desktop)
    assert len([e for e in events(desktop) if e["event"] == "error"]) == 2
    assert desktop.models == []


def test_old_or_nonfinite_metrics_are_unavailable():
    for value in [None, float("inf"), float("nan"), -1, True, "200"]:
        assert metric({"speed": value}, "speed") is None
    assert metric({}, "speed") is None
    assert metric({"speed": 0}, "speed") == 0


def test_disappearing_event_consumer_does_not_expose_exception_details(desktop):
    desktop.output.close()
    desktop.library()
    assert desktop.output_failed


@pytest.mark.parametrize("exit_mode", ["eof", "sigterm"])
def test_app_exit_reaps_active_native_worker(tmp_path, model, executable, exit_mode):
    library = tmp_path / "exit library"
    prepare = Desktop(executable, library, io.StringIO())
    installed(prepare, model)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "stark_convert.desktop",
            "--worker",
            str(executable),
            "--data-dir",
            str(library),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert json.loads(process.stdout.readline())["event"] == "models"
        process.stdout.readline()
        process.stdin.write(json.dumps({"command": "run", "model_id": prepare.models[0]["id"]}) + "\n")
        process.stdin.flush()
        while json.loads(process.stdout.readline())["event"] != "ready":
            pass
        children = psutil.Process(process.pid).children()
        assert len(children) == 1
        child_pid = children[0].pid
        if exit_mode == "eof":
            process.stdin.close()
        else:
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert process.stderr.read() == ""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
