"""Chat contracts with synthetic messages and a private fake worker."""

import io
import json
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest

from stark_convert.chat import ChatError, ChatRequestTooLarge, Worker, load_template, main, run_chat, terminal_text


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "token_codec_config": {
                    "type": "ChatCodecConfig",
                    "bos_token": "<bos>",
                    "default_system_prompt": "system fixture",
                    "prompt_template": (
                        "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"
                    ),
                }
            }
        )
    )
    return tmp_path


class FakeWorker:
    def __init__(self, *, context_limit=8192):
        self.requests = []
        self.context_limit = context_limit

    def read(self):
        return {
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": self.context_limit,
            "max_new_tokens": 6144,
            "generation_budget": "memory-context-v1",
        }

    def request(self, payload):
        self.requests.append(payload)
        if payload["operation"] == "plan_generation":
            context = min(payload["context_tokens"], payload["input_tokens"] + 6144)
            return {
                "context_tokens": context,
                "max_new_tokens": context - payload["input_tokens"],
                "context_capacity_known": self.context_limit != 0xFFFFFFFF,
                "budget_kind": "memory",
            }
        return {"token_count": 20} if payload["operation"] == "tokenize" else {"text": "reply fixture"}


def options(model, **kwargs):
    values = dict(model_path=model, context_tokens=1024, max_new_tokens=32, system=None, thinking=False)
    values.update(kwargs)
    return Namespace(**values)


@pytest.mark.parametrize("context_limit", [512, 32768, 262144, 1048576])
def test_default_context_uses_the_loaded_model_maximum(model, context_limit):
    worker = FakeWorker(context_limit=context_limit)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=None), worker) == 0
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert len(generated) == 1
    assert generated[0]["context_tokens"] == context_limit


def test_explicit_context_override_is_preserved(model):
    worker = FakeWorker(context_limit=262144)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=2048), worker) == 0
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert len(generated) == 1
    assert generated[0]["context_tokens"] == 2048


def test_explicit_context_above_the_model_maximum_is_rejected_before_requests(model):
    worker = FakeWorker(context_limit=32768)
    with pytest.raises(ChatError, match="Requested token limits exceed"):
        run_chat(options(model, context_tokens=32769), worker)
    assert worker.requests == []


def test_context_help_describes_the_model_default(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "loaded model's maximum" in capsys.readouterr().out


@pytest.mark.parametrize("budget", [None, 2048])
def test_unknown_context_capacity_reserves_actual_cli_text_request(model, budget):
    worker = FakeWorker(context_limit=0xFFFFFFFF)
    with patch("sys.stdin", io.StringIO("synthetic turn\nsynthetic follow-up\n/exit\n")):
        assert run_chat(options(model, context_tokens=budget), worker) == 0
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert len(generated) == 2
    assert all(request["context_tokens"] == 20 + 32 for request in generated)


def test_unknown_context_capacity_cli_still_enforces_custom_budget(model, capsys):
    worker = FakeWorker(context_limit=0xFFFFFFFF)
    with patch("sys.stdin", io.StringIO("synthetic overflow\n/exit\n")):
        assert run_chat(options(model, context_tokens=21), worker) == 0
    assert all(request["operation"] != "generate" for request in worker.requests)
    assert "Conversation exceeds the context budget" in capsys.readouterr().out


def test_automatic_cli_output_uses_remaining_finite_context(model):
    worker = FakeWorker(context_limit=512)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=None, max_new_tokens=None), worker) == 0
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert generated[0]["max_new_tokens"] == 492
    assert generated[0]["context_tokens"] == 512


def test_automatic_cli_output_uses_memory_admission_below_finite_capacity(model):
    worker = FakeWorker(context_limit=32768)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=None, max_new_tokens=None), worker) == 0
    planned = [request for request in worker.requests if request["operation"] == "plan_generation"]
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert planned == [{"operation": "plan_generation", "input_tokens": 20, "context_tokens": 32768}]
    assert generated[0]["context_tokens"] == 6164
    assert generated[0]["max_new_tokens"] == 6144


def test_legacy_finite_worker_uses_remaining_context_without_planner(model):
    worker = FakeWorker(context_limit=512)
    read = worker.read
    worker.read = lambda: {key: value for key, value in read().items() if key != "generation_budget"}
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=None, max_new_tokens=None), worker) == 0
    assert not any(request["operation"] == "plan_generation" for request in worker.requests)
    assert worker.requests[-1]["max_new_tokens"] == 492


def test_automatic_cli_output_uses_custom_budget_for_unknown_context(model):
    worker = FakeWorker(context_limit=0xFFFFFFFF)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=2048, max_new_tokens=None), worker) == 0
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert generated[0]["max_new_tokens"] == 2028
    assert generated[0]["context_tokens"] == 2048


def test_automatic_cli_output_plans_unknown_context_without_user_options(model):
    worker = FakeWorker(context_limit=0xFFFFFFFF)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, context_tokens=None, max_new_tokens=None), worker) == 0
    generated = [request for request in worker.requests if request["operation"] == "generate"]
    assert generated[0]["context_tokens"] == 6164
    assert generated[0]["max_new_tokens"] == 6144


@pytest.mark.parametrize(
    "invalid",
    [
        {"context_tokens": True},
        {"context_tokens": 0xFFFFFFFF, "max_new_tokens": 0xFFFFFFFF - 20},
        {"context_tokens": 9000, "max_new_tokens": 8980},
        {"max_new_tokens": True},
        {"max_new_tokens": 1},
        {"budget_kind": "model"},
    ],
)
@pytest.mark.parametrize("known", [False, True])
def test_automatic_budget_rejects_unsafe_or_inconsistent_worker_allowances(invalid, known):
    from types import SimpleNamespace

    from stark_convert.chat import automatic_generation_budget

    response = {
        "context_tokens": 8192,
        "max_new_tokens": 8172,
        "context_capacity_known": known,
        "budget_kind": "memory",
        **invalid,
    }
    worker = SimpleNamespace(generation_budget=True, max_new_tokens=8192, request=lambda *_args, **_kwargs: response)
    with pytest.raises(ChatError, match="Invalid automatic execution budget"):
        automatic_generation_budget(worker, 20, 8192, context_capacity_known=known)


@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("reported", [False, True, 0, 1, None])
def test_automatic_budget_requires_exact_capacity_contract(known, reported):
    from types import SimpleNamespace

    from stark_convert.chat import automatic_generation_budget

    response = {
        "context_tokens": 8192,
        "max_new_tokens": 8172,
        "context_capacity_known": reported,
        "budget_kind": "memory",
    }
    worker = SimpleNamespace(generation_budget=True, max_new_tokens=8192, request=lambda *_args, **_kwargs: response)
    if reported is known:
        assert automatic_generation_budget(worker, 20, 8192, context_capacity_known=known) == (8192, 8172)
    else:
        with pytest.raises(ChatError, match="Invalid automatic execution budget"):
            automatic_generation_budget(worker, 20, 8192, context_capacity_known=known)


def test_history_reset_and_eof(model):
    worker = FakeWorker()
    with patch("sys.stdin", io.StringIO("first\nsecond\n/reset\nthird\n")):
        assert run_chat(options(model), worker) == 0
    prompts = [r["rendered_prompt"] for r in worker.requests if r["operation"] == "generate"]
    assert prompts == [
        "<bos>system:system fixture;user:first;assistant:",
        "<bos>system:system fixture;user:first;assistant:reply fixture;user:second;assistant:",
        "<bos>system:system fixture;user:third;assistant:",
    ]


def test_overflow_rejects_turn_without_mutating_history(model):
    worker = FakeWorker()
    request = worker.request

    def respond(payload):
        if payload["operation"] == "tokenize" and "oversized" in payload["text"]:
            return {"token_count": 1024}
        return request(payload)

    worker.request = respond
    with patch("sys.stdin", io.StringIO("oversized\nsmall\n/exit\n")):
        run_chat(options(model), worker)
    prompts = [r["rendered_prompt"] for r in worker.requests if r["operation"] == "generate"]
    assert len(prompts) == 1
    assert "oversized" not in prompts[0]


def test_bad_metadata_and_sandbox(model):
    (model / "config.json").write_text("{}")
    with pytest.raises(ChatError, match="chat metadata"):
        load_template(model)
    (model / "config.json").write_text(
        json.dumps(
            {
                "token_codec_config": {
                    "type": "ChatCodecConfig",
                    "prompt_template": "{{ ''.__class__.__mro__ }}",
                }
            }
        )
    )
    # The native evaluator rejects reflective access during admission.
    with pytest.raises(ChatError, match="chat metadata"):
        load_template(model)


def test_template_loopcontrols_compile_and_run(model):
    (model / "config.json").write_text(
        json.dumps(
            {
                "token_codec_config": {
                    "type": "ChatCodecConfig",
                    "prompt_template": (
                        "{% for item in messages %}"
                        "{% if item == 'skip' %}{% continue %}{% endif %}"
                        "{% if item == 'stop' %}{% break %}{% endif %}"
                        "{{ item }}{% endfor %}"
                    ),
                }
            }
        )
    )
    _, template = load_template(model)
    assert template.render(messages=["a", "skip", "b", "stop", "unreachable"]) == "ab"


def test_template_syntax_error_is_content_free(model):
    private_source = "{% break %} private template detail"
    (model / "config.json").write_text(
        json.dumps({"token_codec_config": {"type": "ChatCodecConfig", "prompt_template": private_source}})
    )
    with pytest.raises(ChatError, match="Cannot read chat metadata") as error:
        load_template(model)
    assert private_source not in str(error.value)


def test_worker_real_pipes_timeout_and_cleanup(tmp_path):
    executable = tmp_path / "fake worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request.get('wait'): time.sleep(10)\n"
        "    print(json.dumps({'text': request.get('text')}), flush=True)\n"
    )
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 5)
    try:
        assert worker.read() == {"ready": True}
        assert worker.request({"text": "synthetic fixture"}) == {"text": "synthetic fixture"}
        worker.timeout = 0.2
        with pytest.raises(ChatError, match="timed out"):
            worker.request({"wait": True})
    finally:
        worker.close()
    assert worker.process.poll() is not None


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_worker_requests_text_prefix_reuse_for_both_artifact_types_but_trusts_readiness(tmp_path, vision, supported):
    if vision:
        (tmp_path / "config.json").write_text(json.dumps({"type": "Qwen35VisionLanguageModelConfig"}))
    executable = tmp_path / "capability worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, stat, sys\n"
        "assert sys.argv.count('--prefix-cache') == 1\n"
        f"assert ('--vision-staging-root' in sys.argv) is {vision!r}\n"
        "if '--vision-staging-root' in sys.argv:\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('--vision-staging-root') + 1])\n"
        "    assert stat.S_IMODE(root.stat().st_mode) == 0o700 and not list(root.iterdir())\n"
        f"supported = {supported!r}\n"
        "print(json.dumps({'ready': True, 'operations': ['generate', 'reset_cache'],\n"
        "    'state_policy': 'consecutive-prefix-v1' if supported else 'fresh-per-request',\n"
        "    'prefix_cache': {'supported': supported, 'enabled': supported, 'chunk_tokens': 1024}}), flush=True)\n"
        "for _line in sys.stdin: pass\n"
    )
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 5)
    staging = worker.media.root if vision else None
    try:
        assert worker.read()["ready"] is True
        assert worker.prefix_cache is supported
        assert worker.prefix_chunk_tokens == (1024 if supported else None)
        assert worker.images is False
    finally:
        worker.close()
    if staging is not None:
        assert not staging.exists()


def test_main_suppresses_exception_payload_and_cleans_up(model, capsys):
    worker = FakeWorker()
    worker.close = lambda: None
    with (
        patch("stark_convert.chat.Worker", return_value=worker),
        patch("stark_convert.chat.run_chat", side_effect=RuntimeError("private marker")),
        patch.object(worker, "close") as close,
    ):
        assert main(["--model-path", str(model)]) == 1
        close.assert_called_once()
    output = capsys.readouterr()
    assert "private marker" not in output.out + output.err


def test_terminal_control_sequences_are_inert():
    assert terminal_text("hello\x1b[2J\r\x00\nworld") == "hello[2J\nworld"


@pytest.mark.parametrize("reply", ["not json", "[]", ""])
def test_worker_rejects_invalid_reply_or_exit(tmp_path, reply):
    executable = tmp_path / "invalid worker"
    executable.write_text(f"#!{sys.executable}\nprint({reply!r}, flush=True)\n")
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 2)
    try:
        with pytest.raises(ChatError):
            worker.read()
    finally:
        worker.close()
    assert worker.process.poll() is not None


def test_request_limit_is_checked_before_write(tmp_path, monkeypatch):
    from stark_convert import chat

    monkeypatch.setattr(chat, "WIRE_BYTES", 128)
    worker = Worker(sys.executable, tmp_path, 2)
    try:
        with pytest.raises(ChatRequestTooLarge, match="request limit"):
            worker.request({"text": "x" * 256})
    finally:
        worker.close()


@pytest.mark.parametrize("unit", ["x", "界", "\n"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_worker_request_limit_counts_encoded_bytes_and_newline(unit, over_limit):
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker.read = lambda: {"accepted": True}
    from stark_convert import chat

    limit = chat.WIRE_BYTES
    overhead = len(json.dumps({"text": ""}).encode()) + 1
    unit_bytes = len(json.dumps(unit, ensure_ascii=False).encode()) - 2
    count, padding = divmod(limit - overhead + int(over_limit), unit_bytes)
    payload = {"text": unit * count + "x" * padding}
    if over_limit:
        with pytest.raises(ChatRequestTooLarge):
            worker.request(payload)
        assert worker.process.stdin.tell() == 0
    else:
        assert worker.request(payload) == {"accepted": True}
        assert worker.process.stdin.tell() == limit


def test_worker_request_respects_smaller_advertised_limit(monkeypatch):
    from types import SimpleNamespace

    from stark_convert import chat

    monkeypatch.setattr(chat, "WIRE_BYTES", 128)
    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker.read = lambda: {"accepted": True}
    worker.max_request_bytes = 64
    with pytest.raises(ChatRequestTooLarge):
        worker.request({"text": "x" * 100})
    assert worker.process.stdin.tell() == 0


@pytest.mark.parametrize("error", ["request_too_large", "resource_budget_exceeded"])
def test_worker_resource_budget_reply_is_recoverable(error):
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker.read = lambda: {"error": error}
    with pytest.raises(ChatRequestTooLarge):
        worker.request({"text": "short request"})


def test_startup_failure_does_not_echo_worker_diagnostics(model, capsys):
    worker = FakeWorker()
    worker.read = lambda: {"ready": False, "error": "private marker"}
    with pytest.raises(ChatError, match="could not load"):
        run_chat(options(model), worker)
    output = capsys.readouterr()
    assert "private marker" not in output.out + output.err


def test_worker_dispatches_deltas_before_reading_final_response():
    import io
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    frames = iter(
        [
            {"event": "delta", "text": "é"},
            {"event": "delta", "text": "界"},
            {"text": "é界"},
        ]
    )
    received = []
    reads = []

    def read():
        reads.append(True)
        return next(frames)

    def on_delta(text):
        assert len(reads) < 3
        received.append(text)

    worker.read = read
    assert worker.request({"operation": "generate", "stream": True}, on_delta=on_delta) == {"text": "é界"}
    assert received == ["é", "界"]


def test_worker_delivers_progress_without_ending_the_stream():
    import io
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    progress = {
        "event": "progress",
        "input_tokens": 20,
        "sampled_tokens": 0,
        "output_token_budget": 5,
        "elapsed_seconds": 0.0,
        "prefill_seconds": None,
    }
    replies = iter(
        [
            progress,
            {**progress, "sampled_tokens": 1, "elapsed_seconds": 0.2, "prefill_seconds": 0.2},
            {"event": "delta", "text": "fixture"},
            {"text": "fixture"},
        ]
    )
    worker.read = lambda: next(replies)
    received = []
    result = worker.request(
        {"operation": "generate", "stream": True},
        on_delta=lambda _text: None,
        on_progress=lambda value: received.append(value),
    )
    assert result == {"text": "fixture"}
    assert [value["sampled_tokens"] for value in received] == [0, 1]
    assert all("text" not in value and "token_ids" not in value for value in received)


def test_worker_ignores_valid_progress_for_existing_stream_consumers():
    import io
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    replies = iter(
        [
            {
                "event": "progress",
                "input_tokens": 20,
                "sampled_tokens": 0,
                "output_token_budget": 5,
                "elapsed_seconds": 0.0,
                "prefill_seconds": None,
            },
            {"event": "delta", "text": "fixture"},
            {"text": "fixture"},
        ]
    )
    worker.read = lambda: next(replies)
    assert worker.request({"operation": "generate", "stream": True}, on_delta=lambda _text: None) == {
        "text": "fixture"
    }


def test_worker_rejects_regressing_progress():
    import io
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    progress = {
        "event": "progress",
        "input_tokens": 20,
        "sampled_tokens": 2,
        "output_token_budget": 5,
        "elapsed_seconds": 0.3,
        "prefill_seconds": 0.2,
    }
    replies = iter([progress, {**progress, "sampled_tokens": 1}])
    worker.read = lambda: next(replies)
    with pytest.raises(ChatError, match="Invalid worker stream"):
        worker.request({"operation": "generate", "stream": True}, on_delta=lambda _text: None)


@pytest.mark.parametrize(
    "frames",
    [
        [{"event": "delta", "text": 5}],
        [{"event": "unexpected", "text": "hello"}],
        [{"event": "delta", "text": "hello"}, {"text": "different"}],
    ],
)
def test_worker_rejects_invalid_or_mismatched_streams(frames):
    import io
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    replies = iter(frames)
    worker.read = lambda: next(replies)
    with pytest.raises(ChatError, match="Invalid worker stream"):
        worker.request({"operation": "generate", "stream": True}, on_delta=lambda _: None)


def test_worker_cooperative_cancel_before_first_token_keeps_process(tmp_path):
    import threading

    from stark_convert.chat import ChatCancelled

    executable = tmp_path / "cancel worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "fd=int(os.environ['STARK_CANCEL_FD'])\n"
        "print(json.dumps({'ready':True,'cancellation':'pipe-v1'}),flush=True)\n"
        "for line in sys.stdin:\n"
        "    p=json.loads(line)\n"
        "    if p.get('request_id')==1:\n"
        "        data=os.read(fd,8)\n"
        "        assert int.from_bytes(data,'little')==1\n"
        "        print(json.dumps({'error':'generation_cancelled'}),flush=True)\n"
        "    else: print(json.dumps({'text':'next response'}),flush=True)\n"
    )
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 2)
    control, waiting = threading.Event(), threading.Event()
    errors = []
    try:
        worker.read()
        original = worker.read

        def read(*args):
            waiting.set()
            return original(*args)

        worker.read = read

        def request():
            try:
                worker.request({"operation": "generate"}, cancelled=control)
            except ChatCancelled:
                errors.append("cancelled")

        thread = threading.Thread(target=request)
        thread.start()
        assert waiting.wait(2)
        control.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == ["cancelled"]
        assert worker.process.poll() is None
        assert worker.request({"operation": "generate"}) == {"text": "next response"}
    finally:
        worker.close()


def mtp_descriptor(**changes):
    value = {
        "supported": True,
        "max_drafts": 6,
        "recommended_drafts": 4,
        "default_drafts": 0,
        "sampling": "exact-rejection-v1",
        "verification": "model-specific-exact-v1",
        "operations": ["generate"],
    }
    value.update(changes)
    return value


class MtpWorker(FakeWorker):
    def read(self):
        return {**super().read(), "mtp": mtp_descriptor()}


@pytest.mark.parametrize("enabled, expected", [(False, 0), (True, 4)])
@pytest.mark.parametrize("context_limit", [8192, 0xFFFFFFFF])
def test_cli_toggle_resolves_loaded_model_depth_and_forwards_legacy_wire_field(
    model, enabled, expected, context_limit
):
    worker = MtpWorker(context_limit=context_limit)
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, mtp=enabled), worker) == 0
    generated = [r for r in worker.requests if r["operation"] == "generate"]
    assert len(generated) == 1
    assert generated[0]["mtp_drafts"] == expected
    assert "mtp" not in generated[0]
    assert generated[0]["context_tokens"] == (1024 if context_limit != 0xFFFFFFFF else 20 + 32)


def test_cli_default_off_keeps_older_unsupported_worker_requests_compatible(model):
    worker = FakeWorker()
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model), worker) == 0
    assert all("mtp_drafts" not in request and "mtp" not in request for request in worker.requests)


def test_cli_rejects_unsupported_toggle_before_reading_input_or_tokenizing(model):
    worker = FakeWorker()
    with patch("builtins.input", side_effect=AssertionError("input must not be read")):
        with pytest.raises(ChatError, match="MTP is unavailable"):
            run_chat(options(model, mtp=True), worker)
    assert worker.requests == []


def test_cli_rejects_image_mtp_before_reading_image_file_or_input(model):
    worker = MtpWorker()
    with (
        patch("stark_convert.chat._read_image_file", side_effect=AssertionError("image must not be read")),
        patch("builtins.input", side_effect=AssertionError("input must not be read")),
    ):
        with pytest.raises(ChatError, match="MTP is unavailable for image requests"):
            run_chat(options(model, mtp=True, image=model / "private-name.png"), worker)
    assert worker.requests == []


@pytest.mark.parametrize("flags, enabled", [([], False), (["--mtp"], True), (["--no-mtp"], False)])
def test_main_parses_explicit_mtp_toggle_without_enabling_it_by_default(model, flags, enabled):
    worker = FakeWorker()
    worker.close = lambda: None
    with (
        patch("stark_convert.chat.Worker", return_value=worker),
        patch("stark_convert.chat.run_chat", return_value=0) as run,
    ):
        assert main(["--model-path", str(model), *flags]) == 0
    assert run.call_args.args[0].mtp is enabled


def test_main_rejects_conflicting_mtp_flags_before_launch(model):
    with patch("stark_convert.chat.Worker") as launch:
        with pytest.raises(SystemExit) as error:
            main(["--model-path", str(model), "--mtp", "--no-mtp"])
    assert error.value.code == 2
    launch.assert_not_called()


def test_worker_mtp_capabilities_clear_on_timeout_and_close(tmp_path):
    executable = tmp_path / "mtp worker"
    ready = {"ready": True, "mtp": mtp_descriptor()}
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        f"print(json.dumps({ready!r}), flush=True)\n"
        "for line in sys.stdin: time.sleep(10)\n"
    )
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 2)
    try:
        assert worker.mtp["supported"] is False
        worker.read()
        assert worker.mtp == mtp_descriptor()
        worker.timeout = 0.1
        with pytest.raises(ChatError, match="timed out"):
            worker.request({"operation": "generate"})
        assert worker.mtp["supported"] is False
        worker.mtp = mtp_descriptor()
    finally:
        worker.close()
    assert worker.mtp["supported"] is False


@pytest.mark.parametrize("failure", [{"ready": False, "error": "model_load_failed"}, ["invalid"]])
def test_worker_failed_readiness_clears_any_prior_mtp_metadata(tmp_path, failure):
    executable = tmp_path / "failed mtp worker"
    executable.write_text(f"#!{sys.executable}\nimport json\nprint(json.dumps({failure!r}),flush=True)\n")
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 2)
    worker.mtp = mtp_descriptor()
    try:
        if isinstance(failure, list):
            with pytest.raises(ChatError):
                worker.read()
        else:
            worker.read()
        assert worker.mtp["supported"] is False
    finally:
        worker.close()


def test_worker_transport_keeps_explicit_mtp_fields_and_does_not_inject_a_preference():
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker.read = lambda: {"accepted": True}
    worker.mtp = mtp_descriptor()
    for payload in [{"operation": "generate"}, {"operation": "generate", "mtp": False}]:
        assert worker.request(payload) == {"accepted": True}
    sent = [json.loads(line) for line in worker.process.stdin.getvalue().splitlines()]
    assert sent == [{"operation": "generate"}, {"operation": "generate", "mtp": False}]
