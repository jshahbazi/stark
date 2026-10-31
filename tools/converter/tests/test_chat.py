"""Chat contracts with synthetic messages and a private fake worker."""

import io
import json
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest
from jinja2.exceptions import SecurityError

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
    def __init__(self):
        self.requests = []

    def read(self):
        return {"ready": True, "mode": "text-generation-v1", "max_context_tokens": 8192, "max_new_tokens": 4096}

    def request(self, payload):
        self.requests.append(payload)
        return {"token_count": 20} if payload["operation"] == "tokenize" else {"text": "reply fixture"}


def options(model, **kwargs):
    return Namespace(model_path=model, context_tokens=1024, max_new_tokens=32, system=None, thinking=False, **kwargs)


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
    _, template = load_template(model)
    with pytest.raises(SecurityError):
        template.render(messages=[])


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


def test_request_limit_is_checked_before_write(tmp_path):
    worker = Worker(sys.executable, tmp_path, 2)
    try:
        with pytest.raises(ChatRequestTooLarge, match="request limit"):
            worker.request({"text": "x" * 1024 * 1024})
    finally:
        worker.close()


@pytest.mark.parametrize("unit", ["x", "界", "\n"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_worker_request_limit_counts_encoded_bytes_and_newline(unit, over_limit):
    from types import SimpleNamespace

    worker = Worker.__new__(Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker.read = lambda: {"accepted": True}
    limit = 1024 * 1024
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
def test_cli_toggle_resolves_loaded_model_depth_and_forwards_legacy_wire_field(model, enabled, expected):
    worker = MtpWorker()
    with patch("sys.stdin", io.StringIO("synthetic turn\n/exit\n")):
        assert run_chat(options(model, mtp=enabled), worker) == 0
    generated = [r for r in worker.requests if r["operation"] == "generate"]
    assert len(generated) == 1
    assert generated[0]["mtp_drafts"] == expected
    assert "mtp" not in generated[0]


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
