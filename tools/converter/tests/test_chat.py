"""Chat contracts with synthetic messages and a private fake worker."""

import io
import json
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest
from jinja2.exceptions import SecurityError

from stark_convert.chat import ChatError, Worker, load_template, main, run_chat, terminal_text


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"token_codec_config": {
        "type": "ChatCodecConfig", "bos_token": "<bos>", "default_system_prompt": "system fixture",
        "prompt_template": "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:",
    }}))
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
    (model / "config.json").write_text(json.dumps({"token_codec_config": {
        "type": "ChatCodecConfig", "prompt_template": "{{ ''.__class__.__mro__ }}",
    }}))
    _, template = load_template(model)
    with pytest.raises(SecurityError):
        template.render(messages=[])


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
    worker = Worker(executable, tmp_path, 0.2)
    try:
        assert worker.read() == {"ready": True}
        assert worker.request({"text": "synthetic fixture"}) == {"text": "synthetic fixture"}
        with pytest.raises(ChatError, match="timed out"):
            worker.request({"wait": True})
    finally:
        worker.close()
    assert worker.process.poll() is not None


def test_main_suppresses_exception_payload_and_cleans_up(model, capsys):
    worker = FakeWorker()
    worker.close = lambda: None
    with patch("stark_convert.chat.Worker", return_value=worker), patch(
        "stark_convert.chat.run_chat", side_effect=RuntimeError("private marker")
    ), patch.object(worker, "close") as close:
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
        with pytest.raises(ChatError, match="request limit"):
            worker.request({"text": "x" * 1024 * 1024})
    finally:
        worker.close()


def test_startup_failure_does_not_echo_worker_diagnostics(model, capsys):
    worker = FakeWorker()
    worker.read = lambda: {"ready": False, "error": "private marker"}
    with pytest.raises(ChatError, match="could not load"):
        run_chat(options(model), worker)
    output = capsys.readouterr()
    assert "private marker" not in output.out + output.err
