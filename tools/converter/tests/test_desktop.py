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

from stark_convert.desktop import Desktop, DesktopError, metric


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
                    "prompt_template": "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:",
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
        "        for chunk in ['synthetic ', 'answer']: print(json.dumps({'event':'delta','text':chunk}),flush=True)\n"
        "    reply={'token_count':20,'token_ids':[2]*20} if request['operation']=='tokenize' else {\n"
        "        'text':'synthetic answer','input_tokens':20,'output_tokens':4,'finish_reason':'eos',\n"
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
    desktop.dispatch({"command": "generate", "text": "synthetic desktop request"})
    finish(desktop)
    assert len(desktop.messages) == 3
    assert desktop.worker.process is process


@pytest.mark.parametrize("source, expected", [
    ("huggingface:Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    ("local:/synthetic/JevK5", "JevK5"),
])
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


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 3},
        {"temperature": False},
        {"n": 2},
        {"tools": []},
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
    desktop.start_api()
    details = next(e for e in events(desktop) if e["event"] == "connection")
    try:
        request = urllib.request.Request(
            details["base_url"] + "/chat/completions",
            data=json.dumps(
                {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True}
            ).encode(),
            headers={"Authorization": "Bearer " + details["api_key"], "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        assert text == "synthetic answer"
        assert b"data: [DONE]" in body
        assert details["api_key"] not in desktop.registry.read_text()
    finally:
        desktop.api.close()


def test_api_worker_failure_clears_job_and_can_restart(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    with pytest.raises(APIError) as failed:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "failure fixture"}]})
    assert failed.value.status == 503
    assert desktop.job is None
    assert desktop.worker is None
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
    assert failed.value.status == 503
    assert desktop.worker is None
    assert desktop.job is None
    assert process.poll() is not None


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


def test_raw_conversion_is_offline_and_cancel_reaps_converter(desktop, tmp_path):
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
    assert "--offline" in commands[0]
    assert "hf-generic" in commands[0]
    assert desktop.models == []
    assert not list(desktop.model_dir.iterdir())


@pytest.mark.parametrize("repo, generic", [("example/model", True), ("Qwen/Qwen3-0.6B", False)])
def test_huggingface_conversion_publishes_complete_model(desktop, model, repo, generic):
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
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": repo})
        finish(desktop)
    assert "--offline" not in commands[0]
    assert ("--metadata-profile" in commands[0]) is generic
    assert ("hf-generic" in commands[0]) is generic
    assert len(desktop.models) == 1
    assert desktop.models[0]["name"] == repo.split("/")[-1]
    assert Path(desktop.models[0]["path"]).parent == desktop.model_dir
    assert not list(desktop.model_dir.glob(".install-*"))


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
