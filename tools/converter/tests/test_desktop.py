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
        "print(json.dumps({'ready':True,'mode':'text-generation-v1','max_context_tokens':8192,'max_new_tokens':4096}),flush=True)\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if 'wait fixture' in request.get('rendered_prompt',''): time.sleep(30)\n"
        "    if 'failure fixture' in request.get('rendered_prompt',''): raise SystemExit(2)\n"
        "    reply={'token_count':20} if request['operation']=='tokenize' else {\n"
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
