"""Model-free lifecycle and aggregate privacy checks for MaxxBench orchestration."""

import asyncio
import contextlib
import io
import json
import logging
import stat
import sys
import traceback
from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace

import pytest
import run_maxxbench as runner


def config_module(monkeypatch):
    module = ModuleType("pier.models.trial.config")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    for name in ("AgentConfig", "EnvironmentConfig", "TaskConfig", "TrialConfig"):
        setattr(module, name, Config)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_atomic_json_write_is_private_and_bad_serialization_preserves_old_receipt(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    runner.write_json(destination, {"passed": True})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert json.loads(destination.read_text()) == {"passed": True}
    with pytest.raises(ValueError):
        runner.write_json(destination, {"invalid": float("nan")})
    assert json.loads(destination.read_text()) == {"passed": True}


@pytest.mark.parametrize("field", ["worker_sha256", "profile", "artifact_sha256"])
def test_campaign_binding_rejects_changed_identity_and_preserves_old_receipt(tmp_path, field):
    destination = tmp_path / "campaign.json"
    original = {"worker_sha256": "worker-a", "profile": {"context": 32768}, "artifact_sha256": "artifact-a"}
    campaign_id = runner.bind_campaign(destination, original)
    assert runner.bind_campaign(destination, original) == campaign_id
    changed = {**original, field: "different"}
    with pytest.raises(ValueError, match="campaign_identity_mismatch"):
        runner.bind_campaign(destination, changed)
    assert json.loads(destination.read_text()) == original


def test_release_preflight_validates_pinned_revision_and_all_tasks_without_output(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    for index in range(89):
        path = tasks / f"task-{index:02}"
        path.mkdir(parents=True)
        (path / "task.toml").write_text("")
    commands = []
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: runner.UPSTREAM_COMMIT + "\n")

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert len(runner.preflight(tmp_path)) == 89
    assert commands[0][0] == [sys.executable, str(tmp_path / "tools/validate_release.py")]
    assert all(options["stdout"] == runner.subprocess.DEVNULL for _, options in commands)
    assert all(options["stderr"] == runner.subprocess.DEVNULL for _, options in commands)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "different-revision")
    with pytest.raises(ValueError, match="upstream_revision_mismatch"):
        runner.preflight(tmp_path)


def test_gpu_reservation_unlocks_after_failure(tmp_path, monkeypatch):
    operations = []
    monkeypatch.setattr(runner.fcntl, "flock", lambda _handle, operation: operations.append(operation))
    with pytest.raises(RuntimeError, match="stopped"), runner.gpu_reservation(tmp_path / "gpu.lock"):
        raise RuntimeError("stopped")
    assert operations == [runner.fcntl.LOCK_EX, runner.fcntl.LOCK_UN]


class Pipe:
    def __init__(self):
        self.value = ""
        self.closed = False

    def write(self, value):
        self.value += value

    def close(self):
        self.closed = True


class BridgeProcess:
    def __init__(self, *, ready=True, timeout=False):
        self.stdin = Pipe()
        self.ready_frame = {
            "event": "ready", "base_url": "http://127.0.0.1:12345/v1", "model": "sample",
            "context_tokens": 32768, "max_output_tokens": 4096, "template_enable_thinking": True,
            "mtp_requested": "default", "mtp_default_enabled": False, "prefix_cache_supported": True,
            "prefix_chunk_tokens": 64, "worker_pid": 1234, "worker_sha256": "b" * 64,
            "template_helper_sha256": "c" * 64, "logging": "aggregate_only",
        }
        self.stopped_frame = {
            "event": "stopped", "failure": None, "cleanup_failed": False, "elapsed_seconds": 1.0,
            "requests": 1, "completed": 1, "prompt_tokens": 10, "completion_tokens": 20,
            "request_elapsed_seconds": 0.5, "http_failures": {},
        }
        self.stdout = io.StringIO()
        self.frames = [self.ready_frame, self.stopped_frame] if ready else []
        self.refresh_output()
        self.returncode = None if ready else 1
        self.pid = 4321
        self.terminated = False
        self.killed = False
        self.timeout = timeout
        self.native_alive = True
        self.native_stays_alive = False
        self.native_parent = self.pid
        self.native_signals = []
        self.exit_code = 0

    def refresh_output(self):
        self.stdout = io.StringIO("".join(json.dumps(frame) + "\n" for frame in self.frames))

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.timeout and not self.killed:
            raise runner.subprocess.TimeoutExpired("bridge", timeout)
        self.returncode = -9 if self.killed else self.exit_code
        return self.returncode

    def kill(self):
        self.killed = True


def bridge_patches(monkeypatch, tmp_path, process):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def run(command, **_kwargs):
        if command[:2] == ["/bin/ps", "-A"]:
            return SimpleNamespace(
                returncode=0,
                stdout=f"1234 {process.native_parent} Sun Nov 8 01:00:00 2026 {tmp_path / 'stark-generate'}\n",
            )
        assert command[:2] == ["/bin/ps", "-p"]
        assert command[2] == "1234"
        alive = process.native_alive and (not process.terminated or process.native_stays_alive)
        return SimpleNamespace(
            returncode=0 if alive else 1,
            stdout=f"{process.native_parent} Sun Nov 8 01:00:00 2026 {tmp_path / 'stark-generate'}\n" if alive else "",
        )

    def kill(pid, signum):
        assert pid == 1234
        process.native_signals.append(signum)
        process.native_alive = False

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.os, "kill", kill)


def profile(tmp_path):
    return {"id": "sample", "path": str(tmp_path / "model"), "context": 32768,
            "max_tokens": 4096, "thinking": True}


def test_model_bridge_keeps_key_out_of_command_environment_and_cleans_up_on_failure(tmp_path, monkeypatch):
    process = BridgeProcess()
    bridge_patches(monkeypatch, tmp_path, process)
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.secrets, "token_hex", lambda _size: "a" * 64)
    with (
        pytest.raises(RuntimeError, match="stop trial"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate") as (ready, key, _output),
    ):
        assert ready["model"] == "sample" and key == "a" * 64
        raise RuntimeError("stop trial")
    assert process.terminated and process.stdin.closed
    assert process.stdin.value == "a" * 64 + "\n"
    command, options = calls[0]
    assert "a" * 64 not in json.dumps(command)
    assert "a" * 64 not in json.dumps(options["env"])
    assert options["stderr"] == runner.subprocess.DEVNULL


def test_model_bridge_kills_and_reaps_unresponsive_bridge(tmp_path, monkeypatch):
    process = BridgeProcess(timeout=True)
    process.native_stays_alive = True
    bridge_patches(monkeypatch, tmp_path, process)
    with (
        pytest.raises(ValueError, match="model_bridge_shutdown_failed"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pass
    assert process.terminated and process.killed and process.returncode == -9
    assert process.native_signals == [runner.signal.SIGTERM]


def test_model_bridge_clears_adapter_credentials_before_process_shutdown(tmp_path, monkeypatch):
    process = BridgeProcess()
    bridge_patches(monkeypatch, tmp_path, process)
    events = []
    adapter = ModuleType("maxxbench_agent")
    adapter.clear_connection = lambda: events.append("cleared")
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    terminate = process.terminate

    def tracked_terminate():
        events.append("terminated")
        terminate()

    process.terminate = tracked_terminate
    with runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"):
        pass
    assert events == ["cleared", "terminated"]


def test_model_bridge_reports_startup_failure_without_private_output(tmp_path, monkeypatch):
    process = BridgeProcess(ready=False)
    process.stdout = io.StringIO("private text from child\n")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    with (
        pytest.raises(ValueError, match="model_bridge_protocol_failed"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pytest.fail("failed bridge became available")


def test_model_bridge_discovers_owned_native_child_when_startup_fails_before_ready(tmp_path, monkeypatch):
    process = BridgeProcess(timeout=True)
    process.native_stays_alive = True
    process.frames = [{"private": "private-marker startup failure"}]
    process.refresh_output()
    bridge_patches(monkeypatch, tmp_path, process)
    with (
        pytest.raises(ValueError, match="model_bridge_protocol_failed"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pytest.fail("invalid startup became ready")
    assert process.killed and process.native_signals == [runner.signal.SIGTERM]


def test_model_bridge_refuses_worker_pid_without_matching_parent(tmp_path, monkeypatch):
    process = BridgeProcess()
    process.native_parent = 9999
    bridge_patches(monkeypatch, tmp_path, process)
    with (
        pytest.raises(ValueError, match="model_bridge_ownership_unverified"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pytest.fail("unrelated worker PID was accepted")
    assert not process.native_signals


@pytest.mark.parametrize("change", [{"private": "private-marker"}, {"failure": "bridge_failed"},
                                   {"cleanup_failed": True}])
def test_model_bridge_refuses_unexpected_or_failed_terminal_frame(tmp_path, monkeypatch, change):
    process = BridgeProcess()
    process.stopped_frame.update(change)
    process.refresh_output()
    bridge_patches(monkeypatch, tmp_path, process)
    failure = "protocol" if "private" in change else "shutdown"
    with (
        pytest.raises(ValueError, match=f"model_bridge_{failure}_failed"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pass


def test_model_bridge_rejects_success_frame_with_nonzero_exit_status(tmp_path, monkeypatch):
    process = BridgeProcess()
    process.exit_code = 1
    bridge_patches(monkeypatch, tmp_path, process)
    with (
        pytest.raises(ValueError, match="model_bridge_shutdown_failed"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pass


def test_model_bridge_surfaces_native_cleanup_failure(tmp_path, monkeypatch):
    process = BridgeProcess()
    process.native_stays_alive = True
    bridge_patches(monkeypatch, tmp_path, process)
    monkeypatch.setattr(runner.os, "kill", lambda *_: (_ for _ in ()).throw(PermissionError("private-marker")))
    with (
        pytest.raises(ValueError, match="model_bridge_cleanup_failed"),
        runner.model_bridge(profile(tmp_path), tmp_path / "stark-generate"),
    ):
        pass


def test_trial_run_aggregates_do_not_serialize_rollout_or_arbitrary_metadata(tmp_path, monkeypatch):
    config_module(monkeypatch)
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1}), exception_info=None,
        agent_result=SimpleNamespace(
            n_agent_steps=3, n_input_tokens=10, n_output_tokens=20, n_cache_tokens=0, cost_usd=None,
            metadata={"model_calls": 3, "agent_status": "submitted", "private": "private-marker response"},
            model_dump=lambda: {"n_agent_steps": 3, "rollout_details": [{"tokens": ["private-marker"]}],
                                "metadata": {"private": "private-marker response"}},
        ),
        started_at=datetime.now(UTC), finished_at=datetime.now(UTC),
    )

    class Trial:
        @classmethod
        async def create(cls, _config):
            return cls()

        async def run(self):
            print("private-marker transient command")
            return result

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        receipt = asyncio.run(runner.trial_run(Trial, tmp_path, "sample-task", tmp_path / "trials"))
    assert output.getvalue() == ""
    assert receipt["passed"] is True
    assert "private-marker" not in json.dumps(receipt)
    assert "rollout_details" not in json.dumps(receipt)


def test_patch_pier_suppresses_exception_strings_and_traceback(tmp_path, monkeypatch):
    exception_module = ModuleType("pier.models.trial.result")
    trial_module = ModuleType("pier.trial.trial")

    class ExceptionInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    exception_module.ExceptionInfo = ExceptionInfo
    trial_module.Trial = type("Trial", (), {})
    monkeypatch.setitem(sys.modules, exception_module.__name__, exception_module)
    monkeypatch.setitem(sys.modules, trial_module.__name__, trial_module)
    (tmp_path / "run.py").write_text("def patch_verifier_image_handoff(trial):\n    trial.patched = True\n")
    original_disable = logging.root.manager.disable
    original_traceback = traceback.format_exc
    try:
        patched = runner.patch_pier(tmp_path)
        exception = ExceptionInfo.from_exception(RuntimeError("private-marker"))
        assert patched.patched is True
        assert "private-marker" not in json.dumps(exception.__dict__, default=str)
        assert traceback.format_exc() == "Exception details suppressed by Stark policy."
    finally:
        logging.disable(original_disable)
        traceback.format_exc = original_traceback
