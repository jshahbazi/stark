# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["datacurve-pier==0.3.1", "mini-swe-agent==2.4.6"]
# ///
"""Run pinned MaxxBench tasks through one local Stark model at a time.

Use ``uv run --locked scripts/run_maxxbench.py --help``. This is a local
configuration with the stock task verifiers, not a reproduction of the public
leaderboard's provider models or reasoning settings. No conversations are logged.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from maxxbench_verifier_compat import (
    TRIAL_IMAGE_ATTRIBUTE,
    capture_environment_image,
    patch_verifier_image_compat,
)

UPSTREAM_COMMIT = "1a0df5e9c5d4bccbae131b01830d478088fea947"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bind_campaign(path, identity):
    """Refuse to reuse a directory containing measurements from another run."""
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("campaign_identity_mismatch")
    else:
        write_json(path, identity)
    return identity_digest(identity)


def bind_environment_image(images, task, image_id):
    import re

    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("environment_identity_unavailable")
    if task in images and images[task] != image_id:
        raise ValueError("environment_identity_changed")
    images[task] = image_id


def artifact_files(directory):
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts))


def file_identity(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def fingerprint_artifact(directory, cache):
    """Hash contents once; reuse only while the complete filesystem identity matches."""
    directory = Path(directory).expanduser().resolve(strict=True)
    previous = json.loads(cache.read_text()) if cache.exists() else {}
    records = {}
    for path in artifact_files(directory):
        name = str(path.relative_to(directory))
        before = file_identity(path)
        old = previous.get("files", {}).get(name, {})
        content_hash = old.get("sha256") if old.get("identity") == before else digest(path)
        if before != file_identity(path):
            raise ValueError("artifact_changed")
        records[name] = {"identity": before, "sha256": content_hash}
    if not records:
        raise ValueError("artifact_empty")
    write_json(cache, {"directory": str(directory), "files": records})
    return {"sha256": identity_digest({name: row["sha256"] for name, row in records.items()}),
            "files": records, "directory": directory}


def assert_artifact_unchanged(artifact):
    directory = artifact["directory"]
    current = {str(path.relative_to(directory)): file_identity(path) for path in artifact_files(directory)}
    if current != {name: row["identity"] for name, row in artifact["files"].items()}:
        raise ValueError("artifact_changed")


def aggregate_agent_metrics(result):
    if result is None:
        return None
    values = result.model_dump()
    counts = {key: value for key, value in values.items() if key in {
        "n_input_tokens", "n_output_tokens", "n_cache_tokens", "n_agent_steps", "cost_usd",
    } and (value is None or type(value) in (int, float))}
    metadata = values.get("metadata") or {}
    safe = {key: value for key, value in metadata.items() if key in {
        "model_calls", "duration_seconds", "context_overflow_errors", "submission_present",
    } and type(value) in (int, float, bool)}
    if metadata.get("agent_status") in {
        "submitted", "Submitted", "LimitsExceeded", "FormatError", "run_error", "unknown",
        "step_limit", "wall_time_limit", "format_error", "context_overflow", "model_error",
        "adapter_error",
    }:
        safe["agent_status"] = metadata["agent_status"]
    errors = metadata.get("http_errors")
    if isinstance(errors, dict):
        safe["http_errors"] = {key: value for key, value in errors.items() if key in {
            "transport", "configuration", "request_rejected", "authentication", "endpoint",
            "rate_limited", "server", "http_other", "response_too_large", "invalid_response",
            "environment", "environment_timeout", "cancelled", "agent",
        } and type(value) is int}
    codes = metadata.get("api_error_codes")
    adapter = sys.modules.get("maxxbench_agent")
    allowed_codes = set(getattr(adapter, "SAFE_API_ERROR_CODES", ())) | {"unknown"}
    if isinstance(codes, dict):
        safe["api_error_codes"] = {key: value for key, value in codes.items()
                                   if key in allowed_codes and type(value) is int and value >= 0}
    counts["metadata"] = safe
    return counts


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def preflight(upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("upstream_revision_mismatch")
    validation = subprocess.run(
        [sys.executable, str(upstream / "tools/validate_release.py")],
        cwd=upstream, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if validation.returncode:
        raise ValueError("upstream_release_validation_failed")
    tasks = sorted(path.name for path in (upstream / "tasks").iterdir() if (path / "task.toml").is_file())
    if len(tasks) != 89:
        raise ValueError("upstream_task_count_mismatch")
    if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode:
        raise ValueError("docker_unavailable")
    if subprocess.run(["docker", "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode:
        raise ValueError("compose_unavailable")
    return tasks


def patch_pier(upstream):
    """Install stock grading/handoff, scoped compatibility and private errors."""
    from pier.models.trial.result import ExceptionInfo
    from pier.trial.trial import Trial

    spec = importlib.util.spec_from_file_location("maxxbench_upstream_run", upstream / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_verifier_image_handoff(Trial)
    patch_verifier_image_compat(Trial)
    logging.disable(logging.CRITICAL)
    # Pier writes this directly as well as through ExceptionInfo. Neither path
    # may serialize model-derived command/output text from an exception.
    traceback.format_exc = lambda *args, **kwargs: "Exception details suppressed by Stark policy."

    def private_exception(cls, error):
        return cls(exception_type=type(error).__name__, exception_message="suppressed",
                   exception_traceback="suppressed", occurred_at=datetime.now(UTC))

    ExceptionInfo.from_exception = classmethod(private_exception)
    return Trial


@contextlib.contextmanager
def gpu_reservation(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        emit({"event": "waiting_for_gpu"})
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            emit({"event": "gpu_reserved"})
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            emit({"event": "gpu_released"})


@contextlib.contextmanager
def model_bridge(profile, worker):
    """Own the bridge and its verified native child; retain aggregate frames only."""
    import math

    key = secrets.token_hex(32)
    command = [str(ROOT / "tools/converter/.venv/bin/python"),
               str(ROOT / "scripts/maxxbench_stark_bridge.py"), "--worker", str(worker),
               "--model", str(Path(profile["path"]).expanduser()), "--model-id", profile["id"],
               "--context", str(profile["context"]), "--max-tokens", str(profile["max_tokens"]),
               "--mtp", profile.get("mtp", "default"), "--request-timeout", "1800"]
    if profile.get("thinking", False):
        command.append("--thinking")
    environment = dict(os.environ, PATH=str(worker.parent) + os.pathsep + os.environ.get("PATH", ""),
                       HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True, env=environment)
    output = []
    protocol_failed = threading.Event()
    worker_identities = []
    body_failed = False
    ready_fields = {
        "event", "base_url", "model", "context_tokens", "max_output_tokens", "template_enable_thinking",
        "mtp_requested", "mtp_default_enabled", "prefix_cache_supported", "prefix_chunk_tokens", "worker_pid",
        "worker_sha256", "template_helper_sha256", "logging",
    }
    stopped_fields = {
        "event", "failure", "cleanup_failed", "elapsed_seconds", "requests", "completed", "prompt_tokens",
        "completion_tokens", "request_elapsed_seconds", "http_failures",
    }

    def count(value):
        return type(value) is int and value >= 0

    def seconds(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def valid_frame(value):
        if not isinstance(value, dict):
            return False
        if value.get("event") == "ready":
            import re

            url = value.get("base_url")
            port = re.fullmatch(r"http://127\.0\.0\.1:([0-9]{1,5})/v1", url) if isinstance(url, str) else None
            return (
                set(value) == ready_fields and port is not None and 0 < int(port[1]) <= 65535
                and value["model"] == profile["id"] and value["context_tokens"] == profile["context"]
                and value["max_output_tokens"] == profile["max_tokens"]
                and value["template_enable_thinking"] is profile.get("thinking", False)
                and value["mtp_requested"] == profile.get("mtp", "default")
                and type(value["mtp_default_enabled"]) is bool
                and type(value["prefix_cache_supported"]) is bool
                and (value["prefix_chunk_tokens"] is None or count(value["prefix_chunk_tokens"]))
                and type(value["worker_pid"]) is int and value["worker_pid"] > 1
                and all(isinstance(value[name], str) and re.fullmatch(r"[0-9a-f]{64}", value[name])
                        for name in ("worker_sha256", "template_helper_sha256"))
                and value["logging"] == "aggregate_only"
            )
        if value.get("event") != "stopped" or set(value) - stopped_fields:
            return False
        if value.get("failure") not in {
            None, "credential_input_failed", "invalid_arguments", "invalid_model_id", "model_invalid",
            "model_admission_failed", "executable_changed_during_load", "requested_context_unavailable",
            "requested_output_limit_unavailable", "startup_cancelled", "api_model_unavailable", "bridge_failed",
        }:
            return False
        if "cleanup_failed" in value and type(value["cleanup_failed"]) is not bool:
            return False
        if any(not count(value[name]) for name in ("requests", "completed", "prompt_tokens", "completion_tokens")
               if name in value):
            return False
        if any(not seconds(value[name]) for name in ("elapsed_seconds", "request_elapsed_seconds") if name in value):
            return False
        errors = value.get("http_failures", {})
        return isinstance(errors, dict) and all(
            (name == "internal" or name.isdigit() and 400 <= int(name) <= 599) and count(total)
            for name, total in errors.items()
        )

    def snapshot(pid):
        # lstart plus executable identity prevents killing a subsequently reused
        # PID. Initial PPID proves the worker belongs to this exact bridge.
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "ppid=,lstart=,comm="],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 1 and not result.stdout.strip():
            return None
        if result.returncode:
            raise ValueError("model_bridge_cleanup_failed")
        parts = result.stdout.strip().split(maxsplit=6)
        if len(parts) != 7 or not parts[0].isdigit():
            raise ValueError("model_bridge_cleanup_failed")
        return int(parts[0]), tuple(parts[1:6]), str(Path(parts[6]).resolve())

    def discover_workers():
        result = subprocess.run(
            ["/bin/ps", "-A", "-o", "pid=,ppid=,lstart=,comm="],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode:
            raise ValueError("model_bridge_cleanup_failed")
        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=7)
            if (len(parts) == 8 and parts[0].isdigit() and parts[1].isdigit()
                    and int(parts[1]) == process.pid and str(Path(parts[7]).resolve()) == str(worker.resolve())):
                candidate = int(parts[0]), (int(parts[1]), tuple(parts[2:7]), str(worker.resolve()))
                if candidate not in worker_identities:
                    worker_identities.append(candidate)

    def cleanup_worker(pid, identity):

        def still_owned():
            current = snapshot(pid)
            return current is not None and current[1:] == identity[1:]

        for signum in (signal.SIGTERM, signal.SIGKILL):
            if not still_owned():
                return
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 5
            while still_owned() and time.monotonic() < deadline:
                time.sleep(0.05)
        if still_owned():
            raise ValueError("model_bridge_cleanup_failed")

    def drain():
        try:
            for line in iter(lambda: process.stdout.readline(16385), ""):
                try:
                    value = json.loads(line)
                    expected = "ready" if not output else "stopped"
                    if len(line) > 16384 or not valid_frame(value) or len(output) >= 2 or value["event"] != expected:
                        protocol_failed.set()
                        continue
                    output.append(value)
                except (ValueError, TypeError, AttributeError):
                    protocol_failed.set()
        except (OSError, ValueError):
            protocol_failed.set()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        process.stdin.write(key + "\n")
        process.stdin.close()
        deadline = time.monotonic() + 900
        while not output and not protocol_failed.is_set() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if protocol_failed.is_set():
            raise ValueError("model_bridge_protocol_failed")
        if not output or output[0].get("event") != "ready":
            raise ValueError("model_bridge_start_failed")
        native_pid = output[0]["worker_pid"]
        native_identity = snapshot(native_pid)
        if native_identity is None or native_identity[0] != process.pid or native_identity[2] != str(worker.resolve()):
            raise ValueError("model_bridge_ownership_unverified")
        worker_identities.append((native_pid, native_identity))
        yield output[0], key, output
    except BaseException:
        body_failed = True
        raise
    finally:
        cleanup_failed = False
        hard_killed = False
        adapter = sys.modules.get("maxxbench_agent")
        if adapter is not None:
            try:
                adapter.clear_connection()
            except BaseException:  # noqa: BLE001 - clear credentials without leaking private adapter errors.
                cleanup_failed = True
        try:
            if process.poll() is None:
                # A load can fail or hang before readiness names its native PID.
                # Capture exact matching direct children before their parent exits.
                if not worker_identities:
                    try:
                        discover_workers()
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        cleanup_failed = True
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    hard_killed = True
                    process.kill()
                    process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            cleanup_failed = True
        finally:
            for pid, identity in worker_identities:
                try:
                    cleanup_worker(pid, identity)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    cleanup_failed = True
            reader.join(timeout=5)
            if reader.is_alive():
                cleanup_failed = True
            if not process.stdin.closed:
                process.stdin.close()
            if not reader.is_alive():
                process.stdout.close()
        if cleanup_failed:
            raise ValueError("model_bridge_cleanup_failed")
        if not body_failed:
            if protocol_failed.is_set():
                raise ValueError("model_bridge_protocol_failed")
            if (hard_killed or process.returncode != 0 or len(output) != 2
                    or set(output[-1]) != stopped_fields or output[-1]["failure"] is not None
                    or output[-1]["cleanup_failed"] is not False):
                raise ValueError("model_bridge_shutdown_failed")


def inspect_environment_image(trial):
    """Keep immutable evidence and a private buildable reference for handoff."""
    image = capture_environment_image(trial)
    setattr(trial, TRIAL_IMAGE_ATTRIBUTE, image)
    return image["id"] if image is not None else None


async def trial_run(Trial, upstream, task, directory, profile=None):
    from pier.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )

    agent = AgentConfig(name="oracle") if profile is None else AgentConfig(
        import_path="maxxbench_agent:StarkMiniSweAgent", model_name=profile["id"],
        kwargs={"max_tokens": profile["max_tokens"], "temperature": profile.get("temperature", 0),
                "step_limit": profile.get("step_limit", 250)},
    )
    config = TrialConfig(task=TaskConfig(path=upstream / "tasks" / task), trials_dir=directory,
                         trial_name=task, agent=agent, environment=EnvironmentConfig(force_build=True))
    progress_path = Path(directory) / task / "progress.json"
    started = time.monotonic()
    trial = None
    running = None
    original_setup = None
    unset = object()
    original_setup_override = unset
    original_image = unset
    image_id = None
    image_checked = False
    state = "creating"
    counter_names = ("calls", "prompt_tokens", "completion_tokens", "context_overflow_errors")
    error_names = {
        "transport", "configuration", "request_rejected", "authentication", "endpoint",
        "rate_limited", "server", "http_other", "response_too_large", "invalid_response",
        "environment", "environment_timeout", "cancelled", "agent",
    }

    def counters():
        adapter = sys.modules.get("maxxbench_agent")
        metrics = getattr(adapter, "METRICS", None)
        values = metrics.snapshot() if metrics is not None else {}
        safe = {name: values.get(name, 0) if type(values.get(name)) is int and values[name] >= 0 else 0
                for name in counter_names}
        errors = values.get("http_error_categories", {})
        safe["http_error_categories"] = {
            name: value for name, value in errors.items()
            if name in error_names and type(value) is int and value >= 0
        } if isinstance(errors, dict) else {}
        codes = values.get("api_error_codes", {})
        allowed_codes = set(getattr(adapter, "SAFE_API_ERROR_CODES", ())) | {"unknown"}
        safe["api_error_codes"] = {
            key: value for key, value in codes.items()
            if key in allowed_codes and type(value) is int and value >= 0
        } if isinstance(codes, dict) else {}
        return safe

    baseline = counters()

    def progress(active):
        current = counters()
        delta = {name: max(0, current[name] - baseline[name]) for name in counter_names}
        delta["http_error_categories"] = {
            name: value - baseline["http_error_categories"].get(name, 0)
            for name, value in current["http_error_categories"].items()
            if value > baseline["http_error_categories"].get(name, 0)
        }
        delta["api_error_codes"] = {
            key: value - baseline["api_error_codes"].get(key, 0)
            for key, value in current["api_error_codes"].items()
            if value > baseline["api_error_codes"].get(key, 0)
        }
        phase = state
        if active and state == "running":
            # These are the pinned Pier result's timing fields; no task content
            # or agent messages are inspected to infer a phase.
            partial = getattr(trial, "_result", None)
            for name in ("verifier", "agent_execution", "agent_setup", "environment_setup"):
                timing = getattr(partial, name, None)
                if (getattr(timing, "started_at", None) is not None
                        and getattr(timing, "finished_at", None) is None):
                    phase = name
                    break
        write_json(progress_path, {
            "task": task, "state": state, "phase": phase, "active": active,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "environment_image_id": image_id,
            "environment_image_capture": "captured" if image_id else ("unavailable" if image_checked else "pending"),
            "updated_at": datetime.now(UTC).isoformat(), **delta,
        })

    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):  # noqa: ASYNC230 - local null sink only.
        try:
            progress(True)
            trial = await Trial.create(config)
            original_image = getattr(trial, TRIAL_IMAGE_ATTRIBUTE, unset)
            original_setup = getattr(trial, "_setup_environment", None)
            original_setup_override = trial.__dict__.get("_setup_environment", unset)
            if callable(original_setup):
                async def setup_and_capture():
                    nonlocal image_id, image_checked
                    value = await original_setup()
                    # Capture before agent execution; Pier can remove the image
                    # while finalizing even a short attempt.
                    image_id = inspect_environment_image(trial)
                    image_checked = True
                    progress(True)
                    return value

                trial._setup_environment = setup_and_capture
            state = "running"
            running = asyncio.create_task(trial.run(), name="maxxbench-trial-" + task)
            while True:
                progress(True)
                # The inspection is bounded to five seconds. A 25-second poll
                # also keeps progress updates within thirty seconds during it.
                done, _pending = await asyncio.wait({running}, timeout=25)
                if done:
                    result = await running
                    if not image_checked:
                        image_id = inspect_environment_image(trial)
                        image_checked = True
                    state = "finished"
                    break
        except BaseException as error:
            state = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            raise
        finally:
            if running is not None and not running.done():
                running.cancel()
                # Pier owns environment cleanup in Trial.run's finally. Await it
                # before returning; there is no independent monitor task to leak.
                await asyncio.gather(running, return_exceptions=True)
            if callable(original_setup):
                if original_setup_override is unset:
                    trial.__dict__.pop("_setup_environment", None)
                else:
                    trial._setup_environment = original_setup_override
            if trial is not None:
                if original_image is unset:
                    trial.__dict__.pop(TRIAL_IMAGE_ATTRIBUTE, None)
                else:
                    setattr(trial, TRIAL_IMAGE_ATTRIBUTE, original_image)
            progress(False)
    rewards = result.verifier_result.rewards if result.verifier_result else None
    return {
        "task": task, "rewards": rewards,
        "passed": bool(rewards and rewards.get("reward") == 1 and result.exception_info is None),
        "infrastructure_exception": result.exception_info.exception_type if result.exception_info else None,
        "agent_metrics": aggregate_agent_metrics(result.agent_result),
        "environment_image_id": image_id,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "finished_at": result.finished_at.isoformat() if result.finished_at else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--worker", type=Path, default=ROOT / "target/release/stark-generate")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fingerprint-only", action="store_true")
    parser.add_argument("--gpu-lock", type=Path, default=ROOT.parent / "stark/runtime/stark-benchmark-gpu.lock")
    args = parser.parse_args(argv)
    args.upstream = args.upstream.resolve()
    args.output = args.output.resolve()
    args.worker = args.worker.resolve()
    tasks = preflight(args.upstream)
    if args.task:
        if len(set(args.task)) != len(args.task) or set(args.task) - set(tasks):
            raise ValueError("invalid_task_selection")
        tasks = args.task
    identity = {"upstream_commit": UPSTREAM_COMMIT, "manifest_sha256": digest(args.upstream / "tasks/manifest.json"),
                "runner_sha256": digest(__file__),
                "verifier_compat_sha256": digest(ROOT / "scripts/maxxbench_verifier_compat.py"),
                "selected_tasks": tasks, "release_tasks": 89,
                "created_at": datetime.now(UTC).isoformat(), "logging": "aggregate_only"}
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.preflight_only:
        write_json(args.output / "preflight.json", identity)
        emit({"event": "preflight_passed", "tasks": len(tasks)})
        return 0
    Trial = patch_pier(args.upstream)
    if args.oracle:
        with gpu_reservation(args.gpu_lock):
            for task in tasks:
                emit({"event": "oracle_start", "task": task})
                result = asyncio.run(trial_run(Trial, args.upstream, task, args.output / "oracle-trials"))
                write_json(args.output / f"oracle-{task}.json", {**identity, **result})
                emit({"event": "oracle_done", "task": task, "passed": result["passed"],
                      "infrastructure_exception": result["infrastructure_exception"]})
                if not result["passed"]:
                    return 1
        return 0
    if args.matrix is None:
        raise ValueError("matrix_required")
    profiles = json.loads(args.matrix.read_text())["profiles"]
    import maxxbench_agent

    artifacts = {}
    with gpu_reservation(args.gpu_lock):
        for profile in profiles:
            emit({"event": "fingerprint_start", "profile": profile["id"]})
            artifacts[profile["id"]] = fingerprint_artifact(
                profile["path"], args.output / "fingerprints" / (profile["id"] + ".json"))
            emit({"event": "fingerprint_done", "profile": profile["id"],
                  "sha256": artifacts[profile["id"]]["sha256"]})
    if args.fingerprint_only:
        emit({"event": "fingerprints_completed", "profiles": len(profiles)})
        return 0
    stable_identity = {key: value for key, value in identity.items() if key != "created_at"}
    stable_identity.update({"profiles": profiles, "worker_sha256": digest(args.worker),
                            "adapter_sha256": digest(ROOT / "scripts/maxxbench_agent.py"),
                            "bridge_sha256": digest(ROOT / "scripts/maxxbench_stark_bridge.py"),
                            "dependency_lock_sha256": digest(Path(__file__).with_suffix(".py.lock")),
                            "template_helper_sha256": digest(args.worker.parent / "stark-template"),
                            "artifact_sha256": {name: row["sha256"] for name, row in artifacts.items()},
                            "host_platform": platform.platform(), "runner_python": sys.version,
                            "stark_source_commit": subprocess.check_output(
                                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()})
    campaign_id = bind_campaign(args.output / "campaign.json", stable_identity)
    image_receipt = args.output / "environment-images.json"
    task_images = json.loads(image_receipt.read_text()) if image_receipt.exists() else {}
    with gpu_reservation(args.gpu_lock):
        for profile in profiles:
            assert_artifact_unchanged(artifacts[profile["id"]])
            emit({"event": "model_start", "profile": profile["id"]})
            with model_bridge(profile, args.worker) as (ready, key, bridge_output):
                maxxbench_agent.configure_connection(ready["base_url"], key, ready["model"])
                write_json(args.output / profile["id"] / "identity.json", {**identity, **ready, "profile": profile})
                for task in tasks:
                    assert_artifact_unchanged(artifacts[profile["id"]])
                    destination = args.output / profile["id"] / f"{task}.json"
                    if destination.exists():
                        existing = json.loads(destination.read_text())
                        if existing.get("campaign_sha256") != campaign_id:
                            raise ValueError("campaign_identity_mismatch")
                        if existing.get("infrastructure_exception"):
                            raise ValueError("trial_infrastructure_failed")
                        bind_environment_image(task_images, profile["id"] + "/" + task,
                                               existing.get("environment_image_id"))
                        continue
                    if shutil.disk_usage(args.output).free < 12 * 1024**3:
                        raise ValueError("insufficient_disk_space")
                    emit({"event": "trial_start", "profile": profile["id"], "task": task})
                    result = asyncio.run(trial_run(Trial, args.upstream, task,
                                                  args.output / profile["id"] / "trials", profile))
                    assert_artifact_unchanged(artifacts[profile["id"]])
                    image_id = result.get("environment_image_id")
                    if not result["infrastructure_exception"]:
                        bind_environment_image(task_images, profile["id"] + "/" + task, image_id)
                        write_json(image_receipt, task_images)
                    write_json(destination, {**result, "campaign_sha256": campaign_id})
                    emit({"event": "trial_done", "profile": profile["id"], "task": task,
                          "passed": result["passed"], "infrastructure_exception": result["infrastructure_exception"]})
                    if result["infrastructure_exception"]:
                        raise ValueError("trial_infrastructure_failed")
            write_json(args.output / profile["id"] / "bridge-totals.json", bridge_output[-1])
    emit({"event": "campaign_completed", "profiles": len(profiles), "tasks_per_profile": len(tasks)})
    return 0


if __name__ == "__main__":
    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit({"event": "stopped", "failure": "interrupted"})
        sys.exit(130)
    except Exception as error:  # noqa: BLE001 - third-party errors may contain private data.
        # Never expose exception strings: third-party exceptions can contain a
        # full HTTP request, model command, model reply, or credential.
        category = str(error) if isinstance(error, ValueError) and str(error) in {
            "upstream_revision_mismatch", "upstream_release_validation_failed", "upstream_task_count_mismatch",
            "docker_unavailable", "compose_unavailable", "invalid_task_selection", "matrix_required",
            "model_bridge_start_failed", "trial_infrastructure_failed", "insufficient_disk_space",
            "model_bridge_protocol_failed", "model_bridge_ownership_unverified", "model_bridge_cleanup_failed",
            "model_bridge_shutdown_failed",
            "environment_identity_unavailable", "environment_identity_changed",
            "campaign_identity_mismatch",
            "artifact_changed", "artifact_empty",
        } else "runner_failed"
        emit({"event": "stopped", "failure": category})
        sys.exit(1)
