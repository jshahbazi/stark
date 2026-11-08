"""Serve one isolated Stark model to a local benchmark without conversation logs.

Run with the locked converter environment. Supply a fresh 64-character lowercase
hex credential through stdin or ``--key-fd``; stdout contains aggregate JSON only.
The HTTP contract and inference implementation are Stark's production classes.
``--thinking`` explicitly controls the template flag for this benchmark process;
unsupported request fields, including ``reasoning_effort``, remain rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from stark_convert.api_limits import model_byte_policy
from stark_convert.api_tools import tool_parser_for
from stark_convert.desktop import Desktop, model_valid
from stark_convert.local_api import APIError, LocalAPI
from stark_convert.template_runtime import _template_executable


class BridgeFailure(Exception):
    """A fixed content-free failure category."""


def read_key(descriptor: int) -> str:
    """Read one bounded credential line, without a file or environment variable."""
    data = bytearray()
    while len(data) < 66:
        part = os.read(descriptor, 66 - len(data))
        if not part:
            break
        data.extend(part)
        if b"\n" in data:
            break
    if re.fullmatch(rb"[0-9a-f]{64}\n?", data) is None:
        raise BridgeFailure("credential_input_invalid")
    return data.decode("ascii").rstrip("\n")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class BenchmarkDesktop(Desktop):
    """An explicit per-process template setting absent from the HTTP contract."""

    benchmark_thinking = False

    def infer(self, *args, **kwargs):
        kwargs["thinking"] = self.benchmark_thinking
        return super().infer(*args, **kwargs)


class AggregateCompletions:
    """Keep counts and total tokens only; request/reply contents are not retained."""

    def __init__(self, desktop):
        self.desktop = desktop
        self.lock = threading.Lock()
        self.requests = 0
        self.completed = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed_seconds = 0.0
        self.http_failures = Counter()

    def __call__(self, payload, on_delta, cancelled, *, byte_limits=None):
        started = time.monotonic()
        with self.lock:
            self.requests += 1
        try:
            options = {} if byte_limits is None else {"byte_limits": byte_limits}
            reply = self.desktop.api_complete(payload, on_delta, cancelled, **options)
            usage = reply.get("usage", {})
            with self.lock:
                self.completed += 1
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
            return reply
        except APIError as error:
            with self.lock:
                self.http_failures[str(error.status)] += 1
            raise
        except BaseException:  # noqa: BLE001 - discard private inference exception details.
            with self.lock:
                self.http_failures["internal"] += 1
            raise APIError(503, "Benchmark completion failed.", "benchmark_completion_failed") from None
        finally:
            with self.lock:
                self.elapsed_seconds += time.monotonic() - started

    def summary(self):
        with self.lock:
            return {
                "requests": self.requests,
                "completed": self.completed,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "request_elapsed_seconds": round(self.elapsed_seconds, 3),
                "http_failures": dict(self.http_failures),
            }


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def serve_model(args, key, stop, *, desktop_class=BenchmarkDesktop, api_class=LocalAPI):
    """Own the isolated model, API and cleanup through startup failures or stop."""
    desktop = None
    api = None
    totals = None
    failure = None
    cleanup_failed = False
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stark-maxxbench-") as directory, open(os.devnull, "w") as sink:
        try:
            model = args.model.expanduser().resolve(strict=True)
            worker = args.worker.expanduser().resolve(strict=True)
            if not model_valid(model):
                raise BridgeFailure("model_invalid")
            worker_hash = file_sha256(worker)
            helper_hash = file_sha256(_template_executable())
            try:
                byte_limits, byte_policy, policy_digest = model_byte_policy(
                    model, args.context, args.max_tokens, args.step_limit, args.protocol_envelope)
            except (OSError, ValueError, TypeError):
                raise BridgeFailure("benchmark_protocol_capacity_unavailable") from None
            desktop = desktop_class(worker, Path(directory), output=sink, timeout=args.request_timeout,
                                    benchmark_byte_policy=byte_policy)
            desktop.benchmark_thinking = args.thinking
            identifier = "maxxbench-model"
            row = {"id": identifier, "name": args.model_id, "path": str(model), "source": "local:maxxbench"}
            if args.mtp != "default":
                row["mtp_enabled"] = args.mtp == "on"
            desktop.models = [row]
            desktop.run(desktop.epoch, identifier, "chat", guided_context=args.context)
            if file_sha256(worker) != worker_hash or file_sha256(_template_executable()) != helper_hash:
                raise BridgeFailure("executable_changed_during_load")
            with desktop.lock:
                if desktop.worker is None or desktop.active_model != identifier or desktop.active_mode != "chat":
                    raise BridgeFailure("model_admission_failed")
                if desktop.api_context != args.context:
                    raise BridgeFailure("requested_context_unavailable")
                if args.max_tokens > min(desktop.output_limit, args.context - 1):
                    raise BridgeFailure("requested_output_limit_unavailable")
                desktop.api_maximum = args.max_tokens
                actual_parser = tool_parser_for(desktop.codec)
                if actual_parser != byte_policy["tool_profile"]["parser"]:
                    raise BridgeFailure("benchmark_tool_profile_mismatch")
            if stop.is_set():
                raise BridgeFailure("startup_cancelled")
            totals = AggregateCompletions(desktop)
            api = api_class(
                desktop.api_models, totals, api_key=key, port=args.port,
                status=desktop.api_status, cancellable=True, request_timeout=args.request_timeout,
                byte_limits=byte_limits, max_connections=1,
            )
            desktop.api = api
            details = api.start()
            models = desktop.api_models()
            if len(models) != 1:
                raise BridgeFailure("api_model_unavailable")
            emit({
                "event": "ready", "base_url": details["base_url"], "model": models[0]["id"],
                "context_tokens": args.context, "max_output_tokens": args.max_tokens,
                "template_enable_thinking": args.thinking, "mtp_requested": args.mtp,
                "mtp_default_enabled": models[0]["defaults"]["mtp"],
                "prefix_cache_supported": desktop.worker.prefix_cache,
                "prefix_chunk_tokens": desktop.worker.prefix_chunk_tokens,
                "worker_pid": getattr(getattr(desktop.worker, "process", None), "pid", None),
                "worker_sha256": worker_hash, "template_helper_sha256": helper_hash,
                "api_byte_policy": byte_policy, "api_byte_policy_sha256": policy_digest,
                "tool_parser": actual_parser,
                "logging": "aggregate_only",
            })
            stop.wait()
        except BridgeFailure as error:
            failure = str(error)
        except BaseException:  # noqa: BLE001 - never serialize private exception details.
            failure = "bridge_failed"
        finally:
            if desktop is not None:
                with desktop.lock:
                    desktop.api_closing = True
                    desktop.api_condition.notify_all()
                # Stop first to cancel native work; close the listener afterward.
                for cleanup in (lambda: desktop.stop(quiet=True), lambda: api.close() if api else None):
                    try:
                        cleanup()
                    except BaseException:  # noqa: BLE001 - attempt every cleanup without exception dumps.
                        cleanup_failed = True
    emit({
        "event": "stopped", "failure": failure, "cleanup_failed": cleanup_failed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **(totals.summary() if totals else {}),
    })
    return 1 if failure or cleanup_failed else 0


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--model-id", required=True)
    result.add_argument("--context", type=positive, required=True)
    result.add_argument("--max-tokens", type=positive, default=4096)
    result.add_argument("--step-limit", type=positive, required=True)
    result.add_argument("--protocol-envelope", type=json.loads, required=True)
    result.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--mtp", choices=("default", "on", "off"), default="default")
    result.add_argument("--port", type=int, default=0)
    result.add_argument("--request-timeout", type=positive, default=1800)
    result.add_argument("--key-fd", type=int, default=0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 <= args.port <= 65535 or args.key_fd < 0 or not 2 <= args.context <= 0xFFFFFFFF:
        emit({"event": "stopped", "failure": "invalid_arguments"})
        return 1
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", args.model_id) is None:
        emit({"event": "stopped", "failure": "invalid_model_id"})
        return 1
    stop = threading.Event()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        try:
            key = read_key(args.key_fd)
        except BaseException:  # noqa: BLE001 - credential errors must remain content-free.
            emit({"event": "stopped", "failure": "credential_input_failed"})
            return 1
        try:
            return serve_model(args, key, stop)
        except BaseException:  # noqa: BLE001 - cover temporary-directory and output failures.
            emit({"event": "stopped", "failure": "bridge_failed"})
            return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
