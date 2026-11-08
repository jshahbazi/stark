"""Model-free checks of the benchmark bridge's privacy and resource ownership."""

import argparse
import io
import json
import os
import threading
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import maxxbench_stark_bridge as bridge
import pytest
from stark_convert.local_api import APIError


def args(tmp_path, **changes):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    worker = tmp_path / "worker"
    worker.write_bytes(b"worker")
    values = {"model": model, "worker": worker, "model_id": "sample", "context": 32768,
              "max_tokens": 4096, "thinking": True, "mtp": "off", "port": 0, "request_timeout": 10}
    values.update(changes)
    return argparse.Namespace(**values)


class FakeDesktop:
    instances: ClassVar[list] = []

    def __init__(self, worker, data_dir, **kwargs):
        self.__class__.instances.append(self)
        self.lock = threading.RLock()
        self.api_condition = threading.Condition(self.lock)
        self.data_dir = data_dir
        self.epoch = 0
        self.worker = SimpleNamespace(prefix_cache=True, prefix_chunk_tokens=64, process=SimpleNamespace(pid=1234))
        self.output_limit = 4096
        self.active_mode = None
        self.active_model = None
        self.stopped = False
        self.sink = kwargs["output"]

    def run(self, epoch, identifier, mode, *, guided_context):
        self.active_model, self.active_mode = identifier, mode
        self.api_context = guided_context
        self.sink.write("private model prompt and response")

    def api_models(self):
        return [{"id": "sample", "defaults": {"mtp": False}}]

    def api_status(self):
        return {}

    def stop(self, quiet):
        self.stopped = quiet


class FakeAPI:
    instances: ClassVar[list] = []
    fail_start = False

    def __init__(self, *args, **kwargs):
        self.__class__.instances.append(self)
        self.closed = False
        self.key = kwargs["api_key"]

    def start(self):
        if self.fail_start:
            raise RuntimeError("private-marker exception")
        return {"base_url": "http://127.0.0.1:12345/v1", "api_key": self.key, "api_session": "private-session"}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def metadata(monkeypatch, tmp_path):
    FakeAPI.instances = []
    FakeDesktop.instances = []
    monkeypatch.setattr(bridge, "model_valid", lambda _: True)
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    monkeypatch.setattr(bridge, "_template_executable", lambda: helper)


def run_bridge(arguments):
    stop = threading.Event()
    output = io.StringIO()
    with mock.patch.object(stop, "wait", return_value=True), redirect_stdout(output):
        result = bridge.serve_model(arguments, "a" * 64, stop, desktop_class=FakeDesktop, api_class=FakeAPI)
    return result, output.getvalue(), [json.loads(line) for line in output.getvalue().splitlines()]


def test_ready_has_only_aggregates_and_cleanup_removes_isolated_data(tmp_path):
    result, raw, rows = run_bridge(args(tmp_path))
    assert result == 0
    assert [row["event"] for row in rows] == ["ready", "stopped"]
    assert rows[0]["template_enable_thinking"] is True
    assert rows[0]["max_output_tokens"] == 4096
    assert rows[0]["worker_pid"] == 1234
    assert "a" * 64 not in raw and "private" not in raw
    desktop = FakeDesktop.instances[0]
    assert desktop.stopped and FakeAPI.instances[0].closed
    assert not desktop.data_dir.exists()
    assert desktop.models[0]["mtp_enabled"] is False


def test_startup_failure_is_content_free_and_closes_both_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeAPI, "fail_start", True)
    result, raw, rows = run_bridge(args(tmp_path))
    assert result == 1
    assert len(rows) == 1 and rows[0]["failure"] == "bridge_failed"
    assert "private-marker" not in raw and "Traceback" not in raw
    assert FakeDesktop.instances[0].stopped and FakeAPI.instances[0].closed


def test_requested_output_limit_is_not_silently_clamped(tmp_path):
    result, _, rows = run_bridge(args(tmp_path, max_tokens=8192))
    assert result == 1
    assert rows[0]["failure"] == "requested_output_limit_unavailable"
    assert not FakeAPI.instances and FakeDesktop.instances[0].stopped


def test_requested_context_is_not_silently_clamped(tmp_path, monkeypatch):
    original = FakeDesktop.run

    def clamp(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.api_context = 4096

    monkeypatch.setattr(FakeDesktop, "run", clamp)
    result, _, rows = run_bridge(args(tmp_path))
    assert result == 1 and rows[0]["failure"] == "requested_context_unavailable"
    assert FakeDesktop.instances[0].stopped


def test_key_pipe_is_bounded_and_accepts_fragmented_reads(monkeypatch):
    parts = iter([b"b" * 20, b"b" * 44, b"\n"])
    monkeypatch.setattr(os, "read", lambda *_: next(parts))
    assert bridge.read_key(3) == "b" * 64
    monkeypatch.setattr(os, "read", lambda *_: b"c" * 66)
    with pytest.raises(bridge.BridgeFailure, match="credential_input_invalid"):
        bridge.read_key(3)


def test_aggregate_wrapper_does_not_retain_conversation_or_private_exception():
    payload = {"messages": [{"role": "user", "content": "private prompt"}]}
    desktop = SimpleNamespace(api_complete=lambda *_: {"usage": {"prompt_tokens": 10, "completion_tokens": 20},
                                                      "choices": [{"message": {"content": "private reply"}}]})
    totals = bridge.AggregateCompletions(desktop)
    assert totals(payload, None, threading.Event())["choices"]
    assert totals.summary()["prompt_tokens"] == 10
    assert "private" not in json.dumps(totals.summary())
    desktop.api_complete = mock.Mock(side_effect=RuntimeError("private exception"))
    with pytest.raises(APIError, match="Benchmark completion failed"):
        totals(payload, None, threading.Event())
    assert totals.summary()["http_failures"] == {"internal": 1}
    assert totals.summary()["requests"] == 2 and totals.summary()["completed"] == 1


def test_thinking_flag_reaches_template_without_changing_request_fields():
    desktop = object.__new__(bridge.BenchmarkDesktop)
    desktop.benchmark_thinking = True
    with mock.patch.object(bridge.Desktop, "infer", return_value="reply") as infer:
        assert desktop.infer([], 100, seed=7) == "reply"
    assert infer.call_args.kwargs == {"seed": 7, "thinking": True}
