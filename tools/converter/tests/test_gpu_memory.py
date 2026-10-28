"""Private, bounded GPU memory telemetry transport."""

import io
import json
import os
import threading
import time
from unittest.mock import patch

import pytest

from stark_convert.chat import _MAX_GPU_MEMORY_BYTES, Worker, _parse_gpu_memory_line, _popen_with_gpu_memory
from stark_convert.decision import ScoreWorker
from stark_convert.desktop import Desktop


@pytest.mark.parametrize("value", [0, 123456, _MAX_GPU_MEMORY_BYTES])
def test_gpu_memory_parser_accepts_bounded_integers(value):
    assert _parse_gpu_memory_line(json.dumps({"gpu_memory_bytes": value}).encode()) == value


@pytest.mark.parametrize("value", [True, 1.0, -1, _MAX_GPU_MEMORY_BYTES + 1, "12", None])
def test_gpu_memory_parser_rejects_invalid_numbers(value):
    assert _parse_gpu_memory_line(json.dumps({"gpu_memory_bytes": value}).encode()) is None


def test_gpu_memory_reader_delivers_updates_independently_and_closes_pipe():
    read_fd, write_fd = os.pipe()
    received = []
    ready = threading.Event()
    worker = object.__new__(Worker)
    worker._start_gpu_memory_reader(read_fd, lambda _worker, amount: (received.append(amount), ready.set()))
    try:
        os.write(
            write_fd,
            b'{"gpu_memory_bytes":true}\n'
            b'{"gpu_memory_bytes":4096}\n' + b"x" * 256 + b"\n" + b'{"gpu_memory_bytes":8192}\n',
        )
        assert ready.wait(2)
        deadline = time.monotonic() + 2
        while len(received) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [4096, 8192]
    finally:
        worker._close_gpu_memory_reader()
        os.close(write_fd)
    assert worker._gpu_memory_fd is None
    assert worker._gpu_memory_thread is None or not worker._gpu_memory_thread.is_alive()


def test_gpu_memory_reader_clears_sample_on_unexpected_eof():
    read_fd, write_fd = os.pipe()
    received = []
    ready = threading.Event()
    worker = object.__new__(Worker)
    worker._closed = False
    worker._start_gpu_memory_reader(read_fd, lambda _worker, amount: (received.append(amount), ready.set()))
    os.write(write_fd, b'{"gpu_memory_bytes":4096}\n')
    assert ready.wait(2)
    ready.clear()
    os.close(write_fd)
    deadline = time.monotonic() + 2
    while not received or received[-1] is not None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert received == [4096, None]
    worker._close_gpu_memory_reader()


def test_gpu_memory_reader_closes_its_fd_after_child_exits():
    read_fd, write_fd = os.pipe()
    worker = object.__new__(Worker)
    worker._start_gpu_memory_reader(read_fd, lambda *_args: None)
    os.close(write_fd)
    worker._gpu_memory_thread.join(timeout=2)
    assert not worker._gpu_memory_thread.is_alive()
    assert worker._gpu_memory_fd is None
    with pytest.raises(OSError):
        os.fstat(read_fd)
    worker._close_gpu_memory_reader()


def test_gpu_memory_reader_start_failure_closes_owned_fd():
    read_fd, write_fd = os.pipe()
    worker = object.__new__(Worker)
    with patch("stark_convert.chat.threading.Thread.start", side_effect=RuntimeError("start failed")):
        with pytest.raises(RuntimeError):
            worker._start_gpu_memory_reader(read_fd, lambda *_args: None)
    os.close(write_fd)
    assert worker._gpu_memory_fd is None
    assert worker._gpu_memory_thread is None
    assert worker._gpu_memory_stop is None
    with pytest.raises(OSError):
        os.fstat(read_fd)
    worker._close_gpu_memory_reader()


def test_gpu_memory_spawn_uses_nonblocking_child_fd_and_closes_parent_writer():
    observed = {}

    class Process:
        pass

    def fake_popen(_command, **kwargs):
        fd = int(kwargs["env"]["STARK_GPU_MEMORY_FD"])
        observed["fd"] = fd
        observed["blocking"] = os.get_blocking(fd)
        observed["passed"] = fd in kwargs["pass_fds"]
        return Process()

    with patch("stark_convert.chat.subprocess.Popen", fake_popen):
        process, read_fd = _popen_with_gpu_memory(["worker"], callback=lambda *_args: None, env={}, pass_fds=())
    try:
        assert isinstance(process, Process)
        assert observed["blocking"] is False
        assert observed["passed"] is True
        with pytest.raises(OSError):
            os.fstat(observed["fd"])
    finally:
        os.close(read_fd)


def test_gpu_memory_spawn_is_disabled_without_callback():
    observed = {}

    def fake_popen(_command, **kwargs):
        observed.update(kwargs)
        return object()

    with patch("stark_convert.chat.subprocess.Popen", fake_popen):
        process, read_fd = _popen_with_gpu_memory(["worker"], callback=None, env={"STARK_GPU_MEMORY_FD": "99"})
    assert process is not None
    assert read_fd is None
    assert "STARK_GPU_MEMORY_FD" not in observed["env"]
    assert "pass_fds" not in observed


def test_score_worker_launch_keeps_telemetry_opt_in(monkeypatch):
    observed = {}
    process = object()

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return process

    monkeypatch.setenv("STARK_GPU_MEMORY_FD", "99")
    with patch("stark_convert.chat.subprocess.Popen", fake_popen):
        worker = ScoreWorker("score-worker", "model", 5)
    assert worker.process is process
    assert "STARK_GPU_MEMORY_FD" not in observed["env"]
    assert worker._gpu_memory_thread is None
    assert worker._gpu_memory_stop is None


def test_gpu_memory_spawn_failure_closes_both_pipe_ends(monkeypatch):
    descriptors = []
    original_pipe = os.pipe

    def tracked_pipe():
        pair = original_pipe()
        descriptors.extend(pair)
        return pair

    monkeypatch.setattr("stark_convert.chat.os.pipe", tracked_pipe)
    monkeypatch.setattr(
        "stark_convert.chat.subprocess.Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed"))
    )
    with pytest.raises(OSError):
        _popen_with_gpu_memory(["worker"], callback=lambda *_args: None)
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_worker_close_falls_back_to_child_signal_if_group_signal_is_denied():
    import signal

    class Process:
        pid = 12345
        stdin = io.BytesIO()
        stdout = io.BytesIO()

        def __init__(self):
            self.returncode = None
            self.signals = []

        def poll(self):
            return self.returncode

        def send_signal(self, signal_number):
            self.signals.append(signal_number)
            self.returncode = -signal_number

        def wait(self, timeout=None):
            return self.returncode

    worker = object.__new__(Worker)
    worker.process = Process()
    worker._closed = False
    worker.images = False
    worker.image_verification = "unverified"
    with patch("stark_convert.chat.os.killpg", side_effect=PermissionError):
        worker.close()
    assert worker.process.signals == [signal.SIGTERM]
    assert worker.process.poll() is not None


def test_desktop_gpu_memory_events_are_fenced_to_current_worker(tmp_path):
    output = io.StringIO()
    desktop = Desktop(tmp_path / "worker", tmp_path / "library", output)
    try:

        class StubWorker:
            def close(self):
                pass

        worker = StubWorker()
        desktop.worker = worker
        desktop.report_gpu_memory(desktop.epoch, "model-1", "chat", worker, 1234)
        desktop.report_gpu_memory(desktop.epoch, "old-model", "chat", object(), 5678)
        desktop.report_gpu_memory(desktop.epoch - 1, "stale-epoch", "decision", worker, 9012)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        assert events == [
            {
                "event": "gpu_memory",
                "model_id": "model-1",
                "mode": "chat",
                "gpu_memory_bytes": 1234,
            }
        ]
    finally:
        desktop.stop(quiet=True)
