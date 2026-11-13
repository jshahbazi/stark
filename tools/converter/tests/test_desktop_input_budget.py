"""Private command framing stays recoverable when parsing headroom is low."""

import json
from types import SimpleNamespace

import pytest
from test_desktop import desktop as desktop
from test_desktop import events, running
from test_desktop import executable as executable
from test_desktop import model as model

import stark_convert.desktop as desktop_module


def serve_chunks(monkeypatch, controller, chunks):
    stdin = SimpleNamespace(fileno=lambda: 424242)
    remaining = iter(chunks)
    read = desktop_module.os.read
    select = desktop_module.select.select

    def read_chunk(descriptor, size):
        return next(remaining, b"") if descriptor == stdin.fileno() else read(descriptor, size)

    def select_input(readers, writers, errors, timeout):
        return ([stdin], [], []) if readers == [stdin] else select(readers, writers, errors, timeout)

    with monkeypatch.context() as patch:
        patch.setattr(desktop_module.sys, "stdin", stdin)
        patch.setattr(desktop_module.os, "read", read_chunk)
        patch.setattr(desktop_module.select, "select", select_input)
        desktop_module.serve(controller)


def chunks(data):
    return [data[start : start + 65536] for start in range(0, len(data), 65536)]


def large_command():
    return json.dumps(
        {"command": "generate", "text": "x" * (2 * 1024 * 1024) + "\U0001f600"},
        ensure_ascii=False,
    ).encode("utf-8")


class RecordingDesktop:
    output_failed = False

    def __init__(self):
        self.commands = []
        self.errors = []

    def library(self):
        pass

    def families(self):
        pass

    def status(self, *_args):
        pass

    def monitor(self):
        pass

    def emit(self, event, **fields):
        assert event == "error"
        self.errors.append(fields["message"])

    def dispatch(self, command):
        self.commands.append(command)
        return command["command"] != "quit"


def test_low_memory_command_is_drained_before_small_retry_and_stop(monkeypatch):
    monkeypatch.setattr(desktop_module, "available_memory_bytes", lambda: 0)
    controller = RecordingDesktop()
    decoded_lengths = []
    loads = desktop_module.json.loads

    def decode(data):
        decoded_lengths.append(len(data))
        return loads(data)

    monkeypatch.setattr(desktop_module.json, "loads", decode)
    data = large_command() + b'\n{"command":"reset"}\n{"command":"stop"}\n'
    serve_chunks(monkeypatch, controller, chunks(data))
    assert controller.commands == [{"command": "reset"}, {"command": "stop"}]
    assert len(controller.errors) == 1
    assert "Available memory is too low" in controller.errors[0]
    assert max(decoded_lengths) < 1024


def test_complete_command_rechecks_memory_before_json_decode(monkeypatch):
    headroom = [128 * 1024 * 1024]
    monkeypatch.setattr(desktop_module, "available_memory_bytes", lambda: headroom[0])
    check = desktop_module.check_json_structure

    def check_and_reduce_memory(data):
        check(data)
        headroom[0] = 0

    monkeypatch.setattr(desktop_module, "check_json_structure", check_and_reduce_memory)
    controller = RecordingDesktop()
    serve_chunks(monkeypatch, controller, chunks(large_command() + b'\n{"command":"reset"}\n'))
    assert controller.commands == [{"command": "reset"}]
    assert len(controller.errors) == 1
    assert "Available memory is too low" in controller.errors[0]


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
def test_rejected_command_keeps_worker_and_real_small_commands_work(desktop, model, monkeypatch, separator):
    running(desktop, model)
    worker = desktop.worker
    desktop.messages.append({"role": "user", "content": "synthetic prior turn"})
    monkeypatch.setattr(desktop_module, "available_memory_bytes", lambda: 0)
    serve_chunks(monkeypatch, desktop, chunks(large_command() + separator + b'{"command":"reset"}\n'))
    assert desktop.worker is worker and worker.process.poll() is None
    assert desktop.messages == [{"role": "system", "content": "synthetic system"}]
    errors = [event for event in events(desktop) if event["event"] == "error"]
    assert len(errors) == 1
    assert "Available memory is too low" in errors[0]["message"]
    serve_chunks(monkeypatch, desktop, [b'{"command":"stop"}\n'])
    assert desktop.worker is None and worker.process.poll() is not None


def test_large_command_with_headroom_preserves_json_dispatch(monkeypatch):
    monkeypatch.setattr(desktop_module, "available_memory_bytes", lambda: 128 * 1024 * 1024)
    controller = RecordingDesktop()
    command = large_command()
    serve_chunks(monkeypatch, controller, chunks(command + b'\n{"command":"quit"}\n'))
    assert controller.commands == [json.loads(command), {"command": "quit"}]
    assert not controller.errors
