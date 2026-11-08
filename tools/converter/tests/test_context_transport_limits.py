"""Explicit context transport budgets leave default limits unchanged."""

import io
import json
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest import mock

import pytest

from stark_convert import template_runtime as runtime
from stark_convert.chat import ChatRequestTooLarge, Worker


def limits(**changes):
    values = dict(request_bytes=4 * 1024**2, value_bytes=4 * 1024**2,
                  output_bytes=4 * 1024**2, sequence_items=16384, value_nodes=65536,
                  fuel=1_000_000, cumulative_allocation_bytes=128 * 1024**2,
                  live_allocation_bytes=256 * 1024**2, cpu_seconds=4, wall_seconds=10.0)
    values.update(changes)
    return runtime.TemplateRuntimeLimits(**values)


def test_explicit_template_budget_preserves_larger_values_and_default_rejection():
    value = {"value": "x" * (2 * 1024**2)}
    with pytest.raises(runtime.TemplateSizeLimit):
        runtime._request_bytes("{{ value }}", value)
    encoded = runtime._request_bytes("{{ value }}", value, limits=limits())
    assert len(encoded) > 2 * 1024**2
    assert json.loads(encoded)["arguments"] == value
    with pytest.raises(runtime.TemplateSizeLimit):
        runtime._request_bytes("{{ value }}", value)


def test_template_policy_is_frozen_and_validated():
    selected = limits()
    with pytest.raises(FrozenInstanceError):
        selected.output_bytes = 1
    for key, value in (("request_bytes", 0), ("fuel", True), ("wall_seconds", float("inf"))):
        with pytest.raises(ValueError):
            limits(**{key: value})
    assert limits(cumulative_allocation_bytes=2 * sys.maxsize + 1).cumulative_allocation_bytes > sys.maxsize
    with pytest.raises(ValueError):
        limits(cumulative_allocation_bytes=2 * sys.maxsize + 2)
    with pytest.raises(ValueError):
        limits(output_bytes=sys.maxsize + 1)


def test_template_transport_round_trips_unicode_and_controls_without_normalization():
    arguments = {"value": "e\u0301𐐀\n\u0000\\\""}
    data = runtime._request_bytes("{{ value }}", arguments, limits=limits())
    assert json.loads(data)["arguments"] == arguments
    assert "e\u0301𐐀".encode() in data


def test_template_instances_own_their_policy_without_global_mutation():
    with mock.patch.object(runtime, "_template_executable", return_value="/private/helper"), \
            mock.patch.object(runtime.IsolatedTemplate, "_evaluate", return_value=""):
        default = runtime.IsolatedTemplate("{{ value }}")
        selected = runtime.IsolatedTemplate("{{ value }}", runtime_limits=limits())
    assert default._runtime_limits is None
    assert selected._runtime_limits == limits()
    assert runtime.MAX_REQUEST_BYTES == 2 * 1024**2
    assert runtime.MAX_OUTPUT_BYTES == 1024**2


def fake_worker(request_limit=None):
    worker = Worker.__new__(Worker)
    worker.request_byte_limit = request_limit
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    worker.read = lambda: {"token_count": 1}
    return worker


def test_worker_outgoing_budget_is_explicit_and_does_not_truncate():
    request = {"operation": "tokenize", "text": "x" * (2 * 1024**2) + "e\u0301𐐀\n\u0000\\\""}
    default = fake_worker()
    with pytest.raises(ChatRequestTooLarge):
        default._request(request)
    selected = fake_worker(4 * 1024**2)
    assert selected._request(request) == {"token_count": 1}
    encoded = selected.process.stdin.getvalue()
    assert len(encoded) > 2 * 1024**2
    assert json.loads(encoded) == request
    assert default.process.stdin.getvalue() == b""


def long_render_limits():
    return limits(sequence_items=100000, fuel=1_000_000_000_000,
                  cumulative_allocation_bytes=2 * 1024**3, cpu_seconds=30, wall_seconds=60.0)


LONG_RENDER = (
    "{% for a in range(100000) %}{% for b in range(100000) %}"
    "{% if a < 0 %}x{% endif %}{% endfor %}{% endfor %}"
)


def test_expanded_render_budget_still_cancels_and_reaps_child(monkeypatch):
    import psutil

    children = []
    original = runtime.Popen

    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runtime, "Popen", capture)
    template = runtime.IsolatedTemplate(LONG_RENDER, runtime_limits=long_render_limits())
    cancelled = threading.Event()
    timer = threading.Timer(0.15, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(runtime.TemplateCancelled):
            template.render_with_cancellation(cancelled)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3
    assert all(child.poll() is not None and not psutil.pid_exists(child.pid) for child in children)


def test_expanded_helper_stops_when_owner_exits_without_cleanup():
    import psutil

    policy = runtime.asdict(long_render_limits())
    request = json.dumps({"source": LONG_RENDER, "arguments": {}})
    owner = (
        "import json,os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.argv[1],'--runtime-limits',sys.argv[2]],stdin=subprocess.PIPE,"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "p.stdin.write(sys.argv[3].encode()); p.stdin.close(); print(p.pid,flush=True); "
        "time.sleep(0.2); os._exit(0 if p.poll() is None else 4)"
    )
    result = subprocess.run([sys.executable, "-c", owner, str(runtime._template_executable()),
                             json.dumps(policy), request], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, timeout=3, check=True)
    assert result.stdout.strip().isdigit()
    pid = int(result.stdout.strip())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            child = psutil.Process(pid)
            if child.status() == psutil.STATUS_ZOMBIE:
                assert child.memory_info().rss == 0
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.025)
    pytest.fail("Expanded template helper outlived its owner.")
