"""Model-free tests for opt-in aggregate host metrics."""

from __future__ import annotations

import json

import pytest
from hybrid_host_metrics import (
    BUCKETS,
    HostMetricsSampler,
    RequestMarker,
    cpu_interval,
    parse_cpu_seconds,
    parse_gpu_clients,
    parse_gpu_device_utilization,
    parse_process_snapshot,
    parse_vm_counters,
)


def process(parent, start="one", cpu=0.0, name="other"):
    return {
        "parent": parent,
        "start": start,
        "cpu_seconds": cpu,
        "name": name,
    }


def test_parse_process_snapshot_preserves_identity_and_cpu_time():
    raw = "123 1 Wed Oct 29 09:00:00 2026 1-02:03:04.50 /Applications/WindowServer"
    parsed = parse_process_snapshot(raw)

    assert parsed[123] == {
        "parent": 1,
        "start": "Wed Oct 29 09:00:00 2026",
        "cpu_seconds": 93784.5,
        "name": "WindowServer",
    }
    assert parse_cpu_seconds("00:00:01.25") == 1.25
    with pytest.raises(ValueError, match="invalid_process_cpu_time"):
        parse_cpu_seconds("bad")


def test_parse_gpu_clients_only_aggregates_metal_counters():
    parsed = parse_gpu_clients(
        [
            {
                "IOUserClientCreator": "pid 10, stark-generate",
                "IORegistryEntryID": 77,
                "AppUsage": [
                    {"API": "Metal", "accumulatedGPUTime": 12},
                    {"API": "Metal", "accumulatedGPUTime": 7},
                    {"API": "OpenGL", "accumulatedGPUTime": 500},
                ],
            },
            {
                "IOUserClientCreator": "pid 10, stark-generate",
                "IORegistryEntryID": 77,
                "AppUsage": [{"API": "Metal", "accumulatedGPUTime": 3}],
            },
            {
                "IOUserClientCreator": "pid 10, stark-generate",
                "AppUsage": [{"API": "Metal", "accumulatedGPUTime": 5}],
            },
            {
                "IOUserClientCreator": "unparseable",
                "AppUsage": [{"API": "Metal", "accumulatedGPUTime": 99}],
            },
        ]
    )

    assert parsed == {
        (10, 77): {
            "pid": 10,
            "name": "stark-generate",
            "counter": 22,
            "entry_id_available": True,
        },
        (10, None): {
            "pid": 10,
            "name": "stark-generate",
            "counter": 5,
            "entry_id_available": False,
        },
    }


def test_parse_vm_counters_requires_both_values():
    assert parse_vm_counters("Pageouts: 2.\nSwapouts: 3.\n") == {
        "pageouts": 2,
        "swapouts": 3,
    }
    with pytest.raises(ValueError, match="vm_counters_unavailable"):
        parse_vm_counters("Pageouts: 2.\n")


def test_parse_gpu_device_utilization_returns_none_when_unavailable():
    assert (
        parse_gpu_device_utilization(
            [
                {"PerformanceStatistics": {"Device Utilization %": 42}},
                {"PerformanceStatistics": {"Device Utilization %": 31}},
            ]
        )
        == 42
    )
    assert parse_gpu_device_utilization([{"PerformanceStatistics": "missing"}]) is None


def test_cpu_interval_uses_fixed_buckets_and_keeps_worker_child(monkeypatch):
    monkeypatch.setattr("hybrid_host_metrics.os.getpid", lambda: 99)
    previous = {
        99: process(1, cpu=1, name="benchmark"),
        20: process(99, cpu=10, name="stark-generate"),
        21: process(1, cpu=10, name="WindowServer"),
        22: process(1, cpu=10, name="Claude Helper"),
    }
    current = {
        99: process(1, cpu=2, name="benchmark"),
        20: process(99, cpu=10.5, name="stark-generate"),
        21: process(1, cpu=10.25, name="WindowServer"),
        22: process(1, cpu=10.75, name="Claude Helper"),
    }

    values, discontinuities = cpu_interval(previous, current, 1.0, worker_pid=20)

    assert tuple(values) == BUCKETS
    assert values["worker"] == 50
    assert values["window_server"] == 25
    assert values["other"] == 75
    assert not discontinuities


def test_gpu_delta_handles_appearance_disappearance_reset_and_pid_reuse():
    monitor = HostMetricsSampler(worker_pid=2)
    previous_gpu = {
        (1, 11): {"pid": 1, "name": "WindowServer", "counter": 100},
        (2, 22): {"pid": 2, "name": "stark-generate", "counter": 100},
        (3, 33): {"pid": 3, "name": "Other", "counter": 20},
        (4, None): {"pid": 4, "name": "Other", "counter": 30},
        (5, 55): {"pid": 5, "name": "Claude Helper", "counter": 300},
    }
    current_gpu = {
        (1, 11): {"pid": 1, "name": "WindowServer", "counter": 150},
        (2, 22): {"pid": 2, "name": "stark-generate", "counter": 90},
        (4, None): {"pid": 4, "name": "Other", "counter": 40},
        (5, 55): {"pid": 5, "name": "Claude Helper", "counter": 350},
        (6, 66): {"pid": 6, "name": "SystemGraphics", "counter": 200},
    }
    previous_processes = {
        1: process(0, name="WindowServer"),
        2: process(0, name="stark-generate"),
        3: process(0, name="Other"),
        4: process(0, name="Other"),
        5: process(0, start="old", name="Claude Helper"),
        6: process(0, name="SystemGraphics"),
    }
    current_processes = {
        1: process(0, name="WindowServer"),
        2: process(0, name="stark-generate"),
        4: process(0, name="Other"),
        5: process(0, start="new", name="Claude Helper"),
        6: process(0, name="SystemGraphics"),
    }

    values, changes = monitor._gpu_interval(previous_gpu, current_gpu, previous_processes, current_processes)

    assert values["window_server"] == 50
    assert values["worker"] == 0
    assert values["other"] == 10
    assert changes["gpu_client_appeared"] == 1
    assert changes["gpu_client_disappeared"] == 1
    assert changes["gpu_counter_reset"] == 1
    assert changes["gpu_pid_identity_changed"] == 1
    assert changes["gpu_pid_aggregate_ambiguity"] == 1


def test_request_aggregation_excludes_boundary_intervals_and_sanitizes():
    monitor = HostMetricsSampler(worker_pid=10)
    monitor.rows = [
        {
            "start": 9.5,
            "end": 10.5,
            "interval_seconds": 1.0,
            "cpu_percent": {bucket: 10.0 for bucket in BUCKETS},
            "gpu_counter_delta_raw": {bucket: 100 for bucket in BUCKETS},
            "gpu_device_utilization_percent": 20,
            "thermal_state": 0,
            "counter_discontinuities": {"gpu_counter_reset": 1},
            "source_availability": {"gpu_counter_delta": True},
        },
        {
            "start": 10.5,
            "end": 11.5,
            "interval_seconds": 1.0,
            "cpu_percent": {bucket: 20.0 for bucket in BUCKETS},
            "gpu_counter_delta_raw": {bucket: 200 for bucket in BUCKETS},
            "gpu_device_utilization_percent": 40,
            "thermal_state": 1,
            "counter_discontinuities": {},
            "source_availability": {"gpu_counter_delta": True},
        },
        {
            "start": 11.5,
            "end": 12.5,
            "interval_seconds": 1.0,
            "cpu_percent": None,
            "gpu_counter_delta_raw": None,
            "gpu_device_utilization_percent": None,
            "thermal_state": None,
            "counter_discontinuities": {},
            "source_availability": {"gpu_counter_delta": False},
        },
    ]
    marker = RequestMarker(
        started=10.0,
        vm_before={"pageouts": 10, "swapouts": 20},
        finished=12.0,
        vm_after={"pageouts": 12, "swapouts": 19},
    )

    result = monitor.request_result(marker)
    serialized = json.dumps(result)

    assert result["sample_interval_count"] == 1
    assert result["boundary_intervals_excluded"] == 2
    assert result["covered_seconds"] == 1.0
    assert result["cpu_percent_mean_by_bucket"]["other"] == 20
    assert result["gpu_counter_delta_raw_by_bucket"]["other"] == 200
    assert result["gpu_device_utilization_percent_mean"] == 40
    assert result["vm_counter_deltas"] == {"pageouts": 2, "swapouts": None}
    assert result["vm_counters_available"] is True
    assert result["counter_discontinuities"] == {"vm_swapouts_counter_reset": 1}
    assert "WindowServer" not in serialized
    assert '"pid"' not in serialized


def test_unavailable_gpu_source_is_null_not_zero():
    monitor = HostMetricsSampler(worker_pid=None)
    monitor.rows = [
        {
            "start": 1.0,
            "end": 2.0,
            "interval_seconds": 1.0,
            "cpu_percent": None,
            "gpu_counter_delta_raw": None,
            "gpu_device_utilization_percent": None,
            "thermal_state": None,
            "counter_discontinuities": {},
            "source_availability": {},
        }
    ]
    marker = RequestMarker(1.0, None, 2.0, None)

    result = monitor.request_result(marker)

    assert result["gpu_counter_delta_raw_by_bucket"] is None
    assert result["gpu_counter_rate_raw_per_covered_second_by_bucket"] is None
    assert result["cpu_percent_mean_by_bucket"] is None
    assert result["gpu_device_utilization_percent_mean"] is None
    assert result["thermal_state_counts"] is None
    assert result["vm_counters_available"] is False
