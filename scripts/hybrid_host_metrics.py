"""Opt-in, privacy-preserving host measurements for hybrid benchmarks."""

from __future__ import annotations

import ctypes
import os
import plistlib
import re
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BUCKETS = ("worker", "model_processes", "window_server", "system_graphics", "other")
GPU_UNIT = "unverified IORegistry accumulatedGPUTime units"
MODEL_NAME_MARKERS = ("llama", "stark", "tensorfold", "vllm", "mlx", "qwen")
GRAPHICS_NAME_MARKERS = (
    "dock",
    "controlcenter",
    "windowmanager",
    "screencapture",
    "coreanimation",
)


@dataclass
class RequestMarker:
    started: float
    vm_before: dict[str, int] | None
    finished: float | None = None
    vm_after: dict[str, int] | None = None


def parse_cpu_seconds(value: str) -> float:
    days, separator, clock = value.partition("-")
    if not separator:
        days, clock = "0", value
    parts = clock.split(":")
    try:
        return int(days) * 86400 + sum(float(part) * (60**index) for index, part in enumerate(reversed(parts)))
    except ValueError:
        raise ValueError("invalid_process_cpu_time") from None


def parse_process_snapshot(text: str) -> dict[int, dict]:
    """Parse ps output while keeping process identifiers in transient memory only."""
    result = {}
    for line in text.splitlines():
        fields = line.split(None, 8)
        if len(fields) != 9:
            continue
        try:
            pid, parent = int(fields[0]), int(fields[1])
            cpu_seconds = parse_cpu_seconds(fields[7])
        except ValueError:
            continue
        result[pid] = {
            "parent": parent,
            "start": " ".join(fields[2:7]),
            "cpu_seconds": cpu_seconds,
            "name": Path(fields[8]).name,
        }
    return result


def parse_gpu_clients(devices: list[dict]) -> dict[tuple[int, int | None], dict]:
    """Parse Metal AppUsage totals, retaining entry IDs only in transient memory."""
    result: dict[tuple[int, int | None], dict] = {}
    for device in devices:
        creator = device.get("IOUserClientCreator", "")
        match = re.fullmatch(r"pid\s+(\d+),\s*(.*)", creator)
        if match is None:
            continue
        pid, name = int(match.group(1)), match.group(2)
        entry_id = device.get("IORegistryEntryID")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id < 0:
            entry_id = None
        key = (pid, entry_id)
        row = result.setdefault(
            key,
            {
                "pid": pid,
                "name": name,
                "counter": 0,
                "entry_id_available": entry_id is not None,
            },
        )
        if row["name"] != name:
            row["name"] = ""
        usage_rows = device.get("AppUsage", [])
        if not isinstance(usage_rows, list):
            continue
        for usage in usage_rows:
            if not isinstance(usage, dict) or usage.get("API") != "Metal":
                continue
            value = usage.get("accumulatedGPUTime")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                row["counter"] += value
    return result


def parse_vm_counters(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        normalized = key.strip().lower()
        if separator and normalized in {"pageouts", "swapouts"}:
            try:
                result[normalized] = int(value.strip().rstrip("."))
            except ValueError:
                continue
    if set(result) != {"pageouts", "swapouts"}:
        raise ValueError("vm_counters_unavailable")
    return result


def process_snapshot() -> dict[int, dict]:
    raw = subprocess.check_output(
        ["/bin/ps", "-A", "-o", "pid=,ppid=,lstart=,time=,comm="],
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        timeout=5,
    )
    result = parse_process_snapshot(raw)
    if not result:
        raise ValueError("process_snapshot_unavailable")
    return result


def gpu_clients() -> dict[int, dict]:
    raw = subprocess.check_output(
        ["/usr/sbin/ioreg", "-r", "-c", "IOGPUDeviceUserClient", "-a"],
        timeout=8,
    )
    devices = plistlib.loads(raw)
    if not isinstance(devices, list):
        raise TypeError("gpu_clients_unavailable")
    return parse_gpu_clients(devices)


def gpu_device_utilization() -> float | None:
    raw = subprocess.check_output(["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"], timeout=5)
    devices = plistlib.loads(raw)
    if not isinstance(devices, list):
        return None
    return parse_gpu_device_utilization(devices)


def parse_gpu_device_utilization(devices: list[dict]) -> float | None:
    values = [
        value
        for device in devices
        if isinstance(device, dict)
        for value in [
            device.get("PerformanceStatistics", {}).get("Device Utilization %")
            if isinstance(device.get("PerformanceStatistics"), dict)
            else None
        ]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return max(values) if values else None


def vm_counters() -> dict[str, int]:
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
    return parse_vm_counters(raw)


def thermal_state() -> int | None:
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        message = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        process = message(
            objc.objc_getClass(b"NSProcessInfo"),
            objc.sel_registerName(b"processInfo"),
        )
        scalar = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        return scalar(process, objc.sel_registerName(b"thermalState"))
    except Exception:  # noqa: BLE001 - absent frameworks are reported as unavailable.
        return None


def bucket_for(pid: int, name: str, worker_pids: set[int]) -> str:
    if pid in worker_pids:
        return "worker"
    lowered = name.lower()
    if "windowserver" in lowered:
        return "window_server"
    if any(marker in lowered for marker in MODEL_NAME_MARKERS):
        return "model_processes"
    if any(marker in lowered for marker in GRAPHICS_NAME_MARKERS):
        return "system_graphics"
    return "other"


def descendants(snapshot: dict[int, dict], roots: set[int]) -> set[int]:
    owned = set(roots)
    while True:
        added = {pid for pid, row in snapshot.items() if row["parent"] in owned and pid not in owned}
        if not added:
            return owned
        owned.update(added)


def cpu_interval(
    previous: dict[int, dict], current: dict[int, dict], elapsed: float, worker_pid: int
) -> tuple[dict[str, float], Counter]:
    totals = Counter({bucket: 0.0 for bucket in BUCKETS})
    discontinuities = Counter()
    worker_pids = descendants(current, {worker_pid} if worker_pid > 0 else set())
    harness_pids = {os.getpid()} | {
        pid for pid, row in current.items() if row["parent"] == os.getpid() and row["name"].lower() in {"ps", "ioreg"}
    }
    for pid, row in current.items():
        if pid in harness_pids:
            continue
        old = previous.get(pid)
        if old is None:
            discontinuities["cpu_process_appeared"] += 1
            continue
        if old["start"] != row["start"]:
            discontinuities["cpu_pid_identity_changed"] += 1
            continue
        delta = row["cpu_seconds"] - old["cpu_seconds"]
        if delta < 0:
            discontinuities["cpu_counter_reset"] += 1
            continue
        bucket = bucket_for(pid, row["name"], worker_pids)
        totals[bucket] += delta / max(0.001, elapsed) * 100
    disappeared = len(set(previous) - set(current))
    if disappeared:
        discontinuities["cpu_process_disappeared"] += disappeared
    return dict(totals), discontinuities


class HostMetricsSampler:
    """Sample host counters at 1 Hz and retain only fixed aggregate buckets."""

    def __init__(self, worker_pid: int | None):
        self.worker_pid = worker_pid or 0
        self.done = threading.Event()
        self.rows: list[dict] = []
        self.failures = Counter()
        self.discontinuities = Counter()
        self.source_availability = Counter()
        self.lock = threading.Lock()
        self.origin = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _safe(self, source: str, call):
        try:
            value = call()
        except Exception:  # noqa: BLE001 - never persist exceptions or host paths.
            with self.lock:
                self.failures[source] += 1
                self.source_availability[f"{source}:unavailable"] += 1
            return None
        with self.lock:
            self.source_availability[f"{source}:available"] += int(value is not None)
            self.source_availability[f"{source}:unavailable"] += int(value is None)
        return value

    def _run(self) -> None:
        try:
            self._sample_loop()
        except Exception:  # noqa: BLE001 - do not leak command or host data.
            with self.lock:
                self.failures["monitor_loop"] += 1

    def _sample_loop(self) -> None:
        previous_processes = self._safe("process_snapshot", process_snapshot)
        previous_gpu = self._safe("gpu_clients", gpu_clients)
        previous_time = time.monotonic()
        while True:
            finishing = self.done.wait(1.0)
            current_processes = self._safe("process_snapshot", process_snapshot)
            current_gpu = self._safe("gpu_clients", gpu_clients)
            current_time = time.monotonic()
            cpu_values = None
            cpu_discontinuities = Counter()
            if current_processes is not None and previous_processes:
                cpu_values, cpu_discontinuities = cpu_interval(
                    previous_processes,
                    current_processes,
                    current_time - previous_time,
                    self.worker_pid,
                )
            gpu_values, gpu_discontinuities = self._gpu_interval(
                previous_gpu, current_gpu, previous_processes, current_processes
            )
            self.discontinuities.update(cpu_discontinuities)
            for key, value in gpu_discontinuities.items():
                self.discontinuities[key] += value
            gpu_utilization = self._safe("gpu_device_utilization", gpu_device_utilization)
            current_thermal = self._safe("thermal_state", thermal_state)
            self.rows.append(
                {
                    "start": previous_time,
                    "end": current_time,
                    "interval_seconds": current_time - previous_time,
                    "cpu_percent": cpu_values,
                    "gpu_counter_delta_raw": gpu_values,
                    "counter_discontinuities": {
                        **dict(cpu_discontinuities),
                        **dict(gpu_discontinuities),
                    },
                    "source_availability": {
                        "cpu_delta": cpu_values is not None,
                        "gpu_counter_delta": gpu_values is not None,
                        "gpu_device_utilization": gpu_utilization is not None,
                        "thermal_state": current_thermal is not None,
                    },
                    "gpu_device_utilization_percent": gpu_utilization,
                    "thermal_state": current_thermal,
                }
            )
            previous_processes = current_processes
            previous_gpu = current_gpu
            previous_time = current_time
            if finishing:
                break

    def _gpu_interval(
        self,
        previous_gpu: dict[tuple[int, int | None], dict] | None,
        current_gpu: dict[tuple[int, int | None], dict] | None,
        previous_processes: dict[int, dict] | None,
        current_processes: dict[int, dict] | None,
    ) -> tuple[dict[str, int] | None, Counter]:
        changes = Counter()
        if previous_gpu is None or current_gpu is None or current_processes is None:
            return None, changes
        totals = Counter({bucket: 0 for bucket in BUCKETS})
        worker_pids = descendants(current_processes, {self.worker_pid} if self.worker_pid > 0 else set())
        for client_key, now in current_gpu.items():
            pid = now["pid"]
            if client_key[1] is None:
                changes["gpu_pid_aggregate_ambiguity"] += 1
            before = previous_gpu.get(client_key)
            if before is None:
                changes["gpu_client_appeared"] += 1
                continue
            old_proc = previous_processes.get(pid) if previous_processes else None
            new_proc = current_processes.get(pid)
            old_identity = (before["name"], old_proc["start"] if old_proc else None)
            new_identity = (now["name"], new_proc["start"] if new_proc else None)
            if old_identity != new_identity:
                changes["gpu_pid_identity_changed"] += 1
                continue
            delta = now["counter"] - before["counter"]
            if delta < 0:
                changes["gpu_counter_reset"] += 1
                continue
            name = new_proc["name"] if new_proc else now["name"]
            totals[bucket_for(pid, name, worker_pids)] += delta
        disappeared = len(set(previous_gpu) - set(current_gpu))
        if disappeared:
            changes["gpu_client_disappeared"] += disappeared
        return dict(totals), changes

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.done.set()
        self.thread.join(timeout=25)
        if self.thread.is_alive():
            self.failures["monitor_timeout"] += 1

    def begin_request(self) -> RequestMarker:
        marker = RequestMarker(
            started=0.0,
            vm_before=self._safe("vm_counters", vm_counters),
        )
        marker.started = time.monotonic()
        return marker

    def end_request(self, marker: RequestMarker) -> None:
        marker.finished = time.monotonic()
        marker.vm_after = self._safe("vm_counters", vm_counters)

    def request_result(self, marker: RequestMarker) -> dict:
        if marker.finished is None:
            raise ValueError("request_metrics_not_finished")
        overlapping = [row for row in self.rows if row["start"] < marker.finished and row["end"] > marker.started]
        full = [row for row in overlapping if row["start"] >= marker.started and row["end"] <= marker.finished]
        covered = sum(row["interval_seconds"] for row in full)
        max_interval = max((row["interval_seconds"] for row in full), default=None)
        cpu = _weighted_bucket_means(full, "cpu_percent", covered)
        gpu = _sum_gpu_deltas(full)
        utilization = [
            (row["gpu_device_utilization_percent"], row["interval_seconds"])
            for row in full
            if row["gpu_device_utilization_percent"] is not None
        ]
        thermal = Counter(str(row["thermal_state"]) for row in full if row["thermal_state"] is not None)
        vm_delta = {}
        discontinuities = Counter()
        for key in ("pageouts", "swapouts"):
            before = (marker.vm_before or {}).get(key)
            after = (marker.vm_after or {}).get(key)
            if before is None or after is None:
                vm_delta[key] = None
            elif after < before:
                vm_delta[key] = None
                self.discontinuities[f"vm_{key}_counter_reset"] += 1
                discontinuities[f"vm_{key}_counter_reset"] += 1
            else:
                vm_delta[key] = after - before
        availability = Counter()
        for row in full:
            discontinuities.update(row["counter_discontinuities"])
            for source, available in row["source_availability"].items():
                availability[source] += int(available)
        return {
            "sample_interval_count": len(full),
            "boundary_intervals_excluded": len(overlapping) - len(full),
            "covered_seconds": covered,
            "max_sample_interval_seconds": max_interval,
            "sample_intervals_over_two_seconds": sum(row["interval_seconds"] > 2 for row in full),
            "cpu_percent_mean_by_bucket": cpu,
            "gpu_counter_delta_raw_by_bucket": gpu,
            "gpu_counter_unit": GPU_UNIT,
            "gpu_counter_rate_raw_per_covered_second_by_bucket": _rates(gpu, covered),
            "gpu_device_utilization_percent_mean": _weighted_mean(utilization),
            "thermal_state_counts": dict(thermal) or None,
            "vm_counter_deltas": vm_delta,
            "vm_counters_available": marker.vm_before is not None and marker.vm_after is not None,
            "source_available_interval_count": dict(availability),
            "counter_discontinuities": dict(discontinuities),
        }

    def summary(self) -> dict:
        return {
            "schema_version": 1,
            "sampling_interval_seconds": 1.0,
            "gpu_counter_unit": GPU_UNIT,
            "process_cpu_source": "ps cumulative process CPU time; host resolution dependent",
            "gpu_source": "IORegistry IOGPUDeviceUserClient AppUsage Metal counter",
            "gpu_client_identity": "IORegistryEntryID when available; per-PID aggregate fallback otherwise",
            "gpu_pid_aggregate_ambiguity_note": "Missing entry IDs can hide client changes within a PID.",
            "gpu_device_utilization_source": "IORegistry AGXAccelerator PerformanceStatistics Device Utilization %",
            "gpu_device_utilization_note": "Reported utilization does not measure GPU clocks or residency.",
            "vm_source": "vm_stat Pageouts and Swapouts counters",
            "thermal_state_labels": {
                "0": "nominal",
                "1": "fair",
                "2": "serious",
                "3": "critical",
            },
            "gpu_counter_unit_verified": False,
            "process_names_and_pids_persisted": False,
            "sample_count": len(self.rows),
            "source_failures": dict(self.failures),
            "source_read_counts": dict(self.source_availability),
            "counter_discontinuities": dict(self.discontinuities),
        }


def _weighted_bucket_means(rows: list[dict], key: str, covered: float) -> dict | None:
    if not rows or covered <= 0 or any(row[key] is None for row in rows):
        return None
    return {bucket: sum(row[key][bucket] * row["interval_seconds"] for row in rows) / covered for bucket in BUCKETS}


def _sum_gpu_deltas(rows: list[dict]) -> dict | None:
    if not rows or any(row["gpu_counter_delta_raw"] is None for row in rows):
        return None
    totals = Counter({bucket: 0 for bucket in BUCKETS})
    for row in rows:
        totals.update(row["gpu_counter_delta_raw"])
    return dict(totals)


def _rates(values: dict | None, covered: float) -> dict | None:
    if values is None or covered <= 0:
        return None
    return {bucket: values[bucket] / covered for bucket in BUCKETS}


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    covered = sum(duration for _, duration in values)
    if not values or covered <= 0:
        return None
    return sum(value * duration for value, duration in values) / covered
