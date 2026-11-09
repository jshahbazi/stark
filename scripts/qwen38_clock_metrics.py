"""Aggregate clock telemetry from a source-timestamped private macmon build."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
import time
from pathlib import Path


def summarize(rows, start, end):
    """Join conservative source brackets entirely within a request window."""
    selected = [r for r in rows if r["start"] >= start and r["end"] <= end]
    duration = sum(r["duration"] for r in selected)
    active = sum(r["duration"] * r["gpu_active_ratio"] for r in selected)

    def mean(key):
        values = [(r.get(key), r["duration"]) for r in selected]
        values = [(v, w) for v, w in values if isinstance(v, (float, int)) and math.isfinite(v)]
        return sum(v * w for v, w in values) / sum(w for _, w in values) if values else None

    return {
        "fully_contained_intervals": len(selected),
        "covered_seconds": duration,
        "mean_active_gpu_mhz": sum(r["gpu_freq_mhz"] * r["duration"] * r["gpu_active_ratio"] for r in selected) / active if active else None,
        "gpu_active_ratio": active / duration if duration else None,
        "gpu_temperature_c_mean": mean("gpu_temperature_c"),
        "gpu_temperature_c_max": max((r["gpu_temperature_c"] for r in selected if r.get("gpu_temperature_c") is not None), default=None),
        "gpu_power_watts_mean": mean("gpu_power_watts"),
        "cpu_power_watts_mean": mean("cpu_power_watts"),
        "top_cpu_tier_mhz_mean": mean("top_cpu_tier_mhz"),
        "fan_rpm_mean": mean("fan_rpm_mean"),
        "boundary_uncertainty_seconds_max": max((r["boundary_uncertainty"] for r in selected), default=None),
    }


class ClockSampler:
    def __init__(self, binary: Path, interval_ms=500):
        self.binary = binary.resolve()
        self.binary_sha256 = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.interval_ms = interval_ms
        self.rows = []
        self.failures = 0
        self.soc = None
        self.lock = threading.Lock()
        self.process = None
        self.thread = None

    def start(self):
        self.process = subprocess.Popen(
            [str(self.binary), "pipe", "--interval", str(self.interval_ms), "--soc-info"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    doc = json.loads(line)
                    start, end = doc["sample_start_unix_seconds"], doc["sample_end_unix_seconds"]
                    duration = doc["sample_duration_seconds"]
                    if not (0 < start < end and duration > 0 and all(math.isfinite(x) for x in (start, end, duration))):
                        raise ValueError("source_interval_unavailable")
                    fans = [f["rpm"] for f in doc.get("fans", [])]
                    tiers = doc.get("cpu_tiers", [])
                    row = {
                        "start": start, "end": end, "duration": duration,
                        "boundary_uncertainty": doc["sample_boundary_uncertainty_seconds"],
                        "observed_unix_seconds": time.time(),
                        "gpu_freq_mhz": doc["gpu_freq_mhz"],
                        "gpu_active_ratio": doc["gpu_active_ratio"],
                        "gpu_scaled_ratio": doc["gpu_scaled_ratio"],
                        "gpu_temperature_c": doc.get("temp", {}).get("gpu_temp_avg"),
                        "cpu_temperature_c": doc.get("temp", {}).get("cpu_temp_avg"),
                        "gpu_power_watts": doc.get("gpu_power"),
                        "cpu_power_watts": doc.get("cpu_power"),
                        "system_power_watts": doc.get("sys_power"),
                        "top_cpu_tier_mhz": tiers[-1].get("freq_mhz") if tiers else None,
                        "fan_rpm_mean": sum(fans) / len(fans) if fans else None,
                        "ram_usage_bytes": doc.get("memory", {}).get("ram_usage"),
                        "swap_usage_bytes": doc.get("memory", {}).get("swap_usage"),
                    }
                    with self.lock:
                        self.rows.append(row)
                        if self.soc is None:
                            soc = doc.get("soc", {})
                            self.soc = {k: soc.get(k) for k in ("chip_name", "mac_model", "gpu_freqs", "gpu_cores", "memory_gb", "vm")}
                except (ValueError, KeyError, TypeError):
                    with self.lock:
                        self.failures += 1
        except Exception:
            with self.lock:
                self.failures += 1

    def snapshot(self):
        with self.lock:
            return list(self.rows)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.thread is not None:
            self.thread.join(timeout=5)

    def report(self):
        with self.lock:
            return {
                "binary_sha256": self.binary_sha256,
                "interval_ms": self.interval_ms,
                "source": "Pinned macmon 0.10.0 with source-read bracket patch; private IOReport and SMC APIs.",
                "frequency_policy": "Residency-weighted active MHz; every raw source interval retained. Request joins require full containment of conservative source-read brackets.",
                "powermetrics_cross_checked": False,
                "sample_count": len(self.rows), "failures": self.failures,
                "soc": self.soc, "samples": list(self.rows),
            }
