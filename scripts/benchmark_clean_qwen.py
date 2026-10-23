"""Attributed, quiet-window Qwen replay; retain only scalars and hashes."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import plistlib
import signal
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmark_stock_engines import (
    Engine,
    generation_fixtures,
    public_spec,
    save,
    sha,
    sleep_events,
    stop,
    token_sha,
)


def processes():
    raw = subprocess.check_output(
        ["/bin/ps", "-A", "-o", "pid=,ppid=,lstart=,time=,comm="],
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        timeout=5,
    )
    result = {}
    for line in raw.splitlines():
        fields = line.split(None, 8)
        if len(fields) != 9:
            continue
        pid, parent = map(int, fields[:2])
        cpu = fields[7]
        days, cpu = cpu.split("-", 1) if "-" in cpu else ("0", cpu)
        seconds = int(days) * 86400 + sum(
            float(x) * 60**i for i, x in enumerate(reversed(cpu.split(":")))
        )
        result[pid] = {
            "parent": parent,
            "start": " ".join(fields[2:7]),
            "cpu": seconds,
            "name": Path(fields[8]).name,
        }
    return result


def cpu_deltas(previous, current, elapsed, root):
    owned = {root}
    while True:
        expanded = owned | {
            pid for pid, row in current.items() if row["parent"] in owned
        }
        if expanded == owned:
            break
        owned = expanded
    values = []
    for pid, row in current.items():
        old = previous.get(pid)
        if pid in owned:
            continue
        prior_cpu = old["cpu"] if old and row["start"] == old["start"] else 0
        load = max(0, row["cpu"] - prior_cpu) / max(0.001, elapsed) * 100
        values.append({"pid": pid, "name": row["name"], "cpu_percent": load})
    return sorted(values, key=lambda r: r["cpu_percent"], reverse=True)


def gpu_percent():
    devices = plistlib.loads(
        subprocess.check_output(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"], timeout=5
        )
    )
    values = [
        d["PerformanceStatistics"]["Device Utilization %"]
        for d in devices
        if "Device Utilization %" in d.get("PerformanceStatistics", {})
    ]
    if not values:
        raise RuntimeError("gpu_monitor_unavailable")
    return max(values)


def thermal_state():
    # Public NSProcessInfo.thermalState, read without a compiler or extra package.
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    message = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    obj = message(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    scalar = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    return scalar(obj, objc.sel_registerName(b"thermalState"))


def paging():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
    values = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in {"Pageouts", "Swapouts"}:
                values[key.lower()] = int(value.strip().rstrip("."))
    if len(values) != 2:
        raise RuntimeError("paging_monitor_unavailable")
    return values


def model_process(name):
    return any(
        x in name.lower() for x in ("llama", "stark", "tensorfold", "vllm", "mlx")
    )


class Monitor:
    def __init__(self):
        self.phase = "load"
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.rows = []
        self.failures = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def run(self):
        try:
            previous = processes()
            prior_time = time.monotonic()
            prior_phase = self.phase
            while True:
                finishing = self.done.wait(0.5)
                current = processes()
                now = time.monotonic()
                with self.lock:
                    phase = self.phase
                values = cpu_deltas(previous, current, now - prior_time, os.getpid())
                self.rows.append(
                    {
                        "start_seconds": prior_time - self.started,
                        "end_seconds": now - self.started,
                        "interval_seconds": now - prior_time,
                        "phases": sorted({prior_phase, phase}),
                        "external_cpu_percent": sum(v["cpu_percent"] for v in values),
                        "max_process_cpu_percent": values[0]["cpu_percent"]
                        if values
                        else 0,
                        "max_model_cpu_percent": max(
                            (
                                v["cpu_percent"]
                                for v in values
                                if model_process(v["name"])
                            ),
                            default=0,
                        ),
                        "active_processes": [
                            v for v in values if v["cpu_percent"] >= 5
                        ],
                        "gpu_device_utilization_percent": gpu_percent(),
                        "thermal_state": thermal_state(),
                    }
                )
                previous, prior_time, prior_phase = current, now, phase
                if finishing:
                    break
        except Exception:  # noqa: BLE001 - never persist exception payloads.
            self.failures += 1

    def __enter__(self):
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            self.failures += 1

    def result(self):
        measured = [r for r in self.rows if "measurement" in r["phases"]]
        return {
            "samples": self.rows,
            "sampling_failures": self.failures,
            "measurement_samples": len(measured),
            "sampling_gap_exceeded": any(r["interval_seconds"] > 2 for r in measured),
            "cpu_guard_exceeded": any(
                r["max_process_cpu_percent"] > 100 or r["external_cpu_percent"] > 200
                for r in measured
            ),
            "background_model_active": any(
                r["max_model_cpu_percent"] > 10 for r in measured
            ),
            "thermal_guard_exceeded": any(r["thermal_state"] != 0 for r in measured),
        }


def quiet_window(monitor, seconds=10, timeout=120):
    cursor = len(monitor.rows)
    monitor.set_phase("quiet")
    started = time.monotonic()
    streak = None
    while time.monotonic() - started < timeout:
        time.sleep(0.5)
        if monitor.failures:
            raise RuntimeError("host_monitor_failed")
        rows = list(monitor.rows)
        for last in rows[cursor:]:
            quiet = (
                last["phases"] == ["quiet"]
                and last["interval_seconds"] <= 2
                and last["max_process_cpu_percent"] <= 80
                and last["external_cpu_percent"] <= 150
                and last["max_model_cpu_percent"] <= 10
                and last["gpu_device_utilization_percent"] <= 5
                and last["thermal_state"] == 0
            )
            if not quiet:
                streak = None
            elif streak is None:
                streak = last["start_seconds"]
            if streak is not None and last["end_seconds"] - streak >= seconds:
                return time.monotonic() - started
        cursor = len(rows)
    raise RuntimeError("quiet_window_timeout")


def accept_timing(report):
    return (
        report["protocol"].get("require_quiet_windows", True)
        and report["power_log_screen"]["checked"]
        and report["power_log_screen"]["sleep_events"] == 0
        and abs(report["sleep_clock_discrepancy_seconds"]) < 2
        and all(b["status"] == "complete" for b in report["blocks"])
    )


def run(config, output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["source"], local_files_only=True)
    fixtures = generation_fixtures(tokenizer)
    specs = config["engines"]
    report = {
        "schema": 1,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "stock_harness_sha256": sha(
            Path(__file__).with_name("benchmark_stock_engines.py")
        ),
        "worker_sha256": sha(Path(__file__).with_name("stock_compare_mlx.py")),
        "model": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "model_revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3",
        "protocol": {
            **{k: config[k] for k in ("context", "blocks", "repeats")},
            "require_quiet_windows": config.get("require_quiet_windows", True),
        },
        "engines": [public_spec(s) for s in specs],
        "fixtures": [
            {
                "name": f["name"],
                "input_tokens": len(f["ids"]),
                "output_limit": f["limit"],
                "input_token_sha256": token_sha(f["ids"]),
                "population": f["population"],
            }
            for f in fixtures
        ],
        "blocks": [],
        "failures": [],
    }
    save(output, report)
    start_wall, start_mono = time.time(), time.monotonic()
    with contextlib.ExitStack() as outer:
        assertion = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        outer.callback(stop, assertion)
        for block in range(config["blocks"]):
            rotation = (block // 2) % len(specs)
            order = specs[rotation:] + specs[:rotation]
            if block % 2:
                order = list(reversed(order))
            for spec in order:
                row = {
                    "block": block,
                    "engine": spec["id"],
                    "status": "running",
                    "samples": [],
                }
                report["blocks"].append(row)
                monitor = Monitor()
                try:
                    with monitor:
                        row["preload_quiet_seconds"] = (
                            quiet_window(monitor)
                            if report["protocol"]["require_quiet_windows"]
                            else None
                        )
                        monitor.set_phase("load")
                        with contextlib.ExitStack() as stack:
                            engine = Engine(
                                spec, stack, config["task_root"], config["context"]
                            )
                            row["load_seconds"] = engine.load_seconds
                            row["runtime"] = engine.metadata
                            monitor.set_phase("warmup")
                            for fixture in fixtures:
                                checked = engine.tokenize(fixture["rendered"])
                                if checked is not None and checked != fixture["ids"]:
                                    raise RuntimeError("tokenizer_mismatch")
                                engine.generate(fixture, config["context"])
                            row["postwarm_quiet_seconds"] = (
                                quiet_window(monitor)
                                if report["protocol"]["require_quiet_windows"]
                                else None
                            )
                            before_paging = paging()
                            monitor.set_phase("measurement")
                            row["measurement_start_seconds"] = (
                                time.monotonic() - monitor.started
                            )
                            for fixture in fixtures:
                                count = (
                                    config["repeats"]
                                    if fixture["population"] == "performance"
                                    else 1
                                )
                                for repeat in range(count):
                                    result = engine.generate(fixture, config["context"])
                                    row["samples"].append(
                                        {
                                            "fixture": fixture["name"],
                                            "repeat": repeat,
                                            **result,
                                        }
                                    )
                            row["measurement_end_seconds"] = (
                                time.monotonic() - monitor.started
                            )
                            monitor.set_phase("teardown")
                            after_paging = paging()
                            row["paging_delta"] = {
                                k: after_paging[k] - v for k, v in before_paging.items()
                            }
                    row["status"] = "complete"
                except Exception as exc:  # noqa: BLE001 - private worker payloads stay private.
                    safe = (
                        str(exc)
                        if isinstance(exc, RuntimeError)
                        and str(exc).replace("_", "").isalnum()
                        else "engine_operation_failed"
                    )
                    row.update(status="failed", error=safe)
                    report["failures"].append(
                        {"engine": spec["id"], "block": block, "error": safe}
                    )
                row["host_screen"] = monitor.result()
                if row["status"] == "complete" and (
                    any(
                        row["host_screen"][k]
                        for k in (
                            "sampling_failures",
                            "sampling_gap_exceeded",
                            "cpu_guard_exceeded",
                            "background_model_active",
                            "thermal_guard_exceeded",
                        )
                    )
                    or not row["host_screen"]["measurement_samples"]
                    or any(row["paging_delta"].values())
                    or any(
                        not s["fixed_work_eligible"]
                        for s in row["samples"]
                        if next(
                            f["population"]
                            for f in fixtures
                            if f["name"] == s["fixture"]
                        )
                        == "performance"
                    )
                ):
                    row["status"] = "excluded_interference"
                save(output, report)
                print(
                    json.dumps(
                        {
                            "engine": spec["id"],
                            "block": block,
                            "status": row["status"],
                            "samples": len(row["samples"]),
                        }
                    ),
                    flush=True,
                )
    report["ended_utc"] = datetime.now(UTC).isoformat()
    report["wall_elapsed_seconds"] = time.time() - start_wall
    report["monotonic_elapsed_seconds"] = time.monotonic() - start_mono
    report["sleep_clock_discrepancy_seconds"] = (
        report["wall_elapsed_seconds"] - report["monotonic_elapsed_seconds"]
    )
    report["power_log_screen"] = sleep_events(start_wall, time.time())
    report["timing_accepted"] = accept_timing(report)
    report["status"] = "complete_with_failures" if report["failures"] else "complete"
    save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        run(json.loads(args.config.read_text()), args.output)
    except KeyboardInterrupt:
        if args.output.exists():
            report = json.loads(args.output.read_text())
            report.update(status="interrupted", timing_accepted=False)
            report["ended_utc"] = datetime.now(UTC).isoformat()
            save(args.output, report)
        raise SystemExit("clean_benchmark_interrupted") from None
    except Exception:  # noqa: BLE001 - never print private worker failures.
        raise SystemExit("clean_benchmark_failed") from None


if __name__ == "__main__":
    main()
