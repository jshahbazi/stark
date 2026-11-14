"""Measure one exact guided artifact; retain only aggregates and hashes.

Requests, generated text, token arrays and process identities stay in memory.
Run one worker at a time. Existing native request admission remains enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import signal
import stat
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path

import psutil
from qwen38_process_metrics import snapshot
from stark_convert.chat import Worker, load_template
from stark_convert.guided_models import (
    matches_recipe,
    native_recipes,
    with_resident_profile,
)

GIB = 1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(model):
    rows = {}
    if model.is_symlink():
        raise ValueError("linked_model")
    for path in sorted(model.iterdir()):
        if path.suffix in {".json", ".safetensors", ".jinja", ".bin"}:
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("nonregular_model_file")
            row = {
                "bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "ctime_ns": st.st_ctime_ns,
                "inode": st.st_ino,
                "links": st.st_nlink,
            }
            if path.suffix not in {".safetensors", ".bin"}:
                row["sha256"] = sha(path)
            rows[path.name] = row
    encoded = json.dumps(rows, sort_keys=True).encode()
    return rows, hashlib.sha256(encoded).hexdigest()


class Monitor:
    def __init__(self):
        self.pid = None
        self.phase = "load"
        self.phases = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.guard = None
        self.baseline_swap = psutil.swap_memory().used
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def gpu(self, _worker, amount):
        if amount is None:
            return
        with self.lock:
            row = self.phases.setdefault(self.phase, {})
            row["peak_metal_allocated_bytes"] = max(
                row.get("peak_metal_allocated_bytes", 0), amount
            )

    def run(self):
        while not self.stop.wait(0.1):
            if self.pid is None:
                continue
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            usage = snapshot(self.pid)
            values = usage.get("cumulative", {})
            with self.lock:
                row = self.phases.setdefault(self.phase, {})
                row["samples"] = row.get("samples", 0) + 1
                row["process_memory_samples"] = row.get(
                    "process_memory_samples", 0
                ) + int(usage.get("available") is True)
                row["minimum_available_bytes"] = min(
                    row.get("minimum_available_bytes", memory.available),
                    memory.available,
                )
                row["maximum_swap_used_bytes"] = max(
                    row.get("maximum_swap_used_bytes", swap.used), swap.used
                )
                for key in (
                    "resident_size",
                    "phys_footprint",
                    "lifetime_max_phys_footprint",
                ):
                    if key in values:
                        row["peak_" + key + "_bytes"] = max(
                            row.get("peak_" + key + "_bytes", 0), values[key]
                        )
                reason = (
                    "available_memory_floor"
                    if memory.available < 6 * GIB
                    else "swap_growth"
                    if swap.used > self.baseline_swap + GIB
                    else "worker_footprint_ceiling"
                    if values.get("phys_footprint", 0) > 96 * GIB
                    else None
                )
                if reason and self.guard is None:
                    self.guard = reason
                    try:
                        os.killpg(self.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


def atomic_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def request(worker, payload):
    """Read terminal errors without the interactive client's error translation."""
    worker.request_sequence += 1
    data = (
        json.dumps({**payload, "request_id": worker.request_sequence}).encode() + b"\n"
    )
    if len(data) > worker.max_request_bytes:
        raise ValueError("wire_limit")
    view = memoryview(data)
    while view:
        count = worker.process.stdin.write(view)
        if not count:
            raise ValueError("worker_write_failed")
        view = view[count:]
    while True:
        reply = worker.read()
        if "event" not in reply:
            return reply


def run(args):
    if args.output.exists():
        raise ValueError("output_exists")
    recipe = next(r for r in native_recipes() if r.native_choice == args.choice)
    if not matches_recipe(args.model, recipe):
        raise ValueError("artifact_mismatch")
    recipe = with_resident_profile(recipe, args.model)
    # Qualification must begin inside the same available-memory envelope that
    # Beginner startup requires. Compilation/other work can release it slowly.
    stable_since = None
    deadline = time.monotonic() + 300
    announced = False
    while True:
        now = time.monotonic()
        available = psutil.virtual_memory().available
        if available >= recipe.runtime_memory:
            stable_since = stable_since or now
            if now - stable_since >= 5:
                break
        else:
            stable_since = None
        if not announced:
            print(
                json.dumps(
                    {
                        "phase": "waiting_for_startup_headroom",
                        "required_bytes": recipe.runtime_memory,
                    }
                ),
                flush=True,
            )
            announced = True
        if now >= deadline:
            raise ValueError("startup_headroom_unavailable")
        time.sleep(0.5)
    if any(p.info["name"] == "stark-generate" for p in psutil.process_iter(["name"])):
        raise ValueError("concurrent_worker")
    # Remove diagnostic/capture selectors; report no environment contents.
    for key in tuple(os.environ):
        if key.startswith(
            ("STARK_CAPTURE_", "UZU_CAPTURE_", "STARK_QWEN38_", "STARK_NATIVE_PROFILE_")
        ):
            os.environ.pop(key)
    before, seal = identity(args.model)
    monitor = Monitor()
    report = {
        "schema": "stark-guided-memory-observation-v1",
        "passed": False,
        "choice": args.choice,
        "repo": recipe.repo,
        "revision": recipe.revision,
        "worker_sha256": sha(args.worker),
        "harness_sha256": sha(Path(__file__)),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "policy_files_sha256": {
            name: sha(Path(name))
            for name in (
                "tools/converter/stark_convert/guided_memory_profiles.json",
                "crates/stark-generate/src/hybrid.rs",
                "crates/stark-engine/src/engine/hybrid/guided_admission.rs",
                "vendor/lily-core/src/qwen4exp/request_memory.rs",
            )
        },
        "artifact_metadata_stat_sha256": seal,
        "files": before,
        "payload_verification": "Pinned source/metadata/layout match plus before/after file stats; payloads not freshly rehashed.",
        "hardware": {
            "os": platform.mac_ver()[0],
            "machine": platform.machine(),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "baseline_available_bytes": psutil.virtual_memory().available,
        "planned_startup_bytes": recipe.runtime_memory,
        "qualified_resident_plan": recipe.resident_profile is not None,
        "baseline_swap_used_bytes": monitor.baseline_swap,
        "context_tokens": args.context,
        "output_budget": args.tokens,
        "sampling": "greedy",
        "seed": 42,
        "purpose": "memory_observation_only",
        "requests": [],
    }
    worker = None
    phase = "load"
    started = time.monotonic()
    monitor.thread.start()
    try:
        worker = Worker(args.worker, args.model, 900, gpu_memory_callback=monitor.gpu)
        monitor.pid = worker.process.pid
        ready = worker.read()
        if ready.get("ready") is not True:
            raise ValueError("worker_not_ready")
        report["load_seconds"] = time.monotonic() - started
        report["capabilities"] = {
            "prefix_cache_enabled": worker.prefix_cache,
            "prefix_chunk_tokens": worker.prefix_chunk_tokens,
            "recommended_mtp_drafts": ready.get("mtp", {}).get("recommended_drafts", 0),
        }
        print(
            json.dumps({"phase": "ready", "seconds": round(report["load_seconds"], 2)}),
            flush=True,
        )
        monitor.set_phase("ready")
        time.sleep(2)
        codec, template = load_template(args.model)
        # Synthetic code-like input provides varied n-gram rows without user data.
        body = "\n".join(
            f"def task_{i}(value): return value + {i}  # review item {i % 97}"
            for i in range(12000)
        )
        rendered = template.render(
            messages=[{"role": "user", "content": body}],
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token=codec.get("bos_token") or "",
            eos_token=codec.get("eos_token") or "",
        )
        full = worker.request(
            {"operation": "tokenize", "text": rendered, "add_special_tokens": False}
        )["token_ids"]
        scope = secrets.token_hex(16)
        recommended = ready.get("mtp", {}).get("recommended_drafts", 0)
        modes = [recommended] + ([0] if recommended else [])
        for drafts in modes:
            counts = (
                [1024, 8192, 16384, args.context - args.tokens - 32]
                if drafts == recommended
                else [args.context - args.tokens - 32]
            )
            counts = [n for n in counts if n + args.tokens <= args.context]
            reference_hash = None
            for index, count in enumerate([*counts, counts[-1], counts[-1]]):
                fresh = index == len(counts) + 1
                phase = f"mtp{drafts}_input{count}_request{index}"
                monitor.set_phase(phase)
                ids = full[: count - 64] + full[-64:]
                if len(ids) != count:
                    raise ValueError("input_size_mismatch")
                begin = time.monotonic()
                reply = request(
                    worker,
                    {
                        "operation": "generate",
                        "input_token_ids": ids,
                        "context_tokens": args.context,
                        "max_new_tokens": args.tokens,
                        "sampling": {"mode": "greedy"},
                        "mtp_drafts": drafts,
                        "seed": 42,
                        "cache_scope": secrets.token_hex(16) if fresh else scope,
                    },
                )
                error = reply.get("error")
                if error is not None or reply.get("sampled_tokens", 0) < 1:
                    report["request_failure"] = (
                        error
                        if error
                        in {"resource_budget_exceeded", "native_generation_failed"}
                        else "request_failed"
                    )
                    raise ValueError("request_failed")
                output = reply.get("generated_token_ids", [])
                if (
                    not output
                    or reply.get("input_tokens") != count
                    or reply.get("mtp", {}).get("drafts") != drafts
                ):
                    raise ValueError("reply_contract_mismatch")
                item = {
                    "input_tokens": count,
                    "mtp_drafts": drafts,
                    "sampled_tokens": reply["sampled_tokens"],
                    "finish_reason": reply.get("finish_reason"),
                    "seconds": time.monotonic() - begin,
                    "reused_input_tokens": reply.get("reused_input_tokens", 0),
                    "output_sha256": hashlib.sha256(
                        json.dumps(output).encode()
                    ).hexdigest(),
                }
                item["fresh_scope"] = fresh
                if count == counts[-1]:
                    if (
                        reference_hash is not None
                        and item["output_sha256"] != reference_hash
                    ):
                        raise ValueError("prefix_output_mismatch")
                    reference_hash = item["output_sha256"]
                if fresh and item["reused_input_tokens"] != 0:
                    raise ValueError("fresh_scope_reused")
                if (
                    index == len(counts)
                    and worker.prefix_cache
                    and item["reused_input_tokens"] <= 0
                ):
                    raise ValueError("repeat_prefix_not_reused")
                report["requests"].append(item)
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "input_tokens": count,
                            "sampled_tokens": item["sampled_tokens"],
                            "reused_input_tokens": item["reused_input_tokens"],
                            "seconds": round(item["seconds"], 2),
                        }
                    ),
                    flush=True,
                )
                with monitor.lock:
                    report["memory_phases"] = deepcopy(monitor.phases)
                atomic_report(args.output, report)
        report["passed"] = monitor.guard is None
    except BaseException as error:  # noqa: BLE001 - never expose request or native exception contents.
        report["failure_stage"] = phase
        allowed = {
            "wire_limit",
            "worker_write_failed",
            "worker_not_ready",
            "input_size_mismatch",
            "request_failed",
            "reply_contract_mismatch",
            "prefix_output_mismatch",
            "fresh_scope_reused",
            "repeat_prefix_not_reused",
        }
        report["failure_code"] = (
            error.args[0]
            if isinstance(error, ValueError) and error.args and error.args[0] in allowed
            else "observation_failed"
        )
        print(json.dumps({"phase": phase, "status": "failed"}), flush=True)
    finally:
        if worker is not None:
            worker.close()
        monitor.stop.set()
        monitor.thread.join(timeout=3)
        report["memory_phases"] = monitor.phases
        report["memory_measurement_available"] = any(
            row.get("process_memory_samples", 0) > 0
            and row.get("peak_metal_allocated_bytes", 0) > 0
            for row in monitor.phases.values()
        )
        report["guard"] = monitor.guard
        report["elapsed_seconds"] = time.monotonic() - started
        report["artifact_unchanged"] = identity(args.model)[1] == seal
        report["worker_unchanged"] = sha(args.worker) == report["worker_sha256"]
        report["passed"] = (
            report["passed"]
            and report["artifact_unchanged"]
            and report["worker_unchanged"]
            and report["memory_measurement_available"]
        )
        atomic_report(args.output, report)
    print(
        json.dumps(
            {
                "status": "passed" if report["passed"] else "failed",
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--choice",
        choices=("qwen38-lily-q4-xl", "qwen38-mlx-serve-mixed"),
        required=True,
    )
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except BaseException:  # noqa: BLE001 - content-free failure; inputs and native exceptions stay private.
        print("guided_memory_observation_failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
