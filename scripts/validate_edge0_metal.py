"""Matched, private-pipe Edge0 reference/candidate readouts and forward timing.

The identical probe must be built separately against each frozen engine source.
Registration is a separate, model-free action required before execution. Forced
ordinary continuation IDs bypass EOS only for fixed-work numerical/timing tests;
these timings do not describe free-running generation quality or stop behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import qwen38_process_metrics as process_metrics
from validate_edge0_reference import (
    RECEIPT,
    RSS_LIMIT,
    Failure,
    digest,
    require,
    write_new,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_ABS = 0.002
REL_L2 = 0.0002
PREFIX_LENGTHS = (32, 128)
DECODE_STEPS = 16
WARMUPS = 1
REPEATS = 3
TIMEOUT = 1200
TEXT = (
    "The small workshop has a wooden table beside the window. "
    "Each morning the keeper sorts the blue and green cards into neat piles. "
    "A paper notebook records the number of items on every shelf. "
) * 32


def progress(arm, stage, **counts):
    """Fixed stage names and aggregate counters only; never model payloads."""
    print(
        json.dumps({"arm": arm, "stage": stage, **counts}, sort_keys=True), flush=True
    )


def selectors(values):
    result = {"STARK_EDGE0_REFERENCE": "1"}
    for item in values:
        key, separator, value = item.partition("=")
        require(
            bool(separator)
            and key.startswith("STARK_EDGE0_")
            and key != "STARK_EDGE0_REFERENCE"
            and value in {"0", "1"},
            "invalid_candidate_selector",
        )
        require(key not in result, "duplicate_candidate_selector")
        result[key] = value
    return result


def bindings(args):
    require(digest(args.model / "native-source.json") == RECEIPT, "source_receipt")
    files = {
        "baseline_probe": args.baseline_probe,
        "candidate_probe": args.candidate_probe,
        "qualifier": Path(__file__),
        "probe_source": ROOT / "crates/stark-engine/examples/edge0_forward_probe.rs",
        "process_metrics": ROOT / "scripts/qwen38_process_metrics.py",
        "reference_helpers": ROOT / "scripts/validate_edge0_reference.py",
        "tokenizer": args.model / "tokenizer.json",
        "config": args.model / "config.json",
    }
    return {name: digest(path) for name, path in files.items()}


def recipe(args):
    return {
        "schema": 1,
        "attempt": "20261112-EDGE0-06",
        "scope": "local_noncommercial_teacher_forced_forward_only",
        "files_sha256": bindings(args),
        "source_receipt_sha256": RECEIPT,
        "large_payloads_rehashed": False,
        "baseline_selectors": {"STARK_EDGE0_REFERENCE": "1"},
        "candidate_selectors": selectors(args.candidate_env),
        "input_text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
        "input_recipe": "ordinary-text-repeat-v1; encode without specials; prefix and following IDs",
        "context_tokens": 512,
        "prefix_lengths": list(PREFIX_LENGTHS),
        "decode_steps": DECODE_STEPS,
        "warmups": WARMUPS,
        "timed_repeats": REPEATS,
        "forward_advances_per_backend": (WARMUPS + REPEATS)
        * sum(length + DECODE_STEPS for length in PREFIX_LENGTHS)
        + PREFIX_LENGTHS[0]
        + DECODE_STEPS,
        "full_readout_rows_per_backend": len(PREFIX_LENGTHS)
        * (WARMUPS + REPEATS)
        * (1 + DECODE_STEPS)
        + 1
        + DECODE_STEPS,
        "retained_comparison_rows": len(PREFIX_LENGTHS) * (1 + DECODE_STEPS),
        "sampling": "none; teacher-forced fixed continuation; EOS does not stop",
        "execution_gate": {
            "baseline_backend": "edge0-f32-reference",
            "candidate_backend": "edge0-metal-graph-v1",
            "baseline_source_binding": "0721afc5",
            "baseline_prefill_max_batch": 1,
            "candidate_prefill_max_batch": 8,
            "activation_precision": "f32",
            "accumulation_precision": "f32",
            "recurrent_state_precision": "f32",
            "attention_cache_precision": "f32",
            "prefix_reuse": False,
            "registered_bank_bytes_scope": "retained source-owner bank registration/virtual span bytes; not physical residency or total mapped weights; baseline Phase4 registry absent",
            "candidate_required_counter_increases": [
                "batched_matrix_calls",
                "batched_q8_calls",
                "batched_expert_calls",
            ],
            "counter_scope": "actual dispatches with more than one input token",
            "graph_scratch_high_water_bytes_gate": "candidate final: greater than zero and at most 67108864 bytes",
        },
        "max_abs_gate": MAX_ABS,
        "relative_l2_gate": REL_L2,
        "top1_gate": "exact when reference margin exceeds twice max_abs_gate",
        "rss_limit_bytes": RSS_LIMIT,
        "minimum_available_bytes": args.minimum_available_gib * 1024**3,
        "request_timeout_seconds": TIMEOUT,
        "watchdog": "owned-worker RSS sampled while awaiting replies, at most 100ms select interval",
        "timing_scope": "engine-only synchronous ingest_prefix, initial step, forced decode steps; excludes IPC",
        "order": "baseline session reaped; candidate session reaped; each session includes reset and request rejection recovery",
        "inherited_stark_uzu_selectors_removed": True,
    }


def row(value, vocab):
    require(isinstance(value, list) and len(value) == vocab, "readout_shape")
    require(all(type(item) in {int, float} for item in value), "readout_type")
    result = np.asarray(value, dtype=np.float64)
    require(bool(np.isfinite(result).all()), "readout_nonfinite")
    return result


def execution_metadata(value, name, initial=None):
    require(isinstance(value, dict), "execution_metadata_shape")
    backend = "edge0-f32-reference" if name == "baseline" else "edge0-metal-graph-v1"
    batch = 1 if name == "baseline" else 8
    require(value.get("backend") == backend, "execution_backend")
    for precision in (
        "activation_precision",
        "accumulation_precision",
        "recurrent_state_precision",
        "attention_cache_precision",
    ):
        require(value.get(precision) == "f32", "execution_precision")
    require(value.get("prefix_reuse") is False, "execution_prefix_reuse")
    require(
        type(value.get("prefill_max_batch")) is int
        and value["prefill_max_batch"] == batch,
        "execution_prefill_batch",
    )
    weights = value.get("weights")
    require(isinstance(weights, dict), "execution_weights")
    require(type(weights.get("resident_enabled")) is bool, "execution_residency")
    require(
        type(weights.get("registered_bank_bytes")) is int
        and weights["registered_bank_bytes"] >= 0,
        "execution_registered_bank_bytes",
    )
    scratch = weights.get("graph_scratch_high_water_bytes", 0)
    require(
        type(scratch) is int and 0 <= scratch <= 64 * 1024**2,
        "execution_graph_scratch_cap",
    )
    if name == "candidate" and initial is not None:
        require(scratch > 0, "execution_graph_scratch_missing")
    if name == "baseline":
        require(value.get("source_binding") == "0721afc5", "baseline_source_binding")
    counters = weights.get("counters")
    require(isinstance(counters, dict), "execution_counters")
    for counter in ("batched_matrix_calls", "batched_q8_calls", "batched_expert_calls"):
        count = counters.get(counter)
        require(type(count) is int and count >= 0, "execution_counter_type")
        if name == "baseline":
            require(count == 0, "baseline_batched_dispatch")
        elif initial is not None:
            require(
                count > initial["weights"]["counters"][counter],
                "candidate_batched_dispatch_missing",
            )
    return value


def row_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def comparison(reference, candidate):
    require(
        reference.shape == candidate.shape and reference.size >= 2, "comparison_shape"
    )
    require(
        bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
        "comparison_nonfinite",
    )
    absolute = float(np.max(np.abs(reference - candidate)))
    denominator = float(np.linalg.norm(reference))
    relative = float(np.linalg.norm(reference - candidate)) / max(denominator, 1e-300)
    best = np.partition(reference, -2)[-2:]
    margin = float(best[1] - best[0])
    same_top1 = bool(np.argmax(reference) == np.argmax(candidate))
    return {
        "max_abs": absolute,
        "relative_l2": relative,
        "top1_agrees": same_top1,
        "top1_margin_gate_applies": margin > 2 * MAX_ABS,
        "passed": absolute <= MAX_ABS
        and relative <= REL_L2
        and (margin <= 2 * MAX_ABS or same_top1),
    }


class Probe:
    def __init__(self, binary, model, environment, minimum_available):
        require(
            psutil.virtual_memory().available >= minimum_available, "available_memory"
        )
        gpu_read, gpu_write = os.pipe()
        os.set_blocking(gpu_read, False)
        os.set_blocking(gpu_write, False)
        clean = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("STARK_", "UZU_"))
        }
        try:
            self.process = subprocess.Popen(
                [str(binary), str(model)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={**clean, **environment, "STARK_GPU_MEMORY_FD": str(gpu_write)},
                pass_fds=(gpu_write,),
            )
        except BaseException:
            os.close(gpu_read)
            raise
        finally:
            os.close(gpu_write)
        self.gpu_read = gpu_read
        self.gpu_buffer = bytearray()
        self.owner = psutil.Process(self.process.pid)
        self.buffer = bytearray()
        self.metrics = {
            "peak_rss_bytes": 0,
            "peak_gpu_allocated_bytes": 0,
            "gpu_samples": 0,
            "peak_phys_footprint_bytes": 0,
            "process_metrics_samples": 0,
            "minimum_available_bytes": psutil.virtual_memory().available,
        }

    def sample(self, deadline):
        require(time.monotonic() < deadline, "probe_timeout")
        try:
            rss = self.owner.memory_info().rss
        except psutil.NoSuchProcess:
            raise Failure("probe_exited") from None
        self.metrics["peak_rss_bytes"] = max(self.metrics["peak_rss_bytes"], rss)
        require(rss <= RSS_LIMIT, "rss_limit")
        self.metrics["minimum_available_bytes"] = min(
            self.metrics["minimum_available_bytes"], psutil.virtual_memory().available
        )
        snapshot = process_metrics.snapshot(self.process.pid)
        if snapshot.get("available"):
            self.metrics["process_metrics_samples"] += 1
            counters = snapshot["cumulative"]
            self.metrics["peak_phys_footprint_bytes"] = max(
                self.metrics["peak_phys_footprint_bytes"],
                counters["phys_footprint"],
                counters["lifetime_max_phys_footprint"],
            )
        try:
            self.gpu_buffer.extend(os.read(self.gpu_read, 65536))
        except BlockingIOError:
            pass
        require(len(self.gpu_buffer) <= 1024 * 1024, "gpu_telemetry_limit")
        while b"\n" in self.gpu_buffer:
            line, _, self.gpu_buffer = self.gpu_buffer.partition(b"\n")
            value = json.loads(line)["gpu_memory_bytes"]
            require(type(value) is int and value >= 0, "gpu_telemetry_invalid")
            self.metrics["peak_gpu_allocated_bytes"] = max(
                self.metrics["peak_gpu_allocated_bytes"], value
            )
            self.metrics["gpu_samples"] += 1

    def read(self):
        deadline = time.monotonic() + TIMEOUT
        while b"\n" not in self.buffer:
            self.sample(deadline)
            if select.select([self.process.stdout], [], [], 0.1)[0]:
                block = os.read(self.process.stdout.fileno(), 65536)
                require(bool(block), "probe_exited")
                self.buffer.extend(block)
                require(len(self.buffer) <= 16 * 1024 * 1024, "reply_limit")
        line, _, self.buffer = self.buffer.partition(b"\n")
        result = json.loads(line)
        require(isinstance(result, dict), "reply_shape")
        return result

    def request(self, operation, **values):
        self.process.stdin.write(
            json.dumps({"operation": operation, **values}, allow_nan=False).encode()
            + b"\n"
        )
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        os.close(self.gpu_read)


def forward(probe, tokens, length, *, reset=True):
    reply = probe.request("prefill", ids=tokens[:length], reset=reset)
    require(reply.get("prefix_tokens") == length, "prefix_count")
    rows = [row(reply.get("logits"), 248320)]
    timing = {
        "ingest_tokens": length - 1,
        "prefix_tokens": length,
        "ingest_seconds": reply["ingest_seconds"],
        "initial_readout_seconds": reply["readout_seconds"],
        "decode_steps": DECODE_STEPS,
        "decode_seconds": 0.0,
    }
    for token in tokens[length : length + DECODE_STEPS]:
        reply = probe.request("step", id=token)
        rows.append(row(reply.get("logits"), 248320))
        timing["decode_seconds"] += reply["seconds"]
    require(len(rows) == DECODE_STEPS + 1, "decode_count")
    require(
        all(
            type(value) in {int, float} and np.isfinite(value) and value >= 0
            for value in timing.values()
        ),
        "invalid_timing",
    )
    require(timing["decode_seconds"] > 0, "zero_timing")
    timing["prefill_seconds"] = (
        timing["ingest_seconds"] + timing["initial_readout_seconds"]
    )
    require(timing["prefill_seconds"] > 0, "zero_prefill_timing")
    timing["prefill_tokens_per_second"] = length / timing["prefill_seconds"]
    timing["decode_steps_per_second"] = DECODE_STEPS / timing["decode_seconds"]
    return rows, timing


def run_session(args, name, registration, summary, reference=None):
    probe = Probe(
        getattr(args, name + "_probe"),
        args.model,
        registration[name + "_selectors"],
        registration["minimum_available_bytes"],
    )
    stored = {}
    summary.update({"cases": [], "passed": True})
    try:
        ready = probe.read()
        require(
            ready.get("ready") is True
            and ready.get("protocol") == 1
            and ready.get("vocab_size") == 248320
            and ready.get("max_context") == 512,
            "probe_ready",
        )
        summary["load_seconds"] = ready["load_seconds"]
        summary["request_memory_estimate_bytes"] = ready["request_memory_bytes"]
        summary["execution_ready"] = ready.get("execution")
        execution_metadata(summary["execution_ready"], name)
        progress(name, "ready")
        tokens = probe.request("encode", text=TEXT)["ids"]
        require(len(tokens) >= max(PREFIX_LENGTHS) + DECODE_STEPS, "input_too_short")
        require(
            all(
                type(token) is int
                and 0 <= token < 248320
                and token not in ready["eos_ids"]
                for token in tokens
            ),
            "input_ids",
        )
        summary["input_ids_sha256"] = hashlib.sha256(
            np.asarray(tokens, dtype="<u4").tobytes()
        ).hexdigest()
        for length in PREFIX_LENGTHS:
            for repeat in range(WARMUPS):
                forward(probe, tokens, length)
                progress(name, "warmup_complete", prefix_tokens=length, repeat=repeat)
            rows, measurements = None, []
            for repeat in range(REPEATS):
                current, timing = forward(probe, tokens, length)
                measurements.append(timing)
                if rows is not None:
                    require(
                        all(
                            row_hash(a) == row_hash(b)
                            for a, b in zip(rows, current, strict=True)
                        ),
                        "nondeterministic_replay",
                    )
                rows = current
                progress(
                    name, "timed_replay_complete", prefix_tokens=length, repeat=repeat
                )
            stored[length] = rows
            case = {
                "prefix_tokens": length,
                "readout_rows": len(rows),
                "timed_runs": measurements,
                "readout_sha256": hashlib.sha256(
                    "".join(map(row_hash, rows)).encode()
                ).hexdigest(),
                "median_prefill_tokens_per_second": statistics.median(
                    m["prefill_tokens_per_second"] for m in measurements
                ),
                "median_decode_steps_per_second": statistics.median(
                    m["decode_steps_per_second"] for m in measurements
                ),
            }
            if reference is not None:
                results = [
                    comparison(a, b)
                    for a, b in zip(reference[length], rows, strict=True)
                ]
                case["comparison"] = {
                    "max_abs": max(r["max_abs"] for r in results),
                    "maximum_relative_l2": max(r["relative_l2"] for r in results),
                    "top1_agreements": sum(r["top1_agrees"] for r in results),
                    "margin_gate_rows": sum(
                        r["top1_margin_gate_applies"] for r in results
                    ),
                    "failed_rows": sum(not r["passed"] for r in results),
                    "passed": all(r["passed"] for r in results),
                }
                summary["passed"] &= case["comparison"]["passed"]
            summary["cases"].append(case)
        require(
            probe.request("reject_and_reset") == {"rejected": True}, "rejection_failed"
        )
        recovered, _ = forward(probe, tokens, PREFIX_LENGTHS[0], reset=False)
        require(
            all(
                row_hash(a) == row_hash(b)
                for a, b in zip(stored[PREFIX_LENGTHS[0]], recovered, strict=True)
            ),
            "recovery_mismatch",
        )
        summary["rejection_recovery_exact"] = True
        progress(name, "recovery_complete")
        summary["execution_final"] = probe.request("metadata")
        execution_metadata(summary["execution_final"], name, summary["execution_ready"])
        return stored
    finally:
        summary["memory"] = probe.metrics
        probe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--baseline-probe", type=Path, required=True)
    parser.add_argument("--candidate-probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--minimum-available-gib", type=int, default=64)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = recipe(args)
    require(
        expected["candidate_selectors"].get("STARK_EDGE0_METAL") == "1",
        "candidate_selector_required",
    )
    if args.action == "register":
        write_new(args.registration, expected)
        return
    require(
        args.output is not None and not args.output.exists(),
        "output_required_or_exists",
    )
    require(
        json.loads(args.registration.read_text()) == expected, "registration_changed"
    )
    report = {
        "registration_sha256": digest(args.registration),
        "status": "failed",
        "memory_scope": "sampled owned-worker RSS/physical footprint and device resource allocations; not total residency",
    }
    try:
        report["baseline"] = {}
        reference = run_session(args, "baseline", expected, report["baseline"])
        report["candidate"] = {}
        run_session(args, "candidate", expected, report["candidate"], reference)
        require(
            report["baseline"]["input_ids_sha256"]
            == report["candidate"]["input_ids_sha256"],
            "tokenizer_mismatch",
        )
        require(report["candidate"]["passed"], "numerical_gate")
        require(bindings(args) == expected["files_sha256"], "files_changed_during_run")
        report["matched_speed"] = [
            {
                "prefix_tokens": baseline["prefix_tokens"],
                "median_prefill_speedup": candidate["median_prefill_tokens_per_second"]
                / baseline["median_prefill_tokens_per_second"],
                "median_decode_speedup": candidate["median_decode_steps_per_second"]
                / baseline["median_decode_steps_per_second"],
            }
            for baseline, candidate in zip(
                report["baseline"]["cases"], report["candidate"]["cases"], strict=True
            )
        ]
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- private IPC failures must never emit exception dumps.
        report["failure"] = (
            str(error)
            if isinstance(error, Failure)
            else "qualification_io_or_protocol_failure"
        )
    finally:
        write_new(args.output, report)
    print(
        json.dumps({"status": report["status"], "report_sha256": digest(args.output)}),
        flush=True,
    )
    require(report["status"] == "passed", "qualification_failed")


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        raise SystemExit(str(error)) from None
    except Exception:  # noqa: BLE001 -- the command boundary exposes fixed, content-free codes only.
        raise SystemExit("qualification_unhandled_failure") from None
