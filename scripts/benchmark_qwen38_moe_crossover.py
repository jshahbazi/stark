"""Within-worker MoE diagnostic crossover; requires the archived private worker patch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time

import benchmark_qwen38_mtp as mtp
from benchmark_qwen38_pilot import clean_environment, weight_guard
from benchmark_clean_qwen import Monitor, paging
from benchmark_glm_mlx_serve import screen_attempt
from benchmark_hybrid import hardware_identity
from benchmark_hybrid_context import execution_profile
from hybrid_host_metrics import HostMetricsSampler
from qwen38_clock_metrics import ClockSampler, summarize as clock_summary
from qwen38_process_metrics import snapshot, delta
from stark_convert.chat import ChatError, Worker


def plan(seed, repetitions):
    rng = random.Random(seed)
    result = []
    for repetition in range(repetitions):
        pairs = [(route, candidate) for route in ("off", "on") for candidate in (1, 2)]
        rng.shuffle(pairs)
        for route, candidate in pairs:
            order = [0, candidate, candidate, 0] if repetition % 2 == 0 else [candidate, 0, 0, candidate]
            result.append({"repetition": repetition, "route": route, "candidate": candidate, "order": order})
    return result


def summarize_groups(groups):
    result = []
    for group in groups:
        reference = [r for r in group["requests"] if r["variant"] == 0]
        candidate = [r for r in group["requests"] if r["variant"] == group["candidate"]]
        if len(reference) != 2 or len(candidate) != 2:
            raise ValueError("incomplete_crossover_group")
        result.append({
            "repetition": group["repetition"], "route": group["route"], "candidate": group["candidate"],
            "reference_decode_tok_s": 510 / sum(r["decode_seconds"] for r in reference),
            "candidate_decode_tok_s": 510 / sum(r["decode_seconds"] for r in candidate),
            "candidate_over_reference_decode_ratio": sum(r["decode_seconds"] for r in reference) / sum(r["decode_seconds"] for r in candidate),
            "reference_median_wall_seconds": statistics.median(r["wall_seconds"] for r in reference),
            "candidate_median_wall_seconds": statistics.median(r["wall_seconds"] for r in candidate),
        })
    return result


def run(args):
    if args.output.exists() or args.native_profile_output.exists():
        raise ValueError("output_exists")
    if args.repetitions not in (2, 4):
        raise ValueError("invalid_repetitions")
    args.input_tokens, args.tokens, args.blocks = 512, 256, 4
    args.drafts, args.top_k, args.top_p, args.seed = 2, 20, .95, 42
    args.checkpoint_receipt = None
    mtp.validate_args(args)
    guard = weight_guard(args.model, args.weights_receipt)
    report = {
        "schema": "stark-flashnext-moe-crossover-v1", "harness_sha256": mtp.sha256_file(Path(__file__)),
        "worker_sha256": mtp.sha256_file(args.worker), "hardware": hardware_identity(),
        "weights_receipt_sha256": guard, "configuration_sha256": mtp.sha256_file(args.model / "config.json"),
        "corpus_sha256": mtp.sha256_file(args.corpus), "sampling": mtp.sampling_options(args), "seed": args.seed,
        "input_tokens": 512, "output_tokens": 256, "context_tokens": 768,
        "variants": {"0": "original generic gate/up and shared-after", "1": "fixed gate/up and shared-after", "2": "fixed gate/up and shared-first"},
        "policy": "One loaded model; every variant setter clears retained prefix state. Six condition warmups and six settling requests precede measured ABBA/BAAB groups. All arms retained, no quiet admission or automatic promotion.",
        "native_cpu_profile": True, "gpu_stage_timestamp_writes": False,
        "warmups": [], "settling": [], "groups": [],
    }
    report.update(mtp.checkpoint_metadata(args))
    worker = clocks = monitor = host = None
    with clean_environment():
        try:
            report["execution_profile"] = execution_profile(screened=True)
            report["telemetry_helper_sha256"] = {name: mtp.sha256_file(Path(__file__).with_name(name)) for name in ("qwen38_clock_metrics.py", "qwen38_process_metrics.py", "hybrid_host_metrics.py")}
            args.native_profile_output.parent.mkdir(parents=True, exist_ok=True)
            os.environ["STARK_NATIVE_PROFILE_OUTPUT"] = str(args.native_profile_output.resolve())
            try:
                worker = Worker(args.worker.resolve(), args.model.resolve(), 3600)
            finally:
                os.environ.pop("STARK_NATIVE_PROFILE_OUTPUT", None)
            ready = worker.read()
            if ready.get("ready") is not True:
                raise ValueError("worker_not_ready")
            tokenized = worker.request({"operation": "tokenize", "text": args.corpus.read_text(), "add_special_tokens": False})
            if not mtp._valid_ids(tokenized.get("token_ids")):
                raise ValueError("corpus_tokenization_failed")
            ids = mtp.exact_prefix(tokenized["token_ids"], 512)
            report["input_sha256"] = mtp.ids_sha256(ids)
            clocks = ClockSampler(args.clock_sampler, 500).start()
            monitor = Monitor().__enter__()
            host = HostMetricsSampler(worker.process.pid).__enter__()
            request_id = 0

            def request(variant, route):
                nonlocal request_id
                ack = worker.request({"operation": "diagnostic_moe_variant", "variant": variant})
                if type(ack.get("variant")) is not int or ack["variant"] != variant:
                    raise ValueError("variant_ack_failed")
                request_id += 1
                phase = f"crossover_{request_id}"
                monitor.set_phase(phase)
                before = paging()
                proc_before = snapshot(worker.process.pid)
                marker = host.begin_request()
                start_unix, started = time.time(), time.monotonic()
                raw = worker.request({"operation": "generate", "input_token_ids": ids,
                    "context_tokens": 768, "max_new_tokens": 256, "sampling": report["sampling"],
                    "seed": 42, "mtp_drafts": 2 if route == "on" else 0, "request_id": request_id})
                elapsed, end_unix = time.monotonic() - started, time.time()
                proc_after = snapshot(worker.process.pid)
                host.end_request(marker)
                item = mtp._validate_result(raw, ready, args, route, 2, ready["vocab_size"])
                item.update(variant=variant, request_id=request_id, wall_seconds=elapsed,
                            started_unix_seconds=start_unix, finished_unix_seconds=end_unix)
                item["host_screen"] = screen_attempt(monitor, phase, before, paging())
                item["process_metrics"] = delta(proc_before, proc_after)
                item["host_attribution"] = host.request_result(marker)
                item["clock_full_request"] = clock_summary(clocks.snapshot(), start_unix, end_unix)
                item["clock_approximate_decode"] = clock_summary(clocks.snapshot(), start_unix + item["prefill_seconds"], end_unix)
                return item

            for variant in (0, 1, 2):
                for route in ("off", "on"):
                    report["warmups"].append(request(variant, route))
            for _ in range(6):
                report["settling"].append(request(0, "on"))
            for number, group in enumerate(plan(42, args.repetitions)):
                group["requests"] = [request(variant, group["route"]) for variant in group["order"]]
                report["groups"].append(group)
                print(json.dumps({"group": number, "summary": summarize_groups([group])[0]}), flush=True)
            all_items = report["warmups"] + report["settling"] + [r for g in report["groups"] for r in g["requests"]]
            report["group_summary"] = summarize_groups(report["groups"])
            report["same_route_hashes_equal"] = all(len({r["output_sha256"] for r in all_items if r["route"] == route}) == 1 for route in ("off", "on"))
            report["artifact_unchanged"] = weight_guard(args.model, args.weights_receipt) == guard
            report["worker_unchanged"] = mtp.sha256_file(args.worker) == report["worker_sha256"]
            report["passed"] = all(report[k] for k in ("same_route_hashes_equal", "artifact_unchanged", "worker_unchanged"))
            report["timing_eligible"] = False
            report["all_measured_host_screens_eligible"] = all(r["host_screen"]["eligible"] for g in report["groups"] for r in g["requests"])
            clocks.close()
            report["clock_capture"] = clocks.report()
            report["clock_capture_valid"] = report["clock_capture"]["failures"] == 0 and all(r["clock_approximate_decode"]["fully_contained_intervals"] > 0 for r in all_items)
            report["execution_profile_unchanged"] = execution_profile(screened=True) == report["execution_profile"]
            report["passed"] = report["passed"] and report["execution_profile_unchanged"]
            host.__exit__(None, None, None)
            report["host_attribution_sampler"] = host.summary()
            mtp._atomic_report(args.output, report)
            return report
        finally:
            if worker is not None:
                worker.close()
            if clocks is not None:
                clocks.close()
            if monitor is not None:
                monitor.__exit__(None, None, None)
            if host is not None:
                host.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "worker", "corpus", "weights-receipt", "clock-sampler", "native-profile-output", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ChatError):
        print(json.dumps({"passed": False, "stage": "crossover_failed"}))
        return 1
    print(json.dumps({"passed": report["passed"], "all_screens": report["all_measured_host_screens_eligible"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
