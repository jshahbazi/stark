"""Probe request cadence with one sealed Flash-Next worker; retain aggregates only."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import benchmark_qwen38_mtp as mtp
from benchmark_clean_qwen import Monitor, paging
from benchmark_glm_mlx_serve import screen_attempt
from benchmark_hybrid import hardware_identity
from benchmark_hybrid_context import execution_profile
from benchmark_qwen38_pilot import clean_environment, weight_guard
from stark_convert.chat import ChatError, Worker
from qwen38_clock_metrics import ClockSampler, summarize as summarize_clocks
from qwen38_process_metrics import snapshot as process_snapshot, delta as process_delta
from hybrid_host_metrics import HostMetricsSampler


def run(args):
    if args.output.exists():
        raise ValueError("output_exists")
    if args.native_profile_output is not None and args.native_profile_output.exists():
        raise ValueError("profile_output_exists")
    if not 2 <= args.cycles <= 8 or not 2 <= args.consecutive <= 8 or not 1 <= args.rest_seconds <= 30:
        raise ValueError("invalid_cadence")
    args.input_tokens, args.tokens, args.blocks = 512, 256, 4
    args.drafts, args.top_k, args.top_p, args.seed = 2, 20, 0.95, 42
    args.checkpoint_receipt = None
    mtp.validate_args(args)
    weights_before = weight_guard(args.model, args.weights_receipt)
    worker_hash = mtp.sha256_file(args.worker)
    report = {
        "schema": "stark-flashnext-cadence-v3",
        "harness_sha256": mtp.sha256_file(Path(__file__)),
        "paired_harness_sha256": mtp.sha256_file(Path(mtp.__file__)),
        "worker_sha256": worker_hash,
        "hardware": hardware_identity(),
        "weights_receipt_sha256": weights_before,
        "configuration_sha256": mtp.sha256_file(args.model / "config.json"),
        "corpus_sha256": mtp.sha256_file(args.corpus),
        "sampling": mtp.sampling_options(args),
        "seed": args.seed,
        "input_tokens": args.input_tokens,
        "output_tokens": args.tokens,
        "context_tokens": args.input_tokens + args.tokens,
        "drafts": args.drafts,
        "cycles": args.cycles,
        "consecutive_requests": args.consecutive,
        "rest_seconds_before_each_cycle": args.rest_seconds,
        "policy": "Fixed rest before each group; subsequent requests retain the ordinary 0.6s host-screen interval. No quiet admission or selective removal. Same loaded worker, input, seed and MTP route throughout.",
        "requests": [],
        "clock_sampler_requested": args.clock_sampler is not None,
        "clock_helper_sha256": mtp.sha256_file(Path(__file__).with_name("qwen38_clock_metrics.py")),
        "process_metrics_requested": args.process_metrics,
        "process_helper_sha256": mtp.sha256_file(Path(__file__).with_name("qwen38_process_metrics.py")),
        "native_aggregate_profiling": args.native_profile_output is not None,
    }
    report.update(mtp.checkpoint_metadata(args))
    with clean_environment():
        report["execution_profile"] = execution_profile(screened=True)
        try:
            if args.native_profile_output is not None:
                args.native_profile_output.parent.mkdir(parents=True, exist_ok=True)
                os.environ["STARK_NATIVE_PROFILE_OUTPUT"] = str(args.native_profile_output.resolve())
            worker = Worker(args.worker.resolve(), args.model.resolve(), 3600)
        finally:
            os.environ.pop("STARK_NATIVE_PROFILE_OUTPUT", None)
        monitor = None
        clocks = None
        attribution = None
        try:
            ready = worker.read()
            if ready.get("ready") is not True or ready.get("mode") != "text-generation-v1":
                raise ValueError("worker_not_ready")
            if ready.get("mtp", {}).get("verification") not in mtp.EXACT_VERIFIERS:
                raise ValueError("mtp_verifier_unavailable")
            tokenized = worker.request({"operation": "tokenize", "text": args.corpus.read_text(), "add_special_tokens": False})
            if not mtp._valid_ids(tokenized.get("token_ids")):
                raise ValueError("corpus_tokenization_failed")
            ids = mtp.exact_prefix(tokenized["token_ids"], args.input_tokens)
            report["input_sha256"] = mtp.ids_sha256(ids)
            monitor = Monitor()
            monitor.__enter__()
            if args.clock_sampler is not None:
                clocks = ClockSampler(args.clock_sampler, args.clock_interval_ms).start()
            if args.process_metrics:
                attribution = HostMetricsSampler(worker.process.pid)
                attribution.__enter__()
            counter = 0

            def request(route):
                nonlocal counter
                counter += 1
                phase = f"cadence_request_{counter}"
                monitor.set_phase(phase)
                before = paging()
                process_before = process_snapshot(worker.process.pid) if args.process_metrics else None
                marker = attribution.begin_request() if attribution is not None else None
                started_unix = time.time()
                started = time.monotonic()
                result = worker.request({
                    "operation": "generate", "input_token_ids": ids,
                    "context_tokens": report["context_tokens"],
                    "max_new_tokens": args.tokens, "sampling": report["sampling"],
                    "seed": args.seed, "mtp_drafts": args.drafts if route == "on" else 0,
                    "request_id": counter,
                })
                elapsed = time.monotonic() - started
                finished_unix = time.time()
                process_after = process_snapshot(worker.process.pid) if args.process_metrics else None
                if attribution is not None:
                    attribution.end_request(marker)
                item = mtp._validate_result(result, ready, args, route, args.drafts, ready["vocab_size"])
                item["wall_seconds"] = elapsed
                item["started_unix_seconds"] = started_unix
                item["finished_unix_seconds"] = finished_unix
                item["host_screen"] = screen_attempt(monitor, phase, before, paging())
                if args.process_metrics:
                    item["process_metrics"] = process_delta(process_before, process_after)
                    item["host_attribution"] = attribution.request_result(marker)
                if clocks is not None:
                    samples = clocks.snapshot()
                    item["clock_full_request"] = summarize_clocks(samples, started_unix, finished_unix)
                    item["clock_approximate_decode"] = summarize_clocks(samples, started_unix + item["prefill_seconds"], finished_unix)
                return item

            report["warmups"] = [request("off"), request("on")]
            for cycle in range(args.cycles):
                monitor.set_phase("fixed_rest")
                time.sleep(args.rest_seconds)
                for position in range(args.consecutive):
                    item = request("on")
                    item.update(cycle=cycle, position_after_rest=position)
                    report["requests"].append(item)
                    print(json.dumps({"cycle": cycle, "position": position, "decode_tok_s": 255 / item["decode_seconds"], "decode_clocks": item.get("clock_approximate_decode")}), flush=True)
            report["by_position"] = []
            for position in range(args.consecutive):
                items = [r for r in report["requests"] if r["position_after_rest"] == position]
                report["by_position"].append({
                    "position_after_rest": position, "requests": len(items),
                    "pooled_decode_tok_s": 255 * len(items) / sum(r["decode_seconds"] for r in items),
                    "median_prefill_seconds": statistics.median(r["prefill_seconds"] for r in items),
                    "median_wall_seconds": statistics.median(r["wall_seconds"] for r in items),
                    "eligible_arms": sum(r["host_screen"]["eligible"] for r in items),
                })
            report["artifact_unchanged"] = weight_guard(args.model, args.weights_receipt) == weights_before
            report["worker_unchanged"] = mtp.sha256_file(args.worker) == worker_hash
            report["execution_profile_unchanged"] = execution_profile(screened=True) == report["execution_profile"]
            hashes = {r["output_sha256"] for r in report["requests"] + [report["warmups"][1]]}
            report["passed"] = len(hashes) == 1 and all(report[k] for k in ("artifact_unchanged", "worker_unchanged", "execution_profile_unchanged"))
            report["timing_eligible"] = False
            report["all_host_screens_eligible"] = all(r["host_screen"]["eligible"] for r in report["requests"])
            if clocks is not None:
                clocks.close()
                report["clock_capture"] = clocks.report()
                report["clock_capture_valid"] = report["clock_capture"]["failures"] == 0 and all(r["clock_approximate_decode"]["fully_contained_intervals"] > 0 for r in report["requests"])
            if attribution is not None:
                report["host_attribution_sampler"] = attribution.summary()
            mtp._atomic_report(args.output, report)
            return report
        finally:
            worker.close()
            if monitor is not None:
                monitor.__exit__(None, None, None)
            if clocks is not None:
                clocks.close()
            if attribution is not None:
                attribution.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "worker", "corpus", "weights-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--consecutive", type=int, default=4)
    parser.add_argument("--rest-seconds", type=float, default=10)
    parser.add_argument("--clock-sampler", type=Path)
    parser.add_argument("--clock-interval-ms", type=int, choices=(250, 500, 1000), default=500)
    parser.add_argument("--process-metrics", action="store_true")
    parser.add_argument("--native-profile-output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ChatError):
        print(json.dumps({"passed": False, "stage": "cadence_probe_failed"}))
        return 1
    print(json.dumps({"passed": report["passed"], "by_position": report["by_position"]}))
    return 0 if report["passed"] and report.get("clock_capture_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
