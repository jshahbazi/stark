"""Probe request cadence with one sealed Flash-Next worker; retain aggregates only."""

from __future__ import annotations

import argparse
import json
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


def run(args):
    if args.output.exists():
        raise ValueError("output_exists")
    if not 2 <= args.cycles <= 8 or not 2 <= args.consecutive <= 8 or not 1 <= args.rest_seconds <= 30:
        raise ValueError("invalid_cadence")
    args.input_tokens, args.tokens, args.blocks = 512, 256, 4
    args.drafts, args.top_k, args.top_p, args.seed = 2, 20, 0.95, 42
    args.checkpoint_receipt = None
    mtp.validate_args(args)
    weights_before = weight_guard(args.model, args.weights_receipt)
    worker_hash = mtp.sha256_file(args.worker)
    report = {
        "schema": "stark-flashnext-cadence-v1",
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
    }
    report.update(mtp.checkpoint_metadata(args))
    with clean_environment():
        report["execution_profile"] = execution_profile(screened=True)
        worker = Worker(args.worker.resolve(), args.model.resolve(), 3600)
        monitor = None
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
            counter = 0

            def request(route):
                nonlocal counter
                counter += 1
                phase = f"cadence_request_{counter}"
                monitor.set_phase(phase)
                before = paging()
                started = time.monotonic()
                result = worker.request({
                    "operation": "generate", "input_token_ids": ids,
                    "context_tokens": report["context_tokens"],
                    "max_new_tokens": args.tokens, "sampling": report["sampling"],
                    "seed": args.seed, "mtp_drafts": args.drafts if route == "on" else 0,
                    "request_id": counter,
                })
                elapsed = time.monotonic() - started
                item = mtp._validate_result(result, ready, args, route, args.drafts, ready["vocab_size"])
                item["wall_seconds"] = elapsed
                item["host_screen"] = screen_attempt(monitor, phase, before, paging())
                return item

            report["warmups"] = [request("off"), request("on")]
            for cycle in range(args.cycles):
                monitor.set_phase("fixed_rest")
                time.sleep(args.rest_seconds)
                for position in range(args.consecutive):
                    item = request("on")
                    item.update(cycle=cycle, position_after_rest=position)
                    report["requests"].append(item)
                    print(json.dumps({"cycle": cycle, "position": position, "decode_tok_s": 255 / item["decode_seconds"]}), flush=True)
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
            mtp._atomic_report(args.output, report)
            return report
        finally:
            worker.close()
            if monitor is not None:
                monitor.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "worker", "corpus", "weights-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--consecutive", type=int, default=4)
    parser.add_argument("--rest-seconds", type=float, default=10)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ChatError):
        print(json.dumps({"passed": False, "stage": "cadence_probe_failed"}))
        return 1
    print(json.dumps({"passed": report["passed"], "by_position": report["by_position"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
