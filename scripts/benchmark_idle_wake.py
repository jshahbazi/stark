"""Serialized cross-backend idle latency controls; publish hashes and aggregates."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import time

from benchmark_clean_qwen import Monitor, paging
from benchmark_glm_mlx_serve import quiet_admission, screen_attempt
from benchmark_hybrid import hardware_identity
from benchmark_hybrid_context import execution_profile
from benchmark_qwen38_dense import require_mtp_depth, summarize as generation_summary
from benchmark_qwen38_mtp import exact_prefix, ids_sha256, sha256_file
from benchmark_qwen38_pilot import clean_environment
from benchmark_stock_engines import Pipe
from hybrid_host_metrics import HostMetricsSampler
from qwen38_process_metrics import delta, snapshot
from qwen38_clock_metrics import ClockSampler, summarize as clock_summary


def artifact_guard(model, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "stark-idle-artifact-v1" or receipt.get("passed") is not True:
        raise ValueError("invalid_artifact_receipt")
    files = receipt["files"]
    payload = lambda n: n.endswith(".safetensors") or n == "ngram_table.bin"
    present = {p.name for p in model.iterdir() if p.is_file() and payload(p.name)}
    if not files or present != {n for n in files if payload(n)}:
        raise ValueError("payload_inventory_changed")
    for name, expected in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid_artifact_receipt")
        stat = (model / path).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != expected["identity"]:
            raise ValueError("artifact_identity_changed")
        if "identity5" in expected and [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != expected["identity5"]:
            raise ValueError("artifact_identity_changed")
    if any(sha256_file(model / n) != h for n, h in receipt["metadata_sha256"].items()):
        raise ValueError("artifact_metadata_changed")
    return {"receipt_sha256": sha256_file(receipt_path), "source": receipt["source"],
            "files": len(files), "payload_bytes": sum(v["identity"][0] for n, v in files.items() if payload(n)),
            "configuration_sha256": receipt["metadata_sha256"]["config.json"]}


def scoring_summary(reply, *, digest=False):
    if digest:
        value = reply.get("vocab_logits_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("invalid_scoring_digest")
        return {"output_sha256": value, "cache_hit": reply["cache_hit"], "cache": reply["cache"],
                "input_tokens": reply["input_tokens"], "vocab_size": reply["vocab_size"]}
    vectors = [reply.get(k) for k in ("logits", "log_probabilities", "candidate_probabilities")]
    if any(not isinstance(v, list) or len(v) != 4 for v in vectors) or not all(
        type(x) in (int, float) and math.isfinite(x) for v in vectors for x in v
    ):
        raise ValueError("invalid_scoring_output")
    fingerprint = hashlib.sha256(b"".join(struct.pack("<d", x) for v in vectors for x in v)).hexdigest()
    return {"output_sha256": fingerprint, "cache_hit": reply["cache_hit"], "cache": reply["cache"],
            "input_tokens": reply["input_tokens"], "metrics": reply["metrics"]}


def generation_item(raw, ready, requested):
    item = generation_summary(raw)
    metadata = item["mtp"]
    nested = raw.get("mtp", {}).get("stats", {})
    for key in ("rounds", "proposed", "accepted", "verified_rows", "draft_seconds", "verify_seconds", "rollback_seconds"):
        if key in nested:
            metadata[key] = nested[key]
    if metadata.get("enabled") is not (requested > 0) or metadata.get("drafts") != requested:
        raise ValueError("mtp_route_acknowledgement_mismatch")
    if requested and (metadata.get("verification") != ready["mtp"]["verification"] or not metadata.get("proposed", 0)):
        raise ValueError("mtp_verifier_acknowledgement_mismatch")
    return item


def summarize_positions(requests):
    return [{"route": route, "position": pos, "requests": len(rows),
             "median_wall_seconds": statistics.median(r["wall_seconds"] for r in rows),
             "median_prefill_seconds": statistics.median(r["prefill_seconds"] for r in rows) if "prefill_seconds" in rows[0] else None,
             "eligible": sum(r["host_screen"]["eligible"] for r in rows)}
            for route, pos in sorted({(r["route"], r["position"]) for r in requests})
            if (rows := [r for r in requests if r["route"] == route and r["position"] == pos])]


def request_screen(monitor, phase, before, after, *, cache_hit):
    if cache_hit:
        monitor.set_phase("between_requests")
        return {"eligible": False, "reason": "cache_correctness_only"}
    try:
        return screen_attempt(monitor, phase, before, after)
    except ValueError as error:
        if str(error) != "host_observations_missing":
            raise
        return {"eligible": False, "reason": "host_observations_missing"}


def checkpoint(output, report, stage):
    report["stage"] = stage
    partial = output.with_suffix(output.suffix + ".partial")
    temporary = partial.with_suffix(partial.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(partial)


def run(args):
    if args.output.exists() or not 1 <= args.rest_seconds <= 30 or args.input_tokens < 2 or args.tokens < 2:
        raise ValueError("invalid_idle_benchmark")
    guard = artifact_guard(args.model, args.artifact_receipt)
    worker_hash = sha256_file(args.worker)
    report = {"schema": "stark-cross-backend-idle-v4", "endpoint": args.endpoint,
        "worker_sha256": worker_hash, "harness_sha256": sha256_file(Path(__file__)),
        "artifact": guard, "hardware": hardware_identity(), "corpus_sha256": sha256_file(args.corpus),
        "input_tokens": args.input_tokens, "output_budget": args.tokens if args.endpoint == "generation" else None,
        "score_temperature": 1.22 if args.endpoint != "generation" else None,
        "context_tokens": args.input_tokens + args.tokens,
        "seed": 42, "sampling": {"mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": .95},
        "rest_seconds": args.rest_seconds, "quiet_admission": args.screened,
        "policy": "One loaded worker; one warmup per route; fixed pause then cold and warm requests. Score cache reset before fresh requests; separate post-idle hit control. All arms retained.",
        "warmups": [], "requests": [], "idle_intervals": [], "cache_controls": [], "prefix_controls": []}
    checkpoint(args.output, report, "load_worker")
    routes = ["off", "on"] if args.endpoint == "generation" else ["score"]
    with clean_environment(), contextlib.ExitStack() as stack:
        report["execution_profile"] = execution_profile(screened=True)
        command = [str(args.worker.resolve()), "--model-path" if args.endpoint == "generation" else "--score-model-path", str(args.model.resolve()),
                   "--prefix-cache" if args.endpoint == "generation" else "--consecutive-cache"]
        pipe = Pipe(command, stack, str(Path.cwd()), None)
        checkpoint(args.output, report, "worker_ready")
        ready = pipe.ready
        if args.endpoint == "generation":
            require_mtp_depth(ready, args.drafts)
        checkpoint(args.output, report, "tokenize")
        ids = exact_prefix(pipe.request({"operation": "tokenize", "text": args.corpus.read_text(), "add_special_tokens": False})["token_ids"], args.input_tokens)
        report["input_sha256"] = ids_sha256(ids)
        candidates = list(dict.fromkeys(ids))[:4]
        if len(candidates) != 4:
            raise ValueError("insufficient_candidates")
        report["candidates_sha256"] = ids_sha256(candidates)
        checkpoint(args.output, report, "host_monitors")
        monitor = stack.enter_context(Monitor())
        host = stack.enter_context(HostMetricsSampler(pipe.process.pid))
        checkpoint(args.output, report, "clock_sampler")
        clocks = ClockSampler(args.clock_sampler, 500).start() if args.clock_sampler else None
        if clocks is not None:
            stack.callback(clocks.close)
        counter = 0

        def request(route, cache_hit=False):
            nonlocal counter
            checkpoint(args.output, report, "cache_hit" if cache_hit else "fresh_request")
            if args.endpoint != "generation" and not cache_hit:
                if pipe.request({"operation": "reset_cache"}).get("cache_reset") is not True:
                    raise ValueError("cache_reset_failed")
            payload = {"operation": "generate" if args.endpoint == "generation" else "score",
                       "input_token_ids": ids, "context_tokens": report["context_tokens"]}
            if args.endpoint == "generation":
                payload.update(max_new_tokens=args.tokens, seed=42, sampling=report["sampling"], mtp_drafts=args.drafts if route == "on" else 0)
            else:
                payload.update(candidate_token_ids=candidates, temperature=1.22)
            counter += 1
            phase = f"idle_request_{counter}"
            monitor.set_phase(phase)
            before = paging()
            proc = snapshot(pipe.process.pid)
            marker = host.begin_request()
            started = time.monotonic()
            raw = pipe.request(payload)
            wall = time.monotonic() - started
            proc_after = snapshot(pipe.process.pid)
            host.end_request(marker)
            item = generation_item(raw, ready, args.drafts if route == "on" else 0) if args.endpoint == "generation" else scoring_summary(raw, digest=args.endpoint == "score-digest")
            if args.endpoint != "generation" and item["cache_hit"] != cache_hit:
                raise ValueError("unexpected_score_cache_state")
            item.update(route=route, wall_seconds=wall, process_metrics=delta(proc, proc_after),
                        host_attribution=host.request_result(marker),
                        host_screen=request_screen(monitor, phase, before, paging(), cache_hit=cache_hit))
            return item

        def rest(route):
            admission = quiet_admission(monitor) if args.screened else None
            monitor.set_phase("idle")
            before = snapshot(pipe.process.pid)
            start = time.monotonic()
            start_unix = time.time()
            time.sleep(args.rest_seconds)
            end_unix = time.time()
            item = {"route": route, "seconds": time.monotonic()-start, "admission": admission,
                    "process_metrics": delta(before, snapshot(pipe.process.pid))}
            if clocks is not None:
                item["clocks"] = clock_summary(clocks.snapshot(), start_unix, end_unix)
            report["idle_intervals"].append(item)
            checkpoint(args.output, report, "idle_complete")

        for route in routes:
            report["warmups"].append(request(route))
            checkpoint(args.output, report, "warmup_complete")
        for route in routes:
            if args.endpoint != "generation":
                pipe.request({"operation": "reset_cache"})
            rest(route)
            for position in (0, 1):
                item = request(route)
                item["position"] = position
                report["requests"].append(item)
                checkpoint(args.output, report, "request_complete")
            if args.endpoint != "generation":
                previous_calls = item["cache"]["native_score_calls"]
                rest("cache_hit")
                hit = request(route, True)
                if hit["cache"]["native_score_calls"] != previous_calls or hit["output_sha256"] != item["output_sha256"]:
                    raise ValueError("score_cache_changed_during_idle")
                report["cache_controls"].append(hit)
                checkpoint(args.output, report, "cache_control_complete")
        all_requests = report["warmups"] + report["requests"] + report["cache_controls"]
        report["same_route_hashes_equal"] = all(len({r["output_sha256"] for r in all_requests if r["route"] == route}) == 1 for route in routes)
        report["summary"] = summarize_positions(report["requests"])
        report["artifact_unchanged"] = artifact_guard(args.model, args.artifact_receipt) == guard
        report["worker_unchanged"] = sha256_file(args.worker) == worker_hash
        report["execution_profile_unchanged"] = execution_profile(screened=True) == report["execution_profile"]
        report["all_host_screens_eligible"] = all(r["host_screen"]["eligible"] for r in report["requests"])
        report["passed"] = all(report[k] for k in ("same_route_hashes_equal", "artifact_unchanged", "worker_unchanged", "execution_profile_unchanged"))
        if args.endpoint == "generation" and args.prefix_control:
            chunk = ready.get("prefix_cache", {}).get("chunk_tokens")
            if not ready.get("prefix_cache", {}).get("enabled") or type(chunk) is not int or chunk < 1:
                raise ValueError("prefix_reuse_unavailable")
            prefix_ids = exact_prefix(pipe.request({"operation": "tokenize", "text": args.corpus.read_text(), "add_special_tokens": False})["token_ids"], chunk + 512)
            for route in routes:
                rows = []
                for phase in ("capture", "reuse_after_idle", "fresh"):
                    if phase == "reuse_after_idle":
                        time.sleep(args.rest_seconds)
                    payload = {"operation": "generate", "input_token_ids": prefix_ids,
                        "context_tokens": len(prefix_ids)+64, "max_new_tokens": 64, "seed": 42,
                        "sampling": report["sampling"], "mtp_drafts": args.drafts if route == "on" else 0}
                    if phase != "fresh":
                        payload["cache_scope"] = "1234567890abcdef1234567890abcdef"
                    raw = pipe.request(payload)
                    row = generation_item(raw, ready, args.drafts if route == "on" else 0)
                    row.update(phase=phase, reused_input_tokens=raw["reused_input_tokens"])
                    rows.append(row)
                if len({r["output_sha256"] for r in rows}) != 1 or [r["reused_input_tokens"] for r in rows] != [0, chunk, 0]:
                    raise ValueError("prefix_restore_mismatch")
                report["prefix_controls"].append({"route": route, "input_tokens": len(prefix_ids),
                    "input_sha256": ids_sha256(prefix_ids), "chunk_tokens": chunk, "requests": rows, "passed": True})
                checkpoint(args.output, report, "prefix_control_complete")
            report["artifact_unchanged"] = artifact_guard(args.model, args.artifact_receipt) == guard
            report["passed"] = report["passed"] and report["artifact_unchanged"]
        started = time.monotonic()
        checkpoint(args.output, report, "eof_cleanup")
        pipe.process.stdin.close()
        report["eof_exit_code"] = pipe.process.wait(timeout=10)
        report["eof_exit_seconds"] = time.monotonic() - started
        report["passed"] = report["passed"] and report["eof_exit_code"] == 0
        if clocks is not None:
            clocks.close()
            report["clock_capture"] = {k: v for k, v in clocks.report().items() if k != "samples"}
            report["passed"] = report["passed"] and report["clock_capture"]["failures"] == 0
        report["correctness_passed"] = report["passed"]
        report["timing_qualified"] = report["passed"] and args.screened and all(
            r["host_screen"]["eligible"] for r in report["requests"]
        )
        # ExitStack closes the worker and monitors before releasing the caller's GPU lease.
    checkpoint(args.output, report, "complete")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "worker", "artifact-receipt", "corpus", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--endpoint", choices=("generation", "score", "score-digest"), required=True)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--rest-seconds", type=float, default=10)
    parser.add_argument("--screened", action="store_true")
    parser.add_argument("--prefix-control", action="store_true")
    parser.add_argument("--clock-sampler", type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        safe = {"mtp_route_acknowledgement_mismatch", "mtp_verifier_acknowledgement_mismatch",
                "prefix_restore_mismatch", "prefix_reuse_unavailable", "artifact_identity_changed",
                "artifact_metadata_changed", "unexpected_score_cache_state", "score_cache_changed_during_idle",
                "worker_not_ready", "worker_request_failed", "worker_closed", "worker_timeout"}
        code = str(error) if str(error) in safe else "idle_benchmark_failed"
        print(json.dumps({"passed": False, "stage": code, "error_type": type(error).__name__,
                          "errno": getattr(error, "errno", None)}))
        return 1
    print(json.dumps({"passed": result["passed"], "eligible": result["all_host_screens_eligible"], "summary": result["summary"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
