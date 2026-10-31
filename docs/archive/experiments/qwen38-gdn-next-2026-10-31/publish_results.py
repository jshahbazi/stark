"""Export approved aggregate fields from private GDN trial receipts.

This exporter never copies private worker, monitor, path, or raw-output data.
It refuses partial model trials and creates results.json exclusively.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent
RUNTIME = ARCHIVE.parents[3] / "runtime/gdn-next-gpu-20261031"


def HASH(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def clean_metric(value):
    if not isinstance(value, str) or len(value) > 1000:
        raise RuntimeError("invalid_metric_line")
    lowered = value.lower()
    if any(term in lowered for term in ("/users/", "/private/", "pid", "prompt", "token_ids", "logits", "probabilities")):
        raise RuntimeError("unsafe_metric_line")
    return value


def controls():
    result = []
    names = ("correctness-v1.json", "correctness-v2.json", "components-v1.json", "components-v2.json")
    for name in names:
        src = RUNTIME / name
        report = read(src)
        if report.get("completed") is not True or not isinstance(report.get("rows"), list):
            raise RuntimeError("incomplete_controls")
        rows = []
        for row in report["rows"]:
            if not all(HASH(row.get(key)) for key in ("binary_sha256", "harness_sha256", "raw_output_sha256", "sanitized_log_sha256", "shader_sha256")):
                raise RuntimeError("invalid_control_hash")
            item = {key: row[key] for key in ("suite", "test", "status", "exit_code", "started_utc", "completed_utc", "elapsed_seconds", "binary_sha256", "harness_sha256", "shader_sha256", "raw_output_sha256", "sanitized_log_sha256") if key in row}
            item["metrics"] = [clean_metric(m) for m in row.get("metrics", [])]
            if row.get("status") == "failed":
                item["failure_category"] = ("pipeline_cache_capture_collision"
                    if name == "components-v1.json" and row.get("suite") == "q8conv"
                    else "recorded_control_failure")
            for key in ("native_mismatch_counts", "native_reported_nonfinite"):
                if key in row:
                    item[key] = row[key]
            rows.append(item)
        result.append({"receipt": name, "receipt_sha256": digest(src), "runner_sha256": report.get("runner_sha256"), "phase": report.get("phase"), "rows": rows})
    return result


ROW_FIELDS = ("block", "arm", "replay", "warmup", "status", "historical_output_exact", "output_token_sha256", "input_tokens", "output_tokens", "useful_decode_tokens", "prefill_seconds", "decode_seconds", "wall_seconds", "decode_tokens_per_second", "mtp_enabled", "mtp")
ARM_ORDERS = {"single": ["baseline", "q8conv", "fused1024", "fused1024", "q8conv", "baseline"],
              "mtp": ["baseline", "q8reuse", "q8reuse", "baseline"]}


def aggregate(rows):
    rates = [r["decode_tokens_per_second"] for r in rows]
    return {"measured_requests": len(rows),
            "pooled_decode_tokens_per_second": sum(r["useful_decode_tokens"] for r in rows) / sum(r["decode_seconds"] for r in rows),
            "median_decode_tokens_per_second": statistics.median(rates),
            "minimum_decode_tokens_per_second": min(rates), "maximum_decode_tokens_per_second": max(rates),
            "request_decode_tokens_per_second": rates,
            "pooled_prefill_tokens_per_second": sum(r["input_tokens"] for r in rows) / sum(r["prefill_seconds"] for r in rows),
            "median_wall_seconds": statistics.median(r["wall_seconds"] for r in rows)}


def safe_seals(seals):
    if not HASH(seals.get("preparation_sha256")):
        raise RuntimeError("invalid_preparation_seal")
    arms = {}
    for arm, v in seals.get("arms", {}).items():
        keep = ("worker_sha256", "build_receipt_sha256", "source_manifest_sha256", "source_tree_manifest_sha256", "source_file_count")
        if not all(HASH(v.get(k)) for k in keep[:4]) or type(v.get("source_file_count")) is not int:
            raise RuntimeError("invalid_source_seal")
        arms[arm] = {k: v[k] for k in keep}
    return {"preparation_sha256": seals.get("preparation_sha256"), "arms": arms}


def summarize_rows(rows, arms, blocks):
    summaries = {}
    for arm in arms:
        calls = [r for r in rows if r["arm"] == arm]
        measured = [r for r in calls if not r["warmup"]]
        s = {"qualified": True, "attempted_calls": len(calls), "completed_calls": len(calls), **aggregate(measured)}
        block_sums = []
        for number in sorted({r["block"] for r in measured}):
            b = {"block": number, **aggregate([r for r in measured if r["block"] == number])}
            block_sums.append(b)
        s["blocks"] = block_sums
        if len(block_sums) == 2:
            s["later_vs_earlier_block_pooled_percent"] = (block_sums[1]["pooled_decode_tokens_per_second"] / block_sums[0]["pooled_decode_tokens_per_second"] - 1) * 100
        if any("mtp" in r for r in measured):
            s["mtp_totals"] = {k: sum(r["mtp"][k] for r in measured) for k in ("rounds", "proposed", "accepted", "verified_rows")}
            s["mtp_timing_totals_seconds"] = {k: sum(r["mtp"]["timings_seconds"][k] for r in measured) for k in ("draft_seconds", "verify_seconds", "rollback_seconds")}
        summaries[arm] = s
    comparisons = {}
    baseline = summaries["baseline"]["pooled_decode_tokens_per_second"]
    for arm in arms:
        if arm != "baseline":
            comparisons[arm] = {"pooled_decode_change_percent": (summaries[arm]["pooled_decode_tokens_per_second"] / baseline - 1) * 100,
                                "scope": "six measured requests per arm; descriptive, order/drift remain"}
    return summaries, comparisons


def model_trial(arm):
    path = RUNTIME / arm / "model-trial-v1.json"
    d = read(path)
    if d.get("completed") is not True or d.get("all_arms_qualified") is not True:
        raise RuntimeError("model_trial_incomplete")
    if d.get("quiet_window", {}).get("failed") is not False:
        raise RuntimeError("model_trial_monitor_failed")
    block_order = d.get("block_order")
    rows = d.get("rows")
    blocks = d.get("blocks")
    if not isinstance(block_order, list) or not isinstance(rows, list) or not isinstance(blocks, list):
        raise TypeError("model_trial_shape")
    if block_order != ARM_ORDERS[arm] or d.get("calls_per_block") != 4 or d.get("warmups_per_block") != 1:
        raise RuntimeError("model_trial_order_policy")
    expected_n = len(block_order) * d.get("calls_per_block", 0)
    if len(rows) != expected_n or len(blocks) != len(block_order):
        raise RuntimeError("model_trial_row_count")
    out_rows = []
    per_block = [0 for _ in block_order]
    for i, row in enumerate(rows):
        if row.get("status") != "complete" or row.get("historical_output_exact") is not True or not HASH(row.get("output_token_sha256")) or row.get("output_token_sha256") != d.get("historical_output_token_sha256", d.get("historical_mtp_output_token_sha256")):
            raise RuntimeError("model_trial_row_unqualified")
        safe = {key: row[key] for key in ROW_FIELDS if key in row}
        block_index = row.get("block")
        if not isinstance(block_index, int) or not 0 <= block_index < len(block_order):
            raise RuntimeError("model_trial_block_order")
        expected_arm = block_order[block_index]
        replay = per_block[block_index]
        per_block[block_index] += 1
        if block_index != i // 4 or row.get("arm") != expected_arm or row.get("replay") != replay or row.get("warmup") is not (replay == 0):
            raise RuntimeError("model_trial_order_mismatch")
        if "mtp" in safe:
            mtp = safe["mtp"]
            timings = mtp.get("timings_seconds", {})
            if set(timings) != {"draft_seconds", "verify_seconds", "rollback_seconds"}:
                raise RuntimeError("mtp_timing_schema")
            if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in timings.values()):
                raise RuntimeError("mtp_timing_values")
            safe["mtp"] = {k: mtp[k] for k in ("enabled", "drafts", "sampling", "verification", "rounds", "proposed", "accepted", "verified_rows") if k in mtp}
            safe["mtp"]["timings_seconds"] = {k: timings[k] for k in ("draft_seconds", "verify_seconds", "rollback_seconds")}
        safe["sequence_index"] = i
        for k in ("prefill_seconds", "decode_seconds", "wall_seconds", "decode_tokens_per_second"):
            if type(safe.get(k)) not in (int, float) or not math.isfinite(safe[k]) or safe[k] <= 0:
                raise RuntimeError("invalid_model_timing")
        if safe["decode_tokens_per_second"] != safe["useful_decode_tokens"] / safe["decode_seconds"]:
            raise RuntimeError("decode_rate_mismatch")
        out_rows.append(safe)
    if any(count != d.get("calls_per_block") for count in per_block):
        raise RuntimeError("model_trial_block_call_count")
    safe_blocks = []
    for i, block in enumerate(blocks):
        if block.get("block") != i or block.get("arm") != block_order[i] or block.get("status") != "complete" or block.get("before_seals_verified") is not True or block.get("after_seals_verified") is not True:
            raise RuntimeError("model_trial_block_unsealed")
        safe_blocks.append({k: block[k] for k in ("block", "arm", "status", "before_seals_verified", "after_seals_verified", "ready_policy_sha256", "precision_metadata_exposed") if k in block})
        if arm == "mtp":
            readiness = block["mtp_readiness"]
            safe_blocks[-1]["mtp_readiness"] = {
                key: readiness[key] for key in
                ("supported", "sampling", "verification", "max_drafts", "default_drafts")
            }
    seals = d.get("source_and_worker_seals", {})
    arms = sorted(set(block_order))
    summaries, comparisons = summarize_rows(out_rows, arms, safe_blocks)
    monitor = d["quiet_window"]
    observations = monitor.get("observations", [])
    observed_python = monitor.get("observed_small_or_vanished_python", [])
    observed_builds = monitor.get("observed_cpu_builds", [])
    public_observations = [{"category": o.get("category", "unclassified"), "inspection_status": o.get("inspection_status", "not_applicable"),
                            "mapping_count": o.get("txt_mapping_count", 0),
                            "gpu_marker_count": len(o.get("gpu_or_model_mapping_markers", [])) if isinstance(o.get("gpu_or_model_mapping_markers", []), list) else 0}
                           for o in observed_python + observed_builds + observations]
    monitor_counts = {"polls": int(monitor.get("polls", 0)), "observed_python": len(observed_python),
                      "observed_cpu_builds": len(observed_builds), "competing_observations": len(observations),
                      "mapped_python_records": sum(o.get("inspection_status") == "inspected" for o in observed_python),
                      "python_with_gpu_or_model_markers": sum(bool(o.get("gpu_or_model_mapping_markers")) for o in observed_python),
                      "vanished_python_records": sum(o.get("inspection_status") in ("vanished_before_inspection", "vanished_before_mapping_inspection") for o in observed_python)}
    monitor_hash = hashlib.sha256(json.dumps({"monitor_semantics": monitor.get("monitor_semantics"), "counts": monitor_counts,
                                              "observations": public_observations}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": d["schema"], "trial_sha256": digest(path), "driver_sha256": d.get("driver_sha256"),
        "source_base_commit": d.get("source_base_commit"), "checkpoint_manifest_sha256": d.get("checkpoint_manifest_sha256"),
        "checkpoint_payload_receipt_sha256": d.get("checkpoint_payload_receipt_sha256"), "checkpoint_stat_snapshot_sha256": d.get("checkpoint_stat_snapshot_sha256"),
        "corpus_sha256": d.get("corpus_sha256"), "input_token_sha256": d.get("input_token_sha256"),
        "support_code_sha256": d.get("support_code_sha256"), "source_and_worker_seals": safe_seals(seals),
        "started_utc": d.get("started_utc"), "finished_utc": d.get("finished_utc"),
        "completed": True, "all_arms_qualified": True,
        "block_order": block_order, "calls_per_block": d.get("calls_per_block"), "warmups_per_block": d.get("warmups_per_block"),
        "input_tokens": d.get("input_tokens"), "output_tokens": d.get("output_tokens"),
        "sampling": {k: d["sampling"][k] for k in ("temperature", "top_k", "top_p", "seed")},
        "speculation": d.get("speculation"), "mtp_enabled_for_all_arms": d.get("mtp_enabled_for_all_arms"),
        "mtp_drafts": d.get("mtp_drafts"), "mtp_sampling": d.get("mtp_sampling"), "mtp_verification": d.get("mtp_verification"),
        "context_limit_tokens": 4352, "prefix_reuse": d.get("prefix_reuse"),
        "historical_output_token_sha256": d.get("historical_output_token_sha256", d.get("historical_mtp_output_token_sha256")),
        "checkpoint_payloads_rehashed_this_run": d.get("checkpoint_payloads_rehashed_this_run"),
        "stored_weights_or_quantization_changed": d.get("stored_weights_or_quantization_changed"),
        "qualification_scope": d.get("qualification_scope"), "timing_note": d.get("timing_note"), "uncertainty": d.get("uncertainty"),
        "host": {"model": "Mac17,7", "chip": "Apple M5 Max", "memory_bytes": 137438953472, "macos": "26.7.1"},
        "rows": out_rows, "blocks": safe_blocks, "summary": summaries, "comparisons": comparisons,
        "monitor_aggregate": {**monitor_counts, "failed": False, "monitor_semantics": monitor.get("monitor_semantics"),
                              "identity_sha256": monitor_hash, "auxiliary_manifest_sha256": d.get("reviewed_auxiliary_manifest_sha256")},
    }


def main():
    target = ARCHIVE / "results.json"
    if target.exists():
        raise FileExistsError("results.json_exists")
    data = {"schema": "qwen38-gdn-next-published-results-v1", "exporter_sha256": digest(Path(__file__)),
            "controls": controls(), "model_trials": {}}
    for arm in ("single", "mtp"):
        path = RUNTIME / arm / "model-trial-v1.json"
        if not path.is_file():
            raise RuntimeError("required_model_trial_missing")
        data["model_trials"][arm] = model_trial(arm)
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def failure_main():
    try:
        main()
    except (OSError, RuntimeError, TypeError, KeyError, ValueError, json.JSONDecodeError):
        failure = {"schema": "qwen38-gdn-next-publish-failure-v1", "completed": False,
                   "failure_category": "publisher_validation_or_io_failure"}
        target = RUNTIME / "results-failure.json"
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(failure, stream, sort_keys=True)
                stream.write("\n")
        except OSError:
            pass
        print("publisher_validation_or_io_failure", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    failure_main()
