"""Publish the matched MTP repeat using v1 only as an immutable metadata template."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent
RUNTIME = ARCHIVE.parents[3] / "runtime/gdn-next-gpu-20261031/mtp"
INITIAL = ARCHIVE / "results.json"
TARGET = ARCHIVE / "mtp-repeat-results.json"
PARENT = ARCHIVE / "publish_results.py"
FIXED = {"schema": "qwen38-gdn-next-publish-repeat-failure-v1", "completed": False,
         "failure_category": "repeat_export_validation_or_io_failure"}
IMMUTABLE = ("schema", "driver_sha256", "source_base_commit", "checkpoint_manifest_sha256",
             "checkpoint_payload_receipt_sha256", "checkpoint_stat_snapshot_sha256", "corpus_sha256",
             "input_token_sha256", "support_code_sha256", "source_and_worker_seals", "block_order",
             "calls_per_block", "warmups_per_block", "input_tokens", "output_tokens", "sampling",
             "speculation", "mtp_enabled_for_all_arms", "mtp_drafts", "mtp_sampling", "mtp_verification",
             "prefix_reuse", "stored_weights_or_quantization_changed",
             "qualification_scope", "timing_note", "uncertainty", "checkpoint_payloads_rehashed_this_run",
             "historical_mtp_output_token_sha256")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_parent():
    spec = importlib.util.spec_from_file_location("gdn_publisher", PARENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def monitor_summary(value):
    py = value.get("observed_small_or_vanished_python", [])
    builds = value.get("observed_cpu_builds", [])
    compete = value.get("observations", [])
    records = [{"category": o.get("category", "unclassified"),
                "inspection_status": o.get("inspection_status", "not_applicable"),
                "mapping_count": o.get("txt_mapping_count", 0),
                "gpu_marker_count": len(o.get("gpu_or_model_mapping_markers", []))
                if isinstance(o.get("gpu_or_model_mapping_markers", []), list) else 0}
               for o in py + builds + compete]
    counts = {"polls": int(value.get("polls", 0)), "observed_python": len(py),
              "observed_cpu_builds": len(builds), "competing_observations": len(compete),
              "mapped_python_records": sum(o.get("inspection_status") == "inspected" for o in py),
              "python_with_gpu_or_model_markers": sum(bool(o.get("gpu_or_model_mapping_markers")) for o in py),
              "vanished_python_records": sum(o.get("inspection_status") in
                  ("vanished_before_inspection", "vanished_before_mapping_inspection") for o in py)}
    identity = hashlib.sha256(json.dumps({"semantics": value.get("monitor_semantics"), "counts": counts,
                                          "observations": records}, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    return {**counts, "failed": False, "monitor_semantics": value.get("monitor_semantics"),
            "identity_sha256": identity, "auxiliary_manifest_sha256": value.get("reviewed_auxiliary_manifest_sha256")}


def run():
    if TARGET.exists():
        raise RuntimeError("repeat_output_exists")
    public = read(INITIAL)
    public_v1 = public["model_trials"]["mtp"]
    private_v1 = read(RUNTIME / "model-trial-v1.json")
    trial = read(RUNTIME / "model-trial-v2.json")
    parent = load_parent()
    if public["exporter_sha256"] != sha(PARENT) or public_v1["trial_sha256"] != sha(RUNTIME / "model-trial-v1.json"):
        raise RuntimeError("initial_public_binding_changed")
    if trial.get("completed") is not True or trial.get("all_arms_qualified") is not True:
        raise RuntimeError("repeat_trial_incomplete")
    if trial.get("quiet_window", {}).get("failed") is not False:
        raise RuntimeError("repeat_monitor_failed")
    for key in IMMUTABLE:
        if trial.get(key) != private_v1.get(key):
            raise RuntimeError("repeat_immutable_trial_mismatch")
        prior_value = parent.safe_seals(private_v1[key]) if key == "source_and_worker_seals" else private_v1.get(key)
        if key in public_v1 and public_v1.get(key) != prior_value:
            raise RuntimeError("public_v1_template_mismatch")
    if trial.get("block_order") != parent.ARM_ORDERS["mtp"] or trial.get("calls_per_block") != 4 or trial.get("warmups_per_block") != 1:
        raise RuntimeError("repeat_protocol_mismatch")
    rows = trial.get("rows")
    blocks = trial.get("blocks")
    if not isinstance(rows, list) or len(rows) != 16 or not isinstance(blocks, list) or len(blocks) != 4:
        raise RuntimeError("repeat_row_count")
    golden = trial.get("historical_mtp_output_token_sha256")
    safe_rows = []
    per_block = [0, 0, 0, 0]
    for index, row in enumerate(rows):
        block_index = row.get("block")
        if type(block_index) is not int or not 0 <= block_index < 4 or block_index != index // 4:
            raise RuntimeError("repeat_row_order")
        replay = per_block[block_index]
        per_block[block_index] += 1
        if row.get("arm") != trial["block_order"][block_index] or row.get("replay") != replay or row.get("warmup") is not (replay == 0):
            raise RuntimeError("repeat_row_order")
        if row.get("status") != "complete" or row.get("historical_output_exact") is not True or row.get("output_token_sha256") != golden:
            raise RuntimeError("repeat_row_parity")
        stats = row.get("mtp")
        timings = stats.get("timings_seconds") if isinstance(stats, dict) else None
        if row.get("mtp_enabled") is not True or not isinstance(stats, dict) or stats.get("enabled") is not True or stats.get("drafts") != 2 or stats.get("sampling") != "exact-rejection-v1" or stats.get("verification") != "batched-affine-decode-exact-v1":
            raise RuntimeError("repeat_mtp_route")
        if not isinstance(timings, dict) or set(timings) != {"draft_seconds", "verify_seconds", "rollback_seconds"}:
            raise RuntimeError("repeat_mtp_timing_schema")
        counters = ("rounds", "proposed", "accepted", "verified_rows")
        if any(type(stats.get(k)) is not int or stats[k] < 0 for k in counters) or stats["rounds"] == 0 or stats["proposed"] == 0 or stats["accepted"] > stats["proposed"] or stats["verified_rows"] < stats["accepted"]:
            raise RuntimeError("repeat_mtp_counters")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in timings.values()):
            raise RuntimeError("repeat_mtp_timings")
        item = {k: row[k] for k in ("block", "arm", "replay", "warmup", "status", "historical_output_exact", "output_token_sha256", "input_tokens", "output_tokens", "useful_decode_tokens", "prefill_seconds", "decode_seconds", "wall_seconds", "decode_tokens_per_second", "mtp_enabled")}
        item["mtp"] = {k: stats[k] for k in (*counters, "enabled", "drafts", "sampling", "verification")}
        item["mtp"]["timings_seconds"] = {k: timings[k] for k in ("draft_seconds", "verify_seconds", "rollback_seconds")}
        item["sequence_index"] = index
        if (item["input_tokens"], item["output_tokens"], item["useful_decode_tokens"]) != (4096, 256, 255):
            raise RuntimeError("repeat_token_counts")
        if any(type(item[k]) not in (int, float) or not math.isfinite(item[k]) or item[k] <= 0
               for k in ("prefill_seconds", "decode_seconds", "wall_seconds", "decode_tokens_per_second")):
            raise RuntimeError("repeat_duration")
        if item["decode_tokens_per_second"] != 255 / item["decode_seconds"]:
            raise RuntimeError("repeat_rate_arithmetic")
        if any(stats[k] != private_v1["rows"][index]["mtp"][k] for k in counters):
            raise RuntimeError("repeat_per_request_counter_drift")
        safe_rows.append(item)
    safe_blocks = []
    for index, block in enumerate(blocks):
        if block.get("block") != index or block.get("arm") != trial["block_order"][index] or block.get("status") != "complete" or block.get("before_seals_verified") is not True or block.get("after_seals_verified") is not True:
            raise RuntimeError("repeat_block_unsealed")
        safe_blocks.append({k: block[k] for k in ("block", "arm", "status", "before_seals_verified", "after_seals_verified", "ready_policy_sha256", "precision_metadata_exposed") if k in block})
        if block["mtp_readiness"] != private_v1["blocks"][index]["mtp_readiness"]:
            raise RuntimeError("repeat_readiness_changed")
        safe_blocks[-1]["mtp_readiness"] = {key: block["mtp_readiness"][key] for key in
            ("supported", "sampling", "verification", "max_drafts", "default_drafts")}
    summaries, comparisons = parent.summarize_rows(safe_rows, ["baseline", "q8reuse"], safe_blocks)
    for arm in ("baseline", "q8reuse"):
        prior_totals = public_v1["summary"][arm]["mtp_totals"]
        if summaries[arm].get("mtp_totals") != prior_totals:
            raise RuntimeError("repeat_mtp_counter_drift")
    result = {k: public_v1[k] for k in IMMUTABLE if k in public_v1}
    result["context_limit_tokens"] = public_v1["context_limit_tokens"]
    result["historical_output_token_sha256"] = golden
    result.update({"trial_sha256": sha(RUNTIME / "model-trial-v2.json"),
                   "started_utc": trial.get("started_utc"), "finished_utc": trial.get("finished_utc"),
                   "completed": True, "all_arms_qualified": True,
                   "rows": safe_rows, "blocks": safe_blocks, "summary": summaries,
                   "comparisons": comparisons, "monitor_aggregate": monitor_summary(trial["quiet_window"])})
    parent_bytes = PARENT.read_bytes()
    result["export_provenance"] = {"repeat_exporter_sha256": sha(Path(__file__)),
                                   "parent_exporter_sha256": hashlib.sha256(parent_bytes).hexdigest(),
                                   "initial_public_results_sha256": sha(INITIAL)}
    envelope = {"schema": "qwen38-gdn-next-mtp-repeat-results-v1", "host": public_v1["host"], "mtp_repeat": result}
    payload = (json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(TARGET, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def failure_main():
    try:
        run()
    except (OSError, RuntimeError, TypeError, KeyError, ValueError, json.JSONDecodeError):
        try:
            fd = os.open(RUNTIME / "mtp-repeat-export-failure.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(FIXED, stream, sort_keys=True)
                stream.write("\n")
        except OSError:
            pass
        print(FIXED["failure_category"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    failure_main()
