"""Aggregate the private Lily top-k policy run without exposing content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "runtime/lily-topk-policy/performance-private"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float:
    if len(values) != 8:
        raise ValueError("paired_comparison_count_mismatch")
    return statistics.median(values)


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    center = statistics.median(values)
    return {
        "n": len(values),
        "median": center,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mad": statistics.median(abs(value - center) for value in values),
    }


def work_tuple(row: dict) -> tuple:
    mtp = row["mtp"]
    return tuple(mtp[key] for key in ("rounds", "proposed", "accepted", "verified_rows"))


def sanitized_request(row: dict, input_sha256: str | None = None) -> dict:
    """Copy only numeric metrics, identifiers, MTP counts, and hashes."""
    allowed = (
        "session", "build", "cell", "role", "ordinal", "actual_drafts",
        "sampled_tokens", "output_tokens", "output_sha256", "decode_seconds",
        "generation_seconds", "prefill_seconds", "useful_decode_tokens_per_second",
        "mtp_selector", "input_tokens", "context_tokens", "output_budget", "seed",
        "wall_seconds",
    )
    result = {key: row[key] for key in allowed if key in row}
    if input_sha256 is not None:
        result["input_sha256"] = input_sha256
    if "mtp" in row:
        mtp = row["mtp"]
        result["mtp"] = {
            key: mtp[key] for key in (
                "rounds", "proposed", "accepted", "verified_rows",
                "timings_seconds", "verification", "sampling",
            ) if key in mtp
        }
    if "host" in row:
        host = row["host"]
        result["host"] = {
            key: host[key] for key in (
                "samples", "observation_missing", "external_cpu_max_percent",
                "max_process_cpu_percent", "max_model_cpu_percent",
                "thermal_max_state", "sampling_gap_exceeded",
                "monitor_failures_so_far", "paging_delta",
            ) if key in host
        }
    return result


def summarize_build_cell(sessions: list[dict], build: str, cell_name: str) -> dict:
    selected = [cell for session in sessions if session["build"] == build
                for cell in session["cells"] if cell["name"] == cell_name]
    measured = [row for cell in selected for row in cell["measurements"]]
    warmups = [cell["warmup"] for cell in selected]
    if len(measured) != 8 or len(warmups) != 2:
        raise ValueError("build_cell_summary_count_mismatch")
    depths = {row["actual_drafts"] for row in measured + warmups}
    input_hashes = {cell["input_sha256"] for cell in selected}
    if len(depths) != 1:
        raise ValueError("build_cell_depth_changed")
    if len(input_hashes) != 1:
        raise ValueError("build_cell_input_changed")
    repeated = measured + warmups
    hashes = {row["output_sha256"] for row in repeated}
    counters = {work_tuple(row) for row in repeated}
    if len(hashes) != 1 or len(counters) != 1:
        raise ValueError("within_depth_repeat_mismatch")
    decodes = [row["decode_seconds"] for row in measured]
    useful_tokens = sum(row["output_tokens"] - 1 for row in measured)
    pooled_decode = useful_tokens / sum(decodes)
    metrics = {}
    for key in ("wall_seconds", "decode_seconds", "generation_seconds",
                "prefill_seconds", "useful_decode_tokens_per_second"):
        metrics[key] = describe([row[key] for row in measured if row.get(key) is not None])
    return {
        "build": build,
        "cell": cell_name,
        "input_sha256": next(iter(input_hashes)),
        "measured_n": len(measured),
        "warmup_n": len(warmups),
        "actual_drafts": next(iter(depths)),
        "metrics": metrics,
        "pooled_useful_decode_tokens_per_second": pooled_decode,
        "pooled_useful_decode_tokens": useful_tokens,
        "pooled_decode_seconds": sum(decodes),
        "mtp_work_counters": dict(zip(
            ("rounds", "proposed", "accepted", "verified_rows"), next(iter(counters))
        )),
        "output_sha256": next(iter(hashes)),
        "repeat_assertions": {
            "same_depth_for_warmups_and_measurements": True,
            "output_hash_unchanged": True,
            "mtp_work_counters_unchanged": True,
        },
    }


def load_complete() -> tuple[dict, list[dict]]:
    manifest = json.loads((PRIVATE / "manifest.json").read_text())
    if manifest.get("completed") is not True or len(manifest.get("sessions", [])) != 4:
        raise ValueError("run_incomplete")
    sessions = []
    for number, expected_build in enumerate(("before", "candidate", "candidate", "before"), start=1):
        session = json.loads((PRIVATE / f"session-{number}.json").read_text())
        if manifest.get("session_file_sha256", {}).get(str(number)) != sha(PRIVATE / f"session-{number}.json"):
            raise ValueError("session_receipt_hash_mismatch")
        if session.get("session") != number or session.get("build") != expected_build:
            raise ValueError("session_order_mismatch")
        if session.get("cell_order") != manifest["session_cell_orders"][number - 1]:
            raise ValueError("session_cell_order_mismatch")
        if session != manifest["sessions"][number - 1]:
            raise ValueError("session_manifest_mismatch")
        if len(session.get("cells", [])) != 4 or session.get("setup") is None:
            raise ValueError("session_content_mismatch")
        cells = {cell["name"]: cell for cell in session["cells"]}
        if len(cells) != 4:
            raise ValueError("cell_identity_mismatch")
        for cell in cells.values():
            if cell.get("warmup", {}).get("role") != "fixed_cell_warmup":
                raise ValueError("warmup_missing")
            rows = cell.get("measurements", [])
            if len(rows) != 4 or [row.get("ordinal") for row in rows] != [1, 2, 3, 4]:
                raise ValueError("measured_request_count_mismatch")
            if any(row.get("role") != "measured" for row in rows):
                raise ValueError("measurement_role_mismatch")
        sessions.append(session)
    return manifest, sessions


def contrast(manifest: dict, sessions: list[dict], cell_name: str) -> dict:
    # ABBA positional pairing: B1/C2 and C3/B4. Replicate 1..4 is paired
    # within the corresponding adjacent sessions; no medians are used to form
    # the decision statistic.
    pairs = []
    for before_index, candidate_index in ((0, 1), (3, 2)):
        before_cell = next(c for c in sessions[before_index]["cells"] if c["name"] == cell_name)
        candidate_cell = next(c for c in sessions[candidate_index]["cells"] if c["name"] == cell_name)
        if before_cell["input_sha256"] != candidate_cell["input_sha256"]:
            raise ValueError("paired_input_mismatch")
        before_rows = before_cell["measurements"]
        candidate_rows = candidate_cell["measurements"]
        for position, (before, candidate) in enumerate(zip(before_rows, candidate_rows), start=1):
            if before["ordinal"] != candidate["ordinal"] or before["cell"] != candidate["cell"]:
                raise ValueError("pair_position_mismatch")
            for row in (before, candidate):
                reference = manifest["reference_output_sha256"][cell_name][str(row["actual_drafts"])]
                if row["output_sha256"] != reference:
                    raise ValueError("per_depth_reference_output_mismatch")
            if (before["actual_drafts"] == candidate["actual_drafts"]
                    and before["output_sha256"] != candidate["output_sha256"]):
                raise ValueError("same_depth_cross_build_output_mismatch")
            pairs.append({
                "before_session": before["session"], "candidate_session": candidate["session"],
                "replicate_position": position,
                "before_wall_seconds": before["wall_seconds"],
                "candidate_wall_seconds": candidate["wall_seconds"],
                "candidate_vs_before_wall_change": candidate["wall_seconds"] / before["wall_seconds"] - 1,
                "before_useful_decode_tokens_per_second": before["useful_decode_tokens_per_second"],
                "candidate_useful_decode_tokens_per_second": candidate["useful_decode_tokens_per_second"],
                "candidate_vs_before_decode_rate_change": candidate["useful_decode_tokens_per_second"] / before["useful_decode_tokens_per_second"] - 1,
                "before_drafts": before["actual_drafts"], "candidate_drafts": candidate["actual_drafts"],
            })
    if len(pairs) != 8:
        raise ValueError("paired_comparison_count_mismatch")
    wall = [pair["candidate_vs_before_wall_change"] for pair in pairs]
    decode = [pair["candidate_vs_before_decode_rate_change"] for pair in pairs]
    split_medians = [statistics.median(wall[:4]), statistics.median(wall[4:])]
    return {
        "pair_count": len(pairs),
        "paired_candidate_vs_before_wall_change_median": median(wall),
        "paired_wall_reduction_percent_median": -100 * median(wall),
        "paired_wall_wins": sum(value < 0 for value in wall),
        "paired_candidate_vs_before_decode_rate_change_median": median(decode),
        "paired_decode_rate_wins": sum(value > 0 for value in decode),
        "abba_half_paired_wall_change_medians": split_medians,
        "cross_session_timing_spread": abs(split_medians[0] - split_medians[1]),
        "cross_session_timing_consistent": (
            split_medians[0] < 0 and split_medians[1] < 0
            and abs(split_medians[0] - split_medians[1]) <= 0.10
        ),
        "pairs": pairs,
    }


def main() -> int:
    manifest, sessions = load_complete()
    cell_names = [cell["name"] for cell in manifest["cell_definitions"]]
    if len(cell_names) != 4 or [s["build"] for s in sessions] != ["before", "candidate", "candidate", "before"]:
        raise ValueError("fixed_matrix_mismatch")
    all_rows = [row for session in sessions for cell in session["cells"]
                for row in cell["measurements"]]
    all_warmups = [cell["warmup"] for session in sessions for cell in session["cells"]]
    setups = [session["setup"] for session in sessions]
    if len(all_rows) != 64 or len(all_warmups) != 16 or len(setups) != 4:
        raise ValueError("planned_request_count_mismatch")
    if any(row["role"] != "measured" for row in all_rows):
        raise ValueError("measurement_role_mismatch")
    for session in sessions:
        build = session["build"]
        setup = session["setup"]
        if (setup.get("role") != "separate_max_context_setup"
                or setup.get("input_tokens") != 4096
                or setup.get("context_tokens") != 4352
                or setup.get("output_budget") != 2):
            raise ValueError("max_context_setup_mismatch")
        for cell in session["cells"]:
            kind = next(item["kind"] for item in manifest["cell_definitions"]
                        if item["name"] == cell["name"])
            expected_depth = 2 if build == "before" or kind != "top20" else 1
            for row in [cell["warmup"], *cell["measurements"]]:
                if row.get("mtp_selector") != "auto" or row.get("actual_drafts") != expected_depth:
                    raise ValueError("auto_depth_matrix_mismatch")
                if row.get("output_sha256") != manifest["reference_output_sha256"][cell["name"]][str(expected_depth)]:
                    raise ValueError("per_depth_reference_output_mismatch")
        if setup.get("mtp_selector") != "auto" or setup.get("actual_drafts") != 2:
            raise ValueError("setup_depth_mismatch")
    comparisons = {name: contrast(manifest, sessions, name) for name in cell_names}
    build_cell_summaries = {
        build: {cell_name: summarize_build_cell(sessions, build, cell_name)
                for cell_name in cell_names}
        for build in ("before", "candidate")
    }
    guard_matches = {}
    for cell_name in (name for name in cell_names if name.startswith("guard-")):
        before = build_cell_summaries["before"][cell_name]
        candidate = build_cell_summaries["candidate"][cell_name]
        guard_matches[cell_name] = {
            "before_depth": before["actual_drafts"],
            "candidate_depth": candidate["actual_drafts"],
            "both_depth_two": before["actual_drafts"] == candidate["actual_drafts"] == 2,
            "output_sha256_matches": before["output_sha256"] == candidate["output_sha256"],
            "mtp_work_counters_match": before["mtp_work_counters"] == candidate["mtp_work_counters"],
        }
        guard_matches[cell_name]["passed"] = all((
            guard_matches[cell_name]["both_depth_two"],
            guard_matches[cell_name]["output_sha256_matches"],
            guard_matches[cell_name]["mtp_work_counters_match"],
        ))
    primary = next(name for name in cell_names if name.startswith("primary-"))
    heldout = next(name for name in cell_names if name.startswith("heldout-"))
    observed_thermal_states = [row["host"]["thermal_max_state"] for row in all_rows
                               if "thermal_max_state" in row["host"]]
    observed_external_cpu = [row["host"]["external_cpu_max_percent"] for row in all_rows
                             if "external_cpu_max_percent" in row["host"]]
    observed_process_cpu = [row["host"]["max_process_cpu_percent"] for row in all_rows
                            if "max_process_cpu_percent" in row["host"]]
    observed_model_cpu = [row["host"]["max_model_cpu_percent"] for row in all_rows
                          if "max_model_cpu_percent" in row["host"]]
    primary_direction = comparisons[primary]["paired_candidate_vs_before_wall_change_median"] < 0
    held = comparisons[heldout]
    checks = {
        "heldout_paired_wall_reduction_at_least_10_percent":
            held["paired_candidate_vs_before_wall_change_median"] <= -0.10,
        "heldout_at_least_six_of_eight_wall_wins": held["paired_wall_wins"] >= 6,
        "heldout_cross_session_timing_consistent": held["cross_session_timing_consistent"],
        "primary_short_case_direction_positive": primary_direction,
        "depth2_guards_match_output_and_work": all(row["passed"] for row in guard_matches.values()),
    }
    # A small and cross-session-noisy sample can only qualify the declared
    # policy endpoint. It does not support chasing another favorable run.
    report = {
        "schema": "stark-lily-topk-policy-performance-aggregate-v1",
        "attempt": manifest["attempt"],
        "prior_preflight_attempt": manifest["prior_preflight_attempt"],
        "manifest_sha256": sha(PRIVATE / "manifest.json"),
        "publication_harness_sha256": sha(Path(__file__)),
        "worker_sha256": manifest["worker_sha256"],
        "build_identity_sha256": manifest["build_identity_sha256"],
        "native_patch_sha256": manifest["native_patch_sha256"],
        "python_patch_sha256": manifest["python_patch_sha256"],
        "reference_file_sha256": manifest["reference_file_sha256"],
        "reference_output_sha256": manifest["reference_output_sha256"],
        "reference_input_sha256": manifest["reference_input_sha256"],
        "tool_sha256": manifest["tool_sha256"],
        "source_commit": manifest["source_commit"],
        "runtime": manifest["runtime"],
        "model_content_manifest_sha256": manifest["model_content_manifest_sha256"],
        "model_extra_file_identity": manifest["model_extra_file_identity"],
        "harness_sha256": manifest["harness_sha256"],
        "corpora_sha256": manifest["corpora_sha256"],
        "cell_definitions": manifest["cell_definitions"],
        "session_build_order": manifest["session_build_order"],
        "session_cell_orders": manifest["session_cell_orders"],
        "request_counts": {"measured": len(all_rows), "cell_warmups": len(all_warmups),
                            "warmmax4352_setups": len(setups)},
        "actual_draft_counts": {
            build: {cell_name: {str(depth): sum(row["actual_drafts"] == depth
                for session in sessions if session["build"] == build
                for cell in session["cells"] if cell["name"] == cell_name
                for row in cell["measurements"]) for depth in (1, 2)}
                for cell_name in cell_names} for build in ("before", "candidate")
        },
        "per_build_cell_summary": build_cell_summaries,
        "depth2_guard_crossbuild_matches": guard_matches,
        "sanitized_request_records": {
            "measurements": [sanitized_request(row, cell["input_sha256"])
                for session in sessions for cell in session["cells"] for row in cell["measurements"]],
            "fixed_cell_warmups": [sanitized_request(cell["warmup"], cell["input_sha256"])
                for session in sessions for cell in session["cells"]],
            "warmmax4352_setups": [sanitized_request(row, manifest["input_token_sha256"]["warmmax4352statefirst"])
                for row in setups],
        },
        "request_record_policy": "Contains only build/cell/session identifiers, counts, timings, sampling/MTP counters, host metrics and SHA-256 hashes. Session readiness objects, input/output token IDs and text are excluded.",
        "comparisons": comparisons,
        "host_observations": {
            "missing_per_request_samples": sum(row["host"].get("observation_missing", False) for row in all_rows),
            "missing_cpu_metric_samples": sum("external_cpu_max_percent" not in row["host"] for row in all_rows),
            "missing_thermal_samples": sum("thermal_max_state" not in row["host"] for row in all_rows),
            "external_cpu_max_percent_observed": max(observed_external_cpu) if observed_external_cpu else None,
            "max_process_cpu_percent_observed": max(observed_process_cpu) if observed_process_cpu else None,
            "max_model_cpu_percent_observed": max(observed_model_cpu) if observed_model_cpu else None,
            "monitor_sampling_failures_by_session": {
                str(session["session"]): session.get("monitor_summary", {}).get("sampling_failures", 0)
                for session in sessions
            },
            "max_monitor_sampling_failures_in_one_session": max(
                (session.get("monitor_summary", {}).get("sampling_failures", 0) for session in sessions),
                default=0,
            ),
            "thermal_max_state": max(observed_thermal_states) if observed_thermal_states else None,
            "paging_deltas": {key: sum(row["host"].get("paging_delta", {}).get(key, 0)
                for row in all_rows) for key in ("pageouts", "swapouts")},
            "failure_count_semantics": "Monitor failures are cumulative per worker session; report each session once and the maximum, never sum per-request snapshots.",
            "policy": manifest["host_observation_policy"],
        },
        "predeclared_checks": checks,
        "endpoint_met": all(checks.values()),
        "interpretation_limit": "This is a cross-build AUTO-policy comparison with verified MTP depths, not an isolated draft-depth effect. Heldout session halves must both improve and their paired medians must differ by no more than 10 percentage points; otherwise report inconclusive. The endpoint is a bounded confirmation, not a general performance claim.",
    }
    output = PRIVATE / "aggregate.json"
    if output.exists():
        raise SystemExit("aggregate_already_exists")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"measured": 64, "warmups": 16, "setups": 4,
                      "heldout_reduction_percent": held["paired_wall_reduction_percent_median"],
                      "heldout_wall_wins": held["paired_wall_wins"],
                      "endpoint_met": report["endpoint_met"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
