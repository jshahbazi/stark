"""Offline, aggregate-only analysis of the Lily projection-reuse comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from analyze_screen import exclusive_stage_categories, finite_number, safe_counter

HERE = Path(__file__).resolve().parent
COMPARE = HERE / "compare-v4"
DETAILED = HERE / "detailed-v3"
EXPECTED_VERIFICATION = "batched-decode-exact-v1"
WORK_KEYS = ("rounds", "proposed", "accepted", "verified_rows")
HASH_KEYS = ("output_token_ids_sha256", "text_sha256")
EXPECTED_ORDER = ("before", "candidate", "candidate", "before")
SESSION_PAIRS = ((1, 2), (4, 3))
REPETITIONS = (0, 1, 2, 3, 4)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def quantile(values, probability):
    clean = sorted(value for value in values if finite_number(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return clean[lower] * (1 - fraction) + clean[upper] * fraction


def distribution(values):
    clean = [value for value in values if finite_number(value)]
    if not clean:
        return {"n": 0, "min": None, "median": None, "max": None,
                "q1": None, "q3": None, "iqr": None, "mad": None}
    med = statistics.median(clean)
    q1, q3 = quantile(clean, 0.25), quantile(clean, 0.75)
    return {"n": len(clean), "min": min(clean), "median": med, "max": max(clean),
            "q1": q1, "q3": q3, "iqr": q3 - q1,
            "mad": statistics.median(abs(value - med) for value in clean)}


def load_sessions(root, count=4):
    manifest = read_json(root / "manifest.json")
    sessions, missing = [], []
    if manifest is None:
        missing.append("manifest.json")
    for index in range(1, count + 1):
        path = root / f"session-{index}.json"
        value = read_json(path)
        if value is None:
            missing.append(path.name)
        else:
            sessions.append(value)
    return manifest, sessions, missing


def row_map(session):
    return {(row.get("case"), row.get("repetition")): row
            for row in session.get("requests", [])}


def request_summary(rows):
    return {
        "requests": len(rows),
        "output_tokens": distribution([row.get("output_tokens") for row in rows]),
        "sampled_tokens": distribution([row.get("sampled_tokens") for row in rows]),
        "natural_eos": sum(row.get("natural_eos") is True for row in rows),
        "finish_reasons": {reason: sum(row.get("finish_reason") == reason for row in rows)
                           for reason in ("eos", "length", "unknown")},
        "mtp_counters": {
            key: distribution([(row.get("mtp_counters") or {}).get(key) for row in rows])
            for key in WORK_KEYS
        },
    }


def host_summary(rows):
    hosts = [row.get("host") for row in rows if isinstance(row.get("host"), dict)]
    paging = [host.get("paging_delta") or {} for host in hosts]
    return {
        "requests_with_host_observation": len(hosts),
        "external_cpu_max": distribution([h.get("external_cpu_max") for h in hosts]),
        "external_model_cpu_max": distribution([h.get("external_model_cpu_max") for h in hosts]),
        "thermal_max": distribution([h.get("thermal_max") for h in hosts]),
        "monitor_samples": distribution([h.get("samples") for h in hosts]),
        "sampling_failures": sum(h.get("sampling_failures", 0) for h in hosts
                                  if type(h.get("sampling_failures")) is int),
        "pageouts_total": sum(p.get("pageouts", 0) for p in paging if type(p.get("pageouts", 0)) is int),
        "swapouts_total": sum(p.get("swapouts", 0) for p in paging if type(p.get("swapouts", 0)) is int),
        "compiler_active_true": sum(h.get("compiler_active") is True for h in hosts),
        "eligible_true": sum(h.get("eligible") is True for h in hosts),
        "eligible_false": sum(h.get("eligible") is False for h in hosts),
    }


def input_identity_guard(sessions, cases):
    identities = {}
    for session in sessions:
        for case, value in session.get("inputs", {}).items():
            identities.setdefault(case, []).append({
                key: value.get(key) for key in
                ("input_tokens", "codec_sha256", "rendered_sha256", "token_ids_sha256")
            })
    checks = {}
    for case in cases:
        values = identities.get(case, [])
        keys = ("input_tokens", "codec_sha256", "rendered_sha256", "token_ids_sha256")
        checks[case] = {"sessions": len(values), "match": len(values) == 4 and
                        len({tuple(item.get(k) for k in keys) for item in values}) == 1,
                        "identity": values[0] if values else None}
    return checks


def compare_pair(before, candidate):
    hash_match = all(before.get(key) == candidate.get(key) and before.get(key)
                     for key in HASH_KEYS)
    output_match = all(before.get(key) == candidate.get(key)
                       for key in ("input_tokens", "output_tokens", "sampled_tokens", "finish_reason",
                                   "natural_eos"))
    work_match = all((before.get("mtp_counters") or {}).get(key) ==
                     (candidate.get("mtp_counters") or {}).get(key) for key in WORK_KEYS)
    verifier_match = (before.get("actual_drafts") == candidate.get("actual_drafts") == 2
                      and before.get("mtp_verification") == candidate.get("mtp_verification")
                      == EXPECTED_VERIFICATION)
    prefix_match = (before.get("prefix_reused_tokens") == 0
                    and candidate.get("prefix_reused_tokens") == 0)
    changes = {}
    for metric in ("decode_seconds", "useful_decode_tokens_per_second", "wall_seconds"):
        b, c = before.get(metric), candidate.get(metric)
        if finite_number(b) and finite_number(c) and b > 0:
            # Positive means candidate improved: less time or more useful tokens/s.
            changes[metric] = ((b - c) / b * 100 if metric != "useful_decode_tokens_per_second"
                               else (c - b) / b * 100)
        else:
            changes[metric] = None
    return {
        "output_hashes_match": hash_match,
        "lengths_and_finish_match": output_match,
        "work_counters_match": work_match,
        "verifier_id_and_depth_match": verifier_match,
        "prefix_disabled_match": prefix_match,
        "improvement_percent": changes,
        "before": {key: before.get(key) for key in
                   (*HASH_KEYS, "input_tokens", "output_tokens", "sampled_tokens", "finish_reason",
                    "natural_eos", "actual_drafts", "mtp_verification", "prefix_reused_tokens",
                    "mtp_counters")},
        "candidate": {key: candidate.get(key) for key in
                       (*HASH_KEYS, "input_tokens", "output_tokens", "sampled_tokens", "finish_reason",
                        "natural_eos", "actual_drafts", "mtp_verification", "prefix_reused_tokens",
                        "mtp_counters")},
    }


def build_detailed_stage_report(root):
    manifest, sessions, missing = load_sessions(root, count=2)
    result = {"status": "incomplete", "missing_files": missing, "sessions": {}}
    if manifest is None:
        return result
    errors = []
    complete_sessions = len(sessions) == 2 and manifest.get("completed") is True
    for session in sessions:
        session_id = str(session.get("index"))
        rows_by_case = {}
        for row in session.get("requests", []):
            if row.get("warmup") is False:
                rows_by_case.setdefault(row.get("case"), []).append(row)
        result["sessions"][session_id] = {}
        for case in (manifest.get("cases") or {}):
            rows = rows_by_case.get(case, [])
            categories = {}
            for row in rows:
                profile = row.get("aggregate_profile") or {}
                cats, local_errors = exclusive_stage_categories(profile)
                for error in local_errors:
                    errors.append({"session": session.get("index"), "case": case, **error})
                decode = row.get("decode_seconds")
                command = safe_counter(profile, "qwen.mtp.verify.gpu.command")
                for name, seconds in cats.items():
                    values = categories.setdefault(name, {"seconds": [], "decode_fraction": [],
                                                            "gpu_command_fraction": []})
                    values["seconds"].append(seconds)
                    if finite_number(decode) and decode > 0:
                        values["decode_fraction"].append(seconds / decode)
                    if finite_number(command) and command > 0:
                        values["gpu_command_fraction"].append(seconds / command)
            result["sessions"][session_id][case] = {
                "measured_requests": len(rows),
                "exclusive_categories": {
                    name: {"median_seconds_per_request": statistics.median(values["seconds"]),
                           "median_fraction_of_request_decode": statistics.median(values["decode_fraction"]),
                           "median_fraction_of_gpu_command": statistics.median(values["gpu_command_fraction"]),
                           "n": len(values["seconds"])}
                    for name, values in sorted(categories.items()) if values["seconds"]
                },
            }
            complete_sessions &= len(rows) == 1 and all(
                isinstance(row.get("aggregate_profile"), dict) for row in rows
            )
    result["profile_errors"] = errors
    result["status"] = "complete" if complete_sessions and not errors else "incomplete"
    result["attempt"] = manifest.get("attempt")
    result["source_commit"] = manifest.get("source_commit")
    return result


def analyze(root: Path, detailed_root: Path):
    manifest, sessions, missing = load_sessions(root)
    if manifest is None:
        return {"status": "incomplete", "missing_files": missing, "sessions_loaded": len(sessions),
                "detailed_v3_stage_fractions": build_detailed_stage_report(detailed_root)}

    cases = list((manifest.get("cases") or {}).keys())
    integrity = {
        "manifest_completed": manifest.get("completed") is True,
        "attempt_matches": manifest.get("attempt") == "20261115-LILY-VERIFY-04",
        "two_expected_cases": set(cases) == {"coding", "planning"},
        "session_order_matches": manifest.get("session_order") == list(EXPECTED_ORDER),
        "all_sessions_present": len(sessions) == 4,
        "session_arms_match": len(sessions) == 4 and tuple(s.get("arm") for s in sessions) == EXPECTED_ORDER,
    }
    per_session_rows = {s.get("index"): s.get("requests", []) for s in sessions}
    arm_rows = {"before": [], "candidate": []}
    per_session_case = {}
    request_checks = []
    for session in sessions:
        arm = session.get("arm")
        if arm not in arm_rows:
            continue
        if session.get("failure_type") is not None or session.get("model_guard_passed") is not True:
            request_checks.append({"session": session.get("index"), "check": "session_completion"})
        ready = session.get("ready") or {}
        if ready.get("mtp_recommended_drafts") != 2:
            request_checks.append({"session": session.get("index"), "check": "recommended_depth"})
        for row in session.get("requests", []):
            arm_rows[arm].append(row)
            key = (row.get("case"), row.get("repetition"))
            per_session_case.setdefault((session.get("index"), *key), []).append(row)
            failed = []
            if row.get("request_status") != "ok": failed.append("request_status")
            if row.get("actual_drafts") != 2 or row.get("automatic_depth_valid") is not True:
                failed.append("draft_depth")
            if row.get("mtp_verification") != EXPECTED_VERIFICATION:
                failed.append("verifier_id")
            if row.get("prefix_reused_tokens") != 0 or row.get("prefix_reuse_valid") is not True:
                failed.append("prefix_reuse")
            if row.get("counts_valid") is not True or row.get("timings_valid") is not True:
                failed.append("counts_or_timing")
            counters = row.get("mtp_counters") or {}
            if not all(type(counters.get(k)) is int and counters[k] >= 0 for k in WORK_KEYS):
                failed.append("mtp_counters")
            if not all(isinstance(row.get(k), str) and len(row[k]) == 64 for k in HASH_KEYS):
                failed.append("response_hashes")
            if failed:
                request_checks.append({"session": session.get("index"), "case": row.get("case"),
                                       "repetition": row.get("repetition"), "checks": failed})
    input_checks = input_identity_guard(sessions, cases)
    integrity["input_identities_match"] = bool(cases) and all(
        input_checks[case]["match"] for case in cases
    )
    expected_case_grid = {(case, rep) for case in cases for rep in REPETITIONS}
    session_grid = {}
    for session in sessions:
        keys = [(row.get("case"), row.get("repetition")) for row in session.get("requests", [])]
        session_grid[str(session.get("index"))] = {
            "request_count": len(keys),
            "grid_complete": len(keys) == len(expected_case_grid)
            and set(keys) == expected_case_grid and len(set(keys)) == len(keys),
            "warmup_count": sum(row.get("warmup") is True for row in session.get("requests", [])),
            "measured_count": sum(row.get("warmup") is False for row in session.get("requests", [])),
        }
    integrity["request_grid_complete"] = len(session_grid) == 4 and all(
        item["grid_complete"] and item["warmup_count"] == 2 and item["measured_count"] == 8
        for item in session_grid.values()
    )

    paired = {case: [] for case in cases}
    for before_session, candidate_session in SESSION_PAIRS:
        before_map = row_map(next((s for s in sessions if s.get("index") == before_session), {}))
        candidate_map = row_map(next((s for s in sessions if s.get("index") == candidate_session), {}))
        for case in cases:
            for repetition in REPETITIONS:
                key = (case, repetition)
                before, candidate = before_map.get(key), candidate_map.get(key)
                if before is None or candidate is None:
                    continue
                record = compare_pair(before, candidate)
                record.update({"before_session": before_session, "candidate_session": candidate_session,
                               "case": case, "repetition": repetition,
                               "warmup": before.get("warmup") is True})
                paired[case].append(record)
    comparison = {}
    timing = {}
    full_range = {}
    for case in cases:
        measured_pairs = [pair for pair in paired[case] if not pair["warmup"]]
        comparison[case] = {
            "all_pairs": len(paired[case]),
            "measured_pairs": len(measured_pairs),
            "output_hash_matches": sum(p["output_hashes_match"] for p in paired[case]),
            "length_finish_matches": sum(p["lengths_and_finish_match"] for p in paired[case]),
            "work_counter_matches": sum(p["work_counters_match"] for p in paired[case]),
            "depth_verifier_matches": sum(p["verifier_id_and_depth_match"] for p in paired[case]),
            "zero_prefix_matches": sum(p["prefix_disabled_match"] for p in paired[case]),
            "measured_pair_changes": {},
            "session_halves": {},
        }
        for metric in ("decode_seconds", "useful_decode_tokens_per_second", "wall_seconds"):
            changes = [p["improvement_percent"].get(metric) for p in measured_pairs]
            changes = [x for x in changes if finite_number(x)]
            wins = sum(x > 0 for x in changes)
            comparison[case]["measured_pair_changes"][metric] = {
                **distribution(changes), "candidate_wins": wins,
                "candidate_losses": sum(x < 0 for x in changes), "ties": sum(x == 0 for x in changes),
            }
        for half_index, (before_session, candidate_session) in enumerate(SESSION_PAIRS, 1):
            before_rows = [r for r in per_session_rows.get(before_session, [])
                           if r.get("case") == case and r.get("warmup") is False]
            candidate_rows = [r for r in per_session_rows.get(candidate_session, [])
                              if r.get("case") == case and r.get("warmup") is False]
            half_pairs = [pair for pair in measured_pairs
                          if pair["before_session"] == before_session
                          and pair["candidate_session"] == candidate_session]
            half = {"before_session": before_session, "candidate_session": candidate_session,
                    "before": {}, "candidate": {},
                    "arm_median_ratio_improvement_percent": {},
                    "paired_improvement_percent": {}}
            for metric in ("decode_seconds", "useful_decode_tokens_per_second", "wall_seconds"):
                half["before"][metric] = distribution([r.get(metric) for r in before_rows])
                half["candidate"][metric] = distribution([r.get(metric) for r in candidate_rows])
                b = half["before"][metric]["median"]
                c = half["candidate"][metric]["median"]
                if finite_number(b) and finite_number(c) and b > 0:
                    half["arm_median_ratio_improvement_percent"][metric] = (
                        (b - c) / b * 100 if metric != "useful_decode_tokens_per_second"
                        else (c - b) / b * 100
                    )
                else:
                    half["arm_median_ratio_improvement_percent"][metric] = None
                changes = [pair["improvement_percent"].get(metric) for pair in half_pairs]
                changes = [value for value in changes if finite_number(value)]
                half["paired_improvement_percent"][metric] = {
                    **distribution(changes),
                    "candidate_wins": sum(value > 0 for value in changes),
                    "candidate_losses": sum(value < 0 for value in changes),
                    "ties": sum(value == 0 for value in changes),
                    "statistic": "median of same-repetition paired percentage changes",
                }
            comparison[case]["session_halves"][str(half_index)] = half

        before_measured = [r for r in arm_rows["before"] if r.get("case") == case and r.get("warmup") is False]
        candidate_measured = [r for r in arm_rows["candidate"] if r.get("case") == case and r.get("warmup") is False]
        timing[case] = {}
        full_range[case] = {}
        for metric in ("decode_seconds", "useful_decode_tokens_per_second", "wall_seconds"):
            bvals = [r.get(metric) for r in before_measured if finite_number(r.get(metric))]
            cvals = [r.get(metric) for r in candidate_measured if finite_number(r.get(metric))]
            timing[case][metric] = {"before": distribution(bvals), "candidate": distribution(cvals)}
            base_med = statistics.median(bvals) if bvals else None
            relative_range = ((max(bvals) - min(bvals)) / base_med * 100
                              if bvals and base_med and base_med > 0 else None)
            improvement = comparison[case]["measured_pair_changes"][metric]["median"]
            hurdle = max(10.0, relative_range) if relative_range is not None else None
            full_range[case][metric] = {
                "before_full_range_percent_of_median": relative_range,
                "hurdle_percent": hurdle,
                "median_paired_improvement_percent": improvement,
                "hurdle_met": improvement >= hurdle if improvement is not None and hurdle is not None else False,
            }

    arm_summaries = {}
    host = {}
    for arm in ("before", "candidate"):
        measured = [row for row in arm_rows[arm] if row.get("warmup") is False]
        warmups = [row for row in arm_rows[arm] if row.get("warmup") is True]
        arm_summaries[arm] = {
            "all_requests": len(arm_rows[arm]), "measured_requests": len(measured), "warmups": len(warmups),
            "cases": {case: request_summary([r for r in measured if r.get("case") == case]) for case in cases},
            "warmup_cases": {case: request_summary([r for r in warmups if r.get("case") == case]) for case in cases},
        }
        host[arm] = {"measured": host_summary(measured), "warmups": host_summary(warmups)}

    pair_count_ok = all(len(paired[case]) == 10 and
                        sum(not p["warmup"] for p in paired[case]) == 8 for case in cases)
    integrity["pair_grid_complete"] = pair_count_ok
    integrity["all_request_checks_pass"] = not request_checks
    all_same = pair_count_ok and all(
        comparison[case][field] == comparison[case]["all_pairs"]
        for case in cases for field in ("output_hash_matches", "length_finish_matches", "work_counter_matches",
                                        "depth_verifier_matches", "zero_prefix_matches")
    )
    integrity["all_paired_outputs_and_work_match"] = all_same

    # Pairwise matches alone can miss a changed same-seed result in another
    # session/repetition. Require one output/text hash and one work tuple for
    # the full 40-request grid, independently for each case and warmup mode.
    full_grid_checks = {}
    flat_rows = [row for session in sessions for row in session.get("requests", [])]
    for case in cases:
        for warmup in (True, False):
            key = f"{case}:{'warmup' if warmup else 'measured'}"
            rows = [row for row in flat_rows if row.get("case") == case and row.get("warmup") is warmup]
            expected_count = 4 if warmup else 16
            output_tuples = {
                (row.get("output_token_ids_sha256"), row.get("text_sha256"))
                for row in rows
            }
            length_tuples = {
                (row.get("input_tokens"), row.get("output_tokens"), row.get("sampled_tokens"),
                 row.get("finish_reason"), row.get("natural_eos"))
                for row in rows
            }
            work_tuples = {
                tuple((row.get("mtp_counters") or {}).get(counter) for counter in WORK_KEYS)
                for row in rows
            }
            output_valid = (len(rows) == expected_count and len(output_tuples) == 1
                            and all(all(isinstance(value, str) and len(value) == 64 for value in values)
                                    for values in output_tuples))
            length_valid = len(rows) == expected_count and len(length_tuples) == 1
            work_valid = len(rows) == expected_count and len(work_tuples) == 1
            full_grid_checks[key] = {
                "requests": len(rows), "expected_requests": expected_count,
                "unique_output_text_hash_pairs": len(output_tuples),
                "unique_length_finish_tuples": len(length_tuples),
                "unique_mtp_work_tuples": len(work_tuples),
                "output_text_hashes_match": output_valid,
                "length_finish_tuples_match": length_valid,
                "mtp_work_tuples_match": work_valid,
            }
    integrity["all_40_requests_present"] = len(flat_rows) == 40
    integrity["full_grid_same_seed_hashes_and_work_match"] = (
        integrity["all_40_requests_present"] and len(full_grid_checks) == len(cases) * 2
        and all(check["output_text_hashes_match"] and check["length_finish_tuples_match"]
                and check["mtp_work_tuples_match"] for check in full_grid_checks.values())
    )

    detailed_stage = build_detailed_stage_report(detailed_root)
    all_integrity = all(integrity.values())
    complete = (not missing and all_integrity and not request_checks and
                detailed_stage.get("status") == "complete")
    return {
        "schema": "stark-lily-verifier-compare-analysis-v1",
        "status": "complete" if complete else ("incomplete" if missing or not integrity["request_grid_complete"]
                                               or not integrity["pair_grid_complete"] else "review_required"),
        "attempt": manifest.get("attempt"), "source_commit": manifest.get("source_commit"),
        "missing_files": missing, "integrity": integrity, "request_checks": request_checks,
        "session_grid": session_grid, "input_identity_checks": input_checks,
        "arm_summaries": arm_summaries, "timing_distributions": timing,
        "paired_changes": comparison, "paired_rows": paired,
        "full_grid_same_seed_checks": full_grid_checks,
        "same_code_baseline_range_and_hurdle": full_range, "host_summaries": host,
        "detailed_v3_stage_fractions": detailed_stage,
    }


def compact(report):
    paired = report.get("paired_changes") or {}
    return {
        "status": report.get("status"),
        "integrity": report.get("integrity"),
        "arm_counts": {arm: {"measured": values.get("measured_requests"),
                              "warmups": values.get("warmups")}
                       for arm, values in (report.get("arm_summaries") or {}).items()},
        "timing_medians": {
            case: {metric: {arm: values[arm].get("median")
                            for arm in ("before", "candidate")}
                  for metric, values in metrics.items()}
            for case, metrics in (report.get("timing_distributions") or {}).items()
        },
        "paired_median_improvement_percent": {
            case: {metric: values.get("median") for metric, values in
                   (case_data.get("measured_pair_changes") or {}).items()}
            for case, case_data in paired.items()
        },
        "paired_median_improvement_by_half_percent": {
            case: {
                half: {metric: values.get("median") for metric, values in
                       (half_data.get("paired_improvement_percent") or {}).items()}
                for half, half_data in (case_data.get("session_halves") or {}).items()
            }
            for case, case_data in paired.items()
        },
        "same_code_hurdles": report.get("same_code_baseline_range_and_hurdle"),
        "detailed_v3_stage_status": (report.get("detailed_v3_stage_fractions") or {}).get("status"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", type=Path, default=COMPARE)
    parser.add_argument("--detailed-dir", type=Path, default=DETAILED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.compare_dir, args.detailed_dir)
    full = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = args.output or args.compare_dir / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(full)
    temporary.replace(output)
    print(json.dumps(compact(report), indent=2, sort_keys=True))
    return 0 if report.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
