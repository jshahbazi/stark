"""Offline, aggregate-only analysis of the Lily verifier stage screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SCREEN = HERE / "screen-v2"
EXPECTED_VERIFICATION = "batched-decode-exact-v1"
WORK_KEYS = ("rounds", "proposed", "accepted", "verified_rows")
HASH_KEYS = ("output_token_ids_sha256", "text_sha256")


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def median(values):
    clean = [value for value in values if finite_number(value)]
    return statistics.median(clean) if clean else None


def safe_counter(profile, name):
    counter = (profile or {}).get("counters", {}).get(name, {})
    value = counter.get("seconds") if isinstance(counter, dict) else None
    return value if finite_number(value) else None


def exclusive_stage_categories(profile):
    counters = (profile or {}).get("counters", {})
    prefix = "qwen.mtp.verify.gpu."
    phase_names = {
        name[len(prefix):]: item.get("seconds")
        for name, item in counters.items()
        if name.startswith(prefix)
        and isinstance(item, dict)
        and finite_number(item.get("seconds"))
    }
    excluded = {"command", "measured", "profile_resolve"}
    result = {}
    errors = []
    for phase, seconds in phase_names.items():
        if phase in excluded:
            continue
        # Coarse GDN/MoE counters include their child intervals. Keep only
        # leaves, so reported stage fractions do not count a parent twice.
        if any(other.startswith(phase + "_") or other.startswith(phase + ".")
               for other in phase_names if other != phase):
            children = [value for other, value in phase_names.items()
                        if other.startswith(phase + "_") or other.startswith(phase + ".")]
            residual = seconds - sum(children)
            tolerance = 1e-9
            if residual < -tolerance:
                errors.append({"phase": phase, "parent_seconds": seconds,
                               "children_seconds": sum(children), "residual_seconds": residual})
            else:
                result[phase + "_other"] = max(0.0, residual)
            continue
        result[phase] = seconds
    return result, errors


def request_key(row):
    return (row.get("case"), row.get("repetition"))


def finite_distribution(rows, field):
    vals = [row.get(field) for row in rows if finite_number(row.get(field))]
    return {
        "n": len(vals),
        "min": min(vals) if vals else None,
        "median": median(vals),
        "max": max(vals) if vals else None,
    }


def load_screen(root: Path):
    manifest = read_json(root / "manifest.json")
    missing = []
    sessions = []
    if manifest is None:
        missing.append("manifest.json")
    for index in range(1, 5):
        path = root / f"session-{index}.json"
        data = read_json(path)
        if data is None:
            missing.append(path.name)
        else:
            sessions.append(data)
    return manifest, sessions, missing


def analyze(root: Path):
    manifest, sessions, missing = load_screen(root)
    expected_order = ["off", "stage", "stage", "off"]
    if manifest is None:
        return {"status": "incomplete", "missing": missing, "sessions_loaded": len(sessions)}

    integrity = {
        "manifest_completed": manifest.get("completed") is True,
        "session_order_matches": manifest.get("session_order") == expected_order,
        "all_sessions_present": len(sessions) == 4,
        "session_arms_match": len(sessions) == 4 and [s.get("arm") for s in sessions] == expected_order,
    }
    arms = {"off": [], "stage": []}
    all_rows = []
    stage_failures = []
    input_identities = {}
    for session in sessions:
        arm = session.get("arm")
        if arm not in arms:
            continue
        if session.get("failure_type") is not None or session.get("model_guard_passed") is not True:
            integrity["sessions_valid"] = False
        for case, identity in session.get("inputs", {}).items():
            key = f"{case}:{session.get('index')}"
            input_identities[key] = {
                field: identity.get(field)
                for field in ("input_tokens", "codec_sha256", "rendered_sha256", "token_ids_sha256")
            }
        for row in session.get("requests", []):
            row = dict(row)
            row["session"] = session.get("index")
            row["arm"] = arm
            all_rows.append(row)
            arms[arm].append(row)
            if row.get("mtp_verification") != EXPECTED_VERIFICATION:
                stage_failures.append({"session": session.get("index"), "case": row.get("case"),
                                       "repetition": row.get("repetition"), "check": "verifier_id"})
            if arm == "stage" and not isinstance(row.get("aggregate_profile"), dict):
                stage_failures.append({"session": session.get("index"), "case": row.get("case"),
                                       "repetition": row.get("repetition"), "check": "aggregate_profile_missing"})
    integrity["sessions_valid"] = integrity.get("sessions_valid", True)

    # Input and rendered hashes must stay fixed across all four loaded sessions.
    input_by_case = {}
    for session in sessions:
        for case, identity in session.get("inputs", {}).items():
            input_by_case.setdefault(case, []).append(identity)
    integrity["input_hashes_match"] = bool(input_by_case) and all(
        len({(item.get("input_tokens"), item.get("codec_sha256"), item.get("rendered_sha256"),
              item.get("token_ids_sha256")) for item in values}) == 1
        and len(values) == 4
        for values in input_by_case.values()
    )

    # Screen completeness means every expected case/repetition exists in each arm.
    expected_keys = {(case, rep) for case in (manifest.get("cases") or {}) for rep in (0, 1)}
    from collections import Counter
    by_arm_key = {arm: Counter(request_key(row) for row in rows) for arm, rows in arms.items()}
    integrity["request_grid_complete"] = all(
        len(arms[arm]) == len(expected_keys) * 2
        and all(by_arm_key[arm][key] == 2 for key in expected_keys)
        for arm in ("off", "stage")
    )

    matched = []
    comparison_checks = {"matched_pairs": 0, "output_hash_matches": 0, "length_matches": 0,
                         "work_counter_matches": 0, "verifier_id_matches": 0}
    pair_sessions = ((1, 2), (3, 4))
    rows_by_session = {session.get("index"): session.get("requests", []) for session in sessions}
    for off_session, stage_session in pair_sessions:
        for key in sorted(expected_keys):
            for warmup in (True, False):
                off_rows = [row for row in rows_by_session.get(off_session, [])
                            if request_key(row) == key and row.get("warmup") is warmup]
                stage_rows = [row for row in rows_by_session.get(stage_session, [])
                              if request_key(row) == key and row.get("warmup") is warmup]
                if len(off_rows) != 1 or len(stage_rows) != 1:
                    continue
                off, stage = off_rows[0], stage_rows[0]
                comparison_checks["matched_pairs"] += 1
                hash_match = all(off.get(k) == stage.get(k) and off.get(k) for k in HASH_KEYS)
                length_match = all(off.get(k) == stage.get(k)
                                   for k in ("input_tokens", "output_tokens", "sampled_tokens"))
                work_match = all(off.get("mtp_counters", {}).get(k) == stage.get("mtp_counters", {}).get(k)
                                 for k in WORK_KEYS)
                verifier_match = off.get("mtp_verification") == stage.get("mtp_verification") == EXPECTED_VERIFICATION
                comparison_checks["output_hash_matches"] += int(hash_match)
                comparison_checks["length_matches"] += int(length_match)
                comparison_checks["work_counter_matches"] += int(work_match)
                comparison_checks["verifier_id_matches"] += int(verifier_match)
                matched.append({"case": key[0], "repetition": key[1], "warmup": warmup,
                                "off_session": off_session, "stage_session": stage_session,
                                "output_hashes_match": hash_match, "lengths_match": length_match,
                                "work_counters_match": work_match, "verifier_id_matches": verifier_match})
                for label, row in (("off", off), ("stage", stage)):
                    matched[-1][label] = {
                        "input_tokens": row.get("input_tokens"),
                        "output_tokens": row.get("output_tokens"),
                        "sampled_tokens": row.get("sampled_tokens"),
                        "output_token_ids_sha256": row.get("output_token_ids_sha256"),
                        "text_sha256": row.get("text_sha256"),
                        "actual_drafts": row.get("actual_drafts"),
                        "mtp_verification": row.get("mtp_verification"),
                        "mtp_counters": row.get("mtp_counters"),
                    }

    measured_by_arm = {
        arm: [row for row in rows if row.get("warmup") is False]
        for arm, rows in arms.items()
    }
    timing = {}
    for arm, rows in measured_by_arm.items():
        timing[arm] = {}
        for case in (manifest.get("cases") or {}):
            subset = [row for row in rows if row.get("case") == case]
            timing[arm][case] = {
                "requests": len(subset),
                "wall_seconds": finite_distribution(subset, "wall_seconds"),
                "decode_seconds": finite_distribution(subset, "decode_seconds"),
                "generation_seconds": finite_distribution(subset, "generation_seconds"),
                "useful_decode_tokens_per_second": finite_distribution(subset, "useful_decode_tokens_per_second"),
            }

    stage_sessions = {}
    stage_profile_errors = []
    for session in sessions:
        if session.get("arm") != "stage":
            continue
        session_id = str(session.get("index"))
        stage_sessions[session_id] = {}
        for case in (manifest.get("cases") or {}):
            subset = [row for row in measured_by_arm["stage"]
                      if row.get("session") == session.get("index") and row.get("case") == case]
            category_samples = {}
            for row in subset:
                profile = row.get("aggregate_profile") or {}
                decode = row.get("decode_seconds")
                command = safe_counter(profile, "qwen.mtp.verify.gpu.command")
                categories, errors = exclusive_stage_categories(profile)
                for error in errors:
                    stage_profile_errors.append({"session": session.get("index"), "case": case, **error})
                for category, seconds in categories.items():
                    sample = category_samples.setdefault(category, {"seconds": [], "decode_fraction": [],
                                                                      "gpu_command_fraction": []})
                    sample["seconds"].append(seconds)
                    if finite_number(decode) and decode > 0:
                        sample["decode_fraction"].append(seconds / decode)
                    if finite_number(command) and command > 0:
                        sample["gpu_command_fraction"].append(seconds / command)
            stage_sessions[session_id][case] = {
                "measured_requests": len(subset),
                "categories": {
                    name: {
                        "median_seconds_per_request": median(values["seconds"]),
                        "median_fraction_of_request_decode": median(values["decode_fraction"]),
                        "median_fraction_of_gpu_command": median(values["gpu_command_fraction"]),
                        "n": len(values["seconds"]),
                    }
                    for name, values in sorted(category_samples.items())
                },
            }

    complete = (not missing and all(integrity.values()) and not stage_failures
                and comparison_checks["matched_pairs"] == len(expected_keys) * 2
                and comparison_checks["output_hash_matches"] == comparison_checks["matched_pairs"]
                and comparison_checks["length_matches"] == comparison_checks["matched_pairs"]
                and comparison_checks["work_counter_matches"] == comparison_checks["matched_pairs"]
                and comparison_checks["verifier_id_matches"] == comparison_checks["matched_pairs"]
                and not stage_profile_errors)
    return {
        "schema": "stark-lily-verifier-screen-analysis-v1",
        "status": "complete" if complete else ("incomplete" if missing or not integrity["request_grid_complete"] else "review_required"),
        "screen_dir": str(root),
        "manifest_attempt": manifest.get("attempt"),
        "source_commit": manifest.get("source_commit"),
        "missing_files": missing,
        "integrity": integrity,
        "expected_verification": EXPECTED_VERIFICATION,
        "stage_failures": stage_failures,
        "stage_profile_errors": stage_profile_errors,
        "input_identities": input_identities,
        "comparison_checks": comparison_checks,
        "matched_off_stage_requests": matched,
        "request_counts": {arm: {"all": len(rows), "measured": len(measured_by_arm[arm]),
                                  "warmup": sum(1 for row in rows if row.get("warmup") is True)}
                           for arm, rows in arms.items()},
        "timing_distributions": timing,
        "stage_sessions": stage_sessions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", type=Path, default=SCREEN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.screen_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = args.output or args.screen_dir / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(output)
    sys.stdout.write(rendered)
    return 0 if report.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
