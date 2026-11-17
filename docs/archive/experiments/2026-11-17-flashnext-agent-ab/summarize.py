#!/usr/bin/env python3
"""Summarize the content-free Pi/Flash Next campaign receipt as aggregate JSON."""

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
from statistics import median

BASELINE, CANDIDATE = 4096, 2048
WORK_KEYS = ("finish", "input_tokens", "output_tokens", "sampled_tokens", "output_budget")
MTP_KEYS = ("rounds", "proposed", "accepted", "verified_rows")
HOST_KEYS = ("Pageins", "Pageouts", "Swapins", "Swapouts")
TOOLS = {"read", "edit", "write", "grep", "find", "ls", "other"}
PI_FINISH = {"stop", "length", "toolUse", "tool_limit", "aborted", "error", "missing", "unknown"}
CLIENT_ERRORS = {"missing_usage", "missing_finish_reason", "missing_done", "invalid_usage", "upstream_error",
                 "transport_failed", "invalid_request_route"}
API_ERRORS = {"invalid_request", "model_not_found", "model_not_loaded", "model_busy", "invalid_model_output", "tool_parse_failed",
              "model_changed", "runtime_load_failed", "queue_full", "queue_timeout", "generation_failed",
              "context_overflow", "request_too_large", "request_timeout", "generation_cancelled",
              "mtp_not_supported", "tools_not_supported"}
RUN_ERRORS = {"request_limit", "request_settings", "native_acknowledgement", "pi_exited", "turn_timeout",
              "pi_prompt_rejected", "pi_rpc_failed", "pi_model_selection", "api_model", "api_mtp", "native_ready",
              "OSError", "ValueError", "TimeoutError", "HTTPError", "URLError"}


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def sum_known(rows, key):
    return sum(row[key] for row in rows if number(row.get(key)))


def distribution(values):
    valid = [value for value in values if number(value)]
    return {"samples": len(valid), "min": min(valid), "median": median(valid), "max": max(valid)} if valid else {"samples": 0}


def safe_code(value, allowed):
    return value if isinstance(value, str) and value in allowed else "other"


def client_error(value):
    if isinstance(value, str) and value.startswith("stark_http_"):
        status = value.removeprefix("stark_http_")
        if len(status) == 3 and status.isascii() and status.isdecimal() and 400 <= int(status) <= 599:
            return value
    return safe_code(value, CLIENT_ERRORS)


def host_delta(before, after):
    return {key: after[key] - before[key] for key in HOST_KEYS
            if number(before.get(key)) and number(after.get(key)) and after[key] >= before[key]}


def relative(pairs):
    timed = [(base, candidate) for base, candidate in pairs if number(base) and base > 0 and number(candidate)]
    if not timed:
        return {"pairs": 0}
    ratios = [candidate / base for base, candidate in timed]
    return {
        "pairs": len(timed), "median_2048_over_4096": median(ratios),
        "median_percent_change": (median(ratios) - 1) * 100,
        "2048_faster": sum(candidate < base for base, candidate in timed),
        "4096_faster": sum(base < candidate for base, candidate in timed),
        "ties": sum(base == candidate for base, candidate in timed),
        "4096_total_seconds": sum(base for base, _ in timed),
        "2048_total_seconds": sum(candidate for _, candidate in timed),
    }


def work_key(request):
    return (request.get("phase"), request.get("input_sha256"), request.get("output_sha256"),
            *(request.get(key) for key in WORK_KEYS),
            *((request.get("mtp") or {}).get(key) for key in MTP_KEYS))


def request_integrity(row, grid):
    cached = row.get("cached_tokens")
    input_count = row.get("input_tokens")
    output = row.get("output_tokens")
    sampled = row.get("sampled_tokens")
    budget = row.get("output_budget")
    finish = row.get("finish")
    counts = row.get("mtp") or {}
    rounds, proposed, accepted, verified = (counts.get(key) for key in MTP_KEYS)
    return (
        all(type(value) is int and value >= 0 for value in
            (cached, input_count, output, sampled, budget, rounds, proposed, accepted, verified))
        and cached == row.get("expected_cached") and cached % grid == 0 and cached < input_count
        and (cached == 0 or input_count - cached >= 2)
        and row.get("mtp_valid") is True and accepted <= proposed <= 2 * rounds
        and verified == rounds + proposed and (sampled <= 3 or proposed > 0)
        and finish in ("length", "eos", "stop_token", "stop_sequence", "context")
        and sampled == output + (finish in ("eos", "stop_token")) and sampled <= budget
        and (finish != "length" or sampled == budget)
        and all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ("input_sha256", "output_sha256"))
    )


def quality(turn):
    grade = turn.get("grade") or {}
    passed, total = grade.get("passed"), grade.get("total")
    return (grade.get("status") == "ok" and type(total) is int and type(passed) is int
            and total > 0 and passed == total)


def accepted_turn(turn):
    return quality(turn) and turn.get("client_finish", "stop") == "stop"


def read_lines(path):
    rows = []
    partial_lines = 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                partial_lines += 1
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                partial_lines += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                partial_lines += 1
    return rows, partial_lines


def collect(rows):
    sessions = {}
    arm_failures = {}
    for row in rows:
        kind = row.get("type")
        label = row.get("session")
        if kind == "session_start" and isinstance(label, str):
            sessions[label] = {"start": row, "turns": [], "requests": [], "native_errors": 0, "api_errors": [],
                               "end": None, "failure": None}
        elif kind == "arm_failure":
            arm_failures[row.get("block")] = safe_code(row.get("code"), RUN_ERRORS)
        elif label in sessions:
            session = sessions[label]
            if kind == "turn":
                session["turns"].append(row)
            elif kind == "native_request":
                session["requests"].append(row)
            elif kind == "native_error":
                session["native_errors"] += 1
            elif kind == "api_error":
                session["api_errors"].append(row)
            elif kind == "session_end":
                session["end"] = row
            elif kind == "session_failure":
                session["failure"] = safe_code(row.get("code"), RUN_ERRORS)
    return sessions, arm_failures


def session_status(session):
    start, end = session["start"], session["end"] or {}
    turns, requests = session["turns"], session["requests"]
    phase_counts = Counter(request.get("phase") for request in requests)
    integrity = (
        all(request_integrity(request, start["grid"]) for request in requests)
        and not session["native_errors"]
        and (not requests or requests[0].get("cached_tokens") == 0)
        and all(turn.get("native_requests") == phase_counts[turn.get("phase")] for turn in turns)
        and all(sum(request.get("phase") == phase and request.get("request", 0) >= 1
                    for request in requests) == phase_counts[phase] for phase in phase_counts)
    )
    grade_total = sum_known([turn.get("grade") or {} for turn in turns], "total")
    grade_passed = sum_known([turn.get("grade") or {} for turn in turns], "passed")
    phases = [turn.get("phase") for turn in turns]
    completed = bool(end)
    if completed:
        integrity = (integrity and phases == [1, 2, 3]
                     and set(phase_counts) == set(phases)
                     and all([request["request"] for request in requests if request.get("phase") == phase]
                             == list(range(1, phase_counts[phase] + 1)) for phase in phases))
    accepted = completed and all(map(accepted_turn, turns)) and end.get("passed") is True and integrity
    return {
        "complete": completed, "accepted": accepted, "integrity": integrity, "failed": session["failure"] is not None,
        "failure_code": session["failure"],
        "turns": len(turns), "accepted_turns": sum(map(quality, turns)),
        "turns_passing_grade_and_pi_finish": sum(map(accepted_turn, turns)),
        "checks_passed": grade_passed, "checks_total": grade_total,
        "tools": sum_known(turns, "calls"), "tool_errors": sum_known(turns, "errors"),
        "requests": len(requests), "native_errors": session["native_errors"],
        "positive_mtp_requests": sum(type(proposed) is int and proposed > 0
            for proposed in ((request.get("mtp") or {}).get("proposed") for request in requests)),
        "hits": sum((request.get("cached_tokens") or 0) > 0 for request in requests),
        "reused_tokens": sum_known(requests, "cached_tokens"),
        "input_tokens": sum_known(requests, "input_tokens"),
        "output_tokens": sum_known(requests, "output_tokens"),
        "turn_seconds": sum_known(turns, "wall_seconds"),
        "turn_first_text_observations": sum(number(turn.get("first_text_seconds")) for turn in turns),
        "turn_first_text_seconds": sum_known(turns, "first_text_seconds"),
        "session_seconds": end.get("wall_seconds"),
        "host_deltas": host_delta(start.get("host", {}), end.get("host", {})),
        "turn_timeline": [{key: turn.get(key) for key in (
            "phase", "wall_seconds", "first_text_seconds", "calls", "errors", "native_requests"
        )} | {"accepted": quality(turn), "grade_and_pi_finish_accepted": accepted_turn(turn),
              "client_finish": safe_code(turn.get("client_finish"), PI_FINISH)} for turn in turns],
    }


def exact_pairs(base, candidate):
    by_work = defaultdict(deque)
    for row in candidate["requests"]:
        if request_integrity(row, CANDIDATE):
            by_work[work_key(row)].append(row)
    for row in base["requests"]:
        if request_integrity(row, BASELINE) and by_work[work_key(row)]:
            yield row, by_work[work_key(row)].popleft()


def phase_work(session, turn):
    phase = turn["phase"]
    grid = session["start"]["grid"]
    requests = [row for row in session["requests"] if row.get("phase") == phase]
    if (not requests or len(requests) != turn.get("native_requests")
            or [row.get("request") for row in requests] != list(range(1, len(requests) + 1))
            or not all(request_integrity(row, grid) for row in requests)):
        return None
    return [work_key(row) for row in requests]


def paired_user_turns(base, candidate):
    confirmed = []
    candidate_turns = {row.get("phase"): row for row in candidate["turns"]}
    for left in base["turns"]:
        phase = left.get("phase")
        right = candidate_turns.get(phase)
        if (phase not in (1, 2, 3) or right is None or not quality(left) or not quality(right)
                or left.get("client_finish") != "stop" or right.get("client_finish") != "stop"
                or left["grade"]["total"] != right["grade"]["total"]
                or not number(left.get("wall_seconds")) or left["wall_seconds"] == 0
                or not number(right.get("wall_seconds"))):
            continue
        left_work = phase_work(base, left)
        right_work = phase_work(candidate, right)
        confirmed.append({"phase": phase, "4096_seconds": left["wall_seconds"],
                          "2048_seconds": right["wall_seconds"],
                          "identical_full_native_work": left_work is not None and left_work == right_work})
    return confirmed


def turn_comparison(records):
    return {
        "all_grade_and_pi_accepted": relative([(row["4096_seconds"], row["2048_seconds"]) for row in records]),
        "identical_full_native_work": relative([(row["4096_seconds"], row["2048_seconds"]) for row in records
                                                if row["identical_full_native_work"]]),
    }


def input_observations(base, candidate, exact):
    def key(row):
        return row.get("phase"), row.get("input_sha256"), row.get("input_tokens")

    left = [row for row in base["requests"] if request_integrity(row, BASELINE)]
    right = [row for row in candidate["requests"] if request_integrity(row, CANDIDATE)]
    inputs = Counter(map(key, left)) & Counter(map(key, right))
    left_outputs = Counter((key(row), row["output_sha256"]) for row in left)
    right_outputs = Counter((key(row), row["output_sha256"]) for row in right)
    same_input = sum(inputs.values())
    same_output = sum((left_outputs & right_outputs).values())
    return {"same_input": same_input, "same_input_and_output": same_output,
            "same_input_output_and_work": exact, "same_input_different_output": same_input - same_output,
            "same_input_output_nonmatching_work": same_output - exact}


def extended_arm(sessions):
    requests = [row for session in sessions for row in session["requests"]]
    turns = [row for session in sessions for row in session["turns"]]
    api = [row for session in sessions for row in session["api_errors"]]
    tool_totals = defaultdict(Counter)
    for turn in turns:
        for name, counts in (turn.get("tool_totals") or {}).items():
            if isinstance(counts, dict):
                for field in ("calls", "errors"):
                    if number(counts.get(field)):
                        tool_totals[safe_code(name, TOOLS)][field] += counts[field]
    return {
        "native_prefill_seconds": sum_known(requests, "prefill_seconds"),
        "native_prefill_observations": sum(number(row.get("prefill_seconds")) for row in requests),
        "remaining_uncached_tokens": sum(row["input_tokens"] - row["cached_tokens"] for row in requests
            if number(row.get("input_tokens")) and number(row.get("cached_tokens")) and row["cached_tokens"] <= row["input_tokens"]),
        "prompt_length_tokens": distribution(row.get("input_tokens") for row in requests),
        "prompt_below_4096": sum(number(row.get("input_tokens")) and row["input_tokens"] < 4096 for row in requests),
        "prompt_at_least_4096": sum(number(row.get("input_tokens")) and row["input_tokens"] >= 4096 for row in requests),
        "tool_totals_by_name": dict(tool_totals),
        "pi_finish_counts": dict(Counter(safe_code(row.get("client_finish"), PI_FINISH) for row in turns
                                         if "client_finish" in row)),
        "pi_error_counts": dict(Counter(client_error(row["client_error"]) for row in turns if row.get("client_error"))),
        "api_error_count": len(api),
        "api_error_codes": dict(Counter(safe_code(row.get("code"), API_ERRORS) for row in api)),
        "api_error_statuses": dict(Counter(row["status"] for row in api
                                          if type(row.get("status")) is int and 400 <= row["status"] <= 599)),
        "session_failure_codes": dict(Counter(session["failure"] for session in sessions if session["failure"])),
    }


def summarize(rows, partial_lines):
    manifest = next((row for row in rows if row.get("type") == "manifest"), {})
    schedule = manifest.get("schedule") or []
    sessions, arm_failures = collect(rows)
    status = {label: session_status(session) for label, session in sessions.items()}
    indexed = {(session["start"]["block"], session["start"]["task"]): label for label, session in sessions.items()}
    arms = {}
    for grid in (BASELINE, CANDIDATE):
        labels = [label for label, session in sessions.items() if session["start"]["grid"] == grid]
        samples = [status[label] for label in labels]
        arms[str(grid)] = {
            "sessions_started": len(samples),
            "sessions_completed": sum(sample["complete"] for sample in samples),
            "sessions_accepted": sum(sample["accepted"] for sample in samples),
            "sessions_with_failure": sum(sample["failed"] for sample in samples),
            "sessions_passing_recorded_integrity": sum(sample["integrity"] for sample in samples),
            "host_observed_sessions": sum(bool(sample["host_deltas"]) for sample in samples),
            "host_deltas": {key: sum(sample["host_deltas"].get(key, 0) for sample in samples) for key in HOST_KEYS},
            **{key: sum_known(samples, key) for key in ("turns", "accepted_turns", "turns_passing_grade_and_pi_finish", "checks_passed", "checks_total",
                "tools", "tool_errors", "requests", "native_errors", "positive_mtp_requests", "hits", "reused_tokens",
                "input_tokens", "output_tokens", "turn_seconds", "turn_first_text_observations", "turn_first_text_seconds")},
            **extended_arm([sessions[label] for label in labels]),
        }
    pairs = []
    paired_turns, paired_sessions = [], []
    user_turn_rows = []
    matched = defaultdict(list)
    matched_sessions = defaultdict(list)
    inputs_total = Counter()
    for first in range(0, len(schedule) - 1, 2):
        left, right = schedule[first:first + 2]
        if {left[0], right[0]} != {BASELINE, CANDIDATE} or set(left[1]) != set(right[1]):
            continue
        blocks = {left[0]: first + 1, right[0]: first + 2}
        for task in left[1]:
            labels = {grid: indexed.get((block, task)) for grid, block in blocks.items()}
            base_label, candidate_label = labels[BASELINE], labels[CANDIDATE]
            base, candidate = status.get(base_label), status.get(candidate_label)
            equal = False
            native_pairs = []
            user_turns = []
            inputs = None
            if base and candidate:
                base_session, candidate_session = sessions[base_label], sessions[candidate_label]
                user_turns = paired_user_turns(base_session, candidate_session)
                user_turn_rows.extend(user_turns)
                equal = (base["accepted"] and candidate["accepted"]
                         and [(turn["phase"], turn["grade"]["total"]) for turn in base_session["turns"]]
                         == [(turn["phase"], turn["grade"]["total"]) for turn in candidate_session["turns"]])
                native_pairs = list(exact_pairs(base_session, candidate_session))
                inputs = input_observations(base_session, candidate_session, len(native_pairs))
                inputs_total.update(inputs)
                by_reuse = defaultdict(list)
                for current, trial in native_pairs:
                    group = ("fresh_both" if current["cached_tokens"] == trial["cached_tokens"] == 0 else
                             "same_reuse" if current["cached_tokens"] == trial["cached_tokens"] else "different_reuse")
                    matched[group].append((current, trial))
                    by_reuse[group].append((current, trial))
                    if group == "fresh_both":
                        length_group = "fresh_both_at_least_4096" if current["input_tokens"] >= 4096 else "fresh_both_below_4096"
                        matched[length_group].append((current, trial))
                        by_reuse[length_group].append((current, trial))
                for group, group_pairs in by_reuse.items():
                    matched_sessions[group].append(group_pairs)
                if equal:
                    paired_turns.append((base["turn_seconds"], candidate["turn_seconds"]))
                    paired_sessions.append((base["session_seconds"], candidate["session_seconds"]))
            pairs.append({
                "task": task, "order": [left[0], right[0]], "blocks": blocks, "equal_accepted_work": equal,
                "exact_matched_native_requests": len(native_pairs), "4096": base, "2048": candidate,
                "same_input_observations": inputs,
                "accepted_user_turn_pairs": user_turns,
                "accepted_user_turn_summary": turn_comparison(user_turns),
            })

    groups = {}
    for group in ("fresh_both", "same_reuse", "different_reuse", "fresh_both_at_least_4096", "fresh_both_below_4096"):
        records = matched[group]
        fields = ("prefill_seconds", "decode_seconds", "worker_wall_seconds")
        groups[group] = {
            "requests": len(records),
            "prompt_length_tokens": distribution(base["input_tokens"] for base, _ in records),
            "4096_reused": sum(base["cached_tokens"] for base, _ in records),
            "2048_reused": sum(candidate["cached_tokens"] for _, candidate in records),
            **{field: relative([(base.get(field), candidate.get(field)) for base, candidate in records])
               for field in fields},
            "session_total_ratios": {field: relative([
                (sum(base[field] for base, _ in group_pairs), sum(candidate[field] for _, candidate in group_pairs))
                for group_pairs in matched_sessions[group]
                if all(number(base.get(field)) and number(candidate.get(field)) for base, candidate in group_pairs)
            ]) for field in fields},
        }
    return {
        "campaign_end_marker": any(row.get("type") == "complete" for row in rows),
        "ignored_partial_or_invalid_lines": partial_lines,
        "planned_pairs": len(pairs),
        "arm_failure_count": len(arm_failures),
        "arm_failure_codes": dict(Counter(arm_failures.values())),
        "arms": arms, "paired_groups": pairs,
        "equal_accepted_pi_turn_seconds": relative(paired_turns),
        "equal_accepted_session_wall_seconds": relative(paired_sessions),
        "exact_native_matches": groups,
        "same_input_observations": dict(inputs_total),
        "paired_user_turn_seconds": {
            "overall": turn_comparison(user_turn_rows),
            "by_phase": {phase: turn_comparison([row for row in user_turn_rows if row["phase"] == phase])
                         for phase in (1, 2, 3)},
        },
        "limits": [
            "Session-level pairs require both sessions to pass the same grade phases and check counts.",
            "Descriptive user-turn pairs require both turns to pass grading and finish normally in Pi, even when their sessions fail. The exact subset also matches all native requests in order; user-turn pairs within one session are dependent.",
            "Exact native matches require the same input and generated-token hashes, finish, output budget, token counts and MTP work within a paired task and phase. Unmatched requests remain in arm totals.",
            "Native phase timings are dependent observations within sessions. The paired session ratios are descriptive and launch order and host paging may affect them.",
            "Session wall time includes Pi lifecycle and grading. Pi turn time includes client, tool and request time. First-text totals count only observations that had text.",
            "Host deltas are system-wide counters over each session and only available when sampled before and after.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(*read_lines(args.receipt)), ensure_ascii=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
