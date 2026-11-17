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


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def sum_known(rows, key):
    return sum(row[key] for row in rows if number(row.get(key)))


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
        and finish in ("length", "eos", "stop_sequence")
        and sampled == output + (finish == "eos") and sampled <= budget
        and (finish != "length" or sampled == budget)
        and all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ("input_sha256", "output_sha256"))
    )


def quality(turn):
    grade = turn.get("grade") or {}
    passed, total = grade.get("passed"), grade.get("total")
    return (grade.get("status") == "ok" and type(total) is int and type(passed) is int
            and total > 0 and passed == total)


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
    arm_failures = set()
    for row in rows:
        kind = row.get("type")
        label = row.get("session")
        if kind == "session_start" and isinstance(label, str):
            sessions[label] = {"start": row, "turns": [], "requests": [], "native_errors": 0, "end": None, "failure": False}
        elif kind == "arm_failure":
            arm_failures.add(row.get("block"))
        elif label in sessions:
            session = sessions[label]
            if kind == "turn":
                session["turns"].append(row)
            elif kind == "native_request":
                session["requests"].append(row)
            elif kind == "native_error":
                session["native_errors"] += 1
            elif kind == "session_end":
                session["end"] = row
            elif kind == "session_failure":
                session["failure"] = True
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
        integrity = (integrity and phases == list(range(1, len(phases) + 1))
                     and set(phase_counts) == set(phases)
                     and all([request["request"] for request in requests if request.get("phase") == phase]
                             == list(range(1, phase_counts[phase] + 1)) for phase in phases))
    accepted = completed and bool(turns) and all(map(quality, turns)) and end.get("passed") is True and integrity
    return {
        "complete": completed, "accepted": accepted, "integrity": integrity, "failed": session["failure"],
        "turns": len(turns), "accepted_turns": sum(map(quality, turns)),
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
        )} | {"accepted": quality(turn)} for turn in turns],
    }


def exact_pairs(base, candidate):
    by_work = defaultdict(deque)
    for row in candidate["requests"]:
        if request_integrity(row, CANDIDATE):
            by_work[work_key(row)].append(row)
    for row in base["requests"]:
        if request_integrity(row, BASELINE) and by_work[work_key(row)]:
            yield row, by_work[work_key(row)].popleft()


def summarize(rows, partial_lines):
    manifest = next((row for row in rows if row.get("type") == "manifest"), {})
    schedule = manifest.get("schedule") or []
    sessions, arm_failures = collect(rows)
    status = {label: session_status(session) for label, session in sessions.items()}
    indexed = {(session["start"]["block"], session["start"]["task"]): label for label, session in sessions.items()}
    arms = {}
    for grid in (BASELINE, CANDIDATE):
        samples = [status[label] for label, session in sessions.items() if session["start"]["grid"] == grid]
        arms[str(grid)] = {
            "sessions_started": len(samples),
            "sessions_completed": sum(sample["complete"] for sample in samples),
            "sessions_accepted": sum(sample["accepted"] for sample in samples),
            "sessions_with_failure": sum(sample["failed"] for sample in samples),
            "sessions_passing_recorded_integrity": sum(sample["integrity"] for sample in samples),
            "host_observed_sessions": sum(bool(sample["host_deltas"]) for sample in samples),
            "host_deltas": {key: sum(sample["host_deltas"].get(key, 0) for sample in samples) for key in HOST_KEYS},
            **{key: sum_known(samples, key) for key in ("turns", "accepted_turns", "checks_passed", "checks_total",
                "tools", "tool_errors", "requests", "native_errors", "positive_mtp_requests", "hits", "reused_tokens",
                "input_tokens", "output_tokens", "turn_seconds", "turn_first_text_observations", "turn_first_text_seconds")},
        }
    pairs = []
    paired_turns, paired_sessions = [], []
    matched = defaultdict(list)
    matched_sessions = defaultdict(list)
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
            if base and candidate:
                base_session, candidate_session = sessions[base_label], sessions[candidate_label]
                equal = (base["accepted"] and candidate["accepted"]
                         and [(turn["phase"], turn["grade"]["total"]) for turn in base_session["turns"]]
                         == [(turn["phase"], turn["grade"]["total"]) for turn in candidate_session["turns"]])
                native_pairs = list(exact_pairs(base_session, candidate_session))
                by_reuse = defaultdict(list)
                for current, trial in native_pairs:
                    group = ("fresh_both" if current["cached_tokens"] == trial["cached_tokens"] == 0 else
                             "same_reuse" if current["cached_tokens"] == trial["cached_tokens"] else "different_reuse")
                    matched[group].append((current, trial))
                    by_reuse[group].append((current, trial))
                for group, group_pairs in by_reuse.items():
                    matched_sessions[group].append(group_pairs)
                if equal:
                    paired_turns.append((base["turn_seconds"], candidate["turn_seconds"]))
                    paired_sessions.append((base["session_seconds"], candidate["session_seconds"]))
            pairs.append({
                "task": task, "order": [left[0], right[0]], "blocks": blocks, "equal_accepted_work": equal,
                "exact_matched_native_requests": len(native_pairs), "4096": base, "2048": candidate,
            })

    groups = {}
    for group in ("fresh_both", "same_reuse", "different_reuse"):
        records = matched[group]
        fields = ("prefill_seconds", "decode_seconds", "worker_wall_seconds")
        groups[group] = {
            "requests": len(records),
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
        "arms": arms, "paired_groups": pairs,
        "equal_accepted_pi_turn_seconds": relative(paired_turns),
        "equal_accepted_session_wall_seconds": relative(paired_sessions),
        "exact_native_matches": groups,
        "limits": [
            "Session-level pairs require both sessions to pass the same grade phases and check counts.",
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
