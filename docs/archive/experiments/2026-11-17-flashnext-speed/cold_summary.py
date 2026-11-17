#!/usr/bin/env python3
"""Summarize an ABBA fresh-input Lily coding receipt without exposing request hashes."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

BASELINE = 4096
CANDIDATE = 2048
LENGTHS = (8192, 12288, 32768)
MTP_KEYS = ("rounds", "proposed", "accepted", "verified_rows")
HOST_KEYS = ("Pageins", "Pageouts", "Swapins", "Swapouts")


def is_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def is_count(value):
    return type(value) is int and value >= 0


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_receipt(path):
    rows = []
    dropped = 0
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line) if line.endswith(b"\n") else None
            except (ValueError, UnicodeError):
                row = None
            if isinstance(row, dict):
                rows.append(row)
            else:
                dropped += 1
    return rows, dropped


def timing_pairs(pairs, field):
    ratios = []
    before = []
    after = []
    for left, right in pairs:
        base, trial = left.get(field), right.get(field)
        if is_time(base) and base > 0 and is_time(trial):
            ratios.append(trial / base)
            before.append(base)
            after.append(trial)
    if not ratios:
        return {"pairs": 0}
    return {
        "pairs": len(ratios),
        "2048_faster": sum(ratio < 1 for ratio in ratios),
        "4096_faster": sum(ratio > 1 for ratio in ratios),
        "ties": sum(ratio == 1 for ratio in ratios),
        "2048_at_least_5_percent_faster": sum(ratio <= 0.95 for ratio in ratios),
        "2048_at_least_5_percent_slower": sum(ratio >= 1.05 for ratio in ratios),
        "median_2048_over_4096": median(ratios),
        "min_2048_over_4096": min(ratios),
        "max_2048_over_4096": max(ratios),
        "4096_seconds_range": [min(before), max(before)],
        "2048_seconds_range": [min(after), max(after)],
        "4096_total_seconds": sum(before),
        "2048_total_seconds": sum(after),
    }


def same_cold_input(left, right):
    return (type(left.get("cached_tokens")) is int and type(right.get("cached_tokens")) is int
            and left["cached_tokens"] == right["cached_tokens"] == 0
            and is_hash(left.get("input_sha256")) and left["input_sha256"] == right.get("input_sha256"))


def work_signature(row):
    finish = row.get("finish")
    emitted, sampled = row.get("output_tokens"), row.get("sampled_tokens")
    counts = row.get("mtp_counts")
    if not isinstance(counts, dict) or not all(is_count(counts.get(key)) for key in MTP_KEYS):
        return None
    rounds, proposals, accepted, verified = (counts[key] for key in MTP_KEYS)
    if (not is_count(emitted) or not is_count(sampled) or not sampled
            or finish not in ("eos", "stop_token", "stop_sequence", "length", "context")
            or sampled != emitted + (finish in ("eos", "stop_token"))
            or not 0 <= accepted <= proposals <= 2 * rounds or verified != rounds + proposals
            or (sampled > 3 and not proposals)
            or not all(is_hash(row.get(key)) for key in ("generated_sha256", "text_sha256"))):
        return None
    return (row["generated_sha256"], row["text_sha256"], finish, emitted, sampled,
            row.get("output_budget", row.get("max_new_tokens")), *(counts[key] for key in MTP_KEYS))


def classify(pairs, *, midpoint=False):
    equal_input = []
    exact = []
    nonmatching_work = []
    for left, right in pairs:
        if same_cold_input(left, right):
            equal_input.append((left, right))
            work = work_signature(left)
            if work is not None and work == work_signature(right):
                exact.append((left, right))
            else:
                nonmatching_work.append((left, right))
    result = {
        "paired_observations": len(pairs),
        "same_fresh_input_pairs": len(equal_input),
        "exact_work_pairs": len(exact),
        "prefill_same_fresh_input": timing_pairs(exact if midpoint else equal_input, "prefill_seconds"),
        "prefill_exact_work": timing_pairs(exact, "prefill_seconds"),
        "prefill_same_input_nonmatching_work": timing_pairs(nonmatching_work, "prefill_seconds"),
        "decode_exact_work": timing_pairs(exact, "decode_seconds"),
        "wall_exact_work": timing_pairs(exact, "wall_seconds"),
    }
    if midpoint:
        output_fields = ("generated_sha256", "text_sha256", "sampled_tokens", "output_tokens", "finish")
        output_changed = sum(any(left.get(key) != right.get(key) for key in output_fields)
                             for left, right in nonmatching_work)
        result.update(input_mismatch_pairs=len(pairs) - len(equal_input), output_change_pairs=output_changed,
                      unchanged_output_nonmatching_work_pairs=len(nonmatching_work) - output_changed)
    return result


def midpoint_geometry(row):
    grid = row.get("grid")
    return (grid in (BASELINE, CANDIDATE) and row.get("prefix_chunk_tokens") == grid
            and row.get("physical_chunk_tokens") == BASELINE and row.get("midpoint_enabled") is (grid == CANDIDATE))


def host_stats(rows):
    summed = Counter()
    observed = Counter()
    affected = Counter()
    incomplete = 0
    for row in rows:
        before = row.get("host_before") or {}
        after = row.get("host_after") or {}
        if before.get("incomplete") or after.get("incomplete"):
            incomplete += 1
        for key in HOST_KEYS:
            left, right = before.get(key), after.get(key)
            if is_count(left) and is_count(right) and right >= left:
                observed[key] += 1
                summed[key] += right - left
                affected[key] += int(right > left)
    return {
        key: {"observations": observed[key], "total_delta_pages": summed[key], "positive_observations": affected[key]}
        for key in HOST_KEYS
    } | {"snapshots_marked_incomplete": incomplete}


def aggregate(rows, dropped):
    manifests = [row for row in rows if row.get("type") == "manifest"]
    manifest = manifests[0] if manifests else {}
    midpoint = manifest.get("mode") == "midpoint"
    probe = midpoint and manifest.get("probe_only") is True
    ready = [row for row in rows if row.get("type") == "ready"]
    requests = [row for row in rows if row.get("type") == "request"]
    blocks = defaultdict(set)
    for row in (*ready, *requests):
        block, grid = row.get("block"), row.get("grid")
        if block in (1, 2, 3, 4) and grid in (BASELINE, CANDIDATE):
            blocks[block].add(grid)
    expected_ready = {row.get("block") for row in ready if row.get("block") in (1, 2, 3, 4)}
    ready_seen = Counter(row.get("block") for row in ready)
    valid_ready = {row.get("block"): row for row in ready if midpoint_geometry(row) and ready_seen[row.get("block")] == 1}
    scope_declared = manifest.get("scored_unique_scopes") is True
    scope_digests_present = any("scope_sha256" in row for row in requests)
    scope_digests = Counter(row.get("scope_sha256") for row in requests if is_hash(row.get("scope_sha256")))

    by_fixture = defaultdict(lambda: defaultdict(list))
    fixture_hashes = defaultdict(set)
    unscheduled = 0
    timing_requests = [row for row in requests if row.get("warmup") is False]
    invalid_midpoint = 0
    for row in timing_requests:
        block, pair, grid = (row.get(key) for key in ("block", "pair", "grid"))
        length, prompt = row.get("input_tokens"), row.get("prompt")
        if (type(block) is not int or block not in (1, 2, 3, 4) or pair != (block + 1) // 2
                or grid not in (BASELINE, CANDIDATE) or length not in LENGTHS or not isinstance(prompt, str)
                or len(blocks[block]) != 1):
            unscheduled += 1
            continue
        if midpoint and (not scope_declared or not midpoint_geometry(row) or block not in valid_ready
                         or valid_ready[block].get("grid") != grid or row.get("expected_cached_tokens") != 0
                         or not is_count(row.get("cached_tokens")) or row["cached_tokens"] != 0
                         or (scope_digests_present and scope_digests[row.get("scope_sha256")] != 1)):
            invalid_midpoint += 1
            continue
        by_fixture[(pair, prompt, length)][grid].append(row)
        if is_hash(row.get("input_sha256")):
            fixture_hashes[(prompt, length)].add(row["input_sha256"])

    paired = defaultdict(list)
    missing = Counter()
    for (pair, _, length), grids in by_fixture.items():
        base, candidate = grids[BASELINE], grids[CANDIDATE]
        missing[BASELINE] += max(0, len(candidate) - len(base))
        missing[CANDIDATE] += max(0, len(base) - len(candidate))
        for left, right in zip(base, candidate):
            if left["block"] == right["block"]:
                unscheduled += 2
                continue
            paired[(pair, length)].append((left, right))

    all_pairs = [match for matches in paired.values() for match in matches]
    block_grids = [next(iter(blocks[i])) if len(blocks[i]) == 1 else None for i in range(1, 5)]
    complete = None not in block_grids
    counterbalanced = complete and block_grids in ([4096, 2048, 2048, 4096], [2048, 4096, 4096, 2048])
    arms = {}
    for grid in (BASELINE, CANDIDATE):
        measured = [row for row in timing_requests if row.get("grid") == grid]
        arms[str(grid)] = {
            "measured_requests": len(measured),
            "warmups": sum(row.get("grid") == grid and row.get("warmup") is True for row in requests),
            "measured_requests_with_reuse": sum(is_count(row.get("cached_tokens")) and row["cached_tokens"] > 0 for row in measured),
            "measured_requests_missing_reuse": sum(not is_count(row.get("cached_tokens")) for row in measured),
            "invalid_work_signature": sum(work_signature(row) is None for row in measured),
            "host_paging": host_stats(measured),
            "by_length": {length: {"measured_requests": sum(row.get("input_tokens") == length for row in measured),
                                  "host_paging": host_stats([row for row in measured if row.get("input_tokens") == length])}
                          for length in LENGTHS},
        }

    identity_fields = ("worker_sha256", "model_config_sha256", "model_manifest_sha256", "image_manifest_sha256",
                       "artifact_sha256", "corpus_sha256", "fixtures_sha256", "fixture_sha256", "runner_sha256",
                       "cargo_lock_sha256")
    identities = {}
    for field in identity_fields:
        values = [row[field] for row in (*manifests, *ready) if field in row]
        if values:
            identities[field] = {"observations": len(values), "valid_hashes": sum(map(is_hash, values)),
                                 "distinct_valid_hashes": len({value for value in values if is_hash(value)})}
    result = {
        "end_marker": any(row.get("type") == "complete" for row in rows),
        "artifact_identity_unchanged": any(row.get("type") == "complete" and row.get("artifact_identity_unchanged") is True
                                           for row in rows),
        "failure_markers": sum(row.get("type") == "failure" for row in rows),
        "ignored_partial_or_invalid_lines": dropped,
        "worker_ready_blocks": sorted(expected_ready),
        "block_grids": block_grids,
        "counterbalanced_abba": counterbalanced if complete else None,
        "observed_grids_match_manifest": (block_grids == manifests[0].get("schedule"))
                                         if complete and manifests and "schedule" in manifests[0] else None,
        "fixture_lengths_with_input_hash_drift": sum(len(values) > 1 for values in fixture_hashes.values()),
        "identity_observations": identities,
        "unpaired_observations_missing_grid": {str(grid): missing[grid] for grid in (BASELINE, CANDIDATE)},
        "invalid_schedule_observations": unscheduled,
        "arms": arms,
        "overall": classify(all_pairs, midpoint=midpoint),
        "by_length": {length: classify([match for (pair, n), matches in paired.items() if n == length for match in matches], midpoint=midpoint)
                      for length in LENGTHS},
        "by_pair_order": {
            pair: {"first_grid": block_grids[2 * pair - 2], "second_grid": block_grids[2 * pair - 1],
                   "overall": classify([match for (p, _), matches in paired.items() if p == pair for match in matches], midpoint=midpoint),
                   "by_length": {length: classify(paired[(pair, length)], midpoint=midpoint) for length in LENGTHS}}
            for pair in (1, 2)
        },
        "limits": [
            "Recorded warmups are excluded. Fresh requires both requests to report zero reused tokens and the same input hash.",
            "Decode and elapsed request comparisons additionally require exact generated token and text hashes, token counts, budget, finish, and MTP work.",
            "Repeated observations of one fixture within the same worker block are dependent. Ratios and the 5% threshold are descriptive.",
            "Host paging counters are system-wide deltas, with no attribution to the worker or implication of exclusive host access.",
        ],
    }
    if midpoint:
        result.update(mode="midpoint", grid_labels={"4096": "baseline prefix grid", "2048": "midpoint prefix grid"},
                      physical_chunk_tokens=BASELINE, probe_only=probe,
                      capture_proofs={str(grid): sum(row.get("type") == "capture_proof" and midpoint_geometry(row)
                          and row.get("grid") == grid and row.get("exact_work") is True and is_count(row.get("input_tokens"))
                          and row.get("confirmed_cached_tokens") == (row["input_tokens"] - 1) // grid * grid
                          for row in rows) for grid in (BASELINE, CANDIDATE)},
                      midpoint_controls={"valid_ready_blocks": sorted(valid_ready), "invalid_measured_requests": invalid_midpoint,
                                         "scored_unique_scope_declared": scope_declared,
                                         "scope_digest_evidence_present": scope_digests_present,
                                         "all_request_scope_digests_unique": (len(scope_digests) == len(requests))
                                         if scope_digests_present else None})
        result["limits"].append("Midpoint headline ratios use exact-work pairs only. Each arm uses a 4,096-token physical chunk; arm labels identify prefix grids. Scope uniqueness is runner-declared unless digests are present.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(*read_receipt(args.receipt)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
