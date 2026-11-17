#!/usr/bin/env python3
"""Verify both unchanged 4K Lily kernel receipts and emit content-free results."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from cold_summary import host_stats, is_count, is_hash, is_time

GRID, CONTEXT, DRAFTS, SEED = 4096, 40960, 2, 42
LENGTHS = (8192, 12288, 32768)
SCHEDULE = [False, True, True, False]
KERNELS = {
    "moe": ("STARK_FLASHNEXT_PREFILL_GROUPED_SWIGLU", 96),
    "hc": ("STARK_LILY_VERIFY_HC_DOWN_REUSE", 192),
}
SAMPLING = {"mode": "stochastic", "temperature": 0.6, "top_k": 20, "top_p": 0.95}
MODEL = "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
REVISION = "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
ARTIFACT = "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4"
IDENTITIES = ("worker_sha256", "runner_sha256", "fixture_source_sha256", "cargo_lock_sha256",
              "image_manifest_sha256", "artifact_sha256")
MTP_COUNTS = ("rounds", "proposed", "accepted", "verified_rows")


def require(condition, error):
    if not condition:
        raise ValueError(error)


def read_receipt(path):
    data = path.read_bytes()
    require(data and data.endswith(b"\n"), "incomplete_receipt_line")
    try:
        rows = [json.loads(line) for line in data.splitlines()]
    except (ValueError, UnicodeError) as error:
        raise ValueError("malformed_json_record") from error
    require(all(isinstance(row, dict) for row in rows), "malformed_record")
    return rows, sha256(data).hexdigest()


def request_signature(row, budget):
    counts = row.get("mtp_counts")
    times = row.get("mtp_seconds")
    sampled, emitted, finish = (row.get(key) for key in ("sampled_tokens", "output_tokens", "finish"))
    require(type(row.get("cached_tokens")) is int and row["cached_tokens"] == 0
            and row.get("max_new_tokens") == budget and row.get("requested_drafts") == DRAFTS,
            "request_control")
    require(all(is_hash(row.get(key)) for key in ("input_sha256", "generated_sha256", "text_sha256")),
            "request_hash")
    require(is_count(sampled) and is_count(emitted) and 0 < sampled <= budget
            and finish in ("length", "eos") and sampled == emitted + (finish == "eos")
            and (finish != "length" or sampled == budget), "token_work")
    require(isinstance(counts, dict) and all(is_count(counts.get(key)) for key in MTP_COUNTS), "mtp_counts")
    rounds, proposed, accepted, verified = (counts[key] for key in MTP_COUNTS)
    require(0 <= accepted <= proposed <= 2 * rounds and proposed > 0 and verified == rounds + proposed, "mtp_work")
    require(isinstance(times, dict) and all(is_time(times.get(key)) for key in
            ("draft_seconds", "verify_seconds", "rollback_seconds")), "mtp_times")
    require(all(is_time(row.get(key)) and row[key] > 0 for key in ("prefill_seconds", "decode_seconds", "wall_seconds")),
            "native_timing")
    return (row["input_sha256"], row["generated_sha256"], row["text_sha256"],
            sampled, emitted, finish, *(counts[key] for key in MTP_COUNTS))


def paired_stats(before_candidate):
    extract = {
        "prefill": lambda row: row["prefill_seconds"],
        "decode": lambda row: row["decode_seconds"],
        "verify": lambda row: row["mtp_seconds"]["verify_seconds"],
        "wall": lambda row: row["wall_seconds"],
    }
    results = {}
    for key, get_seconds in extract.items():
        times = [(get_seconds(before), get_seconds(candidate)) for before, candidate in before_candidate]
        ratios = [candidate / before for before, candidate in times if before > 0]
        require(len(ratios) == len(times), "zero_baseline_timing")
        results[key] = {
            "pairs": len(ratios), "candidate_wins": sum(value < 1 for value in ratios),
            "baseline_wins": sum(value > 1 for value in ratios), "ties": sum(value == 1 for value in ratios),
            "median_candidate_over_baseline": median(ratios),
            "ratio_range": [min(ratios), max(ratios)],
            "baseline_median_seconds": median(before for before, _ in times),
            "candidate_median_seconds": median(candidate for _, candidate in times),
        }
    return results


def verify(path):
    rows, receipt_sha = read_receipt(path)
    kinds = Counter(row.get("type") for row in rows)
    require(kinds == {"manifest": 1, "ready": 4, "request": 16, "comparison": 6, "complete": 1}
            and rows[0].get("type") == "manifest" and rows[-1].get("type") == "complete",
            "unexpected_or_incomplete_record_set")
    manifest, complete = rows[0], rows[-1]
    kernel = manifest.get("kernel")
    require(isinstance(kernel, str) and kernel in KERNELS, "unknown_kernel")
    selector, budget = KERNELS[kernel]
    require(manifest.get("schema") == 1 and manifest.get("selector") == selector and manifest.get("grid") == GRID
            and manifest.get("schedule") == SCHEDULE and all(type(value) is bool for value in manifest["schedule"])
            and manifest.get("selected_blocks") == [1, 2, 3, 4] and manifest.get("context_tokens") == CONTEXT
            and manifest.get("output_cap") == budget and manifest.get("seed") == SEED and manifest.get("drafts") == DRAFTS
            and manifest.get("sampling") == SAMPLING, "manifest_control")
    require(manifest.get("model_source") == MODEL and manifest.get("model_revision") == REVISION
            and manifest.get("artifact_sha256") == ARTIFACT and all(is_hash(manifest.get(key)) for key in IDENTITIES),
            "artifact_identity")
    source_commit = manifest.get("source_commit")
    require(isinstance(source_commit, str) and len(source_commit) == 40
            and all(c in "0123456789abcdef" for c in source_commit), "source_identity")
    jobs = manifest.get("pair_one_jobs")
    require(isinstance(jobs, list) and len(jobs) == len(LENGTHS)
            and all(isinstance(job, list) and len(job) == 2 and isinstance(job[0], str)
                    and job[0] and job[1] == length for job, length in zip(jobs, LENGTHS))
            and len({job[0] for job in jobs}) == len(LENGTHS), "fixture_control")
    signatures = {}
    ready = set()
    block_requests = []
    by_join = {}
    comparisons = {}
    for row in rows[1:-1]:
        kind = row["type"]
        if kind == "comparison":
            key = row.get("pair"), row.get("prompt"), row.get("input_tokens")
            require(key in by_join and set(by_join[key]) == {False, True} and key not in comparisons,
                    "unexpected_comparison_marker")
            comparisons[key] = row
            continue
        block = row.get("block")
        require(type(block) is int and 1 <= block <= 4, "block_control")
        enabled = SCHEDULE[block - 1]
        pair = (block + 1) // 2
        require(row.get("candidate_enabled") is enabled and row.get("pair") == pair and row.get("grid") == GRID
                and row.get("kernel") == kernel, "arm_control")
        if kind == "ready":
            require(block == len(ready) + 1 and not block_requests
                    and row.get("image_profile") == "qwen38-lily-retained-bf16-vision-v1"
                    and is_time(row.get("startup_seconds")), "ready_control")
            ready.add(block)
            continue
        require(block == len(ready), "request_before_ready")
        job = [row.get("prompt"), row.get("input_tokens")]
        ordered = [jobs[0], *(jobs if pair == 1 else reversed(jobs))]
        require(job == ordered[len(block_requests)] and row.get("warmup") is (not block_requests), "request_schedule")
        block_requests.append(row)
        signature = request_signature(row, budget)
        fixture = tuple(job)
        prior = signatures.setdefault(fixture, signature)
        require(signature[0] == prior[0], "fixture_input_drift")
        if not row["warmup"]:
            logical_key = pair, *job
            join = by_join.setdefault(logical_key, {})
            require(enabled not in join, "duplicate_arm_observation")
            join[enabled] = (row, signature)
        if len(block_requests) == 4:
            block_requests.clear()
    require(ready == {1, 2, 3, 4} and not block_requests and len(by_join) == 6 and set(comparisons) == set(by_join),
            "incomplete_join")
    paired = []
    for (pair, prompt, length), arms in sorted(by_join.items(), key=lambda item: (item[0][0], item[0][2])):
        require(set(arms) == {False, True}, "incomplete_pair")
        (before, baseline_work), (after, candidate_work) = arms[False], arms[True]
        marker = comparisons[pair, prompt, length]
        same_input, exact_work = baseline_work[0] == candidate_work[0], baseline_work == candidate_work
        require(marker.get("same_input") is same_input and marker.get("exact_work") is exact_work,
                "comparison_marker_disagrees")
        require(exact_work, "paired_work_mismatch")
        paired.append((pair, length, before, after))
    require(complete.get("artifact_identity_unchanged") is True and complete.get("paired_requests") == len(paired)
            and complete.get("exact_work_pairs") == len(paired), "final_receipt_guard")
    return {
        "kernel": kernel, "source_commit": source_commit, "receipt_sha256": receipt_sha,
        "identities": {key: manifest[key] for key in IDENTITIES},
        "counts": {"launches": len(ready), "warmups": 4, "measured_requests": len(paired) * 2,
                   "exact_pairs": len(paired)},
        "overall": paired_stats([(before, after) for _, _, before, after in paired]),
        "by_length": {length: paired_stats([(before, after) for _, n, before, after in paired if n == length])
                      for length in LENGTHS},
        "by_pair_order": {
            pair: {"first_arm": "baseline" if pair == 1 else "candidate",
                   "overall": paired_stats([(before, after) for p, _, before, after in paired if p == pair]),
                   "by_length": {length: paired_stats([(before, after) for p, n, before, after in paired
                                                      if p == pair and n == length]) for length in LENGTHS}}
            for pair in (1, 2)
        },
        "measured_host_paging": {
            arm: host_stats([match[index] for match in paired])
            for arm, index in (("baseline", 2), ("candidate", 3))
        },
    }, jobs, tuple(signatures[tuple(job)][0] for job in jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs=2)
    args = parser.parse_args()
    try:
        left, right = (verify(path) for path in args.receipts)
        require({left[0]["kernel"], right[0]["kernel"]} == set(KERNELS), "require_both_kernels")
        require(left[0]["identities"] == right[0]["identities"] and left[0]["source_commit"] == right[0]["source_commit"]
                and left[1:] == right[1:], "cross_campaign_identity_drift")
    except (OSError, TypeError, KeyError, IndexError):
        parser.exit(1, "Kernel receipt verification failed.\n")
    except ValueError as error:
        parser.exit(1, f"Kernel receipt verification failed ({error}).\n")
    arms = sorted((left[0], right[0]), key=lambda arm: arm["kernel"])
    print(json.dumps({"schema": 1, "receipts_verified": len(arms), "arm_schedule": ["baseline", "candidate", "candidate", "baseline"],
                      "results": {arm["kernel"]: arm for arm in arms},
                      "limits": "Descriptive paired timings on shared worker launches. Paging counters are system-wide."},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
