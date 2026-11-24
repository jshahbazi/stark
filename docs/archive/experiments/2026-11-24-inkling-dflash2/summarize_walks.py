"""Validate private native walk receipts and print aggregate verifier timings."""

import argparse
import json
import statistics
from pathlib import Path

CASES = ("coding", "sustained", "prose", "context4k")
ARMS = ("dspark", "dflash")
DEPTHS = ("depth_three", "depth_seven_offline_chunk_emulation")
TOKENS_PER_CASE = 192


def validate_walk(receipt: dict, depth: str) -> dict:
    assert receipt["prefill_anchor_matches_teacher"]
    assert receipt["teacher_len"] == TOKENS_PER_CASE
    arm = receipt[depth]
    walk = arm["walk"]
    assert arm["complete"] and arm["final_state_exact"]
    assert walk["token_count_invariant"]
    assert walk["emitted_tokens"] == TOKENS_PER_CASE
    assert walk["top1_drift_rows"] == 0
    assert walk["target_rows_checked"] == walk["exact_logit_rows"] == TOKENS_PER_CASE
    assert walk["prefix_state_matches"] == walk["prefix_state_checks"]
    assert walk["committed_state_matches"] == walk["committed_state_checks"]
    assert walk["rounds"] > 0 and walk["rounds"] + walk["accepted_proposals"] == TOKENS_PER_CASE
    assert walk["transaction_calls"] + walk["singleton_calls"] > 0
    assert walk["verify_ms"] >= 0 and walk["commit_ms"] >= 0
    return walk


def collect(traces: Path, blocks: int) -> dict:
    rows = {(arm, depth): {case: [] for case in CASES} for arm in ARMS for depth in (3, 7)}
    for case in CASES:
        for block in range(blocks):
            receipts = {
                arm: json.loads((traces / case / f"actual{block}-{arm}.json").read_text())
                for arm in ARMS
            }
            order = [3, 7] if block < 3 else [7, 3]
            assert all(receipt.get("depth_order", [3, 7]) == order for receipt in receipts.values())
            checked = {
                (arm, depth): validate_walk(receipts[arm], DEPTHS[index])
                for arm in ARMS
                for index, depth in enumerate((3, 7))
            }
            for key, walk in checked.items():
                rows[key][case].append(walk)

    for key, cases in rows.items():
        for case, walks in cases.items():
            counts = {(w["rounds"], w["transaction_calls"], w["singleton_calls"]) for w in walks}
            assert len(counts) == 1, (key, case)
    return rows


def summarize(case_walks: dict[str, list[dict]]) -> dict:
    per_block = [
        [case_walks[case][block] for case in CASES]
        for block in range(len(case_walks[CASES[0]]))
    ]
    tokens = len(CASES) * TOKENS_PER_CASE
    rounds = sum(walk["rounds"] for walk in per_block[0])
    verify_ms_by_block = [sum(walk["verify_ms"] for walk in block) for block in per_block]
    combined_ms_by_block = [
        sum(walk["verify_ms"] + walk["commit_ms"] for walk in block) for block in per_block
    ]
    calls = sum(
        walk["transaction_calls"] + walk["singleton_calls"]
        for walk in per_block[0]
    )
    median_verify_ms = statistics.median(verify_ms_by_block)
    return {
        "tokens": tokens,
        "rounds": rounds,
        "physical_target_calls": calls,
        "median_verify_ms_per_round": median_verify_ms / rounds,
        "median_verify_ms_per_token": median_verify_ms / tokens,
        "median_verify_and_commit_ms_per_token": statistics.median(combined_ms_by_block) / tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="private directory containing case receipts")
    parser.add_argument("--blocks", type=int, default=6, help="number of paired blocks to include")
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    rows = collect(args.traces, args.blocks)
    output = {
        f"{arm}_{depth}": summarize(rows[(arm, depth)])
        for arm in ARMS
        for depth in (3, 7)
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
