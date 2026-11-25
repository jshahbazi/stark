"""CPU-only replay of fixed-width DFlash2 sibling self-correction.

The input is one private JSON sidecar per trace. Its schema is:
``primary`` [n, 3], ``alternate`` [n, 2], ``gap`` [n, 2],
``teacher`` [n] (token IDs), ``trace_sha256`` and ``case``. Alternate
columns correspond to 1-based proposal slots 1 and 2; each alternate is the
conditional runner-up under the ordinary primary predecessor. The gap is the
corrected primary-versus-alternate logit margin. Inputs and token IDs remain
private; output contains only aggregate counts, hashes and modeled costs.

Usage: uv run --no-project python self_correction.py --set archived DIR
       uv run --no-project python self_correction.py --set fresh DIR
DIR contains one ``selector.json`` or ``self_correction.json`` per case,
directly or nested below it. The two sets are run separately so fresh and archived evidence cannot be
silently pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


THRESHOLD = math.log(2.0)
DEFAULT_ROW_MS = (17.79, 19.24, 21.72, 23.78)


@dataclass
class Walk:
    rounds: int = 0
    accepted: int = 0
    draft_rounds: int = 0
    ordinary_rounds: int = 0
    branch_rounds: int = 0
    branch_attempts: int = 0
    eligible_opportunities: int = 0
    branch_catches: int = 0
    baseline_first_misses: int = 0
    lost_primary_tail: int = 0
    target_rows: int = 0
    packet_calls: list[int] = field(default_factory=lambda: [0] * 4)


def _matrix(value: Any, name: str, rows: int, cols: int) -> list[list[int | float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} has invalid row count")
    if any(not isinstance(row, list) or len(row) != cols for row in value):
        raise ValueError(f"{name} has invalid column count")
    return value


def load_sidecar(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    primary = record.get("primary")
    alternate = record.get("alternate")
    gaps = record.get("gap")
    teacher = record.get("teacher")
    if not isinstance(primary, list) or not primary:
        raise ValueError(f"{path}: primary must be a nonempty matrix")
    n = len(primary)
    _matrix(primary, "primary", n, 3)
    _matrix(alternate, "alternate", n, 2)
    _matrix(gaps, "gap", n, 2)
    if not isinstance(teacher, list) or len(teacher) != n:
        raise ValueError(f"{path}: teacher must contain n anchor tokens")
    if not isinstance(record.get("trace_sha256"), str) or len(record["trace_sha256"]) != 64:
        raise ValueError(f"{path}: missing trace_sha256")
    if not isinstance(record.get("case"), str) or not record["case"]:
        raise ValueError(f"{path}: missing case")
    if any(not math.isfinite(x) or x < 0 for row in gaps for x in row):
        raise ValueError(f"{path}: invalid corrected gap")
    if any(a == p for row, alt in zip(primary, alternate) for p, a in zip(row, alt)):
        raise ValueError(f"{path}: runner-up duplicates the primary")
    if record.get("ordinary_primary_ids_equal") != n * 7:
        raise ValueError(f"{path}: ordinary selector parity missing")
    return record


def eligible_slot(gaps: list[float], threshold: float, forced: int | None = None) -> int | None:
    if forced is not None:
        return forced if forced in (0, 1) else None
    return next((i for i, gap in enumerate(gaps) if gap <= threshold), None)


def replay(record: dict[str, Any], mode: str, threshold: float = THRESHOLD) -> Walk:
    """Replay teacher-forced fixed width three, optionally using one sibling.

    A fork at slot 1 or 2 preserves up to two primary nodes and adds the
    sibling within the three-candidate-node budget. It displaces the third
    primary node only when that node is in the remaining output budget. A
    caught sibling commits the verified primary prefix plus that sibling; a
    primary path continues through the retained nodes.
    This is a token-path/cost model; it does not claim target tree execution is
    implemented or bit-exact in the runtime.
    """
    primary, alternate, gaps, teacher = (record[k] for k in ("primary", "alternate", "gap", "teacher"))
    n = len(primary)
    out = Walk()
    cursor = 0
    while cursor < n:
        # Each proposal row predicts a token after the current anchor. At the
        # final anchor, the output budget has no further target token to check.
        available = min(3, n - cursor - 1)
        out.draft_rounds += int(available > 0)
        truth = teacher[cursor + 1:cursor + 1 + available]
        p = primary[cursor][:available]
        hit = 0
        while hit < len(p) and p[hit] == truth[hit]:
            hit += 1
        if hit < len(p):
            out.baseline_first_misses += 1

        selected = None
        if mode == "gated":
            selected = eligible_slot(gaps[cursor], threshold)
        elif mode == "unconditional":
            selected = eligible_slot(gaps[cursor], threshold, forced=0)
        # Keep up to two primary nodes and one sibling inside the four-row
        # verifier cap (anchor + at most three candidate nodes). Near the end a
        # fork can use an otherwise idle slot without displacing primary work.
        if selected is not None and selected < available:
            out.eligible_opportunities += 1
            out.branch_rounds += 1
            out.branch_attempts += 1
            retained_primary = min(available, 2)
            rows = 2 + retained_primary
            accepted = min(hit, retained_primary)
            out.lost_primary_tail += int(hit > retained_primary)
            # A sibling only helps if the primary path reaches the fork.
            if hit == selected and alternate[cursor][selected] == truth[selected]:
                out.branch_catches += 1
                accepted = selected + 1
        else:
            out.ordinary_rounds += 1
            accepted = hit
            rows = 1 + available

        out.accepted += accepted
        out.rounds += 1
        out.target_rows += rows
        out.packet_calls[rows - 1] += 1
        cursor += 1 + accepted
    if out.rounds + out.accepted != n:
        raise AssertionError("replay token accounting invariant failed")
    return out


def baseline_catches(record: dict[str, Any], threshold: float) -> int:
    """Count eligible alternates that catch baseline first misses at baseline anchors."""
    teacher, primary = record["teacher"], record["primary"]
    alternate, gaps = record["alternate"], record["gap"]
    n, cursor, caught = len(primary), 0, 0
    while cursor < n:
        available = min(3, n - cursor - 1)
        truth = teacher[cursor + 1:cursor + 1 + available]
        p = primary[cursor][:available]
        hit = 0
        while hit < len(p) and p[hit] == truth[hit]:
            hit += 1
        selected = eligible_slot(gaps[cursor], threshold)
        if selected is not None and selected < available and selected < 2 and selected == hit:
            caught += int(alternate[cursor][selected] == truth[selected])
        cursor += 1 + hit
    return caught


def _cost(walk: Walk, prices: tuple[float, float, float, float]) -> float:
    return sum(count * price for count, price in zip(walk.packet_calls, prices, strict=True))


def paired_round_interval(rows: list[dict], seed: int) -> dict:
    """Resample prompts as pairs; positive percentages mean fewer gated rounds."""
    rng = random.Random(seed)
    pairs = [(row["baseline_rounds"], row["gated_rounds"]) for row in rows]
    savings = []
    for _ in range(20_000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        base = sum(a for a, _ in sample)
        savings.append(100 * (base - sum(b for _, b in sample)) / base)
    savings.sort()
    return {"unit": "prompt", "seed": seed, "replicates": len(savings),
            "round_savings_pct_95_interval": [savings[499], savings[19499]]}


def analyze(records: list[dict[str, Any]], threshold: float, prices: tuple[float, ...]) -> dict[str, Any]:
    per_case = {}
    for record in records:
        base = replay(record, "baseline", threshold)
        gated = replay(record, "gated", threshold)
        unconditional = replay(record, "unconditional", threshold)
        base_ms, gated_ms, unconditional_ms = (_cost(w, prices) for w in (base, gated, unconditional))
        per_case[record["case"]] = {
            "trace_sha256": record["trace_sha256"],
            "tokens": len(record["primary"]),
            "baseline_rounds": base.rounds,
            "baseline_nonterminal_draft_rounds": base.draft_rounds,
            "baseline_first_misses": base.baseline_first_misses,
            "baseline_packet_calls_rows_1_to_4": base.packet_calls,
            "gated_rounds": gated.rounds,
            "gated_nonterminal_draft_rounds": gated.draft_rounds,
            "gated_ordinary_rounds": gated.ordinary_rounds,
            "gated_branch_rounds": gated.branch_rounds,
            "branch_attempts": gated.branch_attempts,
            "eligible_opportunities": gated.eligible_opportunities,
            "branch_catches": gated.branch_catches,
            "baseline_misses_caught": baseline_catches(record, threshold),
            "lost_primary_tail": gated.lost_primary_tail,
            "baseline_target_rows": base.target_rows,
            "gated_target_rows": gated.target_rows,
            "gated_packet_calls_rows_1_to_4": gated.packet_calls,
            "unconditional_attempts": unconditional.branch_attempts,
            "unconditional_rounds": unconditional.rounds,
            "unconditional_nonterminal_draft_rounds": unconditional.draft_rounds,
            "unconditional_eligible_opportunities": unconditional.eligible_opportunities,
            "unconditional_packet_calls_rows_1_to_4": unconditional.packet_calls,
            "unconditional_catches": unconditional.branch_catches,
            "unconditional_lost_primary_tail": unconditional.lost_primary_tail,
            "baseline_modeled_verify_ms": round(base_ms, 4),
            "gated_modeled_verify_ms": round(gated_ms, 4),
            "unconditional_modeled_verify_ms": round(unconditional_ms, 4),
            "modeled_verify_savings_ms": round(base_ms - gated_ms, 4),
            "savings_ms_per_attempt": round((base_ms - gated_ms) / gated.branch_attempts, 4) if gated.branch_attempts else None,
            "savings_ms_per_baseline_round": round((base_ms - gated_ms) / base.rounds, 4) if base.rounds else None,
            "unconditional_savings_ms": round(base_ms - unconditional_ms, 4),
        }

    groups = {}
    grouped_names = {
        "prose": [name for name in per_case if name == "prose" or name.startswith("prose")],
        "coding": [name for name in per_case if name == "coding" or name.startswith("coding")
                   or name in ("sustained", "context4k")],
    }
    for group, names in grouped_names.items():
        selected = [per_case[name] for name in names]
        if not selected:
            continue
        delta = [row["baseline_modeled_verify_ms"] - row["gated_modeled_verify_ms"] for row in selected]
        round_delta = [row["baseline_rounds"] - row["gated_rounds"] for row in selected]
        wins = sum(x > 0 for x in round_delta)
        ties = sum(x == 0 for x in round_delta)
        losses = sum(x < 0 for x in round_delta)
        attempts = sum(row["branch_attempts"] for row in selected)
        catches = sum(row["branch_catches"] for row in selected)
        paid_rounds = sum(row["gated_nonterminal_draft_rounds"] for row in selected)
        groups[group] = {
            "prompts": len(selected),
            "tokens": sum(row["tokens"] for row in selected),
            "baseline_rounds": sum(row["baseline_rounds"] for row in selected),
            "gated_rounds": sum(row["gated_rounds"] for row in selected),
            "baseline_nonterminal_draft_rounds": sum(row["baseline_nonterminal_draft_rounds"] for row in selected),
            "gated_nonterminal_draft_rounds": sum(row["gated_nonterminal_draft_rounds"] for row in selected),
            "branch_attempts": attempts,
            "eligible_opportunities": sum(row["eligible_opportunities"] for row in selected),
            "branch_catches": catches,
            "catch_rate_per_attempt": catches / attempts if attempts else None,
            "baseline_misses_caught": sum(row["baseline_misses_caught"] for row in selected),
            "baseline_first_misses": sum(row["baseline_first_misses"] for row in selected),
            "lost_primary_tail": sum(row["lost_primary_tail"] for row in selected),
            "paired_prompt_round_direction": {"wins": wins, "ties": ties, "losses": losses},
            "mean_round_savings_per_prompt": statistics.mean(round_delta),
            "total_round_savings": sum(round_delta),
            "round_savings_per_all_candidate_draft_rounds": sum(round_delta) / paid_rounds if paid_rounds else None,
            "gated_attempts_per_all_candidate_draft_rounds": attempts / paid_rounds if paid_rounds else None,
            "mean_savings_ms_per_prompt": statistics.mean(delta),
            "total_modeled_verify_savings_ms": sum(delta),
            "mean_savings_ms_per_all_candidate_draft_round": sum(delta) / paid_rounds if paid_rounds else None,
            "mean_savings_ms_per_attempt": sum(delta) / attempts if attempts else None,
            "unconditional_rounds": sum(row["unconditional_rounds"] for row in selected),
            "unconditional_attempts": sum(row["unconditional_attempts"] for row in selected),
            "unconditional_catches": sum(row["unconditional_catches"] for row in selected),
            "unconditional_lost_primary_tail": sum(row["unconditional_lost_primary_tail"] for row in selected),
            "paired_prompt_round_savings": round_delta,
            "paired_prompt_savings_ms_secondary": delta,
        }
        if len(selected) > 1:
            groups[group]["bootstrap"] = paired_round_interval(selected, 20261125 + int(group == "coding"))
    return {
        "rule": {"primary_width": 3, "threshold_corrected_logit_gap": threshold,
                 "threshold_source": "prespecified ln(2)" if threshold == THRESHOLD else "caller override",
                 "alternate_slots_1_based": [1, 2], "fork_nodes_max": 3,
                 "near_end": "an in-range alternative can occupy a spare node at the output cap"},
        "target_price_ms_rows_1_to_4": list(prices),
        "pricing_scope": "modeled target verification only; branch-aware execution, draft/IPC/commit/state costs are unmeasured",
        "groups": groups,
        "cases": per_case,
    }


def discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(set(root.rglob("self_correction.json")) | set(root.rglob("selector.json")))


def self_test() -> None:
    base = {"case": "synthetic", "trace_sha256": "0" * 64,
            "teacher": [10, 11, 12, 13],
            "primary": [[99, 12, 13], [12, 13, 14], [13, 14, 0], [14, 0, 0]],
            "alternate": [[11, 12], [90, 90], [90, 90], [90, 90]],
            "gap": [[0.1, 0.1], [9, 9], [9, 9], [9, 9]]}
    # Slot 1 low-gap sibling catches the first primary miss, matching target.
    result = replay(base, "gated")
    assert result.branch_attempts == 1
    assert result.branch_catches == 1
    assert baseline_catches(base, THRESHOLD) == 1
    assert result.rounds + result.accepted == 4
    # Unconditional slot 1 makes the same attempt even when its gap is high.
    base["gap"][0] = [9, 9]
    assert replay(base, "gated").branch_attempts == 0
    assert replay(base, "unconditional").branch_attempts > 0
    # One remaining target token permits a slot-1 sibling without displacing
    # any in-budget primary work; it uses a three-row packet.
    tail = {"case": "tail", "trace_sha256": "1" * 64, "teacher": [1, 2],
            "primary": [[99, 0, 0], [0, 0, 0]], "alternate": [[2, 0], [0, 0]],
            "gap": [[0, 9], [9, 9]]}
    tail_walk = replay(tail, "gated")
    assert tail_walk.branch_attempts == 1 and tail_walk.branch_catches == 1
    assert tail_walk.packet_calls[2] == 1
    # A fully accepted three-token primary block loses its third node to a
    # fork, while slot 2 is eligible and the retained primary prefix stays exact.
    full = {"case": "full", "trace_sha256": "2" * 64, "teacher": [1, 2, 3, 4],
            "primary": [[2, 3, 4], [3, 4, 0], [4, 0, 0], [0, 0, 0]],
            "alternate": [[2, 3], [3, 4], [4, 0], [0, 0]],
            "gap": [[9, 0.1], [9, 9], [9, 9], [9, 9]]}
    full_walk = replay(full, "gated")
    assert full_walk.branch_attempts == 1 and full_walk.lost_primary_tail == 1
    assert baseline_catches(full, THRESHOLD) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="label", choices=("archived", "fresh"), required=True,
                        help="label kept in output to prevent silent pooling")
    parser.add_argument("path", type=Path, nargs="?", help="sidecar file or directory")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--row-prices-ms", type=float, nargs=4, default=DEFAULT_ROW_MS,
                        metavar=("ROW1", "ROW2", "ROW3", "ROW4"),
                        help="historical M5 Max verifier medians for target rows 1..4")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"self_test": "pass", "gpu_used": False}))
        return
    if args.path is None:
        parser.error("path is required unless --self-test is used")
    if args.threshold < 0 or any(value <= 0 for value in args.row_prices_ms):
        parser.error("threshold must be nonnegative and row prices positive")
    paths = discover(args.path)
    if not paths:
        parser.error("no self_correction.json sidecars found")
    records = [load_sidecar(path) for path in paths]
    cases = [record["case"] for record in records]
    if len(cases) != len(set(cases)):
        parser.error("case names must be unique within a set")
    result = analyze(records, args.threshold, tuple(args.row_prices_ms))
    result["set"] = args.label
    result["sidecars"] = len(records)
    result["sidecar_sha256"] = {record["case"]: hashlib.sha256(path.read_bytes()).hexdigest()
                                for record, path in zip(records, paths, strict=True)}
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
