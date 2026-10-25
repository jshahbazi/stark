"""Summarize a current multi-engine Qwen receipt without retaining payloads."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ENGINES = ("stark_optimized", "mlx_lm", "tensorfold_default", "tensorfold_plain", "llama_cpp", "vllm_metal")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    values = list(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive_values_required")
    return math.exp(statistics.mean(math.log(value) for value in values))


def fixture_map(receipt):
    return {row["name"]: row for row in receipt["fixtures"]}


def summarize(receipt):
    fixtures = fixture_map(receipt)
    performance = [name for name, row in fixtures.items() if row["population"] == "performance"]
    sanity = [name for name, row in fixtures.items() if row["population"] == "sanity"]
    coverage_ok = len(performance) == 6 and len(sanity) == 4 and len(receipt["fixtures"]) == 10
    blocks = receipt.get("blocks", [])
    by_engine = {engine: [b for b in blocks if b.get("engine") == engine] for engine in ENGINES}
    planned_ids = set(range(int(receipt["protocol"]["blocks"])))
    complete = (
        {b.get("engine") for b in blocks} == set(ENGINES)
        and all(len(by_engine[engine]) == len(planned_ids)
                and {b.get("block") for b in by_engine[engine]} == planned_ids for engine in ENGINES)
    )
    statuses = {engine: [b.get("status") for b in by_engine[engine]] for engine in ENGINES}
    all_complete = complete and all(all(status == "complete" for status in values) for values in statuses.values())
    timing_accepted = bool(receipt.get("timing_accepted"))
    sanity_coverage_ok = all(
        len([s for s in block.get("samples", []) if s.get("fixture") == fixture]) == 1
        and all(s.get("input_tokens") == fixtures[fixture]["input_tokens"]
                for s in block.get("samples", []) if s.get("fixture") == fixture)
        for block in blocks for fixture in sanity
    )
    public_allowed = timing_accepted and all_complete and coverage_ok and sanity_coverage_ok

    rates = {engine: {} for engine in ENGINES}
    hash_summary = {engine: {} for engine in ENGINES}
    for engine in ENGINES:
        for fixture in performance:
            medians = []
            median_by_block = {}
            rows = []
            for block in by_engine[engine]:
                if block.get("status") != "complete":
                    continue
                samples = [s for s in block.get("samples", []) if s.get("fixture") == fixture]
                expected = fixtures[fixture]
                eligible = [
                    s for s in samples
                    if s.get("fixed_work_eligible")
                    and s.get("input_tokens") == expected.get("input_tokens")
                    and s.get("output_tokens") == expected.get("output_limit")
                    and s.get("finish_reason") == "length"
                ]
                if len(eligible) != int(receipt["protocol"]["repeats"]) and all_complete:
                    raise ValueError("fixed_work_count_mismatch")
                if not eligible:
                    continue
                values = [s["output_tokens"] / s["wall_seconds"] for s in eligible]
                median_by_block[block["block"]] = statistics.median(values)
                medians.append(median_by_block[block["block"]])
                rows.extend(eligible)
            if len(medians) != int(receipt["protocol"]["blocks"]):
                rates[engine][fixture] = {
                    "geomean_tokens_per_second": None,
                    "per_load_median_tokens_per_second": medians,
                    "min_per_load_median_tokens_per_second": min(medians) if medians else None,
                    "max_per_load_median_tokens_per_second": max(medians) if medians else None,
                }
                hash_summary[engine][fixture] = {
                    "samples": len(rows),
                    "token_hashes": sorted({s.get("token_sha256") for s in rows if s.get("token_sha256")}),
                    "text_hashes": sorted({s.get("text_sha256") for s in rows if s.get("text_sha256")}),
                    "token_hash_stable": len({s.get("token_sha256") for s in rows if s.get("token_sha256")}) == 1 if rows else False,
                    "text_hash_stable": len({s.get("text_sha256") for s in rows if s.get("text_sha256")}) == 1 if rows else False,
                    "token_hash_comparable": engine not in {"tensorfold_default", "tensorfold_plain"},
                }
                continue
            rates[engine][fixture] = {
                "geomean_tokens_per_second": gm(medians),
                "per_load_median_tokens_per_second": medians,
                "per_block_median_tokens_per_second": median_by_block,
                "min_per_load_median_tokens_per_second": min(medians),
                "max_per_load_median_tokens_per_second": max(medians),
            }
            hash_summary[engine][fixture] = {
                "samples": len(rows),
                "token_hashes": sorted({s.get("token_sha256") for s in rows if s.get("token_sha256")}),
                "text_hashes": sorted({s.get("text_sha256") for s in rows if s.get("text_sha256")}),
                "token_hash_stable": len({s.get("token_sha256") for s in rows if s.get("token_sha256")}) == 1,
                "text_hash_stable": len({s.get("text_sha256") for s in rows if s.get("text_sha256")}) == 1,
                "token_hash_comparable": engine not in {"tensorfold_default", "tensorfold_plain"},
            }

    paired = {fixture: {} for fixture in performance}
    for engine in ENGINES:
        if engine == "stark_optimized":
            continue
        for fixture in performance:
            left = rates[engine][fixture].get("per_block_median_tokens_per_second", {})
            right = rates["stark_optimized"][fixture].get("per_block_median_tokens_per_second", {})
            common_blocks = sorted(set(left) & set(right))
            if not common_blocks:
                paired[fixture][engine] = {"per_block_engine_over_stark": [], "geomean_engine_over_stark": None}
                continue
            ratios = [left[block] / right[block] for block in common_blocks]
            paired[fixture][engine] = {"block_ids": common_blocks, "per_block_engine_over_stark": ratios, "geomean_engine_over_stark": gm(ratios)}

    overall = {
        engine: gm(
            ratio
            for fixture in performance
            for ratio in paired[fixture][engine]["per_block_engine_over_stark"]
        )
        for engine in ENGINES
        if engine != "stark_optimized"
        and all(paired[fixture][engine]["per_block_engine_over_stark"] for fixture in performance)
    }
    cross_engine = {
        "token_hash_agreement_counts": {},
        "text_hash_agreement_counts": {},
    }
    for engine in ENGINES:
        if engine == "stark_optimized":
            continue
        cross_engine["token_hash_agreement_counts"][engine] = None if engine.startswith("tensorfold") else sum(
            len(hash_summary["stark_optimized"][f]["token_hashes"]) == 1
            and hash_summary["stark_optimized"][f]["token_hashes"] == hash_summary[engine][f]["token_hashes"]
            for f in performance if hash_summary[engine][f]["token_hash_comparable"]
        )
        cross_engine["text_hash_agreement_counts"][engine] = sum(
            len(hash_summary["stark_optimized"][f]["text_hashes"]) == 1
            and hash_summary["stark_optimized"][f]["text_hashes"] == hash_summary[engine][f]["text_hashes"]
            for f in performance
        )

    sanity_summary = {}
    for engine in ENGINES:
        rows = [s for b in by_engine[engine] for s in b.get("samples", []) if s.get("fixture") in sanity]
        sanity_summary[engine] = {
            "samples": len(rows),
            "strict_exact_match_passes": sum(bool(s.get("strict_sanity_correct")) for s in rows),
            "strict_exact_match_total": len(rows),
            "per_fixture": {
                fixture: {
                    "passes": sum(bool(s.get("strict_sanity_correct")) for s in rows if s.get("fixture") == fixture),
                    "total": sum(1 for s in rows if s.get("fixture") == fixture),
                }
                for fixture in sanity
            },
            "token_hashes": sorted({s.get("token_sha256") for s in rows if s.get("token_sha256")}),
            "text_hashes": sorted({s.get("text_sha256") for s in rows if s.get("text_sha256")}),
        }

    return {
        "schema": 1,
        "summary_program_sha256": sha(Path(__file__)),
        "receipt_sha256": None,
        "engine_ids": list(ENGINES),
        "timing_accepted": timing_accepted,
        "all_blocks_complete": all_complete,
        "fixture_coverage_ok": coverage_ok,
        "sanity_coverage_ok": sanity_coverage_ok,
        "allow_public_table": public_allowed,
        "diagnostic_only_reason": None if public_allowed else "timing_or_completeness_gate_failed",
        "planned_blocks": receipt["protocol"]["blocks"],
        "realized_order": [[b.get("engine") for b in sorted(blocks, key=lambda x: x.get("block", -1)) if b.get("block") == block] for block in sorted({b.get("block") for b in blocks})],
        "block_statuses": statuses,
        "rates": rates if public_allowed else {},
        "diagnostic_rates": {} if public_allowed else rates,
        "paired_comparisons": paired if public_allowed else {},
        "diagnostic_paired_comparisons": {} if public_allowed else paired,
        "overall_engine_over_stark_geomean": overall if public_allowed else {},
        "hash_stability": hash_summary,
        "cross_engine_hash_agreement_performance_fixture_count": cross_engine,
        "sanity": sanity_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("summary_output_already_exists")
    receipt = json.loads(args.input.read_text())
    result = summarize(receipt)
    result["receipt_sha256"] = sha(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
