"""Recompute the report from source-bound, aggregate-only receipts."""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("summary_output_already_exists")
    analyzer = root.parent / "mlx-optimization-2026-10-20" / "analyze.py"
    spec = importlib.util.spec_from_file_location("paired_analysis", analyzer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = args.receipt
    raw = json.loads(receipt.read_text())
    analysis = module.analyze(receipt)
    comparison = analysis["paired_performance"][0]
    assert comparison["left"] == "stark_optimized"
    assert comparison["right"] == "mlx_lm"
    paired = comparison["paired_accepted_blocks"]
    result = {
        "summary_program_sha256": sha(Path(__file__)),
        "shared_analyzer_sha256": sha(analyzer),
        "receipt_sha256": sha(receipt),
        "analysis": analysis,
        "paired_accepted_blocks": paired,
        "overall_stark_gain_percent": 100
        * (1 / comparison["all_fixture_block_geomean_ratio"] - 1),
        "per_block_stark_gain_percent": {
            str(b): 100
            * (
                1
                / gm(
                    f["right_over_left_load_median_ratios"][str(b)]
                    for f in comparison["fixtures"].values()
                )
                - 1
            )
            for b in paired
        },
        "table": [],
        "sanity": {},
    }
    for fixture, data in comparison["fixtures"].items():
        rates = {
            e: gm(d["performance"][fixture]["load_medians"][str(b)] for b in paired)
            for e, d in analysis["engines"].items()
        }
        result["table"].append(
            {
                "fixture": fixture,
                "stark_tokens_per_second": rates["stark_optimized"],
                "mlx_tokens_per_second": rates["mlx_lm"],
                "stark_gain_percent": 100 * (1 / data["geomean_ratio"] - 1),
                "per_load_stark_gain_percent": {
                    b: 100 * (1 / v - 1)
                    for b, v in data["right_over_left_load_median_ratios"].items()
                },
            }
        )
    samples = [
        s
        for b in raw["blocks"]
        if b["block"] in paired
        for s in b["host_screen"]["samples"]
        if "measurement" in s["phases"]
    ]
    result["paired_host"] = {
        "samples": len(samples),
        "max_external_process_cpu_percent": max(
            s["max_process_cpu_percent"] for s in samples
        ),
        "max_aggregate_external_cpu_percent": max(
            s["external_cpu_percent"] for s in samples
        ),
        "max_sample_interval_seconds": max(s["interval_seconds"] for s in samples),
        "thermal_states": sorted({s["thermal_state"] for s in samples}),
        "paging_deltas": [
            b["paging_delta"] for b in raw["blocks"] if b["block"] in paired
        ],
    }
    for engine in analysis["engines"]:
        result["sanity"][engine] = {}
        for fixture in raw["fixtures"]:
            if fixture["population"] != "sanity":
                continue
            rows = [
                s
                for b in raw["blocks"]
                if b["engine"] == engine and b["block"] in paired
                for s in b["samples"]
                if s["fixture"] == fixture["name"]
            ]
            result["sanity"][engine][fixture["name"]] = {
                "samples": len(rows),
                "strict_exact_match_passes": sum(
                    s["strict_sanity_correct"] for s in rows
                ),
                "output_token_counts": sorted({s["output_tokens"] for s in rows}),
                "finish_reasons": sorted({s["finish_reason"] for s in rows}),
            }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
