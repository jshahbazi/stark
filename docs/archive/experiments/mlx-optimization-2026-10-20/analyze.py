"""Derive this experiment's tables from aggregate-only benchmark receipts."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def rate(row):
    return row["output_tokens"] / row["wall_seconds"]


def rows_for(blocks, fixture):
    return [s for b in blocks for s in b["samples"] if s["fixture"] == fixture]


def analyze(path):
    report = json.loads(path.read_text())
    ids = [e["id"] for e in report["engines"]]
    fixture_names = [f["name"] for f in report["fixtures"]]
    performance = [
        f["name"] for f in report["fixtures"] if f["population"] == "performance"
    ]
    accepted = {
        e: {
            b["block"]: b
            for b in report["blocks"]
            if b["engine"] == e and b["status"] == "complete"
        }
        for e in ids
    }
    common = sorted(set.intersection(*(set(accepted[e]) for e in ids)))
    result = {
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": report["status"],
        "timing_accepted": report["timing_accepted"],
        "failures": report["failures"],
        "all_engine_common_accepted_blocks": common,
        "engines": {},
        "cross_engine": [],
        "paired_performance": [],
    }
    hashes = {}
    for engine in ids:
        blocks = [b for b in report["blocks"] if b["engine"] == engine]
        hashes[engine] = {
            f: {
                key: sorted({s[key] for s in rows_for(blocks, f)})
                for key in ("token_sha256", "text_sha256")
            }
            for f in fixture_names
        }
        result["engines"][engine] = {
            "block_statuses": {str(b["block"]): b["status"] for b in blocks},
            "samples_by_fixture_all_blocks": {
                f: len(rows_for(blocks, f)) for f in fixture_names
            },
            "hashes": hashes[engine],
            "performance": {},
        }
        for fixture in performance:
            accepted_rows = rows_for(accepted[engine].values(), fixture)
            common_rows = rows_for([accepted[engine][b] for b in common], fixture)
            result["engines"][engine]["performance"][fixture] = {
                "accepted_samples": len(accepted_rows),
                "accepted_median_tokens_per_second": statistics.median(
                    map(rate, accepted_rows)
                )
                if accepted_rows
                else None,
                "common_samples": len(common_rows),
                "common_median_tokens_per_second": statistics.median(
                    map(rate, common_rows)
                )
                if common_rows
                else None,
                "non_fixed_work_samples_all_blocks": sum(
                    not s["fixed_work_eligible"] for s in rows_for(blocks, fixture)
                ),
                "load_medians": {
                    str(b): statistics.median(map(rate, rows_for([block], fixture)))
                    for b, block in accepted[engine].items()
                },
            }
    for left, right in itertools.combinations(ids, 2):
        equality = {"left": left, "right": right}
        for key in ("token_sha256", "text_sha256"):
            equality[key] = {
                "matching_fixtures": [
                    f
                    for f in fixture_names
                    if len(hashes[left][f][key]) == 1
                    and hashes[left][f][key] == hashes[right][f][key]
                ],
                "mismatch_or_missing_fixtures": [
                    f
                    for f in fixture_names
                    if len(hashes[left][f][key]) != 1
                    or hashes[left][f][key] != hashes[right][f][key]
                ],
            }
        result["cross_engine"].append(equality)
        paired = sorted(set(accepted[left]) & set(accepted[right]))
        comparison = {
            "left": left,
            "right": right,
            "paired_accepted_blocks": paired,
            "fixtures": {},
        }
        for fixture in performance:
            ratios = {}
            for block in paired:
                a = rows_for([accepted[left][block]], fixture)
                b = rows_for([accepted[right][block]], fixture)
                if a and b and all(s["fixed_work_eligible"] for s in a + b):
                    ratios[str(block)] = statistics.median(
                        map(rate, b)
                    ) / statistics.median(map(rate, a))
            comparison["fixtures"][fixture] = {
                "right_over_left_load_median_ratios": ratios,
                "geomean_ratio": geomean(list(ratios.values())),
            }
        comparison["all_fixture_block_geomean_ratio"] = geomean(
            [
                ratio
                for fixture in comparison["fixtures"].values()
                for ratio in fixture["right_over_left_load_median_ratios"].values()
            ]
        )
        result["paired_performance"].append(comparison)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": 1,
        "analysis_harness_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "rate_definition": "actual output tokens / complete client request seconds",
        "ratio_definition": "geometric mean of paired-load median-rate ratios; performance fixtures only",
        "receipts": {p.name: analyze(p) for p in args.receipts},
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
