"""Describe all acquired fixed-work results without quiet-host certification."""

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(receipt):
    if receipt["status"] != "complete" or receipt["failures"]:
        raise ValueError("complete_acquisition_required")
    if receipt["protocol"].get("require_quiet_windows", True):
        raise ValueError("explicit_observed_mode_required")
    fixtures = {f["name"]: f for f in receipt["fixtures"]}
    performance = [n for n, f in fixtures.items() if f["population"] == "performance"]
    sanity = [n for n, f in fixtures.items() if f["population"] == "sanity"]
    expected_engines = {
        "stark_optimized", "mlx_lm", "llama_cpp", "tensorfold_default",
        "tensorfold_plain", "vllm_metal",
    }
    if (len(performance), len(sanity), len(receipt["fixtures"])) != (6, 4, 10):
        raise ValueError("fixture_coverage_mismatch")
    if {e["id"] for e in receipt["engines"]} != expected_engines:
        raise ValueError("engine_coverage_mismatch")
    block_ids = set(range(receipt["protocol"]["blocks"]))
    if len(receipt["blocks"]) != len(expected_engines) * len(block_ids):
        raise ValueError("load_count_mismatch")
    result = {
        "schema": 1,
        "mode": "observed_busy_host",
        "timing_accepted": False,
        "allow_public_clean_table": False,
        "estimator": "median output tokens/client seconds per load and workload; geometric mean across four loads, then six workloads",
        "includes_host_flagged_loads": True,
        "engines": {},
    }
    for spec in receipt["engines"]:
        engine = spec["id"]
        loads = [b for b in receipt["blocks"] if b["engine"] == engine]
        if len(loads) != len(block_ids) or {b["block"] for b in loads} != block_ids:
            raise ValueError("block_coverage_mismatch")
        if any(b["status"] not in {"complete", "excluded_interference"} for b in loads):
            raise ValueError("incomplete_load")
        per_fixture = {}
        for name in performance:
            expected = fixtures[name]
            medians = {}
            for load in loads:
                samples = [s for s in load["samples"] if s["fixture"] == name]
                if len(samples) != receipt["protocol"]["repeats"]:
                    raise ValueError("repeat_count_mismatch")
                if {s["repeat"] for s in samples} != set(range(receipt["protocol"]["repeats"])):
                    raise ValueError("repeat_coverage_mismatch")
                if any(
                    not s["fixed_work_eligible"]
                    or s["input_tokens"] != expected["input_tokens"]
                    or s["output_tokens"] != expected["output_limit"]
                    or s["finish_reason"] != "length"
                    or not math.isfinite(s["wall_seconds"])
                    or s["wall_seconds"] <= 0
                    for s in samples
                ):
                    raise ValueError("fixed_work_mismatch")
                medians[load["block"]] = statistics.median(
                    s["output_tokens"] / s["wall_seconds"] for s in samples
                )
            per_fixture[name] = {
                "geomean_tokens_per_second": statistics.geometric_mean(medians.values()),
                "per_block_median_tokens_per_second": medians,
                "min_load_median": min(medians.values()),
                "max_load_median": max(medians.values()),
            }
        sanity_counts = {}
        for name in sanity:
            samples = []
            for load in loads:
                rows = [s for s in load["samples"] if s["fixture"] == name]
                if len(rows) != 1 or rows[0]["input_tokens"] != fixtures[name]["input_tokens"]:
                    raise ValueError("sanity_coverage_mismatch")
                if not isinstance(rows[0].get("strict_sanity_correct"), bool):
                    raise ValueError("sanity_result_missing")
                samples.extend(rows)
            sanity_counts[name] = {
                "passes": sum(s["strict_sanity_correct"] for s in samples),
                "samples": len(samples),
            }
        for load in loads:
            if len(load["samples"]) != len(performance) * receipt["protocol"]["repeats"] + len(sanity):
                raise ValueError("sample_count_mismatch")
        measured = [
            row for load in loads for row in load["host_screen"]["samples"]
            if "measurement" in row["phases"]
        ]
        result["engines"][engine] = {
            "identity": spec,
            "load_statuses": dict(collections.Counter(b["status"] for b in loads)),
            "workloads": per_fixture,
            "overall_geomean_tokens_per_second": statistics.geometric_mean(
                f["geomean_tokens_per_second"] for f in per_fixture.values()
            ),
            "sanity": sanity_counts,
            "measurement_max_external_cpu_percent": max(r["external_cpu_percent"] for r in measured),
            "measurement_max_thermal_state": max(r["thermal_state"] for r in measured),
            "paging_delta_totals": {
                k: sum(b["paging_delta"][k] for b in loads) for k in ("pageouts", "swapouts")
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("summary_output_already_exists")
    result = summarize(json.loads(args.input.read_text()))
    result["receipt_sha256"] = digest(args.input)
    result["summary_program_sha256"] = digest(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
