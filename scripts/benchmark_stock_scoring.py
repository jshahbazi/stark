"""Replay sealed JevBench inputs against stock scoring APIs; aggregates only."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_stock_engines import (
    Engine,
    HostMonitor,
    public_spec,
    save,
    sha,
    sleep_events,
    stop,
)
from stock_compare_cases import load_reference


def calibrated(logits, temperature=1.22):
    high = max(logits)
    values = [math.exp((x - high) / temperature) for x in logits]
    total = math.fsum(values)
    return [x / total for x in values]


def score(engine, case, context):
    ids, candidates = case["input_token_ids"], case["candidate_ids"]
    started = time.perf_counter()
    if engine.kind == "stark_score":
        result = engine.pipe.request(
            {
                "operation": "score",
                "input_token_ids": ids,
                "candidate_token_ids": candidates,
                "context_tokens": context,
                "temperature": 1.22,
            }
        )
        logits, logp = result["logits"], result["log_probabilities"]
        native_probs = result["candidate_probabilities"]
    elif engine.kind == "llama_score":
        line = " ".join(
            map(
                str,
                ["score", context, 1.22, len(ids), len(candidates), *ids, *candidates],
            )
        )
        result = engine.pipe.request(line)
        logits, logp = result["logits"], result["log_probabilities"]
        native_probs = result["candidate_probabilities"]
    elif engine.kind == "mlx":
        result = engine.pipe.request(
            {
                "operation": "score",
                "input_token_ids": ids,
                "candidate_ids": candidates,
                "context_tokens": context,
                "temperature": 1.22,
            }
        )
        logits, logp = result["candidate_logits"], result["candidate_log_probabilities"]
        native_probs = None
    else:
        raise RuntimeError("scoring_unsupported")
    elapsed = time.perf_counter() - started
    if (
        result["input_tokens"] != len(ids)
        or len(logits) != len(candidates)
        or len(logp) != len(candidates)
    ):
        raise RuntimeError("scoring_coverage_mismatch")
    if not all(math.isfinite(x) for x in logits + logp) or any(x > 1e-6 for x in logp):
        raise RuntimeError("scoring_nonfinite")
    probabilities = calibrated(logits)
    if (
        native_probs is not None
        and max(abs(x - y) for x, y in zip(native_probs, probabilities, strict=True))
        > 1e-6
    ):
        raise RuntimeError("calibration_formula_mismatch")
    return elapsed, probabilities, logits, logp


def quality(cases, probabilities):
    public = []
    synthetic = []
    for case in cases:
        p = probabilities[case["id"]]
        index = max(range(len(p)), key=p.__getitem__)
        labels = list(map(str, case["labels"]))
        gold = str(case["expected"])
        row = {
            "correct": labels[index] == gold,
            "confidence": max(p),
            "brier": math.fsum(
                (value - (label == gold)) ** 2
                for value, label in zip(p, labels, strict=True)
            ),
        }
        (synthetic if case["tier"] == "synthetic" else public).append(row)

    def aggregate(rows):
        ece = 0.0
        for index in range(10):
            bucket = [r for r in rows if min(9, int(r["confidence"] * 10)) == index]
            if bucket:
                ece += (
                    len(bucket)
                    / len(rows)
                    * abs(
                        statistics.mean(r["confidence"] for r in bucket)
                        - statistics.mean(r["correct"] for r in bucket)
                    )
                )
        return {
            "cases": len(rows),
            "argmax_correct": sum(r["correct"] for r in rows),
            "brier": statistics.mean(r["brier"] for r in rows),
            "ece_10_bins": ece,
        }

    return {"public": aggregate(public), "synthetic": aggregate(synthetic)}


def run(config, output, probe):
    cases, provenance = load_reference(Path(config["reference"]))
    provenance.pop("reference_path", None)
    selected = cases if not probe else [cases[i] for i in (0, 60, 120, 180)]
    specs = config["engines"]
    report = {
        "schema": 1,
        "phase": "scoring_admission" if probe else "scoring",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "cases_loader_sha256": sha(Path(__file__).with_name("stock_compare_cases.py")),
        "dataset": provenance,
        "model": "alibiserikbay/JevK5",
        "model_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
        "protocol": {
            "context_tokens": 8192,
            "temperature": 1.22,
            "blocks": 1 if probe else config["blocks"],
            "cases_per_block": len(selected),
            "cache": "fresh native state per score, output caching off",
            "host_screen_scope": "sampled process CPU and whole-window sleep; continuous GPU/thermal/power isolation unverified",
            "timing": "private pipe complete scoring request; model load excluded; context initialization/readout/cleanup included",
        },
        "engines": [public_spec(s) for s in specs],
        "blocks": [],
        "failures": [],
    }
    save(output, report)
    all_probs = {}
    reference_values = None
    repeatability = {}
    comparisons = []
    wall_start, mono_start = time.time(), time.monotonic()
    with contextlib.ExitStack() as outer:
        assertion = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(__import__("os").getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        outer.callback(stop, assertion)
        for block in range(1 if probe else config["blocks"]):
            order = specs if block % 2 == 0 else list(reversed(specs))
            for spec in order:
                row = {"engine": spec["id"], "block": block, "status": "running"}
                report["blocks"].append(row)
                try:
                    with contextlib.ExitStack() as stack, HostMonitor() as monitor:
                        engine = Engine(spec, stack, config["task_root"], 8192)
                        row["load_seconds"] = engine.load_seconds
                        row["runtime"] = engine.metadata
                        for case in selected[:3]:
                            score(engine, case, 8192)
                        values, probabilities, timings = {}, {}, []
                        for case in selected:
                            elapsed, p, logits, logp = score(engine, case, 8192)
                            timings.append(
                                {
                                    "input_tokens": len(case["input_token_ids"]),
                                    "wall_seconds": elapsed,
                                }
                            )
                            probabilities[case["id"]] = p
                            values[case["id"]] = (p, logits, logp)
                        row["timings"] = timings
                        row["cases"] = len(selected)
                        row["median_wall_seconds"] = statistics.median(
                            t["wall_seconds"] for t in timings
                        )
                        row["buckets"] = {}
                        for low, high in ((0, 256), (256, 1024), (1024, 8192)):
                            times = [
                                t["wall_seconds"]
                                for t in timings
                                if low <= t["input_tokens"] < high
                            ]
                            row["buckets"][f"{low}-{high - 1}"] = {
                                "count": len(times),
                                "median_seconds": statistics.median(times)
                                if times
                                else None,
                            }
                        if not probe:
                            row["quality"] = quality(cases, probabilities)
                        if spec["id"] in all_probs:
                            repeatability[spec["id"]] = max(
                                repeatability.get(spec["id"], 0.0),
                                max(
                                    abs(a - b)
                                    for key in probabilities
                                    for a, b in zip(
                                        probabilities[key],
                                        all_probs[spec["id"]][key],
                                        strict=True,
                                    )
                                ),
                            )
                        else:
                            all_probs[spec["id"]] = probabilities
                        if block == 0:
                            if reference_values is None:
                                reference_values = (spec["id"], values)
                            else:
                                differences = []
                                logit_differences = []
                                logp_differences = []
                                agreements = 0
                                for key, (p, logits, logp) in values.items():
                                    refp, reflogits, reflogp = reference_values[1][key]
                                    differences.extend(
                                        abs(a - b) for a, b in zip(p, refp, strict=True)
                                    )
                                    logit_differences.extend(
                                        abs(a - b)
                                        for a, b in zip(logits, reflogits, strict=True)
                                    )
                                    logp_differences.extend(
                                        abs(a - b)
                                        for a, b in zip(logp, reflogp, strict=True)
                                    )
                                    agreements += max(
                                        range(len(p)), key=p.__getitem__
                                    ) == max(range(len(refp)), key=refp.__getitem__)
                                comparisons.append(
                                    {
                                        "reference": reference_values[0],
                                        "candidate": spec["id"],
                                        "cases": len(values),
                                        "candidate_argmax_agreement": agreements,
                                        "maximum_probability_difference": max(
                                            differences
                                        ),
                                        "mean_probability_difference": statistics.mean(
                                            differences
                                        ),
                                        "maximum_candidate_logit_difference": max(
                                            logit_differences
                                        ),
                                        "maximum_candidate_log_probability_difference": max(
                                            logp_differences
                                        ),
                                    }
                                )
                        row["status"] = "complete"
                    row["host_screen"] = monitor.result()
                    if not probe and (
                        row["host_screen"]["cpu_guard_exceeded"]
                        or row["host_screen"]["background_model_active"]
                        or row["host_screen"]["sampling_failures"]
                    ):
                        row["status"] = "excluded_interference"
                except Exception as exc:
                    safe = (
                        str(exc)
                        if isinstance(exc, RuntimeError)
                        and str(exc).replace("_", "").isalnum()
                        else "scoring_operation_failed"
                    )
                    row.update(status="failed", error=safe)
                    report["failures"].append(
                        {"engine": spec["id"], "block": block, "error": safe}
                    )
                save(output, report)
                print(
                    json.dumps(
                        {
                            "completed_engine": spec["id"],
                            "block": block,
                            "status": row["status"],
                            "cases": row.get("cases"),
                            "error": row.get("error"),
                        }
                    ),
                    flush=True,
                )
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    report["wall_elapsed_seconds"] = time.time() - wall_start
    report["monotonic_elapsed_seconds"] = time.monotonic() - mono_start
    report["power_log_screen"] = sleep_events(wall_start, time.time())
    report["repeatability_max_probability_difference"] = repeatability
    report["comparisons"] = comparisons
    report["timing_accepted"] = (
        report["power_log_screen"]["checked"]
        and report["power_log_screen"]["sleep_events"] == 0
        and abs(report["wall_elapsed_seconds"] - report["monotonic_elapsed_seconds"])
        < 2
        and all(b["status"] == "complete" for b in report["blocks"])
    )
    report["status"] = "complete_with_failures" if report["failures"] else "complete"
    save(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text()), args.output, args.probe)
    except Exception:
        raise SystemExit("stock_scoring_benchmark_failed") from None
