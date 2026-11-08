"""Finish only unanswered Mercury questions from an interrupted sequential run."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from mercury import (
    Client,
    build_request,
    is_correct,
    parse_response,
    require,
    supported_cases,
    wire,
)
from compare import load_cases, quality, read, save, sha


def digest(cases):
    return hashlib.sha256(b"".join(wire(build_request(c)) for c in cases)).hexdigest()


def probe_cases(cases):
    selected = [
        next(c for c in cases if c["question"]["type"] == kind)
        for kind in ("choice", "noul", "score")
    ]
    extra = max(
        (c for c in cases if c["question"]["type"] == "choice"),
        key=lambda c: len(c["labels"]),
    )
    return (
        selected if extra["id"] in {c["id"] for c in selected} else [*selected, extra]
    )


def remaining_cases(cases, failed, probes):
    eligible, _ = supported_cases(cases)
    require(
        failed["status"] == "failed" and failed.get("http_status") == 429,
        "expected_rate_limited_run",
    )
    require(
        failed["request_population_sha256"] == digest(eligible),
        "prior_population_mismatch",
    )
    require(
        len(failed["rounds"]) == 1 and failed["rounds"][0]["round"] == 0,
        "prior_rounds_invalid",
    )
    completed = failed["rounds"][0]["completed"]
    require(
        type(completed) is int and 0 <= completed <= len(eligible),
        "prior_completion_invalid",
    )
    require(
        failed["physical_requests_succeeded"] == completed, "prior_coverage_invalid"
    )
    answered = {c["id"] for c in eligible[:completed]}
    for probe in probes:
        selected = probe_cases(cases)
        require(
            probe["probe"] is True and probe["status"] == "complete", "probe_incomplete"
        )
        require(
            probe["request_population_sha256"] == digest(selected),
            "probe_population_mismatch",
        )
        require(
            probe["physical_requests_succeeded"] == len(selected),
            "probe_coverage_invalid",
        )
        answered.update(c["id"] for c in selected)
    return (
        [c for c in eligible if c["id"] not in answered],
        len(answered),
        len(eligible),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--answered-probe", type=Path, action="append", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = None
    try:
        config = read(args.config)
        cases, dataset = load_cases(config["reference"])
        remaining, answered_count, eligible_count = remaining_cases(
            cases, read(args.failed_run), list(map(read, args.answered_probe))
        )
        key = args.key_file.read_text().strip()
        require(key and not any(c.isspace() for c in key), "key_invalid")
        with os.fdopen(
            os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as handle:
            handle.write("{}\n")
        report = {
            "schema": 1,
            "scope": "unanswered_questions_only",
            "status": "running",
            "dataset": dataset,
            "harness_sha256": sha(__file__),
            "client_sha256": sha(Path(__file__).with_name("mercury.py")),
            "failed_run_sha256": sha(args.failed_run),
            "probe_receipts_sha256": [sha(p) for p in args.answered_probe],
            "request_population_sha256": digest(remaining),
            "prior_unique_answered": answered_count,
            "eligible_cases": eligible_count,
            "planned_requests": len(remaining),
            "full_corpus_quality_recoverable": False,
            "reason": "Prior successful readouts were not retained after process failure.",
            "physical_requests_started": 0,
            "physical_requests_succeeded": 0,
            "minimum_dispatch_interval_seconds": 1.0,
            "retries": 0,
        }
        save(args.output, report)
        client = Client(key)
        probabilities, timings, usage = {}, [], collections.Counter()
        correct = collections.Counter()
        previous = 0.0
        for case in remaining:
            time.sleep(max(0.0, 1.0 - (time.monotonic() - previous)))
            report["physical_requests_started"] += 1
            save(args.output, report)
            previous = time.monotonic()
            payload, _ = client.request(wire(build_request(case)))
            timings.append(time.monotonic() - previous)
            p, tokens = parse_response(payload, case)
            probabilities[case["id"]] = p
            usage.update(tokens)
            tier = "synthetic" if case["tier"] == "synthetic" else "public"
            correct[tier] += is_correct(case, p)
            report["physical_requests_succeeded"] += 1
            report["correct_counts_on_remaining_only"] = dict(correct)
            report["usage"] = dict(usage)
            save(args.output, report)
        report.update(
            status="complete",
            unique_answered_after=answered_count + len(remaining),
            quality_on_remaining_only=quality(remaining, probabilities),
            estimated_input_charge_usd=usage["input_tokens"] * 0.02 / 1_000_000,
        )
        save(args.output, report)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "new_requests": len(remaining),
                    "unique_answered": report["unique_answered_after"],
                }
            )
        )
    except Exception as exc:
        if report is not None:
            report["status"] = "failed"
            report["failure_type"] = type(exc).__name__
            if hasattr(exc, "code") and type(exc.code) is int:
                report["http_status"] = exc.code
            report.update(getattr(exc, "safe_details", {}))
            save(args.output, report)
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
