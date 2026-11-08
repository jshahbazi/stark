"""Evaluate hosted Mercury Decide on sealed JevBench; write aggregates only."""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from compare import (
    Pipe,
    align,
    clean_environment,
    drex_request,
    load_cases,
    paired,
    quality,
    read,
    save,
    sha,
)

ENDPOINT = "https://api.inceptionlabs.ai/v1/decisions"
MODEL = "mercury-decide"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def wire(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def build_request(case):
    return {"model": MODEL, **drex_request(case)}


def is_correct(case, probabilities):
    selected = case["labels"][
        max(range(len(probabilities)), key=probabilities.__getitem__)
    ]
    return str(selected) == str(case["expected"])


def supported_cases(cases):
    included, excluded = [], collections.Counter()
    for case in cases:
        if case["question"]["type"] == "score" and len(case["labels"]) > 10:
            require(case["tier"] == "synthetic", "public_case_unsupported")
            excluded["synthetic_score_over_10_levels"] += 1
        else:
            included.append(case)
    return included, dict(excluded)


def parse_response(payload, case):
    require(
        isinstance(payload, dict) and payload.get("model") == MODEL,
        "served_model_invalid",
    )
    answers = payload.get("answers")
    require(
        isinstance(answers, dict) and set(answers) == {"decision"},
        "answer_names_invalid",
    )
    answer = answers["decision"]
    kind = case["question"]["type"]
    require(
        isinstance(answer, dict) and answer.get("type") == kind, "answer_type_invalid"
    )
    if kind == "noul":
        positive = answer.get("noul")
        require(
            type(positive) in (int, float)
            and math.isfinite(positive)
            and 0 <= positive <= 1,
            "boolean_invalid",
        )
        probabilities = align(
            case["labels"], ["false", "true"], [1 - positive, positive]
        )
    else:
        values = answer.get("probabilities")
        require(isinstance(values, dict), "readout_invalid")
        probabilities = align(case["labels"], values.keys(), list(values.values()))
        confidence = answer.get("confidence")
        require(
            type(confidence) in (int, float)
            and math.isfinite(confidence)
            and 0 <= confidence <= 1,
            "confidence_invalid",
        )
        if kind == "choice":
            choice = answer.get("choice")
            require(
                choice in values and values[choice] >= max(values.values()) - 1e-12,
                "choice_argmax_invalid",
            )
        else:
            score = answer.get("score")
            expected_score = math.fsum(
                int(key) * value for key, value in values.items()
            )
            require(
                type(score) in (int, float)
                and math.isfinite(score)
                and abs(score - expected_score) <= 0.002,
                "score_readout_invalid",
            )
    usage = payload.get("usage")
    require(
        isinstance(usage, dict)
        and type(usage.get("input_tokens")) is int
        and usage["input_tokens"] > 0
        and type(usage.get("output_tokens")) is int
        and usage["output_tokens"] >= 0,
        "usage_invalid",
    )
    return probabilities, {key: usage[key] for key in ("input_tokens", "output_tokens")}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, key):
        self.key = key
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, body):
        request = Request(
            ENDPOINT,
            data=body,
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                raw = response.read(2_000_001)
        except HTTPError as exc:
            exc.safe_details = {}
            retry = exc.headers.get("Retry-After", "")
            if retry.isdigit():
                exc.safe_details["retry_after_seconds"] = min(int(retry), 86400)
            try:
                error = json.loads(exc.read(2_000_001)).get("error", {})
                for field in ("code", "type"):
                    value = error.get(field)
                    if value in {
                        "rate_limit_exceeded",
                        "insufficient_quota",
                        "quota_exceeded",
                        "account_error",
                        "rate_limit_error",
                        "upstream_error",
                        "model_not_found",
                        "invalid_request_error",
                    }:
                        exc.safe_details["provider_error_" + field] = value
            except Exception:
                pass
            raise
        require(len(raw) <= 2_000_000, "response_too_large")
        return json.loads(raw), hashlib.sha256(raw).digest()


def local_jev(config, config_path, cases, probe):
    worker_cases, _ = load_cases(config["reference"])
    if probe:
        worker_cases = [worker_cases[i] for i in (0, 60, 120, 180)]
    positions = {case["id"]: i for i, case in enumerate(worker_cases)}
    command = [
        sys.executable,
        str(Path(__file__).with_name("compare.py").resolve()),
        "--config",
        str(config_path.resolve()),
        "--worker",
        "jev",
    ]
    if probe:
        command.append("--probe")
    with contextlib.ExitStack() as stack:
        pipe = Pipe(command, stack, Path.cwd(), clean_environment())
        # Worker indexes refer to the original sealed population (or the four-case probe).
        for index in range(min(3, len(cases))):
            pipe.request({"index": positions[cases[index]["id"]]})
        probabilities = {
            case["id"]: pipe.request({"index": positions[case["id"]]})["probabilities"]
            for case in cases
        }
    return probabilities


def run(args):
    key = os.environ.get("INCEPTION_API_KEY")
    if args.key_file:
        key = args.key_file.read_text().strip()
    require(
        isinstance(key, str) and bool(key) and not any(c.isspace() for c in key),
        "inception_key_missing_or_invalid",
    )
    config = read(args.config)
    all_cases, dataset = load_cases(config["reference"])
    if args.probe:
        # Deliberately exercise every supported question type plus the largest choice.
        cases = [
            next(c for c in all_cases if c["question"]["type"] == kind)
            for kind in ("choice", "noul", "score")
        ]
        extra = max(
            (c for c in all_cases if c["question"]["type"] == "choice"),
            key=lambda c: len(c["labels"]),
        )
        if extra["id"] not in {c["id"] for c in cases}:
            cases.append(extra)
    else:
        cases, excluded = supported_cases(all_cases)
    bodies = {case["id"]: wire(build_request(case)) for case in cases}
    for case in cases:
        if case["question"]["type"] == "score":
            require(len(case["labels"]) <= 10, "score_exceeds_provider_limit")
    request_digest = hashlib.sha256(
        b"".join(bodies[c["id"]] for c in cases)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as handle:
        handle.write("{}\n")
    report = {
        "schema": 1,
        "status": "running",
        "probe": args.probe,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "coverage": {
            "public_cases": sum(c["tier"] != "synthetic" for c in cases),
            "synthetic_cases": sum(c["tier"] == "synthetic" for c in cases),
            "excluded": {} if args.probe else excluded,
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "case_loader_sha256": sha(Path(__file__).with_name("compare.py")),
        "dependency_lock_sha256": sha(Path(__file__).with_name("uv.lock")),
        "request_population_sha256": request_digest,
        "protocol": {
            "rounds": 1 if args.probe else args.rounds,
            "cases_per_round": len(cases),
            "concurrency": 1,
            "max_rate_limit_retries_per_question": args.rate_limit_retries,
            "retry_scope": "HTTP 429 only; successful questions are never retried",
            "minimum_dispatch_interval_seconds": args.min_interval,
            "local_cache": "none",
            "question_order": "original supplied criteria order",
            "timing": "HTTP client wall time including connection setup, network and hosted provider; excludes response validation",
            "model_identity": "hosted alias only; no immutable weight digest",
            "estimated_input_usd_per_million": 0.02,
            "limits": "reused public development corpus; hosted latency differs from prepared-input local scoring",
        },
        "physical_requests_started": 0,
        "physical_requests_succeeded": 0,
        "rounds": [],
    }
    save(args.output, report)
    client = Client(key)
    first = None
    try:
        if not args.probe and not args.skip_local_baseline:
            model = Path(config["jev_model"])
            conversion = read(model / "conversion.json")
            require(
                conversion["source_revision"]
                == "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
                "jev_checkpoint_invalid",
            )
            require(
                sha(model / "model.safetensors") == conversion["output_sha256"],
                "jev_weight_hash_invalid",
            )
            report["jev_baseline"] = {
                "worker_sha256": sha(config["jev_worker"]),
                "weight_sha256": conversion["output_sha256"],
                "checkpoint_revision": conversion["source_revision"],
                "config_sha256": sha(model / "config.json"),
                "tokenizer_sha256": sha(model / "tokenizer.json"),
                "cache": "disabled",
                "temperature": 1.22,
            }
            jev = local_jev(config, args.config, cases, False)
            report["jev_baseline"]["quality"] = quality(cases, jev)
        for round_number in range(report["protocol"]["rounds"]):
            row = {"round": round_number, "status": "running", "completed": 0}
            report["rounds"].append(row)
            probabilities, timings, usage, response_hashes = (
                {},
                [],
                collections.Counter(),
                hashlib.sha256(),
            )
            previous_dispatch = 0.0
            progress = {
                tier: {"completed": 0, "correct": 0} for tier in ("public", "synthetic")
            }
            for case in cases:
                time.sleep(
                    max(0.0, args.min_interval - (time.monotonic() - previous_dispatch))
                )
                for attempt in range(args.rate_limit_retries + 1):
                    report["physical_requests_started"] += 1
                    save(args.output, report)
                    started = time.perf_counter()
                    previous_dispatch = time.monotonic()
                    try:
                        payload, response_digest = client.request(bodies[case["id"]])
                        break
                    except HTTPError as exc:
                        details = getattr(exc, "safe_details", {})
                        if (
                            exc.code != 429
                            or attempt == args.rate_limit_retries
                            or details.get("provider_error_code")
                            in {"insufficient_quota", "quota_exceeded"}
                        ):
                            raise
                        report["rate_limit_events"] = (
                            report.get("rate_limit_events", 0) + 1
                        )
                        save(args.output, report)
                        print(
                            json.dumps(
                                {"rate_limited": True, "completed": row["completed"]}
                            ),
                            flush=True,
                        )
                        time.sleep(max(60, details.get("retry_after_seconds", 0)))
                timings.append(time.perf_counter() - started)
                p, tokens = parse_response(payload, case)
                response_hashes.update(response_digest)
                probabilities[case["id"]] = p
                usage.update(tokens)
                row["usage"] = dict(usage)
                report["physical_requests_succeeded"] += 1
                row["completed"] += 1
                tier = "synthetic" if case["tier"] == "synthetic" else "public"
                progress[tier]["completed"] += 1
                progress[tier]["correct"] += is_correct(case, p)
                row["accuracy_progress"] = progress
                save(args.output, report)
                if row["completed"] % 25 == 0:
                    print(
                        json.dumps(
                            {"round": round_number, "completed": row["completed"]}
                        ),
                        flush=True,
                    )
            row.update(
                status="complete",
                median_http_seconds=statistics.median(timings),
                p95_http_seconds=sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
                total_http_seconds=math.fsum(timings),
                usage=dict(usage),
                response_population_sha256=response_hashes.hexdigest(),
                estimated_input_charge_usd=usage["input_tokens"] * 0.02 / 1_000_000,
            )
            if not args.probe:
                row["quality"] = quality(cases, probabilities)
            if not args.probe and not args.skip_local_baseline:
                row["paired_vs_current_jev"] = {
                    tier: {
                        key.replace("drex", "mercury"): value
                        for key, value in result.items()
                    }
                    for tier, result in paired(cases, jev, probabilities).items()
                }
            if first is None:
                first = probabilities
            else:
                report["repeatability_max_probability_delta"] = max(
                    abs(a - b)
                    for c in cases
                    for a, b in zip(first[c["id"]], probabilities[c["id"]], strict=True)
                )
                report["repeatability_argmax_changes"] = sum(
                    max(range(len(first[c["id"]])), key=first[c["id"]].__getitem__)
                    != max(
                        range(len(probabilities[c["id"]])),
                        key=probabilities[c["id"]].__getitem__,
                    )
                    for c in cases
                )
            save(args.output, report)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        if isinstance(exc, HTTPError):
            report["http_status"] = exc.code
            report.update(getattr(exc, "safe_details", {}))
        if report["rounds"] and report["rounds"][-1]["status"] == "running":
            report["rounds"][-1]["status"] = "failed"
        raise
    finally:
        save(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    parser.add_argument("--rate-limit-retries", type=int, choices=range(4), default=3)
    parser.add_argument("--skip-local-baseline", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.min_interval) or args.min_interval < 0:
        parser.error("--min-interval must be finite and nonnegative")
    try:
        report = run(args)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "physical_requests": report["physical_requests_started"],
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
