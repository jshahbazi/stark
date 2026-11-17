#!/usr/bin/env python3
"""Attribute the cost of fresh 12K prefix capture on the aligned and midpoint paths."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import secrets
import signal
from statistics import median
import subprocess
import time
import traceback

from benchmark import ROOT, Worker, digest_ids, generate, hash_file, host_snapshot, record, validate
from cold_fixtures import make_prompts
from cold_summary import host_stats
from fresh_coding import ARTIFACT, CONTEXT, DRAFTS, REVISION, SAMPLING, SEED, VERIFIER
from fresh_coding import require, snapshot_artifact, validate_ready, work_signature

LENGTH = 12288
BUDGET = 96
SCHEDULE = (4096, 2048, 2048, 4096)
PHYSICAL = 4096
PROFILE_SCOPE = "native_hybrid_aggregate_profile"
PHASE_BOUNDARY = "after_final_prompt_model_readout_before_initial_sampling"
SNAPSHOT = "qwen.mixed.prefill.snapshot."
ATTRIBUTION = SNAPSHOT + "attribution"


def focused_metrics(row):
    prefill = row["phase_counters"]["prefill"]

    def seconds(category, *, absent_zero=False):
        counter = prefill.get(category)
        return counter["seconds"] if counter is not None else 0 if absent_zero else None

    def snapshot_seconds(name):
        return seconds(SNAPSHOT + name, absent_zero=True)

    components = {
        "first_sample_wall_seconds": row["prefill_seconds"],
        "midpoint_cpu_allocate_zero_seconds": snapshot_seconds("midpoint.allocate_zero"),
        "midpoint_cpu_history_seconds": snapshot_seconds("midpoint.history"),
        "aligned_cpu_allocate_zero_seconds": snapshot_seconds("aligned.allocate_zero"),
        "aligned_blit_host_wall_seconds": snapshot_seconds("aligned.blit_wall"),
        "aligned_blit_gpu_seconds": (None if prefill.get(SNAPSHOT + "aligned.blit_missing_timestamps", {}).get("calls", 0)
                                      else snapshot_seconds("aligned.blit_gpu")),
        **{f"prefill_phase_{name}_seconds": seconds(f"qwen.mixed.prefill.{name}")
           for name in ("allocate", "ngram_stage", "encode", "gpu", "wait", "gpu_busy")},
    }
    chunk_gpu = [seconds(f"qwen.mixed.prefill.chunk.{index:02d}.gpu") for index in range(3)]
    if any(prefill.get(SNAPSHOT + name, {}).get("calls", 0)
           for name in ("missing_gpu_timestamps", "unindexed_chunks")):
        chunk_gpu = [None, None, None]
    components["final_physical_chunk_gpu_seconds"] = chunk_gpu[2]
    components["untouched_physical_chunks_gpu_seconds"] = (sum(chunk_gpu[:2]) if all(
        value is not None for value in chunk_gpu[:2]) else None)
    return components


def available_contrast(left, right):
    return {key: left[key] - right[key] for key in sorted(left.keys() & right.keys())
            if left[key] is not None and right[key] is not None}


def describe_available(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: describe([{key: row[key]} for row in rows if key in row])[key] for key in keys}


def read_profile(stream, reply):
    line = stream.readline()
    require(line.endswith(b"\n"), "missing_native_profile")
    profile = json.loads(line)
    require(profile.get("scope") == PROFILE_SCOPE and profile.get("success") is True
            and all(profile.get(key) == reply.get(key) for key in ("input_tokens", "sampled_tokens", "output_tokens")),
            "unpaired_native_profile")
    phases = profile.get("native_phase_counters") or {}
    require(phases.get("boundary") == PHASE_BOUNDARY, "unexpected_native_phase_boundary")
    clean = {}
    for phase in ("prefill", "decode"):
        counters = phases.get(phase)
        require(isinstance(counters, dict), "missing_phase_counters")
        clean[phase] = {}
        for name, counter in counters.items():
            require(isinstance(name, str) and all(part.isascii() and part.replace("_", "").isalnum()
                    for part in name.split(".")) and isinstance(counter, dict), "invalid_native_profile_category")
            calls, seconds, size = (counter.get(key) for key in ("calls", "seconds", "bytes"))
            require(type(calls) is int and calls >= 0 and type(size) is int and size >= 0 and
                    type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "invalid_native_profile_value")
            clean[phase][name] = {"calls": calls, "seconds": seconds, "bytes": size}
    return clean


def metrics(row):
    values = {key: row[key] for key in ("prefill_seconds", "decode_seconds", "wall_seconds")}
    for phase, counters in row["phase_counters"].items():
        for name, measure in counters.items():
            for unit in ("seconds", "calls", "bytes"):
                values[f"{phase}/{name}/{unit}"] = measure[unit]
    return values


def contrast(left, right):
    return {key: left.get(key, 0) - right.get(key, 0) for key in sorted(left.keys() | right.keys())}


def describe(rows):
    keys = sorted({key for row in rows for key in row})
    results = {}
    for key in keys:
        values = [row.get(key, 0) for row in rows]
        if any(values) or "/" not in key:
            results[key] = {
                "pairs": len(values), "median": median(values), "range": [min(values), max(values)],
                "positive": sum(value > 0 for value in values), "negative": sum(value < 0 for value in values),
            }
    return results


def summary(rows):
    joined = defaultdict(dict)
    for row in rows:
        key = row["pair"], row["prefix_chunk_tokens"], row["fixture"], row["repetition"]
        require(row["scoped"] not in joined[key], "duplicate_scope_arm")
        joined[key][row["scoped"]] = row
    within = defaultdict(dict)
    focused = defaultdict(dict)
    for (pair, grid, fixture, rep), observed in joined.items():
        require(set(observed) == {True, False}, "missing_scope_arm")
        scoped, unscoped = observed[True], observed[False]
        require(work_signature(scoped) == work_signature(unscoped), "scope_work_mismatch")
        within[grid][pair, fixture, rep] = contrast(metrics(scoped), metrics(unscoped))
        focused[grid][pair, fixture, rep] = available_contrast(focused_metrics(scoped), focused_metrics(unscoped))
    require(within[2048].keys() == within[4096].keys(), "missing_abba_pair")
    for (pair, fixture, rep) in within[4096]:
        baseline = joined[pair, 4096, fixture, rep][False]
        candidate = joined[pair, 2048, fixture, rep][False]
        require(work_signature(baseline) == work_signature(candidate), "cross_arm_work_mismatch")
    difference = {key: contrast(within[2048][key], baseline) for key, baseline in within[4096].items()}
    focused_difference = {key: available_contrast(focused[2048][key], baseline) for key, baseline in focused[4096].items()}
    return {
        "exact_scope_pairs": {str(grid): len(contrasts) for grid, contrasts in within.items()},
        "exact_cross_arm_pairs": len(difference),
        "scoped_minus_unscoped_by_prefix_grid": {str(grid): describe(contrasts.values()) for grid, contrasts in within.items()},
        "midpoint_minus_aligned_capture_delta": describe(difference.values()),
        "focused_seconds": {
            "scoped_minus_unscoped_by_prefix_grid": {str(grid): describe_available(list(groups.values()))
                                                     for grid, groups in focused.items()},
            "midpoint_minus_aligned_capture_delta": describe_available(list(focused_difference.values())),
        },
        "by_launch_pair": {pair: {
            "first_prefix_grid": SCHEDULE[(pair - 1) * 2],
            "scoped_minus_unscoped_by_prefix_grid": {
                str(grid): describe([value for key, value in contrasts.items() if key[0] == pair])
                for grid, contrasts in within.items()
            },
            "midpoint_minus_aligned_capture_delta": describe([value for key, value in difference.items() if key[0] == pair]),
        } for pair in (1, 2)},
        "host_paging": {str(grid): {
            arm: host_stats([row for row in rows if row["prefix_chunk_tokens"] == grid and row["scoped"] is scoped])
            for arm, scoped in (("scoped", True), ("unscoped", False))
        } for grid in (4096, 2048)},
        "limits": [
            "Differences are signed seconds, calls, or bytes as indicated by the key. Positive means the first named arm is greater.",
            "Native profile categories are inclusive. Report CPU, GPU and blit categories separately and do not sum overlapping time.",
            "Raw category contrasts treat an omitted category as zero. Focused GPU phase comparisons omit pairs without the required timestamps. Host paging is system-wide.",
            "Repeated fixtures share launch conditions; ABBA and alternating scope order are controls, not independent hardware samples.",
        ],
    }


def run(args, output, position):
    for key in tuple(os.environ):
        if key.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(key)
    os.environ["STARK_LILY_PREFILL_CHUNK"] = str(PHYSICAL)
    artifact = snapshot_artifact(args.model)
    matrix, fixtures = make_prompts(ROOT, args.model)
    names = sorted(name for name, length in matrix if length == LENGTH)
    require(len(names) == 3 and all(len(matrix[name, LENGTH]) == LENGTH for name in names), "fixture_lengths")
    record(output, {"type": "manifest", "schema": 1, "physical_chunk_tokens": PHYSICAL,
                    "prefix_grid_schedule": SCHEDULE, "input_tokens": LENGTH, "output_cap": BUDGET, "drafts": DRAFTS,
                    "sampling": SAMPLING, "seed": SEED, "context_tokens": CONTEXT, "repetitions": args.reps,
                    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "worker_sha256": hash_file(args.worker), "runner_sha256": hash_file(Path(__file__)),
                    "fixtures_sha256": hash_file(Path(__file__).with_name("cold_fixtures.py")),
                    "image_manifest_sha256": hash_file(args.model / "stark-image-artifact.json"),
                    "artifact_sha256": ARTIFACT, "revision": REVISION, "fixtures": fixtures,
                    "profile_mode": PROFILE_SCOPE, "host": host_snapshot()})
    measured = []
    used_scope_digests = set()
    for block, grid in enumerate(SCHEDULE, 1):
        pair = (block + 1) // 2
        arm = {"block": block, "pair": pair, "prefix_chunk_tokens": grid, "physical_chunk_tokens": PHYSICAL,
               "midpoint_enabled": grid == 2048}
        position.clear()
        position.update(arm)
        if grid == 2048:
            os.environ["STARK_LILY_PREFIX_MIDPOINT"] = "1"
        else:
            os.environ.pop("STARK_LILY_PREFIX_MIDPOINT", None)
        profile_path = args.output.with_name(f"{args.output.name}.block-{block}.profile.jsonl")
        os.environ["STARK_NATIVE_PROFILE_OUTPUT"] = str(profile_path)
        started = time.monotonic()
        with Worker(str(args.worker), str(args.model), prefix_cache=True, vision=True, timeout=args.timeout) as worker:
            image_profile = validate_ready(worker, grid)
            record(output, {"type": "ready", **arm, "image_profile": image_profile,
                            "startup_seconds": time.monotonic() - started, "host": host_snapshot(worker.process.pid)})
            with profile_path.open("rb") as profile_stream:
                def observe(fixture, repetition, scoped, *, warmup=False):
                    position.update(fixture=fixture, repetition=repetition, scoped=scoped, warmup=warmup)
                    scope = secrets.token_hex(16) if scoped else None
                    scope_hash = sha256(scope.encode()).hexdigest() if scoped else None
                    if scoped:
                        require(scope_hash not in used_scope_digests, "duplicate_cache_scope")
                        used_scope_digests.add(scope_hash)
                    ids = matrix[fixture, LENGTH]
                    before = host_snapshot(worker.process.pid)
                    launch = time.monotonic()
                    reply = generate(worker, ids, context=CONTEXT, max_new=BUDGET, sampling=SAMPLING,
                                     seed=SEED, mtp_drafts=DRAFTS, scope=scope)
                    wall = time.monotonic() - launch
                    after = host_snapshot(worker.process.pid)
                    details = validate(reply, LENGTH, BUDGET, DRAFTS, 0, VERIFIER)
                    counts = details["mtp_counts"]
                    require(0 < counts["proposed"] <= DRAFTS * counts["rounds"]
                            and counts["verified_rows"] == counts["rounds"] + counts["proposed"], "unexpected_mtp_work")
                    phases = read_profile(profile_stream, reply)
                    prefill = phases["prefill"]
                    require(prefill.get(ATTRIBUTION, {}).get("calls", 0) == 1, "missing_prefill_attribution_marker")
                    for name, expected_grid in (("aligned", 4096), ("midpoint", 2048)):
                        capture = prefill.get(f"{SNAPSHOT}{name}.capture", {})
                        expected = int(scoped and grid == expected_grid)
                        require(capture.get("calls", 0) == expected and (not expected or capture.get("bytes", 0) > 0),
                                "unexpected_capture_count")
                    row = {"type": "request", **arm, "fixture": fixture, "repetition": repetition,
                           "scoped": scoped, "scope_sha256": scope_hash, "warmup": warmup, "input_tokens": LENGTH,
                           "input_sha256": digest_ids(ids), "cached_tokens": reply["reused_input_tokens"],
                           "wall_seconds": wall, "host_before": before, "host_after": after, **details,
                           "phase_counters": phases}
                    record(output, row)
                    if not warmup:
                        measured.append(row)
                    print(f"Block {block}, {'scoped' if scoped else 'unscoped'}: {row['prefill_seconds']:.3f} s"
                          + (" (warmup)" if warmup else ""), flush=True)

                observe(names[0], 0, False, warmup=True)
                jobs = [(name, rep) for rep in range(1, args.reps + 1) for name in names]
                for fixture, rep in jobs if pair == 1 else reversed(jobs):
                    for scoped in (True, False) if pair == 1 else (False, True):
                        observe(fixture, rep, scoped)
                require(not profile_stream.read(1), "unexpected_extra_profile_record")
    require(snapshot_artifact(args.model) == artifact, "model_files_changed")
    require(len(measured) == len(SCHEDULE) * len(names) * args.reps * 2
            and len(used_scope_digests) * 2 == len(measured), "unexpected_request_count")
    record(output, {"type": "complete", "artifact_identity_unchanged": True,
                    "summary": summary(measured), "host": host_snapshot()})


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=2, help="Repetitions per three fixtures in each launch")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.reps < 1 or args.timeout < 1:
        parser.error("--reps and --timeout must be positive")
    os.umask(0o077)
    args.worker, args.model = args.worker.resolve(strict=True), args.model.resolve(strict=True)
    args.output = args.output.absolute()
    position = {}
    with args.output.open("x", encoding="utf-8") as output:
        try:
            run(args, output, position)
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            location = traceback.extract_tb(error.__traceback__)[-1]
            record(output, {"type": "failure", **position, "exception_type": type(error).__name__,
                            "source": Path(location.filename).name, "line": location.lineno})
            print("Midpoint profile stopped. See the private aggregate receipt.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
