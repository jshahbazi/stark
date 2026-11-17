#!/usr/bin/env python3
"""Compare Lily's physical prefill grids or midpoint capture on exact MTP2 coding prompts."""

import argparse
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
import traceback

from benchmark import ROOT, Worker, digest_ids, file_identity, generate, hash_file, host_snapshot, record, validate
from cold_fixtures import make_prompts


LENGTHS = (8192, 12288, 32768)
SCHEDULE = (4096, 2048, 2048, 4096)
CONTEXT = 40960
DRAFTS = 2
BUDGET = 96
SEED = 42
SAMPLING = {"mode": "stochastic", "temperature": 0.6, "top_k": 20, "top_p": 0.95}
SOURCE = "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
REVISION = "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
ARTIFACT = "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4"
VERIFIER = "batched-decode-exact-v1"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def snapshot_artifact(model):
    # The vision loader rewrites its derived digest cache at startup.
    paths = [path for path in sorted(model.iterdir()) if path.is_file()
             and path.name != ".stark-image-validation.json"
             and path.suffix in (".json", ".safetensors", ".jinja")]
    weights = [path for path in paths if path.suffix == ".safetensors"]
    require(len(weights) == 55 and sum(path.stat().st_size for path in weights) == 106905155192,
            "unexpected_weight_layout")
    return {path.name: file_identity(path) for path in paths}


def validate_ready(worker, grid):
    ready = worker.ready
    cache, mtp, image = (ready.get(key) or {} for key in ("prefix_cache", "mtp", "image_input"))
    require(ready.get("mode") == "text-generation-v1" and ready.get("max_context_tokens", 0) >= CONTEXT
            and cache.get("enabled") is True and cache.get("chunk_tokens") == grid,
            "unexpected_worker_geometry")
    require(mtp.get("verification") == VERIFIER and mtp.get("sampling") == "exact-rejection-v1"
            and mtp.get("recommended_drafts") == DRAFTS, "unexpected_worker_mtp")
    require(image.get("family_profile") == "qwen38-lily-retained-bf16-vision-v1"
            and image.get("model_source") == SOURCE and image.get("model_revision") == REVISION
            and image.get("artifact_sha256") == ARTIFACT, "unexpected_image_artifact")
    return image["family_profile"]


def work_signature(row):
    return (row["input_sha256"], row["generated_sha256"], row["text_sha256"], row["sampled_tokens"],
            row["output_tokens"], row["finish"], row["mtp_counts"])


def run(args, output):
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    # Source checkouts use the installed, pinned native chat-template helper
    # when it is placed on PATH by the caller. Build inputs are read-only here.
    artifact = snapshot_artifact(args.model)
    matrix, fixtures = make_prompts(ROOT, args.model)
    names = sorted({name for name, _length in matrix})
    require(len(names) == 3 and set(matrix) == {(name, length) for name in names for length in LENGTHS},
            "prompt_matrix")
    require(all(len(ids) == length for (_, length), ids in matrix.items()), "prompt_lengths")
    jobs = [(names[(rotation + column) % 3], length)
            for rotation in range(3) for column, length in enumerate(LENGTHS)]
    schedule = SCHEDULE if args.order == "abba" else (2048, 4096, 4096, 2048)
    if args.probe_only:
        schedule = schedule[:2]
        jobs = list(zip(names, LENGTHS, strict=True))
    record(output, {"type": "manifest", "schema": 1,
                    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "worker_sha256": hash_file(args.worker), "runner_sha256": hash_file(Path(__file__)),
                    "fixtures_sha256": hash_file(Path(__file__).with_name("cold_fixtures.py")),
                    "cargo_lock_sha256": hash_file(ROOT / "Cargo.lock"),
                    "image_manifest_sha256": hash_file(args.model / "stark-image-artifact.json"),
                    "source": SOURCE, "revision": REVISION, "artifact_sha256": ARTIFACT,
                    "fixtures": fixtures, "schedule": schedule, "pair1_jobs": jobs,
                    "mode": "midpoint" if args.midpoint else "physical_grid", "probe_only": args.probe_only,
                    "order": args.order, "scored_unique_scopes": args.midpoint and not args.probe_only,
                    "context_tokens": CONTEXT, "output_cap": BUDGET, "drafts": DRAFTS,
                    "sampling": SAMPLING, "seed": SEED, "host": host_snapshot()})

    observed = {}
    exact_work = 0
    work_comparisons = 0
    probe_reports = 0
    for block, grid in enumerate(schedule, 1):
        pair = (block + 1) // 2
        physical = 4096 if args.midpoint else grid
        midpoint_enabled = args.midpoint and grid == 2048
        os.environ["STARK_LILY_PREFILL_CHUNK"] = str(physical)
        if midpoint_enabled:
            os.environ["STARK_LILY_PREFIX_MIDPOINT"] = "1"
        else:
            os.environ.pop("STARK_LILY_PREFIX_MIDPOINT", None)
        arm = {"block": block, "pair": pair, "grid": grid, "physical_chunk_tokens": physical,
               "prefix_chunk_tokens": grid, "midpoint_enabled": midpoint_enabled}
        launched = time.monotonic()
        with Worker(str(args.worker), str(args.model), prefix_cache=True, vision=True, timeout=240) as worker:
            profile = validate_ready(worker, grid)
            record(output, {"type": "ready", **arm,
                            "image_profile": profile, "startup_seconds": time.monotonic() - launched,
                            "host": host_snapshot(worker.process.pid)})

            def observe(name, length, warmup=False, *, scope=None, reused=0, probe=False):
                ids = matrix[name, length]
                if scope is None and args.midpoint:
                    scope = secrets.token_hex(16)
                before = host_snapshot(worker.process.pid)
                started = time.monotonic()
                reply = generate(worker, ids, context=CONTEXT, max_new=BUDGET, sampling=SAMPLING,
                                 seed=SEED, mtp_drafts=DRAFTS, scope=scope)
                elapsed = time.monotonic() - started
                after = host_snapshot(worker.process.pid)
                details = validate(reply, length, BUDGET, DRAFTS, reused, VERIFIER)
                counts = details["mtp_counts"]
                require(0 < counts["proposed"] <= DRAFTS * counts["rounds"]
                        and counts["verified_rows"] == counts["rounds"] + counts["proposed"], "wrong_mtp_work")
                row = {"type": "probe_request" if probe else "request", **arm,
                                "prompt": name, "input_tokens": length, "warmup": warmup,
                                "input_sha256": digest_ids(ids), "cached_tokens": reply["reused_input_tokens"],
                                "expected_cached_tokens": reused,
                                "max_new_tokens": BUDGET, "requested_drafts": DRAFTS,
                                "wall_seconds": elapsed, "host_before": before, "host_after": after, **details}
                record(output, row)
                print(f"Block {block}, grid {grid}, {name}, {length} tokens: {details['prefill_seconds']:.3f} s"
                      + (" (warmup)" if warmup else ""), flush=True)
                return work_signature(row)

            if not args.probe_only:
                observe(names[0], max(LENGTHS), warmup=True)
            for name, length in jobs if pair == 1 else reversed(jobs):
                scope = secrets.token_hex(16) if args.probe_only else None
                work = observe(name, length, scope=scope, probe=args.probe_only)
                if args.probe_only:
                    expected = (length - 1) // grid * grid
                    resumed = observe(name, length, scope=scope, reused=expected, probe=True)
                    record(output, {"type": "capture_proof", **arm, "prompt": name, "input_tokens": length,
                                    "confirmed_cached_tokens": expected, "exact_work": work == resumed})
                    require(work == resumed, "resumed_work_mismatch")
                    probe_reports += 1
                match_key = pair, name, length
                if match_key in observed:
                    work_comparisons += 1
                    exact_work += work == observed[match_key]
                else:
                    observed[match_key] = work

    require(snapshot_artifact(args.model) == artifact, "model_files_changed")
    record(output, {"type": "complete", "artifact_identity_unchanged": True, "host": host_snapshot(),
                    "paired_requests": work_comparisons, "exact_work_pairs": exact_work,
                    "capture_proofs": probe_reports})


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("worker", "model", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--midpoint", action="store_true", help="Keep both arms at physical 4096; test interior capture")
    parser.add_argument("--order", choices=("abba", "baab"), default="abba")
    parser.add_argument("--probe-only", action="store_true", help="Prove capture and exact warm replay; skip scoring")
    args = parser.parse_args()
    if args.probe_only and not args.midpoint:
        parser.error("--probe-only requires --midpoint")
    os.umask(0o077)
    args.worker, args.model = args.worker.resolve(strict=True), args.model.resolve(strict=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            run(args, output)
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            location = traceback.extract_tb(error.__traceback__)[-1]
            record(output, {"type": "failure", "exception_type": type(error).__name__,
                            "source": Path(location.filename).name, "line": location.lineno})
            print("Fresh benchmark stopped. See the private aggregate receipt.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
