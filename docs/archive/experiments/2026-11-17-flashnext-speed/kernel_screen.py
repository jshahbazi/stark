#!/usr/bin/env python3
"""Screen one opt-in Lily kernel at a fixed 4K physical grid using exact MTP2."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from benchmark import ROOT, Worker, digest_ids, generate, hash_file, host_snapshot, record, validate
from cold_fixtures import make_prompts
from fresh_coding import ARTIFACT, CONTEXT, DRAFTS, LENGTHS, REVISION, SAMPLING, SEED, SOURCE, VERIFIER
from fresh_coding import require, snapshot_artifact, validate_ready

GRID = 4096
SCHEDULE = (False, True, True, False)
KERNELS = {
    "moe": ("STARK_FLASHNEXT_PREFILL_GROUPED_SWIGLU", 96),
    "hc": ("STARK_LILY_VERIFY_HC_DOWN_REUSE", 192),
}


def work_signature(row):
    return (row["input_sha256"], row["generated_sha256"], row["text_sha256"], row["finish"],
            row["sampled_tokens"], row["output_tokens"], row["mtp_counts"])


def run(args, output, position):
    for key in tuple(os.environ):
        if key.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(key)
    os.environ["STARK_LILY_PREFILL_CHUNK"] = str(GRID)
    selector, budget = KERNELS[args.kernel]
    artifact = snapshot_artifact(args.model)
    matrix, fixtures = make_prompts(ROOT, args.model)
    names = sorted({name for name, _length in matrix})
    require(len(names) == len(LENGTHS), "unexpected_fixture_count")
    jobs = tuple(zip(names, LENGTHS, strict=True))
    require(all(len(matrix.get((name, length), ())) == length for name, length in jobs), "prompt_lengths")
    blocks = [(block, enabled) for block, enabled in enumerate(SCHEDULE, 1)
              if args.arm is None or enabled == (args.arm == "1")]
    record(output, {"type": "manifest", "schema": 1, "kernel": args.kernel, "selector": selector, "grid": GRID,
                    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "worker_sha256": hash_file(args.worker), "runner_sha256": hash_file(Path(__file__)),
                    "fixture_source_sha256": hash_file(Path(__file__).with_name("cold_fixtures.py")),
                    "cargo_lock_sha256": hash_file(ROOT / "Cargo.lock"),
                    "image_manifest_sha256": hash_file(args.model / "stark-image-artifact.json"),
                    "model_source": SOURCE, "model_revision": REVISION, "artifact_sha256": ARTIFACT,
                    "fixtures": fixtures, "schedule": SCHEDULE, "selected_blocks": [block for block, _ in blocks],
                    "pair_one_jobs": jobs, "context_tokens": CONTEXT, "output_cap": budget,
                    "drafts": DRAFTS, "sampling": SAMPLING, "seed": SEED, "host": host_snapshot()})
    measured = {}
    comparisons = 0
    matching = 0
    for block, enabled in blocks:
        pair = (block + 1) // 2
        position.update(block=block, pair=pair, candidate_enabled=enabled)
        arm = {"block": block, "pair": pair, "grid": GRID, "kernel": args.kernel, "candidate_enabled": enabled}
        if enabled:
            os.environ[selector] = "1"
        else:
            os.environ.pop(selector, None)
        started = time.monotonic()
        with Worker(str(args.worker), str(args.model), prefix_cache=True, vision=True, timeout=args.timeout) as worker:
            profile = validate_ready(worker, GRID)
            record(output, {"type": "ready", **arm, "image_profile": profile,
                            "startup_seconds": time.monotonic() - started, "host": host_snapshot(worker.process.pid)})
            ordered = jobs if pair == 1 else tuple(reversed(jobs))
            for (name, length), warmup in [(jobs[0], True), *((job, False) for job in ordered)]:
                position.update(prompt=name, input_tokens=length, warmup=warmup)
                ids = matrix[(name, length)]
                before = host_snapshot(worker.process.pid)
                start = time.monotonic()
                reply = generate(worker, ids, context=CONTEXT, max_new=budget,
                                 sampling=SAMPLING, seed=SEED, mtp_drafts=DRAFTS)
                elapsed = time.monotonic() - start
                after = host_snapshot(worker.process.pid)
                details = validate(reply, length, budget, DRAFTS, 0, VERIFIER)
                counts = details["mtp_counts"]
                require(0 < counts["proposed"] <= DRAFTS * counts["rounds"]
                        and counts["verified_rows"] == counts["rounds"] + counts["proposed"], "unexpected_mtp_work")
                row = {"type": "request", **arm, "prompt": name, "input_tokens": length, "warmup": warmup,
                       "input_sha256": digest_ids(ids), "cached_tokens": reply["reused_input_tokens"],
                       "max_new_tokens": budget, "requested_drafts": DRAFTS, "wall_seconds": elapsed,
                       "host_before": before, "host_after": after, **details}
                record(output, row)
                print(f"Block {block}, {length} tokens: {details['prefill_seconds']:.3f} s"
                      + (" (warmup)" if warmup else ""), flush=True)
                if not warmup:
                    key = pair, name, length
                    prior = measured.get(key)
                    if prior is None:
                        measured[key] = row
                    else:
                        equal_work = work_signature(prior) == work_signature(row)
                        comparisons += 1
                        matching += equal_work
                        record(output, {"type": "comparison", "pair": pair, "prompt": name, "input_tokens": length,
                                        "same_input": prior["input_sha256"] == row["input_sha256"],
                                        "exact_work": equal_work})
            for key in ("prompt", "input_tokens", "warmup"):
                position.pop(key, None)
    require(snapshot_artifact(args.model) == artifact, "model_files_changed")
    record(output, {"type": "complete", "artifact_identity_unchanged": True, "paired_requests": comparisons,
                    "exact_work_pairs": matching, "host": host_snapshot()})


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("worker", "model", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--kernel", choices=KERNELS, required=True)
    parser.add_argument("--arm", choices=("0", "1"), help="Run only the baseline (0) or candidate (1) blocks")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    os.umask(0o077)
    args.worker, args.model = args.worker.resolve(strict=True), args.model.resolve(strict=True)
    position = {}
    with args.output.open("x", encoding="utf-8") as output:
        try:
            run(args, output, position)
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            location = traceback.extract_tb(error.__traceback__)[-1]
            record(output, {"type": "failure", **position, "exception_type": type(error).__name__,
                            "source": Path(location.filename).name, "line": location.lineno})
            print("Kernel screen stopped. See the private aggregate receipt.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
