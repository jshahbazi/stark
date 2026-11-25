"""Paired full-worker timing on the frozen, private 24-prompt Inkling cohort."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import time
from pathlib import Path

from stark_convert.chat import Worker


def predict_live_counts(record, candidate):
    """Independent check using the worker's one-token reserve for the target draw."""
    teacher = record["teacher"]
    cursor = rounds = accepted = forks = catches = proposed = 0
    while cursor < len(teacher) - 1:
        depth = min(3, len(teacher) - cursor - 2)
        primary = record["primary"][cursor][:depth]
        slot = next((i for i in range(min(depth, 2)) if record["gap"][cursor][i] <= math.log(2)), None)
        branch = candidate and slot is not None
        if branch:
            primary = primary[:2]
            forks += 1
        proposed += len(primary) + int(branch)
        hit = 0
        for i, token in enumerate(primary):
            truth = teacher[cursor + 1 + i]
            if token != truth:
                if branch and i == slot and record["alternate"][cursor][i] == truth:
                    hit += 1
                    catches += 1
                break
            hit += 1
        rounds += 1
        accepted += hit
        cursor += 1 + hit
    return (rounds, proposed, accepted, forks, catches)


def summarize(report):
    """Reduce private per-request observations to domain totals and paired block rates."""
    output = {}
    for domain in ("prose", "coding"):
        pairs = []
        for block in report["blocks"]:
            pair = {}
            for arm, rows in block.items():
                group = [row for case, row in rows.items() if case.startswith(domain)]
                if len(group) != 12:
                    raise ValueError("incomplete timing block")
                counts = [row["self_correction"] for row in group if row["self_correction"] is not None]
                pair[arm] = {
                    "requests": len(group), "emitted_tokens": sum(row["tokens"] for row in group),
                    "matching_policy_replays": sum(row["matches_worker_policy_replay"] for row in group),
                    # The worker decode clock starts after the first emitted token.
                    "decode_tokens": sum(row["tokens"] - 1 for row in group),
                    "decode_s": sum(row["decode_seconds"] for row in group),
                    "wall_s": sum(row["wall_seconds"] for row in group),
                    "prefill_s": sum(row["prefill_seconds"] for row in group),
                    "rounds": sum(sum(row["speculative"]["rounds"]) for row in group),
                    "accepted": sum(row["accepted"] for row in group),
                    "proposed": sum(row["proposed"] for row in group),
                    "draft_s": sum(row["speculative"]["draft_seconds"] for row in group),
                    "verify_s": sum(row["speculative"]["verify_seconds"] for row in group),
                    "finish_s": sum(row["speculative"]["finish_seconds"] for row in group),
                    "forks": {key: sum(counter[key] for counter in counts) for key in counts[0]} if counts else {},
                }
            pairs.append(pair)
        totals = {}
        for arm in ("ordinary", "candidate"):
            sums = {key: sum(pair[arm][key] for pair in pairs) for key in (
                "requests", "decode_tokens", "emitted_tokens", "wall_s", "decode_s")}
            totals[arm] = {**sums, "decode_tok_s": sums["decode_tokens"] / sums["decode_s"],
                           "wall_tok_s": sums["emitted_tokens"] / sums["wall_s"]}
        block_gains = [100 * (pair["ordinary"]["decode_s"] / pair["candidate"]["decode_s"] - 1) for pair in pairs]
        case_ids = [case for case in report["blocks"][0]["ordinary"] if case.startswith(domain)]
        wins = sum(sum(b["candidate"][case]["decode_seconds"] for b in report["blocks"])
                   < sum(b["ordinary"][case]["decode_seconds"] for b in report["blocks"]) for case in case_ids)
        output[domain] = {"totals": totals, "blocks": pairs, "paired_block_speed_change_pct": block_gains,
            "median_paired_block_pct": statistics.median(block_gains), "faster_prompts": wins,
            "pooled_decode_change_pct": 100 * (totals["candidate"]["decode_tok_s"] / totals["ordinary"]["decode_tok_s"] - 1),
            "pooled_wall_change_pct": 100 * (totals["candidate"]["wall_tok_s"] / totals["ordinary"]["wall_tok_s"] - 1)}
    return output


def read_tokens(path):
    with path.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        tensor = json.loads(source.read(size))["tokens"]
        start, stop = tensor["data_offsets"]
        if tensor["dtype"] != "U32" or tensor["shape"] != [(stop - start) // 4]:
            raise ValueError("invalid private token fixture")
        source.seek(8 + size + start)
        return [token for (token,) in struct.iter_unpack("<I", source.read(stop - start))]


def measure(worker, prompt, teacher):
    started = time.perf_counter()
    result = worker.request(
        {"operation": "generate", "input_token_ids": prompt, "context_tokens": 4352,
         "max_new_tokens": len(teacher), "seed": 0, "sampling": {"mode": "greedy"},
         "stream": True, "mtp_drafts": 3},
        on_delta=lambda _: None,
    )
    wall = time.perf_counter() - started
    if result.get("generated_token_ids") != teacher:
        raise ValueError("live output differed from frozen greedy teacher or request failed")
    mtp = result.get("mtp") or {}
    if not mtp.get("enabled") or mtp.get("drafts") != 3 or result.get("reused_input_tokens", 0):
        raise ValueError("unexpected generation configuration")
    stats = mtp["stats"]
    spec = stats["speculative"]
    return {
        "tokens": len(teacher), "output_sha256": hashlib.sha256(struct.pack(f"<{len(teacher)}I", *teacher)).hexdigest(),
        "wall_seconds": wall, "prefill_seconds": result["prefill_seconds"], "decode_seconds": result["decode_seconds"],
        "accepted": stats["accepted"], "proposed": stats["proposed"],
        "speculative": spec, "self_correction": stats.get("self_correction"),
    }


def run(args):
    cohort = {}
    predicted = {}
    for i in range(1, 13):
        for domain in ("prose", "coding"):
            case = f"{domain}{i:02}"
            prompt = read_tokens(args.cohort / "inputs" / case / "tokens.safetensors")
            capture = read_tokens(args.cohort / "cases" / case / "teacher.safetensors")
            if capture[:len(prompt)] != prompt or len(capture) != len(prompt) + 192:
                raise ValueError("teacher fixture is not bound to the expected prompt and length")
            cohort[case] = (prompt, capture[len(prompt):])
            record = json.loads((args.cohort / "cases" / case / "selector.json").read_text())
            if record["teacher"] != cohort[case][1]:
                raise ValueError("selector sidecar disagrees with the frozen native teacher")
            for arm in ("ordinary", "candidate"):
                predicted[arm, case] = predict_live_counts(record, arm == "candidate")
    report = {"schema": 1, "blocks": []}
    for index in range(args.blocks):
        # Rotate the prompt order while keeping each within-block A/B pair matched.
        order = list(cohort)
        offset = (index * 7) % len(order)
        order = order[offset:] + order[:offset]
        arms = [("ordinary", args.ordinary), ("candidate", args.candidate)]
        if index % 2:
            arms.reverse()
        block = {}
        report["blocks"].append(block)
        for arm, executable in arms:
            on_ac = subprocess.check_output(["pmset", "-g", "batt"], text=True).splitlines()[0]
            if "AC Power" not in on_ac:
                raise ValueError("timing block requires AC power")
            worker = Worker(executable, args.model, timeout=240)
            try:
                ready = worker.read()
                if not (ready.get("ready") and worker.images and "dflash2" in ready["mtp"]["verification"]):
                    raise ValueError("expected image-qualified DFlash worker")
                # A full untimed request primes both Metal and the persistent MLX companion.
                measure(worker, *cohort[order[-1]])
                rows = block[arm] = {}
                for case in order:
                    rows[case] = measure(worker, *cohort[case])
                    counters = rows[case]["self_correction"]
                    if arm == "candidate" and (not counters or not counters["packed_rounds"] or counters["unsupported_rounds"]):
                        raise ValueError("candidate did not run the packed short verifier on this cohort")
                    if arm == "ordinary" and counters is not None:
                        raise ValueError("ordinary arm unexpectedly enabled self-correction")
                    row = rows[case]
                    observed = (sum(row["speculative"]["rounds"]), row["proposed"], row["accepted"],
                                (counters or {}).get("packed_rounds", 0), (counters or {}).get("sibling_hits", 0))
                    row["matches_worker_policy_replay"] = observed == predicted[arm, case]
            finally:
                worker.close()
            tmp = args.output.with_suffix(".partial")
            tmp.write_text(json.dumps(report, indent=2) + "\n")
            tmp.replace(args.output)
            print(f"Block {index + 1} {arm} verified {len(rows)} completions", flush=True)
    print(json.dumps(summarize(report), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "model", "ordinary", "candidate", "output", "lock"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=4)
    options = parser.parse_args()
    if options.blocks < 1 or options.output.exists():
        parser.error("positive block count and a new output path required")
    os.umask(0o077)
    with options.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(options)
