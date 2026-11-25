"""Adjacent A/B timing, using the study-only greedy seed switch in one loaded process.

Production snapshot 6af5401b plus one added condition on use_self_correction:
`&& options.seed == 1`. Width three selects the arm; width two ignores both labels.
"""

import argparse
import fcntl
import json
import os
import random
import subprocess
from pathlib import Path

from live_runner_up import Worker, measure, read_tokens


def summarize(pairs):
    result = {}
    for domain in ("prose", "coding"):
        result[domain] = {}
        for width in (3, 2):
            group = [p for p in pairs if p["case"].startswith(domain) and p["width"] == width]
            by_case = {}
            for p in group:
                v = by_case.setdefault(p["case"], [0., 0.])
                for seed in (0, 1):
                    v[seed] += p[str(seed)]["decode_seconds"]
            if len(by_case) != (12 if width == 3 else 4) or len(group) != 2 * len(by_case):
                raise ValueError("incomplete adjacent-pair report")
            def change(sample, key):
                return 100 * (sum(p["0"][key] for p in sample) / sum(p["1"][key] for p in sample) - 1)
            # Keep both repeats and both labels together when resampling prompts.
            rng = random.Random(20261125)
            times = list(by_case.values())
            bootstrap = []
            for _ in range(20000):
                sample = rng.choices(times, k=len(times))
                bootstrap.append(100 * (sum(t[0] for t in sample) / sum(t[1] for t in sample) - 1))
            bootstrap.sort()
            policies = {}
            for seed in (0, 1):
                rows = [p[str(seed)] for p in group]
                nrounds = sum(sum(r["speculative"]["rounds"]) for r in rows)
                active = sum(sum(r["speculative"]["rounds"][1:]) for r in rows)
                decode = sum(r["decode_seconds"] for r in rows)
                policies[str(seed)] = {
                    "requests": len(rows), "decode_tok_s": sum(r["tokens"] - 1 for r in rows) / decode,
                    "wall_tok_s": sum(r["tokens"] for r in rows) / sum(r["wall_seconds"] for r in rows),
                    "rounds": nrounds, "decode_ms_per_round": 1000 * decode / nrounds,
                    **{k + "_ms_per_active_round": 1000 * sum(r["speculative"][k + "_seconds"] for r in rows) / active
                       for k in ("draft", "verify", "finish")},
                }
            ordinary, candidate = policies["0"], policies["1"]
            result[domain][str(width)] = {"policies": policies, "prompts": len(by_case),
                "decode_change_pct": change(group, "decode_seconds"), "wall_change_pct": change(group, "wall_seconds"),
                "pass_decode_changes_pct": [change([p for p in group if p["repeat"] == i], "decode_seconds") for i in (1, 2)],
                "faster_prompts": sum(left > right for left, right in times),
                "paired_prompt_bootstrap_95_pct": [bootstrap[500], bootstrap[19499]],
                "headroom_ms_per_candidate_round": ordinary["decode_ms_per_round"] * (ordinary["rounds"] / candidate["rounds"] - 1),
                "added_ms_per_candidate_round": candidate["decode_ms_per_round"] - ordinary["decode_ms_per_round"]}
    return result


def main(args):
    if args.output.exists():
        raise ValueError("a fresh private output path is required")
    cases = {}
    for i in range(1, 13):
        for domain in ("prose", "coding"):
            case = f"{domain}{i:02}"
            prompt = read_tokens(args.cohort / "inputs" / case / "tokens.safetensors")
            full = read_tokens(args.cohort / "cases" / case / "teacher.safetensors")
            if full[:len(prompt)] != prompt or len(full) != len(prompt) + 192:
                raise ValueError("invalid private teacher binding")
            cases[case] = (prompt, full[len(prompt):])
    report = []
    worker = Worker(args.worker, args.model, timeout=240)
    try:
        ready = worker.read()
        if not (ready.get("ready") and worker.images and "sibling" in ready["mtp"]["verification"]):
            raise ValueError("expected image-qualified study worker")
        for width, seed in ((3, 0), (3, 1), (2, 0)):
            measure(worker, *cases["coding12"], seed=seed, width=width)
        for repeat in range(2):
            if "AC Power" not in subprocess.check_output(["pmset", "-g", "batt"], text=True).splitlines()[0]:
                raise ValueError("timing requires AC power")
            sequence = list(cases)
            if repeat:
                sequence.reverse()
            for case in sequence:
                prompt_number = int(case[-2:])
                arm_first = (prompt_number + int(case.startswith("coding")) + repeat) % 2
                # Interleave four paired width-two controls per domain per repeat.
                widths = [3, 2] if prompt_number in (1, 4, 7, 10) else [3]
                if repeat:
                    widths.reverse()
                for width in widths:
                    first = arm_first if width == 3 else 1 - arm_first
                    pair = {"case": case, "width": width, "repeat": repeat + 1, "seed_order": [first, 1 - first]}
                    for seed in pair["seed_order"]:
                        row = measure(worker, *cases[case], width=width, seed=seed)
                        expected = width == 3 and seed == 1
                        counters = row["self_correction"]
                        if expected != (counters is not None) or expected and (
                            not counters["packed_rounds"] or counters["unsupported_rounds"]
                        ):
                            raise ValueError("unexpected active arm")
                        pair[str(seed)] = row
                    report.append(pair)
                tmp = args.output.with_suffix(".partial")
                tmp.write_text(json.dumps(report) + "\n")
                tmp.replace(args.output)
            print(f"Verified adjacent A/B pass {repeat + 1}", flush=True)
    finally:
        worker.close()
    print(json.dumps(summarize(report), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "model", "worker", "output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main(args)
