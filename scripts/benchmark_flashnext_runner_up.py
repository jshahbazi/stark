"""Time Flash Next's three-slot runner-up against fixed MTP on private, predeclared prompts."""

import argparse
import collections
import contextlib
import hashlib
import json
import os
import random
import runpy
import select
import struct
import subprocess
import tempfile
import time
from pathlib import Path

DOMAINS = ("prose", "coding")
CASES = [f"{domain}{i:02d}" for i in range(1, 13) for domain in DOMAINS]
IMAGE_PROFILE = "qwen38-lily-retained-bf16-vision-v1"
REVISION = "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
ARTIFACT = "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4"
FIXED = {"mtp_policy": "fixed", "mtp_lookup": False}
ARMS = {
    "serial": {"mtp_drafts": 0},
    "ordinary": {**FIXED, "mtp_drafts": 3},
    "runner_up": {**FIXED, "mtp_drafts": 3, "mtp_self_correction": True},
    "control_a": {**FIXED, "mtp_drafts": 2},
    "control_b": {**FIXED, "mtp_drafts": 2},
    "shipped": {"mtp": True},
}
REFERENCE_FIELDS = ("output_sha256", "finish_reason", "sampled_tokens", "output_tokens")
STATS = ("rounds", "proposed", "accepted", "verified_rows", "draft_seconds", "verify_seconds", "rollback_seconds")
BRANCH = (
    "eligible_rounds",
    "packed_rounds",
    "sibling_hits",
    "primary_tail_truncated",
    "primary_tail_would_hit",
    "first_misses",
    "runner_up_offered_at_miss",
    "runner_up_depth",
    "hits_by_depth",
)
LIMIT = 192


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokens_hash(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def write_report(path, report):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


class Worker:
    def __init__(self, child):
        self.child = child

    def read(self):
        if not select.select([self.child.stdout], [], [], 300)[0]:
            raise ValueError("native_worker_timeout")
        line = self.child.stdout.readline()
        if not line:
            raise ValueError("native_worker_exited")
        reply = json.loads(line)
        if "error" in reply:
            raise ValueError("native_generation_failed:" + str(reply["error"])[:80])
        return reply

    def generate(self, prompt, arm):
        payload = {
            "operation": "generate",
            "input_token_ids": prompt,
            "context_tokens": 4352,
            "max_new_tokens": LIMIT,
            "seed": 0,
            "sampling": {"mode": "greedy"},
            **ARMS[arm],
        }
        frame = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
        start = time.perf_counter()
        self.child.stdin.write(frame)
        self.child.stdin.flush()
        reply = self.read()
        wall = time.perf_counter() - start
        ids = reply["generated_token_ids"]
        sampled = reply["sampled_tokens"]
        if (
            reply["input_tokens"] != len(prompt)
            or reply.get("reused_input_tokens", 0) != 0
            or len(ids) != reply["output_tokens"]
            or not 1 <= sampled <= LIMIT
            or sampled != len(ids) + int(reply["stop_token_id"] is not None)
        ):
            raise ValueError("invalid_token_counts_or_reuse")
        decode_seconds = reply["decode_seconds"]
        if sampled == 1 and decode_seconds is None:
            decode_seconds = 0.0
        if decode_seconds is None or (sampled > 1 and decode_seconds <= 0):
            raise ValueError("invalid_worker_decode_clock")
        # Include a terminal EOS in the hash when the worker excluded it from emitted IDs.
        output_hash = tokens_hash(ids + ([reply["stop_token_id"]] if reply["stop_token_id"] is not None else []))
        mtp = reply["mtp"]
        if (
            mtp["enabled"] != (arm != "serial")
            or mtp["drafts"] != (3 if arm == "shipped" else ARMS[arm]["mtp_drafts"])
            or mtp.get("self_correction", False) != (arm == "runner_up")
            or mtp.get("lookup", False) != (arm == "shipped")
            or mtp.get("policy", "fixed") != ("adaptive" if arm == "shipped" else "fixed")
        ):
            raise ValueError("worker_selected_unexpected_arm")
        raw = mtp["stats"]
        branch = raw.get("self_correction")
        if (branch is not None) != (arm == "runner_up"):
            raise ValueError("missing_or_unexpected_runner_up_counters")
        row = {
            "output_sha256": output_hash,
            "sampled_tokens": sampled,
            "output_tokens": len(ids),
            "finish_reason": reply["finish_reason"],
            "wall_seconds": wall,
            "prefill_seconds": reply["prefill_seconds"],
            "decode_seconds": decode_seconds,
            "stats": {k: raw[k] for k in STATS},
        }
        if branch is not None:
            row["branch"] = {k: branch[k] for k in BRANCH}
        return row


@contextlib.contextmanager
def worker_session(args):
    # Launch the normal image-loaded backend. Text requests omit cache_scope.
    with tempfile.TemporaryDirectory(prefix="flashnext-vision-") as staging:
        env = {key: val for key, val in os.environ.items() if not key.startswith(("STARK_", "LILY_", "GGML_", "UZU_"))}
        proc = subprocess.Popen(
            [
                str(args.worker.resolve()),
                "--model-path",
                str(args.model.resolve()),
                "--vision-staging-root",
                staging,
                "--prefix-cache",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            worker = Worker(proc)
            ready = worker.read()
            image = ready.get("image_input", {})
            mtp = ready.get("mtp", {})
            if (
                ready.get("ready") is not True
                or image.get("artifact_sha256") != ARTIFACT
                or image.get("model_revision") != REVISION
                or image.get("family_profile") != IMAGE_PROFILE
                or mtp.get("self_correction") is not True
                or mtp.get("default_policy") != "adaptive"
                or mtp.get("default_lookup") is not True
                or mtp.get("max_drafts") != 3
            ):
                raise ValueError("unexpected_worker_readiness")
            yield (
                worker,
                {
                    "image": {k: image[k] for k in ("artifact_sha256", "model_revision", "family_profile")},
                    "mtp": mtp,
                    "prefix_cache_advertised": ready.get("prefix_cache", {}).get("enabled"),
                },
            )
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.stdout.close()


def totals(rows):
    totals = {
        key: sum(r[key] for r in rows)
        for key in ("sampled_tokens", "output_tokens", "decode_seconds", "prefill_seconds", "wall_seconds")
    }
    totals["requests"] = len(rows)
    totals["matching_references"] = sum(r["matches_reference"] for r in rows)
    totals["decode_tok_s"] = (totals["sampled_tokens"] - len(rows)) / totals["decode_seconds"]
    totals["wall_tok_s"] = totals["output_tokens"] / totals["wall_seconds"]
    totals["stats"] = {k: sum(r["stats"][k] for r in rows) for k in STATS}
    if "branch" in rows[0]:
        totals["branch"] = {}
        for k in BRANCH:
            values = [r["branch"][k] for r in rows]
            totals["branch"][k] = (
                [sum(v) for v in zip(*values, strict=True)] if isinstance(values[0], list) else sum(values)
            )
    return totals


def comparison(rows, a, b):
    cases = sorted({r["case"] for r in rows})
    group = {arm: [r for r in rows if r["arm"] == arm] for arm in (a, b)}
    summaries = {arm: totals(v) for arm, v in group.items()}
    data = collections.defaultdict(dict)
    for arm, entries in group.items():
        for case in cases:
            pack = [r for r in entries if r["case"] == case]
            if sorted(r["pass"] for r in pack) != [1, 2]:
                raise ValueError("incomplete_paired_cohort")
            data[case][arm] = (sum(r["sampled_tokens"] - 1 for r in pack), sum(r["decode_seconds"] for r in pack))

    def change(sample):
        def rate(arm):
            tokens, seconds = zip(*(p[arm] for p in sample), strict=True)
            return sum(tokens) / sum(seconds)

        return 100 * (rate(b) / rate(a) - 1)

    prompts = list(data.values())
    rng = random.Random(20261125)
    boot = sorted(change(rng.choices(prompts, k=len(prompts))) for _ in range(20000))
    return {
        "arms": summaries,
        "paired_prompts": len(prompts),
        "all_reference_matched": all(r["matches_reference"] for r in rows),
        "decode_change_pct": change(prompts),
        "paired_prompt_bootstrap_95_pct": [boot[500], boot[19499]],
        "faster_prompts": sum(p[b][0] * p[a][1] > p[a][0] * p[b][1] for p in prompts),
        "pass_decode_changes_pct": [
            100
            * (
                totals([r for r in group[b] if r["pass"] == i])["decode_tok_s"]
                / totals([r for r in group[a] if r["pass"] == i])["decode_tok_s"]
                - 1
            )
            for i in (1, 2)
        ],
        "wall_change_pct": 100 * (summaries[b]["wall_tok_s"] / summaries[a]["wall_tok_s"] - 1),
    }


def summarize(rows):
    out = {}
    for domain in DOMAINS:
        selected = [r for r in rows if r["case"].startswith(domain)]
        out[domain] = {}
        for label, a, b in (
            ("primary", "ordinary", "runner_up"),
            ("width_two_control", "control_a", "control_b"),
            ("versus_shipped", "shipped", "runner_up"),
        ):
            matched = [r for r in selected if r["arm"] in (a, b)]
            expected = 4 if label == "width_two_control" else 12
            if len(matched) != expected * 4:
                raise ValueError("incomplete_domain_matrix")
            out[domain][label] = comparison(matched, a, b)
    return out


def order(case, rep):
    i = int(case[-2:])
    flip = (i + int(case.startswith("coding")) + rep) % 2
    primary = ["ordinary", "runner_up"][:: -1 if flip else 1]
    # Keep both arms adjacent. Alternate the default schedule's position around that pair.
    shipped_first = (i // 2 + int(case.startswith("coding")) + rep) % 2
    if shipped_first:
        primary.insert(0, "shipped")
    else:
        primary.append("shipped")
    groups = [primary]
    if i in (1, 4, 7, 10):
        controls = ["control_a", "control_b"][:: 1 if flip else -1]
        groups.insert(0 if rep else 1, controls)
    return [arm for group in groups for arm in group]


def ac_power():
    return "AC Power" in subprocess.check_output(["pmset", "-g", "batt"], text=True).splitlines()[0]


def run(args):
    os.umask(0o077)
    if args.output.exists():
        raise ValueError("output_already_exists")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    prompts, preparation = runpy.run_path(str(args.prepare))["load"](args.model)
    if (
        set(prompts) != set(CASES)
        or len({tokens_hash(p) for p in prompts.values()}) != len(CASES)
        or any(not 1 <= len(p) <= 4096 for p in prompts.values())
    ):
        raise ValueError("unexpected_private_cohort")
    report = {
        "schema": "flashnext-three-slot-runner-up-v1",
        "phase": "reference_freeze",
        "prepared": preparation,
        "driver_sha256": digest(Path(__file__)),
        "worker_sha256": digest(args.worker),
        "max_new_tokens": LIMIT,
        "seed": 0,
        "repeats": 2,
        "arms": ARMS,
        "references": {},
        "rows": [],
    }
    if not ac_power():
        raise ValueError("AC_power_required_for_timing")
    with worker_session(args) as (worker, identity):
        report["readiness"] = identity
        for case in CASES:
            row = worker.generate(prompts[case], "serial")
            report["references"][case] = {
                **{k: row[k] for k in REFERENCE_FIELDS},
                "prompt_sha256": tokens_hash(prompts[case]),
                "prompt_tokens": len(prompts[case]),
            }
            write_report(args.output, report)
            print(f"Reference {len(report['references'])}/{len(CASES)}", flush=True)
        report["phase"] = "timing"
        write_report(args.output, report)

        def matched(case, arm):
            row = worker.generate(prompts[case], arm)
            ref = report["references"][case]
            row["matches_reference"] = all(row[k] == ref[k] for k in REFERENCE_FIELDS)
            return row

        for arm in ("ordinary", "runner_up", "control_a", "shipped"):
            matched("coding12", arm)
        for rep in range(2):
            for case in CASES if not rep else list(reversed(CASES)):
                if not ac_power():
                    raise ValueError("AC_power_required_for_timing")
                for arm in order(case, rep):
                    row = matched(case, arm)
                    report["rows"].append({"case": case, "pass": rep + 1, "arm": arm, **row})
                    write_report(args.output, report)
                print(f"Pass {rep + 1}/2 {case} complete ({len(report['rows'])}/176 measured)", flush=True)
    report["summary"] = summarize(report["rows"])
    report["phase"] = "complete"
    write_report(args.output, report)
    print(f"Complete: {sum(r['matches_reference'] for r in report['rows'])}/{len(report['rows'])} matched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "worker", "prepare", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    run(parser.parse_args())
