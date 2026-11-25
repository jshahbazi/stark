"""Replay one alternate sibling from matched, private Inkling draft fixtures.

Run from any directory with uv run --no-project python first_miss.py TRACE_DIR.
The JSON output contains counters, hashes and modeled aggregate costs only.
"""

import argparse
import hashlib
import json
import statistics
import struct
from dataclasses import dataclass, field
from pathlib import Path


DATA = Path(__file__).resolve().parents[3] / "benchmarks" / "data"
MATCHED = DATA / "inkling-mini-dflash2-traces-2026-11-24.json"
LIVE = DATA / "2026-11-24-inkling-mini-live-dflash2.json"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_u32(path, expected_hash, shapes):
    if digest(path) != expected_hash:
        raise ValueError(f"Fixture hash mismatch: {path.name}")
    result = {}
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        for name, shape in shapes.items():
            info = header[name]
            if info["dtype"] != "U32" or info["shape"] != shape:
                raise ValueError(f"Unexpected {name} layout")
            start, end = info["data_offsets"]
            source.seek(8 + header_size + start)
            result[name] = [v for v, in struct.iter_unpack("<I", source.read(end - start))]
    return result


def load_trace(directory, case):
    n, p = case["output_tokens"], case["prompt_tokens"]
    shapes = {"proposals": [n, 7]}
    native = read_u32(directory / "teacher.safetensors", case["fixture_sha256"],
                      {**shapes, "tokens": [n + p]})
    alternate = read_u32(directory / "dflash-f32.safetensors",
                         case["dflash_candidate_fixture_sha256"], shapes)
    teacher = native["tokens"][p:]
    if hashlib.sha256(struct.pack(f"<{n}I", *teacher)).hexdigest() != case["teacher_output_sha256"]:
        raise ValueError("Teacher hash mismatch")
    rows = {name: [data["proposals"][i:i + 7] for i in range(0, n * 7, 7)]
            for name, data in (("dspark", native), ("dflash", alternate))}
    return teacher, rows


def prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


@dataclass
class Counts:
    rounds: int = 0
    draft_rounds: int = 0
    first_misses: int = 0
    other_catches: int = 0
    forks: int = 0
    sibling_taken: int = 0
    calls_by_rows_1_to_4: list = field(default_factory=lambda: [0] * 4)


def walk(teacher, primary, other, width, branch):
    counts = Counts()
    cursor = 0
    while cursor < len(teacher):
        truth = teacher[cursor + 1:cursor + width + 1]
        a, b = primary[cursor][:len(truth)], other[cursor][:len(truth)]
        hit = prefix(a, truth)
        counts.rounds += 1
        counts.draft_rounds += bool(truth)
        if hit < len(truth):
            counts.first_misses += 1
            counts.other_catches += prefix(b, truth) > hit

        # Decide from the proposals and remaining output budget alone. Pack the
        # sibling immediately after its primary counterpart in parent-first order.
        split = prefix(a, b)
        fork = branch and split < len(a) and split < width - 1
        chain = min(len(a), width - int(fork))
        nodes = 1 + chain + int(fork)  # includes the anchor at packed index zero
        kept = min(hit, chain)
        last_row = kept
        if fork:
            counts.forks += 1
            if hit == split and b[split] == truth[split]:
                counts.sibling_taken += 1
                kept = split + 1
                last_row = split + 2
            elif kept > split + 1:
                last_row += 1

        # Like the native chain walk, check the crossing edge using logits from
        # its parent in packet one. Send packet two only to consume a matched node.
        first_rows = min(4, nodes)
        counts.calls_by_rows_1_to_4[first_rows - 1] += 1
        if last_row >= first_rows:
            counts.calls_by_rows_1_to_4[nodes - first_rows - 1] += 1
        cursor += 1 + kept
    return counts


def priced_walk(base, fork, timing, width_prices, worker_seconds):
    """Price matched physical packet counts, retaining the six timing blocks."""
    def cost(counts):
        return sum(n * ms for n, ms in zip(counts.calls_by_rows_1_to_4, width_prices, strict=True))

    verify_ratio = cost(fork) / cost(base)
    # The native singleton updates state inside verify. Only transactions have
    # a separately timed commit. Per-transaction commit uses the baseline mean.
    commit_ratio = sum(fork.calls_by_rows_1_to_4[1:]) / sum(base.calls_by_rows_1_to_4[1:])
    totals, worker_totals = [], []
    saved_draft_rounds = base.draft_rounds - fork.draft_rounds
    for verify, commit in zip(timing["verify_ms_by_block"], timing["commit_ms_by_block"], strict=True):
        savings = verify * (1 - verify_ratio) + commit * (1 - commit_ratio)
        totals.append(savings)
        if worker_seconds:
            # Diagnostic and live clocks are separate runs. Attribute their
            # difference to ordinary non-verifier work per nonterminal draft round.
            ordinary = (1000 * statistics.mean(worker_seconds) - verify - commit) / base.draft_rounds
            worker_totals.append(savings + saved_draft_rounds * ordinary)
    return totals, worker_totals


def summarize(records):
    counts = {key: sum(r[key] for r in records) for key in
              ("baseline_rounds", "first_misses", "other_catches", "fork_rounds",
               "fork_draft_rounds", "forks", "sibling_taken", "saved_draft_rounds")}
    for kind in ("verify_commit", "worker"):
        blocks = [r[kind + "_savings_ms_by_block"] for r in records]
        if not all(blocks):
            continue
        combined = [sum(b) for b in zip(*blocks, strict=True)]
        median = statistics.median(combined)
        counts[kind + "_savings_ms_by_block"] = [round(t, 4) for t in combined]
        for label, n in (("attempt", counts["forks"]), ("taken", counts["sibling_taken"]),
                         ("draft_round", counts["fork_draft_rounds"])):
            counts[f"{kind}_ms_per_{label}"] = round(median / n, 4) if n else None
    return counts


def analyze(root):
    matched = json.loads(MATCHED.read_text())["cases"]
    live = json.loads(LIVE.read_text())["width_screen"]["cases"]
    cases = {}
    for name, case in matched.items():
        teacher, rows = load_trace(root / name, case)
        cases[name] = {}
        for arm, backup in (("dspark", "dflash"), ("dflash", "dspark")):
            receipt_key = f"actual0-{arm}"
            path = root / name / f"{receipt_key}.json"
            if digest(path) != case["native_walk_receipts_sha256"][receipt_key]:
                raise ValueError(f"Walk receipt hash mismatch: {name} {arm}")
            native = json.loads(path.read_text())
            for width in (3, 7):
                key = f"{arm}_{width}"
                timing = case["timed_walks"][key]
                base, fork = (walk(teacher, rows[arm], rows[backup], width, branch)
                              for branch in (False, True))
                depth = "depth_three" if width == 3 else "depth_seven_offline_chunk_emulation"
                reference = native[depth]["walk"]
                if (base.rounds != timing["rounds"]
                        or base.first_misses != sum(reference["rejected_by_position"])
                        or base.calls_by_rows_1_to_4[0] != reference["singleton_calls"]
                        or sum(base.calls_by_rows_1_to_4[1:]) != reference["transaction_calls"]
                        or sum(base.calls_by_rows_1_to_4) != timing["physical_target_calls"]):
                    raise ValueError(f"Archived walk differs: {name} {key}")
                worker = live[name][str(width)]["decode_seconds"] if arm == "dflash" else None
                verify_ms, worker_ms = priced_walk(base, fork, timing,
                    case["full_block_verify_median_ms_by_target_rows_1_to_8"][:4], worker)
                record = {"baseline_rounds": base.rounds, "first_misses": base.first_misses,
                          "other_catches": base.other_catches, "fork_rounds": fork.rounds,
                          "fork_draft_rounds": fork.draft_rounds, "forks": fork.forks,
                          "sibling_taken": fork.sibling_taken,
                          "saved_draft_rounds": base.draft_rounds - fork.draft_rounds,
                          "verify_commit_savings_ms_by_block": verify_ms,
                          "worker_savings_ms_by_block": worker_ms}
                cases[name][key] = {**record, "baseline_packets": base.calls_by_rows_1_to_4,
                                    "fork_packets": fork.calls_by_rows_1_to_4}
    groups = {}
    for group, names in (("prose", ["prose"]), ("coding_pooled", ["coding", "sustained", "context4k"])):
        groups[group] = {key: summarize([cases[name][key] for name in names]) for key in cases["prose"]}
    for arms in cases.values():
        for key, record in arms.items():
            arms[key] = {**summarize([record]), "baseline_packets": record["baseline_packets"],
                         "fork_packets": record["fork_packets"]}
    return {"attempt": "20261125-INKDRAFT-01", "matched_receipt_sha256": digest(MATCHED),
            "live_receipt_sha256": digest(LIVE), "groups": groups, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    print(json.dumps(analyze(parser.parse_args().traces), indent=2))
