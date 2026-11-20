"""Compare matched, full-length serial requests from the two Inkling reference servers."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_samples(paths, normalize):
    samples = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            sample = normalize(row)
            if sample is None:
                continue
            key = sample["id"]
            if key in samples:
                raise ValueError(f"Duplicate case {key}. Supply only the selected measurement sessions.")
            samples[key] = sample
    return samples


def ollama_sample(row):
    if "eval_count" not in row:
        return None
    return {
        "id": row["id"],
        "input": row["requested_input_tokens"],
        "mode": row["mode"],
        "window": row["window"],
        "prompt_sha256": row["prompt_sha256"],
        "seed": row["seed"],
        "start_ns": row["start_unix_ns"],
        "end_ns": row["end_unix_ns"],
        "valid": (
            row["requested_outputs"] == row["eval_count"] == 256
            and row["prompt_eval_count"] == row["requested_input_tokens"]
            and row["prompt_eval_cached_count"] == 0
        ),
        "prefill_tokens_s": row["prefill_tps"],
        "decode_tokens_s": row["useful_decode_tps"],
        "first_visible_s": row["first_visible_seconds"],
        "request_s": row["wall_seconds"],
    }


def gguf_sample(row):
    if row.get("kind") != "sample":
        return None
    row = row["result"]
    t = row["timings"]
    return {
        "id": row["id"],
        "input": row["input_tokens"],
        "mode": row["mode"],
        "window": row["window"],
        "prompt_sha256": row["prompt_sha256"],
        "seed": row["seed"],
        "start_ns": row["epoch_start_ns"],
        "end_ns": row["epoch_end_ns"],
        "valid": row["eligible"] and row["wanted_output"] == t["predicted_n"] == 256,
        "prefill_tokens_s": t["prompt_n"] / (t["prompt_ms"] / 1000),
        "decode_tokens_s": (t["predicted_n"] - 1) / (t["predicted_ms"] / 1000),
        "first_visible_s": row["ttft_seconds"],
        "request_s": row["elapsed_seconds"],
    }


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def apply_host_gate(samples, readings):
    thermal, pressure, observed, gpu = [], [], [], []
    counters = Counter()
    excluded = []
    for key, sample in samples.items():
        window = [r for r in readings if sample["start_ns"] <= r["unix_ns"] <= sample["end_ns"]]
        temperatures = [r["thermal"] for r in window if r["thermal"] is not None]
        levels = [r["pressure"] for r in window]
        if not levels or not temperatures or max(temperatures) >= 2 or max(levels) >= 2:
            sample["valid"] = False
            excluded.append(key)
        thermal.extend(temperatures)
        pressure.extend(levels)
        observed.extend(window)
        gpu.extend(r["gpu_percent"] for r in window if r["gpu_percent"] is not None)
        for row in window:
            counters.update(row["metal_counter_delta"])
    return {
        "excluded_cases": excluded,
        "sample_count": len(observed),
        "thermal_counts": dict(Counter(thermal)),
        "memory_pressure_counts": dict(Counter(pressure)),
        "gpu_utilization_percent": distribution(gpu) if gpu else None,
        "metal_counter_percent": {name: 100 * value / counters.total() for name, value in sorted(counters.items())}
        if counters.total()
        else {},
        "pageout_counter_span_between_first_and_last_samples": (
            max(r["pageouts"] for r in observed) - min(r["pageouts"] for r in observed) if observed else None
        ),
        "swapout_counter_span_between_first_and_last_samples": (
            max(r["swapouts"] for r in observed) - min(r["swapouts"] for r in observed) if observed else None
        ),
    }


def compare(original, gguf):
    groups = {}
    rejected = []
    for key in sorted(original.keys() | gguf.keys()):
        left, right = original.get(key), gguf.get(key)
        if not (left and right and left["valid"] and right["valid"]):
            rejected.append(key)
            continue
        if any(left[k] != right[k] for k in ("input", "mode", "window", "prompt_sha256", "seed")):
            raise ValueError(f"Unmatched request {key}")
        groups.setdefault(f"{left['input']}_{left['mode']}", []).append((left, right))
    cells = {}
    for label, pairs in sorted(groups.items()):
        pairs.sort(key=lambda p: p[0]["window"])
        pairs = pairs[:8]
        entry = {"paired_cases": [a["id"] for a, _ in pairs]}
        for metric in (
            "prefill_tokens_s",
            "decode_tokens_s",
            "first_visible_s",
            "request_s",
        ):
            a = [x[metric] for x, _ in pairs]
            b = [x[metric] for _, x in pairs]
            entry[metric] = {
                "original": distribution(a),
                "unsloth": distribution(b),
                "unsloth_over_original": distribution([y / x for x, y in zip(a, b, strict=True)]),
            }
        cells[label] = entry
    return {"cells": cells, "unmatched_or_ineligible": rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", type=Path, nargs="+", required=True)
    parser.add_argument("--gguf", type=Path, nargs="+", required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = load_samples(args.ollama, ollama_sample)
    gguf = load_samples(args.gguf, gguf_sample)
    readings = [json.loads(line) for line in args.host.read_text().splitlines()]
    host = {
        "original": apply_host_gate(original, readings),
        "unsloth": apply_host_gate(gguf, readings),
    }
    report = compare(original, gguf) | {"host_on_all_attempts": host}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
