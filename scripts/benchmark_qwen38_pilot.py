"""Bounded Flash-Next experiment grid using the existing paired MTP harness.

Each cell owns one loaded worker. Reports contain aggregate measurements and
hashes only. Optional candidate workers run in ABBA session order; no setting
is automatically promoted into the engine or application.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import statistics
from pathlib import Path

import benchmark_qwen38_mtp as mtp
from benchmark_hybrid import hardware_identity
from stark_convert.chat import ChatError


def weight_guard(model, receipt_path):
    if receipt_path is None:
        return None
    receipt = json.loads(receipt_path.read_text())
    files = receipt.get("files")
    if (receipt.get("schema") != "stark-flashnext-pilot-weights-v1"
        or receipt.get("passed") is not True or not isinstance(files, dict)
        or not files or set(files) != {path.name for path in model.glob("*.safetensors")}):
        raise ValueError("weight_receipt_invalid")
    for name, expected in files.items():
        if Path(name).name != name:
            raise ValueError("weight_receipt_invalid")
        stat = (model / name).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
            expected["bytes"], expected["mtime_ns"], expected["inode"]):
            raise ValueError("weight_identity_changed")
    return mtp.sha256_file(receipt_path)


@contextlib.contextmanager
def clean_environment():
    prefixes = ("STARK_QWEN38_", "STARK_GLM53_", "STARK_HYBRID_")
    removed = {key: value for key, value in os.environ.items()
               if key.startswith(prefixes) or key == "STARK_NATIVE_PROFILE_OUTPUT"}
    for key in removed:
        del os.environ[key]
    try:
        yield
    finally:
        os.environ.update(removed)


def summarize(report):
    off, on = (report["route_summary"][route] for route in ("off", "on"))
    decode_ratios = [pair["on_tok_s"] / pair["off_tok_s"] for pair in report["paired_block_rates"]]
    return {
        "passed": report["passed"],
        "timing_eligible": report["timing_eligible"],
        "off_decode_tok_s": off["pooled_decode_tokens_per_second"],
        "on_decode_tok_s": on["pooled_decode_tokens_per_second"],
        "median_paired_decode_speedup": statistics.median(decode_ratios),
        "decode_pair_wins": sum(value > 1 for value in decode_ratios),
        "median_paired_wall_speedup": statistics.median(report["paired_wall_speedups"]),
        "off_median_wall_seconds": off["median_wall_seconds"],
        "on_median_wall_seconds": on["median_wall_seconds"],
        "mtp_totals": on["mtp_totals"],
    }


def compare_workers(cells):
    """Compare identical seeded routes, never require on/off RNG equality."""
    groups = {}
    for cell in cells:
        if "report" not in cell:
            continue
        key = (cell["input_tokens"], cell["drafts"])
        group = groups.setdefault(key, {"workers": set(), "off": set(), "on": set()})
        group["workers"].add(cell["worker_label"])
        for item in cell["report"]["requests"]:
            group[item["route"]].add(item["output_sha256"])
    return [
        {"input_tokens": key[0], "drafts": key[1],
         "workers": sorted(group["workers"]),
         "same_route_hashes_equal": all(len(group[route]) == 1 for route in ("off", "on"))}
        for key, group in sorted(groups.items())
    ]


def run(args, *, cell_runner=mtp.run):
    if args.output.exists() or args.cells.exists():
        raise ValueError("output_exists")
    if (not args.input_lengths or len(args.input_lengths) > 4
        or len(set(args.input_lengths)) != len(args.input_lengths)
        or not args.draft_depths or len(set(args.draft_depths)) != len(args.draft_depths)
        or any(depth not in (1, 2, 3) for depth in args.draft_depths)):
        raise ValueError("invalid_pilot_grid")
    # Validate the entire grid before loading any model.
    for count in args.input_lengths:
        mtp.validate_args(argparse.Namespace(**{**vars(args), "input_tokens": count, "drafts": args.draft_depths[0]}))
    receipt_path = getattr(args, "weights_receipt", None)
    weights_before = weight_guard(args.model, receipt_path)
    args.cells.mkdir(parents=True, mode=0o700)
    plan = [(count, depth) for count in args.input_lengths for depth in args.draft_depths]
    random.Random(args.seed).shuffle(plan)
    sessions = [("baseline", args.worker)]
    if args.candidate_worker is not None:
        sessions = [("baseline", args.worker), ("candidate", args.candidate_worker),
                    ("candidate", args.candidate_worker), ("baseline", args.worker)]
    report = {
        "schema": "stark-flashnext-optimization-pilot-v1",
        "harness_sha256": mtp.sha256_file(Path(__file__)),
        "paired_harness_sha256": mtp.sha256_file(Path(mtp.__file__)),
        "hardware": hardware_identity(),
        "worker_hashes": {label: mtp.sha256_file(worker) for label, worker in sessions},
        "screened": args.screened,
        "observe_host": getattr(args, "observe_host", False),
        "weights_receipt_sha256": weights_before,
        "weight_guard_enabled": weights_before is not None,
        "weight_guard_policy": ("Previously hashed payloads; size, mtime and inode checked before and after the grid."
                                if weights_before is not None else "No payload receipt supplied; artifact integrity is unqualified."),
        "session_order": [label for label, _ in sessions],
        "selection_policy": "Screening evidence only; independent workload confirmation required before promotion.",
        "cells": [],
    }
    with clean_environment():
        for session, (label, worker) in enumerate(sessions):
            # Reverse the grid on alternate sessions to reduce load-order bias.
            for count, depth in (plan if session % 2 == 0 else list(reversed(plan))):
                number = len(report["cells"])
                output = args.cells / f"cell-{number:02d}.json"
                options = argparse.Namespace(**{**vars(args), "worker": worker, "input_tokens": count,
                                                "drafts": depth, "output": output})
                cell = {"session": session, "worker_label": label, "input_tokens": count, "drafts": depth}
                try:
                    result = cell_runner(options)
                    cell.update(summary=summarize(result), report=result)
                except (OSError, ValueError, KeyError, TypeError, TimeoutError, ChatError) as error:
                    # Exception text can contain model data; retain a static stage only.
                    cell["failure"] = "paired_cell_failed"
                    cell["failure_code"] = str(error) if str(error) in mtp.SAFE_FAILURE_CODES else "paired_cell_failed"
                    if output.exists():
                        cell["partial_receipt_sha256"] = mtp.sha256_file(output)
                report["cells"].append(cell)
                print(json.dumps({"cell": number, **{key: cell[key] for key in (
                    "session", "worker_label", "input_tokens", "drafts")},
                    "summary": cell.get("summary"), "failure": cell.get("failure"),
                    "failure_code": cell.get("failure_code")}), flush=True)
                if "failure" in cell:
                    break
            if report["cells"] and "failure" in report["cells"][-1]:
                break
    report["worker_comparisons"] = compare_workers(report["cells"])
    report["artifact_unchanged"] = weight_guard(args.model, receipt_path) == weights_before
    report["workers_unchanged"] = all(
        mtp.sha256_file(worker) == report["worker_hashes"][label] for label, worker in sessions
    )
    report["passed"] = (
        len(report["cells"]) == len(plan) * len(sessions)
        and report["artifact_unchanged"] and report["workers_unchanged"]
        and all(cell.get("summary", {}).get("passed") is True for cell in report["cells"])
        and all(item["same_route_hashes_equal"] for item in report["worker_comparisons"])
    )
    report["timing_eligible"] = report["passed"] and all(
        cell["summary"]["timing_eligible"] for cell in report["cells"]
    )
    mtp._atomic_report(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "worker", "corpus", "output", "cells"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--candidate-worker", type=Path)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--weights-receipt", type=Path, help="Full-payload hash receipt with local file identity guards.")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--draft-depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screened", action="store_true")
    parser.add_argument("--observe-host", action="store_true", help="Complete the grid with host observations and unqualified timings.")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ChatError):
        print(json.dumps({"passed": False, "stage": "pilot_failed"}))
        return 1
    print(json.dumps({"passed": report["passed"], "timing_eligible": report["timing_eligible"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
