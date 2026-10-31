#!/usr/bin/env python3
"""Build one archived arm into its exclusive target and bind aggregate receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRIALS = ROOT / "runtime/gdn-next-gpu-20261031"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("single", "mtp"))
    parser.add_argument("arm", choices=("baseline", "q8conv", "fused1024", "q8reuse"))
    args = parser.parse_args()
    trial = TRIALS / args.mode
    preparation_path = trial / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    item = preparation["source_manifests"][args.arm]
    source = (trial / item["source_root"]).resolve()
    target = (trial / "builds" / args.arm).resolve()
    log = trial / f"build-{args.arm}.log"
    receipt_path = target / "build-receipt.json"
    if log.exists() or receipt_path.exists():
        raise ValueError("build_attempt_exists")
    spec = importlib.util.spec_from_file_location("preparation", HERE / "prepare_trials.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if helper.shared.tree_digest(helper.shared.source_file_map(source)) != item["source_tree_manifest_sha256"]:
        raise ValueError("build_source_changed")
    target.mkdir(parents=True, exist_ok=True)
    command = [str(Path("/Users/johnshahbazian/.cargo/bin/cargo")), "+1.98.1", "build", "--locked", "--release", "-p", "stark-generate"]
    overrides = {"CARGO_TARGET_DIR": str(target), "CARGO_BUILD_JOBS": "3"}
    started = time.monotonic()
    with log.open("xb") as stream:
        code = subprocess.call(command, cwd=source, env={**os.environ, **overrides},
                               stdout=stream, stderr=subprocess.STDOUT)
    receipt = {
        "schema": f"qwen38-gdn-next-{args.mode}-build-receipt-v1",
        "completed": code == 0, "exit_code": code, "locked_toolchain": "1.98.1",
        "command": command, "environment": overrides, "working_directory": str(source),
        "cargo_target_dir": str(target), "elapsed_seconds": time.monotonic() - started,
        "preparation_sha256": sha(preparation_path),
        "source_manifest_sha256": item["source_manifest_sha256"],
        "source_tree_manifest_sha256": item["source_tree_manifest_sha256"],
        "build_log_sha256": sha(log), "build_driver_sha256": sha(Path(__file__)),
    }
    if code == 0:
        after = helper.shared.tree_digest(helper.shared.source_file_map(source))
        if after != item["source_tree_manifest_sha256"]:
            raise ValueError("build_source_changed")
        receipt["worker_sha256"] = sha(target / "release/stark-generate")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "arm": args.arm, "completed": code == 0,
                      "worker_sha256": receipt.get("worker_sha256")}), flush=True)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - never persist native exception dumps.
        print(json.dumps({"completed": False, "category": "build_controller_failed"}), flush=True)
        raise SystemExit(1) from None
