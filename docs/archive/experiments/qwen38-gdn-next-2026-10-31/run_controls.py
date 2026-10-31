#!/usr/bin/env python3
"""Serialize ignored GPU controls; persist metrics/hashes without native dumps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / "runtime/gdn-next-gpu-20261031"
SUITES = {
    "fusion1024": ("gdn_fusion1024_research", "fusion1024.metal", [
        "fused1024_deployed_matches_production", "fused1024_capture_matches_production",
    ], "fused1024_component_abba_after_exactness"),
    "q8conv": ("gdn_q8_conv_research", "q8_conv.metal", [
        "q8_conv_deployed_matches_production", "q8_conv_capture_matches_production",
    ], "q8_conv_component_streaming_after_exactness"),
    "q8reuse": ("gdn_q8_rows_research", "q8_rows.metal", [
        f"q8_rows{i}_matches_stock_exactly" for i in range(1, 5)
    ], "q8_rows_component_rotating_after_exactness"),
}
spec = importlib.util.spec_from_file_location(
    "control", HERE.parent / "qwen38-gdn-fusion-2026-10-31/benchmark_model.py",
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def run_one(suite: str, test: str, version: str) -> dict:
    harness, shader, _, _ = SUITES[suite]
    binaries = sorted((RUN / "component-build/release/deps").glob(harness + "-*"))
    binaries = [p for p in binaries if p.is_file() and p.suffix == "" and os.access(p, os.X_OK)]
    control.check(len(binaries) == 1, "control_binary_set")
    binary = binaries[0]
    log = RUN / f"{test}-{version}.log"
    control.check(not log.exists(), "control_log_exists")
    watch = control.QuietWindow(998)
    child = None
    lines = []
    raw = bytearray()
    started = time.monotonic()
    code = None
    failure = None
    started_utc = datetime.now(UTC).isoformat()
    try:
        watch.start()
        with watch.lock:
            watch.check()
            child = subprocess.Popen([
                str(binary), test, "--exact", "--ignored", "--nocapture", "--test-threads=1", "--color", "never",
            ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "RUST_BACKTRACE": "0", "RUST_LIB_BACKTRACE": "0"})
            watch.worker = SimpleNamespace(process=child)
        raw.extend(child.communicate()[0])
        code = child.returncode
        watch.check()
    except BaseException as exc:  # noqa: BLE001 - preserve interruptions without native dumps.
        failure = control.safe_failure(exc)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            tail, _ = child.communicate()
            raw.extend(tail)
            code = child.returncode
        watch.close()
    text = bytes(raw).decode("utf-8", errors="replace")
    for line in text.splitlines():
        if (re.match(r"^(?:test \w+ \.\.\. )?(?:gdn_|q8_|qwen38_)", line)
                or line.startswith(("running ", "test result:", "test "))
                or line in {"ok", "FAILED"}):
            lines.append(line)
    log.write_text("\n".join(lines) + "\n")
    unsupported = "status=unsupported" in text
    matched = "test result: ok. 1 passed;" in text
    status = "unsupported" if unsupported else "passed" if code == 0 and matched and failure is None else "failed"
    row = {
        "suite": suite, "test": test, "status": status, "exit_code": code,
        "started_utc": started_utc, "completed_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "binary_sha256": control.sha(binary),
        "harness_sha256": control.sha(ROOT / f"vendor/lily-core/tests/{harness}.rs"),
        "shader_sha256": control.sha(HERE / shader),
        "raw_output_sha256": control.hashlib.sha256(raw).hexdigest(),
        "sanitized_log_sha256": control.sha(log), "metrics": lines,
        "quiet_window": watch.snapshot(),
        "native_mismatch_counts": [int(v) for v in re.findall(r"mismatched_elements=(\d+)", text)],
        "native_reported_nonfinite": "nonfinite" in text.lower(),
    }
    if failure:
        row.update(failure)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("correctness", "components"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITES), default=list(SUITES))
    args = parser.parse_args()
    output = RUN / f"{args.phase}-{args.version}.json"
    control.check(not output.exists(), "control_receipt_exists")
    report = {"schema": "qwen38-gdn-next-gpu-controls-v1", "phase": args.phase,
              "runner_sha256": control.sha(Path(__file__)), "rows": [], "completed": False}
    control.persist(output, report, initial=True)
    prior = None
    if args.phase == "components":
        prior = json.loads((RUN / f"correctness-{args.version}.json").read_text())
    for suite, (_, _, tests, component) in SUITES.items():
        if suite not in args.suites:
            continue
        if prior is not None:
            outcomes = [x for x in prior["rows"] if x["suite"] == suite]
            if not outcomes or any(x["status"] != "passed" for x in outcomes):
                continue
        for test in tests if prior is None else [component]:
            row = run_one(suite, test, args.version)
            report["rows"].append(row)
            control.persist(output, report)
            print(json.dumps({"suite": suite, "test": test, "status": row["status"]}), flush=True)
    report["completed"] = True
    control.persist(output, report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - fixed failure category; no private exception text.
        print(json.dumps({"category": "control_runner_failed"}), flush=True)
        raise SystemExit(1) from None
