"""Run bounded native GLM validation, screening, and confirmation phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("benchmark_glm_llama_v1.py")
VERIFIER = Path(__file__).with_name("verify_glm53_llama_weights.py")
COMPARISONS = {
    "whole_profile": ("defaults", "profile"),
    "mma": ("mma_off", "profile"),
    "mla": ("profile", "mla"),
    "kda": ("profile", "kda"),
    "combination": ("profile", "combination"),
}
BASE_CONFIGS = ("defaults", "profile", "mma_off", "mla", "kda")


class CampaignError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CampaignError(code)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _regular(path: Path, code: str) -> Path:
    require(not path.is_symlink() and path.is_file(), code)
    return path.resolve()


def prepare(args) -> tuple[Path, Path, Path, Path, Path, Path]:
    model = args.model
    require(not model.is_symlink() and model.is_dir(), "model_directory_invalid")
    model = model.resolve()
    for name in ("config.json", "tokenizer.json", "native-source.json"):
        _regular(model / name, "model_source_invalid")
    worker = _regular(args.worker, "worker_invalid")
    example = _regular(args.example, "readout_example_invalid")
    corpus = _regular(args.corpus, "corpus_invalid")
    _regular(RUNNER, "runner_source_invalid")
    _regular(VERIFIER, "verifier_source_invalid")
    require(os.access(worker, os.X_OK), "worker_not_executable")
    require(os.access(example, os.X_OK), "readout_example_not_executable")
    receipt = args.verified_weights
    if receipt.exists() or receipt.is_symlink():
        require(not receipt.is_symlink() and receipt.is_file(), "weight_receipt_invalid")
        receipt = receipt.resolve()
    else:
        parent = receipt.parent
        require(parent.exists() and parent.is_dir(), "weight_receipt_parent_invalid")
        receipt = parent.resolve() / receipt.name
    output = args.output
    require(not output.exists() and not output.is_symlink(), "output_already_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    return model, worker, example, corpus, receipt, output


def load_report(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _relative_range(summary: dict, field: str) -> float | None:
    value = summary.get(field, {}).get("relative_range")
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _max_range(summary: dict, field: str) -> float | None:
    before = _relative_range(summary, "before_" + field)
    after = _relative_range(summary, "after_" + field)
    return max(before, after) if before is not None and after is not None else None


def screen_decision(report: dict) -> dict:
    summaries = report.get("summary")
    if report.get("passed") is not True or not isinstance(summaries, list) or not summaries:
        return {"survives": False, "reason": "functional_screen_failed"}
    cells = []
    for summary in summaries:
        wall = summary.get("median_paired_wall_reduction")
        wall_range = _max_range(summary, "wall_seconds")
        gain = summary.get("median_paired_decode_gain")
        decode_range = _max_range(summary, "useful_decode_tokens_per_second")
        wall_signal = type(wall) in (int, float) and wall_range is not None and wall >= 0.05 and wall > wall_range
        decode_signal = (
            type(gain) in (int, float) and decode_range is not None and gain >= 0.05 and gain > decode_range
        )
        no_large_regression = type(wall) in (int, float) and wall >= -0.10
        cells.append({
            "input_tokens": summary.get("input_tokens"),
            "requested_output_tokens": summary.get("requested_output_tokens"),
            "median_wall_reduction": wall,
            "wall_relative_range_max": wall_range,
            "median_useful_decode_gain": gain,
            "decode_relative_range_max": decode_range,
            "wall_signal": wall_signal,
            "decode_signal": decode_signal,
            "no_more_than_10_percent_wall_regression": no_large_regression,
        })
    survives = any(cell["wall_signal"] or cell["decode_signal"] for cell in cells) and all(
        cell["no_more_than_10_percent_wall_regression"] for cell in cells
    )
    return {"survives": survives, "cells": cells,
            "reason": "screen_hurdles_met" if survives else "screen_hurdles_not_met"}


def _full_budget_decode_rows(report: dict, input_tokens: int, budget: int) -> bool:
    rows = [
        request
        for session in report.get("sessions", [])
        for request in session.get("requests", [])
        if request.get("input_tokens") == input_tokens
        and request.get("requested_output_tokens") == budget
        and request.get("warmup") is False
    ]
    return len(rows) == 16 and all(request.get("sampled_tokens") == budget for request in rows)


def confirmation_decision(report: dict) -> dict:
    summaries = report.get("summary")
    if report.get("passed") is not True or not isinstance(summaries, list) or not summaries:
        return {"functional_passed": report.get("passed") is True, "performance_confirmed": False,
                "reason": "functional_confirmation_failed"}
    cells = []
    for summary in summaries:
        wall = summary.get("median_paired_wall_reduction")
        wall_range = _max_range(summary, "wall_seconds")
        gain = summary.get("median_paired_decode_gain")
        decode_range = _max_range(summary, "useful_decode_tokens_per_second")
        pairs = summary.get("pairs")
        wall_confirming = (
            type(wall) in (int, float) and wall_range is not None
            and wall >= 0.10 and wall > wall_range
            and pairs == 8 and summary.get("wall_pair_wins", 0) >= 7
        )
        full_decode_cell = (summary.get("input_tokens"), summary.get("requested_output_tokens")) == (512, 128)
        full_decode_observed = full_decode_cell and _full_budget_decode_rows(report, 512, 128)
        decode_confirming = (
            type(gain) in (int, float) and decode_range is not None
            and gain >= 0.10 and gain > decode_range
            and pairs == 8 and summary.get("decode_pair_wins", 0) >= 7
        )
        cells.append({
            "input_tokens": summary.get("input_tokens"),
            "requested_output_tokens": summary.get("requested_output_tokens"),
            "pairs": pairs,
            "median_wall_reduction": wall,
            "wall_relative_range_max": wall_range,
            "wall_pair_wins": summary.get("wall_pair_wins"),
            "median_useful_decode_gain": gain,
            "decode_relative_range_max": decode_range,
            "decode_pair_wins": summary.get("decode_pair_wins"),
            "full_budget_decode_cell": full_decode_cell,
            "full_budget_decode_observed": full_decode_observed,
            "wall_confirming": wall_confirming,
            "decode_confirming": decode_confirming,
            "no_more_than_5_percent_wall_regression": type(wall) in (int, float) and wall >= -0.05,
            "no_more_than_5_percent_decode_regression": (
                type(gain) in (int, float) and gain >= -0.05
            ),
        })
    wall_confirmed = sum(cell["wall_confirming"] for cell in cells) >= 2
    decode_confirming_cells = [cell for cell in cells if cell["decode_confirming"]]
    long_decode_confirmed = any(cell["full_budget_decode_observed"] for cell in decode_confirming_cells)
    wall_regression_ok = all(cell["no_more_than_5_percent_wall_regression"] for cell in cells)
    decode_regression_ok = all(cell["no_more_than_5_percent_decode_regression"] for cell in cells)
    wall_confirmed = wall_confirmed and wall_regression_ok
    general_decode_confirmed = (
        long_decode_confirmed and len(decode_confirming_cells) >= 2 and decode_regression_ok and wall_regression_ok
    )
    confirming = wall_confirmed or general_decode_confirmed
    effect = "complete_request_latency" if wall_confirmed else (
        "general_useful_decode_throughput" if general_decode_confirmed else None
    )
    return {"functional_passed": True, "performance_confirmed": confirming,
            "confirmed_effect": effect,
            "wall_confirming_cells": sum(cell["wall_confirming"] for cell in cells),
            "decode_confirming_cells": len(decode_confirming_cells),
            "general_decode_long_cell_confirmed": long_decode_confirmed,
            "wall_regression_guard_passed": wall_regression_ok,
            "decode_regression_guard_passed": decode_regression_ok,
            "cells": cells, "reason": "confirmation_hurdles_met" if confirming
            else "confirmation_hurdles_not_met"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "worker", "example", "corpus", "verified-weights", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(argv)
    active_child = None
    interrupted = False
    output_owned = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        if active_child is not None and active_child.poll() is None:
            try:
                active_child.terminate()
            except OSError:
                pass

    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    report = {"schema": "stark-glm-llama-campaign-v1", "status": "starting", "phases": [],
              "screens": {}, "confirmations": {}, "source_sha256": {}}
    try:
        model, worker, example, corpus, receipt, output = prepare(args)
        output_owned = True
        report["source_sha256"] = {"runner": digest(RUNNER), "verifier": digest(VERIFIER),
                                   "campaign": digest(Path(__file__).resolve())}
        log_path = output / "stdout.log"
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        log = os.fdopen(log_fd, "ab", buffering=0)
        report_path = output / "campaign.json"

        def persist():
            write_json(report_path, report)

        def run_child(label, command, phase_output: Path, *, expect_report=True) -> tuple[dict, dict | None]:
            nonlocal active_child, interrupted
            row = {"name": label, "status": "starting"}
            report["phases"].append(row)
            log.write((json.dumps({"phase_start": label}) + "\n").encode())
            persist()
            if interrupted:
                row["status"] = "interrupted"
                persist()
                return row, None
            try:
                active_child = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.DEVNULL,
                    env=dict(os.environ),
                )
                code = active_child.wait()
            except (OSError, subprocess.SubprocessError) as error:
                row.update(status="failed", failure_type=type(error).__name__, failure_code="phase_launch_failed")
                if active_child is not None and active_child.poll() is None:
                    interrupted = True
                else:
                    active_child = None
                persist()
                return row, None
            finally:
                if active_child is not None and active_child.poll() is not None:
                    active_child = None
            value = load_report(phase_output / "report.json") if phase_output.is_dir() else None
            row.update(status="interrupted" if interrupted else ("completed" if code == 0 else "failed"),
                       exit_code=code, report_passed=value.get("passed") if value else None)
            if value is None and expect_report:
                row.update(failure_type="PhaseReportUnavailable", failure_code="phase_report_unavailable")
                if code == 0:
                    row["status"] = "failed"
            log.write((json.dumps({"phase_end": label, "exit_code": code,
                                   "status": row["status"]}) + "\n").encode())
            persist()
            return row, value

        def run_runner(phase, label, *, validation=None, prefix=None, comparison=None,
                       configs=None, combination=False, decode_confirm=False):
            phase_dir = output / f"{len(report['phases']):02d}-{label}"
            command = [sys.executable, str(RUNNER), phase, "--model", str(model), "--worker", str(worker),
                       "--example", str(example), "--corpus", str(corpus), "--verified-weights", str(receipt),
                       "--output", str(phase_dir)]
            if validation is not None:
                command.extend(("--validation-receipt", str(validation)))
            if prefix is not None:
                command.extend(("--prefix-receipt", str(prefix)))
            if comparison is not None:
                command.extend(("--comparison", comparison))
            if configs is not None:
                command.extend(("--configs", ",".join(configs)))
            if combination:
                command.append("--include-combination")
            if decode_confirm:
                command.append("--decode-confirm")
            return run_child(label, command, phase_dir)

        if not receipt.exists():
            verify_dir = output / f"{len(report['phases']):02d}-verify-weights"
            verify_dir.mkdir(mode=0o700)
            row, _ = run_child("verify_weights", [sys.executable, str(VERIFIER), "--model", str(model),
                                                   "--output", str(receipt)], verify_dir, expect_report=False)
            receipt_data = load_report(receipt) if receipt.is_file() and not receipt.is_symlink() else None
            if row["status"] != "completed" or not receipt_data or receipt_data.get("passed") is not True or interrupted:
                report["status"] = "interrupted" if interrupted else "failed"
                if not interrupted:
                    report["failure_code"] = "weight_verification_failed"
                persist()
                return 130 if interrupted else 1

        _validation_row, validation = run_runner("validate", "validate", combination=False)
        validation_dir = output / f"{len(report['phases']) - 1:02d}-validate"
        validation_path = validation_dir / "report.json"
        if validation is None:
            report["status"] = "interrupted" if interrupted else "failed"
            persist()
            return 130 if interrupted else 1
        report["validation"] = {name: bool(validation.get("validation", {}).get(name, {}).get("passed"))
                                for name in BASE_CONFIGS}
        expected_ids, expected_weights = validation.get("identities"), validation.get("weights")
        require(expected_ids is not None and expected_weights is not None, "validation_identity_missing")
        require(report["source_sha256"]["runner"] == digest(RUNNER)
                and report["source_sha256"]["verifier"] == digest(VERIFIER)
                and report["source_sha256"]["campaign"] == digest(Path(__file__).resolve()),
                "campaign_source_changed")

        def identity_ok(value):
            return (
                value is not None and value.get("identities") == expected_ids
                and value.get("weights") == expected_weights
                and digest(RUNNER) == report["source_sha256"]["runner"]
                and digest(VERIFIER) == report["source_sha256"]["verifier"]
                and digest(Path(__file__).resolve()) == report["source_sha256"]["campaign"]
            )

        for comparison, (before, after) in list(COMPARISONS.items())[:4]:
            if interrupted:
                break
            if not (report["validation"].get(before) and report["validation"].get(after)):
                report["screens"][comparison] = {"status": "skipped", "reason": "arms_not_validated"}
                persist()
                continue
            row, screen = run_runner("screen", f"screen-{comparison}", validation=validation_path,
                                     comparison=comparison)
            if screen is not None and not identity_ok(screen):
                screen = None
                row.update(status="failed", failure_type="IdentityMismatch", failure_code="phase_identity_mismatch")
            decision = screen_decision(screen or {}) if row["status"] == "completed" else {
                "survives": False, "reason": "screen_phase_not_completed"
            }
            report["screens"][comparison] = {"phase_status": row["status"], **decision}
            persist()

        for comparison, (before, after) in list(COMPARISONS.items())[:4]:
            if interrupted:
                break
            screen = report["screens"].get(comparison, {})
            if not screen.get("survives"):
                continue
            row, prefix = run_runner("prefix", f"prefix-{comparison}", validation=validation_path,
                                     configs=(before, after))
            prefix_dir = output / f"{len(report['phases']) - 1:02d}-prefix-{comparison}"
            prefix_path = prefix_dir / "report.json"
            if row["status"] != "completed" or prefix is None or not identity_ok(prefix) or prefix.get("passed") is not True:
                report["confirmations"][comparison] = {"functional_passed": False,
                    "performance_confirmed": False, "reason": "prefix_control_failed"}
                persist()
                continue
            row, confirmed = run_runner("confirm", f"confirm-{comparison}", validation=validation_path,
                                        prefix=prefix_path, comparison=comparison, decode_confirm=True)
            if row["status"] != "completed":
                confirmed = None
            elif confirmed is not None and not identity_ok(confirmed):
                confirmed = None
                row.update(status="failed", failure_type="IdentityMismatch", failure_code="phase_identity_mismatch")
            decision = confirmation_decision(confirmed or {})
            report["confirmations"][comparison] = {"phase_status": row["status"], **decision}
            persist()

        if not interrupted and report["screens"].get("mla", {}).get("survives") \
                and report["screens"].get("kda", {}).get("survives"):
            combo_row, combo_validation = run_runner("validate", "validate-combination", combination=True)
            combo_dir = output / f"{len(report['phases']) - 1:02d}-validate-combination"
            combo_validation_path = combo_dir / "report.json"
            combo_valid = combo_row["status"] == "completed" and combo_validation is not None \
                and combo_validation.get("passed") is True and identity_ok(combo_validation) \
                and combo_validation.get("validation", {}).get("combination", {}).get("passed") is True
            if combo_valid and not interrupted:
                row, combo_screen = run_runner("screen", "screen-combination", validation=combo_validation_path,
                                               comparison="combination")
                if combo_screen is not None and not identity_ok(combo_screen):
                    combo_screen = None
                    row.update(status="failed", failure_type="IdentityMismatch", failure_code="phase_identity_mismatch")
                decision = screen_decision(combo_screen or {}) if row["status"] == "completed" else {
                    "survives": False, "reason": "screen_phase_not_completed"
                }
                report["screens"]["combination"] = {"phase_status": row["status"], **decision}
                persist()
                if report["screens"]["combination"].get("survives") and not interrupted:
                    row, prefix = run_runner("prefix", "prefix-combination", validation=combo_validation_path,
                                             configs=COMPARISONS["combination"])
                    prefix_dir = output / f"{len(report['phases']) - 1:02d}-prefix-combination"
                    prefix_path = prefix_dir / "report.json"
                    if row["status"] == "completed" and prefix is not None \
                            and identity_ok(prefix) and prefix.get("passed") is True:
                        row, confirmed = run_runner("confirm", "confirm-combination",
                            validation=combo_validation_path, prefix=prefix_path,
                            comparison="combination", decode_confirm=True)
                        if row["status"] != "completed":
                            confirmed = None
                        elif confirmed is not None and not identity_ok(confirmed):
                            confirmed = None
                            row.update(status="failed", failure_type="IdentityMismatch", failure_code="phase_identity_mismatch")
                        report["confirmations"]["combination"] = {
                            "phase_status": row["status"], **confirmation_decision(confirmed or {})}
                    else:
                        report["confirmations"]["combination"] = {
                            "functional_passed": False, "performance_confirmed": False,
                            "reason": "prefix_control_failed"}
                    persist()
            else:
                report["screens"]["combination"] = {"status": "skipped",
                    "reason": "combination_validation_failed"}
                persist()

        failed_phases = sum(row["status"] == "failed" for row in report["phases"])
        usable_screens = sum(bool(value.get("cells")) for value in report["screens"].values())
        report["failed_phase_count"] = failed_phases
        report["usable_comparison_count"] = usable_screens
        report["status"] = "interrupted" if interrupted else (
            "completed_with_failures" if failed_phases or not usable_screens else "completed"
        )
        persist()
        return 130 if interrupted else (1 if failed_phases or not usable_screens else 0)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - sanitize the CLI boundary; never log exception text.
        report["status"] = "interrupted" if interrupted else "failed"
        report["failure_type"] = type(error).__name__
        if isinstance(error, CampaignError):
            report["failure_code"] = error.code
        try:
            if output_owned and args.output.is_dir():
                write_json(args.output / "campaign.json", report)
        except (OSError, ValueError, TypeError):
            return 130 if interrupted else 1
        return 130 if interrupted else 1
    finally:
        if active_child is not None:
            if active_child.poll() is None:
                try:
                    active_child.terminate()
                except OSError:
                    pass
                try:
                    active_child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        active_child.kill()
                    except OSError:
                        pass
                    active_child.wait()
                except OSError:
                    pass
            active_child = None
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
