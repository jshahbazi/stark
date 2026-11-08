"""Run a guarded, aggregate-only native versus MLX dense Q4 comparison.

The harnesses each perform their fixed one-warmup/three-measurement workload.
This driver owns engine ordering, host screening, identity checks, and private
aggregate receipts. It never stores prompts, token IDs, responses, or logits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from benchmark_stock_engines import HostMonitor, sleep_events

ROOT = Path(__file__).resolve().parents[1]
NATIVE_HARNESS = ROOT / "crates/stark-engine/examples/dense_profile.rs"
MLX_HARNESS = ROOT / "scripts/dense_profile_mlx.py"
RUNS_PER_ENGINE = 4
IDENTITY_FIELDS = (
    "scope",
    "input_sha256",
    "input_tokens",
    "returned_tokens",
    "decode_forward_steps",
    "context_tokens",
    "source_repo",
    "source_revision",
    "source_config_sha256",
    "source_weights_digest",
    "quantization",
    "warmup_runs",
    "measured_runs",
)
FORBIDDEN_FIELDS = {
    "input_token_ids",
    "generated_token_ids",
    "token_ids",
    "tokens",
    "text",
    "response",
    "prompt",
    "logits",
    "log_probs",
    "candidate_logits",
    "candidate_log_probabilities",
    "probabilities",
    "exception",
    "traceback",
    "argv",
    "command_line",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
NATIVE_ARTIFACT_FIELDS = (
    "native_weights_sha256",
    "native_config_sha256",
    "conversion_receipt_sha256",
)
NATIVE_PROFILE_FIELDS = ("backend", "activation_dtype", "device")
MLX_PROFILE_FIELDS = (
    "backend",
    "mlx_lm_revision",
    "mlx_version",
    "transformers_version",
    "floating_parameter_dtypes",
    "device",
)
IDENTITY_FAILURE_CODES = {
    "fixture_changed_during_campaign",
    "native_artifact_changed_during_campaign",
    "native_executable_changed_during_campaign",
    "native_executable_identity_mismatch",
    "mlx_python_changed_during_campaign",
    "mlx_harness_changed_during_campaign",
    "mlx_harness_identity_mismatch",
    "campaign_identity_mismatch",
    "engine_profile_changed_during_campaign",
    "missing_engine_profile_identity",
    "unexpected_engine_backend",
}


class ComparisonError(Exception):
    """A safe, stable error category suitable for aggregate receipts."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def validate_safe_tree(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ComparisonError("unsafe_harness_field")
            validate_safe_tree(child)
    elif isinstance(value, list):
        for child in value:
            validate_safe_tree(child)


def validate_report(report: dict) -> None:
    validate_safe_tree(report)
    if report.get("passed") is not True:
        raise ComparisonError("harness_not_passed")
    if report.get("scope") != "matched-pp512-tg128-greedy-v1":
        raise ComparisonError("unexpected_workload_scope")
    if (
        report.get("input_tokens"),
        report.get("returned_tokens"),
        report.get("decode_forward_steps"),
    ) != (
        512,
        128,
        128,
    ):
        raise ComparisonError("unexpected_work_counts")
    if report.get("warmup_runs") != 1 or report.get("measured_runs") != 3:
        raise ComparisonError("unexpected_repeat_counts")
    if not isinstance(report.get("input_sha256"), str) or not SHA256_RE.fullmatch(
        report["input_sha256"]
    ):
        raise ComparisonError("invalid_input_identity")
    if not isinstance(report.get("source_repo"), str) or not REPO_RE.fullmatch(
        report["source_repo"]
    ):
        raise ComparisonError("invalid_source_identity")
    if not isinstance(report.get("source_revision"), str) or not REVISION_RE.fullmatch(
        report["source_revision"]
    ):
        raise ComparisonError("invalid_source_identity")
    for key in ("source_config_sha256", "source_weights_digest"):
        if not isinstance(report.get(key), str) or not SHA256_RE.fullmatch(report[key]):
            raise ComparisonError("invalid_source_identity")
    quantization = report.get("quantization")
    if not isinstance(quantization, dict) or (
        quantization.get("mode"),
        quantization.get("bits"),
        quantization.get("group_size"),
    ) != ("affine", 4, 64):
        raise ComparisonError("unexpected_quantization")
    source_weights = report.get("source_weights")
    if not isinstance(source_weights, list) or not 1 <= len(source_weights) <= 64:
        raise ComparisonError("invalid_source_weights")
    for weight in source_weights:
        if not isinstance(weight, dict):
            raise ComparisonError("invalid_source_weights")
        if (
            not isinstance(weight.get("filename"), str)
            or Path(weight["filename"]).name != weight["filename"]
            or not weight["filename"].endswith(".safetensors")
            or type(weight.get("bytes")) is not int
            or not 0 < weight["bytes"] <= 128 * 1024**3
            or not isinstance(weight.get("sha256"), str)
            or not SHA256_RE.fullmatch(weight["sha256"])
        ):
            raise ComparisonError("invalid_source_weights")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise ComparisonError("unexpected_run_receipt")
    measured = [
        run for run in runs if isinstance(run, dict) and run.get("measured") is True
    ]
    if len(measured) != 3:
        raise ComparisonError("unexpected_measured_count")
    if [run.get("run") for run in runs if isinstance(run, dict)] != list(range(4)):
        raise ComparisonError("unexpected_run_order")
    if [run.get("measured") for run in runs if isinstance(run, dict)] != [
        False,
        True,
        True,
        True,
    ]:
        raise ComparisonError("unexpected_run_order")
    for run in runs:
        if not isinstance(run, dict):
            raise ComparisonError("invalid_run_receipt")
        if (
            run.get("input_tokens"),
            run.get("returned_tokens"),
            run.get("decode_forward_steps"),
        ) != (
            512,
            128,
            128,
        ):
            raise ComparisonError("unexpected_run_work_counts")
        if not isinstance(run.get("output_sha256"), str) or not SHA256_RE.fullmatch(
            run["output_sha256"]
        ):
            raise ComparisonError("missing_output_digest")
        for timing_key in ("request_total_seconds", "work_tokens_per_second"):
            timing = run.get(timing_key)
            if (
                not isinstance(timing, (int, float))
                or not math.isfinite(timing)
                or timing <= 0
            ):
                raise ComparisonError("invalid_timing")
    if (
        report.get("repeated_outputs_identical") is not True
        or len({run["output_sha256"] for run in runs}) != 1
    ):
        raise ComparisonError("nonrepeatable_outputs")
    measured_seconds = [run["request_total_seconds"] for run in measured]
    median_seconds = statistics.median(measured_seconds)
    reported_seconds = report.get("median_request_total_seconds")
    median_rate = report.get("median_work_tokens_per_second")
    if (
        not isinstance(reported_seconds, (int, float))
        or not math.isfinite(reported_seconds)
        or not math.isclose(reported_seconds, median_seconds, rel_tol=1e-9)
        or not isinstance(median_rate, (int, float))
        or not math.isfinite(median_rate)
        or not math.isclose(median_rate, 640 / median_seconds, rel_tol=1e-9)
        or any(
            not math.isclose(
                run["work_tokens_per_second"],
                640 / run["request_total_seconds"],
                rel_tol=1e-9,
            )
            for run in runs
        )
    ):
        raise ComparisonError("invalid_timing")


def assert_same_identity(reports: list[dict]) -> None:
    if len(reports) < 2:
        raise ComparisonError("insufficient_engine_reports")
    baseline = reports[0]
    for report in reports[1:]:
        if any(baseline.get(key) != report.get(key) for key in IDENTITY_FIELDS):
            raise ComparisonError("engine_identity_mismatch")


def native_artifact_identity(model: Path) -> dict:
    """Freeze the small receipt/config identities without rereading weights."""
    receipt_path = model / "conversion.json"
    if not 0 < receipt_path.stat().st_size <= 8 * 1024 * 1024:
        raise ComparisonError("invalid_conversion_receipt")
    raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError:
        raise ComparisonError("invalid_conversion_receipt") from None
    config_hash = sha256_file(model / "config.json")
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "ok"
        or receipt.get("config_sha256") != config_hash
        or not isinstance(receipt.get("output_sha256"), str)
        or not SHA256_RE.fullmatch(receipt["output_sha256"])
    ):
        raise ComparisonError("invalid_conversion_receipt")
    return {
        "native_weights_sha256": receipt["output_sha256"],
        "native_config_sha256": config_hash,
        "conversion_receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def freeze_file(source: Path, target: Path, mode: int) -> str:
    expected = sha256_file(source)
    with target.open("xb") as stream:
        os.chmod(target, mode)
        with source.open("rb") as original:
            shutil.copyfileobj(original, stream)
    if sha256_file(target) != expected or sha256_file(source) != expected:
        raise ComparisonError("campaign_component_changed_while_freezing")
    return expected


def freeze_engines(
    engines: list[dict], args: argparse.Namespace, run_id: str, fixture_hash: str
) -> None:
    """Snapshot standalone benchmark programs; bind the venv interpreter in place."""
    frozen_harness = args.output_dir / f"{run_id}-dense_profile_mlx.py"
    harness_hash = freeze_file(MLX_HARNESS, frozen_harness, 0o600)
    child_env = dict(os.environ)
    child_env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for engine in engines:
        engine["child_env"] = child_env
        engine["fixture"] = args.fixture
        engine["fixture_sha256"] = fixture_hash
        if engine["kind"] == "native":
            frozen_binary = args.output_dir / f"{run_id}-{engine['id']}-dense_profile"
            engine["binary_sha256"] = freeze_file(
                engine["binary"], frozen_binary, 0o700
            )
            engine["binary"] = frozen_binary
            engine["command"] = (
                str(frozen_binary),
                str(args.model),
                str(args.fixture),
            )
        else:
            # Resolving the venv's Python symlink would discard its environment.
            engine["python"] = args.mlx_python.absolute()
            engine["python_sha256"] = sha256_file(engine["python"])
            engine["harness"] = frozen_harness
            engine["harness_sha256"] = harness_hash
            engine["command"] = (
                str(engine["python"]),
                str(frozen_harness),
                "run",
                "--source",
                str(args.source),
                "--fixture",
                str(args.fixture),
            )


def validate_frozen_engine(engine: dict) -> None:
    if sha256_file(engine["fixture"]) != engine["fixture_sha256"]:
        raise ComparisonError("fixture_changed_during_campaign")
    if engine["kind"] == "native":
        if sha256_file(engine["binary"]) != engine["binary_sha256"]:
            raise ComparisonError("native_executable_changed_during_campaign")
    else:
        if sha256_file(engine["python"]) != engine["python_sha256"]:
            raise ComparisonError("mlx_python_changed_during_campaign")
        if sha256_file(engine["harness"]) != engine["harness_sha256"]:
            raise ComparisonError("mlx_harness_changed_during_campaign")


def bind_campaign_report(report: dict, engine: dict, binding: dict) -> None:
    """One source/workload and one immutable profile per engine for every load."""
    identity = {key: report[key] for key in IDENTITY_FIELDS}
    if binding.get("identity") is None:
        binding["identity"] = copy.deepcopy(identity)
    elif identity != binding["identity"]:
        raise ComparisonError("campaign_identity_mismatch")
    if engine["kind"] == "native":
        if report.get("backend") != "stark-metal":
            raise ComparisonError("unexpected_engine_backend")
        if report.get("executable_sha256") != engine["binary_sha256"]:
            raise ComparisonError("native_executable_identity_mismatch")
        if any(
            report.get(key) != value
            for key, value in binding["native_artifacts"].items()
        ):
            raise ComparisonError("native_artifact_changed_during_campaign")
        fields = NATIVE_PROFILE_FIELDS
    else:
        if report.get("backend") != "mlx-lm-direct-greedy":
            raise ComparisonError("unexpected_engine_backend")
        if report.get("harness_sha256") != engine["harness_sha256"]:
            raise ComparisonError("mlx_harness_identity_mismatch")
        fields = MLX_PROFILE_FIELDS
    if any(key not in report for key in fields):
        raise ComparisonError("missing_engine_profile_identity")
    profile = {key: report[key] for key in fields}
    profile["output_sha256"] = report["runs"][0]["output_sha256"]
    previous = binding["engine_profiles"].get(engine["id"])
    if previous is None:
        binding["engine_profiles"][engine["id"]] = copy.deepcopy(profile)
    elif profile != previous:
        raise ComparisonError("engine_profile_changed_during_campaign")


def engine_order(engine_ids: list[str], block: int) -> list[str]:
    """Return alternating/rotated orders with three-arm position counts within one."""
    if len(engine_ids) == 2:
        return engine_ids[:] if block % 2 == 0 else list(reversed(engine_ids))
    if len(engine_ids) == 3:
        table = (
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1),
            (0, 2, 1),
        )
        return [engine_ids[index] for index in table[block % 4]]
    raise ComparisonError("unsupported_engine_count")


def host_guard_result(
    monitor_result: dict,
    started_wall: float,
    ended_wall: float,
    started_mono: float,
    ended_mono: float,
) -> dict:
    discrepancy = (ended_wall - started_wall) - (ended_mono - started_mono)
    accepted = (
        monitor_result.get("samples", 0) > 0
        and monitor_result.get("sampling_failures", 0) == 0
        and not monitor_result.get("cpu_guard_exceeded", True)
        and not monitor_result.get("background_model_active", True)
        and abs(discrepancy) < 2
    )
    return {
        **monitor_result,
        "wall_monotonic_discrepancy_seconds": discrepancy,
        "accepted": accepted,
    }


def run_engine(engine: dict, args: argparse.Namespace, run_key: str) -> dict:
    output_path = (
        args.output_dir / f"{run_key}-{engine['id']}-mlx.json"
        if engine["kind"] == "mlx"
        else None
    )
    if output_path is not None and output_path.exists():
        raise ComparisonError("output_collision")
    validate_frozen_engine(engine)
    command = list(engine["command"])
    if output_path is not None:
        command += ["--output", str(output_path)]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=engine["child_env"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ComparisonError("engine_timeout") from None
    except OSError:
        raise ComparisonError("engine_launch_failed") from None
    if result.returncode != 0:
        raise ComparisonError("engine_process_failed")
    try:
        if engine["kind"] == "native":
            report = json.loads(result.stdout)
        else:
            report = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise ComparisonError("invalid_harness_report") from None
    if not isinstance(report, dict):
        raise ComparisonError("invalid_harness_report")
    validate_frozen_engine(engine)
    validate_report(report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        type=Path,
        required=True,
        help="Candidate native dense_profile executable",
    )
    parser.add_argument(
        "--baseline-native",
        type=Path,
        help="Optional baseline native dense_profile executable",
    )
    parser.add_argument(
        "--mlx-python",
        type=Path,
        required=True,
        help="Python executable in the frozen MLX reference environment",
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Immutable MLX Q4 source snapshot"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Converted native Q4 model directory"
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Existing private matched pp512/tg128 fixture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored private directory for aggregate receipts",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.timeout_seconds < 60:
        parser.error("--timeout-seconds must be at least 60")
    return args


def ignored_path(path: Path) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(path)], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def build_engines(args: argparse.Namespace) -> list[dict]:
    engines = []
    if args.baseline_native:
        engines.append(
            {
                "id": "baseline_native",
                "kind": "native",
                "binary": args.baseline_native.resolve(strict=True),
            }
        )
    engines.append(
        {
            "id": "candidate_native",
            "kind": "native",
            "binary": args.native.resolve(strict=True),
        }
    )
    engines.append({"id": "mlx", "kind": "mlx"})
    return engines


def paired_summary(
    blocks: list[dict], native_id: str, campaign_guard_accepted: bool
) -> dict:
    accepted = []
    for block in blocks:
        native = block["engines"].get(native_id)
        mlx = block["engines"].get("mlx")
        if (
            campaign_guard_accepted
            and block.get("identity_match") is True
            and native
            and mlx
            and native.get("guard", {}).get("accepted") is True
            and mlx.get("guard", {}).get("accepted") is True
            and native.get("status") == "complete"
            and mlx.get("status") == "complete"
        ):
            nrate = native["report"]["median_work_tokens_per_second"]
            mrate = mlx["report"]["median_work_tokens_per_second"]
            accepted.append(
                {"block": block["block"], "native_to_mlx_rate_ratio": nrate / mrate}
            )
    ratios = [item["native_to_mlx_rate_ratio"] for item in accepted]
    return {
        "selection_rule": "include every paired block only when both engine guards, campaign sleep/clock checks, and identity checks pass; no best-run selection",
        "accepted_pair_count": len(accepted),
        "accepted_pairs": accepted,
        "median_native_to_mlx_rate_ratio": statistics.median(ratios)
        if ratios
        else None,
    }


def run(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    if not ignored_path(args.output_dir):
        raise ComparisonError("output_directory_not_ignored")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    args.fixture = args.fixture.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.source = args.source.resolve(strict=True)
    for path in (args.fixture, args.model / "config.json", args.source / "config.json"):
        if not path.is_file():
            raise ComparisonError("required_input_missing")
    engines = build_engines(args)
    for engine in engines:
        if engine["kind"] == "native" and not os.access(engine["binary"], os.X_OK):
            raise ComparisonError("native_binary_not_executable")
    if not args.mlx_python.is_file():
        raise ComparisonError("mlx_python_missing")
    run_id = uuid.uuid4().hex
    output_path = args.output_dir / f"dense-q4-comparison-{run_id}.json"
    fixture_hash = sha256_file(args.fixture)
    artifacts = native_artifact_identity(args.model)
    freeze_engines(engines, args, run_id, fixture_hash)
    binding = {
        "identity": None,
        "native_artifacts": artifacts,
        "engine_profiles": {},
    }
    frozen_mlx = next(engine for engine in engines if engine["kind"] == "mlx")
    static_meta = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "matched-pp512-tg128-greedy-v1",
        "loads_per_engine": RUNS_PER_ENGINE,
        "fixture_sha256": fixture_hash,
        "native_harness_sha256": sha256_file(NATIVE_HARNESS),
        "mlx_harness_sha256": frozen_mlx["harness_sha256"],
        "mlx_python_sha256": frozen_mlx["python_sha256"],
        "native_artifact_identity": artifacts,
        "engines": [
            {
                "id": engine["id"],
                "kind": engine["kind"],
                "binary_sha256": engine["binary_sha256"]
                if engine["kind"] == "native"
                else None,
            }
            for engine in engines
        ],
        "blocks": [],
        "failures": [],
        "campaign_status": "running",
        "timing_accepted": False,
        "identity_guard_accepted": True,
    }
    private_write(output_path, static_meta)
    campaign_wall_start, campaign_mono_start = time.time(), time.monotonic()
    try:
        for block_index in range(RUNS_PER_ENGINE):
            block = {
                "block": block_index,
                "order": engine_order(
                    [engine["id"] for engine in engines], block_index
                ),
                "engines": {},
            }
            static_meta["blocks"].append(block)
            reports = []
            for engine_id in block["order"]:
                engine = next(item for item in engines if item["id"] == engine_id)
                started_wall, started_mono = time.time(), time.monotonic()
                with HostMonitor() as monitor:
                    try:
                        report = run_engine(engine, args, f"{run_id}-b{block_index}")
                        bind_campaign_report(report, engine, binding)
                        status, failure = "complete", None
                    except ComparisonError as exc:
                        report, status, failure = None, "failed", exc.code
                        if exc.code in IDENTITY_FAILURE_CODES:
                            static_meta["identity_guard_accepted"] = False
                            static_meta["failures"].append(
                                {
                                    "block": block_index,
                                    "engine": engine_id,
                                    "error": exc.code,
                                }
                            )
                    except Exception:  # noqa: BLE001 - keep unexpected exceptions out of private reports.
                        report, status, failure = (
                            None,
                            "failed",
                            "runner_operation_failed",
                        )
                ended_wall, ended_mono = time.time(), time.monotonic()
                guard = host_guard_result(
                    monitor.result(), started_wall, ended_wall, started_mono, ended_mono
                )
                entry = {"status": status, "failure": failure, "guard": guard}
                if report is not None:
                    reports.append(report)
                    entry["report"] = report
                block["engines"][engine_id] = entry
                private_write(output_path, static_meta)
            block["identity_match"] = len(reports) == len(engines)
            if block["identity_match"]:
                try:
                    assert_same_identity(reports)
                except ComparisonError:
                    block["identity_match"] = False
                    static_meta["failures"].append(
                        {"block": block_index, "error": "engine_identity_mismatch"}
                    )
            if block["identity_match"]:
                block["cross_engine_output_hashes_agree"] = (
                    len({report["runs"][0]["output_sha256"] for report in reports}) == 1
                )
            private_write(output_path, static_meta)
        campaign_wall_end, campaign_mono_end = time.time(), time.monotonic()
        static_meta["campaign_elapsed_seconds"] = (
            campaign_mono_end - campaign_mono_start
        )
        static_meta["sleep_clock_discrepancy_seconds"] = (
            campaign_wall_end - campaign_wall_start
        ) - (campaign_mono_end - campaign_mono_start)
        static_meta["power_log_screen"] = sleep_events(
            campaign_wall_start, campaign_wall_end
        )
        sleep = static_meta["power_log_screen"]
        campaign_guard_accepted = (
            sleep.get("checked") is True
            and sleep.get("sleep_events") == 0
            and abs(static_meta["sleep_clock_discrepancy_seconds"]) < 2
            and static_meta["identity_guard_accepted"] is True
        )
        static_meta["campaign_guard_accepted"] = campaign_guard_accepted
        native_ids = [engine["id"] for engine in engines if engine["kind"] == "native"]
        static_meta["paired_summaries"] = {
            engine_id: paired_summary(
                static_meta["blocks"], engine_id, campaign_guard_accepted
            )
            for engine_id in native_ids
        }
        static_meta["timing_accepted"] = campaign_guard_accepted and all(
            block.get("identity_match") is True
            and all(
                item.get("status") == "complete"
                and item.get("guard", {}).get("accepted") is True
                for item in block["engines"].values()
            )
            for block in static_meta["blocks"]
        )
        static_meta["campaign_status"] = "complete"
    except Exception:  # noqa: BLE001 - record only a stable failure category.
        static_meta["campaign_status"] = "failed"
        static_meta["failures"].append({"error": "comparison_campaign_failed"})
    finally:
        if static_meta.get("campaign_status") == "running":
            static_meta["campaign_status"] = "failed"
            static_meta["failures"].append({"error": "comparison_campaign_incomplete"})
        private_write(output_path, static_meta)
    print(
        json.dumps(
            {
                "status": static_meta["campaign_status"],
                "timing_accepted": static_meta["timing_accepted"],
                "receipt": str(output_path),
            }
        )
    )
    return 0 if static_meta["campaign_status"] == "complete" else 1


def main() -> int:
    try:
        return run(parse_args())
    except ComparisonError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}))
        return 2
    except Exception:  # noqa: BLE001 - never expose exception text from model tooling.
        print(json.dumps({"status": "failed", "error": "comparison_runner_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
