"""Validate and aggregate the private Lily long-completion comparison.

The output contains only content-free timings, counters, host summaries, flags,
and hashes. It never reads or republishes prompt or generated text content.
This publisher performs no model, worker, or checker execution.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "runtime/lily-ttt-long/private-results-20261115"
OUT = RESULTS / "aggregate.json"
MODEL = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
IDENTITY = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2/identity.json")
OLD_RUNNER = ROOT / "runtime/lily-topk-policy/run_performance.py"
RUNNER = ROOT / "runtime/lily-ttt-long/run.py"
INITIAL_RUNNER = ROOT / "runtime/lily-ttt-long/run_initial_stdin.py"
PREFLIGHT = ROOT / "runtime/lily-ttt-long/stdin-preflight.json"
CHECKER = ROOT / "runtime/lily-ttt-long/check_program.py"
BASE = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate")
CANDIDATE = ROOT / "runtime/build-topk-candidate/release/stark-generate"
TEMPLATE_HELPER = ROOT / "runtime/build-candidate/release/stark-template"
BUILD_IDENTITY = ROOT / "runtime/lily-topk-policy/build-identity.json"
NATIVE_PATCH = ROOT / "runtime/lily-topk-policy/native.patch"
PYTHON_PATCH = ROOT / "runtime/lily-topk-policy/python.patch"
CONFIG = MODEL / "config.json"
NATIVE_RECEIPT = MODEL / "native-source.json"
TEMPLATE = MODEL / "chat_template.jinja"

EXPECTED = {
    "before_worker": "61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640",
    "candidate_worker": "e1e432fcd5d490c8ab900be23c7cbc3855fc4515d75a38089bac672a85e9be22",
    "runner": "9ffbb51121a91d1a5dd663a80fdf01fa968269dbe8b66dd34bddb7802ecbfb84",
    "initial_runner": "5dbb975b6c8867c4d654b5cf98da9d144e685d345b8e35e15f1b4c5a6f1d3c15",
    "checker": "9b31f9ea7868ee35b8b5923c625fc3a97074d6d6934739849c4b071028bf9700",
    "old_runner": "d5943a941b56c2065a0ee562214161e77e8a4939168057f58c42ae6c7d0147dc",
    "build_identity": "273c56d4470b36bf7257e62b5431b8b55081dbbc1e88a2d08aaeb6e79e93a691",
    "native_patch": "24bc87a033386e9150a7d264afb546e83cf5dd86b1aee9e9f0bce8d3aa3797c0",
    "python_patch": "e947c9919583e68d09678c684b7de80ed400a472ff8873150465d7efc6efd307",
    "template_helper": "a3697940cf58166e31a022b1788c5c48ed2605792e97f686ce4626dc05a27ca5",
    "config": "2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899",
    "template": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}

SOURCE_FILES = {
    "runner": RUNNER,
    "initial_stdin_runner": INITIAL_RUNNER,
    "checker": CHECKER,
    "identity_guard_runner": OLD_RUNNER,
    "build_identity": BUILD_IDENTITY,
    "native_patch": NATIVE_PATCH,
    "python_patch": PYTHON_PATCH,
    "template_helper": TEMPLATE_HELPER,
    "model_config": CONFIG,
    "native_source_receipt": NATIVE_RECEIPT,
    "chat_template": TEMPLATE,
    "chat_runtime": ROOT / "tools/converter/stark_convert/chat.py",
    "reasoning_runtime": ROOT / "tools/converter/stark_convert/reasoning.py",
    "template_runtime": ROOT / "tools/converter/stark_convert/template_runtime.py",
    "prefix_worker_tools": ROOT / "scripts/validate_generation_prefix_cache.py",
    "host_monitor_tools": ROOT / "scripts/benchmark_clean_qwen.py",
    "host_screen_tools": ROOT / "scripts/benchmark_glm_mlx_serve.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def summary(values: list[float | int]) -> dict:
    ordered = sorted(values)
    middle = statistics.median(ordered)
    return {
        "count": len(ordered),
        "median": middle,
        "min": ordered[0],
        "max": ordered[-1],
        "median_absolute_deviation": statistics.median(abs(value - middle) for value in ordered),
    }


def paired_change(candidate: float, before: float) -> float:
    return (candidate / before - 1.0) * 100.0


def import_old_guard():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_lily_long_old_runner", OLD_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("identity_guard_source_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source_and_model(manifest: dict) -> dict:
    for path in (BASE, CANDIDATE, RUNNER, INITIAL_RUNNER, PREFLIGHT, CHECKER,
                 OLD_RUNNER, BUILD_IDENTITY, NATIVE_PATCH, PYTHON_PATCH,
                 TEMPLATE_HELPER, CONFIG, NATIVE_RECEIPT, TEMPLATE, IDENTITY):
        if not path.is_file():
            raise RuntimeError("sealed_source_missing")
    actual = {
        "before_worker": sha256_file(BASE),
        "candidate_worker": sha256_file(CANDIDATE),
        "runner": sha256_file(RUNNER),
        "initial_runner": sha256_file(INITIAL_RUNNER),
        "checker": sha256_file(CHECKER),
        "old_runner": sha256_file(OLD_RUNNER),
        "build_identity": sha256_file(BUILD_IDENTITY),
        "native_patch": sha256_file(NATIVE_PATCH),
        "python_patch": sha256_file(PYTHON_PATCH),
        "template_helper": sha256_file(TEMPLATE_HELPER),
        "config": sha256_file(CONFIG),
        "template": sha256_file(TEMPLATE),
    }
    if actual != EXPECTED:
        raise RuntimeError("source_hash_mismatch")
    expected_from_run = {
        "before_worker": manifest["worker_sha256"]["before"],
        "candidate_worker": manifest["worker_sha256"]["candidate"],
        "runner": manifest["harness_sha256"],
        "checker": manifest["quality_checker_sha256"],
        "build_identity": manifest["build_identity_sha256"],
        "native_patch": manifest["native_patch_sha256"],
        "python_patch": manifest["python_patch_sha256"],
        "template_helper": manifest["template_helper_sha256"],
        "config": manifest["model_config_sha256"],
        "native_source_receipt": manifest["native_source_receipt_sha256"],
        "template": manifest["template_source_sha256"],
    }
    for key, value in expected_from_run.items():
        if key == "native_source_receipt":
            current = sha256_file(NATIVE_RECEIPT)
        else:
            current = actual[key]
        if current != value:
            raise RuntimeError("run_receipt_source_mismatch")

    old = import_old_guard()
    identity = json.loads(IDENTITY.read_text())
    expected_file_count = len(identity.get("model_file_identities", {})) + len(old.MODEL_EXTRA_FILE_SHA256)
    model_guard = bool(old.current_model_identity(identity))
    if not model_guard or expected_file_count != 72:
        raise RuntimeError("post_run_model_identity_mismatch")
    current_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    supplemental_source_hashes = {
        name: sha256_file(path) for name, path in SOURCE_FILES.items()
        if name not in {"runner", "initial_stdin_runner", "checker", "identity_guard_runner",
                        "build_identity", "native_patch", "python_patch", "template_helper",
                        "model_config", "native_source_receipt", "chat_template"}
    }
    return {
        "sha256": actual,
        "model_72_file_guard": {
            "passed": model_guard,
            "file_count": expected_file_count,
            "prior_manifest_files": len(identity.get("model_file_identities", {})),
            "ancillary_files_sha256_guarded": len(old.MODEL_EXTRA_FILE_SHA256),
            "method": "63 files checked against prior size/mtime/inode manifest; nine ancillary files checked by size/SHA-256",
            "weight_payloads_rehashed": False,
            "model_content_manifest_sha256": manifest["model_content_manifest_sha256"],
            "identity_receipt_sha256": sha256_file(IDENTITY),
        },
        "build_source_base": json.loads(BUILD_IDENTITY.read_text()).get("source_base"),
        "current_repository_commit_at_publication": current_commit,
        "source_hashes_not_sealed_in_generation_manifest": supplemental_source_hashes,
    }


def validate_run(manifest: dict, sessions: list[dict]) -> tuple[list[dict], list[dict]]:
    if (manifest.get("schema") != "stark-lily-ttt-long-completion-v1"
            or manifest.get("attempt") != "20261115-LILY-LONG-01"
            or manifest.get("completed") is not True
            or manifest.get("measured_rows_written") != 8
            or manifest.get("measured_count") != 8
            or manifest.get("warmup_count") != 4
            or manifest.get("session_build_order") != ["before", "candidate", "candidate", "before"]
            or manifest.get("prefix_cache") is not False
            or manifest.get("vision_worker") is not True
            or manifest.get("context_tokens") != 8192
            or manifest.get("max_new_tokens") != 2048
            or manifest.get("request_budget_mib") != 16
            or manifest.get("sampling") != {
                "mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": 0.95
            }):
        raise RuntimeError("run_manifest_profile_mismatch")
    if (len(sessions) != 4 or [item.get("build") for item in sessions]
            != ["before", "candidate", "candidate", "before"]):
        raise RuntimeError("session_matrix_mismatch")
    prompt_hashes = {manifest.get("prompt_sha256")}
    token_hashes = {manifest.get("input_token_ids_sha256", [None])[0]}
    rendered_hashes = {manifest.get("rendered_prompt_sha256", [None])[0]}
    rows = []
    warmups = []
    for expected_session, session in enumerate(sessions, start=1):
        if (session.get("session") != expected_session or session.get("setup_failure_type") is not None
                or session.get("input_tokens") != 24
                or session.get("input_token_ids_sha256") not in token_hashes
                or session.get("rendered_prompt_sha256") not in rendered_hashes
                or session.get("template_source_sha256") != manifest["template_source_sha256"]
                or session.get("template_helper_sha256") != manifest["template_helper_sha256"]
                or session.get("ready", {}).get("image_family_profile") != manifest["image_profile_required"]
                or session.get("ready", {}).get("mtp_supported") is not True):
            raise RuntimeError("session_identity_or_readiness_mismatch")
        warm = session.get("warmup")
        if (not isinstance(warm, dict) or warm.get("request_status") != "ok"
                or warm.get("actual_drafts") != (2 if session["build"] == "before" else 1)
                or warm.get("output_tokens") != 64 or warm.get("finish_reason") != "length"
                or warm.get("prefix_reuse_valid") is not True or warm.get("prefix_reused_tokens") != 0):
            raise RuntimeError("warmup_validation_mismatch")
        warmups.append({
            "session": expected_session,
            "build": session["build"],
            **warm,
        })
        measurements = session.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != 2:
            raise RuntimeError("measurement_count_mismatch")
        for repetition, row in enumerate(measurements, start=1):
            expected_depth = 2 if session["build"] == "before" else 1
            if (row.get("request_status") != "ok"
                    or row.get("session") != expected_session
                    or row.get("repetition") != repetition
                    or row.get("build") != session["build"]
                    or row.get("actual_drafts") != expected_depth
                    or row.get("automatic_depth_valid") is not True
                    or row.get("counts_valid") is not True
                    or row.get("timings_valid") is not True
                    or row.get("input_tokens") != 24
                    or row.get("prefix_reused_tokens") != 0
                    or row.get("prefix_reuse_valid") is not True
                    or row.get("finish_reason") != "eos"
                    or row.get("natural_eos") is not True
                    or row.get("output_tokens") != row.get("sampled_tokens") - 1
                    or row.get("output_tokens") >= 2048):
                raise RuntimeError("measurement_validation_mismatch")
            rows.append(dict(row))
    if len(prompt_hashes) != 1 or manifest.get("prompt_tokenization_match") is not True:
        raise RuntimeError("prompt_hash_consistency_mismatch")
    if (len(token_hashes) != 1 or len(rendered_hashes) != 1
            or len(set(manifest.get("prompt_codec_sha256", []))) != 1
            or len(set(manifest.get("input_token_ids_sha256", []))) != 1
            or len(set(manifest.get("rendered_prompt_sha256", []))) != 1):
        raise RuntimeError("cross_session_prompt_mismatch")
    return rows, warmups


def arm_summary(rows: list[dict], build: str) -> dict:
    arm = [row for row in rows if row["build"] == build]
    return {
        "request_wall_seconds": summary([row["wall_seconds"] for row in arm]),
        "generation_seconds": summary([row["generation_seconds"] for row in arm]),
        "decode_seconds": summary([row["decode_seconds"] for row in arm]),
        "prefill_seconds": summary([row["prefill_seconds"] for row in arm]),
        "useful_decode_tokens_per_second": summary([
            row["useful_decode_tokens_per_second"] for row in arm
        ]),
        "output_tokens": summary([row["output_tokens"] for row in arm]),
        "sampled_tokens": summary([row["sampled_tokens"] for row in arm]),
        "natural_eos_count": sum(row["natural_eos"] is True for row in arm),
        "length_capped_count": sum(row["finish_reason"] == "length" for row in arm),
        "output_token_id_sha256_values": sorted({row["output_token_ids_sha256"] for row in arm}),
        "text_sha256_values": sorted({row["text_sha256"] for row in arm}),
    }


def make_pairs(rows: list[dict]) -> list[dict]:
    by_key = {(row["session"], row["repetition"]): row for row in rows}
    schedule = (
        (1, 2, 1), (1, 2, 2),
        (4, 3, 1), (4, 3, 2),
    )
    pairs = []
    for index, (before_session, candidate_session, repetition) in enumerate(schedule, start=1):
        before = by_key[(before_session, repetition)]
        candidate = by_key[(candidate_session, repetition)]
        pairs.append({
            "pair": index,
            "half": "first" if index <= 2 else "second",
            "before_session": before_session,
            "candidate_session": candidate_session,
            "repetition": repetition,
            "before_wall_seconds": before["wall_seconds"],
            "candidate_wall_seconds": candidate["wall_seconds"],
            "request_wall_change_percent_candidate_vs_before": paired_change(
                candidate["wall_seconds"], before["wall_seconds"]
            ),
            "request_wall_delta_seconds_candidate_minus_before": (
                candidate["wall_seconds"] - before["wall_seconds"]
            ),
            "before_decode_seconds": before["decode_seconds"],
            "candidate_decode_seconds": candidate["decode_seconds"],
            "decode_time_change_percent_candidate_vs_before": paired_change(
                candidate["decode_seconds"], before["decode_seconds"]
            ),
            "before_useful_decode_tokens_per_second": before["useful_decode_tokens_per_second"],
            "candidate_useful_decode_tokens_per_second": candidate["useful_decode_tokens_per_second"],
            "useful_decode_rate_change_percent_candidate_vs_before": paired_change(
                candidate["useful_decode_tokens_per_second"],
                before["useful_decode_tokens_per_second"],
            ),
            "before_output_tokens": before["output_tokens"],
            "candidate_output_tokens": candidate["output_tokens"],
            "output_length_change_percent_candidate_vs_before": paired_change(
                candidate["output_tokens"], before["output_tokens"]
            ),
            "before_finish_reason": before["finish_reason"],
            "candidate_finish_reason": candidate["finish_reason"],
            "before_text_sha256": before["text_sha256"],
            "candidate_text_sha256": candidate["text_sha256"],
        })
    return pairs


def run() -> int:
    if OUT.exists():
        raise RuntimeError("private_aggregate_already_exists")
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    sessions = [json.loads((RESULTS / f"session-{index}.json").read_text())
                for index in range(1, 5)]
    rows, warmups = validate_run(manifest, sessions)
    source = validate_source_and_model(manifest)
    pairs = make_pairs(rows)
    wall_pair_changes = [row["request_wall_change_percent_candidate_vs_before"] for row in pairs]
    rate_pair_changes = [row["useful_decode_rate_change_percent_candidate_vs_before"] for row in pairs]
    decode_pair_changes = [row["decode_time_change_percent_candidate_vs_before"] for row in pairs]
    first_half = [row for row in pairs if row["half"] == "first"]
    second_half = [row for row in pairs if row["half"] == "second"]
    quality = manifest.get("quality_by_text_sha256", {})
    if (len(quality) != manifest.get("quality_unique_output_count")
            or set(quality) != {row["text_sha256"] for row in rows}):
        raise RuntimeError("quality_hash_set_mismatch")
    for value in quality.values():
        if value.get("syntax_valid") is not True or value.get("sandbox_or_runtime_failure") is not True:
            raise RuntimeError("quality_metadata_mismatch")

    preflight = json.loads(PREFLIGHT.read_text())
    if (preflight.get("worker_started") is not False
            or preflight.get("native_requests") != 0
            or preflight.get("output_directory_created") is not False
            or preflight.get("harness_sha256") != EXPECTED["initial_runner"]):
        raise RuntimeError("initial_stdin_preflight_mismatch")

    host_measurements = [row["host"] for row in rows]
    host_warmups = [row["host"] for row in warmups]
    all_host = host_measurements + host_warmups
    cpu_samples = [item.get("external_cpu_max") for item in all_host
                   if isinstance(item.get("external_cpu_max"), (int, float))]
    thermal_samples = [item.get("thermal_max") for item in all_host
                       if isinstance(item.get("thermal_max"), (int, float))]
    paging = {
        "pageouts_total": sum(item.get("paging_delta", {}).get("pageouts", 0) for item in all_host),
        "swapouts_total": sum(item.get("paging_delta", {}).get("swapouts", 0) for item in all_host),
    }
    report = {
        "schema": "stark-lily-long-completion-aggregate-v1",
        "attempt": manifest["attempt"],
        "started_utc": manifest["started_utc"],
        "finished_utc": manifest["finished_utc"],
        "design": {
            "sessions": manifest["session_build_order"],
            "measured_requests": len(rows),
            "warmups": len(warmups),
            "positional_pairs": len(pairs),
            "context_tokens": manifest["context_tokens"],
            "max_new_tokens": manifest["max_new_tokens"],
            "sampling": manifest["sampling"],
            "seed": 42,
            "mtp_selector": "auto",
            "actual_depths": {"before": 2, "candidate": 1},
            "prefix_cache_enabled": False,
            "image_profile_loaded": manifest["image_profile_required"],
            "request_budget_mib": manifest["request_budget_mib"],
            "prompt_sha256": manifest["prompt_sha256"],
            "rendered_prompt_sha256": manifest["rendered_prompt_sha256"][0],
            "input_tokens": sessions[0]["input_tokens"],
            "input_token_ids_sha256": manifest["input_token_ids_sha256"][0],
        },
        "source_and_artifact_identity": source,
        "stdin_preflight_failure": preflight,
        "sampler_and_prefix_guards": {
            "actual_depths_valid": all(row["automatic_depth_valid"] for row in rows + warmups),
            "counts_and_timings_valid": all(row["counts_valid"] and row["timings_valid"] for row in rows + warmups),
            "zero_prefix_reuse_valid": all(row["prefix_reuse_valid"] for row in rows + warmups),
            "cross_session_prompt_match": manifest["prompt_tokenization_match"],
            "same_arm_repeatability": manifest["same_arm_repeatability"],
        },
        "per_request_observations": rows,
        "warmup_observations": warmups,
        "per_build_summary": {
            "before_two_drafts": arm_summary(rows, "before"),
            "candidate_one_draft": arm_summary(rows, "candidate"),
        },
        "paired_comparisons": pairs,
        "paired_change_summary": {
            "request_wall_change_percent_candidate_vs_before": summary(wall_pair_changes),
            "decode_time_change_percent_candidate_vs_before": summary(decode_pair_changes),
            "useful_decode_rate_change_percent_candidate_vs_before": summary(rate_pair_changes),
            "first_half_median_request_wall_change_percent": statistics.median(
                row["request_wall_change_percent_candidate_vs_before"] for row in first_half
            ),
            "second_half_median_request_wall_change_percent": statistics.median(
                row["request_wall_change_percent_candidate_vs_before"] for row in second_half
            ),
            "first_half_median_useful_decode_rate_change_percent": statistics.median(
                row["useful_decode_rate_change_percent_candidate_vs_before"] for row in first_half
            ),
            "second_half_median_useful_decode_rate_change_percent": statistics.median(
                row["useful_decode_rate_change_percent_candidate_vs_before"] for row in second_half
            ),
        },
        "host_observations": {
            "policy": "descriptive only; no filtering or quiet gate",
            "external_cpu_max_percent_observed": max(cpu_samples) if cpu_samples else None,
            "coarse_thermal_max_observed": max(thermal_samples) if thermal_samples else None,
            "paging_delta_totals": paging,
            "monitor_failures": sum(item.get("sampling_failures", 0) for item in all_host),
            "monitor_sample_count": sum(item.get("samples", 0) for item in all_host),
            "all_request_screens": all_host,
        },
        "quality_check": {
            "checker_sha256": manifest["quality_checker_sha256"],
            "unique_measured_text_hashes": manifest["quality_unique_output_count"],
            "results_by_text_sha256": quality,
            "scope": "syntax parsing passed for both unique responses; isolated program replay was unavailable (sandbox/runtime failure flag); zero replays; no conclusion about playable behavior",
            "generated_output_execution_count": 0,
            "checker_profile_self_check_run": False,
        },
        "interpretation_limits": [
            "This is one natural short prompt with a long program answer, not a long-context prefill test.",
            "The candidate and before builds produced different output lengths because the MTP depths use different valid stochastic draw schedules. Complete-request latency therefore compares different amounts of generated output and is not a fixed-length causal speed estimate.",
            "Useful decode rates are reported separately from complete-request latency; the four paired observations are descriptive and do not qualify general coding workloads or quality.",
            "Both unique response strings passed syntax parsing, while isolated runtime replay was unavailable; playable game behavior is unverified.",
        ],
        "source_hashes": {
            "publisher_sha256": sha256_file(Path(__file__)),
            "generation_manifest_sha256": sha256_file(RESULTS / "manifest.json"),
            "session_json_sha256": {
                str(index): sha256_file(RESULTS / f"session-{index}.json")
                for index in range(1, 5)
            },
            "stdin_preflight_sha256": sha256_file(PREFLIGHT),
            "initial_harness_sha256": sha256_file(INITIAL_RUNNER),
            "generation_harness_sha256": sha256_file(RUNNER),
            "checker_sha256": sha256_file(CHECKER),
        },
        "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write(OUT, report)
    return 0


if __name__ == "__main__":
    try:
        status = run()
    except BaseException as error:
        print(json.dumps({"status": "failed", "failure_type": type(error).__name__}), flush=True)
        status = 2
    raise SystemExit(status)
