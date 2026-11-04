"""Measure serialized decoding from an exact, corpus-derived prefix.

The report contains only hashes, token counts, finish reasons, and timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

from stark_convert.chat import ChatError, Worker


class BenchmarkFailure(ValueError):
    """A sanitized failure whose partial receipt has already been written."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def execution_profile(*, screened: bool) -> dict:
    """Bind observed selectors and helper source, without inferring dispatch."""
    import benchmark_hybrid
    from stark_convert import chat

    script_root = Path(__file__).resolve().parent
    helpers = {
        "benchmark_hybrid_context.py": Path(__file__),
        "benchmark_hybrid.py": Path(benchmark_hybrid.__file__),
        "stark_convert/chat.py": Path(chat.__file__),
    }
    if screened:
        helpers.update({
            "benchmark_clean_qwen.py": script_root / "benchmark_clean_qwen.py",
            "benchmark_glm_mlx_serve.py": script_root / "benchmark_glm_mlx_serve.py",
        })
    try:
        wrapper_hash = sha256_file(script_root / "with_glm53_optimized.sh")
        helper_hashes = {name: sha256_file(path) for name, path in helpers.items()}
    except OSError:
        raise ValueError("execution_profile_source_unavailable") from None
    return {
        "configuration_flags": benchmark_hybrid.configuration_flags(),
        "configuration_flags_semantics": "Known selector environment values, including inactive selectors; not dispatch proof.",
        "recipe_wrapper_source_sha256": wrapper_hash,
        "recipe_wrapper_binding": "Source hash only; invocation is not observed.",
        "dependency_source_sha256": helper_hashes,
    }


def model_provenance(model: Path) -> dict:
    """Bind the report to the receipt format shipped with this model artifact."""
    native_receipt = model / "native-source.json"
    if native_receipt.is_file():
        receipt_hash = sha256_file(native_receipt)
        return {
            "model_artifact_kind": "native_source",
            "model_artifact_receipt_file": native_receipt.name,
            "model_artifact_receipt_sha256": receipt_hash,
            # Keep this field unchanged for existing native-source benchmarks.
            "source_receipt_sha256": receipt_hash,
        }

    conversion_receipt = model / "conversion.json"
    if conversion_receipt.is_file():
        try:
            receipt = json.loads(conversion_receipt.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("conversion_receipt_invalid") from None
        source_manifest_hash = receipt.get("source_manifest_sha256") if isinstance(receipt, dict) else None
        if (
            not isinstance(source_manifest_hash, str)
            or len(source_manifest_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_manifest_hash)
        ):
            raise ValueError("conversion_source_manifest_hash_invalid")
        return {
            "model_artifact_kind": "stark_conversion",
            "model_artifact_receipt_file": conversion_receipt.name,
            "model_artifact_receipt_sha256": sha256_file(conversion_receipt),
            "source_manifest_sha256": source_manifest_hash,
        }

    raise FileNotFoundError("model_provenance_receipt_missing")


def token_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def exact_prefix(ids: list[int], count: int) -> list[int]:
    if not ids or count < 1:
        raise ValueError("corpus_tokenization_empty")
    repeats = (count + len(ids) - 1) // len(ids)
    return (ids * repeats)[:count]


def validate_args(args: argparse.Namespace) -> None:
    if args.input_tokens < 1 or args.tokens < 1 or args.replays not in (3, 5):
        raise ValueError("invalid_benchmark_limits")
    if args.input_tokens + args.tokens > 16384:
        raise ValueError("context_limit_exceeded")
    if getattr(args, "screened", False) and getattr(args, "max_attempts", 24) < args.replays:
        raise ValueError("invalid_screened_attempt_limit")
    if getattr(args, "sampling", "greedy") == "stochastic":
        temperature = getattr(args, "temperature", 1.0)
        top_k = getattr(args, "top_k", 20)
        top_p = getattr(args, "top_p", 0.95)
        seed = getattr(args, "seed", 42)
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or temperature > 100
            or top_k < 0
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
            or not 0 <= seed <= 2**64 - 1
        ):
            raise ValueError("invalid_sampling_parameters")


def sampling_options(args: argparse.Namespace) -> dict:
    if getattr(args, "sampling", "greedy") == "greedy":
        return {"mode": "greedy"}
    options = {
        "mode": "stochastic",
        "temperature": getattr(args, "temperature", 1.0),
        "top_p": getattr(args, "top_p", 0.95),
    }
    top_k = getattr(args, "top_k", 20)
    if top_k:
        options["top_k"] = top_k
    return options


def screen_monitor_diagnostics(monitor, *, now: float | None = None) -> dict:
    """Summarize existing observations without exporting host identities or rows."""
    observed = getattr(monitor, "rows", None)
    available = isinstance(observed, (list, tuple))
    rows = list(observed) if available else []

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def values(key):
        return [row[key] for row in rows if isinstance(row, dict) and finite(row.get(key))]

    intervals = values("interval_seconds")
    started = getattr(monitor, "started", None)
    latest_end = rows[-1].get("end_seconds") if rows and isinstance(rows[-1], dict) else None
    latest_age = None
    if finite(started) and finite(latest_end):
        latest_age = max(0.0, (time.monotonic() if now is None else now) - started - latest_end)
    result = {
        "data_available": available,
        "sample_count": len(rows) if available else None,
        "latest_sample_age_seconds": latest_age,
        "max_sample_interval_seconds": max(intervals, default=None),
        "thermal_state_max": max(values("thermal_state"), default=None),
        "compiler_active": any(
            isinstance(item, dict) and type(item.get("name")) is str
            and item["name"] in {"cargo", "rustc", "(rustc)", "clang", "clang++", "ld"}
            for row in rows if isinstance(row, dict)
            for item in (row.get("active_processes") or [])
        ) if available else None,
    }
    for label, field in (
        ("external_cpu_percent", "external_cpu_percent"),
        ("external_model_cpu_percent", "max_model_cpu_percent"),
        ("gpu_device_utilization_percent", "gpu_device_utilization_percent"),
    ):
        samples = values(field)
        result[f"{label}_min"] = min(samples, default=None)
        result[f"{label}_max"] = max(samples, default=None)
    return result


def run(
    args: argparse.Namespace,
    worker_factory=Worker,
    host_metrics_factory=None,
    *,
    monitor_factory=None,
    paging_reader=None,
    admission_fn=None,
    screen_fn=None,
) -> dict:
    validate_args(args)
    screened = getattr(args, "screened", False)
    corpus_bytes = args.corpus.read_bytes()
    try:
        corpus_text = corpus_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("corpus_not_utf8") from None
    if not corpus_text:
        raise ValueError("corpus_empty")

    report = {
        "schema_version": 3,
        "scope": "serialized_native_text_fixed_context",
        "sampling": sampling_options(args),
        "worker_sha256": sha256_file(args.worker),
        "configuration_sha256": sha256_file(args.model / "config.json"),
        **model_provenance(args.model),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "requested_input_tokens": args.input_tokens,
        "requested_output_tokens": args.tokens,
        "context_tokens": args.input_tokens + args.tokens,
        "replay_count": args.replays,
        "warmup_replays": 1,
        "screened": screened,
        "execution_profile": execution_profile(screened=screened),
        "excluded_attempts": [],
        "replays": [],
    }
    verified_weights = getattr(args, "verified_weights", None)
    weights_before = None
    if verified_weights is not None:
        from benchmark_hybrid import checkout_identity, hardware_identity, weight_identity

        weights_before = weight_identity(args.model, verified_weights)
        report.update(
            weights=weights_before,
            harness_sha256=sha256_file(Path(__file__)),
            harness_checkout=checkout_identity(),
            hardware=hardware_identity(),
            prefill_configuration_flags={name: os.environ.get(name) for name in (
                "STARK_GLM53_PREFILL_MMA", "STARK_GLM53_PREFILL_CHUNK",
                "STARK_GLM53_PREFILL_WIDE", "STARK_GLM53_CONTROL_PREFILL_BATCH",
                "STARK_GLM53_KDA_REGISTER_PREFILL",
                "STARK_GLM53_PREFILL_OVERLAP", "STARK_GLM53_PREFILL_QUEUED",
                "STARK_GLM53_KDA_PARALLEL_NORM",
                "STARK_GLM53_IQ_FUSED_GATE_UP", "STARK_GLM53_IQ_FUSED_GATE_UP_R2",
            )},
        )
    if report["sampling"]["mode"] == "stochastic":
        report["seed"] = getattr(args, "seed", 42)
    if screened:
        from benchmark_clean_qwen import Monitor, paging
        from benchmark_glm_mlx_serve import quiet_admission, screen_attempt

        monitor_factory = monitor_factory or Monitor
        paging_reader = paging_reader or paging
        admission_fn = admission_fn or quiet_admission
        screen_fn = screen_fn or screen_attempt
        report["max_attempts"] = getattr(args, "max_attempts", 24)
        report["screening_policy"] = {
            "quiet_admission": "benchmark_glm_mlx_serve.quiet_admission",
            "screen_attempt": "benchmark_glm_mlx_serve.screen_attempt",
            "monitor": "benchmark_clean_qwen.Monitor",
            "quiet_seconds_before_each_measured_attempt": 10,
            "monitor_source_sha256": sha256_file(Path(__file__).with_name("benchmark_clean_qwen.py")),
            "screen_source_sha256": sha256_file(Path(__file__).with_name("benchmark_glm_mlx_serve.py")),
        }

    worker = monitor = host_monitor = None
    monitor_entered = host_entered = False
    host_rows = []
    active_item = None
    failure = None
    stage = "worker_start"
    try:
        worker = worker_factory(args.worker.resolve(), args.model.resolve(), 3600)
        stage = "worker_ready"
        ready = worker.read()
        if ready.get("ready") is not True or ready.get("mode") != "text-generation-v1":
            raise ValueError("worker_not_ready")
        worker_context = ready.get("max_context_tokens")
        if type(worker_context) is int and args.input_tokens + args.tokens > worker_context:
            raise ValueError("worker_context_limit_exceeded")
        stage = "tokenization"
        tokenized = worker.request({"operation": "tokenize", "text": corpus_text, "add_special_tokens": False})
        tokenized_ids = tokenized.get("token_ids")
        if (
            "error" in tokenized
            or not isinstance(tokenized_ids, list)
            or any(type(token) is not int or token < 0 for token in tokenized_ids)
        ):
            raise ValueError("corpus_tokenization_failed")
        ids = exact_prefix(tokenized_ids, args.input_tokens)
        report["input_tokens"] = len(ids)
        report["corpus_token_count"] = len(tokenized_ids)
        report["input_sha256"] = token_ids_sha256(ids)

        if screened:
            stage = "screen_monitor_start"
            monitor = monitor_factory()
            monitor_entered = True
            monitor.__enter__()
        if getattr(args, "host_metrics", False):
            stage = "host_metrics_start"
            if host_metrics_factory is None:
                from hybrid_host_metrics import HostMetricsSampler

                host_metrics_factory = HostMetricsSampler
            host_monitor = host_metrics_factory(getattr(getattr(worker, "process", None), "pid", None))
            host_entered = True
            host_monitor.__enter__()

        attempts = getattr(args, "max_attempts", 24) if screened else args.replays
        for attempt in range(attempts):
            replay = len(report["replays"])
            active_item = {"replay": replay, "attempt": attempt, "warmup": replay == 0, "admission": None}
            report["attempt_count"] = attempt + 1
            if monitor is not None and replay > 0:
                stage = "quiet_admission"
                active_item["admission"] = admission_fn(monitor)
            phase = f"attempt_{attempt}"
            if monitor is not None:
                stage = "before_paging"
                monitor.set_phase(phase)
                before_paging = paging_reader()
            stage = "host_metrics_begin"
            marker = host_monitor.begin_request() if host_monitor else None
            if marker is not None:
                # Associate the marker with the row itself, not its accepted
                # replay index: rejected attempts also own complete metrics.
                host_rows.append((active_item, marker))
            stage = "generation"
            started = time.monotonic()
            try:
                result = worker.request({
                    "operation": "generate",
                    "input_token_ids": ids,
                    "context_tokens": args.input_tokens + args.tokens,
                    "max_new_tokens": args.tokens,
                    "sampling": report["sampling"],
                    **({"seed": report["seed"]} if "seed" in report else {}),
                    "request_id": attempt + 1,
                })
            finally:
                active_item["wall_seconds"] = time.monotonic() - started
                if host_monitor and marker is not None:
                    host_monitor.end_request(marker)
            stage = "response_validation"
            generated = result.get("generated_token_ids")
            sampled = result.get("sampled_tokens")
            finish = result.get("finish_reason")
            stop = result.get("stop_token_id")
            stopped = type(stop) is int and stop in ready.get("eos_token_ids", [])
            if (
                "error" in result
                or not isinstance(generated, list)
                or any(type(token) is not int or token < 0 for token in generated)
                or type(sampled) is not int
                or sampled != len(generated) + int(stopped)
                or sampled < 2
                or sampled > args.tokens
                or finish not in {"length", "context", "eos"}
                or (finish == "eos") != stopped
                or (not stopped and stop is not None)
            ):
                raise ValueError("generation_failed_or_too_short")
            active_item.update(
                sampled_tokens=sampled,
                output_tokens=len(generated),
                finish_reason=finish,
                output_sha256=token_ids_sha256(generated),
            )
            for key in (
                "prefill_seconds", "decode_seconds", "input_tokens_per_second",
                "output_tokens_per_second", "generation_seconds",
            ):
                value = result.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(value):
                        raise ValueError("generation_timing_invalid")
                    active_item[key] = value
            if monitor is not None:
                stage = "host_screen"
                active_item["host_screen"] = screen_fn(monitor, phase, before_paging, paging_reader())
                if not active_item["warmup"] and not active_item["host_screen"]["eligible"]:
                    active_item["exclusion_reason"] = "host_screen_ineligible"
                    report["excluded_attempts"].append(active_item)
                    active_item = None
                    print(json.dumps({"excluded_attempt": attempt}), flush=True)
                    continue
            report["replays"].append(active_item)
            active_item = None
            print(json.dumps({"completed_replay": replay, "sampled_tokens": sampled, "finish_reason": finish}), flush=True)
            if len(report["replays"]) == args.replays:
                break

        stage = "aggregation"
        if len(report["replays"]) != args.replays:
            raise ValueError("insufficient_screened_requests")
        if verified_weights is not None:
            stage = "weight_identity_guard"
            if weight_identity(args.model, verified_weights) != weights_before:
                raise ValueError("verified_weight_changed_during_execution")
        stage = "execution_profile_guard"
        profile_after = execution_profile(screened=screened)
        report["execution_profile_after"] = profile_after
        report["execution_profile_unchanged"] = profile_after == report["execution_profile"]
        if profile_after != report["execution_profile"]:
            raise ValueError("execution_profile_changed_during_execution")
    except (Exception, KeyboardInterrupt) as error:
        failure = {
            "error": "interrupted" if isinstance(error, KeyboardInterrupt) else "native_context_benchmark_failed",
            "failed_stage": stage,
            "error_type": type(error).__name__,
        }
        if active_item is not None:
            report["failed_attempt"] = active_item
    finally:
        # Snapshot before stopping observers: a final drain sample must not
        # hide stale observations at the point admission succeeded or failed.
        if monitor is not None:
            report["screening_monitor"] = {
                "diagnostic_snapshot_stage": "before_observer_cleanup",
                **screen_monitor_diagnostics(monitor),
            }
        # Drain observers while the worker still exists. Every cleanup runs
        # even when admission, generation or another observer has failed.
        cleanup = []
        if host_monitor is not None and host_entered:
            cleanup.append(("host_metrics_stop", lambda: host_monitor.__exit__(None, None, None)))
        if monitor is not None and monitor_entered:
            cleanup.append(("screen_monitor_stop", lambda: monitor.__exit__(None, None, None)))
        if worker is not None:
            cleanup.append(("worker_close", worker.close))
        for cleanup_stage, close in cleanup:
            try:
                close()
            except Exception as error:
                entry = {"stage": cleanup_stage, "error_type": type(error).__name__}
                report.setdefault("cleanup_failures", []).append(entry)
                if failure is None:
                    failure = {"error": "native_context_benchmark_failed", "failed_stage": cleanup_stage,
                               "error_type": type(error).__name__}
        if host_monitor is not None:
            try:
                for item, marker in host_rows:
                    item["host_metrics"] = host_monitor.request_result(marker)
                report["host_metrics_monitor"] = host_monitor.summary()
            except Exception as error:
                if failure is None:
                    failure = {"error": "native_context_benchmark_failed", "failed_stage": "host_metrics_result",
                               "error_type": type(error).__name__}
        if monitor is not None:
            report["screening_monitor"]["sampling_failures"] = monitor.failures
            if monitor.failures and failure is None:
                failure = {"error": "native_context_benchmark_failed", "failed_stage": "screen_monitor_stop",
                           "error_type": "MonitorFailure"}

    rows = report["replays"]
    warm = rows[1:]
    completed = sorted(rows + report["excluded_attempts"], key=lambda row: row["attempt"])
    if "output_sha256" in report.get("failed_attempt", {}):
        completed.append(report["failed_attempt"])
    report["repeat_exact"] = len({item["output_sha256"] for item in rows}) == 1
    report["all_attempts_repeat_exact"] = len({item["output_sha256"] for item in completed}) == 1
    report["consistent_sample_counts"] = len({item["sampled_tokens"] for item in rows}) == 1
    if rows:
        report["median_wall_seconds"] = statistics.median(item["wall_seconds"] for item in rows)
        report["median_generated_tokens"] = statistics.median(item["sampled_tokens"] for item in rows)
        rates = [item["output_tokens_per_second"] for item in rows if "output_tokens_per_second" in item]
        if len(rates) == len(rows):
            report["median_decode_tokens_per_second"] = statistics.median(rates)
    if warm:
        report["median_warm_wall_seconds"] = statistics.median(item["wall_seconds"] for item in warm)
        for key, field in (
            ("output_tokens_per_second", "median_warm_decode_tokens_per_second"),
            ("input_tokens_per_second", "median_warm_prefill_tokens_per_second"),
        ):
            rates = [item[key] for item in warm if key in item]
            if len(rates) == len(warm):
                report[field] = statistics.median(rates)
    report["passed"] = failure is None and report["repeat_exact"] and report["consistent_sample_counts"]
    if failure is not None:
        report.update(failure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failure is not None:
        raise BenchmarkFailure("native_context_benchmark_failed") from None
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--verified-weights", type=Path, help="Prior GGUF hash receipt; check shard identity before and after.")
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--replays", type=int, choices=(3, 5), default=5)
    parser.add_argument("--screened", action="store_true", help="Use the existing quiet-admission and paging policy for every measured attempt.")
    parser.add_argument("--max-attempts", type=int, default=24, help="Bound screened attempts, including the accepted warmup.")
    parser.add_argument("--sampling", choices=("greedy", "stochastic"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20, help="Zero disables top-k filtering.")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--host-metrics",
        action="store_true",
        help="Attach opt-in aggregate CPU, GPU, memory, and thermal measurements.",
    )
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, ChatError):
        print(json.dumps({"passed": False, "stage": "native_benchmark_failed"}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
