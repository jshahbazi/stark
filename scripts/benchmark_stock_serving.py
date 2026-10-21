"""Measure one stock vLLM Metal server at concurrency 1, 4 and 8.

This is a within-stack scaling experiment. It does not rank inference engines.
Payloads remain in private HTTP exchanges; reports contain scalars and hashes.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from benchmark_stock_engines import (
    Engine,
    HostMonitor,
    generation_fixtures,
    public_spec,
    save,
    sha,
    sleep_events,
    stop,
    token_sha,
)


CONCURRENCIES = (1, 4, 8)


def set_option(command, name, value):
    """Replace an ordinary CLI option without modifying engine code."""
    command = list(command)
    matches = [
        i
        for i, item in enumerate(command)
        if item == name or str(item).startswith(name + "=")
    ]
    if len(matches) > 1:
        raise ValueError("duplicate_server_option")
    if not matches:
        return command + [name, str(value)]
    index = matches[0]
    if command[index] == name:
        if index + 1 == len(command):
            raise ValueError("server_option_value_missing")
        command[index + 1] = str(value)
    else:
        command[index] = name + "=" + str(value)
    return command


def serving_config(generation_config):
    selected = [
        copy.deepcopy(spec)
        for spec in generation_config["engines"]
        if spec["kind"] == "vllm"
    ]
    if len(selected) != 1:
        raise ValueError("one_vllm_engine_required")
    spec = selected[0]
    command = set_option(spec["command"], "--max-num-seqs", 8)
    command = set_option(command, "--gpu-memory-utilization", 0.1)
    if any(str(arg).split("=", 1)[0] == "--enable-prefix-caching" for arg in command):
        raise ValueError("prefix_cache_must_be_disabled")
    if "--no-enable-prefix-caching" not in command:
        command.append("--no-enable-prefix-caching")
    spec["command"] = command
    return {
        "scope": "vllm_metal_within_stack_scaling",
        "source": generation_config["source"],
        "task_root": generation_config["task_root"],
        "context": generation_config["context"],
        "blocks": 3,
        "rounds": 5,
        "concurrencies": list(CONCURRENCIES),
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.1,
        "fixture": "fibonacci-module",
        "output_limit": 128,
        "engine": spec,
    }


def option_value(command, name):
    matches = [
        index
        for index, item in enumerate(command)
        if item == name or str(item).startswith(name + "=")
    ]
    if len(matches) != 1:
        raise ValueError("server_option_missing_or_duplicated")
    index = matches[0]
    if command[index] != name:
        return command[index].split("=", 1)[1]
    if index + 1 == len(command) or str(command[index + 1]).startswith("--"):
        raise ValueError("server_option_value_missing")
    return str(command[index + 1])


def validate_config(config):
    if (
        config.get("scope") != "vllm_metal_within_stack_scaling"
        or config["engine"]["kind"] != "vllm"
    ):
        raise ValueError("serving_scope_invalid")
    if (
        config["concurrencies"] != list(CONCURRENCIES)
        or config["max_num_seqs"] != 8
        or config["fixture"] != "fibonacci-module"
        or config["output_limit"] != 128
        or any(
            type(config[key]) is not int or config[key] < 1
            for key in ("blocks", "rounds", "context")
        )
    ):
        raise ValueError("serving_protocol_invalid")
    budget = config["gpu_memory_utilization"]
    if type(budget) not in (int, float) or budget not in (0.1, 0.2):
        raise ValueError("serving_memory_budget_invalid")
    command = config["engine"]["command"]
    if (
        int(option_value(command, "--max-num-seqs")) != 8
        or float(option_value(command, "--gpu-memory-utilization")) != budget
    ):
        raise ValueError("server_cli_configuration_mismatch")
    context = option_value(command, "--max-model-len")
    if context != "{context}" and int(context) != config["context"]:
        raise ValueError("server_context_configuration_mismatch")
    required_values = {
        "--host": "127.0.0.1",
        "--port": "{port}",
        "--served-model-name": "stock-qwen35",
        "--generation-config": "vllm",
    }
    if any(
        option_value(command, name) != value for name, value in required_values.items()
    ):
        raise ValueError("server_cli_configuration_mismatch")
    for name in ("--no-enable-prefix-caching", "--language-model-only"):
        if (
            sum(str(arg).split("=", 1)[0] == name for arg in command) != 1
            or name not in command
        ):
            raise ValueError("server_required_control_missing")
    forbidden = {
        "--enable-prefix-caching",
        "--enable-log-requests",
        "--disable-log-requests",
    }
    if any(str(arg).split("=", 1)[0] in forbidden for arg in command):
        raise ValueError("server_forbidden_control")


def safe_error(exc):
    message = str(exc)
    if isinstance(exc, RuntimeError) and message.replace("_", "").isalnum():
        return message
    return "serving_operation_failed"


def wave(engine, fixture, context, concurrency, pool):
    """Release ready client threads together; time through complete reduction."""
    gate = threading.Barrier(concurrency + 1)

    def request():
        gate.wait(timeout=30)
        return engine.generate(fixture, context)

    futures = [pool.submit(request) for _ in range(concurrency)]
    started = time.perf_counter()
    gate.wait(timeout=30)
    rows, failures = [], []
    for index, future in enumerate(futures):
        try:
            rows.append({"request_index": index, **future.result(timeout=310)})
        except Exception as exc:
            failures.append({"request_index": index, "error": safe_error(exc)})
    elapsed = time.perf_counter() - started
    limit = fixture["limit"]
    fixed_work = (
        not failures
        and len(rows) == concurrency
        and all(
            row["finish_reason"] == "length"
            and row["output_tokens"] == limit
            and row.get("returned_token_id_count") == limit
            for row in rows
        )
    )
    total = sum(row["output_tokens"] for row in rows)
    return {
        "requests": concurrency,
        "completed_requests": len(rows),
        "failed_requests": failures,
        "request_rows": rows,
        "wall_seconds": elapsed,
        "completed_output_tokens": total,
        "fixed_work_eligible": fixed_work,
        "aggregate_output_tokens_per_second": total / elapsed if fixed_work else None,
        "cache_counter_known_requests": sum(
            row.get("cached_tokens") is not None for row in rows
        ),
        "cache_counter_unknown_requests": sum(
            row.get("cached_tokens") is None for row in rows
        ),
        "unique_output_token_hashes": len(
            {row["token_sha256"] for row in rows if row.get("token_sha256")}
        ),
    }


def summaries(blocks):
    result = {}
    for concurrency in CONCURRENCIES:
        waves = [
            item
            for block in blocks
            if block["status"] == "complete"
            for item in block["waves"]
            if item["concurrency"] == concurrency and item["fixed_work_eligible"]
        ]
        if not waves:
            continue
        rates = [item["aggregate_output_tokens_per_second"] for item in waves]
        latencies = [
            row["wall_seconds"] for item in waves for row in item["request_rows"]
        ]
        result[str(concurrency)] = {
            "accepted_load_blocks": len(
                {
                    block["block"]
                    for block in blocks
                    if block["status"] == "complete"
                    and any(
                        item["concurrency"] == concurrency
                        and item["fixed_work_eligible"]
                        for item in block["waves"]
                    )
                }
            ),
            "waves": len(waves),
            "requests": len(latencies),
            "median_aggregate_output_tokens_per_second": statistics.median(rates),
            "minimum_aggregate_output_tokens_per_second": min(rates),
            "maximum_aggregate_output_tokens_per_second": max(rates),
            "median_request_wall_seconds": statistics.median(latencies),
            "minimum_request_wall_seconds": min(latencies),
            "maximum_request_wall_seconds": max(latencies),
            "cache_counter_known_requests": sum(
                item["cache_counter_known_requests"] for item in waves
            ),
            "cache_counter_unknown_requests": sum(
                item["cache_counter_unknown_requests"] for item in waves
            ),
            "unique_output_token_hashes": len(
                {
                    row["token_sha256"]
                    for item in waves
                    for row in item["request_rows"]
                    if row.get("token_sha256")
                }
            ),
        }
    return result


def run(config, output):
    validate_config(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["source"], local_files_only=True)
    fixture = next(
        item
        for item in generation_fixtures(tokenizer)
        if item["name"] == "fibonacci-module"
    )
    if len(fixture["ids"]) + fixture["limit"] > config["context"]:
        raise ValueError("serving_context_too_small")
    spec = config["engine"]
    report = {
        "schema": 1,
        "phase": "serving_scaling",
        "status": "running",
        "scope": config["scope"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "shared_harness_sha256": sha(
            Path(__file__).with_name("benchmark_stock_engines.py")
        ),
        "configuration_sha256": hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "model": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "model_revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3",
        "engine": public_spec(spec),
        "protocol": {
            "blocks": config["blocks"],
            "timed_waves_per_concurrency_per_block": config["rounds"],
            "concurrencies": list(CONCURRENCIES),
            "max_num_seqs": 8,
            "gpu_memory_utilization": config["gpu_memory_utilization"],
            "context_tokens": config["context"],
            "prefix_cache_requested": False,
            "cache_counter_policy": "missing remains unknown; positive hits fail the request",
            "warmup": "one unmeasured full wave at each concurrency on every fresh process",
            "sampling": "greedy, seed 0",
            "output_limit": 128,
            "fixture": fixture["name"],
            "input_tokens": len(fixture["ids"]),
            "input_token_sha256": token_sha(fixture["ids"]),
            "timing": "wave dispatch to completed private response reduction; per-request client complete latency",
            "order": "concurrency order rotates once per independent process load",
            "claims": "within-vLLM-Metal scaling only; no cross-engine ranking, quality or tail-latency claim",
        },
        "host_screen_coverage": {
            "cpu_process_samples": True,
            "gpu_activity": False,
            "thermal_or_gpu_clock_stability": False,
            "python_named_background_model_detection": False,
        },
        "blocks": [],
        "failures": [],
    }
    save(output, report)
    wall_start, mono_start = time.time(), time.monotonic()
    with contextlib.ExitStack() as outer:
        assertion = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        outer.callback(stop, assertion)
        for block in range(config["blocks"]):
            shift = block % len(CONCURRENCIES)
            order = CONCURRENCIES[shift:] + CONCURRENCIES[:shift]
            row = {
                "block": block,
                "status": "running",
                "concurrency_order": list(order),
                "warmups": [],
                "waves": [],
            }
            report["blocks"].append(row)
            try:
                with contextlib.ExitStack() as stack, HostMonitor() as monitor:
                    engine = Engine(spec, stack, config["task_root"], config["context"])
                    row["load_seconds"] = engine.load_seconds
                    row["runtime"] = engine.metadata
                    for concurrency in order:
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            warmup = wave(
                                engine, fixture, config["context"], concurrency, pool
                            )
                            row["warmups"].append(
                                {
                                    "concurrency": concurrency,
                                    "completed_requests": warmup["completed_requests"],
                                    "fixed_work_eligible": warmup[
                                        "fixed_work_eligible"
                                    ],
                                    "failed_requests": warmup["failed_requests"],
                                }
                            )
                            if not warmup["fixed_work_eligible"]:
                                raise RuntimeError("serving_warmup_invalid")
                            for round_index in range(config["rounds"]):
                                item = {
                                    "concurrency": concurrency,
                                    "round": round_index,
                                    **wave(
                                        engine,
                                        fixture,
                                        config["context"],
                                        concurrency,
                                        pool,
                                    ),
                                }
                                row["waves"].append(item)
                                save(output, report)
                    row["status"] = (
                        "complete"
                        if all(item["fixed_work_eligible"] for item in row["waves"])
                        else "invalid_fixed_work"
                    )
                row["host_screen"] = monitor.result()
                if row["status"] == "complete" and (
                    row["host_screen"]["cpu_guard_exceeded"]
                    or row["host_screen"]["background_model_active"]
                    or row["host_screen"]["sampling_failures"]
                    or not row["host_screen"]["samples"]
                ):
                    row["status"] = "excluded_interference"
            except Exception as exc:
                row.update(status="failed", error=safe_error(exc))
                report["failures"].append({"block": block, "error": row["error"]})
            save(output, report)
            print(
                json.dumps(
                    {
                        "block": block,
                        "status": row["status"],
                        "completed_waves": len(row["waves"]),
                    }
                ),
                flush=True,
            )
    wall_end = time.time()
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    report["wall_elapsed_seconds"] = wall_end - wall_start
    report["monotonic_elapsed_seconds"] = time.monotonic() - mono_start
    report["power_log_screen"] = sleep_events(wall_start, wall_end)
    report["timing_accepted"] = (
        report["power_log_screen"]["checked"]
        and report["power_log_screen"]["sleep_events"] == 0
        and abs(report["wall_elapsed_seconds"] - report["monotonic_elapsed_seconds"])
        < 2
        and len(report["blocks"]) == config["blocks"]
        and all(row["status"] == "complete" for row in report["blocks"])
    )
    report["summary"] = summaries(report["blocks"])
    report["status"] = (
        "complete" if report["timing_accepted"] else "complete_with_exclusions"
    )
    save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--generation-config", type=Path)
    sources.add_argument("--config", type=Path)
    parser.add_argument("--write-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = (
            serving_config(json.loads(args.generation_config.read_text()))
            if args.generation_config
            else json.loads(args.config.read_text())
        )
        validate_config(config)
        if args.write_config:
            save(args.write_config, config)
        if args.output:
            run(config, args.output)
        elif not args.write_config:
            parser.error("--output or --write-config is required")
    except Exception:
        raise SystemExit("stock_serving_benchmark_failed") from None


if __name__ == "__main__":
    main()
