"""Paired native prefill timing and exact generation checks on synthetic inputs.

Both workers stay loaded; every request still starts fresh native state. Reports
contain only counts, booleans, timings and hashes. Model text/token vectors and
rendered conversations remain in RAM. Use the locked converter environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

from benchmark_hybrid_context import model_provenance, sha256_file
from stark_convert.api_tools import template_messages
from stark_convert.chat import ChatCancelled, Worker, load_template

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("baseline", "candidate", "candidate", "baseline")
MODEL_BINDING_KEYS = (
    "configuration_sha256",
    "tokenizer_sha256",
    "weights_sha256",
    "model_provenance",
)
HOST_FAILURE_CATEGORIES = frozenset(
    {
        "gpu_monitor_unavailable",
        "paging_monitor_unavailable",
        "host_monitor_failed",
        "quiet_window_timeout",
    }
)
BENCHMARK_DEPENDENCIES = (
    "benchmark_clean_qwen",
    "benchmark_stock_engines",
    "benchmark_hybrid_context",
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def source_seal(root=ROOT):
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "Cargo.lock",
            "rust-toolchain.toml",
            "crates",
            "vendor",
        ],
        cwd=root,
    )
    digest = hashlib.sha256()
    for name in sorted(filter(None, paths.decode().split("\0"))):
        path = root / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def benchmark_dependency_bindings():
    sources = {}
    for name in BENCHMARK_DEPENDENCIES:
        origin = importlib.util.find_spec(name).origin
        if not isinstance(origin, str):
            raise TypeError("benchmark_dependency_source_missing")
        sources[name] = sha256_file(Path(origin))
    return sources


def input_bindings(model):
    sources = {}
    for name in ("api_tools", "api_request", "chat", "desktop"):
        origin = importlib.util.find_spec("stark_convert." + name).origin
        if not isinstance(origin, str):
            raise TypeError("rendering_source_missing")
        sources[name] = sha256_file(Path(origin))
    return {
        "configuration_sha256": sha256_file(model / "config.json"),
        "tokenizer_sha256": sha256_file(model / "tokenizer.json"),
        "weights_sha256": sha256_file(model / "model.safetensors"),
        "model_provenance": model_provenance(model),
        "harness_sha256": sha256_file(Path(__file__)),
        "rendering_sources_sha256": sources,
        "benchmark_dependencies_sha256": benchmark_dependency_bindings(),
    }


def fixture_payload(repeats=0, *, with_tools=True):
    """All descriptions/results/history are intentionally retained in rendering."""
    tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": f"synthetic_lookup_{index}",
                    "description": "Synthetic lookup. "
                    + ("Reference context. " * repeats if index == 0 else ""),
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
            for index in range(8)
        ]
        if with_tools
        else None
    )
    messages = [
        {
            "role": "system",
            "content": "Use the complete supplied history and definitions. Do not execute tools for this synthetic reply.",
        }
    ]
    if with_tools:
        messages += [
            {"role": "user", "content": "Look up the synthetic reference."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_synthetic",
                        "type": "function",
                        "function": {
                            "name": "synthetic_lookup_0",
                            "arguments": '{"key":"reference"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_synthetic",
                "content": "Synthetic reference result: preserve all history.",
            },
        ]
    messages += [
        {
            "role": "user",
            "content": "Write a numbered list of twenty synthetic sample items. Do not call tools.",
        }
    ]
    return messages, tools


def render_fixture(codec, template, repeats=0, *, with_tools=True):
    messages, tools = fixture_payload(repeats, with_tools=with_tools)
    text = template.render(
        messages=template_messages(messages),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token=codec.get("bos_token") or "",
        eos_token=codec.get("eos_token") or "",
    )
    return text, {
        "messages": len(messages),
        "tools": len(tools or []),
        "tool_results": sum(m["role"] == "tool" for m in messages),
    }


def tokenize(worker, text):
    reply = worker.request(
        {"operation": "tokenize", "text": text, "add_special_tokens": False}
    )
    ids = reply.get("token_ids")
    if (
        "error" in reply
        or not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or i < 0 for i in ids)
    ):
        raise ValueError("tokenization_failed")
    return ids


def prepare_fixtures(workers, model, target_tokens, context, output_tokens):
    codec, template = load_template(model)
    short, short_metadata = render_fixture(codec, template, with_tools=False)
    low, high = 0, target_tokens
    # CPU tokenization only; never truncate a message, definition or result.
    for _ in range(20):
        if low >= high:
            break
        middle = (low + high + 1) // 2
        rendered, _ = render_fixture(codec, template, middle)
        if len(tokenize(workers["baseline"], rendered)) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    long, long_metadata = render_fixture(codec, template, low)
    fixtures = []
    for name, text, metadata in [
        ("short_chat", short, short_metadata),
        ("long_tools_history", long, long_metadata),
    ]:
        ids = tokenize(workers["baseline"], text)
        if tokenize(workers["candidate"], text) != ids:
            raise ValueError("tokenization_mismatch")
        if len(ids) + output_tokens > context or (
            name == "long_tools_history"
            and not target_tokens - 16 <= len(ids) <= target_tokens
        ):
            raise ValueError("fixture_context_invalid")
        fixtures.append(
            {
                "name": name,
                "ids": ids,
                "metadata": {
                    **metadata,
                    "input_tokens": len(ids),
                    "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
                },
            }
        )
    return fixtures


def request_for(fixture, context, maximum, sampling=None, seed=0, **extra):
    return {
        "operation": "generate",
        "input_token_ids": fixture["ids"],
        "context_tokens": context,
        "max_new_tokens": maximum,
        "sampling": sampling or {"mode": "greedy"},
        "seed": seed,
        "stream": True,
        **extra,
    }


def checked_reply(reply, input_count, maximum):
    ids = reply.get("generated_token_ids")
    if (
        "error" in reply
        or not isinstance(ids, list)
        or any(type(i) is not int or i < 0 for i in ids)
        or not isinstance(reply.get("text"), str)
        or reply.get("input_tokens") != input_count
        or type(reply.get("sampled_tokens")) is not int
        or not 0 <= reply["sampled_tokens"] <= maximum
        or reply.get("output_tokens", len(ids)) != len(ids)
        or reply["sampled_tokens"]
        != len(ids) + int(reply.get("stop_token_id") is not None)
        or reply.get("finish_reason")
        not in {"eos", "stop_token", "stop_sequence", "length", "context"}
    ):
        raise ValueError("generation_invalid")
    return (
        ids,
        reply["text"],
        reply["finish_reason"],
        reply.get("stop_token_id"),
        reply["sampled_tokens"],
    )


def signature_comparison(reference, actual):
    """Aggregate exact differences; mismatch indices are zero-based, never IDs/text."""

    def first_mismatch(left, right):
        for index, (expected, got) in enumerate(zip(left, right)):
            if expected != got:
                return index
        return min(len(left), len(right)) if len(left) != len(right) else None

    return {
        "same_ids": reference[0] == actual[0],
        "same_text": reference[1] == actual[1],
        "same_finish_reason": reference[2] == actual[2],
        "same_stop_token_id": reference[3] == actual[3],
        "same_usage": (len(reference[0]), reference[4]) == (len(actual[0]), actual[4]),
        "first_token_mismatch_index": first_mismatch(reference[0], actual[0]),
        "first_text_mismatch_index": first_mismatch(reference[1], actual[1]),
        "reference_output_tokens": len(reference[0]),
        "actual_output_tokens": len(actual[0]),
        "reference_sampled_tokens": reference[4],
        "actual_sampled_tokens": actual[4],
        "reference_finish_reason": reference[2],
        "actual_finish_reason": actual[2],
    }


def generate(worker, request):
    first = None
    started = time.monotonic()

    def delta(text):
        nonlocal first
        if text and first is None:
            first = time.monotonic() - started

    reply = worker.request(request, on_delta=delta)
    signature = checked_reply(
        reply, len(request["input_token_ids"]), request["max_new_tokens"]
    )
    row = {
        "wall_seconds": time.monotonic() - started,
        "first_text_seconds": first,
        "output_tokens": len(signature[0]),
        "sampled_tokens": signature[4],
        "finish_reason": signature[2],
    }
    for key in (
        "prefill_seconds",
        "decode_seconds",
        "generation_seconds",
        "input_tokens_per_second",
        "output_tokens_per_second",
    ):
        if finite(reply.get(key)):
            row[key] = reply[key]
    return signature, row


def controls(workers, fixture, short_fixture, context, maximum):
    results = {}
    reference, _ = generate(
        workers["baseline"], request_for(short_fixture, context, maximum)
    )
    for label, worker in workers.items():
        overflow = worker.request(
            request_for(fixture, len(fixture["ids"]), 1), on_delta=lambda _text: None
        )
        overflow_ok = overflow.get("error") == "context_overflow"
        event = threading.Event()
        timer = threading.Timer(0.05, event.set)
        timer.start()
        started = time.monotonic()
        try:
            worker.request(
                request_for(fixture, context, maximum),
                on_delta=lambda _text: None,
                cancelled=event,
            )
            cancelled = False
        except ChatCancelled:
            cancelled = True
        finally:
            timer.cancel()
        recovery, _ = generate(worker, request_for(short_fixture, context, maximum))
        results[label] = {
            "context_overflow_recoverable": overflow_ok,
            "cancelled_before_terminal": cancelled,
            "cancel_and_recovery_seconds": time.monotonic() - started,
            "recovery_exact": recovery == reference,
            "recovery_comparison": signature_comparison(reference, recovery),
        }
    return results


def summarize(rows):
    summaries = {}
    for name in sorted({row["fixture"] for row in rows}):
        selected = [row for row in rows if row["fixture"] == name]
        summary = {}
        for label in ORDER[:2]:
            samples = [
                row.get("prefill_seconds") for row in selected if row["worker"] == label
            ]
            if samples and all(finite(value) and value > 0 for value in samples):
                summary[label + "_median_prefill_seconds"] = statistics.median(samples)
        if len(summary) == 2:
            summary["prefill_speedup"] = (
                summary["baseline_median_prefill_seconds"]
                / summary["candidate_median_prefill_seconds"]
            )
        summaries[name] = summary
    return summaries


def run(args, *, worker_factory=Worker, fixture_factory=prepare_fixtures):
    if (
        args.target_tokens < 64
        or args.context < args.target_tokens + args.tokens
        or not 1 <= args.tokens <= 256
        or not 1 <= args.cycles <= 5
    ):
        raise ValueError("benchmark_limits_invalid")
    start_seal = source_seal()
    start_wall, start_monotonic = time.time(), time.monotonic()
    bindings = input_bindings(args.model)
    worker_hashes = {
        "baseline": sha256_file(args.baseline),
        "candidate": sha256_file(args.candidate),
    }
    report = {
        "schema_version": 1,
        "scope": "native_prefill_fixed_quantization_synthetic_full_history",
        "source_tree_sha256": start_seal,
        **bindings,
        "worker_sha256": worker_hashes,
        "context_tokens": args.context,
        "timing_rows": [],
        "warmup_exact": [],
        "parity": [],
        "numeric_logits_compared": False,
        "quiet_windows_required": not args.no_quiet,
    }
    monitor = None
    with contextlib.ExitStack() as stack:
        if not args.no_quiet:
            from benchmark_clean_qwen import Monitor, paging, quiet_window

            monitor = stack.enter_context(Monitor())
            quiet_window(monitor, seconds=10)
            before_paging = paging()
        workers = {}
        for label in ORDER[:2]:
            worker = worker_factory(
                getattr(args, label).resolve(), args.model.resolve(), 120
            )
            stack.callback(worker.close)
            ready = worker.read()
            if (
                ready.get("ready") is not True
                or ready.get("mode") != "text-generation-v1"
                or ready.get("streaming") != "text-delta-v1"
                or ready.get("max_context_tokens", 0) < args.context
            ):
                raise ValueError("worker_capability_invalid")
            workers[label] = worker
        fixtures = fixture_factory(
            workers, args.model, args.target_tokens, args.context, args.tokens
        )
        report["fixtures"] = [{"name": f["name"], **f["metadata"]} for f in fixtures]
        if args.mode in {"timing", "all"}:
            references = {}
            if monitor:
                monitor.set_phase("warmup")
            for fixture in fixtures:
                request = request_for(fixture, args.context, 1)
                for label, worker in workers.items():
                    signature, _ = generate(worker, request)
                    if label == "baseline":
                        references[fixture["name"]] = signature
                    else:
                        report["warmup_exact"].append(
                            signature == references[fixture["name"]]
                        )
            if monitor:
                quiet_window(monitor, seconds=10)
                monitor.set_phase("measurement")
            for cycle in range(args.cycles):
                for fixture in fixtures:
                    for label in ORDER:
                        signature, row = generate(
                            workers[label], request_for(fixture, args.context, 1)
                        )
                        row.update(
                            worker=label,
                            fixture=fixture["name"],
                            cycle=cycle,
                            exact=signature == references[fixture["name"]],
                            comparison=signature_comparison(
                                references[fixture["name"]], signature
                            ),
                        )
                        report["timing_rows"].append(row)
            if monitor:
                monitor.set_phase("parity")
        if args.mode in {"parity", "all"}:
            for fixture in fixtures:
                cases = [
                    ("greedy", None, 0),
                    (
                        "sampled_top_k",
                        {
                            "mode": "stochastic",
                            "temperature": 0.7,
                            "top_k": 40,
                            "top_p": 0.95,
                        },
                        17,
                    ),
                    (
                        "sampled_nucleus",
                        {"mode": "stochastic", "temperature": 0.9, "top_p": 0.9},
                        42,
                    ),
                ]
                for name, sampling, seed in cases:
                    request = request_for(
                        fixture, args.context, args.tokens, sampling, seed
                    )
                    baseline, _ = generate(workers["baseline"], request)
                    candidate, _ = generate(workers["candidate"], request)
                    report["parity"].append(
                        {
                            "fixture": fixture["name"],
                            "case": name,
                            "exact": baseline == candidate,
                            "comparison": signature_comparison(baseline, candidate),
                        }
                    )
                    if name == "greedy" and baseline[0] and baseline[1]:
                        for stop_name, extra in [
                            ("token_stop", {"stop_token_ids": [baseline[0][0]]}),
                            ("text_stop", {"stop": [baseline[1][:8]]}),
                        ]:
                            request = request_for(
                                fixture, args.context, args.tokens, **extra
                            )
                            a, _ = generate(workers["baseline"], request)
                            b, _ = generate(workers["candidate"], request)
                            expected = (
                                "stop_token"
                                if stop_name == "token_stop"
                                else "stop_sequence"
                            )
                            report["parity"].append(
                                {
                                    "fixture": fixture["name"],
                                    "case": stop_name,
                                    "exact": a == b,
                                    "stop_observed": a[2] == b[2] == expected,
                                    "comparison": signature_comparison(a, b),
                                }
                            )
            report["lifecycle"] = controls(
                workers, fixtures[-1], fixtures[0], args.context, args.tokens
            )
        if monitor:
            after_paging = paging()
            report["paging_delta"] = {
                key: after_paging[key] - value for key, value in before_paging.items()
            }
    if monitor:
        guard = monitor.result()
        report["host_guard"] = {
            key: value for key, value in guard.items() if key != "samples"
        }
        # Keep every boundary-inclusive guard calculation from Monitor.result;
        # only the positive sample-count gate requires an entire measured interval.
        report["host_guard"]["guard_interval_samples"] = guard["measurement_samples"]
        report["host_guard"]["measurement_samples"] = sum(
            row.get("phases") == ["measurement"] for row in guard.get("samples", [])
        )
    report["source_unchanged"] = start_seal == source_seal()
    report["workers_unchanged"] = all(
        sha256_file(getattr(args, label)) == value
        for label, value in worker_hashes.items()
    )
    final_bindings = input_bindings(args.model)
    report["model_unchanged"] = all(
        final_bindings[key] == bindings[key] for key in MODEL_BINDING_KEYS
    )
    report["harness_unchanged"] = (
        final_bindings["harness_sha256"] == bindings["harness_sha256"]
    )
    report["rendering_sources_unchanged"] = (
        final_bindings["rendering_sources_sha256"]
        == bindings["rendering_sources_sha256"]
    )
    report["benchmark_dependencies_unchanged"] = (
        final_bindings["benchmark_dependencies_sha256"]
        == bindings["benchmark_dependencies_sha256"]
    )
    report["summary"] = summarize(report["timing_rows"])
    report["passed"] = (
        report["source_unchanged"]
        and report["workers_unchanged"]
        and report["model_unchanged"]
        and report["harness_unchanged"]
        and report["rendering_sources_unchanged"]
        and report["benchmark_dependencies_unchanged"]
        and all(report["warmup_exact"])
        and all(row["exact"] for row in report["timing_rows"])
        and all(
            row["exact"] and row.get("stop_observed", True) for row in report["parity"]
        )
    )
    report["passed"] &= all(
        all(
            value[key]
            for key in (
                "context_overflow_recoverable",
                "cancelled_before_terminal",
                "recovery_exact",
            )
        )
        for value in report.get("lifecycle", {}).values()
    )
    guard = report.get("host_guard", {})
    if not args.no_quiet:
        from benchmark_stock_engines import sleep_events

        report["sleep_screen"] = sleep_events(start_wall, time.time())
    # The sleep-screen helper runs after the other end bindings are captured.
    report["benchmark_dependencies_unchanged"] &= (
        benchmark_dependency_bindings() == bindings["benchmark_dependencies_sha256"]
    )
    report["passed"] &= report["benchmark_dependencies_unchanged"]
    report["wall_monotonic_discrepancy_seconds"] = abs(
        (time.time() - start_wall) - (time.monotonic() - start_monotonic)
    )
    report["timing_accepted"] = (
        bool(report["timing_rows"])
        and report["passed"]
        and not args.no_quiet
        and type(guard.get("measurement_samples")) is int
        and guard["measurement_samples"] > 0
        and all(
            finite(row.get("prefill_seconds")) and row["prefill_seconds"] > 0
            for row in report["timing_rows"]
        )
        and report.get("sleep_screen", {}).get("checked") is True
        and report["sleep_screen"]["sleep_events"] == 0
        and report["wall_monotonic_discrepancy_seconds"] < 2
        and not any(
            guard.get(key, True)
            for key in (
                "sampling_failures",
                "sampling_gap_exceeded",
                "cpu_guard_exceeded",
                "background_model_active",
                "thermal_guard_exceeded",
            )
        )
        and not any(report.get("paging_delta", {}).values())
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=18544)
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--mode", choices=("timing", "parity", "all"), default="all")
    parser.add_argument(
        "--no-quiet",
        action="store_true",
        help="Exploratory only; timing_accepted stays false.",
    )
    args = parser.parse_args(argv)
    try:
        args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            args.output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "w") as output:
            try:
                report = run(args)
            except Exception as error:  # noqa: BLE001 - never expose private exception payloads.
                # Host guards use a small public vocabulary. All other failures,
                # including worker/template failures, retain no exception text.
                category = (
                    str(error)
                    if isinstance(error, RuntimeError)
                    and str(error) in HOST_FAILURE_CATEGORIES
                    else "operation_failed"
                )
                report = {
                    "passed": False,
                    "timing_accepted": False,
                    "stage": "native_prefill_benchmark_failed",
                    "failure_category": category,
                }
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0 if report["passed"] else 1
    except (OSError, ValueError):
        print(
            json.dumps({"passed": False, "stage": "benchmark_output_unavailable"}),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
