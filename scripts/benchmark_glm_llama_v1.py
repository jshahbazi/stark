"""Fixed GLM execution-profile experiments; persist aggregates and hashes only.

Readouts use the existing anonymous-pipe comparator. Timed sessions retain
every request under current host load, with no quiet admission or retries.
"""

# Unexpected failures must also produce sanitized receipts, never exception dumps.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import compare_glm_prefill_readouts as readouts
from benchmark_hybrid import checkout_identity, hardware_identity, weight_identity
from benchmark_hybrid_context import exact_prefix, token_ids_sha256
from stark_convert.chat import Worker

COMPARISONS = {
    "whole_profile": ("defaults", "profile"),
    "mma": ("mma_off", "profile"),
    "mla": ("profile", "mla"),
    "kda": ("profile", "kda"),
    "combination": ("profile", "combination"),
}


class ExperimentFailure(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("STARK_", "UZU_", "LILY_", "GGML_"))}


def environments():
    base = clean_env()
    wrapper = Path(__file__).with_name("with_glm53_optimized.sh")
    code = "import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('STARK_','UZU_'))}))"
    selected = json.loads(subprocess.check_output([str(wrapper), sys.executable, "-c", code], env=base, text=True))
    profile = {**base, **selected}
    return {
        "defaults": base,
        "profile": profile,
        "mma_off": {**profile, "STARK_GLM53_PREFILL_MMA": "0"},
        "mla": {**profile, "STARK_GLM53_MLA_PREFILL_BATCH": "1"},
        "kda": {**profile, "STARK_GLM53_KDA_REGISTER_PREFILL": "1"},
        "combination": {**profile, "STARK_GLM53_MLA_PREFILL_BATCH": "1", "STARK_GLM53_KDA_REGISTER_PREFILL": "1"},
    }


def selectors(env):
    return {k: v for k, v in env.items() if k.startswith(("STARK_", "UZU_"))}


@contextlib.contextmanager
def selected_environment(env):
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def require(ok, code):
    if not ok:
        raise ExperimentFailure(code)


def failure(error):
    return {"failure_type": type(error).__name__,
            **({"failure_code": error.code} if isinstance(error, ExperimentFailure) else {})}


def identities(args):
    root = Path(__file__).resolve().parents[1]
    sources = ["scripts/compare_glm_prefill_readouts.py", "scripts/benchmark_hybrid.py",
        "scripts/benchmark_hybrid_context.py", "scripts/hybrid_host_metrics.py",
        "scripts/with_glm53_optimized.sh", "tools/converter/stark_convert/chat.py",
        "crates/stark-engine/examples/hybrid_prefill_readouts.rs",
        "crates/stark-engine/src/engine/hybrid/glm.rs",
        "crates/stark-engine/src/engine/hybrid/glm_metal.rs", "Cargo.lock", "tools/converter/uv.lock"]
    return {
        "worker_sha256": readouts.sha(args.worker),
        "example_sha256": readouts.sha(args.example),
        "corpus_sha256": readouts.sha(args.corpus),
        "configuration_sha256": readouts.sha(args.model / "config.json"),
        "model_receipt_sha256": readouts.sha(args.model / "native-source.json"),
        "tokenizer_sha256": readouts.sha(args.model / "tokenizer.json"),
        "runner_sha256": readouts.sha(Path(__file__)),
        "source_sha256": {name: readouts.sha(root / name) for name in sources},
    }


def route_admission(rows, config):
    records = []
    for meta, _, _ in rows:
        dispatch = meta.get("dispatch", {})
        count = lambda key, dispatch=dispatch: dispatch.get(key, {}).get("calls", 0)
        cpu, gpu = count("native.glm.route.cpu.rows"), count("native.glm.route.gpu.rows")
        require((cpu > 0 and gpu == 0) if config == "defaults" else (gpu > 0 and cpu == 0), "route_not_observed")
        mma = count("native.glm.prefill.q5_mma") + count("native.glm.prefill.q6_mma")
        require(mma > 0 if config not in {"defaults", "mma_off"} else mma == 0, "mma_route_not_observed")
        for selected, key in (
            (config in {"mla", "combination"}, "native.glm.prefill.mla_batch.tokens"),
            (config in {"kda", "combination"}, "glm.kda.prefill.register.tokens"),
        ):
            require((count(key) > 0) == selected, "prefill_route_not_observed")
        records.append(meta)
    return records


def validate(args, report, envs):
    args.lengths = [129, 513]
    retained = {}
    configs = ["defaults", "profile", "mma_off", "mla", "kda"]
    if args.include_combination:
        configs.append("combination")
    for config in configs:
        report["active_config"] = config
        write(args.output / "report.json", report)
        print(json.dumps({"stage": "readout_validation", "config": config}), flush=True)
        try:
            if config == "combination":
                require(all(report["validation"][c]["passed"] for c in ("mla", "kda")), "components_not_validated")
            rows = readouts.run_arm(args, envs[config], config)
            metadata = route_admission(rows, config)
            reference = "defaults" if config == "profile" else "profile"
            compared = [] if config == "defaults" else [readouts.compare(a, b) for a, b in zip(retained[reference], rows, strict=True)]
            passed = all(c["passed"] for c in compared)
            report["validation"][config] = {"passed": passed, "reference": None if config == "defaults" else reference,
                "cases": compared, "admission": metadata, "selectors": selectors(envs[config])}
            if config in {"defaults", "profile"}:
                retained[config] = rows
        except Exception as error:
            report["validation"][config] = {"passed": False, **failure(error)}
        write(args.output / "report.json", report)
    # No arrays, readouts or token IDs are returned or persisted.
    retained.clear()
    report.pop("active_config", None)
    report["passed"] = all(v["passed"] for v in report["validation"].values())


def validate_reply(reply, ready, input_count, budget, expected_reuse=0):
    require("error" not in reply, "native_request_failed")
    ids = reply.get("generated_token_ids")
    sampled = reply.get("sampled_tokens")
    stop = reply.get("stop_token_id")
    finish = reply.get("finish_reason")
    eos = type(stop) is int and stop in ready.get("eos_token_ids", [])
    require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids), "invalid_output")
    require(type(sampled) is int and 2 <= sampled <= budget and sampled == len(ids) + int(eos), "invalid_output_count")
    require(finish in {"length", "context", "eos"} and (finish == "eos") == eos
            and (eos or stop is None), "invalid_finish_reason")
    require("reused_input_tokens" in reply, "missing_reuse_accounting")
    reused = reply["reused_input_tokens"]
    require(type(reused) is int, "invalid_reuse_accounting")
    require(reply.get("input_tokens") == input_count and reused == expected_reuse, "input_or_reuse_mismatch")
    item = {"input_tokens": input_count, "sampled_tokens": sampled, "output_tokens": len(ids),
            "finish_reason": reply.get("finish_reason"), "output_sha256": token_ids_sha256(ids), "reused_input_tokens": reused}
    for key in ("prefill_seconds", "decode_seconds", "generation_seconds"):
        value = reply.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid_timing")
        item[key] = value
    require(item["decode_seconds"] > 0, "empty_decode_interval")
    item["useful_decode_tokens_per_second"] = max(0, len(ids) - 1) / item["decode_seconds"]
    return item


def session(args, config, env, index, workloads, repeats):
    result = {"config": config, "session": index, "selectors": selectors(env), "requests": [], "status": "starting"}
    write(args.output / f"session-{index}.json", result)
    gpu = []
    worker = monitor = None
    markers = []
    stage = "worker_start"
    try:
        with selected_environment(env):
            started = time.monotonic()
            worker = Worker(args.worker.resolve(), args.model.resolve(), args.timeout,
                            gpu_memory_callback=lambda _worker, amount: gpu.append(amount))
            stage = "worker_ready"
            ready = worker.read()
            result["startup_seconds"] = time.monotonic() - started
            if ready.get("ready") is not True:
                error_code = ready.get("error")
                if isinstance(error_code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
                    result["native_error_code"] = error_code
                require(False, "worker_not_ready")
            require(not ready.get("mtp", {}).get("supported", False), "unexpected_native_mtp")
            result["ready"] = {key: ready.get(key) for key in ("mode", "mtp", "prefix_cache", "max_context_tokens")}
            stage = "tokenization"
            tokenized = worker.request({"operation": "tokenize", "text": args.corpus.read_text(), "add_special_tokens": False})
            require(isinstance(tokenized.get("token_ids"), list) and tokenized["token_ids"], "tokenization_failed")
            corpus_ids = tokenized["token_ids"]
            try:
                from hybrid_host_metrics import HostMetricsSampler

                monitor = HostMetricsSampler(worker.process.pid)
                monitor.__enter__()
            except Exception as error:
                result["host_monitor"] = {"available": False, "failure_type": type(error).__name__}
                monitor = None
            request_id = 0
            for length, budget in (workloads if index % 2 == 0 else workloads[::-1]):
                ids = exact_prefix(corpus_ids, length)
                for replay in range(repeats + 1):
                    request_id += 1
                    item = {"input_tokens": length, "requested_output_tokens": budget, "replay": replay,
                            "warmup": replay == 0, "input_sha256": token_ids_sha256(ids), "status": "started"}
                    result["requests"].append(item)
                    result["status"] = "running"
                    write(args.output / f"session-{index}.json", result)
                    marker = None
                    if monitor is not None:
                        try:
                            marker = monitor.begin_request()
                        except Exception as error:
                            item["host_metrics"] = {"available": False, "failure_type": type(error).__name__}
                    started = time.monotonic()
                    stage = "generation"
                    try:
                        reply = worker.request({"operation": "generate", "input_token_ids": ids,
                            "context_tokens": length + budget, "max_new_tokens": budget,
                            "sampling": {"mode": "greedy"}, "request_id": request_id})
                    finally:
                        item["wall_seconds"] = time.monotonic() - started
                        if marker is not None:
                            try:
                                monitor.end_request(marker)
                                markers.append((item, marker))
                            except Exception as error:
                                item["host_metrics"] = {"available": False, "failure_type": type(error).__name__}
                    stage = "response_validation"
                    error_code = reply.get("error")
                    if isinstance(error_code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
                        item["native_error_code"] = error_code
                    item.update(validate_reply(reply, ready, length, budget))
                    item["passed"] = True
                    item["status"] = "completed"
                    write(args.output / f"session-{index}.json", result)
                    print(json.dumps({"stage": "request_completed", "config": config, "session": index,
                                      "input_tokens": length, "warmup": item["warmup"], "wall_seconds": item["wall_seconds"]}), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        result.update(failure(error), failed_stage=stage)
    finally:
        if monitor is not None:
            try:
                monitor.__exit__(None, None, None)
                for item, marker in markers:
                    item["host_metrics"] = monitor.request_result(marker)
                result["host_monitor"] = monitor.summary()
            except Exception as error:
                result["host_monitor"] = {"available": False, "failure_type": type(error).__name__}
        if worker is not None:
            try:
                worker.close()
            except Exception as error:
                result.update(failure(error), failed_stage="worker_close")
    result["sampled_peak_worker_device_allocation_bytes"] = max(gpu, default=None)
    result["worker_device_allocation_samples"] = len(gpu)
    result["passed"] = "failure_type" not in result and all(r.get("passed") for r in result["requests"])
    result["status"] = "completed" if result["passed"] else "failed"
    write(args.output / f"session-{index}.json", result)
    return result


def compare_sessions(sessions, workloads):
    summaries = []
    for length, budget in workloads:
        rows = [[r for r in s["requests"] if not r["warmup"] and r["input_tokens"] == length
                 and r["requested_output_tokens"] == budget] for s in sessions]
        before, after = rows[0] + rows[3], rows[1] + rows[2]
        require(len(before) == len(after) and before, "unbalanced_comparison")
        require(all(a["input_sha256"] == b["input_sha256"] and a["sampled_tokens"] == b["sampled_tokens"]
                    and a["output_sha256"] == b["output_sha256"] for a, b in zip(before, after, strict=True)), "output_parity_failed")
        require(len({r["output_sha256"] for r in before + after}) == 1, "repeatability_failed")
        wall = [a["wall_seconds"] / b["wall_seconds"] - 1 for a, b in zip(before, after, strict=True)]
        reduction = [1 - b["wall_seconds"] / a["wall_seconds"] for a, b in zip(before, after, strict=True)]
        positive_decode = all(r["useful_decode_tokens_per_second"] > 0 for r in before + after)
        decode = [b["useful_decode_tokens_per_second"] / a["useful_decode_tokens_per_second"] - 1
                  for a, b in zip(before, after, strict=True)] if positive_decode else []
        summarize = lambda v: {"min": min(v), "median": statistics.median(v), "max": max(v),
                               "relative_range": (max(v) - min(v)) / statistics.median(v) if statistics.median(v) > 0 else None}
        a_wall, b_wall = summarize([r["wall_seconds"] for r in before]), summarize([r["wall_seconds"] for r in after])
        a_decode, b_decode = summarize([r["useful_decode_tokens_per_second"] for r in before]), summarize([r["useful_decode_tokens_per_second"] for r in after])
        summaries.append({"input_tokens": length, "requested_output_tokens": budget, "pairs": len(wall),
            "sampled_tokens_min": min(r["sampled_tokens"] for r in before + after),
            "sampled_tokens_max": max(r["sampled_tokens"] for r in before + after),
            "before_wall_seconds": a_wall, "after_wall_seconds": b_wall,
            "before_useful_decode_tokens_per_second": a_decode, "after_useful_decode_tokens_per_second": b_decode,
            "median_paired_wall_speedup": statistics.median(wall), "wall_pair_wins": sum(x > 0 for x in wall),
            "median_paired_wall_reduction": statistics.median(reduction),
            "median_paired_decode_gain": statistics.median(decode) if decode else None, "decode_pair_wins": sum(x > 0 for x in decode),
            "wall_hurdle": max(0.10, a_wall["relative_range"], b_wall["relative_range"]),
            "decode_hurdle": max(0.10, a_decode["relative_range"], b_decode["relative_range"]) if positive_decode else None})
    return summaries


def prefix_controls(args, report, envs, configs):
    report["prefix"] = {}
    for config in configs:
        result = {"config": config, "requests": [], "passed": False}
        worker = None
        interrupted = False
        try:
            with selected_environment(envs[config]):
                worker = Worker(args.worker.resolve(), args.model.resolve(), args.timeout)
                ready = worker.read()
                require(ready.get("ready") is True and ready.get("prefix_cache", {}).get("enabled") is True,
                        "prefix_not_available")
                require(ready["prefix_cache"]["chunk_tokens"] == 1024, "unexpected_prefix_grid")
                tokenized = worker.request({"operation": "tokenize", "text": args.corpus.read_text(), "add_special_tokens": False})
                corpus = tokenized["token_ids"]
                require(bool(corpus), "empty_corpus")
                initial = exact_prefix(corpus, 1025)
                offset = next((i for i, token in enumerate(corpus) if token != initial[1024]), None)
                require(offset is not None, "no_distinct_suffix_control")
                shifted = corpus[offset:] + corpus[:offset]
                follow = initial[:1024] + exact_prefix(shifted, 33)
                require(follow[1024] != initial[1024], "unchanged_prefix_control")
                for i, (ids, scope, reused) in enumerate((
                    (initial, "a" * 32, 0), (follow, "a" * 32, 1024), (follow, "b" * 32, 0),
                )):
                    pending = {"request": i, "input_tokens": len(ids), "input_sha256": token_ids_sha256(ids),
                               "expected_reuse": reused, "status": "started"}
                    result["requests"].append(pending)
                    report["prefix"][config] = result
                    write(args.output / "report.json", report)
                    reply = worker.request({"operation": "generate", "input_token_ids": ids,
                        "context_tokens": 2048, "max_new_tokens": 8, "sampling": {"mode": "greedy"},
                        "cache_scope": scope, "request_id": i + 1})
                    pending.update(validate_reply(reply, ready, len(ids), 8, reused), status="completed")
                a, b = result["requests"][1:]
                require(a["output_sha256"] == b["output_sha256"] and a["sampled_tokens"] == b["sampled_tokens"],
                        "prefix_fresh_mismatch")
                result["passed"] = True
        except (Exception, KeyboardInterrupt) as error:
            result.update(failure(error))
            interrupted = isinstance(error, KeyboardInterrupt)
        finally:
            if worker is not None:
                try:
                    worker.close()
                except Exception as error:
                    result.update(failure(error), passed=False)
        report["prefix"][config] = result
        write(args.output / "report.json", report)
        print(json.dumps({"stage": "prefix_control", "config": config, "passed": result["passed"]}), flush=True)
        if interrupted:
            break
    report["passed"] = all(x["passed"] for x in report["prefix"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "prefix", "screen", "confirm"))
    for name in ("model", "worker", "example", "corpus", "verified-weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--comparison", choices=tuple(COMPARISONS), default="whole_profile")
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--prefix-receipt", type=Path)
    parser.add_argument("--configs", default="defaults,profile,mma_off,mla,kda")
    parser.add_argument("--include-combination", action="store_true")
    parser.add_argument("--decode-confirm", action="store_true",
                        help="Include the preregistered 512/128 cell during confirmation")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    envs = environments()
    initial = identities(args)
    weights = weight_identity(args.model, args.verified_weights)
    report = {"schema": "stark-glm-llama-full-model-v1", "phase": args.phase, "identities": initial,
        "weights": weights, "source": checkout_identity(), "hardware": hardware_identity(),
        "host_policy": "retain all measured requests under existing load; no quiet admission or retries",
        "validation": {}, "sessions": []}
    try:
        if args.phase == "validate":
            validate(args, report, envs)
        else:
            require(args.validation_receipt is not None, "validation_required")
            validated = json.loads(args.validation_receipt.read_text())
            require(validated.get("schema") == report["schema"] and validated.get("phase") == "validate",
                    "invalid_validation_receipt")
            require(validated["identities"] == initial and validated.get("weights") == weights,
                    "validation_identity_mismatch")
            if args.phase == "prefix":
                configs = args.configs.split(",")
                require(bool(configs) and all(c in envs and validated["validation"][c]["passed"] for c in configs),
                        "candidate_not_validated")
                prefix_controls(args, report, envs, configs)
                require(identities(args) == initial and weight_identity(args.model, args.verified_weights) == weights,
                        "identity_changed")
                write(args.output / "report.json", report)
                return 0 if report["passed"] else 1
            a, b = COMPARISONS[args.comparison]
            require(all(validated["validation"][c]["passed"] for c in (a, b)), "candidate_not_validated")
            workloads = [(96, 8), (512, 16)] if args.comparison == "whole_profile" else [(512, 16)]
            if args.phase == "confirm":
                require(args.prefix_receipt is not None, "prefix_controls_required")
                prefixes = json.loads(args.prefix_receipt.read_text())
                require(prefixes.get("schema") == report["schema"] and prefixes.get("phase") == "prefix",
                        "invalid_prefix_receipt")
                require(prefixes["identities"] == initial and prefixes.get("weights") == weights
                        and all(prefixes["prefix"][c]["passed"] for c in (a, b)),
                        "prefix_controls_not_passed")
                workloads = [(512, 16), (3072, 16)]
                if args.decode_confirm:
                    workloads.append((512, 128))
            report["comparison"] = args.comparison
            report["order"] = [a, b, b, a]
            for i, config in enumerate(report["order"]):
                print(json.dumps({"stage": "session_start", "config": config, "session": i}), flush=True)
                result = session(args, config, envs[config], i, workloads, 4 if args.phase == "confirm" else 1)
                report["sessions"].append(result)
                write(args.output / "report.json", report)
                require(result["passed"], "session_failed")
            report["summary"] = compare_sessions(report["sessions"], workloads)
            report["passed"] = True
        require(identities(args) == initial and weight_identity(args.model, args.verified_weights) == weights, "identity_changed")
    except (Exception, KeyboardInterrupt) as error:
        report["passed"] = False
        report.update(failure(error))
    write(args.output / "report.json", report)
    print(json.dumps({"phase": args.phase, "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        status = main()
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"passed": False, "failed_stage": "setup", **failure(error)}), flush=True)
        status = 1
    raise SystemExit(status)
