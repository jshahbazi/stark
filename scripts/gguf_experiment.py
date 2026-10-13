"""Serial, aggregate-only Qwen3 GGUF import/runtime experiment.

Run with uv: uv run --project tools/converter --locked python
             scripts/gguf_experiment.py --help
Build worker: cargo +1.98.1 build --locked --release -p stark-score
              --bin stark-gguf-experiment

Prompts, tokens, logits and HTTP responses stay in memory and private pipes.
Only the explicitly selected aggregate output is written, mode 0600. No worker
stdout/stderr transcript is retained. Native captures and inherited llama
logging settings are disabled. Each model instance runs alone and exits before
the next instance starts. Performance is Stark-versus-Stark; optional llama.cpp
is a correctness reference only. Fixed-count greedy decoding includes EOS IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import platform
import queue
import random
import re
import socket
import statistics
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {"runtime", "test-artifacts", "artifacts"}
HASH = re.compile(r"[0-9a-f]{64}")


class ExperimentFailure(RuntimeError):
    """Authored, content-free failure code."""


def require(condition, code):
    if not condition:
        raise ExperimentFailure(code)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def digest_ids(ids):
    digest = hashlib.sha256()
    for token in ids:
        require(type(token) is int and 0 <= token <= 2**32 - 1, "invalid_token_values")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def clean_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STARK_CAPTURE_", "UZU_CAPTURE_", "LLAMA_", "GGML_"))
    }


def stats(values):
    require(
        values
        and all(
            type(value) in (int, float) and math.isfinite(value) for value in values
        ),
        "invalid_metrics",
    )
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def log_partition(logits):
    require(
        logits
        and all(
            type(value) in (int, float) and math.isfinite(value) for value in logits
        ),
        "invalid_logits",
    )
    maximum = max(logits)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in logits))


def compare_logits(reference, candidate, top_k=32):
    """Full-vocabulary aggregates, including shift-invariant softmax divergence."""
    require(
        len(reference) == len(candidate) and len(reference) >= top_k,
        "logit_shape_mismatch",
    )
    ref_partition, cand_partition = log_partition(reference), log_partition(candidate)
    differences = [cand - ref for ref, cand in zip(reference, candidate)]
    ref_norm = math.fsum(value * value for value in reference)
    cand_norm = math.fsum(value * value for value in candidate)
    require(ref_norm > 0 and cand_norm > 0, "zero_logit_norm")
    ref_top = heapq.nlargest(top_k, range(len(reference)), key=reference.__getitem__)
    cand_top = heapq.nlargest(top_k, range(len(candidate)), key=candidate.__getitem__)
    kl = math.fsum(
        math.exp(ref - ref_partition)
        * ((ref - ref_partition) - (cand - cand_partition))
        for ref, cand in zip(reference, candidate)
    )
    return {
        "values": len(reference),
        "mean_absolute_error": statistics.fmean(abs(value) for value in differences),
        "root_mean_square_error": math.sqrt(
            statistics.fmean(value * value for value in differences)
        ),
        "maximum_absolute_error": max(abs(value) for value in differences),
        "cosine_similarity": math.fsum(
            ref * cand for ref, cand in zip(reference, candidate)
        )
        / math.sqrt(ref_norm * cand_norm),
        "reference_to_candidate_kl": max(0.0, kl),
        "top1_agreement": int(ref_top[0] == cand_top[0]),
        "top_k_overlap": len(set(ref_top).intersection(cand_top)) / top_k,
    }


def current_rss(process):
    require(process.poll() is None, "worker_exited")
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(process.pid)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    require(
        result.returncode == 0 and result.stdout.strip().isdigit(),
        "memory_measurement_failed",
    )
    return int(result.stdout.strip()) * 1024


def stop_process(process):
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if process.stdout is not None:
        process.stdout.close()


class Worker:
    def __init__(self, binary, model, timeout):
        self.timeout, self.lines = timeout, queue.Queue()
        self.process = subprocess.Popen(
            [str(binary), "--score-model-path", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=clean_environment(),
            cwd=ROOT,
        )
        self.reader = threading.Thread(target=self.read_lines, daemon=True)
        self.reader.start()
        try:
            self.ready = self.receive()
            require(
                self.ready.get("ready") is True
                and self.ready.get("protocol") == "qwen3-gguf-experiment-v1",
                "worker_not_ready",
            )
        except BaseException:
            self.close()
            raise

    def read_lines(self):
        try:
            while line := self.process.stdout.readline(32 * 1024 * 1024):
                if len(line) >= 32 * 1024 * 1024:
                    self.lines.put(None)
                    return
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise ExperimentFailure("worker_timeout") from None
        require(line is not None, "worker_exited")
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            raise ExperimentFailure("invalid_worker_response") from None
        require(
            isinstance(value, dict) and "error" not in value, "worker_operation_failed"
        )
        return value

    def request(self, value):
        require(self.process.poll() is None, "worker_exited")
        self.process.stdin.write(
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        )
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        stop_process(self.process)
        self.reader.join(timeout=1)


def fixture():
    """Authored deterministic public recipes; no external/private evaluation data."""
    questions = [
        "Compute 17 times 23 and give only the number.",
        "Write a Python function that returns the larger of two integers.",
        "用中文简要说明为什么天空是蓝色的。",
        "Name the planet immediately farther from the Sun than Earth.",
    ]
    prefix = "<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    cases = [
        {"text": prefix + question + suffix, "limit": None} for question in questions
    ]
    filler = "The archive contains numbered observations about rivers, forests, stones, and clouds. "
    cases.extend(
        {"text": prefix + filler * 128 + suffix, "limit": limit} for limit in (128, 512)
    )
    return cases


def model_binding(path):
    names = ("config.json", "tokenizer.json", "model.safetensors")
    return {
        "files": {
            name: {
                "sha256": digest_file(path / name),
                "bytes": (path / name).stat().st_size,
            }
            for name in names
        },
        "total_bytes": sum((path / name).stat().st_size for name in names),
    }


def metric_row(value, length, kind):
    metrics = value.get("metrics")
    require(isinstance(metrics, dict), "invalid_metrics")
    row = {key: val for key, val in metrics.items()}
    require(
        all(
            type(val) in (int, float) and math.isfinite(val) and val >= 0
            for val in row.values()
        ),
        "invalid_metrics",
    )
    if kind == "score":
        require(row.get("prefill_seconds", 0) > 0, "invalid_metrics")
        row["prefill_tokens_per_second"] = length / row["prefill_seconds"]
    else:
        require(row.get("decode_seconds", 0) > 0, "invalid_metrics")
        row["decode_completed_tokens_per_second"] = row["decode_tokens"] / (
            row["decode_seconds"] + row["drain_seconds"]
        )
        row["decode_buffered_iterator_tokens_per_second"] = (
            row["decode_tokens"] / row["decode_seconds"]
        )
        row["end_to_end_tokens_per_second"] = (row["decode_tokens"] + 1) / row[
            "total_seconds"
        ]
    return row


def summarize_rows(rows):
    return {key: stats([row[key] for row in rows]) for key in rows[0]}


def run_stark(args, models, cases, report):
    tokens, outputs = [], defaultdict(dict)
    rng = random.Random(args.order_seed)
    report["models"] = {
        name: {"binding": model_binding(path), "runs": []} for name, path in models
    }
    report["model_order_by_round"] = []
    for round_index in range(args.rounds):
        order = list(models)
        rng.shuffle(order)
        report["model_order_by_round"].append([name for name, _ in order])
        for name, path in order:
            worker = Worker(args.binary, path, args.timeout)
            try:
                require(
                    worker.ready.get("vocab_size") == 151936,
                    "unexpected_qwen3_vocabulary",
                )
                run = {
                    "round": round_index + 1,
                    "load_seconds": worker.ready["load_seconds"],
                    "rss_after_load_bytes": current_rss(worker.process),
                    "score": [],
                    "generate": [],
                    "process_peak_rss_bytes": worker.ready["process_peak_rss_bytes"],
                    "engine_peak_allocated_bytes": worker.ready[
                        "engine_peak_allocated_bytes"
                    ],
                    "rss_after_inference_bytes": 0,
                    "score_repeat_hash_agreements": 0,
                    "generation_repeat_hash_agreements": 0,
                    "generation_first_token_score_agreements": 0,
                }
                by_length = defaultdict(lambda: {"score": [], "generate": []})
                for case_index, case in enumerate(cases):
                    tokenized = worker.request(
                        {"operation": "tokenize", "text": case["text"]}
                    )
                    raw_ids = tokenized.get("token_ids")
                    require(
                        isinstance(raw_ids, list)
                        and raw_ids
                        and digest_ids(raw_ids) == tokenized.get("token_ids_sha256"),
                        "invalid_tokenization_response",
                    )
                    ids = raw_ids[: case["limit"]] if case["limit"] else raw_ids
                    require(
                        case["limit"] is None or len(ids) == case["limit"],
                        "fixture_too_short",
                    )
                    if len(tokens) < len(cases):
                        tokens.append(ids)
                    else:
                        require(ids == tokens[case_index], "tokenizer_disagreement")
                    require(
                        len(ids) + args.output_tokens < args.context_tokens,
                        "fixture_context_overflow",
                    )
                    for iteration in range(args.warmups + args.repeats):
                        score = worker.request(
                            {
                                "operation": "score",
                                "input_token_ids": ids,
                                "context_tokens": args.context_tokens,
                            }
                        )
                        logits = score.get("logits")
                        require(
                            isinstance(logits, list)
                            and len(logits) == worker.ready["vocab_size"],
                            "invalid_logits",
                        )
                        log_partition(logits)
                        require(
                            score.get("input_tokens_sha256") == digest_ids(ids),
                            "scoring_input_mismatch",
                        )
                        require(
                            HASH.fullmatch(score.get("logits_sha256", "")) is not None,
                            "invalid_score_hash",
                        )
                        generation = worker.request(
                            {
                                "operation": "generate",
                                "input_token_ids": ids,
                                "context_tokens": args.context_tokens,
                                "output_tokens": args.output_tokens,
                            }
                        )
                        require(
                            generation.get("generated_tokens") == args.output_tokens
                            and HASH.fullmatch(
                                generation.get("generated_tokens_sha256", "")
                            )
                            is not None,
                            "invalid_generation_response",
                        )
                        reference = outputs[name].get(case_index)
                        if reference is None:
                            outputs[name][case_index] = {
                                "logits": logits,
                                "logits_sha256": score["logits_sha256"],
                                "generation_sha256": generation[
                                    "generated_tokens_sha256"
                                ],
                            }
                        if iteration >= args.warmups:
                            for kind, response in (
                                ("score", score),
                                ("generate", generation),
                            ):
                                row = metric_row(response, len(ids), kind)
                                run[kind].append(row)
                                by_length[len(ids)][kind].append(row)
                            reference = outputs[name][case_index]
                            run["score_repeat_hash_agreements"] += int(
                                score["logits_sha256"] == reference["logits_sha256"]
                            )
                            run["generation_repeat_hash_agreements"] += int(
                                generation["generated_tokens_sha256"]
                                == reference["generation_sha256"]
                            )
                            first_token = generation.get("first_token_id")
                            require(
                                type(first_token) is int
                                and 0 <= first_token < len(logits),
                                "invalid_generation_response",
                            )
                            run["generation_first_token_score_agreements"] += int(
                                logits[first_token] == max(logits)
                            )
                        for response in (score, generation):
                            run["process_peak_rss_bytes"] = max(
                                run["process_peak_rss_bytes"],
                                response["process_peak_rss_bytes"],
                            )
                            run["engine_peak_allocated_bytes"] = max(
                                run["engine_peak_allocated_bytes"],
                                response["engine_peak_allocated_bytes"],
                            )
                    run["rss_after_inference_bytes"] = current_rss(worker.process)
                for kind in ("score", "generate"):
                    run[kind] = summarize_rows(run[kind])
                run["by_input_length"] = {
                    str(length): {
                        kind: summarize_rows(rows) for kind, rows in groups.items()
                    }
                    for length, groups in sorted(by_length.items())
                }
                report["models"][name]["runs"].append(run)
            finally:
                worker.close()
            print(
                json.dumps(
                    {
                        "status": "model_round_complete",
                        "model": name,
                        "round": round_index + 1,
                    }
                ),
                flush=True,
            )
    report["fixture"] = {
        "cases": len(cases),
        "input_lengths": [len(ids) for ids in tokens],
        "input_token_hashes": [digest_ids(ids) for ids in tokens],
        "recipe_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    report["comparisons"] = {}
    for name, _ in models:
        report["models"][name]["score_hashes"] = [
            outputs[name][index]["logits_sha256"] for index in range(len(cases))
        ]
        report["models"][name]["generation_hashes"] = [
            outputs[name][index]["generation_sha256"] for index in range(len(cases))
        ]
    baseline_name = models[0][0]
    for name, _ in models[1:]:
        comparisons = [
            compare_logits(
                outputs[baseline_name][index]["logits"], outputs[name][index]["logits"]
            )
            for index in range(len(cases))
        ]
        report["comparisons"][name + "_versus_" + baseline_name] = {
            "interpretation": "Different source quantizers or revisions can affect this baseline comparison; this is not an importer equality oracle.",
            "cases": len(cases),
            "logits": {
                key: stats([row[key] for row in comparisons]) for key in comparisons[0]
            },
            "generation_hash_agreements": sum(
                outputs[baseline_name][index]["generation_sha256"]
                == outputs[name][index]["generation_sha256"]
                for index in range(len(cases))
            ),
        }
    return tokens, outputs


def http_json(port, endpoint, payload=None, timeout=60):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Never send private local reference traffic through inherited proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def reference_top_rows(response):
    rows = response.get("completion_probabilities", response.get("probs"))
    require(
        isinstance(rows, list) and len(rows) == 1, "invalid_reference_probabilities"
    )
    top = rows[0].get("top_logprobs")
    require(isinstance(top, list) and len(top) == 32, "invalid_reference_probabilities")
    require(
        all(
            type(row.get("id")) is int
            and type(row.get("logprob")) in (int, float)
            and math.isfinite(row["logprob"])
            for row in top
        ),
        "invalid_reference_probabilities",
    )
    require(len({row["id"] for row in top}) == 32, "invalid_reference_probabilities")
    return top


def compare_reference_top(top, logits):
    partition = log_partition(logits)
    require(
        all(0 <= row["id"] < len(logits) for row in top),
        "reference_vocabulary_mismatch",
    )
    differences = [(logits[row["id"]] - partition) - row["logprob"] for row in top]
    candidate_top = heapq.nlargest(32, range(len(logits)), key=logits.__getitem__)
    return {
        "top1_agreement": int(top[0]["id"] == candidate_top[0]),
        "top32_overlap": len({row["id"] for row in top}.intersection(candidate_top))
        / 32,
        "reference_top32_logprob_mae": statistics.fmean(
            abs(value) for value in differences
        ),
        "reference_top32_logprob_rmse": math.sqrt(
            statistics.fmean(value * value for value in differences)
        ),
        "reference_top32_logprob_max_absolute_error": max(
            abs(value) for value in differences
        ),
    }


def run_reference(args, cases, tokens, outputs, report):
    # Bind only loopback. The port handoff can race; require readiness before use.
    with socket.socket() as lease:
        lease.bind(("127.0.0.1", 0))
        port = lease.getsockname()[1]
    process = subprocess.Popen(
        [
            str(args.llama_server),
            "--model",
            str(args.gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(args.context_tokens),
            "--parallel",
            "1",
            "--gpu-layers",
            "all",
            "--no-cache-prompt",
            "--log-disable",
            "--no-warmup",
            "--no-slots",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=clean_environment(),
        cwd=ROOT,
    )
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            require(process.poll() is None, "reference_server_exited")
            try:
                ready = http_json(port, "/health", timeout=1)
                if ready.get("status") == "ok":
                    break
            except (OSError, urllib.error.URLError, ValueError):
                pass
            require(time.monotonic() < deadline, "reference_server_timeout")
            time.sleep(0.1)
        # The source identity and binary hashes bind a locally installed reference.
        reference = {
            "binary_sha256": digest_file(args.llama_server),
            "gguf_sha256": digest_file(args.gguf),
            "purpose": "Independent source-GGUF inference correctness; reference timing is not a Stark performance baseline.",
            "tokenizer_hash_agreements": 0,
            "cases": len(cases),
            "models": {},
        }
        comparisons = defaultdict(list)
        generations = defaultdict(int)
        for index, (case, ids) in enumerate(zip(cases, tokens)):
            encoded = http_json(
                port,
                "/tokenize",
                {"content": case["text"], "add_special": False},
                args.timeout,
            )
            ref_ids = encoded.get("tokens")
            require(isinstance(ref_ids, list), "invalid_reference_tokenization")
            ref_ids = ref_ids[: case["limit"]] if case["limit"] else ref_ids
            reference["tokenizer_hash_agreements"] += int(ref_ids == ids)
            require(ref_ids == ids, "reference_tokenizer_disagreement")
            payload = {
                "prompt": ids,
                "n_predict": 1,
                "temperature": 0,
                "seed": 0,
                "cache_prompt": False,
                "return_tokens": True,
                "n_probs": 32,
                "post_sampling_probs": False,
                "ignore_eos": True,
                "repeat_penalty": 1.0,
                "samplers": ["temperature"],
            }
            result = http_json(port, "/completion", payload, args.timeout)
            require(
                result.get("truncated") is False
                and result.get("tokens_evaluated") == len(ids),
                "reference_prompt_mismatch",
            )
            top = reference_top_rows(result)
            generation = http_json(
                port,
                "/completion",
                {**payload, "n_predict": args.output_tokens, "n_probs": 0},
                args.timeout,
            )
            generated = generation.get("tokens")
            require(
                isinstance(generated, list) and len(generated) == args.output_tokens,
                "invalid_reference_generation",
            )
            generated_hash = digest_ids(generated)
            for name, rows in outputs.items():
                comparisons[name].append(
                    compare_reference_top(top, rows[index]["logits"])
                )
                generations[name] += int(
                    rows[index]["generation_sha256"] == generated_hash
                )
        for name, rows in comparisons.items():
            reference["models"][name] = {
                "top32_comparison": {
                    key: stats([row[key] for row in rows]) for key in rows[0]
                },
                "generation_hash_agreements": generations[name],
            }
        report["source_runtime_reference"] = reference
    finally:
        stop_process(process)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/stark-gguf-experiment"
    )
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--order-seed", type=int, default=1729)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--gguf", type=Path)
    args = parser.parse_args(argv)
    require(
        1 <= args.rounds <= 10
        and 0 <= args.warmups <= 10
        and 1 <= args.repeats <= 20
        and 513 <= args.context_tokens <= 16384
        and 2 <= args.output_tokens <= 128
        and math.isfinite(args.timeout)
        and args.timeout > 0,
        "invalid_experiment_arguments",
    )
    require(
        (args.llama_server is None) == (args.gguf is None),
        "incomplete_reference_arguments",
    )
    models = []
    for item in args.model:
        name, separator, path = item.partition("=")
        require(
            separator and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", name),
            "invalid_model_argument",
        )
        require(
            name not in {existing for existing, _ in models}, "duplicate_model_name"
        )
        models.append((name, Path(path).resolve()))
    require(len(models) >= 2, "comparison_models_required")
    args.output = args.output.absolute()
    require(
        not args.output.exists() and not args.output.is_symlink(),
        "output_already_exists",
    )
    relative = (
        args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else None
    )
    require(
        relative is not None and relative.parts[0] in PRIVATE_ROOTS,
        "output_must_be_private_runtime",
    )
    args.binary = args.binary.resolve()
    return args, models


def run(args, models):
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not args.output.parent.is_symlink(), "output_privacy_invalid")
    os.chmod(args.output.parent, 0o700)
    before = {name: model_binding(path) for name, path in models}
    binary_before = digest_file(args.binary)
    report = {
        "schema_version": 1,
        "status": "running",
        "worker_sha256": binary_before,
        "harness_sha256": digest_file(Path(__file__)),
        "worker_source_sha256": digest_file(
            ROOT / "crates/stark-score/probes/gguf_experiment.rs"
        ),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "configuration": {
            key: getattr(args, key)
            for key in (
                "rounds",
                "warmups",
                "repeats",
                "context_tokens",
                "output_tokens",
                "order_seed",
            )
        },
        "measurement_notes": [
            "Fresh process per model round; fixed-seed shuffled model order; warm filesystem cache is uncontrolled.",
            "Score timing excludes JSON/pipe overhead; prefill uses fresh KV state. No decision cache exists in the experiment worker.",
            "Completed decode throughput counts returned tokens after the first and includes final stream drain. Buffered iterator throughput excludes drain and is separately labelled.",
            "Peak RSS is native getrusage ru_maxrss in bytes; resident snapshots use ps RSS after load and inference.",
            "Engine peak measures tracked Metal allocation bytes. Apple unified memory shares pages; RSS and engine allocation peaks are not additive or total physical-memory accounting.",
            "Fixed-count greedy generation ignores stopping tokens, uses seed zero, and retains only hashes. No quality benchmark or general GGUF support is claimed.",
        ],
    }
    try:
        tokens, outputs = run_stark(args, models, fixture(), report)
        if args.llama_server:
            run_reference(args, fixture(), tokens, outputs, report)
        require(
            {name: model_binding(path) for name, path in models} == before
            and digest_file(args.binary) == binary_before,
            "experiment_binding_changed",
        )
        report["status"] = "complete"
        report["ownership_settled"] = True
    except BaseException as failure:
        report["status"] = "failed"
        report["error"] = (
            str(failure)
            if isinstance(failure, ExperimentFailure)
            else "experiment_failed"
        )
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "models": len(models),
                "cases": report["fixture"]["cases"],
                "aggregate_sha256": digest_file(args.output),
            }
        ),
        flush=True,
    )


def main(argv=None):
    try:
        args, models = parse_arguments(argv)
        run(args, models)
    except (Exception, KeyboardInterrupt) as failure:  # noqa: BLE001 - Never expose native/private exception content.
        code = (
            str(failure)
            if isinstance(failure, ExperimentFailure)
            else "experiment_failed"
        )
        print(json.dumps({"status": "failed", "error": code}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
