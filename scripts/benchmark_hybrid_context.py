"""Measure serialized decoding from an exact, corpus-derived prefix.

The report contains only hashes, token counts, finish reasons, and timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from stark_convert.chat import ChatError, Worker


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def run(args: argparse.Namespace, worker_factory=Worker, host_metrics_factory=None) -> dict:
    validate_args(args)
    corpus_bytes = args.corpus.read_bytes()
    try:
        corpus_text = corpus_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("corpus_not_utf8") from None
    if not corpus_text:
        raise ValueError("corpus_empty")

    report = {
        "schema_version": 2,
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
        "replays": [],
    }
    if report["sampling"]["mode"] == "stochastic":
        report["seed"] = getattr(args, "seed", 42)
    worker = worker_factory(args.worker.resolve(), args.model.resolve(), 3600)
    try:
        ready = worker.read()
        if ready.get("ready") is not True or ready.get("mode") != "text-generation-v1":
            raise ValueError("worker_not_ready")
        worker_context = ready.get("max_context_tokens")
        if type(worker_context) is int and args.input_tokens + args.tokens > worker_context:
            raise ValueError("worker_context_limit_exceeded")
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

        host_monitor = None
        if getattr(args, "host_metrics", False):
            if host_metrics_factory is None:
                from hybrid_host_metrics import HostMetricsSampler

                host_metrics_factory = HostMetricsSampler
            host_monitor = host_metrics_factory(getattr(getattr(worker, "process", None), "pid", None))
            host_monitor.__enter__()
        host_markers = []
        try:
            for replay in range(args.replays):
                marker = host_monitor.begin_request() if host_monitor else None
                if marker is not None:
                    host_markers.append(marker)
                try:
                    started = time.monotonic()
                    result = worker.request(
                        {
                            "operation": "generate",
                            "input_token_ids": ids,
                            "context_tokens": args.input_tokens + args.tokens,
                            "max_new_tokens": args.tokens,
                            "sampling": report["sampling"],
                            **({"seed": report["seed"]} if "seed" in report else {}),
                            "request_id": replay + 1,
                        }
                    )
                    elapsed = time.monotonic() - started
                finally:
                    if host_monitor and marker is not None:
                        host_monitor.end_request(marker)
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
                item = {
                    "replay": replay,
                    "wall_seconds": elapsed,
                    "sampled_tokens": sampled,
                    "output_tokens": len(generated),
                    "finish_reason": finish,
                    "output_sha256": token_ids_sha256(generated),
                }
                for key in (
                    "prefill_seconds",
                    "decode_seconds",
                    "input_tokens_per_second",
                    "output_tokens_per_second",
                    "generation_seconds",
                ):
                    value = result.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        item[key] = value
                report["replays"].append(item)
                print(
                    json.dumps(
                        {
                            "completed_replay": replay,
                            "sampled_tokens": sampled,
                            "finish_reason": finish,
                        }
                    ),
                    flush=True,
                )
        finally:
            if host_monitor:
                host_monitor.__exit__(None, None, None)

        if host_monitor:
            for item, marker in zip(report["replays"], host_markers, strict=True):
                item["host_metrics"] = host_monitor.request_result(marker)
            report["host_metrics_monitor"] = host_monitor.summary()

        report["repeat_exact"] = len({item["output_sha256"] for item in report["replays"]}) == 1
        report["median_wall_seconds"] = statistics.median(item["wall_seconds"] for item in report["replays"])
        report["median_generated_tokens"] = statistics.median(item["sampled_tokens"] for item in report["replays"])
        decode_rates = [
            item["output_tokens_per_second"] for item in report["replays"] if "output_tokens_per_second" in item
        ]
        if len(decode_rates) == len(report["replays"]):
            report["median_decode_tokens_per_second"] = statistics.median(decode_rates)
            report["median_warm_decode_tokens_per_second"] = statistics.median(decode_rates[1:])
        prefill_rates = [
            item["input_tokens_per_second"] for item in report["replays"][1:] if "input_tokens_per_second" in item
        ]
        if len(prefill_rates) == len(report["replays"]) - 1:
            report["median_warm_prefill_tokens_per_second"] = statistics.median(prefill_rates)
        report["warmup_replays"] = 1
        report["consistent_sample_counts"] = len({item["sampled_tokens"] for item in report["replays"]}) == 1
        report["passed"] = report["repeat_exact"] and report["consistent_sample_counts"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        worker.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--replays", type=int, choices=(3, 5), default=5)
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
