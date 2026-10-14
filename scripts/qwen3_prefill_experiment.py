"""Controlled aggregate-only prefill comparison for Qwen3 compressed variants.

Each variant uses a fresh, serial worker process in each balanced-order round.
Native scoring timings are collected before a separate fixed-count generation
phase. Token vectors and full logits remain in memory/private pipes; the output
contains only measurements, hashes and aggregate numerical comparisons.

Run with uv run --project tools/converter --locked python
scripts/qwen3_prefill_experiment.py --help. Never redirect worker IPC to a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import struct
from collections import defaultdict
from pathlib import Path

import gguf_experiment as common

ROOT = common.ROOT
LABEL = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
SCORE_METRICS = {
    "state_creation_seconds",
    "prefill_seconds",
    "readout_seconds",
    "total_seconds",
}
GENERATION_METRICS = {
    "state_creation_seconds",
    "time_to_first_token_seconds",
    "decode_seconds",
    "decode_tokens",
    "drain_seconds",
    "total_seconds",
}


def assignments(values, code):
    result = {}
    for value in values:
        name, separator, item = value.partition("=")
        common.require(
            separator and LABEL.fullmatch(name) and item and name not in result, code
        )
        result[name] = item
    return result


def comparison_pairs(values, names):
    pairs = []
    for value in values:
        candidate, separator, reference = value.partition("=")
        common.require(
            separator
            and candidate in names
            and reference in names
            and candidate != reference
            and (candidate, reference) not in pairs,
            "invalid_comparison_argument",
        )
        pairs.append((candidate, reference))
    return pairs or [(name, names[0]) for name in names[1:]]


def balanced_orders(names, rounds, seed):
    """Every complete cycle puts every model in every process-order position."""
    rng = random.Random(seed)
    orders = []
    while len(orders) < rounds:
        cycle = list(names)
        rng.shuffle(cycle)
        for offset in range(len(cycle)):
            orders.append(cycle[offset:] + cycle[:offset])
            if len(orders) == rounds:
                break
    return orders


def fixture(lengths):
    text = (
        "<|im_start|>user\n"
        + "The archive contains numbered observations about rivers, forests, stones, and clouds. "
        * 1024
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return [{"text": text, "limit": length} for length in lengths]


def logits_hash(values):
    digest = hashlib.sha256()
    try:
        for value in values:
            digest.update(struct.pack("<f", value))
    except (OverflowError, struct.error, TypeError):
        raise common.ExperimentFailure("invalid_logits") from None
    return digest.hexdigest()


def row(response, length, kind):
    allowed = SCORE_METRICS if kind == "score" else GENERATION_METRICS
    metrics = response.get("metrics")
    common.require(
        isinstance(metrics, dict) and allowed <= metrics.keys(), "invalid_metrics"
    )
    return common.metric_row(
        {"metrics": {key: metrics[key] for key in allowed}}, length, kind
    )


def validate_score(response, ids, vocab):
    logits = response.get("logits")
    common.require(isinstance(logits, list) and len(logits) == vocab, "invalid_logits")
    common.log_partition(logits)
    common.require(
        response.get("input_tokens_sha256") == common.digest_ids(ids),
        "scoring_input_mismatch",
    )
    common.require(
        response.get("logits_sha256") == logits_hash(logits), "score_hash_mismatch"
    )
    return logits


def peak_update(run, response):
    for key in ("process_peak_rss_bytes", "engine_peak_allocated_bytes"):
        value = response.get(key)
        common.require(type(value) is int and value >= 0, "invalid_memory_measurement")
        run[key] = max(run[key], value)


def run_workers(args, variants, report):
    outputs, tokens = defaultdict(dict), {}
    cases = fixture(args.lengths)
    orders = balanced_orders(list(variants), args.rounds, args.order_seed)
    report["model_order_by_round"] = orders
    for round_index, order in enumerate(orders):
        # Rotation distributes potential short/long-length heating effects.
        offset = round_index % len(cases)
        ordered_cases = cases[offset:] + cases[:offset]
        for name in order:
            variant = variants[name]
            worker = common.Worker(variant["binary"], variant["model"], args.timeout)
            try:
                vocab = worker.ready.get("vocab_size")
                common.require(vocab == 151936, "unexpected_qwen3_vocabulary")
                run = {
                    "round": round_index + 1,
                    "load_seconds": worker.ready["load_seconds"],
                    "rss_after_load_bytes": common.current_rss(worker.process),
                    "process_peak_rss_bytes": 0,
                    "engine_peak_allocated_bytes": 0,
                    "score_repeat_hash_agreements": 0,
                    "generation_repeat_hash_agreements": 0,
                    "generation_first_token_score_agreements": 0,
                    "length_order": [case["limit"] for case in ordered_cases],
                    "by_input_length": {},
                }
                peak_update(run, worker.ready)
                for case in ordered_cases:
                    length = case["limit"]
                    encoded = worker.request(
                        {"operation": "tokenize", "text": case["text"]}
                    )
                    ids = encoded.get("token_ids")
                    common.require(
                        isinstance(ids, list)
                        and common.digest_ids(ids) == encoded.get("token_ids_sha256"),
                        "invalid_tokenization_response",
                    )
                    ids = ids[:length]
                    common.require(len(ids) == length, "fixture_too_short")
                    if length in tokens:
                        common.require(ids == tokens[length], "tokenizer_disagreement")
                    else:
                        tokens[length] = ids
                    measured = []
                    for iteration in range(args.warmups + args.repeats):
                        response = worker.request(
                            {
                                "operation": "score",
                                "input_token_ids": ids,
                                "context_tokens": args.context_tokens,
                            }
                        )
                        logits = validate_score(response, ids, vocab)
                        reference = outputs[name].get(length)
                        if reference is None:
                            reference = {
                                "logits": logits,
                                "logits_sha256": response["logits_sha256"],
                            }
                            outputs[name][length] = reference
                        if iteration >= args.warmups:
                            measured.append(row(response, length, "score"))
                            run["score_repeat_hash_agreements"] += int(
                                response["logits_sha256"] == reference["logits_sha256"]
                            )
                        peak_update(run, response)
                    run["by_input_length"][str(length)] = {
                        "score": common.summarize_rows(measured)
                    }
                run["rss_after_score_bytes"] = common.current_rss(worker.process)
                for case in ordered_cases:
                    length, ids = case["limit"], tokens[case["limit"]]
                    measured = []
                    reference = outputs[name][length]
                    for iteration in range(args.warmups + args.generation_repeats):
                        response = worker.request(
                            {
                                "operation": "generate",
                                "input_token_ids": ids,
                                "context_tokens": args.context_tokens,
                                "output_tokens": args.output_tokens,
                            }
                        )
                        common.require(
                            response.get("generated_tokens") == args.output_tokens
                            and common.HASH.fullmatch(
                                response.get("generated_tokens_sha256", "")
                            )
                            is not None
                            and response.get("input_tokens_sha256")
                            == common.digest_ids(ids),
                            "invalid_generation_response",
                        )
                        if "generation_sha256" not in reference:
                            reference["generation_sha256"] = response[
                                "generated_tokens_sha256"
                            ]
                        first = response.get("first_token_id")
                        common.require(
                            type(first) is int and 0 <= first < vocab,
                            "invalid_generation_response",
                        )
                        if iteration >= args.warmups:
                            measured.append(row(response, length, "generate"))
                            run["generation_repeat_hash_agreements"] += int(
                                response["generated_tokens_sha256"]
                                == reference["generation_sha256"]
                            )
                            run["generation_first_token_score_agreements"] += int(
                                reference["logits"][first] == max(reference["logits"])
                            )
                        peak_update(run, response)
                    run["by_input_length"][str(length)]["generate"] = (
                        common.summarize_rows(measured)
                    )
                run["rss_after_inference_bytes"] = common.current_rss(worker.process)
                report["variants"][name]["runs"].append(run)
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
        "input_lengths": args.lengths,
        "input_token_hashes": [
            common.digest_ids(tokens[length]) for length in args.lengths
        ],
        "recipe_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest(),
    }
    for name, references in outputs.items():
        report["variants"][name]["score_hashes"] = [
            references[length]["logits_sha256"] for length in args.lengths
        ]
        report["variants"][name]["generation_hashes"] = [
            references[length]["generation_sha256"] for length in args.lengths
        ]
    report["comparisons"] = {}
    for candidate, reference in args.comparisons:
        by_length = {}
        for length in args.lengths:
            left, right = outputs[reference][length], outputs[candidate][length]
            ratios = []
            for left_run, right_run in zip(
                report["variants"][reference]["runs"],
                report["variants"][candidate]["runs"],
            ):
                left_time = left_run["by_input_length"][str(length)]["score"][
                    "prefill_seconds"
                ]["median"]
                right_time = right_run["by_input_length"][str(length)]["score"][
                    "prefill_seconds"
                ]["median"]
                ratios.append(left_time / right_time)
            by_length[str(length)] = {
                "logits": common.compare_logits(left["logits"], right["logits"]),
                "exact_logit_hash_agreement": left["logits_sha256"]
                == right["logits_sha256"],
                "exact_generation_hash_agreement": left["generation_sha256"]
                == right["generation_sha256"],
                "paired_round_median_prefill_speedup": common.stats(ratios),
            }
        report["comparisons"][candidate + "_versus_" + reference] = {
            "by_input_length": by_length
        }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--binary", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--default-binary",
        type=Path,
        default=ROOT / "target/release/stark-gguf-experiment",
    )
    parser.add_argument(
        "--build-commit", action="append", default=[], metavar="NAME=COMMIT"
    )
    parser.add_argument(
        "--worker-source-sha256", action="append", default=[], metavar="NAME=HASH"
    )
    parser.add_argument(
        "--compare", action="append", default=[], metavar="CANDIDATE=REFERENCE"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--generation-repeats", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--order-seed", type=int, default=1729)
    args = parser.parse_args(argv)
    common.require(
        1 <= args.rounds <= 24
        and 0 <= args.warmups <= 10
        and 1 <= args.repeats <= 50
        and 1 <= args.generation_repeats <= 20
        and 2 <= args.output_tokens <= 128
        and math.isfinite(args.timeout)
        and args.timeout > 0
        and 1 <= args.context_tokens <= 16384
        and args.lengths
        and len(args.lengths) == len(set(args.lengths))
        and all(
            1 <= length <= 4096 and length + args.output_tokens <= args.context_tokens
            for length in args.lengths
        ),
        "invalid_experiment_arguments",
    )
    models = assignments(args.model, "invalid_model_argument")
    common.require(len(models) >= 2, "comparison_models_required")
    binaries = assignments(args.binary, "invalid_binary_argument")
    commits = assignments(args.build_commit, "invalid_build_commit_argument")
    sources = assignments(args.worker_source_sha256, "invalid_source_hash_argument")
    common.require(
        set(binaries) <= set(models)
        and set(commits) <= set(models)
        and set(sources) <= set(models),
        "unknown_variant_binding",
    )
    common.require(
        all(re.fullmatch(r"[0-9a-f]{40}", commit) for commit in commits.values()),
        "invalid_build_commit_argument",
    )
    common.require(
        all(common.HASH.fullmatch(source) for source in sources.values()),
        "invalid_source_hash_argument",
    )
    args.comparisons = comparison_pairs(args.compare, list(models))
    variants = {
        name: {
            "model": Path(path).resolve(),
            "binary": Path(binaries.get(name, args.default_binary)).resolve(),
            "declared_build_commit": commits.get(name),
            "declared_worker_source_sha256": sources.get(name),
        }
        for name, path in models.items()
    }
    args.output = args.output.absolute()
    common.require(
        not args.output.exists() and not args.output.is_symlink(),
        "output_already_exists",
    )
    resolved = args.output.resolve()
    relative = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else None
    common.require(
        relative is not None and relative.parts[0] in common.PRIVATE_ROOTS,
        "output_must_be_private_runtime",
    )
    return args, variants


def bindings(variants):
    return {
        name: {
            "model": common.model_binding(variant["model"]),
            "worker_sha256": common.digest_file(variant["binary"]),
            "declared_build_commit": variant["declared_build_commit"],
            "declared_worker_source_sha256": variant["declared_worker_source_sha256"],
        }
        for name, variant in variants.items()
    }


def run(args, variants):
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    common.require(not args.output.parent.is_symlink(), "output_privacy_invalid")
    os.chmod(args.output.parent, 0o700)
    before = bindings(variants)
    report = {
        "schema_version": 1,
        "status": "running",
        "harness_sha256": common.digest_file(Path(__file__)),
        "common_harness_sha256": common.digest_file(Path(common.__file__)),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "configuration": {
            key: getattr(args, key)
            for key in (
                "lengths",
                "rounds",
                "warmups",
                "repeats",
                "generation_repeats",
                "context_tokens",
                "output_tokens",
                "order_seed",
            )
        },
        "variants": {
            name: {"binding": binding, "runs": []} for name, binding in before.items()
        },
        "measurement_notes": [
            "Models run serially; fresh process per model/round; every complete balanced-order cycle gives every variant every position.",
            "Within each process the length order rotates by round; score warmups and repeats precede a separate generation/decode phase.",
            "Native prefill timing excludes state creation, readout, JSON and private pipe overhead; each score starts with fresh KV state.",
            "Paired speedup is reference round-median latency divided by candidate round-median latency; rounds are separate fresh-process observations.",
            "Power, thermal state and filesystem cache remain uncontrolled; process-order balance does not eliminate these effects.",
            "Peak RSS and Metal allocations share unified-memory pages and must not be added. Model weight byte counts exclude runtime buffers.",
            "Declared build commits/source hashes are caller-supplied provenance, not reconstructed from the binary. Binary/model hashes are verified before and after.",
            "Fixed-count greedy generation ignores stop tokens. Full logits/token vectors remain in memory and private pipes; only aggregate metrics/hashes persist.",
        ],
    }
    try:
        run_workers(args, variants, report)
        common.require(bindings(variants) == before, "experiment_binding_changed")
        report["status"] = "complete"
        report["ownership_settled"] = True
    except BaseException as failure:
        report["status"] = "failed"
        report["error"] = (
            str(failure)
            if isinstance(failure, common.ExperimentFailure)
            else "experiment_failed"
        )
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, allow_nan=False, indent=2)
            output.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "variants": len(variants),
                "aggregate_sha256": common.digest_file(args.output),
            }
        ),
        flush=True,
    )


def main(argv=None):
    try:
        args, variants = parse_arguments(argv)
        run(args, variants)
    except (Exception, KeyboardInterrupt) as failure:  # noqa: BLE001 - Never expose private/native exception content.
        code = (
            str(failure)
            if isinstance(failure, common.ExperimentFailure)
            else "experiment_failed"
        )
        print(json.dumps({"status": "failed", "error": code}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
