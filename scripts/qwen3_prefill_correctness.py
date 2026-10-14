"""Private-pipe correctness checks for Qwen3 prefill kernel candidates.

All named binaries load the identical native compressed model and use the six
authored cases from the original GGUF experiment. Full logits/token vectors stay
in memory. Source-GGUF llama.cpp checks run after all Stark workers have closed.
Only aggregate numerical comparisons, consistency counts and hashes persist.

Run with uv run --project tools/converter --locked python
scripts/qwen3_prefill_correctness.py --help. Never retain worker IPC transcripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import gguf_experiment as common
import qwen3_prefill_experiment as prefill

ROOT = common.ROOT


def validate_generation(response, ids, count, logits):
    first = response.get("first_token_id")
    common.require(
        response.get("generated_tokens") == count
        and common.HASH.fullmatch(response.get("generated_tokens_sha256", ""))
        is not None
        and response.get("input_tokens_sha256") == common.digest_ids(ids)
        and type(first) is int
        and 0 <= first < len(logits),
        "invalid_generation_response",
    )
    return int(logits[first] == max(logits))


def observe_variants(args, binaries, cases, report):
    tokens, outputs = [], {}
    for name, binary in binaries.items():
        worker = common.Worker(binary, args.model, args.timeout)
        references = []
        counts = {
            "score_calls": 0,
            "generation_calls": 0,
            "score_repeat_hash_agreements": 0,
            "generation_repeat_hash_agreements": 0,
            "generation_first_token_score_agreements": 0,
        }
        try:
            vocab = worker.ready.get("vocab_size")
            common.require(vocab == 151936, "unexpected_qwen3_vocabulary")
            for index, case in enumerate(cases):
                encoded = worker.request(
                    {"operation": "tokenize", "text": case["text"]}
                )
                raw_ids = encoded.get("token_ids")
                common.require(
                    isinstance(raw_ids, list)
                    and raw_ids
                    and common.digest_ids(raw_ids) == encoded.get("token_ids_sha256"),
                    "invalid_tokenization_response",
                )
                ids = raw_ids[: case["limit"]] if case["limit"] else raw_ids
                common.require(
                    case["limit"] is None or len(ids) == case["limit"],
                    "fixture_too_short",
                )
                common.require(
                    len(ids) + args.output_tokens <= args.context_tokens,
                    "fixture_context_overflow",
                )
                if len(tokens) < len(cases):
                    tokens.append(ids)
                else:
                    common.require(ids == tokens[index], "tokenizer_disagreement")
                reference = None
                for _ in range(args.repeats):
                    score = worker.request(
                        {
                            "operation": "score",
                            "input_token_ids": ids,
                            "context_tokens": args.context_tokens,
                        }
                    )
                    logits = prefill.validate_score(score, ids, vocab)
                    generated = worker.request(
                        {
                            "operation": "generate",
                            "input_token_ids": ids,
                            "context_tokens": args.context_tokens,
                            "output_tokens": args.output_tokens,
                        }
                    )
                    first_agreement = validate_generation(
                        generated, ids, args.output_tokens, logits
                    )
                    if reference is None:
                        reference = {
                            "logits": logits,
                            "logits_sha256": score["logits_sha256"],
                            "generation_sha256": generated["generated_tokens_sha256"],
                        }
                    counts["score_calls"] += 1
                    counts["generation_calls"] += 1
                    counts["score_repeat_hash_agreements"] += int(
                        score["logits_sha256"] == reference["logits_sha256"]
                    )
                    counts["generation_repeat_hash_agreements"] += int(
                        generated["generated_tokens_sha256"]
                        == reference["generation_sha256"]
                    )
                    counts["generation_first_token_score_agreements"] += first_agreement
                references.append(reference)
        finally:
            worker.close()
        outputs[name] = references
        report["variants"][name] = {
            **counts,
            "score_hashes": [reference["logits_sha256"] for reference in references],
            "generation_hashes": [
                reference["generation_sha256"] for reference in references
            ],
            "same_model_score_repeatable": counts["score_repeat_hash_agreements"]
            == counts["score_calls"],
            "same_model_generation_repeatable": counts[
                "generation_repeat_hash_agreements"
            ]
            == counts["generation_calls"],
            "generation_first_token_matches_score": counts[
                "generation_first_token_score_agreements"
            ]
            == counts["generation_calls"],
        }
        print(json.dumps({"status": "variant_complete", "variant": name}), flush=True)
    report["stark_workers_closed_before_reference"] = True
    report["fixture"] = {
        "cases": len(cases),
        "input_lengths": [len(ids) for ids in tokens],
        "input_token_hashes": [common.digest_ids(ids) for ids in tokens],
        "recipe_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    return tokens, outputs


def comparisons(outputs, baseline):
    references = outputs[baseline]
    result = {}
    for name, candidate in outputs.items():
        if name == baseline:
            continue
        common.require(len(candidate) == len(references), "comparison_case_mismatch")
        cases = []
        for index, (reference, row) in enumerate(zip(references, candidate)):
            cases.append(
                {
                    "case": index,
                    "exact_logit_hash_agreement": reference["logits_sha256"]
                    == row["logits_sha256"],
                    "exact_generation_hash_agreement": reference["generation_sha256"]
                    == row["generation_sha256"],
                    "logits": common.compare_logits(reference["logits"], row["logits"]),
                }
            )
        result[name + "_versus_" + baseline] = {
            "cases": len(cases),
            "exact_logit_hash_agreements": sum(
                row["exact_logit_hash_agreement"] for row in cases
            ),
            "exact_generation_hash_agreements": sum(
                row["exact_generation_hash_agreement"] for row in cases
            ),
            "logits": {
                key: common.stats([row["logits"][key] for row in cases])
                for key in cases[0]["logits"]
            },
            "by_case": cases,
        }
    return result


def bindings(args, binaries, source_files):
    binding = {
        "model": common.model_binding(args.model),
        "workers": {
            name: {"sha256": common.digest_file(binary)}
            for name, binary in binaries.items()
        },
        "source_files": {
            name: {"sha256": common.digest_file(path)}
            for name, path in source_files.items()
        },
        "harness_sha256": common.digest_file(Path(__file__)),
        "common_harness_sha256": common.digest_file(Path(common.__file__)),
        "prefill_harness_sha256": common.digest_file(Path(prefill.__file__)),
    }
    if args.llama_server:
        binding["source_reference"] = {
            "binary_sha256": common.digest_file(args.llama_server),
            "gguf_sha256": common.digest_file(args.gguf),
        }
    return binding


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--baseline")
    parser.add_argument(
        "--source-file", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    common.require(
        1 <= args.repeats <= 10
        and 2 <= args.output_tokens <= 128
        and 513 <= args.context_tokens <= 16384
        and math.isfinite(args.timeout)
        and args.timeout > 0,
        "invalid_correctness_arguments",
    )
    binaries = {
        name: Path(path).resolve()
        for name, path in prefill.assignments(
            args.binary, "invalid_binary_argument"
        ).items()
    }
    common.require(len(binaries) >= 2, "comparison_binaries_required")
    args.baseline = args.baseline or next(iter(binaries))
    common.require(args.baseline in binaries, "unknown_baseline")
    source_files = {
        name: Path(path).resolve()
        for name, path in prefill.assignments(
            args.source_file, "invalid_source_argument"
        ).items()
    }
    common.require(
        (args.llama_server is None) == (args.gguf is None),
        "incomplete_reference_arguments",
    )
    args.model = args.model.resolve()
    if args.llama_server:
        args.llama_server, args.gguf = args.llama_server.resolve(), args.gguf.resolve()
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
    return args, binaries, source_files


def run(args, binaries, source_files):
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    common.require(not args.output.parent.is_symlink(), "output_privacy_invalid")
    os.chmod(args.output.parent, 0o700)
    before = bindings(args, binaries, source_files)
    cases = common.fixture()
    report = {
        "schema_version": 1,
        "status": "running",
        "binding": before,
        "baseline": args.baseline,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "configuration": {
            "repeats": args.repeats,
            "context_tokens": args.context_tokens,
            "output_tokens": args.output_tokens,
        },
        "variants": {},
        "measurement_notes": [
            "All named workers load the identical model; binaries run serially and close before the source-reference server starts.",
            "The six authored cases reproduce the original GGUF experiment. Full-vocabulary values/token vectors remain in private pipes and RAM.",
            "Fixed-count greedy generation ignores stopping tokens. This is a scoped correctness check, not a semantic quality benchmark.",
            "Source-runtime top32 comparisons use each runtime's own log-partition normalization. Only aggregate differences and agreement counts persist.",
            "All worker/model/source/harness hashes are checked before and after. Explicit source-file bindings describe files present during execution, not inferred binary provenance.",
        ],
    }
    try:
        tokens, outputs = observe_variants(args, binaries, cases, report)
        report["native_comparisons"] = comparisons(outputs, args.baseline)
        if args.llama_server:
            common.run_reference(args, cases, tokens, outputs, report)
        common.require(
            bindings(args, binaries, source_files) == before,
            "correctness_binding_changed",
        )
        report["status"] = "complete"
        report["ownership_settled"] = True
    except BaseException as failure:
        report["status"] = "failed"
        report["error"] = (
            str(failure)
            if isinstance(failure, common.ExperimentFailure)
            else "correctness_failed"
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
                "variants": len(binaries),
                "aggregate_sha256": common.digest_file(args.output),
            }
        ),
        flush=True,
    )


def main(argv=None):
    try:
        args, binaries, source_files = parse_arguments(argv)
        run(args, binaries, source_files)
    except (Exception, KeyboardInterrupt) as failure:  # noqa: BLE001 - Never expose private/native exception content.
        code = (
            str(failure)
            if isinstance(failure, common.ExperimentFailure)
            else "correctness_failed"
        )
        print(json.dumps({"status": "failed", "error": code}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
