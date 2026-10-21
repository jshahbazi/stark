"""Compare complete-vocabulary digests without persisting request payloads."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

from benchmark_stock_engines import Pipe, generation_fixtures, save, sha


def score_token_sha(ids):
    return hashlib.sha256(
        b"".join(value.to_bytes(4, "little") for value in ids)
    ).hexdigest()


def run(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    fixtures = generation_fixtures(tokenizer)
    cases = [(f["name"], f["ids"]) for f in fixtures]
    source_ids = fixtures[4]["ids"]
    for length in (1, 2, 7, 8, 9, 16, 127, 128, 129, 1023, 1024, 1025, 2048):
        cases.append((f"prefix-{length}", source_ids[:length]))
    results = []
    for binary in (args.baseline, args.candidate):
        rows = []
        with contextlib.ExitStack() as stack:
            worker = Pipe(
                [
                    str(binary.resolve()),
                    "--score-model-path",
                    str(args.model.resolve()),
                ],
                stack,
                Path.cwd(),
                dict(os.environ),
            )
            for name, ids in cases:
                result = worker.request(
                    {
                        "operation": "score",
                        "input_token_ids": ids,
                        "candidate_token_ids": [ids[-1], 0 if ids[-1] else 1],
                        "context_tokens": 8192,
                        "temperature": 1.0,
                    }
                )
                if (
                    result["input_tokens"] != len(ids)
                    or result["prompt_tokens_sha256"] != score_token_sha(ids)
                    or result["cache_hit"]
                    or result["cache"]["enabled"]
                    or result["truncated"]
                ):
                    raise RuntimeError("invalid_digest_contract")
                rows.append(
                    {
                        "case": name,
                        "input_tokens": len(ids),
                        "input_sha256": score_token_sha(ids),
                        "vocab_size": result["vocab_size"],
                        "vocab_logits_sha256": result["vocab_logits_sha256"],
                    }
                )
        results.append(rows)
    mismatches = [
        baseline["case"]
        for baseline, candidate in zip(*results, strict=True)
        if baseline != candidate
    ]
    report = {
        "schema": 1,
        "harness_sha256": sha(__file__),
        "baseline_binary_sha256": sha(args.baseline),
        "candidate_binary_sha256": sha(args.candidate),
        "model_sha256": sha(args.model / "model.safetensors"),
        "cache_enabled": False,
        "timing_eligible": False,
        "cases": len(cases),
        "full_vocabulary_bitwise_matches": len(cases) - len(mismatches),
        "mismatches": mismatches,
        "baseline": results[0],
        "candidate": results[1],
    }
    save(args.output, report)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "full_vocabulary_bitwise_matches": len(cases) - len(mismatches),
            }
        )
    )
    return bool(mismatches)


def main():
    parser = argparse.ArgumentParser()
    for name in ("baseline", "candidate", "model", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - native diagnostics must stay private
        category = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        if not category.replace("_", "").isalnum() or len(category) > 80:
            category = "verification_failed"
        save(
            args.output,
            {
                "schema": 1,
                "status": "failed",
                "error": category,
                "harness_sha256": sha(__file__),
            },
        )
        print(json.dumps({"status": "failed", "error": category}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
