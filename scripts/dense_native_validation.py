#!/usr/bin/env python3
"""Private full-checkpoint references and native worker checks for new dense models.

Run ``reference`` with the frozen MLX-LM validation environment. Run ``workers``
with the converter environment. Reference tensors, prompts and token sequences
are private fixtures under an ignored directory. Only aggregates and hashes are
printed. No remote model code is executed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import struct
import time
from pathlib import Path

from test_hybrid_workers import CheckFailed, digest, require, worker_process

MLX_LM_REVISION = "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o600)


def token_hash(values: list[int]) -> str:
    return hashlib.sha256(
        b"".join(struct.pack("<I", value) for value in values)
    ).hexdigest()


def reference(args) -> dict:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import mlx.core as mx
    import numpy as np
    from huggingface_hub.utils import logging as hub_logging
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model, load_tokenizer
    from transformers.utils import logging as transformers_logging

    hub_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    direct = json.loads(
        importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
    )
    require(
        direct.get("vcs_info", {}).get("commit_id") == MLX_LM_REVISION,
        "reference_revision_changed",
    )
    require(
        importlib.metadata.version("mlx") == "0.32.3", "reference_mlx_version_changed"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    source = args.source.resolve(strict=True)
    model, config = load_model(source, lazy=True, strict=True, trust_remote_code=False)
    tokenizer = load_tokenizer(source)
    mx.eval(model.parameters())
    text = config.get("text_config", config)
    window = text["sliding_window"]
    require(
        type(window) is int and 1 <= window <= 4096,
        "reference_window_outside_test_bounds",
    )
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in (
            "Count from one to five, separated by commas.",
            "Write a Python function that adds two integers. Return only the code.",
        )
    ]
    prompts = [tokenizer.encode(value, add_special_tokens=False) for value in rendered]
    eos = config.get("eos_token_id", text.get("eos_token_id"))
    eos = {eos} if type(eos) is int else set(eos or [])
    generation_path = source / "generation_config.json"
    if generation_path.exists():
        extra = json.loads(generation_path.read_text()).get("eos_token_id", [])
        eos.update([extra] if type(extra) is int else extra)
    eos.update(tokenizer.eos_token_ids)

    def prefill(tokens):
        cache = make_prompt_cache(model)
        started = time.perf_counter()
        # Match the native score/stream contract: cache the prefix, then read
        # one final token. BF16 GEMM and GEMV have different rounding boundaries;
        # a single full-sequence call is a separate numerical profile.
        prefix = tokens[:-1]
        for offset in range(0, len(prefix), 1024):
            result = model(
                mx.array([prefix[offset : offset + 1024]], dtype=mx.int32), cache=cache
            )
            mx.eval(result)
        result = model(mx.array([tokens[-1:]], dtype=mx.int32), cache=cache)
        mx.eval(result)
        return result[0, -1], cache, time.perf_counter() - started

    cases = []
    base = prompts[0]
    repeated = (base * ((window + 5) // len(base) + 1))[: window + 5]
    sequences = [base[:1], base, prompts[1], repeated[:129], repeated]
    for index, tokens in enumerate(sequences):
        logits, _cache, elapsed = prefill(tokens)
        values = np.asarray(logits.astype(mx.float32), dtype="<f4")
        require(bool(np.isfinite(values).all()), "reference_logits_nonfinite")
        path = output / f"reference-{index}.f32"
        path.write_bytes(values.tobytes())
        path.chmod(0o600)
        cases.append({"tokens": tokens, "context_tokens": max(1024, len(tokens) + 32)})
        print(
            json.dumps(
                {
                    "phase": "reference_case",
                    "case": index,
                    "input_tokens": len(tokens),
                    "seconds": elapsed,
                    "reference_sha256": digest(path),
                }
            ),
            flush=True,
        )
    save_json(output / "cases.json", {"cases": cases})
    generations = []
    for prompt, tokens in zip(rendered, prompts, strict=True):
        logits, cache, prefill_seconds = prefill(tokens)
        generated = []
        started = time.perf_counter()
        sampled = 0
        for _ in range(32):
            token = int(mx.argmax(logits).item())
            sampled += 1
            if token in eos:
                break
            generated.append(token)
            if len(generated) == 32:
                break
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1]
            mx.eval(logits)
        generations.append(
            {
                "rendered_prompt": prompt,
                "input_tokens": tokens,
                "reference_tokens": generated,
                "max_new_tokens": 32,
                "context_tokens": 1024,
                "reference_sampled_tokens": sampled,
                "reference_prefill_seconds": prefill_seconds,
                "reference_decode_seconds": time.perf_counter() - started,
            }
        )
    save_json(output / "generations.json", generations)
    report = {
        "passed": True,
        "cases": len(cases),
        "generation_cases": len(generations),
        "source_config_sha256": digest(source / "config.json"),
        "mlx_lm_revision": MLX_LM_REVISION,
        "mlx": importlib.metadata.version("mlx"),
        "forward_profile": "prefix-cache-1024-then-last-token-v2",
        "cases_sha256": digest(output / "cases.json"),
        "generations_sha256": digest(output / "generations.json"),
    }
    save_json(output / "reference-report.json", report)
    return report


async def workers(args) -> dict:
    fixture = args.fixture.resolve(strict=True)
    model = args.model.resolve(strict=True)
    cases = json.loads((fixture / "generations.json").read_text())
    require(
        isinstance(cases, list) and 1 <= len(cases) <= 8, "invalid_generation_fixture"
    )
    reports = []
    async with worker_process(
        args.generate_worker.resolve(strict=True), ["--model-path", str(model)]
    ) as worker:
        ready = await worker.reply()
        require(ready.get("ready") is True, "generator_not_ready")
        for case in cases:
            tokenized = await worker.request(
                {"operation": "tokenize", "text": case["rendered_prompt"]}
            )
            require(
                tokenized.get("token_ids") == case["input_tokens"],
                "native_tokenization_mismatch",
            )
            request = {
                "operation": "generate",
                "rendered_prompt": case["rendered_prompt"],
                "context_tokens": case["context_tokens"],
                "max_new_tokens": case["max_new_tokens"],
            }
            first = await worker.request(request)
            repeat = await worker.request(request)
            require(
                "error" not in first and "error" not in repeat,
                "native_generation_failed",
            )
            require(
                first["generated_token_ids"] == repeat["generated_token_ids"],
                "generation_replay_changed",
            )
            failed = await worker.request(
                {
                    **request,
                    "rendered_prompt": None,
                    "input_token_ids": [ready["vocab_size"]],
                }
            )
            require("error" in failed, "invalid_generation_request_accepted")
            recovered = await worker.request(request)
            require(
                recovered.get("generated_token_ids") == first["generated_token_ids"],
                "failure_recovery_changed",
            )
            expected = case["reference_tokens"]
            actual = first["generated_token_ids"]
            common = next(
                (
                    i
                    for i, pair in enumerate(zip(expected, actual))
                    if pair[0] != pair[1]
                ),
                min(len(expected), len(actual)),
            )
            reports.append(
                {
                    "input_tokens": len(case["input_tokens"]),
                    "output_tokens": len(actual),
                    "reference_output_tokens": len(expected),
                    "common_prefix_tokens": common,
                    "reference_greedy_exact": actual == expected,
                    "replay_exact": True,
                    "invalid_request_recovery": True,
                    "finish_reason": first["finish_reason"],
                    "native_prefill_tokens_per_second": first[
                        "input_tokens_per_second"
                    ],
                    "native_decode_tokens_per_second": first[
                        "output_tokens_per_second"
                    ],
                    "native_output_sha256": token_hash(actual),
                    "reference_output_sha256": token_hash(expected),
                }
            )
        last = first["generated_token_ids"]
    async with worker_process(
        args.generate_worker.resolve(strict=True), ["--model-path", str(model)]
    ) as worker:
        require((await worker.reply()).get("ready") is True, "restart_not_ready")
        restarted = await worker.request(request)
        require(restarted.get("generated_token_ids") == last, "restart_replay_changed")
    scoring = await score_worker(args, fixture, model)
    return {
        "passed": all(row["reference_greedy_exact"] for row in reports),
        "generation": reports,
        "restart_replay_exact": True,
        "scoring": scoring,
        "worker_sha256": digest(args.generate_worker),
        "score_worker_sha256": digest(args.score_worker),
        "model_config_sha256": digest(model / "config.json"),
    }


async def score_worker(args, fixture: Path, model: Path) -> dict:
    cases = json.loads((fixture / "cases.json").read_text())["cases"]
    selected = (1, len(cases) - 1)
    reports = []
    async with worker_process(
        args.score_worker.resolve(strict=True),
        ["--score-model-path", str(model), "--consecutive-cache"],
    ) as worker:
        require((await worker.reply()).get("ready") is True, "scorer_not_ready")
        for index in selected:
            case = cases[index]
            raw = (fixture / f"reference-{index}.f32").read_bytes()
            values = [row[0] for row in struct.iter_unpack("<f", raw)]
            candidates = sorted(
                set(
                    sorted(range(len(values)), key=values.__getitem__, reverse=True)[:8]
                    + [i * (len(values) - 1) // 7 for i in range(8)]
                )
            )
            request = {
                "operation": "score",
                "input_token_ids": case["tokens"],
                "candidate_token_ids": candidates,
                "context_tokens": case["context_tokens"],
                "temperature": 1.0,
            }
            first = await worker.request(request)
            repeat = await worker.request(request)
            require(
                "error" not in first and "error" not in repeat, "native_scoring_failed"
            )
            require(
                first.get("cache_hit") is False and repeat.get("cache_hit") is True,
                "consecutive_cache_policy_failed",
            )
            require(
                first.get("logits") == repeat.get("logits"),
                "score_cache_replay_changed",
            )
            actual = first["logits"]
            require(
                len(actual) == len(candidates)
                and all(math.isfinite(v) for v in actual),
                "score_values_invalid",
            )
            maximum = max(
                abs(v - values[token])
                for v, token in zip(actual, candidates, strict=True)
            )
            require(maximum <= 0.75, "score_reference_error_exceeds_tolerance")
            probabilities = first["candidate_probabilities"]
            require(
                all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
                and abs(sum(probabilities) - 1.0) <= 1e-12,
                "score_probability_normalization_failed",
            )
            intervening = await worker.request(
                {**request, "input_token_ids": case["tokens"][:-1]}
            )
            require(
                intervening.get("cache_hit") is False, "intervening_request_cache_hit"
            )
            replay = await worker.request(request)
            require(
                replay.get("cache_hit") is False and replay.get("logits") == actual,
                "intervening_request_did_not_invalidate_cache",
            )
            failed = await worker.request(
                {**request, "candidate_token_ids": [candidates[0], candidates[0]]}
            )
            require("error" in failed, "invalid_score_request_accepted")
            recovered = await worker.request(request)
            require(
                recovered.get("cache_hit") is False
                and recovered.get("logits") == actual,
                "score_failure_did_not_clear_cache",
            )
            reports.append(
                {
                    "input_tokens": len(case["tokens"]),
                    "candidates": len(candidates),
                    "max_abs_error": maximum,
                    "cache_replay_exact": True,
                    "intervening_request_invalidates": True,
                    "failure_clears_cache": True,
                }
            )
    return {"passed": True, "cases": reports}


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    ref = modes.add_parser("reference")
    ref.add_argument("--source", type=Path, required=True)
    ref.add_argument("--output", type=Path, required=True)
    check = modes.add_parser("workers")
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--generate-worker", type=Path, required=True)
    check.add_argument("--score-worker", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "reference":
            # Framework chatter and warnings never become a validation artifact.
            with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
                report = reference(args)
        else:
            report = asyncio.run(asyncio.wait_for(workers(args), 1800))
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 1
    except CheckFailed as failure:
        print(json.dumps({"passed": False, "check": str(failure)}))
    except (Exception, KeyboardInterrupt):  # noqa: BLE001 - never disclose native payloads or exception dumps
        print(json.dumps({"passed": False, "check": "dense_validation_failed"}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
