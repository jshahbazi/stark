#!/usr/bin/env python3
"""Private MLX-LM reference fixtures for Phi3 checkpoints with LongRoPE.

Run with the locked ``tools/stock_benchmark/mlx_current`` environment (MLX-LM
72be8c955d9a0d3099bd01865af288e0e947c3fa, MIT, Apple). Its ``SuScaledRoPE``
always applies the long factors, which is the static selection the native
``LongRoPEConfig`` makes when the declared context exceeds the original one.
Hugging Face Transformers instead switches to the short factors up to the
original context, so this reference matches the implemented native semantics,
not the dynamic Transformers ones.

The output directory has the layout ``scripts/dense_native_validation.py
workers`` reads: per-case F32 last-token logits, ``cases.json`` and
``generations.json``. They contain prompts, token sequences and logits and are
private; keep them under an ignored directory. Only aggregates and hashes are
printed. No remote model code is executed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import struct
import time
from pathlib import Path

MLX_LM_REVISION = "72be8c955d9a0d3099bd01865af288e0e947c3fa"
MLX_VERSION = "0.32.4"
TRANSFORMERS_VERSION = "5.20.1"
CHUNK = 1024
LONG_CASE_TOKENS = CHUNK + 5
OUTPUT_TOKENS = 32


class CheckFailed(Exception):
    pass


def require(condition: bool, check: str) -> None:
    if not condition:
        raise CheckFailed(check)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            value.update(block)
    return value.hexdigest()


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
        importlib.metadata.version("mlx") == MLX_VERSION,
        "reference_mlx_version_changed",
    )
    require(
        importlib.metadata.version("transformers") == TRANSFORMERS_VERSION,
        "reference_transformers_changed",
    )
    source = args.source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    scaling = config.get("rope_scaling") or {}
    require(config.get("model_type") == "phi3", "reference_not_phi3")
    require(
        scaling.get("type", scaling.get("rope_type")) == "longrope",
        "reference_requires_longrope",
    )
    require(
        config["max_position_embeddings"] > config["original_max_position_embeddings"],
        "reference_requires_long_factor_selection",
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    model, config = load_model(source, lazy=True, strict=True, trust_remote_code=False)
    tied = config.get("tie_word_embeddings", False)
    require(hasattr(model, "lm_head") is (not tied), "reference_head_layout_mismatch")
    tokenizer = load_tokenizer(source)
    mx.eval(model.parameters())
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in (
            "Count from one to five, separated by commas.",
            "Write a Python function that adds two integers. Return only the code.",
        )
    ]
    prompts = [tokenizer.encode(value, add_special_tokens=False) for value in rendered]
    eos = config.get("eos_token_id")
    eos = {eos} if type(eos) is int else set(eos or [])
    generation_path = source / "generation_config.json"
    if generation_path.exists():
        extra = json.loads(generation_path.read_text()).get("eos_token_id", [])
        eos.update([extra] if type(extra) is int else extra)
    eos.update(tokenizer.eos_token_ids)

    def prefill(tokens):
        cache = make_prompt_cache(model)
        started = time.perf_counter()
        # Same profile as dense_native_validation: cache the prefix in 1,024-token
        # chunks, then read one final token through the decode route.
        prefix = tokens[:-1]
        for offset in range(0, len(prefix), CHUNK):
            result = model(
                mx.array([prefix[offset : offset + CHUNK]], dtype=mx.int32), cache=cache
            )
            mx.eval(result)
        result = model(mx.array([tokens[-1:]], dtype=mx.int32), cache=cache)
        mx.eval(result)
        return result[0, -1], cache, time.perf_counter() - started

    cases = []
    base = prompts[0]
    repeated = (base * (LONG_CASE_TOKENS // len(base) + 1))[:LONG_CASE_TOKENS]
    sequences = [base[:1], base, prompts[1], repeated[:129], repeated]
    case_rows = []
    for index, tokens in enumerate(sequences):
        logits, _cache, elapsed = prefill(tokens)
        values = np.asarray(logits.astype(mx.float32), dtype="<f4")
        require(bool(np.isfinite(values).all()), "reference_logits_nonfinite")
        path = output / f"reference-{index}.f32"
        path.write_bytes(values.tobytes())
        path.chmod(0o600)
        cases.append({"tokens": tokens, "context_tokens": max(CHUNK, len(tokens) + 32)})
        case_rows.append(
            {
                "case": index,
                "input_tokens": len(tokens),
                "seconds": elapsed,
                "reference_sha256": digest(path),
            }
        )
    save_json(output / "cases.json", {"cases": cases})
    generations = []
    for prompt, tokens in zip(rendered, prompts, strict=True):
        logits, cache, prefill_seconds = prefill(tokens)
        generated = []
        started = time.perf_counter()
        sampled = 0
        for _ in range(OUTPUT_TOKENS):
            token = int(mx.argmax(logits).item())
            sampled += 1
            if token in eos:
                break
            generated.append(token)
            if len(generated) == OUTPUT_TOKENS:
                break
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1]
            mx.eval(logits)
        generations.append(
            {
                "rendered_prompt": prompt,
                "input_tokens": tokens,
                "reference_tokens": generated,
                "max_new_tokens": OUTPUT_TOKENS,
                "context_tokens": CHUNK,
                "reference_sampled_tokens": sampled,
                "reference_prefill_seconds": prefill_seconds,
                "reference_decode_seconds": time.perf_counter() - started,
            }
        )
    save_json(output / "generations.json", generations)
    weights = sorted(source.glob("*.safetensors"))
    report = {
        "passed": True,
        "cases": case_rows,
        "generation_cases": [
            {
                "input_tokens": len(row["input_tokens"]),
                "reference_output_tokens": len(row["reference_tokens"]),
                "reference_sampled_tokens": row["reference_sampled_tokens"],
                "reference_output_sha256": token_hash(row["reference_tokens"]),
            }
            for row in generations
        ],
        "tied_embeddings": tied,
        "source_config_sha256": digest(source / "config.json"),
        "source_tokenizer_sha256": digest(source / "tokenizer.json"),
        "source_tokenizer_config_sha256": digest(source / "tokenizer_config.json"),
        "source_weights_sha256": {path.name: digest(path) for path in weights},
        "mlx_lm_revision": MLX_LM_REVISION,
        "mlx": MLX_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "reference_dtype": "bfloat16 weights and activations as loaded by MLX-LM",
        "forward_profile": "prefix-cache-1024-then-last-token-v2",
        "rope_factor_selection": "long factors always (SuScaledRoPE)",
        "cases_sha256": digest(output / "cases.json"),
        "generations_sha256": digest(output / "generations.json"),
    }
    save_json(output / "reference-report.json", report)
    return report


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Framework chatter and warnings never become a validation artifact.
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
            report = reference(args)
        print(json.dumps(report, sort_keys=True))
        return 0
    except CheckFailed as failure:
        print(json.dumps({"passed": False, "check": str(failure)}))
    except (Exception, KeyboardInterrupt):  # noqa: BLE001 - never disclose native payloads or exception dumps
        print(json.dumps({"passed": False, "check": "phi3_reference_failed"}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
