"""Create a private synthetic OLMo 3 model and independent NumPy logit oracle.

The equations follow Transformers v4.57.1's Apache-2.0 OLMo 3 definition:
whole-projection Q/K RMS, branch post-normalization, three sliding attention
layers followed by full attention, and YaRN only on the full-attention layer.
Weights, token sequences and logits remain under a Git-ignored destination.
Only aggregate dimensions and hashes are printed.

Run with the locked tools/converter uv environment, then pass its ``model`` and
``fixture`` directories to the ``dense_parity`` Rust example. This is structural
validation; it does not replace real-checkpoint conversion/generation checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
from stark_convert.adapters.olmo import build_olmo3_plan
from stark_convert.streaming import write_plan
from tokenizers import Tokenizer
from tokenizers.models import WordLevel


def save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, payload = {}, bytearray()
    for name, value in sorted(tensors.items()):
        value = value.astype("<f4")
        start = len(payload)
        payload.extend(value.tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def rms(value: np.ndarray, scale: np.ndarray, epsilon: float) -> np.ndarray:
    return (
        value
        * np.reciprocal(
            np.sqrt(np.mean(np.square(value), axis=-1, keepdims=True) + epsilon)
        )
        * scale
    )


def bf16(value: np.ndarray) -> np.ndarray:
    """Round a reference operator boundary to native BF16 storage."""
    return np.asarray(value).astype(ml_dtypes.bfloat16).astype(np.float32)


def rotate(value: np.ndarray, config: dict, *, global_attention: bool) -> np.ndarray:
    length, _, width = value.shape
    indices = np.arange(width // 2, dtype=np.float32)
    frequency = np.reciprocal(
        np.power(np.float32(config["rope_theta"]), 2 * indices / width)
    )
    attention = 1.0
    if global_attention:
        scaling = config["rope_scaling"]
        original, base = (
            scaling["original_max_position_embeddings"],
            config["rope_theta"],
        )
        low = max(
            0,
            math.floor(
                width
                * math.log(original / (scaling["beta_fast"] * 2 * math.pi))
                / (2 * math.log(base))
            ),
        )
        high = min(
            width - 1,
            math.ceil(
                width
                * math.log(original / (scaling["beta_slow"] * 2 * math.pi))
                / (2 * math.log(base))
            ),
        )
        ramp = np.clip((indices - low) / (high - low if high != low else 0.001), 0, 1)
        frequency = frequency * (1 - ramp) + (frequency / scaling["factor"]) * ramp
        attention = scaling["attention_factor"]
    phase = np.arange(length, dtype=np.float32)[:, None] * frequency[None]
    cosine = np.cos(phase)[:, None] * attention
    sine = np.sin(phase)[:, None] * attention
    left, right = value[..., : width // 2], value[..., width // 2 :]
    return bf16(
        np.concatenate(
            (left * cosine - right * sine, right * cosine + left * sine), axis=-1
        )
    )


def reference(
    tokens: list[int], weights: dict[str, np.ndarray], config: dict
) -> np.ndarray:
    hidden = weights["model.embed_tokens.weight"][tokens]
    length, heads, groups = (
        len(tokens),
        config["num_attention_heads"],
        config["num_key_value_heads"],
    )
    head_dim = config["hidden_size"] // heads
    epsilon = config["rms_norm_eps"]
    query_positions = np.arange(length)[:, None]
    key_positions = np.arange(length)[None, :]
    for layer, kind in enumerate(config["layer_types"]):
        src = f"model.layers.{layer}"
        query = bf16(
            rms(
                bf16(hidden @ weights[src + ".self_attn.q_proj.weight"].T),
                weights[src + ".self_attn.q_norm.weight"],
                epsilon,
            )
        )
        key = bf16(
            rms(
                bf16(hidden @ weights[src + ".self_attn.k_proj.weight"].T),
                weights[src + ".self_attn.k_norm.weight"],
                epsilon,
            )
        )
        value = bf16(hidden @ weights[src + ".self_attn.v_proj.weight"].T).reshape(
            length, groups, head_dim
        )
        query = rotate(
            query.reshape(length, heads, head_dim),
            config,
            global_attention=kind == "full_attention",
        )
        key = rotate(
            key.reshape(length, groups, head_dim),
            config,
            global_attention=kind == "full_attention",
        )
        key = np.repeat(key, heads // groups, axis=1)
        value = np.repeat(value, heads // groups, axis=1)
        scores = np.einsum("thd,shd->hts", query, key) / math.sqrt(head_dim)
        mask = key_positions <= query_positions
        if kind == "sliding_attention":
            mask &= (query_positions - key_positions) < config["sliding_window"]
        scores = np.where(mask[None], scores, -np.inf)
        probability = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probability /= np.sum(probability, axis=-1, keepdims=True)
        attention = bf16(
            np.einsum("hts,shd->thd", probability, value).reshape(length, -1)
        )
        attention = bf16(attention @ weights[src + ".self_attn.o_proj.weight"].T)
        hidden = bf16(
            hidden
            + bf16(
                rms(
                    attention,
                    weights[src + ".post_attention_layernorm.weight"],
                    epsilon,
                )
            )
        )
        gate = bf16(hidden @ weights[src + ".mlp.gate_proj.weight"].T)
        up = bf16(hidden @ weights[src + ".mlp.up_proj.weight"].T)
        output = bf16(
            bf16(bf16(gate / (1 + np.exp(-gate))) * up)
            @ weights[src + ".mlp.down_proj.weight"].T
        )
        hidden = bf16(
            hidden
            + bf16(
                rms(
                    output, weights[src + ".post_feedforward_layernorm.weight"], epsilon
                )
            )
        )
    return bf16(
        bf16(rms(hidden[-1], weights["model.norm.weight"], epsilon))
        @ weights["lm_head.weight"].T
    ).astype("<f4")


def fixture(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    if subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
    ).returncode:
        raise ValueError("fixture output must be ignored")
    if output.exists():
        raise ValueError("fixture output already exists")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    source, model, cases_dir = [
        output / name for name in ("source", "model", "fixture")
    ]
    for directory in (source, model, cases_dir):
        directory.mkdir(mode=0o700)
    # Use a compiled native attention width (64); head_dim=32 is only useful
    # for converter tests and has no ordinary Metal attention specialization.
    hidden, intermediate, vocab, layers, heads, groups = 256, 384, 64, 4, 4, 2
    head_dim = hidden // heads
    config = {
        "model_type": "olmo3",
        "architectures": ["Olmo3ForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": groups,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0,
        "max_position_embeddings": 1024,
        "sliding_window": 8,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1 + 0.1 * math.log(8),
        },
        "tie_word_embeddings": False,
        "attention_bias": False,
        "eos_token_id": 0,
    }
    rng = np.random.default_rng(20261028)
    weights = {}

    def add(name, shape, norm=False):
        weights[name] = (
            rng.uniform(0.6, 1.4, shape) if norm else rng.normal(0, 0.06, shape)
        ).astype("<f4")

    add("model.embed_tokens.weight", (vocab, hidden))
    add("lm_head.weight", (vocab, hidden))
    add("model.norm.weight", (hidden,), norm=True)
    for layer in range(layers):
        src = f"model.layers.{layer}"
        for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            add(src + f".{name}.weight", (hidden,), norm=True)
        for name, width in (
            ("q", heads * head_dim),
            ("k", groups * head_dim),
            ("v", groups * head_dim),
        ):
            add(src + f".self_attn.{name}_proj.weight", (width, hidden))
            if name != "v":
                add(src + f".self_attn.{name}_norm.weight", (width,), norm=True)
        add(src + ".self_attn.o_proj.weight", (hidden, heads * head_dim))
        add(src + ".mlp.up_proj.weight", (intermediate, hidden))
        add(src + ".mlp.gate_proj.weight", (intermediate, hidden))
        add(src + ".mlp.down_proj.weight", (hidden, intermediate))
    (source / "config.json").write_text(json.dumps(config))
    save_safetensors(source / "model.safetensors", weights)
    # Native language-model execution currently fixes activations to BF16.
    plan = build_olmo3_plan(source, dtype="bfloat16")
    write_plan(plan, model / "model.safetensors", chunk_bytes=65536)
    generation = {
        key: None
        for key in (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "banned_tokens",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "suffix_repetition_length",
        )
    }
    generation["stop_token_ids"] = [0]
    (model / "config.json").write_text(
        json.dumps(
            {
                "type": "LanguageModelConfig",
                "token_codec_config": {"type": "RawTextCodecConfig"},
                "generation_config": generation,
                "decoder_config": plan.decoder_config,
            }
        )
    )
    Tokenizer(
        WordLevel({f"t{index}": index for index in range(vocab)}, unk_token="t0")
    ).save(str(model / "tokenizer.json"))
    cases, receipts = [], []
    for index, length in enumerate((1, 7, 8, 9, 17, 129, 257)):
        tokens = rng.integers(1, vocab, length).tolist()
        reference_weights = {
            name: bf16(value) if value.ndim == 2 else value
            for name, value in weights.items()
        }
        values = reference(tokens, reference_weights, config)
        if not np.isfinite(values).all():
            raise ValueError("invalid reference")
        raw = values.tobytes()
        (cases_dir / f"reference-{index}.f32").write_bytes(raw)
        cases.append({"tokens": tokens, "context_tokens": 1024})
        receipts.append(
            {
                "input_tokens": length,
                "reference_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    (cases_dir / "cases.json").write_text(json.dumps({"cases": cases}))
    report = {
        "schema_version": 1,
        "architecture": "olmo3",
        "source": "synthetic-seeded-f32",
        "native_dtype": "bfloat16",
        "reference_dtype": "bfloat16-operator-boundaries-f32-accumulation",
        "cases": receipts,
        "weights_sha256": hashlib.sha256(
            (source / "model.safetensors").read_bytes()
        ).hexdigest(),
    }
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = fixture(args.output.resolve())
    except Exception:  # noqa: BLE001 - private model/reference data must never enter exception output.
        print(json.dumps({"status": "failed", "error": "olmo_fixture_failed"}))
        return 1
    print(json.dumps({"status": "ready", **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
