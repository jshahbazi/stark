"""Create private, source-bound Qwen3.8 native-forward parity fixtures.

Run with the repository's locked tools/stock_benchmark/mlx_current environment.
The reference is the reviewed, hash-pinned Apache-2.0 PipeNetwork model module;
it is used only by this validator, never by native model loading. The generated
fixture includes Gated DeltaNet, QSA beyond its sparse budget, MoE, n-gram PLE,
EOS segmentation and gated residual streams. Float32 execution is intentional:
it matches the native reference path's accumulation semantics, including when
the source stores its floating weights and affine scales in BF16.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SOURCE_REPO = "pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit"
SOURCE_REVISION = "b2c422f3c643e36f04227a64d61796b44a4b1029"
SOURCE_SHA256 = "6fae4ec0decbf77ca4a4571de683bc5580ec75e84325ecb432dfcd2fc81df75e"
MAX_SOURCE_BYTES = 1024 * 1024


def source_module(cache: Path):
    if cache.exists():
        if cache.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("reference source exceeds limit")
        data = cache.read_bytes()
    else:
        url = f"https://huggingface.co/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/qwen4_exp.py"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError("reference source exceeds limit")
        cache.parent.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("reference source hash mismatch")
    if not cache.exists():
        cache.write_bytes(data)
    spec = importlib.util.spec_from_file_location(
        "stark_qwen38_pinned_reference", cache
    )
    if spec is None or spec.loader is None:
        raise ValueError("reference source import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(output: Path, cache: Path, quantized: bool, bf16: bool) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx.utils import tree_flatten

    root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
    )
    if ignored.returncode != 0:
        raise ValueError("fixture destination must be ignored by git")
    if output.exists():
        raise ValueError("fixture destination already exists")
    module = source_module(cache)
    mx.set_default_device(mx.cpu)
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 3,
            "output_gate_type": "sigmoid",
            "hc_count": 2,
            "hc_lowrank": 4,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 4,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 7,
            "make_ngram_vocab_size_divisible_by": 2,
            "split_ngram_parts": 2,
            "ple_embed_dim": 16,
            "ple_layer_ids": [1],
            "ple_conv_kernel_size": 3,
            "eos_token_id": 0,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "rope_type": "default",
            },
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "attention_bias": False,
        },
        "quantization": {"group_size": 64, "bits": 4},
    }
    if quantized:
        config["text_config"].update(
            {
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "hc_lowrank": 64,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 6,
                "indexer_head_dim": 32,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64,
                "ple_embed_dim": 128,
            }
        )
    mx.random.seed(1234)
    model = module.Model(module.ModelArgs.from_dict(config))
    if bf16:
        # Module.set_dtype's default predicate keeps the I64 hash buffers exact.
        model.set_dtype(mx.bfloat16)
    if quantized:

        def predicate(path, submodule):
            if not hasattr(submodule, "to_quantized") or path.endswith(
                (
                    "mlp.gate",
                    "mlp.shared_expert_gate",
                    "block_inject_weight",
                    "in_proj_a",
                    "in_proj_b",
                    "indexer.index_qk_proj",
                )
            ):
                return False
            bits = 4 if "switch_mlp" in path or "ngram_embedding" in path else 8
            group = 32 if "ngram_embedding" in path else 64
            config["quantization"][path] = {"bits": bits, "group_size": group}
            return {"bits": bits, "group_size": group, "mode": "affine"}

        nn.quantize(model, class_predicate=predicate)
    parameters = dict(tree_flatten(model.parameters()))
    mx.eval(parameters)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, sort_keys=True))
    # Reproduce the real checkpoint's wrapper namespace. Its per-module
    # quantization descriptions deliberately remain unwrapped in config.json.
    mx.save_safetensors(
        str(output / "model.safetensors"),
        {f"language_model.{name}": value for name, value in parameters.items()},
    )
    model.set_dtype(mx.float32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12]
    cache_state = model.make_cache()
    rows = []
    for token in tokens:
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache_state)
        mx.eval(logits)
        rows.append(np.asarray(logits)[0, -1])
    values = np.stack(rows).astype("<f4")
    if not np.isfinite(values).all():
        raise ValueError("reference produced invalid values")
    reference = values.tobytes()
    (output / "tokens.json").write_text(json.dumps(tokens))
    (output / "reference.f32").write_bytes(reference)
    receipt = {
        "schema_version": 1,
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_module_sha256": SOURCE_SHA256,
        "source_float_dtype": "bfloat16" if bf16 else "float32",
        "execution_dtype": "float32",
        "quantization": "mixed-4_8bit" if quantized else "floating-control",
        "positions": len(tokens),
        "vocabulary": int(values.shape[1]),
        "weights_sha256": hashlib.sha256(
            (output / "model.safetensors").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(reference).hexdigest(),
    }
    (output / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format", choices=("float32", "mixed-4_8bit"), default="mixed-4_8bit"
    )
    parser.add_argument(
        "--source-float-dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--reference-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runtime/native-hybrid-reference/qwen4_exp.py",
    )
    args = parser.parse_args()
    try:
        receipt = fixture(
            args.output.resolve(),
            args.reference_source.resolve(),
            args.format == "mixed-4_8bit",
            args.source_float_dtype == "bfloat16",
        )
    except Exception:  # noqa: BLE001 - never expose private/native reference exception content.
        print(
            json.dumps({"status": "failed", "error": "native_reference_fixture_failed"})
        )
        return 1
    print(json.dumps({"status": "ready", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
