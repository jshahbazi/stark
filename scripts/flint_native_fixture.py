"""Create private, source-bound Flint native-forward parity fixtures.

Run with the repository's locked tools/stock_benchmark/mlx_current environment.
The Apache-2.0 Recursive reference is hash-pinned and used only by this validator.
Only its model module executes: its sole relative dependency is replaced with a
local cancellation exception because validation never supplies should_stop.
The small model covers dense and routed/shared experts, nonzero attention sinks,
expert-selection bias, full attention without RoPE, and sliding attention with
split-half RoPE beyond the window. Float32 execution matches native reference
accumulation, including when stored floating weights/scales use BF16.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

SOURCE_REPO = "recursive-oss/rec"
SOURCE_REVISION = "0e3150ca4262f572dee25c231d9ac7148414d2af"
SOURCE_PATH = "engine/src/rec_mlx/model.py"
SOURCE_SHA256 = "56363f7a1caea2b9db317feb192f4bfea32eccb2b6b190dffc3fa93ee09695b1"
MAX_SOURCE_BYTES = 1024 * 1024


def source_module(cache: Path):
    if cache.exists():
        if cache.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("reference source exceeds limit")
        data = cache.read_bytes()
    else:
        url = (
            f"https://raw.githubusercontent.com/{SOURCE_REPO}/"
            f"{SOURCE_REVISION}/{SOURCE_PATH}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError("reference source exceeds limit")
        cache.parent.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("reference source hash mismatch")
    if not cache.exists():
        cache.write_bytes(data)
    package_name = "stark_flint_pinned_reference"
    package = types.ModuleType(package_name)
    package.__path__ = []
    dependency = types.ModuleType(f"{package_name}.prefix")

    class PrefillCancelled(Exception):
        pass

    dependency.PrefillCancelled = PrefillCancelled
    sys.modules[package_name] = package
    sys.modules[dependency.__name__] = dependency
    spec = importlib.util.spec_from_file_location(f"{package_name}.model", cache)
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
    from mlx_lm.models.cache import KVCache, RotatingKVCache

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
        "architectures": ["FlintForCausalLM"],
        "model_type": "flint",
        "hidden_size": 16,
        "vocab_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "layer_types": [
            "full_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        "no_rope_layer_indices": [0, 1, 3],
        "mlp_only_layers": [0, 1],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "num_experts": 4,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": True,
        "router_score_function": "sigmoid",
        "scoring_func": "sigmoid",
        # The public checkpoint omits this optional bias. A nonzero fixture
        # exercises its selection-only semantics independently of that export.
        "router_expert_bias": True,
        "decoder_sparse_step": 1,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_sink": True,
        "softmax_type": "learnable",
        "sliding_window": 5,
        "use_sliding_window": True,
        "rope_theta": 10000.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "use_cache": True,
        "bos_token_id": 30,
        "eos_token_id": [31],
        "pad_token_id": 31,
        "torch_dtype": "bfloat16" if bf16 else "float32",
        "dtype": "bfloat16" if bf16 else "float32",
    }
    if quantized:
        config.update(
            hidden_size=64,
            head_dim=32,
            intermediate_size=128,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            quantization={"group_size": 64, "bits": 4},
            mlx_export={
                "bits": 4,
                "embedding_bits": 8,
                "routed_expert_bits": 4,
                "full_attention_layers": [0, 1, 3],
            },
        )
    mx.random.seed(1234)
    model = module.Model(module.ModelArgs.from_dict(config))
    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        attention.softmax_offset = mx.array([-0.9, 0.6, 1.4, 2.2]) + index * 0.05
        attention.q_norm.weight = mx.linspace(0.7, 1.3, config["head_dim"])
        attention.k_norm.weight = mx.linspace(1.2, 0.8, config["head_dim"])
        if isinstance(layer.mlp, module.SparseMoeBlock):
            layer.mlp.gate.weight = mx.random.normal(
                (config["num_experts"], config["hidden_size"])
            ) * (0.7 / config["hidden_size"] ** 0.5)
            layer.mlp.gate.expert_bias = mx.array([0.25, -0.15, 0.05, 0.4])
    if bf16:
        model.set_dtype(mx.bfloat16)
    if quantized:

        def predicate(path, submodule):
            if not hasattr(submodule, "to_quantized"):
                return False
            bits = 8 if path == "model.embed_tokens" else 4
            return {"bits": bits, "group_size": 64, "mode": "affine"}

        nn.quantize(model, class_predicate=predicate)
    parameters = dict(tree_flatten(model.parameters()))
    mx.eval(parameters)
    output.mkdir(parents=True, mode=0o700)
    (output / "config.json").write_text(json.dumps(config, sort_keys=True))
    # Official MLX Flint serializes its packed uint32 arrays as U8 byte views.
    mx.save_safetensors(
        str(output / "model.safetensors"),
        {
            name: value.view(mx.uint8)
            if name.endswith(".weight") and value.dtype == mx.uint32
            else value
            for name, value in parameters.items()
        },
    )
    model.set_dtype(mx.float32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12]
    # Explicit floating caches isolate model-forward parity from rec's optional
    # quantized KV policy. Native inference retains these states in float32.
    cache_state = [
        KVCache()
        if kind == "full_attention"
        else RotatingKVCache(max_size=config["sliding_window"], keep=0)
        for kind in config["layer_types"]
    ]
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
        "source_path": SOURCE_PATH,
        "source_module_sha256": SOURCE_SHA256,
        "relative_dependency": "local-cancellation-exception-stub",
        "source_float_dtype": "bfloat16" if bf16 else "float32",
        "execution_dtype": "float32",
        "kv_cache_dtype": "float32",
        "quantization": "mlx-affine-4bit-embedding8-u8"
        if quantized
        else "floating-control",
        "fixture_variant": "nonzero-router-selection-bias",
        "sliding_window": config["sliding_window"],
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
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument(
        "--source-float-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--reference-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runtime/native-hybrid-reference/flint_model.py",
    )
    args = parser.parse_args()
    try:
        receipt = fixture(
            args.output.resolve(),
            args.reference_source.resolve(),
            args.quantized,
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
