"""Independent dense and affine MLX LLaMBA decoder planner.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
This module does not import Lalamo, JAX, Transformers, or PyTorch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..compressed import MLXQuantization, parse_mlx_quantization, plan_mlx_matrix, read_mlx_matrix
from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
    tensor_sha256,
)
from .dense import _DTYPES, _bool, _positive_float, _positive_int
from .mamba import mamba2_config, plan_mamba2

_KEYS = {
    "model_type",
    "architectures",
    "vocab_size",
    "tie_embeddings",
    "pad_vocab_size_multiple",
    "lm_head_bias",
    "d_model",
    "n_layer",
    "resid_dropout",
    "norm_epsilon",
    "mlp_cfg",
    "ssm_cfg",
    "dtype",
    "torch_dtype",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "_name_or_path",
    "transformers_version",
    "quantization",
    "quantization_config",
}
_REQUIRED = {
    "vocab_size",
    "tie_embeddings",
    "pad_vocab_size_multiple",
    "lm_head_bias",
    "d_model",
    "n_layer",
    "resid_dropout",
    "norm_epsilon",
    "mlp_cfg",
    "ssm_cfg",
}


def build_llamba_plan(
    source: Path,
    *,
    dtype: str = "bfloat16",
    config: dict | None = None,
    expected_mlx_bits: int | None = None,
) -> ConversionPlan:
    """Plan every declared layer and preserve the complete source tensor set."""
    config = read_json(source / "config.json") if config is None else config
    if not isinstance(config, dict):
        raise ConversionError("LLaMBA model config must be an object")
    if config.get("model_type") != "llamba":
        raise UnsupportedConversion("Checkpoint model_type does not match selected adapter")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("LLaMBA supports BF16, F16, and F32 output")
    if not _REQUIRED <= set(config):
        raise ConversionError("LLaMBA config is missing required structural fields")
    model_dim = _positive_int(config, "d_model")
    layers = _positive_int(config, "n_layer")
    vocab = _positive_int(config, "vocab_size")
    vocab_multiple = _positive_int(config, "pad_vocab_size_multiple")
    epsilon = _positive_float(config, "norm_epsilon", 1e-5)
    tied = _bool(config, "tie_embeddings", True)
    head_bias = _bool(config, "lm_head_bias", False)
    dropout = config["resid_dropout"]
    if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ConversionError("LLaMBA residual dropout is malformed")
    mlp_config, ssm_config = config["mlp_cfg"], config["ssm_cfg"]
    if not isinstance(mlp_config, dict) or not isinstance(ssm_config, dict):
        raise ConversionError("LLaMBA MLP/SSM config must be an object")
    if not {"intermediate_size", "bias", "act_fn"} <= set(mlp_config):
        raise ConversionError("LLaMBA MLP config is missing structural fields")
    if not {"d_state", "n_v_heads", "n_qk_heads", "expand", "activation", "bias"} <= set(ssm_config):
        raise ConversionError("LLaMBA SSM config is missing structural fields")
    hidden_dim = _positive_int(mlp_config, "intermediate_size")
    mlp_bias = _bool(mlp_config, "bias", False)
    unsupported: list[str] = []
    if set(config) - _KEYS:
        unsupported.append("LLaMBA config options are outside implemented coverage")
    if set(mlp_config) - {"intermediate_size", "bias", "act_fn"} or mlp_config["act_fn"] != "silu":
        unsupported.append("LLaMBA requires the declared SiLU dense MLP")
    if ssm_config["activation"] != "identity":
        unsupported.append("LLaMBA requires the declared identity SSM activation")
    if head_bias:
        unsupported.append("LLaMBA biased output heads are outside implemented coverage")
    if vocab % vocab_multiple:
        unsupported.append("LLaMBA padded vocabulary dimensions are outside implemented coverage")
    architectures = config.get("architectures")
    if architectures is not None and architectures != ["LlambaForCausalLM"]:
        unsupported.append("LLaMBA architectures annotation is outside implemented coverage")
    target_dtype = _DTYPES[dtype]
    tensors = load_checkpoint(source)
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    quantization = parse_mlx_quantization(config)
    if expected_mlx_bits is not None and (type(expected_mlx_bits) is not int or expected_mlx_bits not in {4, 8}):
        raise ConversionError("LLaMBA expected MLX precision is malformed")
    if quantization is not None and expected_mlx_bits is not None and quantization.bits != expected_mlx_bits:
        raise ConversionError("LLaMBA quantization config disagrees with the selected profile")

    def packed(name):
        prefix = name.removesuffix(".weight")
        return any(prefix + suffix in tensors for suffix in (".scales", ".biases")) or (
            name in tensors and tensors[name].dtype in {"I32", "U32"}
        )

    def mlx(name, shape):
        nonlocal quantization
        if quantization is None:
            if expected_mlx_bits is None:
                raise UnsupportedConversion("LLaMBA MLX checkpoints require explicit quantization metadata")
            prefix = name.removesuffix(".weight")
            if name not in tensors or prefix + ".scales" not in tensors or prefix + ".biases" not in tensors:
                raise ConversionError("MLX checkpoint is missing a matrix plane")
            scales = tensors[prefix + ".scales"]
            if len(scales.shape) != 2 or scales.shape[1] <= 0 or shape[1] % scales.shape[1]:
                raise ConversionError("LLaMBA MLX group dimensions are malformed")
            quantization = MLXQuantization(expected_mlx_bits, shape[1] // scales.shape[1])
            # The profile supplies only the bit width. Derive the group size
            # from the declared logical matrix and validate its supported set.
            parse_mlx_quantization(
                {"quantization": {"bits": quantization.bits, "group_size": quantization.group_size}}
            )
        matrix_source = read_mlx_matrix(tensors, name, shape, quantization)
        prefix = name.removesuffix(".weight")
        expected.update((name, prefix + ".scales", prefix + ".biases"))
        return matrix_source

    def tensor(name, shape):
        expected.add(name)
        if name not in tensors:
            raise ConversionError("LLaMBA checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("LLaMBA checkpoint tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("LLaMBA checkpoint dtype is outside implemented coverage")
        return info

    def floating(out, names, shapes, shape, output_dtype):
        inputs = tuple(tensor(name, required) for name, required in zip(names, shapes, strict=True))
        if len({info.dtype for info in inputs}) > 1:
            unsupported.append("Mixed-dtype fusion promotion is outside implemented LLaMBA coverage")
        kind = "fuse" if len(inputs) > 1 else "copy"
        if any(info.dtype != output_dtype for info in inputs):
            kind = "fuse_cast" if len(inputs) > 1 else "cast"
        operations.append(TensorOperation(out, inputs, output_dtype, shape, kind))

    def matrix(out, names, shapes, shape, *, layout="output_input"):
        if any(packed(name) for name in names):
            if not all(packed(name) for name in names):
                raise UnsupportedConversion("Mixed dense/MLX fusion is outside implemented LLaMBA coverage")
            inputs = tuple(mlx(name, required) for name, required in zip(names, shapes, strict=True))
            operations.extend(
                plan_mlx_matrix(out, inputs, dtype=target_dtype, layout=layout, quantization=quantization)
            )
            metadata[out + ".spec"] = json.dumps(
                {"type": "MLXSpec", "bits": quantization.bits, "group_size": quantization.group_size, "layout": layout}
            )
            return
        floating(out + ".weights", names, shapes, shape, target_dtype)
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    if any(name.startswith("backbone.") for name in tensors):
        prefix, embedding_name, output_norm = (
            "backbone",
            "backbone.embedding.weight",
            "backbone.final_layernorm.weight",
        )
        alternating = False
        head_name = "lm_head.weight"
    elif "embedding.encoder.weight" in tensors:
        prefix, embedding_name, output_norm = "model", "embedding.encoder.weight", "model.norm.weight"
        alternating = True
        head_name = "head.linear.weight"
    else:
        raise ConversionError("LLaMBA checkpoint has an unrecognized decoder layout")
    embedding = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    matrix(embedding, (embedding_name,), ((vocab, model_dim),), (vocab, model_dim), layout="input_output")
    if tied:
        if packed(head_name):
            head = mlx(head_name, (vocab, model_dim))
            if packed(embedding_name):
                inputs = mlx(embedding_name, (vocab, model_dim))
                if any(
                    left.dtype != right.dtype or tensor_sha256(left) != tensor_sha256(right)
                    for left, right in zip(head.planes, inputs.planes, strict=True)
                ):
                    raise ConversionError("Tied MLX LLaMBA output head differs from the input embedding")
            elif not alternating:
                unsupported.append("LLaMBA packed auxiliary tied heads require the declared alternating layout")
            # The pinned alternating LLaMBA checkpoint stores a separately
            # quantized auxiliary head even though the config selects tied
            # embeddings. Its decoder consumes the dense embedding for both
            # directions. Validate the unused head's complete declared MLX
            # representation and retain its original source checks.
        elif head_name in tensors:
            head = tensor(head_name, (vocab, model_dim))
            inputs = tensors[embedding_name]
            if head.dtype != inputs.dtype or tensor_sha256(head) != tensor_sha256(inputs):
                raise ConversionError("Tied LLaMBA output head differs from the input embedding")
    else:
        matrix("decoder.embedding.output_embedding", (head_name,), ((vocab, model_dim),), (vocab, model_dim))
    layer_configs = []
    norm = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    for index in range(layers):
        mixer_layer = f"{prefix}.layers.{2 * index if alternating else index}"
        mlp_layer = f"{prefix}.layers.{2 * index + 1 if alternating else index}"
        mixer_src = mixer_layer + (".layer" if alternating else ".mixer")
        mlp_src = mlp_layer + (".layer" if alternating else ".mlp")
        out = f"decoder.transformer.layers.{index}"
        pre_mixer = mixer_layer + (".norm.weight" if alternating else ".input_layernorm.weight")
        pre_mlp = mlp_layer + (".norm.weight" if alternating else ".post_attention_layernorm.weight")
        for output, name in (("pre_mixer_norm", pre_mixer), ("pre_mlp_norm", pre_mlp)):
            floating(out + f".{output}.scales", (name,), ((model_dim,),), (model_dim,), "F32")
        if quantization is None and expected_mlx_bits is not None:
            mixer_shape = mamba2_config(model_dim, ssm_config, unsupported)
            inner = mixer_shape["num_heads"] * mixer_shape["head_dim"]
            conv = inner + 2 * mixer_shape["num_groups"] * mixer_shape["state_dim"]
            for name, shape in (
                (".in_proj.weight", (conv + inner + mixer_shape["num_heads"], model_dim)),
                (".out_proj.weight", (model_dim, inner)),
            ):
                if packed(mixer_src + name):
                    mlx(mixer_src + name, shape)
                    break
        mixer_operations, mixer_metadata, mixer_config = plan_mamba2(
            tensors,
            mixer_src,
            out + ".mixer",
            model_dim,
            ssm_config,
            target_dtype,
            expected,
            unsupported,
            quantization=quantization,
        )
        operations.extend(mixer_operations)
        metadata.update(mixer_metadata)
        up_keys = ("gate_proj", "in_proj") if alternating else ("up_proj", "gate_proj")
        down_key = "out_proj" if alternating else "down_proj"
        matrix(
            out + ".mlp.up_projection.weights",
            tuple(mlp_src + f".{key}.weight" for key in up_keys),
            ((hidden_dim, model_dim),) * 2,
            (2 * hidden_dim, model_dim),
        )
        matrix(
            out + ".mlp.down_projection.weights",
            (mlp_src + f".{down_key}.weight",),
            ((model_dim, hidden_dim),),
            (model_dim, hidden_dim),
        )
        if mlp_bias:
            floating(
                out + ".mlp.up_projection.biases",
                tuple(mlp_src + f".{key}.bias" for key in up_keys),
                ((hidden_dim,),) * 2,
                (2 * hidden_dim,),
                target_dtype,
            )
            floating(
                out + ".mlp.down_projection.biases",
                (mlp_src + f".{down_key}.bias",),
                ((model_dim,),),
                (model_dim,),
                target_dtype,
            )
        layer_configs.append(
            {
                "pre_mixer_norm_config": norm,
                "mixer_conv_config": None,
                "mixer_config": mixer_config,
                "post_mixer_norm_config": None,
                "pre_mlp_norm_config": norm,
                "mlp_conv_config": None,
                "mlp_config": {
                    "type": "DenseMLPConfig",
                    "linear_config": {},
                    "activation": {"type": "SiLU", "alpha": 1.0},
                    "has_up_biases": mlp_bias,
                    "has_down_biases": mlp_bias,
                    "up_clipping": None,
                    "gate_clipping": None,
                },
                "post_mlp_norm_config": None,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": None,
            }
        )
    floating("decoder.transformer.output_norm.scales", (output_norm,), ((model_dim,),), (model_dim,), "F32")
    if set(tensors) != expected:
        raise ConversionError("LLaMBA checkpoint has unexpected tensors")
    if (quantization is not None or expected_mlx_bits is not None) and not any(
        "MLXSpec" in value for value in metadata.values()
    ):
        raise ConversionError("LLaMBA quantization config has no packed checkpoint matrices")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    return ConversionPlan(
        {
            "embedding_config": {
                "type": "TiedEmbeddingConfig" if tied else "UntiedEmbeddingConfig",
                "input_scale": None,
                "logit_soft_cap": None,
                "logit_scale": None,
            },
            "transformer_config": {
                "layer_configs": layer_configs,
                "output_norm_config": norm,
                "model_dim": model_dim,
                "hidden_dim": hidden_dim,
            },
            "vocab_size": vocab,
            "ple_model_config": None,
            "embedding_norm_config": None,
        },
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )
