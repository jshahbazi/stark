"""Independent dense LFM2 short-convolution/attention planner.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
This module does not import Lalamo, JAX, Transformers, or PyTorch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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

_KEYS = {
    "architectures",
    "model_type",
    "block_auto_adjust_ff_dim",
    "block_dim",
    "block_ff_dim",
    "block_ffn_dim_multiplier",
    "block_mlp_init_scale",
    "block_multiple_of",
    "block_norm_eps",
    "block_out_init_scale",
    "block_use_swiglu",
    "block_use_xavier_init",
    "bos_token_id",
    "conv_L_cache",
    "conv_bias",
    "conv_dim",
    "conv_dim_out",
    "conv_use_xavier_init",
    "eos_token_id",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "norm_eps",
    "num_attention_heads",
    "num_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "transformers_version",
    "use_cache",
    "use_pos_enc",
    "vocab_size",
    "dtype",
    "torch_dtype",
    "layer_types",
    "full_attn_idxs",
    "tie_embedding",
    "theta",
    "rope_theta",
    "rope_parameters",
    "quantization",
    "quantization_config",
    "_name_or_path",
}


def build_lfm2_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Validate an explicit dense LFM2 topology and consume every input tensor."""
    config = read_json(source / "config.json")
    if config.get("model_type") != "lfm2":
        raise UnsupportedConversion("Checkpoint model_type does not match selected adapter")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("LFM2 supports BF16, F16, and F32 output")
    required = {
        "block_auto_adjust_ff_dim",
        "block_use_swiglu",
        "use_pos_enc",
        "conv_bias",
        "block_ffn_dim_multiplier",
        "block_norm_eps",
        "norm_eps",
    }
    if not required <= set(config):
        raise ConversionError("LFM2 config is missing required structural fields")
    target_dtype = _DTYPES[dtype]
    unsupported: list[str] = []
    model_dim = _positive_int(config, "hidden_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads")
    if heads != _positive_int(config, "num_heads") or heads % groups or model_dim % heads:
        raise ConversionError("LFM2 attention dimensions disagree")
    head_dim = model_dim // heads
    if head_dim % 2:
        raise ConversionError("LFM2 rotary head dimension must be even")
    vocab = _positive_int(config, "vocab_size")
    sequence_length = _positive_int(config, "max_position_embeddings")
    kernel = _positive_int(config, "conv_L_cache")
    if kernel < 2:
        unsupported.append("LFM2 convolution kernels must contain at least two elements")
    for key in ("block_dim", "conv_dim"):
        if _positive_int(config, key) != model_dim:
            unsupported.append("LFM2 block/convolution dimensions disagree with hidden_size")
    if config.get("conv_dim_out") is not None and _positive_int(config, "conv_dim_out") != model_dim:
        unsupported.append("LFM2 convolution output dimension is outside implemented coverage")
    epsilon = _positive_float(config, "block_norm_eps", 1e-5)
    output_epsilon = _positive_float(config, "norm_eps", 1e-5)
    tied = _bool(config, "tie_embedding", True)
    conv_bias = _bool(config, "conv_bias", False)
    use_swiglu = _bool(config, "block_use_swiglu", True)
    use_pos_enc = _bool(config, "use_pos_enc", True)
    if not use_swiglu or not use_pos_enc:
        unsupported.append("LFM2 requires SwiGLU and positional embeddings")
    for key in ("block_use_xavier_init", "conv_use_xavier_init", "use_cache"):
        _bool(config, key, False)
    for key in ("block_mlp_init_scale", "block_out_init_scale", "initializer_range"):
        if key in config:
            value = config[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ConversionError("LFM2 initialization metadata is malformed")
    ff_dim = _positive_int(config, "block_ff_dim")
    multiplier = _positive_float(config, "block_ffn_dim_multiplier", 1.0)
    multiple = _positive_int(config, "block_multiple_of")
    auto_adjust = _bool(config, "block_auto_adjust_ff_dim", False)
    if config.get("intermediate_size") is not None:
        _positive_int(config, "intermediate_size")
    if auto_adjust:
        adjusted = ff_dim * multiplier * (2 / 3)
        if not math.isfinite(adjusted):
            raise ConversionError("LFM2 adjusted intermediate dimension is invalid")
        hidden_dim = int((adjusted + multiple - 1) // multiple * multiple)
        if hidden_dim <= 0:
            raise ConversionError("LFM2 adjusted intermediate dimension is invalid")
    else:
        hidden_dim = config.get("intermediate_size") or ff_dim
    indices = config.get("full_attn_idxs")
    if indices is not None:
        if (
            not isinstance(indices, list)
            or any(type(index) is not int or not 0 <= index < layers for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ConversionError("LFM2 full_attn_idxs is malformed")
    layer_types = config.get("layer_types")
    if layer_types is not None:
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(layer not in ("conv", "full_attention") for layer in layer_types)
        ):
            raise ConversionError("LFM2 layer_types is malformed")
        if indices is not None and set(indices) != {
            index for index, layer in enumerate(layer_types) if layer == "full_attention"
        }:
            unsupported.append("LFM2 layer_types contradicts full_attn_idxs")
    elif indices is not None:
        layer_types = ["full_attention" if index in indices else "conv" for index in range(layers)]
    else:
        raise ConversionError("LFM2 requires explicit layer_types or full_attn_idxs")
    nested_rope = config.get("rope_parameters")
    if nested_rope is not None:
        if not isinstance(nested_rope, dict):
            raise ConversionError("LFM2 rope_parameters is malformed")
        nested_base = _positive_float(nested_rope, "rope_theta", 10000.0)
        if "rope_theta" not in nested_rope:
            raise ConversionError("LFM2 rope_parameters requires rope_theta")
        if set(nested_rope) - {"rope_theta", "rope_type"} or nested_rope.get("rope_type", "default") != "default":
            unsupported.append("LFM2 scaled/custom rotary embeddings are outside implemented coverage")
    else:
        nested_base = None
    flat_base = None if config.get("rope_theta") is None else _positive_float(config, "rope_theta", 10000.0)
    if flat_base is None and nested_base is None:
        raise ConversionError("LFM2 requires rope_theta or rope_parameters")
    if flat_base is not None and nested_base is not None and flat_base != nested_base:
        raise ConversionError("LFM2 rotary embedding bases disagree")
    base = nested_base if nested_base is not None else flat_base
    if config.get("theta") is not None:
        _positive_float(config, "theta", 10000.0)
    if config.get("architectures", ["Lfm2ForCausalLM"]) != ["Lfm2ForCausalLM"] or set(config) - _KEYS:
        unsupported.append("LFM2 config options are outside implemented coverage")
    for key in ("quantization", "quantization_config"):
        if config.get(key) is not None:
            if not isinstance(config[key], dict) or not config[key]:
                raise ConversionError("LFM2 quantization parameters are malformed")
            raise UnsupportedConversion("LFM2 compressed checkpoints are outside implemented coverage")
    tensors = load_checkpoint(source)
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}

    def tensor(name: str, shapes: tuple[tuple[int, ...], ...]):
        expected.add(name)
        if name not in tensors:
            raise ConversionError("LFM2 checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape not in shapes:
            raise ConversionError("LFM2 checkpoint tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("LFM2 checkpoint dtype is outside implemented coverage")
        return info

    def select(names: tuple[str, ...]):
        present = [name for name in names if name in tensors]
        if len(present) != 1:
            raise ConversionError("LFM2 checkpoint has missing or ambiguous parameter aliases")
        return present[0]

    def matrix(out, names, shapes, shape, *, layout="output_input"):
        inputs = tuple(tensor(name, (required,)) for name, required in zip(names, shapes, strict=True))
        if len({info.dtype for info in inputs}) > 1:
            unsupported.append("Mixed-dtype fusion promotion is outside implemented LFM2 coverage")
        kind = "fuse" if len(inputs) > 1 else "copy"
        if any(info.dtype != target_dtype for info in inputs):
            kind = "fuse_cast" if len(inputs) > 1 else "cast"
        operations.append(TensorOperation(out + ".weights", inputs, target_dtype, shape, kind))
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def f32(out, name, shape, *, source_shapes=None):
        info = tensor(name, source_shapes or (shape,))
        operations.append(TensorOperation(out, (info,), "F32", shape, "cast_f32"))

    embedding = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    matrix(embedding, ("model.embed_tokens.weight",), ((vocab, model_dim),), (vocab, model_dim), layout="input_output")
    if tied:
        if "lm_head.weight" in tensors:
            head = tensor("lm_head.weight", ((vocab, model_dim),))
            inputs = tensors["model.embed_tokens.weight"]
            if head.dtype != inputs.dtype or tensor_sha256(head) != tensor_sha256(inputs):
                raise ConversionError("Tied LFM2 lm_head differs from the input embedding")
    else:
        matrix("decoder.embedding.output_embedding", ("lm_head.weight",), ((vocab, model_dim),), (vocab, model_dim))
    q_width, kv_width = heads * head_dim, groups * head_dim
    for layer, layer_type in enumerate(layer_types):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        f32(out + ".pre_mixer_norm.scales", src + ".operator_norm.weight", (model_dim,))
        f32(out + ".pre_mlp_norm.scales", src + ".ffn_norm.weight", (model_dim,))
        if layer_type == "conv":
            matrix(
                out + ".mixer.in_projection.weights",
                (src + ".conv.in_proj.weight",),
                ((3 * model_dim, model_dim),),
                (3 * model_dim, model_dim),
            )
            matrix(
                out + ".mixer.out_projection.weights",
                (src + ".conv.out_proj.weight",),
                ((model_dim, model_dim),),
                (model_dim, model_dim),
            )
            conv_name = select(
                tuple(src + suffix for suffix in (".conv.conv1d.weight", ".conv.conv_weight", ".conv.conv.weight"))
            )
            f32(
                out + ".mixer.conv.weights",
                conv_name,
                (model_dim, kernel),
                source_shapes=((model_dim, kernel), (model_dim, 1, kernel), (model_dim, kernel, 1)),
            )
            if conv_bias:
                bias_name = select(
                    tuple(src + suffix for suffix in (".conv.conv1d.bias", ".conv.conv_bias", ".conv.conv.bias"))
                )
                f32(out + ".mixer.conv.biases", bias_name, (model_dim,))
        else:
            matrix(
                out + ".mixer.qkvg_projection.weights",
                tuple(src + f".self_attn.{name}_proj.weight" for name in ("q", "k", "v")),
                ((q_width, model_dim), (kv_width, model_dim), (kv_width, model_dim)),
                (q_width + 2 * kv_width, model_dim),
            )
            projection = select((src + ".self_attn.o_proj.weight", src + ".self_attn.out_proj.weight"))
            matrix(out + ".mixer.out_projection.weights", (projection,), ((model_dim, q_width),), (model_dim, q_width))
            for side, export in (("q", "query"), ("k", "key")):
                name = select((src + f".self_attn.{side}_norm.weight", src + f".self_attn.{side}_layernorm.weight"))
                f32(out + f".mixer.{export}_norm.scales", name, (head_dim,))
        matrix(
            out + ".mlp.up_projection.weights",
            (src + ".feed_forward.w3.weight", src + ".feed_forward.w1.weight"),
            ((hidden_dim, model_dim), (hidden_dim, model_dim)),
            (2 * hidden_dim, model_dim),
        )
        matrix(
            out + ".mlp.down_projection.weights",
            (src + ".feed_forward.w2.weight",),
            ((model_dim, hidden_dim),),
            (model_dim, hidden_dim),
        )
    f32("decoder.transformer.output_norm.scales", "model.embedding_norm.weight", (model_dim,))
    if set(tensors) != expected:
        raise ConversionError("LFM2 checkpoint has unexpected tensors")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])

    def norm(eps):
        return {
            "epsilon": eps,
            "scale_offset": None,
            "upcast_mode": "only_normalization",
            "subtract_mean": False,
            "has_scale": True,
            "has_biases": False,
        }

    block_norm = norm(epsilon)
    mlp = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": {"type": "SiLU", "alpha": 1.0},
        "has_up_biases": False,
        "has_down_biases": False,
        "gate_clipping": None,
        "up_clipping": None,
    }
    rope = {"type": "UnscaledRoPEConfig", "base": base, "max_sequence_length": sequence_length, "head_dim": head_dim}
    layer_configs = []
    for layer_type in layer_types:
        if layer_type == "conv":
            mixer = {
                "type": "ShortConvConfig",
                "in_projection_config": {},
                "conv_config": {"has_biases": conv_bias},
                "out_projection_config": {},
                "kernel_size": kernel,
            }
        else:
            mixer = {
                "type": "AttentionConfig",
                "qkvg_projection_config": {},
                "out_projection_config": {},
                "query_norm_config": block_norm,
                "key_norm_config": block_norm,
                "num_heads": heads,
                "num_groups": groups,
                "head_dim": head_dim,
                "is_causal": True,
                "scale": None,
                "sliding_window_size": None,
                "logit_soft_cap": None,
                "has_sinks": False,
                "has_qkvg_biases": False,
                "has_out_biases": False,
                "has_gate": False,
                "normalize_values": False,
                "is_kv_sharing": False,
            }
        layer_configs.append(
            {
                "pre_mixer_norm_config": block_norm,
                "mixer_conv_config": None,
                "mixer_config": mixer,
                "post_mixer_norm_config": None,
                "pre_mlp_norm_config": block_norm,
                "mlp_conv_config": None,
                "mlp_config": mlp,
                "post_mlp_norm_config": None,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": rope if layer_type == "full_attention" else None,
            }
        )
    decoder_config = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig" if tied else "UntiedEmbeddingConfig",
            "input_scale": None,
            "logit_soft_cap": None,
            "logit_scale": None,
        },
        "transformer_config": {
            "layer_configs": layer_configs,
            "output_norm_config": norm(output_epsilon),
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
        },
        "vocab_size": vocab,
        "ple_model_config": None,
        "embedding_norm_config": None,
    }
    return ConversionPlan(
        decoder_config,
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )
