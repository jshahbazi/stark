"""Header-only planners for qualified dense Hugging Face architectures.

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

_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_ARCHITECTURES = {"qwen2": "Qwen2ForCausalLM", "llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM"}
_COMMON_KEYS = {
    "model_type",
    "architectures",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
    "attention_bias",
    "attention_dropout",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
    "rope_scaling",
    "rope_parameters",
    "layer_types",
    "quantization_config",
    "quantization",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "initializer_range",
    "use_cache",
    "transformers_version",
    "_name_or_path",
    "_attn_implementation_autoset",
    "_attn_implementation",
    "return_dict",
    "output_attentions",
    "output_hidden_states",
    "torchscript",
    "problem_type",
    "is_encoder_decoder",
    "is_decoder",
    "add_cross_attention",
    "tie_encoder_decoder",
    "pruned_heads",
    "mlp_bias",
    "pretraining_tp",
}
_FAMILY_KEYS = {
    "qwen2": {"use_sliding_window", "sliding_window", "max_window_layers"},
    # Unsloth's mirror records a packaging annotation ignored by HFLlamaConfig;
    # it changes neither the decoder nor tensor interpretation.
    "llama": {"unsloth_fixed"},
    "mistral": {"sliding_window"},
}


def _positive_int(config: dict, key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise ConversionError(f"Dense config requires a positive integer {key}")
    return value


def _positive_float(config: dict, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConversionError(f"Dense config requires a finite positive {key}")
    return float(value)


def _bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConversionError(f"Dense config requires boolean {key}")
    return value


def _rope(config: dict, family: str, head_dim: int, sequence_length: int, unsupported: list[str]) -> dict:
    rope = {
        "type": "UnscaledRoPEConfig",
        "base": _positive_float(config, "rope_theta", 10000.0),
        "max_sequence_length": sequence_length,
        "head_dim": head_dim,
    }
    scaling = config.get("rope_scaling")
    if scaling is None:
        return rope
    if not isinstance(scaling, dict) or not scaling:
        raise ConversionError("Dense config has malformed rope_scaling")
    rope_type = scaling.get("rope_type", scaling.get("type"))
    if not isinstance(rope_type, str) or not rope_type:
        raise ConversionError("Dense config has malformed RoPE type")
    if "type" in scaling and "rope_type" in scaling and scaling["type"] != scaling["rope_type"]:
        raise ConversionError("Dense config has contradictory RoPE types")
    if "factor" in scaling:
        _positive_float(scaling, "factor", 1.0)
    if rope_type == "llama3":
        factor = _positive_float(scaling, "factor", 1.0)
        low = _positive_float(scaling, "low_freq_factor", 1.0)
        high = _positive_float(scaling, "high_freq_factor", 4.0)
        original = _positive_int(scaling, "original_max_position_embeddings")
        if factor < 1.0 or high <= low or original > sequence_length:
            raise ConversionError("Dense config has invalid Llama 3 RoPE bounds")
        required = {"factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}
        if not required <= set(scaling):
            raise ConversionError("Dense config is missing Llama 3 RoPE parameters")
        if family != "llama" or set(scaling) - required - {"type", "rope_type"}:
            unsupported.append("RoPE scaling options are outside qualified dense coverage")
        rope.update(
            {
                "type": "LlamaRoPEConfig",
                "scaling_factor": factor,
                "original_context_length": original,
                "low_frequency_factor": low,
                "high_frequency_factor": high,
            }
        )
    else:
        unsupported.append("RoPE scaling type is outside qualified dense coverage")
    return rope


def build_dense_plan(source: Path, *, family: str, dtype: str = "bfloat16") -> ConversionPlan:
    """Check every tensor and return only bounded-memory transformation operations.

    Compatibility selection is performed separately by the reviewed source catalog.
    Config variants outside each adapter's explicit semantics fail closed.
    """
    if family not in _ARCHITECTURES:
        raise UnsupportedConversion("Architecture has no dense adapter")
    config = read_json(source / "config.json")
    if config.get("model_type") != family:
        raise UnsupportedConversion("Checkpoint model_type does not match selected adapter")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Dense adapters support BF16, F16, and F32")
    target_dtype = _DTYPES[dtype]
    unsupported = []
    # Known structural fields are checked before unsupported semantic options.
    model_dim = _positive_int(config, "hidden_size")
    hidden_dim = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads", heads)
    if heads % groups:
        raise ConversionError("Dense attention groups must divide the head count")
    if config.get("head_dim") is None:
        if model_dim % heads:
            raise ConversionError("Dense attention head dimension cannot be inferred")
        head_dim = model_dim // heads
    else:
        head_dim = _positive_int(config, "head_dim")
    if head_dim % 2:
        raise ConversionError("Dense rotary head dimension must be even")
    if family == "qwen2" and head_dim * heads != model_dim:
        unsupported.append("Qwen2 explicit head_dim disagrees with its qualified semantics")
    vocab = _positive_int(config, "vocab_size")
    sequence_length = _positive_int(config, "max_position_embeddings", 32768)
    epsilon = _positive_float(config, "rms_norm_eps", 1e-6)
    tied = _bool(config, "tie_word_embeddings", False)
    attention_bias = _bool(config, "attention_bias", family == "qwen2")
    mlp_bias = _bool(config, "mlp_bias", False)
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout):
        raise ConversionError("Dense config requires finite attention_dropout")
    if dropout < 0 or dropout > 1:
        raise ConversionError("Dense attention_dropout is outside its valid range")
    tp = _positive_int(config, "pretraining_tp", 1)
    if tp != 1:
        unsupported.append("Tensor-parallel pretraining config is outside qualified dense coverage")
    if (family == "qwen2" and not attention_bias) or (family != "qwen2" and attention_bias) or mlp_bias:
        unsupported.append("Projection bias variant is outside qualified dense coverage")
    if dropout != 0.0:
        unsupported.append("Attention dropout variant is outside qualified dense coverage")
    if config.get("architectures", [_ARCHITECTURES[family]]) != [_ARCHITECTURES[family]]:
        unsupported.append("Checkpoint architecture is outside qualified dense coverage")
    if config.get("hidden_act", "silu") != "silu":
        unsupported.append("Dense adapters require SiLU")
    if set(config) - _COMMON_KEYS - _FAMILY_KEYS[family]:
        unsupported.append("Config options are outside qualified dense coverage")
    for flag in ("is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder"):
        if _bool(config, flag, False):
            unsupported.append("Encoder/cross-attention config is outside qualified dense coverage")
    pruned = config.get("pruned_heads", {})
    if not isinstance(pruned, dict):
        raise ConversionError("Dense config has malformed pruned_heads")
    if pruned:
        unsupported.append("Pruned attention heads are outside qualified dense coverage")
    if config.get("rope_parameters") is not None:
        if not isinstance(config["rope_parameters"], dict):
            raise ConversionError("Dense config has malformed rope_parameters")
        unsupported.append("Custom RoPE parameters are outside qualified dense coverage")
    rope = _rope(config, family, head_dim, sequence_length, unsupported)
    windows: list[int | None] = [None] * layers
    if family == "qwen2":
        use_sliding = _bool(config, "use_sliding_window", False)
        boundary = config.get("max_window_layers", layers)
        if type(boundary) is not int or boundary < 0:
            raise ConversionError("Qwen2 sliding-window layer boundary is invalid")
        sliding = config.get("sliding_window")
        if sliding is not None:
            sliding = _positive_int(config, "sliding_window")
        if use_sliding and sliding is None:
            raise ConversionError("Qwen2 sliding-window size is missing")
        windows = [sliding if use_sliding and layer < boundary else None for layer in range(layers)]
    elif family == "mistral" and config.get("sliding_window") is not None:
        windows = [_positive_int(config, "sliding_window")] * layers
    layer_types = config.get("layer_types")
    if layer_types is not None:
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(not isinstance(layer, str) for layer in layer_types)
        ):
            raise ConversionError("Dense config has malformed layer_types")
        expected_types = ["sliding_attention" if window is not None else "full_attention" for window in windows]
        if layer_types != expected_types:
            unsupported.append("Layer attention types disagree with qualified sliding-window semantics")
    tensors = load_checkpoint(source)
    # Packed layouts have different tensor names and require later adapters.
    for key in ("quantization", "quantization_config"):
        if config.get(key) is not None:
            if not isinstance(config[key], dict) or not config[key]:
                raise ConversionError("Dense config has malformed quantization parameters")
            raise UnsupportedConversion("Quantized checkpoint is outside qualified dense coverage")
    operations: list[TensorOperation] = []
    expected: set[str] = set()
    metadata: dict[str, str] = {}
    unsupported_dtype = False

    def tensor(name: str, shape: tuple[int, ...]):
        nonlocal unsupported_dtype
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Dense checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("Dense checkpoint tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported_dtype = True
        return info

    def matrix(
        out: str, names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...], shape: tuple[int, ...], layout: str
    ):
        sources = tuple(tensor(name, source_shape) for name, source_shape in zip(names, shapes, strict=True))
        if len({info.dtype for info in sources}) > 1:
            unsupported.append("Mixed-dtype fusion promotion is outside qualified dense coverage")
        kind = "fuse" if len(sources) > 1 else "copy"
        if any(info.dtype != target_dtype for info in sources):
            kind = "fuse_cast" if len(sources) > 1 else "cast"
        operations.append(TensorOperation(out + ".weights", sources, target_dtype, shape, kind))
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def scale(out: str, name: str):
        info = tensor(name, (model_dim,))
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (model_dim,), "cast_f32"))

    embedding_out = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    matrix(embedding_out, ("model.embed_tokens.weight",), ((vocab, model_dim),), (vocab, model_dim), "input_output")
    if tied:
        if "lm_head.weight" in tensors:
            head = tensor("lm_head.weight", (vocab, model_dim))
            embedding = tensors["model.embed_tokens.weight"]
            if head.dtype != embedding.dtype or tensor_sha256(head) != tensor_sha256(embedding):
                raise ConversionError("Tied lm_head differs from the input embedding")
    else:
        matrix(
            "decoder.embedding.output_embedding",
            ("lm_head.weight",),
            ((vocab, model_dim),),
            (vocab, model_dim),
            "output_input",
        )
    q_width, kv_width = heads * head_dim, groups * head_dim
    for layer in range(layers):
        src = f"model.layers.{layer}"
        out = f"decoder.transformer.layers.{layer}"
        scale(out + ".pre_mixer_norm", src + ".input_layernorm.weight")
        scale(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight")
        matrix(
            out + ".mixer.qkvg_projection.weights",
            tuple(src + f".self_attn.{name}_proj.weight" for name in ("q", "k", "v")),
            ((q_width, model_dim), (kv_width, model_dim), (kv_width, model_dim)),
            (q_width + 2 * kv_width, model_dim),
            "output_input",
        )
        matrix(
            out + ".mixer.out_projection.weights",
            (src + ".self_attn.o_proj.weight",),
            ((model_dim, q_width),),
            (model_dim, q_width),
            "output_input",
        )
        matrix(
            out + ".mlp.up_projection.weights",
            (src + ".mlp.up_proj.weight", src + ".mlp.gate_proj.weight"),
            ((hidden_dim, model_dim), (hidden_dim, model_dim)),
            (2 * hidden_dim, model_dim),
            "output_input",
        )
        matrix(
            out + ".mlp.down_projection.weights",
            (src + ".mlp.down_proj.weight",),
            ((model_dim, hidden_dim),),
            (model_dim, hidden_dim),
            "output_input",
        )
        if family == "qwen2":
            sources = tuple(
                tensor(src + f".self_attn.{name}_proj.bias", (width,))
                for name, width in (("q", q_width), ("k", kv_width), ("v", kv_width))
            )
            if len({info.dtype for info in sources}) > 1:
                unsupported.append("Mixed-dtype bias fusion promotion is outside qualified dense coverage")
            kind = "fuse" if all(info.dtype == target_dtype for info in sources) else "fuse_cast"
            operations.append(
                TensorOperation(
                    out + ".mixer.qkvg_projection.biases", sources, target_dtype, (q_width + 2 * kv_width,), kind
                )
            )
    scale("decoder.transformer.output_norm", "model.norm.weight")
    if set(tensors) != expected:
        raise ConversionError("Dense checkpoint has unexpected tensors")
    if unsupported_dtype:
        unsupported.append("Checkpoint dtype is outside qualified dense coverage")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    norm = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    mlp = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": {"type": "SiLU", "alpha": 1.0},
        "has_up_biases": False,
        "has_down_biases": False,
        "gate_clipping": None,
        "up_clipping": None,
    }
    layer_configs = []
    for window in windows:
        mixer = {
            "type": "AttentionConfig",
            "qkvg_projection_config": {},
            "out_projection_config": {},
            "query_norm_config": None,
            "key_norm_config": None,
            "num_heads": heads,
            "num_groups": groups,
            "head_dim": head_dim,
            "is_causal": True,
            "scale": None,
            "sliding_window_size": window,
            "logit_soft_cap": None,
            "has_sinks": False,
            "has_qkvg_biases": family == "qwen2",
            "has_out_biases": False,
            "has_gate": False,
            "normalize_values": False,
            "is_kv_sharing": False,
        }
        layer_configs.append(
            {
                "pre_mixer_norm_config": norm,
                "mixer_conv_config": None,
                "mixer_config": mixer,
                "post_mixer_norm_config": None,
                "pre_mlp_norm_config": norm,
                "mlp_conv_config": None,
                "mlp_config": mlp,
                "post_mlp_norm_config": None,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": rope,
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
            "output_norm_config": norm,
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
