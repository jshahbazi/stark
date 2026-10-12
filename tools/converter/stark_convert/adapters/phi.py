"""Bounded Phi-4-mini planner for the pinned Hugging Face Phi3 layout.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
The pinned reference implements tied Phi-4-mini with LongRoPE; other Phi3
variants require separate qualification. No upstream runtime is imported.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
    tensor_sha256,
)

_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_CONFIG_KEYS = {
    "model_type",
    "architectures",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
    "original_max_position_embeddings",
    "partial_rotary_factor",
    "rope_scaling",
    "tie_word_embeddings",
    "attention_bias",
    "mlp_bias",
    "lm_head_bias",
    "attention_dropout",
    "embd_pdrop",
    "resid_pdrop",
    "full_attn_mod",
    "interpolate_factor",
    "sliding_window",
    "auto_map",
    "torch_dtype",
    "dtype",
    "quantization",
    "quantization_config",
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
}


def _positive_int(config: dict, name: str, default: int | None = None) -> int:
    value = config.get(name, default)
    if type(value) is not int or value <= 0:
        raise ConversionError(f"Phi config requires a positive integer {name}")
    return value


def _positive_float(config: dict, name: str, default: float | None = None) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConversionError(f"Phi config requires a finite positive {name}")
    return float(value)


def _bool(config: dict, name: str, default: bool) -> bool:
    value = config.get(name, default)
    if type(value) is not bool:
        raise ConversionError(f"Phi config requires boolean {name}")
    return value


def _row_slice(tensor: TensorInfo, start: int, stop: int) -> TensorInfo:
    """Represent contiguous rows without opening or materializing the payload."""
    rows, columns = tensor.shape
    if not 0 <= start < stop <= rows:
        raise ConversionError("Phi fused projection slice is invalid")
    row_bytes = columns * tensor.itemsize
    return replace(
        tensor,
        shape=(stop - start, columns),
        offset=tensor.offset + start * row_bytes,
        nbytes=(stop - start) * row_bytes,
    )


def build_phi_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Validate the complete tied Phi3 checkpoint and plan Phi-4-mini output.

    HF stores MLP rows as gate then up. The target stores up then gate; two
    contiguous row views implement that reorder with the shared bounded writer.
    Q/K/V rows already have the required order and need no materialization.
    """
    config = read_json(source / "config.json")
    if config.get("model_type") != "phi3":
        raise UnsupportedConversion("Checkpoint model_type does not match the Phi adapter")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Phi adapter supports BF16, F16, and F32")
    target_dtype = _DTYPES[dtype]
    unsupported = []
    model_dim = _positive_int(config, "hidden_size")
    hidden_dim = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads")
    if model_dim % heads or heads % groups:
        raise ConversionError("Phi attention dimensions are inconsistent")
    head_dim = model_dim // heads
    partial_factor = _positive_float(config, "partial_rotary_factor")
    if partial_factor > 1:
        raise ConversionError("Phi partial_rotary_factor exceeds the head dimension")
    # The pinned reference uses int(head_dim * partial_rotary_factor).
    rotary_dim = int(head_dim * partial_factor)
    if rotary_dim < 2 or rotary_dim % 2:
        raise ConversionError("Phi rotary dimension must be positive and even")
    vocab = _positive_int(config, "vocab_size")
    sequence_length = _positive_int(config, "max_position_embeddings")
    original_length = _positive_int(config, "original_max_position_embeddings")
    if original_length < 2:
        raise ConversionError("Phi original context length must exceed one")
    epsilon = _positive_float(config, "rms_norm_eps")
    base = _positive_float(config, "rope_theta")
    tied = _bool(config, "tie_word_embeddings", False)
    if not tied:
        unsupported.append("Untied Phi3 embeddings are outside pinned Phi-4-mini coverage")
    if config.get("architectures", ["Phi3ForCausalLM"]) != ["Phi3ForCausalLM"]:
        unsupported.append("Checkpoint architecture is outside qualified Phi coverage")
    if config.get("hidden_act", "silu") != "silu":
        unsupported.append("Phi adapter requires SiLU")
    if set(config) - _CONFIG_KEYS:
        unsupported.append("Config options are outside qualified Phi coverage")
    for flag in (
        "attention_bias",
        "mlp_bias",
        "lm_head_bias",
        "is_encoder_decoder",
        "add_cross_attention",
        "tie_encoder_decoder",
    ):
        if _bool(config, flag, False):
            unsupported.append("Projection bias or encoder config is outside qualified Phi coverage")
    for key in ("attention_dropout", "embd_pdrop", "resid_pdrop"):
        value = config.get(key, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ConversionError("Phi config has malformed dropout")
        if value != 0:
            unsupported.append("Dropout config is outside qualified Phi coverage")
    if _positive_int(config, "full_attn_mod", 1) != 1:
        unsupported.append("Mixed sliding attention is outside qualified Phi coverage")
    if _positive_float(config, "interpolate_factor", 1.0) != 1:
        unsupported.append("Interpolated attention is outside qualified Phi coverage")
    if config.get("sliding_window") is not None:
        _positive_int(config, "sliding_window")
    pruned = config.get("pruned_heads", {})
    if not isinstance(pruned, dict):
        raise ConversionError("Phi config has malformed pruned_heads")
    if pruned:
        unsupported.append("Pruned attention heads are outside qualified Phi coverage")
    if "auto_map" in config:
        mapping = config["auto_map"]
        if not isinstance(mapping, dict) or any(not isinstance(value, str) for value in mapping.values()):
            raise ConversionError("Phi config has malformed auto_map")
        known = {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
            "AutoTokenizer": "Xenova/gpt-4o",
        }
        if any(key not in known or value != known[key] for key, value in mapping.items()):
            unsupported.append("Remote architecture annotation is outside qualified Phi coverage")
    scaling = config.get("rope_scaling")
    if not isinstance(scaling, dict) or not scaling:
        raise ConversionError("Phi config requires LongRoPE scaling parameters")
    if "type" in scaling and "rope_type" in scaling and scaling["type"] != scaling["rope_type"]:
        raise ConversionError("Phi config has contradictory RoPE types")
    if scaling.get("rope_type", scaling.get("type")) != "longrope":
        unsupported.append("RoPE scaling type is outside qualified Phi coverage")
    if set(scaling) - {"type", "rope_type", "short_factor", "long_factor"}:
        unsupported.append("LongRoPE scaling options are outside qualified Phi coverage")
    factors = {}
    for name in ("short_factor", "long_factor"):
        values = scaling.get(name)
        if not isinstance(values, list) or len(values) != rotary_dim // 2:
            raise ConversionError("Phi LongRoPE factor count disagrees with the rotary dimension")
        factors[name] = [_positive_float({"factor": value}, "factor") for value in values]
    rope = {
        "type": "LongRoPEConfig",
        "base": base,
        "max_sequence_length": sequence_length,
        "head_dim": rotary_dim,
        **factors,
        "original_context_length": original_length,
        "scaling_factor": sequence_length / original_length,
    }
    tensors = load_checkpoint(source)
    for key in ("quantization", "quantization_config"):
        if config.get(key) is not None:
            if not isinstance(config[key], dict) or not config[key]:
                raise ConversionError("Phi config has malformed quantization parameters")
            raise UnsupportedConversion("Quantized checkpoint is outside qualified Phi coverage")
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    expected: set[str] = set()

    def tensor(name: str, shape: tuple[int, ...]) -> TensorInfo:
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Phi checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("Phi checkpoint tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("Checkpoint dtype is outside qualified Phi coverage")
        return info

    def matrix(out: str, sources: tuple[TensorInfo, ...], shape: tuple[int, ...], layout: str):
        kind = "fuse" if len(sources) > 1 else "copy"
        if any(info.dtype != target_dtype for info in sources):
            kind = "fuse_cast" if len(sources) > 1 else "cast"
        operations.append(TensorOperation(out + ".weights", sources, target_dtype, shape, kind))
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def scale(out: str, name: str):
        info = tensor(name, (model_dim,))
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (model_dim,), "cast_f32"))

    embedding = tensor("model.embed_tokens.weight", (vocab, model_dim))
    matrix("decoder.embedding.embedding", (embedding,), (vocab, model_dim), "input_output")
    if "lm_head.weight" in tensors:
        head = tensor("lm_head.weight", (vocab, model_dim))
        if tied and (head.dtype != embedding.dtype or tensor_sha256(head) != tensor_sha256(embedding)):
            raise ConversionError("Tied lm_head differs from the input embedding")
    elif not tied:
        tensor("lm_head.weight", (vocab, model_dim))
    qkv_width = model_dim + 2 * groups * head_dim
    for layer in range(layers):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        scale(out + ".pre_mixer_norm", src + ".input_layernorm.weight")
        scale(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight")
        qkv = tensor(src + ".self_attn.qkv_proj.weight", (qkv_width, model_dim))
        matrix(out + ".mixer.qkvg_projection.weights", (qkv,), (qkv_width, model_dim), "output_input")
        projection = tensor(src + ".self_attn.o_proj.weight", (model_dim, model_dim))
        matrix(out + ".mixer.out_projection.weights", (projection,), (model_dim, model_dim), "output_input")
        gate_up = tensor(src + ".mlp.gate_up_proj.weight", (2 * hidden_dim, model_dim))
        matrix(
            out + ".mlp.up_projection.weights",
            (_row_slice(gate_up, hidden_dim, 2 * hidden_dim), _row_slice(gate_up, 0, hidden_dim)),
            (2 * hidden_dim, model_dim),
            "output_input",
        )
        down = tensor(src + ".mlp.down_proj.weight", (model_dim, hidden_dim))
        matrix(out + ".mlp.down_projection.weights", (down,), (model_dim, hidden_dim), "output_input")
    scale("decoder.transformer.output_norm", "model.norm.weight")
    if set(tensors) != expected:
        raise ConversionError("Phi checkpoint has unexpected tensors")
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
        "sliding_window_size": None,
        "logit_soft_cap": None,
        "has_sinks": False,
        "has_qkvg_biases": False,
        "has_out_biases": False,
        "has_gate": False,
        "normalize_values": False,
        "is_kv_sharing": False,
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
    layer_config = {
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
    decoder_config = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig",
            "input_scale": None,
            "logit_soft_cap": None,
            "logit_scale": None,
        },
        "transformer_config": {
            "layer_configs": [layer_config for _ in range(layers)],
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
