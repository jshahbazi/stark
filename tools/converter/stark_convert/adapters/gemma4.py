"""Bounded native Gemma 4 E2B/E4B text checkpoint planning.

Architecture and tensor conventions independently implemented from Google's
Gemma 4 E2B checkpoint 3e22461f65e89153144f8adb70e3b8c2cc9845a7
(Apache-2.0) and MLX-LM 1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7
(Copyright © 2025 Apple Inc., MIT). No MLX runtime is imported.
Canonical activation arithmetic follows Transformers 5.15.1 modeling_gemma4.py,
SHA-256 4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3,
Copyright 2026 the HuggingFace Team, Apache-2.0.
The decoder uses Stark's native PLE, shared-KV and Metal execution blocks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import ml_dtypes
import numpy as np

from ..compressed import MLXMatrixSource, parse_mlx_quantization, plan_mlx_matrix, read_mlx_matrix
from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
    tensor_sha256,
    validate_planning_layers,
)
from .dense import _COMMON_KEYS, _DTYPES
from .gemma import _boolean, _integer, _number

_KEYS = _COMMON_KEYS | {
    "attention_k_eq_v",
    "enable_moe_block",
    "expert_intermediate_size",
    "final_logit_softcapping",
    "global_head_dim",
    "global_partial_rotary_factor",
    "hidden_activation",
    "hidden_size_per_layer_input",
    "layer_types",
    "num_experts",
    "num_global_key_value_heads",
    "num_kv_shared_layers",
    "rope_parameters",
    "sliding_window",
    "sliding_window_pattern",
    "top_k_experts",
    "use_bidirectional_attention",
    "use_double_wide_mlp",
    "vocab_size_per_layer_input",
    "rope_traditional",
}


def _activation_scalar(value: float, dtype: str) -> float:
    """Materialize canonical HF's explicitly dtype-cast scale buffers."""
    scalar_dtype = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32}[dtype]
    return float(scalar_dtype(value))


def normalize_gemma4_config(config: dict) -> dict:
    """Keep wrapper token IDs (including end-of-turn EOS) over text defaults."""
    if config.get("model_type") == "gemma4_text":
        return dict(config)
    text = config.get("text_config")
    if config.get("model_type") != "gemma4" or not isinstance(text, dict):
        raise ConversionError("Gemma 4 wrapper requires a text configuration")
    return {
        **text,
        **{key: value for key, value in config.items() if key != "text_config"},
        "model_type": "gemma4_text",
    }


def build_gemma4_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Export E2B/E4B text inference without importing modality towers.

    Official multimodal wrappers and MLX-LM text exports have explicit, distinct
    namespaces. Every original tensor remains source-bound in the plan, including
    omitted declared towers and redundant shared-layer K/V weights.
    """
    outer = read_json(source / "config.json")
    wrapper = outer.get("model_type") == "gemma4"
    if wrapper:
        config = outer.get("text_config")
        if not isinstance(config, dict) or config.get("model_type") != "gemma4_text":
            raise UnsupportedConversion("Gemma 4 wrapper requires an explicit Gemma 4 text config")
        if outer.get("architectures", ["Gemma4ForConditionalGeneration"]) != ["Gemma4ForConditionalGeneration"]:
            raise UnsupportedConversion("Gemma 4 wrapper architecture is outside text coverage")
        if outer.get("tie_word_embeddings", True) is not True:
            raise UnsupportedConversion("Gemma 4 text coverage requires tied embeddings")
        for key in ("vision_config", "audio_config"):
            if outer.get(key) is not None and not isinstance(outer[key], dict):
                raise ConversionError("Gemma 4 modality configuration must be an object")
    else:
        config = outer
        if config.get("model_type") != "gemma4_text":
            raise UnsupportedConversion("Gemma 4 adapter supports E2B/E4B text architectures")
        if config.get("architectures", ["Gemma4ForCausalLM"]) not in (
            ["Gemma4ForCausalLM"],
            ["Gemma4TextModel"],
        ):
            raise UnsupportedConversion("Gemma 4 text architecture is outside coverage")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Gemma 4 text exports require BF16, F16 or F32")
    target = _DTYPES[dtype]
    quantization = parse_mlx_quantization(outer)
    if wrapper and parse_mlx_quantization(config) not in (None, quantization):
        raise ConversionError("Gemma 4 wrapper and text quantization disagree")
    if set(config) - _KEYS:
        raise UnsupportedConversion("Gemma 4 text config has unimplemented options")
    for key in ("attention_bias", "mlp_bias", "attention_k_eq_v", "enable_moe_block", "rope_traditional"):
        if _boolean(config, key, False):
            raise UnsupportedConversion("Gemma 4 E2B/E4B text coverage excludes bias, MoE and K=V variants")
    if not _boolean(config, "tie_word_embeddings", True):
        raise UnsupportedConversion("Gemma 4 text coverage requires tied embeddings")
    if config.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise UnsupportedConversion("Gemma 4 requires approximate GELU")
    if config.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise UnsupportedConversion("Gemma 4 activation aliases disagree")
    for key in ("is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder"):
        if _boolean(config, key, False):
            raise UnsupportedConversion("Gemma 4 text coverage requires a causal decoder")
    if config.get("pruned_heads", {}) or config.get("pretraining_tp", 1) != 1:
        raise UnsupportedConversion("Gemma 4 pruned or tensor-parallel variants require separate support")
    if config.get("rope_scaling") is not None:
        raise UnsupportedConversion("Gemma 4 rotary parameters must be declared per attention type")
    if config.get("attention_dropout", 0.0) != 0.0:
        raise UnsupportedConversion("Gemma 4 dropout is outside inference coverage")
    if config.get("use_bidirectional_attention") not in (None, False):
        raise UnsupportedConversion("Gemma 4 text coverage requires causal attention")
    for key in ("num_experts", "top_k_experts", "expert_intermediate_size", "num_global_key_value_heads"):
        if config.get(key) is not None:
            raise UnsupportedConversion("Gemma 4 E2B/E4B config declares an unsupported architecture")
    dim = _integer(config, "hidden_size")
    hidden = _integer(config, "intermediate_size")
    count = _integer(config, "num_hidden_layers")
    validate_planning_layers(count)
    shared_count = config.get("num_kv_shared_layers", 0)
    if type(shared_count) is not int or not 0 <= shared_count < count:
        raise ConversionError("Gemma 4 shared layer count must fit its decoder")
    owned_count = count - shared_count
    heads = _integer(config, "num_attention_heads")
    groups = _integer(config, "num_key_value_heads")
    local_dim = _integer(config, "head_dim")
    global_dim = _integer(config, "global_head_dim")
    if heads % groups or local_dim % 2 or global_dim % 2:
        raise ConversionError("Gemma 4 attention dimensions must be positive and divisible")
    vocab = _integer(config, "vocab_size")
    ple_dim = _integer(config, "hidden_size_per_layer_input")
    ple_vocab = _integer(config, "vocab_size_per_layer_input")
    if ple_vocab != vocab:
        raise UnsupportedConversion("Gemma 4 text and PLE vocabularies must agree")
    epsilon = _number(config, "rms_norm_eps", 1e-6)
    # Value normalization in the native attention block uses this exact epsilon.
    if epsilon != 1e-6:
        raise UnsupportedConversion("Gemma 4 value normalization requires the reference epsilon")
    context = _integer(config, "max_position_embeddings")
    window = _integer(config, "sliding_window")
    double_wide = _boolean(config, "use_double_wide_mlp", False)
    softcap = _number(config, "final_logit_softcapping", 30.0)
    layer_types = config.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != count
        or any(value not in ("sliding_attention", "full_attention") for value in layer_types)
        or layer_types[-1] != "full_attention"
    ):
        raise ConversionError("Gemma 4 layer_types must describe its complete attention schedule")
    ropes = config.get("rope_parameters")
    if not isinstance(ropes, dict) or set(ropes) != {"sliding_attention", "full_attention"}:
        raise ConversionError("Gemma 4 requires explicit local and global rotary parameters")
    rope_configs = {}
    for kind, width in (("sliding_attention", local_dim), ("full_attention", global_dim)):
        params = ropes[kind]
        if not isinstance(params, dict):
            raise ConversionError("Gemma 4 rotary parameters must be objects")
        if set(params) - {"rope_theta", "rope_type", "partial_rotary_factor"}:
            raise UnsupportedConversion("Gemma 4 rotary configuration has unimplemented options")
        expected_type = "default" if kind == "sliding_attention" else "proportional"
        if params.get("rope_type") != expected_type:
            raise UnsupportedConversion("Gemma 4 rotary type is outside E2B/E4B coverage")
        partial = _number(params, "partial_rotary_factor", 1.0)
        rotary = int(width * partial)
        if partial > 1 or rotary < 2 or rotary % 2 or rotary != width * partial:
            raise ConversionError("Gemma 4 rotary pair count must fit its complete head")
        if kind == "sliding_attention" and rotary != width:
            raise UnsupportedConversion("Gemma 4 sliding attention requires complete rotation")
        rope_configs[kind] = {
            "type": "UnscaledRoPEConfig" if kind == "sliding_attention" else "ProportionalRoPEConfig",
            "base": _number(params, "rope_theta"),
            "head_dim": width,
            "max_sequence_length": context,
        }
        if kind == "full_attention":
            if config.get("global_partial_rotary_factor", partial) != partial:
                raise ConversionError("Gemma 4 global rotary factor aliases disagree")
            rope_configs[kind]["rotary_dim"] = rotary
    owners = {}
    sources = []
    for index, kind in enumerate(layer_types):
        if index < owned_count:
            owners[kind] = index
            sources.append(None)
        else:
            if kind not in owners:
                raise ConversionError("Gemma 4 shared attention has no preceding owner of its type")
            sources.append(owners[kind])
    tensors = load_checkpoint(source)
    validate_planning_layers(count, tensors)
    prefixes = ("model.language_model", "language_model.model") if wrapper else ("model",)
    matches = [prefix for prefix in prefixes if prefix + ".embed_tokens.weight" in tensors]
    if len(matches) != 1:
        raise ConversionError("Gemma 4 text tensor namespace is missing or ambiguous")
    prefix = matches[0]
    expected = set()
    operations = []
    metadata = {}
    used_mlx = False

    def tensor(name, shape):
        expected.add(name)
        if name not in tensors or tensors[name].shape != shape:
            raise ConversionError("Gemma 4 required tensor is missing or has the wrong shape")
        info = tensors[name]
        if info.dtype not in _DTYPES.values():
            raise UnsupportedConversion("Gemma 4 scalar tensors require BF16/F16/F32")
        return info

    def matrix_source(name, shape):
        nonlocal used_mlx
        stem = name.removesuffix(".weight")
        packed = any(stem + suffix in tensors for suffix in (".scales", ".biases"))
        if packed or name in tensors and tensors[name].dtype in ("U32", "I32"):
            if quantization is None:
                raise UnsupportedConversion("Packed Gemma 4 matrices require explicit MLX affine configuration")
            value = read_mlx_matrix(tensors, name, shape, quantization)
            expected.update((name, stem + ".scales", stem + ".biases"))
            used_mlx = True
            return value
        return tensor(name, shape)

    def matrix(out, names, shapes, layout="output_input"):
        values = tuple(matrix_source(name, shape) for name, shape in zip(names, shapes, strict=True))
        if any(isinstance(value, MLXMatrixSource) for value in values):
            if not all(isinstance(value, MLXMatrixSource) for value in values):
                raise UnsupportedConversion("Gemma 4 fused matrices cannot mix affine and full precision")
            operations.extend(plan_mlx_matrix(out, values, dtype=target, layout=layout, quantization=quantization))
            spec = {
                "type": "MLXSpec",
                "layout": layout,
                "bits": quantization.bits,
                "group_size": quantization.group_size,
            }
        else:
            cast = any(value.dtype != target for value in values)
            kind = ("fuse_cast" if cast else "fuse") if len(values) > 1 else ("cast" if cast else "copy")
            shape = (sum(shape[0] for shape in shapes), shapes[0][1])
            operations.append(TensorOperation(out + ".weights", values, target, shape, kind))
            spec = {"type": "FullPrecisionSpec", "layout": layout}
        metadata[out + ".spec"] = json.dumps(spec)

    def norm(out, name, width):
        info = tensor(name, (width,))
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (width,), "cast_f32"))

    matrix("decoder.embedding.embedding", (prefix + ".embed_tokens.weight",), ((vocab, dim),), "input_output")
    norm("decoder.transformer.output_norm", prefix + ".norm.weight", dim)
    matrix(
        "decoder.per_layer_embedding.token_embedding",
        (prefix + ".embed_tokens_per_layer.weight",),
        ((ple_vocab, count * ple_dim),),
        "input_output",
    )
    matrix(
        "decoder.per_layer_embedding.model_projection.weights",
        (prefix + ".per_layer_model_projection.weight",),
        ((count * ple_dim, dim),),
    )
    norm("decoder.per_layer_embedding.projection_norm", prefix + ".per_layer_projection_norm.weight", ple_dim)
    # Tied heads may be serialized by text exporters. Admit only byte-identical
    # copies, including every packed affine plane.
    for head in ("lm_head.weight", "language_model.lm_head.weight"):
        if head not in tensors:
            continue
        left = matrix_source(head, (vocab, dim))
        right = matrix_source(prefix + ".embed_tokens.weight", (vocab, dim))
        left = left.planes if isinstance(left, MLXMatrixSource) else (left,)
        right = right.planes if isinstance(right, MLXMatrixSource) else (right,)
        if len(left) != len(right) or any(
            a.dtype != b.dtype or tensor_sha256(a) != tensor_sha256(b) for a, b in zip(left, right, strict=True)
        ):
            raise ConversionError("Gemma 4 tied output head differs from the embedding")
    norm_config = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "full_layer",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    activation = {"type": "GELU", "approximate": True}
    mlp = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": activation,
        "has_up_biases": False,
        "has_down_biases": False,
        "gate_clipping": None,
        "up_clipping": None,
    }
    layers = []
    for index, (kind, owner) in enumerate(zip(layer_types, sources, strict=True)):
        src = prefix + f".layers.{index}"
        out = f"decoder.transformer.layers.{index}"
        width = local_dim if kind == "sliding_attention" else global_dim
        intermediate = hidden * (2 if double_wide and owner is not None else 1)
        for destination, origin in (
            ("pre_mixer_norm", "input_layernorm"),
            ("post_mixer_norm", "post_attention_layernorm"),
            ("pre_mlp_norm", "pre_feedforward_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm"),
        ):
            norm(out + "." + destination, src + "." + origin + ".weight", dim)
        scalar = tensor(src + ".layer_scalar", (1,))
        operations.append(
            TensorOperation(
                out + ".post_layer_scalar", (scalar,), target, (1,), "copy" if scalar.dtype == target else "cast"
            )
        )
        norm(out + ".mixer.query_norm", src + ".self_attn.q_norm.weight", width)
        branches = ("q", "k", "v") if owner is None else ("q",)
        matrix(
            out + ".mixer.qkvg_projection.weights",
            tuple(src + f".self_attn.{branch}_proj.weight" for branch in branches),
            tuple((heads * width if branch == "q" else groups * width, dim) for branch in branches),
        )
        if owner is None:
            norm(out + ".mixer.key_norm", src + ".self_attn.k_norm.weight", width)
        else:
            # HF serializes redundant K/V weights for shared layers; MLX-LM
            # drops them. Neither representation changes the cache owner.
            for branch in ("k", "v"):
                name = src + f".self_attn.{branch}_proj.weight"
                if name in tensors:
                    matrix_source(name, (groups * width, dim))
            name = src + ".self_attn.k_norm.weight"
            if name in tensors:
                tensor(name, (width,))
        matrix(out + ".mixer.out_projection.weights", (src + ".self_attn.o_proj.weight",), ((dim, heads * width),))
        matrix(
            out + ".mlp.up_projection.weights",
            (src + ".mlp.up_proj.weight", src + ".mlp.gate_proj.weight"),
            ((intermediate, dim), (intermediate, dim)),
        )
        matrix(out + ".mlp.down_projection.weights", (src + ".mlp.down_proj.weight",), ((dim, intermediate),))
        matrix(out + ".ple.gate.weights", (src + ".per_layer_input_gate.weight",), ((ple_dim, dim),))
        matrix(out + ".ple.projection.weights", (src + ".per_layer_projection.weight",), ((dim, ple_dim),))
        norm(out + ".ple.norm", src + ".post_per_layer_input_norm.weight", dim)
        mixer = {
            "type": "AttentionConfig",
            "qkvg_projection_config": {},
            "out_projection_config": {},
            "query_norm_config": norm_config,
            "key_norm_config": norm_config if owner is None else None,
            "num_heads": heads,
            "num_groups": groups,
            "head_dim": width,
            "is_causal": True,
            "scale": 1.0,
            "sliding_window_size": window if kind == "sliding_attention" else None,
            "logit_soft_cap": None,
            "has_sinks": False,
            "has_qkvg_biases": False,
            "has_out_biases": False,
            "has_gate": False,
            "normalize_values": True,
            "is_kv_sharing": owner is not None,
        }
        layers.append(
            {
                "pre_mixer_norm_config": norm_config,
                "mixer_conv_config": None,
                "mixer_config": mixer,
                "post_mixer_norm_config": norm_config,
                "pre_mlp_norm_config": norm_config,
                "mlp_conv_config": None,
                "mlp_config": mlp,
                "post_mlp_norm_config": norm_config,
                "hidden_dim": intermediate if intermediate != hidden else None,
                "ple_config": {
                    "linear_config": {},
                    "norm_config": norm_config,
                    "ple_dim": ple_dim,
                    "activation": activation,
                    "reference_bf16": target == "BF16",
                },
                "has_post_layer_scalar": True,
                "kv_source_layer_index": owner,
                "rope_config": rope_configs[kind],
            }
        )
    ignored = set()
    if wrapper:
        # MLX-VLM sanitizes both the text namespace and the declared modality
        # namespaces. Bind omissions to that one admitted source layout;
        # unrelated or mixed-layout tensors must still fail closed.
        modality_root = "model." if prefix == "model.language_model" else ""
        modality_prefixes = []
        if outer.get("vision_config"):
            modality_prefixes += [modality_root + "vision_tower.", modality_root + "embed_vision."]
        if outer.get("audio_config"):
            modality_prefixes += [modality_root + "audio_tower.", modality_root + "embed_audio."]
        ignored = {name for name in tensors if any(name.startswith(value) for value in modality_prefixes)}
    if set(tensors) != expected | ignored:
        raise ConversionError("Gemma 4 checkpoint has unexpected or unconsumed tensors")
    if quantization is not None and not used_mlx:
        raise ConversionError("Gemma 4 affine configuration contains no packed matrices")
    decoder = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig",
            "input_scale": _activation_scalar(math.sqrt(dim), target),
            "logit_soft_cap": softcap,
            "logit_scale": None,
        },
        "transformer_config": {
            "layer_configs": layers,
            "output_norm_config": norm_config,
            "model_dim": dim,
            "hidden_dim": hidden,
        },
        "vocab_size": vocab,
        "embedding_norm_config": None,
        "ple_model_config": {
            "ple_dim": ple_dim,
            "num_layers": count,
            "ple_vocab_size": ple_vocab,
            "ple_embed_scale": _activation_scalar(math.sqrt(ple_dim), target),
            # Canonical HF keeps these Python scalars in FP32 opmath, unlike
            # its explicit dtype-cast token-embedding scale buffers. MLX uses
            # weak BF16 scalars here; that is a separate numerical profile.
            "model_projection_scale": _activation_scalar(dim**-0.5, "F32"),
            "input_scale": _activation_scalar(2**-0.5, "F32"),
            "linear_config": {},
            "norm_config": norm_config,
            "reference_bf16": target == "BF16",
        },
    }
    return ConversionPlan(
        decoder,
        tuple(operations),
        metadata,
        sum(value.nbytes for value in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )
