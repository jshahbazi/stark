"""Bounded OLMo 3 causal-decoder conversion.

Architecture semantics were checked against Transformers v4.57.1 OLMo 3
(Apache-2.0) and Apple's MLX-LM revision
1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7 (MIT, Copyright 2025 Apple Inc.).
The planner uses Stark's existing streaming conversion primitives; it does not
import or execute either reference engine.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..compressed import parse_mlx_quantization, plan_mlx_matrix, read_mlx_matrix
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
from .dense import _COMMON_KEYS, _bool, _positive_float, _positive_int

_DTYPES = {"bfloat16": "BF16", "float32": "F32"}
_KEYS = _COMMON_KEYS | {"sliding_window"}


def _rope(config: dict, head_dim: int, context: int, unsupported: list[str]) -> dict:
    rope = {
        "type": "UnscaledRoPEConfig",
        "base": _positive_float(config, "rope_theta", 500000.0),
        "max_sequence_length": context,
        "head_dim": head_dim,
    }
    scaling = config.get("rope_scaling")
    if scaling is None:
        return rope
    if not isinstance(scaling, dict) or not scaling:
        raise ConversionError("OLMo 3 has malformed RoPE scaling")
    kind = scaling.get("rope_type", scaling.get("type"))
    if kind == "default":
        if set(scaling) - {"rope_type", "type"}:
            unsupported.append("OLMo 3 default RoPE has unsupported options")
        return rope
    if kind != "yarn":
        unsupported.append("OLMo 3 RoPE requires default or YaRN scaling")
        return rope
    factor = _positive_float(scaling, "factor", 1.0)
    original = _positive_int(scaling, "original_max_position_embeddings")
    fast = _positive_float(scaling, "beta_fast", 32.0)
    slow = _positive_float(scaling, "beta_slow", 1.0)
    attention = _positive_float(scaling, "attention_factor", 0.1 * math.log(factor) + 1.0)
    if factor < 1.0 or original > context or fast <= slow or rope["base"] <= 1.0:
        raise ConversionError("OLMo 3 has invalid YaRN bounds")
    if not math.isclose(attention, 0.1 * math.log(factor) + 1.0, rel_tol=1e-7):
        unsupported.append("OLMo 3 custom YaRN attention factor is unsupported")
    if set(scaling) - {
        "rope_type",
        "type",
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "attention_factor",
        "truncate",
    }:
        unsupported.append("OLMo 3 YaRN has unsupported options")
    rope.update(
        type="YARNRoPEConfig",
        scaling_factor=factor,
        original_context_length=original,
        beta_fast=fast,
        beta_slow=slow,
        truncate=_bool(scaling, "truncate", True),
    )
    return rope


def build_olmo3_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Preserve whole-projection Q/K RMSNorm and branch-postnorm residuals."""
    config = read_json(source / "config.json")
    if config.get("model_type") != "olmo3":
        raise UnsupportedConversion("Checkpoint model_type does not match OLMo 3")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("OLMo 3 conversion supports BF16 and F32 matrix storage")
    target_dtype = _DTYPES[dtype]
    unsupported: list[str] = []
    model_dim = _positive_int(config, "hidden_size")
    hidden_dim = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    validate_planning_layers(layers)
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads", heads)
    if heads % groups or (config.get("head_dim") is None and model_dim % heads):
        raise ConversionError("OLMo 3 attention geometry is invalid")
    head_dim = _positive_int(config, "head_dim", model_dim // heads)
    if head_dim % 2:
        raise ConversionError("OLMo 3 rotary head dimension must be even")
    vocab = _positive_int(config, "vocab_size")
    context = _positive_int(config, "max_position_embeddings")
    window = _positive_int(config, "sliding_window")
    epsilon = _positive_float(config, "rms_norm_eps", 1e-6)
    tied = _bool(config, "tie_word_embeddings", False)
    if config.get("architectures", ["Olmo3ForCausalLM"]) != ["Olmo3ForCausalLM"]:
        unsupported.append("OLMo 3 checkpoint architecture is unsupported")
    if config.get("hidden_act", "silu") != "silu":
        unsupported.append("OLMo 3 requires SiLU")
    for key in ("attention_bias", "mlp_bias", "is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder"):
        if _bool(config, key, False):
            unsupported.append("OLMo 3 bias or encoder variants are unsupported")
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout):
        raise ConversionError("OLMo 3 requires finite attention_dropout")
    if not 0.0 <= dropout <= 1.0:
        raise ConversionError("OLMo 3 attention_dropout is outside its valid range")
    if dropout != 0.0:
        unsupported.append("OLMo 3 attention dropout is unsupported")
    if _positive_int(config, "pretraining_tp", 1) != 1:
        unsupported.append("OLMo 3 tensor-parallel pretraining is unsupported")
    if config.get("rope_parameters") is not None:
        unsupported.append("OLMo 3 nested RoPE parameters are unsupported")
    if config.get("pruned_heads", {}):
        unsupported.append("OLMo 3 pruned heads are unsupported")
    if set(config) - _KEYS:
        unsupported.append("OLMo 3 config has unsupported semantic options")
    layer_types = config.get(
        "layer_types", ["full_attention" if (index + 1) % 4 == 0 else "sliding_attention" for index in range(layers)]
    )
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise ConversionError("OLMo 3 layer_types must describe every layer")
    if any(not isinstance(kind, str) or kind not in {"sliding_attention", "full_attention"} for kind in layer_types):
        raise ConversionError("OLMo 3 has an invalid attention layer type")
    global_rope = _rope(config, head_dim, context, unsupported)
    local_rope = {**global_rope, "type": "UnscaledRoPEConfig"}
    local_rope = {key: local_rope[key] for key in ("type", "base", "max_sequence_length", "head_dim")}
    quantization = parse_mlx_quantization(config)
    source_q4 = (
        quantization is not None
        and quantization.bits == 4
        and quantization.group_size == 64
        and target_dtype == "BF16"
    )
    tensors = load_checkpoint(source)
    validate_planning_layers(layers, tensors)
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    used_mlx = False

    def tensor(name: str, shape: tuple[int, ...]):
        expected.add(name)
        if name not in tensors:
            raise ConversionError("OLMo 3 checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("OLMo 3 tensor shape disagrees with its config")
        if info.dtype not in {"BF16", "F16", "F32"}:
            unsupported.append("OLMo 3 tensor dtype is unsupported")
        return info

    def matrix(
        out: str, names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...], shape: tuple[int, ...], layout: str
    ):
        nonlocal used_mlx
        packed = [
            any(name.removesuffix(".weight") + suffix in tensors for suffix in (".scales", ".biases"))
            or (name in tensors and tensors[name].dtype in {"I32", "U32"})
            for name in names
        ]
        if any(packed):
            if quantization is None or not all(packed):
                raise UnsupportedConversion("OLMo 3 packed matrices require a consistent explicit MLX layout")
            sources = tuple(
                read_mlx_matrix(tensors, name, original, quantization)
                for name, original in zip(names, shapes, strict=True)
            )
            for name in names:
                expected.update(
                    (name, name.removesuffix(".weight") + ".scales", name.removesuffix(".weight") + ".biases")
                )
            operations.extend(
                plan_mlx_matrix(out, sources, dtype=target_dtype, layout=layout, quantization=quantization)
            )
            metadata[out + ".spec"] = json.dumps(
                {
                    "type": "MLXSpec",
                    "bits": quantization.bits,
                    "group_size": quantization.group_size,
                    "layout": layout,
                    **({"materialize_affine_input_sum": True} if source_q4 else {}),
                }
            )
            used_mlx = True
            return
        sources = tuple(tensor(name, original) for name, original in zip(names, shapes, strict=True))
        if len({item.dtype for item in sources}) > 1:
            unsupported.append("OLMo 3 mixed-dtype projection fusion is unsupported")
        cast = any(item.dtype != target_dtype for item in sources)
        kind = ("fuse_cast" if cast else "fuse") if len(sources) > 1 else ("cast" if cast else "copy")
        operations.append(TensorOperation(out + ".weights", sources, target_dtype, shape, kind))
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def norm(out: str, name: str, width: int):
        operations.append(TensorOperation(out + ".scales", (tensor(name, (width,)),), "F32", (width,), "cast_f32"))

    embed_out = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    matrix(embed_out, ("model.embed_tokens.weight",), ((vocab, model_dim),), (vocab, model_dim), "input_output")
    if not tied:
        matrix(
            "decoder.embedding.output_embedding",
            ("lm_head.weight",),
            ((vocab, model_dim),),
            (vocab, model_dim),
            "output_input",
        )
    elif "lm_head.weight" in tensors:
        head = tensor("lm_head.weight", (vocab, model_dim))
        embedding = tensor("model.embed_tokens.weight", (vocab, model_dim))
        if head.dtype != embedding.dtype or tensor_sha256(head) != tensor_sha256(embedding):
            raise ConversionError("OLMo 3 tied output embedding disagrees with input embedding")
    q_width, kv_width = heads * head_dim, groups * head_dim
    for index in range(layers):
        src, out = f"model.layers.{index}", f"decoder.transformer.layers.{index}"
        norm(out + ".post_mixer_norm", src + ".post_attention_layernorm.weight", model_dim)
        norm(out + ".post_mlp_norm", src + ".post_feedforward_layernorm.weight", model_dim)
        norm(out + ".mixer.query_norm", src + ".self_attn.q_norm.weight", q_width)
        norm(out + ".mixer.key_norm", src + ".self_attn.k_norm.weight", kv_width)
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
    norm("decoder.transformer.output_norm", "model.norm.weight", model_dim)
    if set(tensors) != expected:
        raise ConversionError("OLMo 3 checkpoint has unexpected tensors")
    if quantization is not None and not used_mlx:
        raise ConversionError("OLMo 3 quantization configuration has no packed matrices")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    normalization = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "only_normalization" if source_q4 else "full_layer",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    regular_normalization = {
        **normalization,
        **({"mlx_bf16_reduction": True} if source_q4 else {}),
    }
    layer_configs = []
    for kind in layer_types:
        layer_configs.append(
            {
                "post_norm": True,
                "pre_mixer_norm_config": None,
                "mixer_conv_config": None,
                "mixer_config": {
                    "type": "AttentionConfig",
                    "qkvg_projection_config": {},
                    "out_projection_config": {},
                    "query_norm_config": regular_normalization,
                    "key_norm_config": regular_normalization,
                    "qk_norm_full_width": True,
                    "num_heads": heads,
                    "num_groups": groups,
                    "head_dim": head_dim,
                    "is_causal": True,
                    "scale": None,
                    "sliding_window_size": window if kind == "sliding_attention" else None,
                    "logit_soft_cap": None,
                    "has_sinks": False,
                    "has_qkvg_biases": False,
                    "has_out_biases": False,
                    "has_gate": False,
                    "normalize_values": False,
                    "is_kv_sharing": False,
                },
                "post_mixer_norm_config": regular_normalization,
                # The post-norm execution mode uses an ordinary residual add here.
                # Keep the legacy required config field explicitly parameter-free.
                "pre_mlp_norm_config": {**normalization, "has_scale": False},
                "mlp_conv_config": None,
                "mlp_config": {
                    "type": "DenseMLPConfig",
                    "linear_config": {},
                    "activation": {
                        "type": "SiLU",
                        "alpha": 1.0,
                        **({"materialize_sigmoid": True} if source_q4 else {}),
                    },
                    "has_up_biases": False,
                    "has_down_biases": False,
                    "gate_clipping": None,
                    "up_clipping": None,
                },
                "post_mlp_norm_config": regular_normalization,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": global_rope if kind == "full_attention" else local_rope,
            }
        )
    return ConversionPlan(
        {
            **({"separate_prefill_last_token": True} if source_q4 else {}),
            "embedding_config": {
                "type": "TiedEmbeddingConfig" if tied else "UntiedEmbeddingConfig",
                "input_scale": None,
                "logit_soft_cap": None,
                "logit_scale": None,
            },
            "transformer_config": {
                "layer_configs": layer_configs,
                "output_norm_config": regular_normalization,
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
