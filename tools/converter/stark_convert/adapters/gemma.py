"""Bounded dense Gemma 2 and dense/MLX Gemma 3 text decoder planners.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
This module does not import Lalamo, JAX, Transformers, or PyTorch.
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
from .dense import _COMMON_KEYS

_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_ARCHITECTURES = {"gemma2": "Gemma2ForCausalLM", "gemma3_text": "Gemma3ForCausalLM"}
_KEYS = _COMMON_KEYS | {
    "hidden_activation",
    "attn_logit_softcapping",
    "final_logit_softcapping",
    "query_pre_attn_scalar",
    "sliding_window",
    "sliding_window_pattern",
    "_sliding_window_pattern",
    "rope_local_base_freq",
    "cache_implementation",
    "use_bidirectional_attention",
}


def _integer(config: dict, key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise ConversionError(f"Gemma config requires a positive integer {key}")
    return value


def _number(config: dict, key: str, default: float | None = None) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConversionError(f"Gemma config requires a finite positive {key}")
    return float(value)


def _boolean(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConversionError(f"Gemma config requires boolean {key}")
    return value


def _softcap(config: dict, key: str, *, required: bool) -> float | None:
    if not required and config.get(key) is None:
        return None
    return _number(config, key)


def _global_rope(config: dict, head_dim: int, sequence_length: int, unsupported: list[str]) -> dict:
    rope = {
        "type": "UnscaledRoPEConfig",
        "base": _number(config, "rope_theta", 1000000.0),
        "max_sequence_length": sequence_length,
        "head_dim": head_dim,
    }
    scaling = config.get("rope_scaling")
    if scaling is None:
        return rope
    if not isinstance(scaling, dict) or not scaling:
        raise ConversionError("Gemma config has malformed rope_scaling")
    rope_type = scaling.get("rope_type")
    if not isinstance(rope_type, str) or not rope_type:
        raise ConversionError("Gemma config requires a RoPE scaling type")
    factor = _number(scaling, "factor")
    if rope_type == "linear":
        if set(scaling) != {"factor", "rope_type"}:
            unsupported.append("Gemma linear RoPE options are outside qualified coverage")
        rope.update(type="LinearScalingRoPEConfig", scaling_factor=factor)
    elif rope_type == "yarn":
        original = _integer(scaling, "original_max_position_embeddings")
        fast = _number(scaling, "beta_fast")
        slow = _number(scaling, "beta_slow")
        truncate = _boolean(scaling, "truncate", False)
        if original > sequence_length or fast <= slow or rope["base"] <= 1.0:
            raise ConversionError("Gemma YARN RoPE bounds are invalid")
        if set(scaling) - {
            "factor",
            "rope_type",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "truncate",
        }:
            unsupported.append("Gemma YARN RoPE options are outside qualified coverage")
        rope.update(
            type="YARNRoPEConfig",
            scaling_factor=factor,
            original_context_length=original,
            beta_fast=fast,
            beta_slow=slow,
            truncate=truncate,
        )
    else:
        unsupported.append("Gemma RoPE scaling type is outside qualified coverage")
    return rope


def build_gemma_plan(source: Path, *, family: str, dtype: str = "bfloat16") -> ConversionPlan:
    """Validate the complete recognized text layout before transformation.

    Embedding and norm scale offsets remain runtime config semantics. The planner
    neither modifies stored norm values nor introduces full-weight allocations.
    Multimodal Gemma 3 and newer Gemma generations require separate adapters.
    """
    if family not in _ARCHITECTURES:
        raise UnsupportedConversion("Gemma generation has no text decoder adapter")
    config = read_json(source / "config.json")
    if config.get("model_type") != family:
        raise UnsupportedConversion("Checkpoint model_type does not match the Gemma text adapter")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Gemma adapters support BF16, F16, and F32")
    target_dtype = _DTYPES[dtype]
    quantization = None
    if family == "gemma3_text":
        quantization = parse_mlx_quantization(config)
    unsupported: list[str] = []
    model_dim = _integer(config, "hidden_size")
    hidden_dim = _integer(config, "intermediate_size")
    layers = _integer(config, "num_hidden_layers")
    validate_planning_layers(layers)
    heads = _integer(config, "num_attention_heads", 8 if family == "gemma3_text" else None)
    groups = _integer(config, "num_key_value_heads", 4 if family == "gemma3_text" else None)
    head_dim = _integer(config, "head_dim", 256 if family == "gemma3_text" else None)
    if heads % groups or head_dim % 2:
        raise ConversionError("Gemma attention groups or rotary head dimension are invalid")
    vocab = _integer(config, "vocab_size", 262208 if family == "gemma3_text" else None)
    sequence_length = _integer(config, "max_position_embeddings", 131072 if family == "gemma3_text" else None)
    epsilon = _number(config, "rms_norm_eps", 1e-6)
    attention_scale = _number(config, "query_pre_attn_scalar", 256.0 if family == "gemma3_text" else None) ** -0.5
    attention_bias = _boolean(config, "attention_bias", False)
    if not _boolean(config, "tie_word_embeddings", True):
        unsupported.append("Gemma text adapters require tied embeddings")
    if _boolean(config, "mlp_bias", False):
        unsupported.append("Gemma MLP biases are outside qualified coverage")
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout):
        raise ConversionError("Gemma config requires finite attention_dropout")
    if not 0.0 <= dropout <= 1.0:
        raise ConversionError("Gemma attention_dropout is outside its valid range")
    if dropout != 0.0:
        unsupported.append("Gemma attention dropout is outside qualified coverage")
    if config.get("architectures", [_ARCHITECTURES[family]]) != [_ARCHITECTURES[family]]:
        unsupported.append("Checkpoint architecture is outside Gemma text coverage")
    for key in ("hidden_act", "hidden_activation"):
        if config.get(key, "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            unsupported.append("Gemma text adapters require approximate GELU")
    if set(config) - _KEYS:
        unsupported.append("Gemma config options are outside qualified coverage")
    for flag in ("is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder", "use_bidirectional_attention"):
        if _boolean(config, flag, False):
            unsupported.append("Gemma text adapter requires a causal decoder")
    if _integer(config, "pretraining_tp", 1) != 1:
        unsupported.append("Gemma tensor-parallel pretraining is outside qualified coverage")
    pruned = config.get("pruned_heads", {})
    if not isinstance(pruned, dict):
        raise ConversionError("Gemma config has malformed pruned_heads")
    if pruned:
        unsupported.append("Gemma pruned attention heads are outside qualified coverage")
    if config.get("rope_parameters") is not None:
        if not isinstance(config["rope_parameters"], dict):
            raise ConversionError("Gemma config has malformed rope_parameters")
        unsupported.append("New Gemma RoPE parameters require a separate adapter")
    tensors = load_checkpoint(source)
    validate_planning_layers(layers, tensors)
    window = _integer(config, "sliding_window")
    if family == "gemma2":
        windows = [window if layer % 2 == 0 else None for layer in range(layers)]
        rope = {
            "type": "UnscaledRoPEConfig",
            "base": _number(config, "rope_theta"),
            "max_sequence_length": sequence_length,
            "head_dim": head_dim,
        }
        ropes = [rope] * layers
        if any(
            config.get(key) is not None
            for key in ("rope_scaling", "sliding_window_pattern", "_sliding_window_pattern", "rope_local_base_freq")
        ):
            unsupported.append("Gemma 2 rotary/window variant is outside qualified coverage")
    else:
        pattern = _integer(config, "sliding_window_pattern", 6)
        # Modern HF config serialization renames this field. The pinned
        # reference ignores the annotation and uses the public/default pattern.
        # Accept it only when both interpretations and explicit layer types agree.
        if "_sliding_window_pattern" in config and _integer(config, "_sliding_window_pattern") != pattern:
            unsupported.append("Gemma private window pattern disagrees with qualified reference semantics")
        windows = [None if (layer + 1) % pattern == 0 else window for layer in range(layers)]
        global_rope = _global_rope(config, head_dim, sequence_length, unsupported)
        local_rope = {
            "type": "UnscaledRoPEConfig",
            "base": _number(config, "rope_local_base_freq", 10000.0),
            "max_sequence_length": sequence_length,
            "head_dim": head_dim,
        }
        ropes = [global_rope if local is None else local_rope for local in windows]
    if config.get("cache_implementation", "hybrid") != "hybrid":
        unsupported.append("Gemma cache variant is outside qualified coverage")
    layer_types = config.get("layer_types")
    if layer_types is not None:
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(not isinstance(layer_type, str) for layer_type in layer_types)
        ):
            raise ConversionError("Gemma config has malformed layer_types")
        if layer_types != ["sliding_attention" if local is not None else "full_attention" for local in windows]:
            unsupported.append("Gemma layer types disagree with qualified window semantics")
    attention_softcap = _softcap(config, "attn_logit_softcapping", required=family == "gemma2")
    final_softcap = _softcap(config, "final_logit_softcapping", required=family == "gemma2")
    for key in ("quantization", "quantization_config"):
        if family == "gemma2" and config.get(key) is not None:
            if not isinstance(config[key], dict) or not config[key]:
                raise ConversionError("Gemma config has malformed quantization parameters")
            raise UnsupportedConversion("Quantized Gemma checkpoint requires a separate adapter")
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    used_mlx = False

    def tensor(name: str, shape: tuple[int, ...]):
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Gemma checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("Gemma checkpoint tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("Gemma checkpoint dtype is outside qualified coverage")
        return info

    def projected(out: str, names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...], shape: tuple[int, ...]):
        inputs = tuple(tensor(name, source_shape) for name, source_shape in zip(names, shapes, strict=True))
        if len({info.dtype for info in inputs}) > 1:
            unsupported.append("Mixed-dtype Gemma fusion promotion is outside qualified coverage")
        cast = any(info.dtype != target_dtype for info in inputs)
        kind = ("fuse_cast" if cast else "fuse") if len(inputs) > 1 else ("cast" if cast else "copy")
        operations.append(TensorOperation(out, inputs, target_dtype, shape, kind))

    def matrix(
        out: str, names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...], shape: tuple[int, ...], layout: str
    ):
        nonlocal used_mlx
        representations = []
        for name in names:
            prefix = name.removesuffix(".weight")
            representations.append(
                any(prefix + suffix in tensors for suffix in (".scales", ".biases"))
                or (name in tensors and tensors[name].dtype in {"I32", "U32"})
            )
        if any(representations):
            if family != "gemma3_text":
                raise UnsupportedConversion("Packed Gemma 2 checkpoints require a separate adapter")
            if quantization is None:
                raise UnsupportedConversion("MLX Gemma text checkpoint requires explicit quantization configuration")
            if not all(representations):
                raise UnsupportedConversion("Mixed full-precision/MLX Gemma fusion is outside implemented coverage")
            sources = tuple(
                read_mlx_matrix(tensors, name, source_shape, quantization)
                for name, source_shape in zip(names, shapes, strict=True)
            )
            for name in names:
                prefix = name.removesuffix(".weight")
                expected.update((name, prefix + ".scales", prefix + ".biases"))
            operations.extend(
                plan_mlx_matrix(out, sources, dtype=target_dtype, layout=layout, quantization=quantization)
            )
            metadata[out + ".spec"] = json.dumps(
                {
                    "type": "MLXSpec",
                    "bits": quantization.bits,
                    "group_size": quantization.group_size,
                    "layout": layout,
                }
            )
            used_mlx = True
            return
        projected(out + ".weights", names, shapes, shape)
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def norm(out: str, name: str, width: int):
        info = tensor(name, (width,))
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (width,), "cast_f32"))

    matrix(
        "decoder.embedding.embedding",
        ("model.embed_tokens.weight",),
        ((vocab, model_dim),),
        (vocab, model_dim),
        "input_output",
    )
    if any("lm_head" + suffix in tensors for suffix in (".weight", ".scales", ".biases")):
        if json.loads(metadata["decoder.embedding.embedding.spec"])["type"] == "MLXSpec":
            head = read_mlx_matrix(tensors, "lm_head.weight", (vocab, model_dim), quantization)
            embedding = read_mlx_matrix(tensors, "model.embed_tokens.weight", (vocab, model_dim), quantization)
            expected.update(("lm_head.weight", "lm_head.scales", "lm_head.biases"))
            if any(
                left.dtype != right.dtype or tensor_sha256(left) != tensor_sha256(right)
                for left, right in zip(head.planes, embedding.planes, strict=True)
            ):
                raise ConversionError("Tied MLX Gemma lm_head differs from the input embedding")
        else:
            head = tensor("lm_head.weight", (vocab, model_dim))
            embedding = tensors["model.embed_tokens.weight"]
            if head.dtype != embedding.dtype or tensor_sha256(head) != tensor_sha256(embedding):
                raise ConversionError("Tied Gemma lm_head differs from the input embedding")
    norm("decoder.transformer.output_norm", "model.norm.weight", model_dim)
    q_width, kv_width = heads * head_dim, groups * head_dim
    for layer in range(layers):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        for destination, origin in (
            ("pre_mixer_norm", "input_layernorm"),
            ("post_mixer_norm", "post_attention_layernorm"),
            ("pre_mlp_norm", "pre_feedforward_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm"),
        ):
            norm(out + "." + destination, src + "." + origin + ".weight", model_dim)
        if family == "gemma3_text":
            norm(out + ".mixer.query_norm", src + ".self_attn.q_norm.weight", head_dim)
            norm(out + ".mixer.key_norm", src + ".self_attn.k_norm.weight", head_dim)
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
        if attention_bias:
            projected(
                out + ".mixer.qkvg_projection.biases",
                tuple(src + f".self_attn.{name}_proj.bias" for name in ("q", "k", "v")),
                ((q_width,), (kv_width,), (kv_width,)),
                (q_width + 2 * kv_width,),
            )
            if family == "gemma3_text":
                projected(
                    out + ".mixer.out_projection.biases",
                    (src + ".self_attn.o_proj.bias",),
                    ((model_dim,),),
                    (model_dim,),
                )
    if set(tensors) != expected:
        raise ConversionError("Gemma checkpoint has unexpected tensors")
    if quantization is not None and not used_mlx:
        raise ConversionError("MLX Gemma quantization configuration has no packed checkpoint matrices")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    norm_config = {
        "epsilon": epsilon,
        "scale_offset": 1.0,
        "upcast_mode": "full_layer",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    mlp_config = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": {"type": "GELU", "approximate": True},
        "has_up_biases": False,
        "has_down_biases": False,
        "gate_clipping": None,
        "up_clipping": None,
    }
    layer_configs = []
    for local_window, rope in zip(windows, ropes, strict=True):
        mixer_config = {
            "type": "AttentionConfig",
            "qkvg_projection_config": {},
            "out_projection_config": {},
            "query_norm_config": norm_config if family == "gemma3_text" else None,
            "key_norm_config": norm_config if family == "gemma3_text" else None,
            "num_heads": heads,
            "num_groups": groups,
            "head_dim": head_dim,
            "is_causal": True,
            "scale": attention_scale,
            "sliding_window_size": local_window,
            "logit_soft_cap": attention_softcap,
            "has_sinks": False,
            "has_qkvg_biases": attention_bias,
            "has_out_biases": attention_bias and family == "gemma3_text",
            "has_gate": False,
            "normalize_values": False,
            "is_kv_sharing": False,
        }
        layer_configs.append(
            {
                "pre_mixer_norm_config": norm_config,
                "mixer_conv_config": None,
                "mixer_config": mixer_config,
                "post_mixer_norm_config": norm_config,
                "pre_mlp_norm_config": norm_config,
                "mlp_conv_config": None,
                "mlp_config": mlp_config,
                "post_mlp_norm_config": norm_config,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": rope,
            }
        )
    decoder_config = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig",
            "input_scale": model_dim**0.5,
            "logit_soft_cap": final_softcap,
            "logit_scale": None,
        },
        "transformer_config": {
            "layer_configs": layer_configs,
            "output_norm_config": norm_config,
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
