"""Conservative dense Hugging Face Qwen3 to Stark conversion planner.

Config/tensor conventions adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See the bundled Lalamo license/provenance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
    tensor_sha256,
)

_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_ALLOWED_CONFIG_KEYS = {
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
    "use_sliding_window",
    "sliding_window",
    "max_window_layers",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
    "rope_scaling",
    "rope_parameters",
    "layer_types",
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


def _positive_int(config: dict, key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise ConversionError("Qwen3 requires positive integer dimensions")
    return value


def _bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConversionError("Qwen3 configuration has an invalid boolean")
    return value


def _positive_float(config: dict, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConversionError("Qwen3 configuration has an invalid numeric parameter")
    return float(value)


def build_qwen3_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Plan all output tensors from strict checkpoint headers before writing."""
    config = read_json(source / "config.json")
    if config.get("model_type") != "qwen3":
        raise UnsupportedConversion("Streaming prototype supports dense Qwen3")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Streaming prototype supports BF16, F16, and F32")
    target_dtype = _DTYPES[dtype]
    if set(config) - _ALLOWED_CONFIG_KEYS:
        raise UnsupportedConversion("Qwen3 config has options outside validated streaming coverage")
    if config.get("architectures", ["Qwen3ForCausalLM"]) != ["Qwen3ForCausalLM"]:
        raise UnsupportedConversion("Qwen3 architecture is outside validated streaming coverage")
    if config.get("hidden_act", "silu") != "silu":
        raise UnsupportedConversion("Qwen3 streaming requires SiLU")
    if any(config.get(key) is not None for key in ("quantization_config", "rope_scaling", "rope_parameters")):
        raise UnsupportedConversion("Quantization or scaled/custom RoPE is outside implemented coverage")
    if config.get("attention_dropout", 0.0) != 0.0:
        raise UnsupportedConversion("Non-default Qwen3 attention dropout is outside implemented coverage")
    if any(config.get(key, False) for key in ("is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder")):
        raise UnsupportedConversion("Encoder/cross-attention settings are outside implemented coverage")
    if config.get("pruned_heads", {}):
        raise UnsupportedConversion("Pruned attention heads are outside implemented coverage")
    model_dim = _positive_int(config, "hidden_size")
    hidden_dim = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads", heads)
    if heads % groups:
        raise ConversionError("Qwen3 attention groups must divide its head count")
    if "head_dim" not in config and model_dim % heads:
        raise ConversionError("Qwen3 head dimension cannot be inferred")
    head_dim = _positive_int(config, "head_dim", model_dim // heads)
    if head_dim % 2:
        raise ConversionError("Qwen3 rotary head dimension must be even")
    vocab = _positive_int(config, "vocab_size")
    max_sequence_length = _positive_int(config, "max_position_embeddings", 32768)
    epsilon = _positive_float(config, "rms_norm_eps", 1e-6)
    theta = _positive_float(config, "rope_theta", 10000.0)
    tied = _bool(config, "tie_word_embeddings", False)
    attention_bias = _bool(config, "attention_bias", False)
    use_sliding = _bool(config, "use_sliding_window", False)
    max_window_layers = config.get("max_window_layers", layers)
    if type(max_window_layers) is not int or not 0 <= max_window_layers <= layers:
        raise ConversionError("Qwen3 sliding window layer boundary is invalid")
    sliding = _positive_int(config, "sliding_window") if use_sliding else None
    windows = [sliding if use_sliding and layer >= max_window_layers else None for layer in range(layers)]
    if config.get("layer_types") is not None:
        expected = ["sliding_attention" if window is not None else "full_attention" for window in windows]
        if config["layer_types"] != expected:
            raise UnsupportedConversion("Qwen3 layer_types disagree with validated sliding-window semantics")
    norm = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    rope = {
        "type": "UnscaledRoPEConfig",
        "base": theta,
        "max_sequence_length": max_sequence_length,
        "head_dim": head_dim,
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
            "query_norm_config": norm,
            "key_norm_config": norm,
            "num_heads": heads,
            "num_groups": groups,
            "head_dim": head_dim,
            "is_causal": True,
            "scale": None,
            "sliding_window_size": window,
            "logit_soft_cap": None,
            "has_sinks": False,
            "has_qkvg_biases": attention_bias,
            "has_out_biases": attention_bias,
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
    tensors = load_checkpoint(source)
    operations = []
    expected = set()
    metadata = {}
    unsupported_source_dtype = False
    needs_dense_dtype_change = False

    def tensor(name: str, shape: tuple[int, ...], *, norm_scale: bool = False):
        nonlocal unsupported_source_dtype, needs_dense_dtype_change
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Qwen3 checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("Qwen3 checkpoint tensor shape disagrees with its config")
        if info.dtype not in {"BF16", "F16", "F32"}:
            unsupported_source_dtype = True
        if not norm_scale and info.dtype != target_dtype:
            needs_dense_dtype_change = True
        return info

    def matrix(
        out: str, names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...], shape: tuple[int, ...], layout: str
    ):
        sources = tuple(tensor(name, source_shape) for name, source_shape in zip(names, shapes, strict=True))
        operations.append(
            TensorOperation(out + ".weights", sources, target_dtype, shape, "fuse" if len(sources) > 1 else "copy")
        )
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def scale(out: str, name: str, width: int):
        info = tensor(name, (width,), norm_scale=True)
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (width,), "cast_f32"))

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
        scale(out + ".pre_mixer_norm", src + ".input_layernorm.weight", model_dim)
        scale(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight", model_dim)
        scale(out + ".mixer.query_norm", src + ".self_attn.q_norm.weight", head_dim)
        scale(out + ".mixer.key_norm", src + ".self_attn.k_norm.weight", head_dim)
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
            sources = tuple(
                tensor(src + f".self_attn.{name}_proj.bias", (width,))
                for name, width in (("q", q_width), ("k", kv_width), ("v", kv_width))
            )
            operations.append(
                TensorOperation(
                    out + ".mixer.qkvg_projection.biases", sources, target_dtype, (q_width + 2 * kv_width,), "fuse"
                )
            )
            info = tensor(src + ".self_attn.o_proj.bias", (model_dim,))
            operations.append(
                TensorOperation(out + ".mixer.out_projection.biases", (info,), target_dtype, (model_dim,))
            )
    scale("decoder.transformer.output_norm", "model.norm.weight", model_dim)
    if set(tensors) != expected:
        raise ConversionError("Qwen3 checkpoint has unexpected tensors")
    if unsupported_source_dtype:
        raise UnsupportedConversion("Unsupported Qwen3 checkpoint dtype")
    if needs_dense_dtype_change:
        raise UnsupportedConversion("Qwen3 dense dtype conversion is not implemented")
    return ConversionPlan(
        decoder_config,
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )
