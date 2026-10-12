"""Dense ModernBERT sequence-classifier conversion without framework imports.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
)

_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_KEYS = {
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "classifier_activation",
    "classifier_bias",
    "classifier_dropout",
    "classifier_pooling",
    "cls_token_id",
    "decoder_bias",
    "deterministic_flash_attn",
    "embedding_dropout",
    "eos_token_id",
    "global_attn_every_n_layers",
    "global_rope_theta",
    "gradient_checkpointing",
    "hidden_activation",
    "hidden_size",
    "initializer_cutoff_factor",
    "initializer_range",
    "intermediate_size",
    "layer_norm_eps",
    "local_attention",
    "local_rope_theta",
    "max_position_embeddings",
    "mlp_bias",
    "mlp_dropout",
    "model_type",
    "norm_bias",
    "norm_eps",
    "num_attention_heads",
    "num_hidden_layers",
    "pad_token_id",
    "position_embedding_type",
    "sep_token_id",
    "transformers_version",
    "vocab_size",
    "id2label",
    "label2id",
    "num_labels",
    "quantization",
    "quantization_config",
    "torch_dtype",
    "dtype",
    "_name_or_path",
    "_attn_implementation",
    "_attn_implementation_autoset",
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
    "use_cache",
    "tie_word_embeddings",
    "reference_compile",
    "repad_logits_with_grad",
    "sparse_prediction",
    "sparse_pred_ignore_index",
    "rope_scaling",
    "rope_parameters",
    "layer_types",
    "head_dim",
    "num_key_value_heads",
}


def _positive_int(config: dict, key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise ConversionError(f"ModernBERT config requires a positive integer {key}")
    return value


def _positive_float(config: dict, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConversionError(f"ModernBERT config requires a finite positive {key}")
    return float(value)


def _bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConversionError(f"ModernBERT config requires boolean {key}")
    return value


def _labels(config: dict) -> list[str]:
    ids, names = config.get("id2label"), config.get("label2id")
    if not isinstance(ids, dict) or not ids or not isinstance(names, dict) or len(ids) != len(names):
        raise ConversionError("ModernBERT requires both complete classifier label maps")
    expected_keys = {str(index) for index in range(len(ids))}
    if set(ids) != expected_keys or any(not isinstance(value, str) or not value for value in ids.values()):
        raise ConversionError("ModernBERT label identifiers must be contiguous canonical integers")
    ordered = [ids[str(index)] for index in range(len(ids))]
    if len(set(ordered)) != len(ordered) or any(type(value) is not int for value in names.values()):
        raise ConversionError("ModernBERT label maps contain duplicate or invalid labels")
    if names != {label: index for index, label in enumerate(ordered)}:
        raise ConversionError("ModernBERT label maps disagree")
    if "num_labels" in config and _positive_int(config, "num_labels") != len(ordered):
        raise ConversionError("ModernBERT num_labels disagrees with its label maps")
    return ordered


def build_modernbert_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Return a complete classifier plan; its legacy decoder_config field holds ClassifierConfig.

    The CLI chooses classifier_config for this architecture. Wi row halves are
    reversed with two bounded slices to reproduce the pinned classifier loader.
    """
    config = read_json(source / "config.json")
    if config.get("model_type") != "modernbert":
        raise UnsupportedConversion("Checkpoint model_type does not match ModernBERT")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("ModernBERT supports BF16, F16, and F32")
    target = _DTYPES[dtype]
    model = _positive_int(config, "hidden_size")
    hidden = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    vocab = _positive_int(config, "vocab_size")
    maximum = _positive_int(config, "max_position_embeddings", 8192)
    periodicity = _positive_int(config, "global_attn_every_n_layers", 3)
    window = _positive_int(config, "local_attention", 128)
    epsilon = _positive_float(config, "norm_eps", 1e-5)
    _positive_float(config, "layer_norm_eps", epsilon)
    global_base = _positive_float(config, "global_rope_theta", 160000.0)
    local_base = _positive_float(config, "local_rope_theta", 10000.0)
    if model % heads or (model // heads) % 2:
        raise ConversionError("ModernBERT hidden size must divide into even rotary attention heads")
    head_dim = model // heads
    attention_bias = _bool(config, "attention_bias", False)
    classifier_bias = _bool(config, "classifier_bias", False)
    labels = _labels(config)
    unsupported: list[str] = []
    if set(config) - _KEYS:
        unsupported.append("Config options are outside implemented ModernBERT coverage")
    if config.get("architectures", ["ModernBertForSequenceClassification"]) != ["ModernBertForSequenceClassification"]:
        unsupported.append("ModernBERT architecture requires sequence classification")
    for key in ("hidden_activation", "classifier_activation"):
        if config.get(key, "gelu") != "gelu":
            unsupported.append("ModernBERT requires exact GELU activation")
    if config.get("classifier_pooling", "cls") != "mean":
        unsupported.append("ModernBERT supports the pinned mean-pooling classifier")
    # This annotation is retained by the pinned importer; attention actually uses RoPE.
    if config.get("position_embedding_type", "absolute") != "absolute":
        unsupported.append("ModernBERT positional annotation is outside implemented coverage")
    for key in (
        "norm_bias",
        "mlp_bias",
        "is_decoder",
        "is_encoder_decoder",
        "add_cross_attention",
        "tie_encoder_decoder",
    ):
        if _bool(config, key, False):
            unsupported.append("ModernBERT bias or encoder variant is outside implemented coverage")
    for key in ("attention_dropout", "classifier_dropout", "embedding_dropout", "mlp_dropout"):
        value = config.get(key, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ConversionError("ModernBERT requires finite dropout in the interval [0, 1]")
        # Training dropout does not affect the pinned inference representation.
    for key in (
        "deterministic_flash_attn",
        # This controls the unused masked-LM readout, absent from sequence classifiers.
        "decoder_bias",
        "gradient_checkpointing",
        "return_dict",
        "output_attentions",
        "output_hidden_states",
        "torchscript",
        "use_cache",
        "tie_word_embeddings",
        "repad_logits_with_grad",
        "sparse_prediction",
    ):
        if key in config:
            _bool(config, key, False)
    for key in ("quantization", "quantization_config", "rope_scaling", "rope_parameters", "layer_types"):
        if config.get(key) is not None:
            unsupported.append("Compressed or custom-attention ModernBERT config is outside implemented coverage")
    explicit_head_dim = _positive_int(config, "head_dim", head_dim)
    explicit_groups = _positive_int(config, "num_key_value_heads", heads)
    if explicit_head_dim != head_dim or explicit_groups != heads:
        unsupported.append("ModernBERT attention dimensions disagree with the pinned encoder")
    if config.get("pruned_heads", {}) != {}:
        unsupported.append("Pruned ModernBERT heads are outside implemented coverage")
    tensors = load_checkpoint(source)
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    expected: set[str] = set()

    def tensor(name: str, shape: tuple[int, ...]) -> TensorInfo:
        expected.add(name)
        if name not in tensors:
            raise ConversionError("ModernBERT checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("ModernBERT tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("ModernBERT requires full-precision floating-point checkpoint tensors")
        return info

    def array(out: str, name: str, shape: tuple[int, ...], output_dtype: str = target) -> None:
        info = tensor(name, shape)
        kind = "copy" if info.dtype == output_dtype else "cast"
        operations.append(TensorOperation(out, (info,), output_dtype, shape, kind))

    def matrix(
        out: str, name: str, shape: tuple[int, ...], layout: str = "output_input", reverse: bool = False
    ) -> None:
        info = tensor(name, shape)
        sources = (info,)
        kind = "copy" if info.dtype == target else "cast"
        if reverse:
            rows = shape[0] // 2
            half = info.nbytes // 2
            first = TensorInfo(info.path, info.dtype, (rows, shape[1]), info.offset, half, info.file_signature)
            second = TensorInfo(info.path, info.dtype, (rows, shape[1]), info.offset + half, half, info.file_signature)
            sources = (second, first)
            kind = "fuse" if info.dtype == target else "fuse_cast"
        operations.append(TensorOperation(out + ".weights", sources, target, shape, kind))
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def norm(out: str, name: str) -> None:
        array(out + ".scales", name + ".weight", (model,), "F32")

    matrix("classifier.embedding.embedding", "model.embeddings.tok_embeddings.weight", (vocab, model), "input_output")
    norm("classifier.embedding_norm", "model.embeddings.norm")
    for index in range(layers):
        src, out = f"model.layers.{index}", f"classifier.transformer.layers.{index}"
        if index:
            norm(out + ".pre_mixer_norm", src + ".attn_norm")
        norm(out + ".pre_mlp_norm", src + ".mlp_norm")
        matrix(out + ".mixer.qkvg_projection.weights", src + ".attn.Wqkv.weight", (3 * model, model))
        if attention_bias:
            array(out + ".mixer.qkvg_projection.biases", src + ".attn.Wqkv.bias", (3 * model,))
        matrix(out + ".mixer.out_projection.weights", src + ".attn.Wo.weight", (model, model))
        matrix(out + ".mlp.up_projection.weights", src + ".mlp.Wi.weight", (2 * hidden, model), reverse=True)
        matrix(out + ".mlp.down_projection.weights", src + ".mlp.Wo.weight", (model, hidden))
    norm("classifier.transformer.output_norm", "model.final_norm")
    matrix("classifier.prediction_head.dense.weights", "head.dense.weight", (model, model))
    if classifier_bias:
        array("classifier.prediction_head.dense.biases", "head.dense.bias", (model,))
    norm("classifier.prediction_head.norm", "head.norm")
    matrix("classifier.prediction_head.readout.weights", "classifier.weight", (len(labels), model))
    array("classifier.prediction_head.readout.biases", "classifier.bias", (len(labels),))
    if set(tensors) != expected:
        raise ConversionError("ModernBERT checkpoint has unexpected tensors")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    norm_config = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "only_normalization",
        "subtract_mean": True,
        "has_scale": True,
        "has_biases": False,
    }
    activation = {"type": "GELU", "approximate": False}
    mlp_config = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": activation,
        "has_up_biases": False,
        "has_down_biases": False,
        "up_clipping": None,
        "gate_clipping": None,
    }
    layer_configs = []
    for index in range(layers):
        local = index % periodicity != 0
        layer_configs.append(
            {
                "pre_mixer_norm_config": norm_config if index else None,
                "mixer_conv_config": None,
                "mixer_config": {
                    "type": "AttentionConfig",
                    "qkvg_projection_config": {},
                    "out_projection_config": {},
                    "query_norm_config": None,
                    "key_norm_config": None,
                    "logit_soft_cap": None,
                    "has_sinks": False,
                    "has_qkvg_biases": attention_bias,
                    "has_out_biases": False,
                    "num_heads": heads,
                    "num_groups": heads,
                    "head_dim": head_dim,
                    "scale": None,
                    "is_causal": False,
                    "sliding_window_size": window if local else None,
                    "has_gate": False,
                    "normalize_values": False,
                    "is_kv_sharing": False,
                },
                "post_mixer_norm_config": None,
                "pre_mlp_norm_config": norm_config,
                "mlp_conv_config": None,
                "mlp_config": mlp_config,
                "post_mlp_norm_config": None,
                "rope_config": {
                    "type": "UnscaledRoPEConfig",
                    "base": local_base if local else global_base,
                    "max_sequence_length": maximum,
                    "head_dim": head_dim,
                },
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
            }
        )
    classifier_config = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig",
            "input_scale": None,
            "logit_soft_cap": None,
            "logit_scale": None,
        },
        "embedding_norm_config": norm_config,
        "transformer_config": {
            "layer_configs": layer_configs,
            "output_norm_config": norm_config,
            "model_dim": model,
            "hidden_dim": hidden,
        },
        "prediction_head_config": {
            "dense_config": {},
            "activation": activation,
            "normalization_config": {**norm_config, "scale_offset": 0.0},
            "readout_config": {},
            "use_dense_bias": classifier_bias,
        },
        "vocab_size": vocab,
        "model_dim": model,
        "hidden_dim": model,
        "num_labels": len(labels),
        "classifier_pooling": "mean",
        "output_labels": labels,
    }
    return ConversionPlan(
        classifier_config,
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )
