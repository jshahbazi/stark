"""Dense Qwen3.5 text/DeltaNet checkpoint planner.

Config, projection order and loader conventions are adapted from Lalamo
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
No Lalamo, JAX, Transformers or PyTorch dependency is imported here.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from ..compressed import MLXMatrixSource, parse_mlx_quantization, plan_mlx_matrix, read_mlx_matrix
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
from .dense import _COMMON_KEYS, _DTYPES, _bool, _positive_float, _positive_int
from .qwen35_vision import build_qwen35_vision_plan

_KEYS = _COMMON_KEYS | {
    "text_config",
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "full_attention_interval",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "output_router_logits",
    "router_aux_loss_coef",
    "mtp_num_hidden_layers",
    "use_bias",
    "use_qk_norm",
    "attn_output_gate",
    "mamba_ssm_dtype",
    "mlp_only_layers",
    "mtp_use_dedicated_embeddings",
    "partial_rotary_factor",
}


def normalize_qwen35_config(config: dict) -> dict:
    """Normalize explicit text exports or the wrapper's top-level precedence."""
    if config.get("model_type") == "qwen3_5_text":
        if "text_config" in config or config.get("vision_config") is not None:
            raise UnsupportedConversion("Qwen3.5 text-only config cannot declare a model wrapper")
        return {**config, "model_type": "qwen3_5"}
    text = config.get("text_config", {})
    if not isinstance(text, dict):
        raise ConversionError("Qwen3.5 text_config must be an object")
    return {**text, **{key: value for key, value in config.items() if key != "text_config"}}


def _row_range(info: TensorInfo, start: int, count: int) -> TensorInfo:
    """A contiguous row view carries the original file identity and signature."""
    rows, *rest = info.shape
    if start < 0 or count <= 0 or start + count > rows:
        raise ConversionError("Qwen3.5 projection row range is invalid")
    row_bytes = math.prod(rest) * info.itemsize
    return replace(info, shape=(count, *rest), offset=info.offset + start * row_bytes, nbytes=count * row_bytes)


def _matrix_row_range(info: TensorInfo | MLXMatrixSource, start: int, count: int):
    """Apply a decoder row permutation equally to all affine source planes."""
    if isinstance(info, MLXMatrixSource):
        return MLXMatrixSource(
            *(_row_range(plane, start, count) for plane in info.planes),
            logical_shape=(count, info.logical_shape[1]),
        )
    return _row_range(info, start, count)


def build_qwen35_plan(source: Path, *, dtype: str = "bfloat16", include_vision: bool = False) -> ConversionPlan:
    """Plan dense or existing affine MLX text weights from checkpoint dimensions.

    Split, directly fused and group-interleaved DeltaNet projections share this
    adapter. Vision weights are omitted only from an explicitly declared wrapper;
    every original file remains in the final signature/content verification set.
    """
    outer = read_json(source / "config.json")
    text = outer.get("text_config", {})
    config = normalize_qwen35_config(outer)
    if text and text.get("model_type", "qwen3_5_text") not in {"qwen3_5", "qwen3_5_text"}:
        raise UnsupportedConversion("Qwen3.5 dense adapter does not support a different nested text architecture")
    if config.get("model_type") != "qwen3_5":
        raise UnsupportedConversion("Qwen3.5 dense adapter requires model_type qwen3_5")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Qwen3.5 supports BF16, F16 and F32 output")
    if include_vision and dtype != "bfloat16":
        raise UnsupportedConversion("Qwen3.5 vision exports require BF16 output")
    target = _DTYPES[dtype]
    # The pinned Qwen importer uses this specific field to select baked norms.
    # An alias-only description cannot safely establish that interpretation.
    if config.get("quantization_config") is not None and config.get("quantization") is None:
        raise UnsupportedConversion("Qwen3.5 MLX requires the quantization field for baked normalization weights")
    quantization = parse_mlx_quantization(config)
    unsupported: list[str] = []
    dim = _positive_int(config, "hidden_size")
    hidden = _positive_int(config, "intermediate_size")
    count = _positive_int(config, "num_hidden_layers")
    vocab = _positive_int(config, "vocab_size")
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    linear_heads = _positive_int(config, "linear_num_value_heads")
    linear_groups = _positive_int(config, "linear_num_key_heads")
    key_dim = _positive_int(config, "linear_key_head_dim")
    value_dim = _positive_int(config, "linear_value_head_dim")
    kernel = _positive_int(config, "linear_conv_kernel_dim")
    if heads % groups or linear_heads % linear_groups:
        raise ConversionError("Qwen3.5 attention groups must divide their head count")
    if kernel < 2:
        raise ConversionError("Qwen3.5 convolution requires at least two coefficients")
    layers = config.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != count
        or any(not isinstance(layer, str) or layer not in {"linear_attention", "full_attention"} for layer in layers)
    ):
        raise ConversionError("Qwen3.5 layer_types must describe every layer")
    epsilon = _positive_float(config, "rms_norm_eps", 1e-6)
    context = _positive_int(config, "max_position_embeddings")
    tied = _bool(config, "tie_word_embeddings", False)
    biases = _bool(config, "attention_bias", False)
    rope_parameters = config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise ConversionError("Qwen3.5 requires rope_parameters")
    base = _positive_float(rope_parameters, "rope_theta", 10000.0)
    partial = _positive_float(rope_parameters, "partial_rotary_factor", 1.0)
    if "partial_rotary_factor" in config and _positive_float(config, "partial_rotary_factor", 1.0) != partial:
        raise ConversionError("Qwen3.5 rotary factors disagree")
    rotary = int(head_dim * partial)
    if partial > 1 or rotary < 2 or rotary % 2:
        raise ConversionError("Qwen3.5 rotary dimension must be positive and even")
    if set(rope_parameters) - {
        "rope_theta",
        "partial_rotary_factor",
        "rope_type",
        "mrope_section",
        "mrope_interleaved",
    }:
        unsupported.append("Qwen3.5 RoPE options are outside implemented text coverage")
    if rope_parameters.get("rope_type", "default") != "default":
        unsupported.append("Qwen3.5 scaled RoPE is outside implemented coverage")
    if "mrope_section" in rope_parameters and (
        not isinstance(rope_parameters["mrope_section"], list)
        or any(type(item) is not int or item < 0 for item in rope_parameters["mrope_section"])
    ):
        raise ConversionError("Qwen3.5 mrope_section is malformed")
    if "mrope_interleaved" in rope_parameters:
        _bool(rope_parameters, "mrope_interleaved", False)
    if set(config) - _KEYS:
        unsupported.append("Qwen3.5 config options are outside implemented coverage")
    text_only = outer.get("model_type") == "qwen3_5_text"
    architectures = (
        (["Qwen3_5ForCausalLM"],) if text_only else (["Qwen3_5ForConditionalGeneration"], ["Qwen3_5ForCausalLM"])
    )
    if config.get("architectures", architectures[0]) not in architectures:
        unsupported.append("Qwen3.5 architecture is outside implemented text coverage")
    if config.get("hidden_act", "silu") != "silu" or _bool(config, "mlp_bias", False):
        unsupported.append("Qwen3.5 requires unbiased SiLU MLPs")
    if _bool(config, "use_bias", False) or not _bool(config, "use_qk_norm", True):
        unsupported.append("Qwen3.5 projection or Q/K normalization options are unsupported")
    dropout = config.get("attention_dropout", 0.0)
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (float, int))
        or not math.isfinite(dropout)
        or not 0 <= dropout <= 1
    ):
        raise ConversionError("Qwen3.5 attention_dropout is malformed")
    if dropout != 0:
        unsupported.append("Qwen3.5 nonzero attention dropout is outside implemented coverage")
    if "full_attention_interval" in config:
        _positive_int(config, "full_attention_interval")
    if _positive_int(config, "pretraining_tp", 1) != 1:
        unsupported.append("Qwen3.5 tensor-parallel pretraining is outside implemented coverage")
    if config.get("pruned_heads", {}) != {}:
        unsupported.append("Qwen3.5 pruned heads are outside implemented coverage")
    if not _bool(config, "attn_output_gate", True):
        unsupported.append("Qwen3.5 requires gated full attention")
    if config.get("mamba_ssm_dtype", "float32") != "float32":
        unsupported.append("Qwen3.5 requires F32 recurrent state parameters")
    if config.get("mlp_only_layers", []) != []:
        unsupported.append("Qwen3.5 MLP-only layers are outside implemented coverage")
    if config.get("rope_scaling") is not None:
        unsupported.append("Qwen3.5 scaled RoPE is outside implemented coverage")
    for field in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
        if config.get(field) not in (None, 0):
            unsupported.append("Qwen3.5 MoE/MTP variants are outside implemented dense coverage")
    if "mtp_num_hidden_layers" in config and (
        type(config["mtp_num_hidden_layers"]) is not int or config["mtp_num_hidden_layers"] < 0
    ):
        raise ConversionError("Qwen3.5 MTP metadata is malformed")
    _bool(config, "mtp_use_dedicated_embeddings", False)
    for field in ("is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder"):
        if _bool(config, field, False):
            unsupported.append("Qwen3.5 encoder/cross-attention variants are unsupported")
    tensors = load_checkpoint(source)
    prefixes = [
        prefix
        for prefix in ("model.language_model", "language_model.model", "model")
        if prefix + ".embed_tokens.weight" in tensors
    ]
    if len(prefixes) != 1:
        raise ConversionError("Qwen3.5 checkpoint has ambiguous or missing text decoder root")
    prefix = prefixes[0]
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    used_mlx = False

    def tensor(name: str, shape: tuple[int, ...]) -> TensorInfo:
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Qwen3.5 checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("Qwen3.5 tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("Qwen3.5 checkpoint dtype is outside dense coverage")
        return info

    def payload(out: str, sources: tuple[TensorInfo, ...], shape: tuple[int, ...], output_dtype=target):
        if len({info.dtype for info in sources}) != 1:
            unsupported.append("Qwen3.5 mixed-dtype fusion promotion is outside implemented coverage")
        kind = "fuse" if len(sources) > 1 else "copy"
        if any(info.dtype != output_dtype for info in sources):
            kind = "fuse_cast" if len(sources) > 1 else "cast"
        operations.append(TensorOperation(out, sources, output_dtype, shape, kind))

    def matrix_source(name: str, shape: tuple[int, int]) -> TensorInfo | MLXMatrixSource:
        prefix = name.removesuffix(".weight")
        packed = any(prefix + suffix in tensors for suffix in (".scales", ".biases")) or (
            name in tensors and tensors[name].dtype in {"I32", "U32"}
        )
        if packed:
            if quantization is None:
                raise UnsupportedConversion("Qwen3.5 MLX checkpoint requires explicit quantization configuration")
            info = read_mlx_matrix(tensors, name, shape, quantization)
            expected.update((name, prefix + ".scales", prefix + ".biases"))
            return info
        return tensor(name, shape)

    def matrix(out: str, sources, shape: tuple[int, ...], layout="output_input"):
        nonlocal used_mlx
        packed = [isinstance(info, MLXMatrixSource) for info in sources]
        if any(packed):
            if not all(packed):
                raise UnsupportedConversion("Qwen3.5 mixed full-precision/MLX fusion is outside implemented coverage")
            operations.extend(plan_mlx_matrix(out, sources, dtype=target, layout=layout, quantization=quantization))
            metadata[out + ".spec"] = json.dumps(
                {"type": "MLXSpec", "bits": quantization.bits, "group_size": quantization.group_size, "layout": layout}
            )
            used_mlx = True
            return
        payload(out + ".weights", sources, shape)
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def norm(out: str, name: str, size: int):
        info = tensor(name, (size,))
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (size,), "cast_f32"))

    embed = matrix_source(prefix + ".embed_tokens.weight", (vocab, dim))
    embed_out = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    matrix(embed_out, (embed,), (vocab, dim), "input_output")
    head_names = [
        name
        for name in ("lm_head.weight", prefix + ".lm_head.weight", prefix.removesuffix(".model") + ".lm_head.weight")
        if any(name.removesuffix(".weight") + suffix in tensors for suffix in (".weight", ".scales", ".biases"))
    ]
    head_names = list(dict.fromkeys(head_names))
    if len(head_names) > 1 or (not tied and not head_names):
        raise ConversionError("Qwen3.5 checkpoint has ambiguous or missing lm_head")
    if head_names:
        head = matrix_source(head_names[0], (vocab, dim))
        if tied:
            if type(head) is not type(embed):
                raise ConversionError("Qwen3.5 tied lm_head representation differs from its embedding")
            head_planes = head.planes if isinstance(head, MLXMatrixSource) else (head,)
            embed_planes = embed.planes if isinstance(embed, MLXMatrixSource) else (embed,)
            if any(
                left.dtype != right.dtype or tensor_sha256(left) != tensor_sha256(right)
                for left, right in zip(head_planes, embed_planes, strict=True)
            ):
                raise ConversionError("Qwen3.5 tied lm_head differs from its embedding")
        else:
            matrix("decoder.embedding.output_embedding", (head,), (vocab, dim))
    q_width, kv_width = heads * head_dim, groups * head_dim
    key_width, value_width = linear_groups * key_dim, linear_heads * value_dim
    conv_width = 2 * key_width + value_width
    for index, layer in enumerate(layers):
        src, out = f"{prefix}.layers.{index}", f"decoder.transformer.layers.{index}"
        norm(out + ".pre_mixer_norm", src + ".input_layernorm.weight", dim)
        norm(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight", dim)
        if layer == "full_attention":
            attn = src + ".self_attn"
            explicit_gate = attn + ".gate_proj.weight" in tensors
            q = matrix_source(attn + ".q_proj.weight", (q_width if explicit_gate else 2 * q_width, dim))
            k = matrix_source(attn + ".k_proj.weight", (kv_width, dim))
            v = matrix_source(attn + ".v_proj.weight", (kv_width, dim))
            if explicit_gate:
                sources = (q, k, v, matrix_source(attn + ".gate_proj.weight", (q_width, dim)))
            else:
                queries = tuple(_matrix_row_range(q, h * 2 * head_dim, head_dim) for h in range(heads))
                gates = tuple(_matrix_row_range(q, h * 2 * head_dim + head_dim, head_dim) for h in range(heads))
                sources = (*queries, k, v, *gates)
            matrix(out + ".mixer.qkvg_projection.weights", sources, (2 * q_width + 2 * kv_width, dim))
            if biases:
                qb = tensor(attn + ".q_proj.bias", (q_width if explicit_gate else 2 * q_width,))
                kb = tensor(attn + ".k_proj.bias", (kv_width,))
                vb = tensor(attn + ".v_proj.bias", (kv_width,))
                if explicit_gate:
                    sources = (qb, kb, vb, tensor(attn + ".gate_proj.bias", (q_width,)))
                else:
                    sources = (
                        *(_row_range(qb, h * 2 * head_dim, head_dim) for h in range(heads)),
                        kb,
                        vb,
                        *(_row_range(qb, h * 2 * head_dim + head_dim, head_dim) for h in range(heads)),
                    )
                payload(out + ".mixer.qkvg_projection.biases", sources, (2 * q_width + 2 * kv_width,))
            matrix(
                out + ".mixer.out_projection.weights",
                (matrix_source(attn + ".o_proj.weight", (dim, q_width)),),
                (dim, q_width),
            )
            if biases:
                payload(out + ".mixer.out_projection.biases", (tensor(attn + ".o_proj.bias", (dim,)),), (dim,))
            norm(out + ".mixer.query_norm", attn + ".q_norm.weight", head_dim)
            norm(out + ".mixer.key_norm", attn + ".k_norm.weight", head_dim)
        else:
            attn = src + ".linear_attn"
            total = conv_width + value_width + 2 * linear_heads
            if attn + ".in_proj.weight" in tensors:
                sources = (matrix_source(attn + ".in_proj.weight", (total, dim)),)
            elif all(attn + f".in_proj_{branch}.weight" in tensors for branch in ("qkv", "z", "b", "a")):
                sources = tuple(
                    matrix_source(attn + f".in_proj_{branch}.weight", (width, dim))
                    for branch, width in (
                        ("qkv", conv_width),
                        ("z", value_width),
                        ("b", linear_heads),
                        ("a", linear_heads),
                    )
                )
            else:
                qkvz = matrix_source(attn + ".in_proj_qkvz.weight", (conv_width + value_width, dim))
                ba = matrix_source(attn + ".in_proj_ba.weight", (2 * linear_heads, dim))
                per_group_values = value_width // linear_groups
                block = 2 * key_dim + 2 * per_group_values
                qkvz_parts = tuple(
                    _matrix_row_range(qkvz, group * block + offset, width)
                    for offset, width in (
                        (0, key_dim),
                        (key_dim, key_dim),
                        (2 * key_dim, per_group_values),
                        (2 * key_dim + per_group_values, per_group_values),
                    )
                    for group in range(linear_groups)
                )
                per_group_heads = linear_heads // linear_groups
                ba_parts = tuple(
                    _matrix_row_range(ba, group * 2 * per_group_heads + offset, per_group_heads)
                    for offset in (0, per_group_heads)
                    for group in range(linear_groups)
                )
                sources = (*qkvz_parts, *ba_parts)
            matrix(out + ".mixer.in_proj.weights", sources, (total, dim))
            conv_names = [
                name
                for name in (attn + ".conv1d.weight", attn + ".conv_weight", attn + ".conv.weight")
                if name in tensors
            ]
            if len(conv_names) != 1:
                raise ConversionError("Qwen3.5 convolution weights are missing or ambiguous")
            conv_name = conv_names[0]
            shape = tensors[conv_name].shape
            if shape not in ((conv_width, 1, kernel), (conv_width, kernel), (conv_width, kernel, 1)):
                raise ConversionError("Qwen3.5 convolution shape disagrees with its config")
            conv = tensor(conv_name, shape)
            operations.append(
                TensorOperation(out + ".mixer.conv.weights", (conv,), "F32", (conv_width, kernel), "cast_f32")
            )
            matrix(
                out + ".mixer.out_proj.weights",
                (matrix_source(attn + ".out_proj.weight", (dim, value_width)),),
                (dim, value_width),
            )
            norm(out + ".mixer.norm", attn + ".norm.weight", value_dim)
            for input_name, output_name in (("A_log", "a_log"), ("dt_bias", "dt_bias")):
                info = tensor(attn + "." + input_name, (linear_heads,))
                operations.append(
                    TensorOperation(out + ".mixer." + output_name, (info,), "F32", (linear_heads,), "cast_f32")
                )
        matrix(
            out + ".mlp.up_projection.weights",
            tuple(matrix_source(src + f".mlp.{branch}_proj.weight", (hidden, dim)) for branch in ("up", "gate")),
            (2 * hidden, dim),
        )
        matrix(
            out + ".mlp.down_projection.weights",
            (matrix_source(src + ".mlp.down_proj.weight", (dim, hidden)),),
            (dim, hidden),
        )
    norm("decoder.transformer.output_norm", prefix + ".norm.weight", dim)
    vision = outer.get("vision_config")
    if vision is not None and not isinstance(vision, dict):
        raise ConversionError("Qwen3.5 vision_config must be an object")
    # The pinned decoder root dispatch also accepts MLX-VL's language_model.model.
    # Its declared visual tower has a separate exact namespace; do not permit
    # another wrapper's visual alias to hide an otherwise unconsumed tensor.
    ignored = set()
    if include_vision:
        if not (vision and prefix == "model.language_model"):
            raise UnsupportedConversion(
                "Qwen3.5 vision requires the official model.language_model/model.visual wrapper"
            )
        vision_ops, vision_expected = build_qwen35_vision_plan(vision, tensors, dim)
        operations.extend(vision_ops)
        expected.update(vision_expected)
    else:
        vision_prefix = "vision_tower." if prefix == "language_model.model" else "model.visual."
        ignored = {name for name in tensors if name.startswith(vision_prefix)} if text and vision else set()
    # The reference text loader also omits the declared auxiliary MTP decoder.
    # Untrusted prefixes alone never authorize dropping source tensors.
    if config.get("mtp_num_hidden_layers", 0) > 0:
        ignored.update(name for name in tensors if name.startswith("mtp."))
    if set(tensors) != expected | ignored:
        raise ConversionError("Qwen3.5 checkpoint has unexpected tensors")
    if quantization is not None and not used_mlx:
        raise ConversionError("Qwen3.5 MLX quantization configuration has no packed checkpoint matrices")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    norm_config = {
        "epsilon": epsilon,
        "scale_offset": None if quantization is not None else 1.0,
        "upcast_mode": "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    delta_norm = {**norm_config, "scale_offset": None}
    mlp = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": {"type": "SiLU", "alpha": 1.0},
        "has_up_biases": False,
        "has_down_biases": False,
        "up_clipping": None,
        "gate_clipping": None,
    }
    rope = {"type": "UnscaledRoPEConfig", "base": base, "max_sequence_length": context, "head_dim": rotary}
    layer_configs = []
    for layer in layers:
        mixer = {
            "type": "DeltaNetConfig",
            "in_proj_config": {},
            "conv_config": {"has_biases": False},
            "out_proj_config": {},
            "norm_config": delta_norm,
            "num_heads": linear_heads,
            "num_groups": linear_groups,
            "head_dim": key_dim,
            "value_head_dim": value_dim,
            "kernel_size": kernel,
        }
        if layer == "full_attention":
            mixer = {
                "type": "AttentionConfig",
                "qkvg_projection_config": {},
                "out_projection_config": {},
                "query_norm_config": norm_config,
                "key_norm_config": norm_config,
                "logit_soft_cap": None,
                "has_sinks": False,
                "has_qkvg_biases": biases,
                "has_out_biases": biases,
                "num_heads": heads,
                "num_groups": groups,
                "head_dim": head_dim,
                "is_causal": True,
                "scale": None,
                "sliding_window_size": None,
                "has_gate": True,
                "normalize_values": False,
                "is_kv_sharing": False,
            }
        layer_configs.append(
            {
                "pre_mixer_norm_config": norm_config,
                "mixer_conv_config": None,
                "mixer_config": mixer,
                "post_mixer_norm_config": None,
                "pre_mlp_norm_config": norm_config,
                "mlp_conv_config": None,
                "mlp_config": mlp,
                "post_mlp_norm_config": None,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": rope if layer == "full_attention" else None,
            }
        )
    decoder = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig" if tied else "UntiedEmbeddingConfig",
            "input_scale": None,
            "logit_soft_cap": None,
            "logit_scale": None,
        },
        "transformer_config": {
            "layer_configs": layer_configs,
            "output_norm_config": norm_config,
            "model_dim": dim,
            "hidden_dim": hidden,
        },
        "vocab_size": vocab,
        "ple_model_config": None,
        "embedding_norm_config": None,
    }
    return ConversionPlan(
        decoder,
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )
