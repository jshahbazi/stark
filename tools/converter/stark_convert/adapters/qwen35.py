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

from ..compressed import MLXMatrixSource, MLXQuantization, parse_mlx_quantization, plan_mlx_matrix, read_mlx_matrix
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
    "language_model_only",
    "output_gate_type",
    "shared_expert_intermediate_size",
    "norm_topk_prob",
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
            quantization=info.quantization,
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
    moe = config.get("model_type") == "qwen3_5_moe"
    nested_types = {"qwen3_5_moe", "qwen3_5_moe_text"} if moe else {"qwen3_5", "qwen3_5_text"}
    if text and text.get("model_type", next(iter(nested_types))) not in nested_types:
        raise UnsupportedConversion("Qwen3.5 dense adapter does not support a different nested text architecture")
    if config.get("model_type") not in {"qwen3_5", "qwen3_5_moe"}:
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
    quantization = parse_mlx_quantization(config, allow_mixed=True)
    unsupported: list[str] = []
    dim = _positive_int(config, "hidden_size")
    hidden = _positive_int(config, "shared_expert_intermediate_size" if moe else "intermediate_size")
    if moe:
        experts = _positive_int(config, "num_experts")
        active = _positive_int(config, "num_experts_per_tok")
        expert_hidden = _positive_int(config, "moe_intermediate_size")
        if experts > 1024 or active > min(experts, 32) or expert_hidden > 131072:
            raise UnsupportedConversion("Qwen3.5 MoE requires bounded expert banks")
        if quantization is None and not include_vision:
            raise UnsupportedConversion("Qwen3.5 BF16 MoE requires the image artifact reference policy")
        if not _bool(config, "norm_topk_prob", True):
            raise UnsupportedConversion("Qwen3.5 MoE requires normalized selected expert probabilities")
        if target != "BF16":
            raise UnsupportedConversion("Qwen3.5 MoE currently requires BF16 activations")
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
        "type",
        "mrope_section",
        "mrope_interleaved",
    }:
        unsupported.append("Qwen3.5 RoPE options are outside implemented text coverage")
    if rope_parameters.get("rope_type", "default") != "default":
        unsupported.append("Qwen3.5 scaled RoPE is outside implemented coverage")
    if rope_parameters.get("type", "default") != "default":
        unsupported.append("Qwen3.5 scaled RoPE alias is outside implemented coverage")
    if "mrope_section" in rope_parameters and (
        not isinstance(rope_parameters["mrope_section"], list)
        or any(type(item) is not int or item < 0 for item in rope_parameters["mrope_section"])
    ):
        raise ConversionError("Qwen3.5 mrope_section is malformed")
    if "mrope_interleaved" in rope_parameters:
        _bool(rope_parameters, "mrope_interleaved", False)
    annotations = {"model_name": (str, 256), "unsloth_version": (str, 64), "unsloth_fixed": (bool, None)}
    if include_vision and any(
        type(config[key]) is not expected_type or maximum is not None and not 0 < len(config[key]) <= maximum
        for key, (expected_type, maximum) in annotations.items()
        if key in config
    ):
        raise ConversionError("Qwen3.5 training annotation metadata is invalid")
    if set(config) - (_KEYS | set(annotations) if include_vision else _KEYS):
        unsupported.append("Qwen3.5 config options are outside implemented coverage")
    text_only = outer.get("model_type") == "qwen3_5_text"
    architectures = (
        (["Qwen3_5ForCausalLM"],) if text_only else (["Qwen3_5ForConditionalGeneration"], ["Qwen3_5ForCausalLM"])
    )
    if moe:
        architectures = (["Qwen3_5MoeForConditionalGeneration"], ["Qwen3_5MoeForCausalLM"])
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
    _bool(config, "language_model_only", False)
    # Transformers v5.8.0 qwen3_5/modeling_qwen3_5.py uses sigmoid at
    # the attention output (line 700), including the Qwen3.8-27B config's
    # inert `swish` annotation. Do not select a SiLU gate from this field.
    if config.get("output_gate_type", "sigmoid") not in {"sigmoid", "swish"}:
        unsupported.append("Unrecognized Qwen3.5 output gate annotation")
    if config.get("mamba_ssm_dtype", "float32") != "float32":
        unsupported.append("Qwen3.5 requires F32 recurrent state parameters")
    if config.get("mlp_only_layers", []) != []:
        unsupported.append("Qwen3.5 MLP-only layers are outside implemented coverage")
    if config.get("rope_scaling") is not None:
        unsupported.append("Qwen3.5 scaled RoPE is outside implemented coverage")
    for field in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
        if not moe and config.get(field) not in (None, 0):
            unsupported.append("Qwen3.5 MoE/MTP variants are outside implemented dense coverage")
    if not moe and any(field in config for field in ("shared_expert_intermediate_size", "norm_topk_prob")):
        unsupported.append("Qwen3.5 routed expert options require the MoE architecture wrapper")
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
            local_quantization = quantization.for_module(prefix)
            if local_quantization is None:
                raise ConversionError("Qwen3.5 packed tensor contradicts an unquantized module override")
            info = read_mlx_matrix(tensors, name, shape, local_quantization)
            expected.update((name, prefix + ".scales", prefix + ".biases"))
            return info
        return tensor(name, shape)

    def expert_source(name: str, rows: int, columns: int):
        """Flatten only the declared stacked expert axis; retain byte identity."""
        module = name.removesuffix(".weight")
        quant = quantization.for_module(module) if quantization is not None else None
        names = (name, module + ".scales", module + ".biases")
        packed = any(key in tensors for key in names[1:])
        if packed:
            if quant is None:
                raise ConversionError("MoE expert plane contradicts its quantization description")
            shapes = ((experts, rows, columns * quant.bits // 32), (experts, rows, columns // quant.group_size))
            flattened = {}
            for key, shape in zip(names, (shapes[0], shapes[1], shapes[1]), strict=True):
                info = tensors.get(key)
                if info is None or info.shape != shape:
                    raise ConversionError("MoE stacked expert planes disagree with the declared geometry")
                expected.add(key)
                flattened[key] = replace(info, shape=(experts * rows, shape[-1]))
            return read_mlx_matrix(flattened, name, (experts * rows, columns), quant)
        info = tensor(name, (experts, rows, columns))
        return replace(info, shape=(experts * rows, columns))

    def matrix(out: str, sources, shape: tuple[int, ...], layout="output_input"):
        nonlocal used_mlx
        packed = [isinstance(info, MLXMatrixSource) for info in sources]
        if any(packed):
            if not all(packed):
                raise UnsupportedConversion("Qwen3.5 mixed full-precision/MLX fusion is outside implemented coverage")
            source_quants = {info.quantization for info in sources}
            group_sizes = {item.group_size for item in source_quants}
            if len(group_sizes) != 1:
                raise UnsupportedConversion("Qwen3.5 affine fusion requires equal source group sizes")
            widths = {item.bits for item in source_quants}
            group_size = next(iter(group_sizes))
            native_packed = target == "BF16" and layout == "output_input" and group_size == 64
            if len(widths) > 1 and native_packed and not biases:
                # Preserve contiguous output intervals and the decoder's row
                # permutation. Coalesce adjacent equal-width intervals so a
                # split Q/gate layout does not dispatch once per head.
                segments = []
                grouped_sources = []
                for source in sources:
                    bits = source.quantization.bits
                    if segments and segments[-1]["bits"] == bits:
                        segments[-1]["rows"] += source.logical_shape[0]
                        grouped_sources[-1].append(source)
                    else:
                        segments.append({"rows": source.logical_shape[0], "bits": bits, "group_size": group_size})
                        grouped_sources.append([source])
                for index, (segment, group) in enumerate(zip(segments, grouped_sources, strict=True)):
                    operations.extend(
                        plan_mlx_matrix(
                            out + f".segment_{index}",
                            tuple(group),
                            dtype=target,
                            layout=layout,
                            quantization=MLXQuantization(segment["bits"], group_size),
                        )
                    )
                metadata[out + ".spec"] = json.dumps(
                    {
                        "type": "MLXRowPartitionedSpec",
                        "layout": layout,
                        "segments": segments,
                    }
                )
                used_mlx = True
                return
            allowed_bits = {4, 5, 6, 8} if native_packed else {4, 8}
            bits = next(iter(widths)) if len(widths) == 1 and widths <= allowed_bits else 8
            native_quantization = MLXQuantization(bits, next(iter(group_sizes)))
            operations.extend(
                plan_mlx_matrix(out, sources, dtype=target, layout=layout, quantization=native_quantization)
            )
            metadata[out + ".spec"] = json.dumps(
                {"type": "MLXSpec", "bits": bits, "group_size": native_quantization.group_size, "layout": layout}
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
        if moe:
            if quantization is None:
                # Official Transformers v5.8.0 stores [expert, gate+up, input].
                # Native dense experts consume [expert, up+gate, input]. Each
                # row view retains the source file, byte range and signature.
                bank = src + ".mlp.experts"
                fused = expert_source(bank + ".gate_up_proj", 2 * expert_hidden, dim)
                interleaved = tuple(
                    _matrix_row_range(fused, (2 * expert + branch) * expert_hidden, expert_hidden)
                    for expert in range(experts)
                    for branch in (1, 0)
                )
                down_name = bank + ".down_proj"
            else:
                bank = src + ".mlp.switch_mlp"
                up = expert_source(bank + ".up_proj.weight", expert_hidden, dim)
                gate = expert_source(bank + ".gate_proj.weight", expert_hidden, dim)
                interleaved = tuple(
                    _matrix_row_range(value, expert * expert_hidden, expert_hidden)
                    for expert in range(experts)
                    for value in (up, gate)
                )
                down_name = bank + ".down_proj.weight"
            matrix(out + ".mlp.experts.up_projection.weights", interleaved, (experts * 2 * expert_hidden, dim))
            matrix(
                out + ".mlp.experts.down_projection.weights",
                (expert_source(down_name, dim, expert_hidden),),
                (experts * dim, expert_hidden),
            )
            matrix(
                out + ".mlp.router.weights", (matrix_source(src + ".mlp.gate.weight", (experts, dim)),), (experts, dim)
            )
            matrix(
                out + ".mlp.shared_gate.weights",
                (matrix_source(src + ".mlp.shared_expert_gate.weight", (1, dim)),),
                (1, dim),
            )
        dense_source = src + (".mlp.shared_expert" if moe else ".mlp")
        dense_output = out + (".mlp.shared_expert" if moe else ".mlp")
        matrix(
            dense_output + ".up_projection.weights",
            tuple(matrix_source(dense_source + f".{branch}_proj.weight", (hidden, dim)) for branch in ("up", "gate")),
            (2 * hidden, dim),
        )
        matrix(
            dense_output + ".down_projection.weights",
            (matrix_source(dense_source + ".down_proj.weight", (dim, hidden)),),
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
        if not (vision and prefix in {"model.language_model", "language_model.model"}):
            raise UnsupportedConversion("Qwen3.5 vision requires a complete HF or MLX vision-language wrapper")
        vision_ops, vision_expected = build_qwen35_vision_plan(
            vision,
            tensors,
            dim,
            source_prefix="vision_tower." if prefix == "language_model.model" else "model.visual.",
        )
        operations.extend(vision_ops)
        expected.update(vision_expected)
    else:
        vision_prefix = "vision_tower." if prefix == "language_model.model" else "model.visual."
        ignored = {name for name in tensors if name.startswith(vision_prefix)} if text and vision else set()
    # The reference text loader also omits the declared auxiliary MTP decoder.
    # Untrusted prefixes alone never authorize dropping source tensors.
    if config.get("mtp_num_hidden_layers", 0) > 0:
        ignored.update(
            name
            for name in tensors
            if name.startswith(
                ("mtp.", "language_model.mtp.", "model.language_model.mtp.") if include_vision else "mtp."
            )
        )
    if quantization is not None and not used_mlx:
        raise ConversionError("Qwen3.5 MLX quantization configuration has no packed checkpoint matrices")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    norm_config = {
        "epsilon": epsilon,
        "scale_offset": None if quantization is not None else 1.0,
        # Qwen3.5 keeps zero-centered RMS normalization and its gain in FP32
        # through one final activation cast. Preserve the existing text export
        # profile while qualifying the exact full-vision artifact separately.
        "upcast_mode": "full_layer" if include_vision else "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    # RMSNormGated deliberately casts normalized values before its direct gain.
    delta_norm = {**norm_config, "scale_offset": None, "upcast_mode": "only_normalization"}
    mlp = {
        "type": "DenseMLPConfig",
        "linear_config": {},
        "activation": {"type": "SiLU", "alpha": 1.0},
        "has_up_biases": False,
        "has_down_biases": False,
        "up_clipping": None,
        "gate_clipping": None,
    }
    if moe:
        mlp = {
            "type": "MixtureOfExpertsConfig",
            "expert_config": mlp,
            "router_config": {},
            "routing_function": {"type": "SoftmaxRouting"},
            "num_routed_experts": experts,
            "num_active_routed_experts": active,
            "router_has_biases": False,
            "num_shared_experts": 1,
            "expert_hidden_dim": expert_hidden,
            "gate_config": {},
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
        if include_vision:
            # The pinned BF16 conditional model materializes its activation
            # boundaries inside DeltaNet instead of fusing them in FP32.
            mixer["reference_bf16"] = True
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
            if include_vision:
                mixer["reference_bf16"] = True
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
    from .qwen35_mtp import plan_qwen35_mtp

    mtp_config = plan_qwen35_mtp(
        tensors=tensors,
        declaration=config.get("mtp_num_hidden_layers", 0),
        dim=dim,
        hidden=hidden,
        heads=heads,
        groups=groups,
        head_dim=head_dim,
        target_layers=count,
        norm_config=norm_config,
        layer_configs=layer_configs,
        matrix_source=matrix_source,
        matrix=matrix,
        norm=norm,
        row_range=_matrix_row_range,
        include_vision=include_vision,
        quantization=quantization,
        moe=moe,
    )
    if mtp_config is not None:
        ignored = {
            name for name in ignored if not name.startswith(("mtp.", "language_model.mtp.", "model.language_model.mtp."))
        }
    if set(tensors) != expected | ignored:
        raise ConversionError("Qwen3.5 checkpoint has unexpected tensors")
    return ConversionPlan(
        decoder,
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
        mtp_config,
    )
