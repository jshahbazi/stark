"""Quantization-preserving Qwen3.8 text artifact planner.

The architecture and fixed projection policy follow Lily revision
e22d58fa7cc488493fb1b5d810a198bd29730aeb, Apache-2.0; see
LICENSE.qwen38-reference, NOTICE.qwen38-reference and docs/QWEN38_UZU_PROVENANCE.md. Upstream attribution names
Copyright 2026 Perplexity AI, Inc. and Copyright 2026 Fabio Greter.
Stark's Uzu extension uses typed configuration and canonical MLXSpec storage.
No checkpoint Python, quantizer, or source-folder runtime bridge is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..compressed import MLXQuantization, plan_mlx_matrix, read_mlx_matrix
from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    _open_tensor,
    load_checkpoint,
    read_json,
    read_safetensors_header,
    tensor_sha256,
    validate_planning_layers,
)

ROOT_TYPE = "Qwen38LanguageModelConfig"
_GROUPS = {"attn", "gdn", "shared", "head", "embed"}
_TEXT_INTEGER_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "max_position_embeddings",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "hc_count",
    "hc_lowrank",
    "ple_embed_dim",
    "ple_conv_kernel_size",
    "ngram_size",
    "heads_per_ngram",
    "split_ngram_parts",
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def _positive_int(value: object, label: str, maximum: int = 1_048_576) -> int:
    _require(type(value) is int and 0 < value <= maximum, f"Invalid Qwen3.8 {label}")
    return value


def _positive_float(value: object, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"Invalid Qwen3.8 {label}")
    return float(value)


def _quantization(value: object, *, bits: int, group_size: int) -> dict:
    _require(isinstance(value, dict), "Missing Qwen3.8 affine policy")
    _require(value.get("mode", "affine") == "affine", "Qwen3.8 requires MLX affine weights")
    _require(type(value.get("bits")) is int and value["bits"] == bits, "Unsupported Qwen3.8 bit width")
    _require(
        type(value.get("group_size")) is int and value["group_size"] == group_size, "Unsupported Qwen3.8 group size"
    )
    return {"bits": bits, "group_size": group_size}


def normalize_qwen38_config(config: dict) -> dict:
    """Return metadata-writer text fields without altering architecture identity."""
    _require(config.get("model_type") == "qwen4_exp", "Qwen3.8 requires model_type qwen4_exp")
    text = config.get("text_config")
    _require(isinstance(text, dict) and text.get("model_type") == "qwen4_exp_text", "Qwen3.8 requires qwen4_exp_text")
    return {
        **text,
        "model_type": "qwen4_exp",
        **({"_stark_model_repo": config["_stark_model_repo"]} if "_stark_model_repo" in config else {}),
    }


def architecture_config(config: dict) -> tuple[dict, tuple[re.Pattern, ...]]:
    """Select only implemented text fields for the typed executable schema."""
    text = normalize_qwen38_config(config)
    architecture = {field: _positive_int(text.get(field), field) for field in _TEXT_INTEGER_FIELDS}
    validate_planning_layers(architecture["num_hidden_layers"])
    for field, default in (("tie_word_embeddings", False), ("norm_topk_prob", True)):
        value = text.get(field, default)
        _require(type(value) is bool, f"Invalid Qwen3.8 {field}")
        architecture[field] = value
    _require(not architecture["tie_word_embeddings"], "Qwen3.8 tied embeddings are not implemented")
    _require(
        text.get("hidden_act") == "silu" and text.get("attention_bias", False) is False,
        "Unsupported Qwen3.8 activation or attention bias",
    )
    architecture["hidden_act"] = "silu"
    architecture["rms_norm_eps"] = _positive_float(text.get("rms_norm_eps"), "normalization epsilon")
    gate = text.get("output_gate_type", text["hidden_act"])
    _require(gate in {"silu", "sigmoid"}, "Unsupported Qwen3.8 output gate")
    architecture["output_gate_type"] = gate
    count = architecture["num_hidden_layers"]
    layers = text.get("layer_types")
    _require(
        isinstance(layers, list)
        and len(layers) == count
        and all(isinstance(layer, str) and layer in {"linear_attention", "full_attention"} for layer in layers),
        "Invalid Qwen3.8 layer schedule",
    )
    architecture["layer_types"] = layers.copy()
    ple_ids = text.get("ple_layer_ids", [])
    _require(
        isinstance(ple_ids, list)
        and len(ple_ids) <= 1
        and all(type(index) is int and 1 <= index <= count for index in ple_ids),
        "Unsupported Qwen3.8 PLE schedule",
    )
    architecture["ple_layer_ids"] = ple_ids.copy()
    eos = text.get("eos_token_id")
    eos = [eos] if type(eos) is int else eos
    _require(
        isinstance(eos, list)
        and 0 < len(eos) <= 32
        and all(type(value) is int and 0 <= value < architecture["vocab_size"] for value in eos),
        "Invalid Qwen3.8 EOS identifiers",
    )
    architecture["eos_token_id"] = list(dict.fromkeys(eos))
    rope = text.get("rope_parameters")
    _require(
        isinstance(rope, dict) and rope.get("rope_type", "default") == "default",
        "Unsupported Qwen3.8 rotary configuration",
    )
    selected_rope = {
        "rope_theta": _positive_float(rope.get("rope_theta"), "rotary base"),
        "partial_rotary_factor": _positive_float(rope.get("partial_rotary_factor"), "rotary factor"),
    }
    rotary = architecture["head_dim"] * selected_rope["partial_rotary_factor"]
    _require(
        rotary.is_integer() and 0 < rotary <= architecture["head_dim"] and int(rotary) % 2 == 0,
        "Invalid Qwen3.8 rotary dimensions",
    )
    for field in ("mrope_section", "mrope_interleaved", "rope_type"):
        selected_rope[field] = rope.get(field)
    architecture["rope_parameters"] = selected_rope
    _require(
        architecture["num_attention_heads"] % architecture["num_key_value_heads"] == 0
        and architecture["linear_num_value_heads"] % architecture["linear_num_key_heads"] == 0
        and architecture["indexer_kv_heads"] == 1
        and architecture["num_experts_per_tok"] <= architecture["num_experts"]
        and architecture["linear_conv_kernel_dim"] >= 2
        and architecture["ple_conv_kernel_size"] >= 2
        and architecture["ngram_size"] >= 2
        and architecture["indexer_budget"] % architecture["indexer_compress_ratio"] == 0,
        "Unsupported Qwen3.8 geometry",
    )
    lily = config.get("lily")
    _require(
        isinstance(lily, dict) and lily.get("format") == "qwen4_exp-affine-v1",
        "Qwen3.8 requires an explicit Lily affine source contract",
    )
    q = lily.get("quantization")
    _require(isinstance(q, dict), "Missing Qwen3.8 quantization policy")
    default = _quantization(q.get("default"), bits=4, group_size=64)
    ngram = _quantization(q.get("ngram_embedding"), bits=4, group_size=32)
    groups = q.get("q8_groups")
    if groups is None:
        groups = ["attn", "gdn", "shared", "head"] if q.get("q8_dense", False) else []
        groups += ["embed"] if q.get("q8_embed", False) else []
    else:
        _require(not q.get("q8_dense", False) and not q.get("q8_embed", False), "Conflicting Qwen3.8 Q8 declarations")
    _require(
        isinstance(groups, list)
        and all(isinstance(group, str) and group in _GROUPS for group in groups)
        and len(groups) == len(set(groups)),
        "Invalid Qwen3.8 Q8 groups",
    )
    architecture["quantization"] = {"default": default, "ngram_embedding": ngram, "q8_groups": groups.copy()}
    patterns = q.get("q8_suffixes", [])
    _require(
        isinstance(patterns, list)
        and len(patterns) <= 64
        and all(isinstance(pattern, str) and len(pattern) <= 512 for pattern in patterns),
        "Invalid Qwen3.8 Q8 suffix rules",
    )
    try:
        regexes = tuple(re.compile(pattern) for pattern in patterns)
    except re.error:
        raise ConversionError("Invalid Qwen3.8 Q8 suffix expression") from None
    ple = lily.get("ple", {})
    _require(isinstance(ple, dict), "Invalid Qwen3.8 PLE configuration")
    heads = (architecture["ngram_size"] - 1) * architecture["heads_per_ngram"]
    p = {}
    for field, size in (
        ("layer_multipliers", architecture["ngram_size"]),
        ("ngram_heads_vocab_sizes", heads),
        ("ngram_heads_offsets", heads),
    ):
        values = ple.get(field, [])
        _require(
            isinstance(values, list)
            and (len(values) == size if ple_ids else not values)
            and all(type(value) is int and 0 <= value <= 2**64 - 1 for value in values),
            "Invalid Qwen3.8 PLE hash configuration",
        )
        p[field] = values.copy()
    padded = ple.get("padded_vocab_size", 0)
    _require(
        type(padded) is int and 0 <= padded <= 2**40 and (padded > 0 if ple_ids else padded == 0),
        "Invalid Qwen3.8 PLE table size",
    )
    p["padded_vocab_size"] = padded
    architecture["ple"] = p
    return architecture, regexes


def canonical_name(name: str) -> str:
    """Use explicit canonical Uzu tensor identifiers, never source paths."""
    if name.startswith("model.language_model."):
        name = "language_model.model." + name.removeprefix("model.language_model.")
    elif name == "lm_head" or name.startswith("lm_head."):
        name = "language_model." + name
    return name.replace(".mlp.experts.", ".mlp.switch_mlp.")


def _flatten(info: TensorInfo) -> TensorInfo:
    return replace(info, shape=(math.prod(info.shape[:-1]), info.shape[-1]))


def build_qwen38_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Plan text-only standard storage while preserving every consumed source bit."""
    if dtype != "bfloat16":
        raise UnsupportedConversion("Qwen3.8 currently preserves BF16 parameters and requires bfloat16 output")
    config = read_json(source / "config.json")
    architecture, regexes = architecture_config(config)
    tensors = load_checkpoint(source)
    validate_planning_layers(architecture["num_hidden_layers"], tensors)
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    bindings: dict[str, dict] = {}
    metadata: dict[str, str] = {}
    groups = set(architecture["quantization"]["q8_groups"])
    base = "model.language_model"
    h, wide = architecture["hidden_size"], architecture["hidden_size"] * architecture["hc_count"]

    def control(name: str, shape: tuple[int, ...], integers: bool = False):
        _require(name in tensors, "Missing Qwen3.8 control tensor")
        info = tensors[name]
        _require(info.shape == shape, "Qwen3.8 control shape mismatch")
        _require(info.dtype in ({"I64", "U64"} if integers else {"BF16", "F32"}), "Unsupported Qwen3.8 control dtype")
        expected.add(name)
        out = canonical_name(name)
        operations.append(TensorOperation(out, (info,), info.dtype, info.shape))
        bindings[name] = {"name": out, "dtype": info.dtype, "shape": list(info.shape), "layout": "direct"}

    def matrix(
        stem: str,
        shape: tuple[int, ...],
        *,
        group: str | None = None,
        fixed_q8: bool = False,
        embedding: bool = False,
        ngram: bool = False,
    ):
        bits = 4 if ngram else 8 if fixed_q8 or group in groups else 4
        group_size = 32 if ngram else 64
        # Regex policy is evaluated for every original checkpoint matrix. The
        # runtime's fixed small-projection rules must describe the same width.
        declared_bits = 4 if ngram else 8 if group in groups or any(p.search(stem + ".weight") for p in regexes) else 4
        if fixed_q8 and not regexes:
            declared_bits = 8
        _require(bits == declared_bits, "Qwen3.8 suffix policy disagrees with the executable storage policy")
        columns, rows = shape[-1], math.prod(shape[:-1])
        _require(columns % group_size == 0, "Qwen3.8 matrix group mismatch")
        names = tuple(stem + suffix for suffix in (".weight", ".scales", ".biases"))
        _require(all(name in tensors for name in names), "Missing Qwen3.8 matrix plane")
        weight, scales, biases = (tensors[name] for name in names)
        _require(
            weight.dtype == "U32" and weight.shape == (*shape[:-1], columns * bits // 32),
            "Qwen3.8 packed matrix geometry mismatch",
        )
        parameter_shape = (*shape[:-1], columns // group_size)
        _require(
            scales.dtype == biases.dtype == "BF16" and scales.shape == biases.shape == parameter_shape,
            "Qwen3.8 requires matching BF16 affine parameter planes",
        )
        quantization = MLXQuantization(bits, group_size)
        flattened = {name: _flatten(tensors[name]) for name in names}
        mlx = read_mlx_matrix(flattened, names[0], (rows, columns), quantization)
        out = canonical_name(stem)
        layout = "input_output" if embedding else "output_input"
        operations.extend(plan_mlx_matrix(out, (mlx,), dtype="BF16", layout=layout, quantization=quantization))
        metadata[out + ".spec"] = json.dumps(
            {"type": "MLXSpec", "bits": bits, "group_size": group_size, "layout": layout}, sort_keys=True
        )
        bindings[names[0]] = {
            "name": out + ".weights",
            "dtype": "U32",
            "shape": list(weight.shape),
            "layout": "packed_u32",
        }
        for name, info in zip(names[1:], (scales, biases), strict=True):
            bindings[name] = {
                "name": canonical_name(name),
                "dtype": "BF16",
                "shape": list(info.shape),
                "layout": "direct" if embedding else "group_output",
            }
        expected.update(names)

    def hyper(stem: str, combine: bool):
        matrix(stem + ".input_mix_weight_down", (architecture["hc_lowrank"], wide), fixed_q8=True)
        matrix(stem + ".input_mix_weight_up", (wide, architecture["hc_lowrank"]), fixed_q8=True)
        if combine:
            matrix(stem + ".block_inject_weight", (architecture["hc_count"], wide), fixed_q8=True)
        control(stem + ".hc_norm.weight", (wide,))

    matrix(base + ".embed_tokens", (architecture["vocab_size"], h), group="embed", embedding=True)
    matrix("lm_head", (architecture["vocab_size"], h), group="head")
    hyper(base + ".hyper_connection_mixer", False)
    for index, kind in enumerate(architecture["layer_types"]):
        stem = f"{base}.layers.{index}"
        hyper(stem + ".attn_hyper_connection", True)
        hyper(stem + ".mlp_hyper_connection", True)
        if kind == "linear_attention":
            p = stem + ".linear_attn"
            kw = architecture["linear_num_key_heads"] * architecture["linear_key_head_dim"]
            vw = architecture["linear_num_value_heads"] * architecture["linear_value_head_dim"]
            for name, rows in (
                ("in_proj_qkv", 2 * kw + vw),
                ("in_proj_z", vw),
                ("in_proj_a", architecture["linear_num_value_heads"]),
                ("in_proj_b", architecture["linear_num_value_heads"]),
            ):
                matrix(p + "." + name, (rows, h), group="gdn")
            matrix(p + ".out_proj", (h, vw), group="gdn")
            control(p + ".conv1d.weight", (2 * kw + vw, 1, architecture["linear_conv_kernel_dim"]))
            control(p + ".A_log", (architecture["linear_num_value_heads"],))
            control(p + ".dt_bias", (architecture["linear_num_value_heads"],))
            control(p + ".norm.weight", (architecture["linear_value_head_dim"],))
        else:
            p = stem + ".self_attn"
            for name, rows in (
                ("q_proj", 2 * architecture["num_attention_heads"] * architecture["head_dim"]),
                ("k_proj", architecture["num_key_value_heads"] * architecture["head_dim"]),
                ("v_proj", architecture["num_key_value_heads"] * architecture["head_dim"]),
            ):
                matrix(p + "." + name, (rows, h), group="attn")
            matrix(p + ".o_proj", (h, architecture["num_attention_heads"] * architecture["head_dim"]), group="attn")
            matrix(
                p + ".indexer.index_qk_proj",
                ((architecture["indexer_n_heads"] + 1) * architecture["indexer_head_dim"], h),
                fixed_q8=True,
            )
            for name, width in (
                ("q_norm", architecture["head_dim"]),
                ("k_norm", architecture["head_dim"]),
                ("indexer.q_layernorm", architecture["indexer_head_dim"]),
                ("indexer.k_layernorm", architecture["indexer_head_dim"]),
            ):
                control(p + "." + name + ".weight", (width,))
        p = stem + ".mlp"
        matrix(p + ".gate", (architecture["num_experts"], h), fixed_q8=True)
        matrix(p + ".shared_expert_gate", (1, h), fixed_q8=True)
        e, m, s = (
            architecture["num_experts"],
            architecture["moe_intermediate_size"],
            architecture["shared_expert_intermediate_size"],
        )
        for name in ("gate_proj", "up_proj"):
            matrix(p + ".experts." + name, (e, m, h))
            matrix(p + ".shared_expert." + name, (s, h), group="shared")
        matrix(p + ".experts.down_proj", (e, h, m))
        matrix(p + ".shared_expert.down_proj", (h, s), group="shared")
        if index + 1 in architecture["ple_layer_ids"]:
            p = stem + ".ple"
            pe = architecture["ple_embed_dim"]
            heads = (architecture["ngram_size"] - 1) * architecture["heads_per_ngram"]
            _require(pe % heads == 0, "Invalid Qwen3.8 PLE head width")
            shard_rows, remainder = divmod(architecture["ple"]["padded_vocab_size"], architecture["split_ngram_parts"])
            _require(shard_rows > 0 and remainder == 0, "Invalid Qwen3.8 PLE shard geometry")
            for shard in range(architecture["split_ngram_parts"]):
                matrix(
                    f"{p}.ple_embedding.ngram_embedding.shard_{shard}",
                    (shard_rows, pe // heads),
                    embedding=True,
                    ngram=True,
                )
            matrix(p + ".key_proj", (wide, pe), fixed_q8=True)
            matrix(p + ".value_proj", (h, pe), fixed_q8=True)
            for name in ("norm_key", "norm_query", "norm_conv"):
                control(p + "." + name + ".weight", (wide,))
            control(p + ".conv1d.weight", (wide, 1, architecture["ple_conv_kernel_size"]))
            for name, width in (
                ("layer_multipliers", architecture["ngram_size"]),
                ("ngram_heads_vocab_sizes", heads),
                ("ngram_heads_offsets", heads),
            ):
                key = p + ".ple_embedding." + name
                # Lily stores these in typed config; retain exact tensor copies
                # when the source also carries them (older exports do).
                if key in tensors:
                    control(key, (width,), integers=True)
    omitted = {"vision": {"tensors": 0, "bytes": 0}, "mtp": {"tensors": 0, "bytes": 0}}
    for name, info in tensors.items():
        if name in expected:
            continue
        category = "vision" if name.startswith("model.visual.") else "mtp" if name.startswith("mtp.") else None
        _require(category is not None, "Unexpected Qwen3.8 text tensor")
        omitted[category]["tensors"] += 1
        omitted[category]["bytes"] += info.nbytes
    metadata["stark.qwen38.omitted"] = json.dumps(omitted, sort_keys=True)
    metadata["stark.qwen38.schema_version"] = "1"
    _require(
        len({binding["name"] for binding in bindings.values()}) == len(bindings) == len(operations)
        and {binding["name"] for binding in bindings.values()} == {operation.name for operation in operations},
        "Qwen3.8 canonical bindings are not bijective",
    )
    return ConversionPlan(
        {"architecture": architecture, "tensor_bindings": bindings},
        tuple(operations),
        metadata,
        sum(info.nbytes for info in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )


def validate_qwen38_artifact(plan: ConversionPlan, path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> dict:
    """Verify code/control byte identity and invert every BF16 plane bit-exactly.

    This performs payload reads only during the caller-owned conversion run.
    Workspace is bounded independently of the full model or expert-bank size.
    The receipt contains hashes and aggregate counts only.
    """
    _require(type(chunk_bytes) is int and chunk_bytes >= 4096, "Invalid Qwen3.8 validation workspace")
    metadata, output = read_safetensors_header(path)
    operations = {operation.name: operation for operation in plan.operations}
    _require(set(output) == set(operations), "Qwen3.8 output tensor inventory changed")
    _require(metadata == plan.metadata, "Qwen3.8 output spec metadata changed")
    records = []
    codes = parameters = controls = 0
    for logical, binding in sorted(plan.decoder_config["tensor_bindings"].items()):
        name = binding["name"]
        op, converted = operations[name], output[name]
        _require(len(op.sources) == 1, "Qwen3.8 validation requires individual source planes")
        original = op.sources[0]
        _require(converted.dtype == op.dtype and converted.shape == op.shape, "Qwen3.8 output geometry changed")
        if binding["layout"] != "group_output":
            source_hash = tensor_sha256(original, chunk_bytes)
            output_hash = tensor_sha256(converted, chunk_bytes)
            _require(source_hash == output_hash, "Qwen3.8 packed codes or controls changed")
            codes += int(binding["layout"] == "packed_u32")
            controls += int(binding["layout"] == "direct")
            if original.dtype in {"I64", "U64"}:
                field = logical.rsplit(".", 1)[-1]
                expected_values = plan.decoder_config["architecture"]["ple"].get(field)
                _require(isinstance(expected_values, list), "Unexpected Qwen3.8 integer control")
                with _open_tensor(original) as control_stream:
                    for expected_value in expected_values:
                        raw_value = control_stream.read(8)
                        _require(
                            len(raw_value) == 8 and struct.unpack("<Q", raw_value)[0] == expected_value,
                            "Qwen3.8 PLE control disagrees with typed configuration",
                        )
        else:
            _require(original.dtype == converted.dtype == "BF16", "Qwen3.8 parameter precision changed")
            rows, groups = original.shape
            padded_rows = rows + (-rows % 4)
            _require(converted.shape == (groups, padded_rows), "Qwen3.8 parameter layout changed")
            tile_rows = max(1, min(rows, chunk_bytes // (groups * 8)))
            inverse_digest = hashlib.sha256()
            source_digest = hashlib.sha256()
            with _open_tensor(original) as src, _open_tensor(converted) as dst:
                for row in range(0, rows, tile_rows):
                    height = min(tile_rows, rows - row)
                    expected_bytes = src.read(height * groups * 2)
                    _require(len(expected_bytes) == height * groups * 2, "Truncated Qwen3.8 source parameters")
                    inverse = np.empty((height, groups), dtype="<u2")
                    for group in range(groups):
                        dst.seek(converted.offset + (group * padded_rows + row) * 2)
                        raw = dst.read(height * 2)
                        _require(len(raw) == height * 2, "Truncated Qwen3.8 output parameters")
                        inverse[:, group] = np.frombuffer(raw, dtype="<u2")
                    actual = inverse.tobytes()
                    _require(actual == expected_bytes, "Qwen3.8 parameter bits changed")
                    inverse_digest.update(actual)
                    source_digest.update(expected_bytes)
                    del expected_bytes, inverse, actual, raw
                for group in range(groups):
                    dst.seek(converted.offset + (group * padded_rows + rows) * 2)
                    padding = dst.read((padded_rows - rows) * 2)
                    _require(
                        len(padding) == (padded_rows - rows) * 2 and not any(padding),
                        "Qwen3.8 nonzero parameter padding",
                    )
            source_hash, output_hash = source_digest.hexdigest(), inverse_digest.hexdigest()
            parameters += 1
        records.append({"logical": logical, "source_sha256": source_hash, "inverse_sha256": output_hash})
    return {
        "schema_version": 1,
        "quantization_preserved": True,
        "packed_code_planes": codes,
        "inverse_bit_exact_parameter_planes": parameters,
        "direct_byte_exact_planes": controls,
        "plane_hashes_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "omitted": json.loads(plan.metadata["stark.qwen38.omitted"]),
        "native_loading": "unverified",
        "execution": "unverified",
    }
