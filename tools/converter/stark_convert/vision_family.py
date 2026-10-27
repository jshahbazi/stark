"""Structural admission for the first merged BF16 Qwen3.5 vision family.

This profile describes implemented operations and geometry. It grants neither
checkpoint reference parity nor a successful execution result on the local Mac.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

from tokenizers import Tokenizer

from .profiles import generic_files
from .streaming import DTYPE_BYTES, ConversionError, UnsupportedConversion, open_regular

FAMILY_PROFILE = "qwen35-dense-bf16-merged-v1"
FAMILY_VERSION = 1
_SOURCE_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
OUTPUT_TOKENIZER_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
_NORMALIZED_SOURCE_TOKENIZER_SHA256 = OUTPUT_TOKENIZER_SHA256
_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
_MAPPING_TEMPLATE_SHA256 = "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67"
COMPONENT_FILES = ("model.safetensors", "tokenizer.json", "preprocessor_config.json", "chat_template.jinja")
_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
_PROCESSOR_FILES = {"preprocessor_config.json", "processor_config.json"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SOURCE_LABEL = re.compile(r"[A-Za-z0-9._/:\-]{1,256}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TEXT_GEOMETRY = {
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_hidden_layers": 24,
    "vocab_size": 248320,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_value_heads": 16,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}
_VISION_GEOMETRY = {
    "model_type": "qwen3_5",
    "depth": 12,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_heads": 12,
    "num_position_embeddings": 2304,
    "out_hidden_size": 1024,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
}
_PROCESSOR = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
_PROCESSOR_DEFAULTS = {
    "data_format": "channels_first",
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "resample": 3,
    "rescale_factor": 1 / 255,
}
_ROPE = {
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
}
_LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
_TOKEN_FIELDS = {
    "vision_start": "vision_start_token_id",
    "vision_end": "vision_end_token_id",
    "image_pad": "image_token_id",
    "video_pad": "video_token_id",
}
_TOKEN_METADATA_FIELDS = {
    "vision_start": "vision_bos_token",
    "vision_end": "vision_eos_token",
    "image_pad": "image_token",
    "video_pad": "video_token",
}


def family_source_files(repo: str | None, names: tuple[str, ...]) -> tuple[dict, ...]:
    """Bind the generic checkpoint's own processor and standalone template."""
    rules = generic_files(repo, names)
    if "chat_template.jinja" not in names:
        raise ConversionError("Vision family metadata source is incomplete")
    return (*rules, {"role": "processor_config", "repo": repo, "filename": family_processor_filename(names)})


def family_processor_filename(names: Any) -> str:
    """Choose exactly one processor serialization from this checkpoint."""
    selected = _PROCESSOR_FILES.intersection(names)
    if len(selected) != 1:
        raise ConversionError("Vision family processor metadata is incomplete or ambiguous")
    return next(iter(selected))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _json(payload: bytes) -> dict:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ConversionError("Vision family metadata is invalid") from None


def _read_metadata(snapshot: Any, name: str) -> bytes:
    try:
        path = snapshot.resolve_file(name)
    except OSError:
        raise ConversionError("Vision family metadata source is incomplete") from None
    return _read_metadata_path(path, name)


def _read_metadata_path(path, name: str) -> bytes:
    maximum = {
        "tokenizer.json": 64 * 1024 * 1024,
        "preprocessor_config.json": 64 * 1024,
        "processor_config.json": 64 * 1024,
        "chat_template.jinja": 256 * 1024,
    }.get(name, 1024 * 1024)
    try:
        with open_regular(path) as stream:
            before = os.fstat(stream.fileno())
            if not 0 < before.st_size <= maximum:
                raise ValueError()
            data = stream.read(before.st_size + 1)
            after = os.fstat(stream.fileno())
            if len(data) != before.st_size or any(
                getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            ):
                raise ValueError()
            return data
    except (OSError, ValueError):
        raise ConversionError("Vision family metadata is unavailable or exceeds its size limit") from None


def read_family_config(path) -> dict:
    """Bound configuration work before generic architecture detection."""
    return _json(_read_metadata_path(path, "config.json"))


def family_config_bytes(config: dict) -> bytes:
    """Keep the complete generated configuration inside native admission's cap."""
    data = json.dumps(config, indent=4).encode("utf-8")
    if len(data) > 1024 * 1024:
        raise UnsupportedConversion("Vision family configuration exceeds the native size limit")
    return data


def validate_family_output_header(plan: Any) -> None:
    """Bound the generated header before streaming any output tensor payloads."""
    header = {"__metadata__": plan.metadata} if plan.metadata else {}
    offset = 0
    for operation in sorted(plan.operations, key=lambda operation: (-DTYPE_BYTES[operation.dtype], operation.name)):
        header[operation.name] = {
            "dtype": operation.dtype,
            "shape": list(operation.shape),
            "data_offsets": [offset, offset + operation.nbytes],
        }
        offset += operation.nbytes
    size = len(json.dumps(header).encode("utf-8"))
    if size + (-size % 8) > 1024 * 1024:
        raise UnsupportedConversion("Vision family tensor header exceeds the native size limit")


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_equal(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if type(expected) is float:
        return type(actual) in (float, int) and math.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def normalize_family_processor(processor: dict, filename: str) -> dict:
    """Recognize still-image serialization defaults without running HF code."""
    if filename == "processor_config.json":
        if (
            set(processor) - {"image_processor", "processor_class", "video_processor"}
            or processor.get("processor_class") != _PROCESSOR["processor_class"]
            or not isinstance(processor.get("image_processor"), dict)
            or ("video_processor" in processor and not isinstance(processor["video_processor"], dict))
            or processor["image_processor"].get("processor_class", processor["processor_class"])
            != processor["processor_class"]
        ):
            raise UnsupportedConversion("Vision family processor recipe is outside the implemented profile")
        # The image-only profile neither interprets nor emits video processing.
        processor = {**processor["image_processor"], "processor_class": processor["processor_class"]}
    permitted = set(_PROCESSOR) | set(_PROCESSOR_DEFAULTS)
    if (
        set(processor) - permitted
        or any(not _equal(processor.get(key), value) for key, value in _PROCESSOR.items())
        or any(not _equal(processor.get(key, value), value) for key, value in _PROCESSOR_DEFAULTS.items())
    ):
        raise UnsupportedConversion("Vision family processor recipe is outside the implemented profile")
    return json.loads(json.dumps(_PROCESSOR))


def _source_label(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _SOURCE_LABEL.fullmatch(value) is not None
        and not value.startswith("/")
        and not any(part in {"", ".", ".."} for part in value.split("/"))
    )


def validate_family_source(snapshot: Any, dtype: str) -> dict:
    """Admit a complete supported family source without an official-weight claim."""
    from .adapters.qwen35 import normalize_qwen35_config

    if dtype != "bfloat16":
        raise UnsupportedConversion("Vision family conversion requires BF16")
    if snapshot.repo is not None and (
        not isinstance(snapshot.repo, str) or len(snapshot.repo) > 256 or _REPO.fullmatch(snapshot.repo) is None
    ):
        raise ConversionError("Vision family source identity is invalid")
    if snapshot.revision is not None and (
        not isinstance(snapshot.revision, str) or _REVISION.fullmatch(snapshot.revision) is None
    ):
        raise ConversionError("Vision family source revision must be immutable")
    processor_filename = family_processor_filename(record.get("filename") for record in snapshot.manifest["files"])
    payloads = {name: _read_metadata(snapshot, name) for name in (*_METADATA_FILES, processor_filename)}
    if any(
        record.get("filename") == "generation_config.json" and record.get("role") == "generation_config"
        for record in snapshot.manifest.get("files", [])
    ):
        payloads["generation_config.json"] = _read_metadata(snapshot, "generation_config.json")
        _json(payloads["generation_config.json"])
    if hashlib.sha256(payloads["tokenizer.json"]).hexdigest() not in {
        _SOURCE_TOKENIZER_SHA256,
        _NORMALIZED_SOURCE_TOKENIZER_SHA256,
    } or hashlib.sha256(payloads["chat_template.jinja"]).hexdigest() not in {
        _TEMPLATE_SHA256,
        _MAPPING_TEMPLATE_SHA256,
    }:
        raise UnsupportedConversion("Vision family tokenizer or template is outside the implemented profile")
    config, processor, token_config = (
        _json(payloads[name]) for name in ("config.json", processor_filename, "tokenizer_config.json")
    )
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or not isinstance(config.get("text_config"), dict)
        or not isinstance(config.get("vision_config"), dict)
        or any(config.get(key) is not None for key in ("quantization", "quantization_config", "auto_map"))
        or token_config.get("auto_map") is not None
    ):
        raise UnsupportedConversion("Vision family requires a merged dense Qwen3.5 checkpoint")
    text = normalize_qwen35_config(config)
    if any(not _equal(text.get(key), expected) for key, expected in _TEXT_GEOMETRY.items()):
        raise UnsupportedConversion("Vision family decoder geometry is outside the implemented profile")
    if (
        text.get("layer_types") != _LAYER_TYPES
        or not _equal(text.get("rope_parameters"), _ROPE)
        or text.get("dtype", "bfloat16") != "bfloat16"
        or text.get("torch_dtype", "bfloat16") != "bfloat16"
        or any(text.get(key) is not None for key in ("quantization", "quantization_config", "auto_map"))
    ):
        raise UnsupportedConversion("Vision family decoder operations are outside the implemented profile")
    expected_operations = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "hidden_act": "silu",
        "mlp_bias": False,
        "use_bias": False,
        "use_qk_norm": True,
        "mamba_ssm_dtype": "float32",
        "mlp_only_layers": [],
        "full_attention_interval": 4,
        "mtp_use_dedicated_embeddings": False,
    }
    if any(not _equal(text.get(key, value), value) for key, value in expected_operations.items()):
        raise UnsupportedConversion("Vision family decoder operations are outside the implemented profile")
    vision = config["vision_config"]
    if set(vision) - (
        set(_VISION_GEOMETRY) | {"hidden_act", "deepstack_visual_indexes", "initializer_range", "dtype", "torch_dtype"}
    ) or any(not _equal(vision.get(key), expected) for key, expected in _VISION_GEOMETRY.items()):
        raise UnsupportedConversion("Vision family tower geometry is outside the implemented profile")
    initializer = vision.get("initializer_range", 0.02)
    if (
        vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh"
        or vision.get("deepstack_visual_indexes", []) not in ([], None)
        or type(initializer) not in (float, int)
        or not math.isfinite(initializer)
        or initializer < 0
        or any(vision.get(key, "bfloat16") != "bfloat16" for key in ("dtype", "torch_dtype"))
    ):
        raise UnsupportedConversion("Vision family tower operations are outside the implemented profile")
    vision = {
        **{key: value for key, value in vision.items() if key not in {"dtype", "torch_dtype"}},
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
        "initializer_range": initializer,
    }
    processor = normalize_family_processor(processor, processor_filename)
    try:
        template = payloads["chat_template.jinja"].decode("utf-8")
        if not template:
            raise ValueError()
        if token_config.get("chat_template") not in (None, template):
            raise ValueError()
        tokenizer = Tokenizer.from_str(payloads["tokenizer.json"].decode("utf-8"))
        vocab_size = _TEXT_GEOMETRY["vocab_size"]
        tokenizer_size = tokenizer.get_vocab_size(with_added_tokens=True)
        # Supported embedding/output matrices include unused padded rows.
        if not 0 < tokenizer_size <= vocab_size:
            raise ValueError()
        vocab = tokenizer.get_vocab()
        if (
            any(type(value) is not int or not 0 <= value < vocab_size for value in vocab.values())
            or len(set(vocab.values())) != tokenizer_size
        ):
            raise ValueError()
        added = token_config.get("added_tokens_decoder")
        if added is None:
            # New TokenizersBackend saves preserve the complete token records
            # in tokenizer.json. Admit this form only for its reviewed normalized
            # bytes, then validate the embedded IDs and flags identically.
            if hashlib.sha256(payloads["tokenizer.json"]).hexdigest() != _NORMALIZED_SOURCE_TOKENIZER_SHA256:
                raise ValueError()
            embedded = _json(payloads["tokenizer.json"])["added_tokens"]
            if not isinstance(embedded, list) or any(
                not isinstance(item, dict) or type(item.get("id")) is not int for item in embedded
            ):
                raise ValueError()
            added = {str(item["id"]): item for item in embedded}
            if len(added) != len(embedded):
                raise ValueError()
        if not isinstance(added, dict):
            raise ValueError()
        symbolic = token_config.get("model_specific_special_tokens", {})
        if not isinstance(symbolic, dict) or any(
            token_config.get(key, False) is not False for key in ("split_special_tokens", "add_prefix_space")
        ):
            raise ValueError()
        seen = set()
        for key, item in added.items():
            value = int(key)
            if (
                str(value) != key
                or not 0 <= value < vocab_size
                or not isinstance(item, dict)
                or not isinstance(item.get("content"), str)
                or item["content"] in seen
                or type(item.get("id", value)) is not int
                or item.get("id", value) != value
            ):
                raise ValueError()
            seen.add(item["content"])
        tokens = {}
        for name, field in _TOKEN_FIELDS.items():
            token = f"<|{name}|>"
            metadata_field = _TOKEN_METADATA_FIELDS[name]
            token_id = config[field]
            item = added.get(str(token_id))
            if (
                type(token_id) is not int
                or token_config.get(metadata_field, token) != token
                or symbolic.get(metadata_field, token) != token
                or not 0 <= token_id < vocab_size
                or tokenizer.token_to_id(token) != token_id
                or not isinstance(item, dict)
                or item.get("content") != token
                or item.get("id", token_id) != token_id
                or item.get("special") is not True
                or any(item.get(key, False) is not False for key in ("normalized", "single_word", "lstrip", "rstrip"))
                or tokenizer.encode(token, add_special_tokens=False).ids != [token_id]
            ):
                raise ValueError()
            tokens[name] = token_id
        if len(set(tokens.values())) != 4:
            raise ValueError()
    except Exception:
        raise ConversionError("Vision family tokenizer or template metadata is invalid") from None
    records = snapshot.manifest.get("files", [])
    if not records or any(
        record.get("repo") != snapshot.repo
        or record.get("revision") != snapshot.revision
        or not isinstance(record.get("sha256"), str)
        or _HASH.fullmatch(record["sha256"]) is None
        for record in records
    ):
        raise ConversionError("Vision family source manifest is invalid")
    weight_records = [record for record in records if record.get("role") in {"weights", "weight_index"}]
    if (
        not 0 < len(weight_records) <= 256
        or any(
            not _source_label(record.get("filename"))
            or type(record.get("size")) is not int
            or not 0 < record["size"] <= 2**64 - 1
            for record in weight_records
        )
        or len({record["filename"] for record in weight_records}) != len(weight_records)
    ):
        raise ConversionError("Vision family weight provenance is outside the native limits")
    weights = {
        record["filename"]: {key: record[key] for key in ("role", "size", "sha256")} for record in weight_records
    }
    if not any(record["role"] == "weights" for record in weights.values()):
        raise ConversionError("Vision family requires complete merged weights")
    return {
        "type": "Qwen35VisionLanguageArtifact",
        "schema_version": 3,
        "family": {"profile": FAMILY_PROFILE, "version": FAMILY_VERSION},
        "input_modalities": ["text", "image"],
        "dtype": dtype,
        "source": {
            "repo": snapshot.repo or "local",
            "revision": snapshot.revision,
            "hashes": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
            "weight_files": weights,
        },
        "special_token_ids": tokens,
        "vision_config": vision,
        "processor_config": processor,
        "position_contract": {"mrope_section": list(_ROPE["mrope_section"]), "interleaved": True},
        "tensor_layout": "hf-qwen35-vision-v1",
        "transport": "mediaref-v1",
        "qualification": "unverified",
        "runtime_supported": True,
    }
