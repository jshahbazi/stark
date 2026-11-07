"""Validated processor and complete source provenance for Gemma 3 image artifacts."""

from __future__ import annotations

import hashlib
import json
import re

from .adapters.gemma3_vision import FAMILY, TOKENS, gemma3_config
from .streaming import ConversionError, UnsupportedConversion, read_regular_bytes

SCHEMA_VERSION = 5
MODEL_TYPE = "Gemma3VisionLanguageModelConfig"
COMPONENT_FILES = ("model.safetensors", "tokenizer.json", "preprocessor_config.json", "chat_template.jinja")
SOURCE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
PROCESSOR = {
    "do_convert_rgb": True, "do_normalize": True, "do_pan_and_scan": False,
    "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "image_processor_type": "Gemma3ImageProcessor", "image_seq_length": 256,
    "resample": 2, "rescale_factor": 1 / 255, "size": {"height": 896, "width": 896},
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"


def gemma3_source_files(repo: str | None, names: tuple[str, ...]) -> tuple[dict, ...]:
    from .profiles import generic_files

    if "preprocessor_config.json" not in names:
        raise ConversionError("Gemma 3 image processor metadata is missing")
    return (*generic_files(repo, names),
        {"role": "processor_config", "repo": repo, "filename": "preprocessor_config.json"})


def normalize_processor(value: dict) -> dict:
    """Resolve disabled optional PaS and RGB defaults without changing pixels."""
    if not isinstance(value, dict):
        raise ConversionError("Gemma 3 image processor must be an object")
    annotations = {"processor_class", "pan_and_scan_min_crop_size", "pan_and_scan_max_num_crops",
                   "pan_and_scan_min_ratio_to_activate"}
    if set(value) - (set(PROCESSOR) | annotations):
        raise UnsupportedConversion("Gemma 3 processor declares unimplemented options")
    result = {**PROCESSOR, **{key: item for key, item in value.items() if key in PROCESSOR}}
    for key, default in (("do_convert_rgb", True), ("do_pan_and_scan", False)):
        if result[key] is None:
            result[key] = default
    if result != PROCESSOR or any(type(result[key]) is not type(expected) for key, expected in PROCESSOR.items()):
        raise UnsupportedConversion("Gemma 3 image processor differs from the fixed-resolution profile")
    if value.get("processor_class", "Gemma3Processor") != "Gemma3Processor":
        raise UnsupportedConversion("Gemma 3 processor class is unsupported")
    return dict(PROCESSOR)


def validate_gemma3_source(snapshot, dtype: str) -> dict:
    """Consume metadata through selected, fully hashed SourceSnapshot files."""
    if dtype != "bfloat16":
        raise UnsupportedConversion("Gemma 3 images require BF16 source and execution")
    payloads = {name: read_regular_bytes(snapshot.resolve_file(name)) for name in SOURCE_FILES}
    config = json.loads(payloads["config.json"])
    text, vision = gemma3_config(config)
    processor = normalize_processor(json.loads(payloads["preprocessor_config.json"]))
    tokenizer_config = json.loads(payloads["tokenizer_config.json"])
    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ConversionError("Gemma 3 requires its source-bound chat template")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != TEMPLATE_SHA256:
        raise UnsupportedConversion("Gemma 3 image template is outside the reviewed program profile")
    # A source may supply the template as a sidecar. It must agree byte-for-byte
    # with the authoritative tokenizer metadata if both are present.
    records = snapshot.manifest.get("files", [])
    record_names = {record.get("filename") for record in records}
    if "chat_template.jinja" in record_names:
        payloads["chat_template.jinja"] = read_regular_bytes(snapshot.resolve_file("chat_template.jinja"))
        if payloads["chat_template.jinja"].decode("utf-8") != template:
            raise ConversionError("Gemma 3 source chat templates disagree")
    else:
        payloads["chat_template.jinja"] = template.encode("utf-8")
    if (not records or any(record.get("repo") != snapshot.repo or record.get("revision") != snapshot.revision
            or not isinstance(record.get("sha256"), str) or _HASH.fullmatch(record["sha256"]) is None
            for record in records)):
        raise ConversionError("Gemma 3 source manifest is invalid")
    weights = {record["filename"]: {key: record[key] for key in ("role", "size", "sha256")}
               for record in records if record.get("role") in {"weights", "weight_index"}}
    if (not 0 < len(weights) <= 256 or not any(record["role"] == "weights" for record in weights.values())
        or any(type(record["size"]) is not int or record["size"] <= 0 for record in weights.values())):
        raise ConversionError("Gemma 3 source weight provenance is incomplete")
    return {
        "type": "Gemma3VisionLanguageArtifact", "schema_version": SCHEMA_VERSION, "family": dict(FAMILY),
        "input_modalities": ["text", "image"], "dtype": dtype,
        "source": {"repo": snapshot.repo or "local", "revision": snapshot.revision,
            "hashes": {**{record["filename"]: record["sha256"] for record in records
                           if record["role"] not in {"weights", "weight_index"}},
                       **{name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}},
            "weight_files": weights},
        "special_token_ids": dict(TOKENS), "vision_config": vision, "processor_config": processor,
        "decoder_model_dim": text["hidden_size"], "tensor_layout": "hf-gemma3-siglip-v1",
        "position_contract": {"type": "scalar", "bidirectional_image_attention": True},
        "transport": "mediaref-v1", "qualification": "unverified", "runtime_supported": True,
    }
