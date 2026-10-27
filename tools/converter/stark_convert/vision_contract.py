"""Opt-in Qwen3.5 vision contract (HF revision 2fc06364715b967f1860aea9cf38778875588b17)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenizers import Tokenizer

from .registry import required_files
from .streaming import ConversionError, read_regular_bytes

REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
_HASHES = {
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
}
_TOKENS = {"vision_start": 248053, "vision_end": 248054, "image_pad": 248056, "video_pad": 248057}
# The immutable HF tree reports the LFS payload identity, not its Git pointer.
# A local same-shape fine-tune must not inherit the official checkpoint tuple.
_WEIGHT_FILES = {
    "model.safetensors-00001-of-00001.safetensors": (
        "weights",
        1746942600,
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    ),
    "model.safetensors.index.json": (
        "weight_index",
        50900,
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    ),
}


def source_rules(entry: dict, artifact_mode: str = "text") -> tuple[dict, ...]:
    """Return catalog source descriptors, with an explicit vision opt-in."""
    if artifact_mode == "text":
        return required_files(entry)
    if artifact_mode != "vision-language":
        raise ConversionError("Unsupported artifact mode")
    if entry.get("repo") != REPO:
        raise ConversionError("Vision source is outside the pinned repository")
    return (
        {"role": "model_config", "repo": REPO, "filename": "config.json"},
        {"role": "tokenizer", "repo": REPO, "filename": "tokenizer.json"},
        {"role": "tokenizer_config", "repo": REPO, "filename": "tokenizer_config.json"},
        {"role": "processor_config", "repo": REPO, "filename": "preprocessor_config.json"},
        {"role": "chat_template", "repo": REPO, "filename": "chat_template.jinja"},
    )


def _fail() -> None:
    raise ConversionError("Vision source does not match the pinned artifact")


def _read(snapshot: Any, filename: str) -> bytes:
    try:
        return read_regular_bytes(snapshot.resolve_file(filename, repo=REPO))
    except Exception:
        _fail()
    raise AssertionError


def validate_vision_source(snapshot: Any, dtype: str) -> dict:
    """Validate the pinned metadata and return a private, versioned contract."""
    if dtype != "bfloat16" or snapshot.repo != REPO or snapshot.revision != REVISION:
        _fail()
    records = snapshot.manifest.get("files", [])
    if not records or any(r.get("repo") != REPO or r.get("revision") != REVISION for r in records):
        _fail()
    weight_files = {
        r.get("filename"): (r.get("role"), r.get("size"), r.get("sha256"))
        for r in records
        if r.get("role") in {"weights", "weight_index"}
    }
    if weight_files != _WEIGHT_FILES:
        _fail()
    payloads = {name: _read(snapshot, name) for name in _HASHES}
    if any(
        hashlib.sha256(data).hexdigest() != digest
        for (name, digest), data in zip(_HASHES.items(), payloads.values(), strict=True)
    ):
        _fail()
    try:
        config = json.loads(payloads["config.json"])
        processor = json.loads(payloads["preprocessor_config.json"])
        try:
            tokenizer = Tokenizer.from_str(_read(snapshot, "tokenizer.json").decode("utf-8"))
        except Exception:
            _fail()
        config_tokens = json.loads(_read(snapshot, "tokenizer_config.json")).get("added_tokens_decoder")
        if not isinstance(config_tokens, dict):
            _fail()
        seen: dict[str, int] = {}
        for key, item in config_tokens.items():
            if (
                not isinstance(item, dict)
                or type(item.get("content")) is not str
                or type(item.get("id", int(key))) is not int
            ):
                _fail()
            token_id = int(key)
            if token_id < 0 or str(token_id) != key or item.get("id", token_id) != token_id:
                _fail()
            if item["content"] in seen and seen[item["content"]] != token_id:
                _fail()
            seen[item["content"]] = token_id
        if any(
            tokenizer.token_to_id(f"<|{name}|>") != value or seen.get(f"<|{name}|>") != value
            for name, value in _TOKENS.items()
        ):
            _fail()
        vision = config["vision_config"]
        text = config["text_config"]
        rope = text["rope_parameters"]
        if config.get("model_type") != "qwen3_5" or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            _fail()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        _fail()
    return {
        "type": "Qwen35VisionLanguageArtifact",
        "schema_version": 2,
        "input_modalities": ["text", "image"],
        "dtype": dtype,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "hashes": dict(_HASHES),
            "weight_files": {
                name: {"role": role, "size": size, "sha256": digest}
                for name, (role, size, digest) in _WEIGHT_FILES.items()
            },
        },
        "special_token_ids": dict(_TOKENS),
        "vision_config": vision,
        "processor_config": processor,
        "position_contract": {
            "mrope_section": rope.get("mrope_section"),
            "interleaved": rope.get("mrope_interleaved"),
        },
        "tensor_layout": "hf-qwen35-vision-v1",
        "transport": "mediaref-v1",
        "qualification": "unverified",
        "runtime_supported": True,
    }
