"""Import classifier tokenizer metadata without model framework imports.

Adapted from Lalamo's ClassifierModelSpec and shared chat-codec import at revision
831accb8252a6b6606c378048caeb44178f1a050.
Copyright (c) 2025 Mirai Tech Inc. MIT licensed; see ../LICENSE.lalamo.

Classifier codecs use the source template and tokenizer, including their special
tokens and post processor. They have no language generation policy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from stark_convert.metadata import (
    UnsupportedMetadata,
    _add_tokens,
    _chat_template,
    _file_path,
    _optional_text,
    _profile_codec,
    _read_object,
    _read_text,
    _token_ids,
    _token_ids_to_text,
)
from stark_convert.profiles import GENERIC_PROFILE
from stark_convert.streaming import ConversionError

__all__ = ["SUPPORTED_CLASSIFIER_METADATA_REPOS", "write_classifier_metadata"]

SUPPORTED_CLASSIFIER_METADATA_REPOS = frozenset({"trymirai/chat-moderation-router", "trymirai/health-related-router"})
_SPECIAL_TOKEN_NAMES = ("bos", "eos", "unk", "pad", "sep", "cls", "mask")


def _label_count(config: dict[str, Any]) -> int:
    """Reject label maps that cannot preserve the classifier readout order."""
    labels = config.get("id2label")
    reverse = config.get("label2id")
    if not isinstance(labels, dict) or not labels or not isinstance(reverse, dict):
        raise ConversionError("Invalid classifier label metadata.")
    normalized: dict[int, str] = {}
    string_indices = {str(index): index for index in range(len(labels))}
    for key, label in labels.items():
        if type(key) is int:
            index = key
        elif isinstance(key, str) and key in string_indices:
            index = string_indices[key]
        else:
            raise ConversionError("Invalid classifier label metadata.")
        if index < 0 or index in normalized or not isinstance(label, str) or not label:
            raise ConversionError("Invalid classifier label metadata.")
        normalized[index] = label
    count = len(normalized)
    if set(normalized) != set(range(count)) or len(set(normalized.values())) != count:
        raise ConversionError("Invalid classifier label metadata.")
    if any(not isinstance(label, str) or type(index) is not int for label, index in reverse.items()):
        raise ConversionError("Invalid classifier label metadata.")
    if reverse != {label: index for index, label in normalized.items()}:
        raise ConversionError("Invalid classifier label metadata.")
    explicit = config.get("num_labels")
    if explicit is not None and (type(explicit) is not int or explicit != count):
        raise ConversionError("Invalid classifier label metadata.")
    return count


def _validate_sources(entry: dict[str, Any]) -> None:
    files = entry.get("files")
    if not isinstance(files, dict) or any(
        role not in files for role in ("model_config", "tokenizer", "tokenizer_config")
    ):
        raise ConversionError("Invalid classifier metadata sources.")
    anonymous_generic = entry.get("metadata_profile") == GENERIC_PROFILE and entry.get("repo") is None
    for role, spec in files.items():
        if not isinstance(spec, dict) or (
            not (anonymous_generic and spec.get("repo") is None)
            and (not isinstance(spec.get("repo"), str) or not spec["repo"])
        ):
            raise ConversionError("Invalid classifier metadata sources.")
        filename = spec.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or "\\" in filename
        ):
            raise ConversionError("Invalid classifier metadata sources.")
        if "field" in spec and (role != "chat_template" or not isinstance(spec["field"], str) or not spec["field"]):
            raise ConversionError("Invalid classifier metadata sources.")
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        raise ConversionError("Invalid classifier metadata profile.")
    if any(metadata.get(name) is not None for name in ("parser_id", "reasoning_id", "end_of_thinking_id")):
        raise UnsupportedMetadata("Unsupported classifier codec profile.")
    if any(metadata.get(name) is not None for name in ("generation_config", "generation_params_overrides")):
        raise UnsupportedMetadata("Unsupported classifier generation profile.")
    if metadata.get("inline_sources", {}):
        raise UnsupportedMetadata("Unsupported classifier inline profile.")


def _validate_tokenizer(tokenizer: Tokenizer, tokenizer_config: dict[str, Any], config: dict[str, Any]) -> None:
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size < 1:
        raise ConversionError("Invalid classifier vocabulary metadata.")
    vocab = tokenizer.get_vocab()
    if not vocab or any(type(index) is not int or not 0 <= index < vocab_size for index in vocab.values()):
        raise ConversionError("Classifier tokenizer exceeds configured vocabulary.")
    if len(set(vocab.values())) != len(vocab):
        raise ConversionError("Classifier tokenizer has ambiguous identifiers.")
    serialized = json.loads(tokenizer.to_str())
    _validate_post_processor(serialized.get("post_processor"), tokenizer)
    padding = serialized.get("padding")
    if padding is not None and (
        not isinstance(padding, dict)
        or type(padding.get("pad_id")) is not int
        or not isinstance(padding.get("pad_token"), str)
        or tokenizer.token_to_id(padding["pad_token"]) != padding["pad_id"]
    ):
        raise ConversionError("Classifier padding tokens disagree with tokenizer.")
    added = tokenizer_config.get("added_tokens_decoder")
    if added is not None:
        for key, token in added.items():
            if (
                not isinstance(key, str)
                or len(key) > len(str(vocab_size))
                or not key.isascii()
                or not key.isdecimal()
                or str(int(key)) != key
            ):
                raise ConversionError("Invalid classifier added-token identifiers.")
            if tokenizer.token_to_id(token["content"]) != int(key):
                raise ConversionError("Classifier added-token identifiers disagree with tokenizer.")
    for name in _SPECIAL_TOKEN_NAMES:
        text = _optional_text(tokenizer_config.get(name + "_token"))
        identifiers = _token_ids(config.get(name + "_token_id"), optional=True)
        if any(index >= vocab_size or tokenizer.id_to_token(index) is None for index in identifiers):
            raise ConversionError("Invalid classifier special-token identifiers.")
        if text is not None:
            index = tokenizer.token_to_id(text)
            if index is None or (identifiers and index != identifiers[0]):
                raise ConversionError("Classifier special tokens disagree with tokenizer.")
    for name in (
        "add_bos_token",
        "add_eos_token",
        "add_prefix_space",
        "use_default_system_prompt",
        "spaces_between_special_tokens",
        "do_lower_case",
    ):
        value = tokenizer_config.get(name)
        if value is not None and type(value) is not bool:
            raise ConversionError("Invalid classifier tokenizer behavior metadata.")


def _validate_post_processor(processor: object, tokenizer: Tokenizer) -> None:
    """Check literal IDs emitted by the preserved tokenizer post processor."""
    if processor is None:
        return
    if not isinstance(processor, dict):
        raise ConversionError("Invalid classifier tokenizer post processor.")
    kind = processor.get("type")
    pairs = []
    if kind == "TemplateProcessing":
        specials = processor.get("special_tokens")
        if not isinstance(specials, dict):
            raise ConversionError("Invalid classifier tokenizer post processor.")
        for value in specials.values():
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("ids"), list)
                or not isinstance(value.get("tokens"), list)
            ):
                raise ConversionError("Invalid classifier tokenizer post processor.")
            if len(value["ids"]) != len(value["tokens"]):
                raise ConversionError("Invalid classifier tokenizer post processor.")
            pairs.extend(zip(value["tokens"], value["ids"], strict=True))
    elif kind in {"BertProcessing", "RobertaProcessing"}:
        for name in ("cls", "sep"):
            pair = processor.get(name)
            if not isinstance(pair, list) or len(pair) != 2:
                raise ConversionError("Invalid classifier tokenizer post processor.")
            pairs.append(pair)
    elif kind == "Sequence":
        children = processor.get("processors")
        if not isinstance(children, list):
            raise ConversionError("Invalid classifier tokenizer post processor.")
        for child in children:
            _validate_post_processor(child, tokenizer)
    elif kind != "ByteLevel":
        raise UnsupportedMetadata("Unsupported classifier tokenizer post processor.")
    for text, index in pairs:
        if not isinstance(text, str) or type(index) is not int or tokenizer.token_to_id(text) != index:
            raise ConversionError("Classifier post-processor tokens disagree with tokenizer.")


def write_classifier_metadata(
    source: Path,
    output: Path,
    hf_config: dict[str, Any],
    *,
    model_entry: dict[str, Any],
    resolve_file: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    """Return ClassifierModelConfig codec fragments and normalize its tokenizer.

    The caller pins every selected file before calling this function. An
    external selected source requires the snapshot's immutable resolver. All
    metadata is validated before creating output; errors never include content.
    """
    profile = model_entry.get("metadata_profile")
    repo = hf_config.get("_stark_model_repo")
    if (
        model_entry.get("repo") != repo
        or model_entry.get("model_kind") != "ClassifierModelSpec"
        or model_entry.get("config_class") != "ModernBERTConfig"
        or profile not in {"modernbert-classifier", GENERIC_PROFILE}
        or (profile != GENERIC_PROFILE and repo not in SUPPORTED_CLASSIFIER_METADATA_REPOS)
    ):
        raise UnsupportedMetadata("Unverified classifier metadata profile.")
    if hf_config.get("model_type") != "modernbert":
        raise ConversionError("Model architecture does not match classifier metadata profile.")
    _label_count(hf_config)
    _validate_sources(model_entry)
    source = Path(source)
    output = Path(output)
    if profile == GENERIC_PROFILE:
        if any(spec["repo"] != repo for spec in model_entry["files"].values()):
            raise ConversionError("Generic classifier metadata must be self-contained.")
    tokenizer_config = _read_object(_file_path(source, model_entry, "tokenizer_config", resolve_file), "tokenizer")
    if profile == GENERIC_PROFILE:
        if tokenizer_config.get("chat_template") is not None and "chat_template" in model_entry["files"]:
            supplied = _read_text(_file_path(source, model_entry, "chat_template", resolve_file), "chat-template")
            if tokenizer_config["chat_template"] != supplied:
                raise ConversionError("Conflicting generic classifier chat-template metadata.")
            model_entry = {
                **model_entry,
                "files": {key: value for key, value in model_entry["files"].items() if key != "chat_template"},
            }
    prompt_template = _chat_template(source, model_entry, tokenizer_config, resolve_file)
    codec_profile = _profile_codec(model_entry["metadata"])
    system_prompt = None
    if "system_prompt" in model_entry["files"]:
        system_prompt = _read_text(_file_path(source, model_entry, "system_prompt", resolve_file), "system-prompt")
    bos_token = _optional_text(tokenizer_config.get("bos_token"))
    eos_token = _optional_text(tokenizer_config.get("eos_token"))
    try:
        tokenizer = Tokenizer.from_str(
            _read_text(_file_path(source, model_entry, "tokenizer", resolve_file), "tokenizer")
        )
        _add_tokens(tokenizer, tokenizer_config)
        _validate_tokenizer(tokenizer, tokenizer_config, hf_config)
        if bos_token is None:
            bos_token = _token_ids_to_text(tokenizer, hf_config.get("bos_token_id"))
        if eos_token is None:
            eos_token = _token_ids_to_text(tokenizer, hf_config.get("eos_token_id"))
    except (UnsupportedMetadata, ConversionError):
        raise
    except Exception:
        raise ConversionError("Classifier tokenizer normalization failed.") from None
    codec = {
        "type": "ChatCodecConfig",
        "prompt_template": prompt_template,
        **codec_profile,
        "eos_token": eos_token,
        "bos_token": bos_token,
        "default_system_prompt": system_prompt,
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output / "tokenizer.json"))
    except Exception:
        raise ConversionError("Cannot write classifier tokenizer metadata.") from None
    return {"token_codec_config": codec}
