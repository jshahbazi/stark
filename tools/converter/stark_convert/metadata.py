"""Import registered Hugging Face metadata without model framework imports.

Adapted from Lalamo's metadata import, tokenizer configuration, generation
configuration, and Qwen model specifications at revision
831accb8252a6b6606c378048caeb44178f1a050.
Copyright (c) 2025 Mirai Tech Inc. MIT licensed; see ../LICENSE.lalamo.

The registry identity is supplied by the caller as ``_stark_model_repo`` in the
in-memory Hugging Face configuration. It is never written into model metadata.
Architecture alone does not determine a model's reasoning defaults or output
parser. Source files are supplied by the caller's immutable source resolver.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tokenizers import AddedToken, Tokenizer

from stark_convert.registry import CONFIG_ADAPTERS, get_model, models
from stark_convert.streaming import ConversionError, read_json

__all__ = ["SUPPORTED_METADATA_REPOS", "UnsupportedMetadata", "write_metadata"]

_OPTIONAL_THINKING_OUTPUT_PARSER_REGEX = (
    r"(?s)(?=(<think>|.*?</think>)?)"
    r"(?(1)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*|\Z))"
    r"(?P<response>.*)\Z"
)
_GENERATION_BLOCK_TAG = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
_MODEL_TYPES = CONFIG_ADAPTERS
_GRANITE_THINKING_OUTPUT_PARSER_REGEX = (
    r"(?s)<think>(?P<chain_of_thought>.*?)"
    r"(?:</think>\s*(?:<response>\s*)?(?P<response>.*?)(?:</response>\s*)?)?\Z"
)
# These are public, static upstream format defaults, never evaluation content.
# The inventory stores only hashes; the runtime resolves an explicit reviewed
# role/hash pair and rechecks it instead of accepting arbitrary inline content.
_INLINE_TEXT = {
    (
        "system_prompt",
        "ad1f2d358132633f575755436201edaeb47de52256c93b6876d7233276e6c224",
    ): "You are a model that can do function calling with the following functions",
    (
        "chat_template",
        "61f8687f0fca06cb9702de0a3e029cdb9cc90208fd3e7f24e57577b877bf2678",
    ): "{{ bos_token or '' }}{% for message in messages %}{{ message.content }}{% endfor %}",
}
SUPPORTED_METADATA_REPOS = frozenset(entry["repo"] for entry in models() if entry.get("metadata_profile") is not None)


class UnsupportedMetadata(ValueError):
    """Valid metadata is outside implemented profiles; never include content."""


def _inline_text(role: str, spec: object) -> str:
    if not isinstance(spec, dict) or set(spec) != {"sha256"}:
        raise ConversionError("Invalid inline metadata profile.")
    digest = spec["sha256"]
    if not isinstance(digest, str):
        raise ConversionError("Invalid inline metadata profile.")
    value = _INLINE_TEXT.get((role, digest))
    if value is None:
        raise UnsupportedMetadata("Unsupported inline metadata profile.")
    if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != digest:
        raise ConversionError("Inline metadata profile identity changed.")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        return read_json(path)
    except ConversionError:
        raise ConversionError(f"Cannot read {label} metadata.") from None


def _optional_text(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        raise UnsupportedMetadata("Unsupported special-token representation.")
    raise ConversionError("Invalid special-token metadata.")


def _token_ids(value: object, *, optional: bool) -> list[int]:
    if value is None and optional:
        return []
    if type(value) is int:
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ConversionError("Invalid stop-token metadata.")
    if any(type(token_id) is not int or token_id < 0 for token_id in values):
        raise ConversionError("Invalid stop-token metadata.")
    return values


def _token_ids_to_text(tokenizer: Tokenizer, value: object) -> str | None:
    # Lalamo falls back to config.json and decodes only the first identifier.
    if value is None:
        return None
    return tokenizer.decode(_token_ids(value, optional=False)[:1], skip_special_tokens=False)


def _add_tokens(tokenizer: Tokenizer, tokenizer_config: dict[str, Any]) -> None:
    added_decoder = tokenizer_config.get("added_tokens_decoder")
    if added_decoder is None:
        return
    if not isinstance(added_decoder, dict):
        raise ConversionError("Invalid added-token metadata.")
    special: list[AddedToken] = []
    regular: list[AddedToken] = []
    boolean_fields = ("single_word", "normalized", "special", "lstrip", "rstrip")
    for token in added_decoder.values():
        if not isinstance(token, dict) or not isinstance(token.get("content"), str):
            raise ConversionError("Invalid added-token metadata.")
        if any(type(token.get(field)) is not bool for field in boolean_fields):
            raise ConversionError("Invalid added-token metadata.")
        added = AddedToken(token["content"], **{field: token[field] for field in boolean_fields})
        (special if token["special"] else regular).append(added)
    # Order matches Lalamo: register all special tokens before regular tokens.
    tokenizer.add_special_tokens(special)
    tokenizer.add_tokens(regular)


def _optional_sampling_number(source: dict[str, Any], name: str, *, integer: bool = False) -> int | float | None:
    value = source.get(name)
    if value is None:
        return None
    if integer:
        if type(value) is not int:
            raise ConversionError("Invalid generation metadata.")
        return value
    if type(value) not in (int, float):
        raise ConversionError("Invalid generation metadata.")
    try:
        result = float(value)
    except OverflowError:
        raise ConversionError("Invalid generation metadata.") from None
    if not math.isfinite(result):
        raise ConversionError("Invalid generation metadata.")
    return result


def _generation_config(source: dict[str, Any], hf_config: dict[str, Any]) -> dict[str, Any]:
    stop_ids = _token_ids(hf_config.get("eos_token_id"), optional=False)
    stop_ids.extend(_token_ids(source.get("eos_token_id"), optional=True))
    # This reproduces _policy_from_hf_config, including its intentional omission
    # of do_sample and HF fields that Lalamo does not import.
    return {
        "stop_token_ids": sorted(set(stop_ids)),
        "temperature": _optional_sampling_number(source, "temperature"),
        "top_k": _optional_sampling_number(source, "top_k", integer=True),
        "top_p": _optional_sampling_number(source, "top_p"),
        "min_p": _optional_sampling_number(source, "min_p"),
        "banned_tokens": None,
        "repetition_penalty": _optional_sampling_number(source, "repetition_penalty"),
        "presence_penalty": None,
        "frequency_penalty": None,
        "suffix_repetition_length": None,
    }


def _file_path(
    source: Path,
    model_entry: dict[str, Any],
    role: str,
    resolve_file: Callable[..., Path] | None,
) -> Path:
    spec = model_entry["files"].get(role)
    if not isinstance(spec, dict) or not isinstance(spec.get("filename"), str):
        raise UnsupportedMetadata("Unsupported metadata source specification.")
    if resolve_file is not None:
        try:
            return Path(resolve_file(spec["filename"], repo=spec["repo"]))
        except Exception:
            raise ConversionError("Cannot resolve metadata source file.") from None
    if spec["repo"] != model_entry["repo"]:
        raise ConversionError("External metadata source requires a resolved snapshot.")
    return source / spec["filename"]


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ConversionError(f"Cannot read {label} metadata.") from None


def _chat_template(
    source: Path,
    model_entry: dict[str, Any],
    tokenizer_config: dict[str, Any],
    resolve_file: Callable[..., Path] | None,
) -> str:
    prompt_template = tokenizer_config.get("chat_template")
    template_spec = model_entry["files"].get("chat_template")
    inline_id = model_entry["metadata"].get("inline_sources", {}).get("chat_template")
    if prompt_template is not None and (template_spec is not None or inline_id is not None):
        raise ConversionError("Conflicting chat-template metadata.")
    if prompt_template is None and template_spec is not None:
        template_path = _file_path(source, model_entry, "chat_template", resolve_file)
        field_name = template_spec.get("field")
        if field_name is None:
            prompt_template = _read_text(template_path, "chat-template")
        else:
            prompt_template = _read_object(template_path, "chat-template").get(field_name)
    elif prompt_template is None and inline_id is not None:
        prompt_template = _inline_text("chat_template", inline_id)
    if not isinstance(prompt_template, str):
        named_templates = (
            isinstance(prompt_template, list)
            and bool(prompt_template)
            and all(
                isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("template"), str)
                for item in prompt_template
            )
        )
        mapped_templates = (
            isinstance(prompt_template, dict)
            and bool(prompt_template)
            and all(isinstance(value, str) for value in prompt_template.values())
        )
        if named_templates or mapped_templates:
            raise UnsupportedMetadata("Unsupported chat-template representation.")
        raise ConversionError("Invalid chat-template metadata.")
    return _GENERATION_BLOCK_TAG.sub("", prompt_template)


def _profile_codec(metadata: dict[str, Any]) -> dict[str, Any]:
    parser_id = metadata.get("parser_id")
    end_id = metadata.get("end_of_thinking_id")
    reasoning_id = metadata.get("reasoning_id")
    parsers = {
        None: None,
        "optional_thinking": _OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        "granite_thinking": _GRANITE_THINKING_OUTPUT_PARSER_REGEX,
    }
    ends = {
        None: None,
        "qwen_think_end": "\n</think>",
        "double_newline_think_end": "\n\n</think>",
        "think_end": "</think>",
    }
    if parser_id not in parsers:
        raise UnsupportedMetadata("Unsupported output-parser profile.")
    if end_id not in ends:
        raise UnsupportedMetadata("Unsupported reasoning-boundary profile.")
    if reasoning_id not in (None, "boolean_default_on", "boolean_default_off", "granite_boolean_default_off"):
        raise UnsupportedMetadata("Unsupported reasoning-default profile.")
    roles = metadata.get("roles")
    if not isinstance(roles, dict) or any(
        not isinstance(roles.get(role), str) for role in ("system", "user", "assistant")
    ):
        raise ConversionError("Invalid role-name profile.")
    reasoning = None
    if reasoning_id is not None:
        reasoning = {
            "default_reasoning_effort": "medium" if reasoning_id == "boolean_default_on" else "no_reasoning",
            "field_name": "thinking" if reasoning_id == "granite_boolean_default_off" else "enable_thinking",
            "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
        }
    return {
        "output_parser_regex": parsers[parser_id],
        "system_role_name": roles["system"],
        "user_role_name": roles["user"],
        "assistant_role_name": roles["assistant"],
        "end_of_thinking_tag": ends[end_id],
        "reasoning_config": reasoning,
    }


def _native_generation_config(source: dict[str, Any], hf_config: dict[str, Any]) -> dict[str, Any]:
    """Interpret an inline native GenerationConfig, rather than an HF file."""
    result = _generation_config({}, hf_config)
    result["stop_token_ids"] = sorted(
        set(result["stop_token_ids"]) | set(_token_ids(source.get("stop_token_ids", []), optional=False))
    )
    for name in result:
        if name == "stop_token_ids":
            continue
        if name == "banned_tokens":
            value = source.get(name)
            result[name] = None if value is None else _token_ids(value, optional=False)
        else:
            result[name] = _optional_sampling_number(
                source, name, integer=name in ("top_k", "suffix_repetition_length")
            )
    return result


def _profile_generation_config(
    source: Path,
    model_entry: dict[str, Any],
    hf_config: dict[str, Any],
    resolve_file: Callable[..., Path] | None,
) -> dict[str, Any]:
    metadata = model_entry["metadata"]
    if "generation_config" in model_entry["files"]:
        generation_source = _read_object(
            _file_path(source, model_entry, "generation_config", resolve_file), "generation"
        )
        result = _generation_config(generation_source, hf_config)
    else:
        inline_source = metadata.get("generation_config")
        if inline_source is not None and not isinstance(inline_source, dict):
            raise ConversionError("Invalid generation-config profile.")
        result = _native_generation_config(inline_source or {}, hf_config)

    overrides = metadata.get("generation_params_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ConversionError("Invalid generation-override profile.")
        defaults = _native_generation_config({}, {"eos_token_id": []})
        normalized = _native_generation_config(overrides, {"eos_token_id": []})
        # Match GenerationConfig.override_with: defaults do not override and
        # the token-list fields merge only when the existing field is nondefault.
        for name, value in normalized.items():
            if value == defaults[name]:
                continue
            if name in ("stop_token_ids", "banned_tokens") and result[name] != defaults[name]:
                result[name] = sorted(set(result[name]) | set(value))
            else:
                result[name] = value
    return result


def write_metadata(
    source: Path,
    output: Path,
    hf_config: dict[str, Any],
    *,
    model_entry: dict[str, Any] | None = None,
    resolve_file: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    """Normalize tokenizer metadata using an explicit registered model profile.

    The caller resolves metadata files at immutable source revisions, including
    cross-repository ConfigMap entries, before invoking this adapter. Files are
    read through ``resolve_file(filename, repo=...)`` when supplied. Without a
    resolver, only files in the model's own local source directory are accepted.
    This function creates output only after validating every metadata fragment.
    """
    repo = hf_config.get("_stark_model_repo")
    if model_entry is None:
        model_entry = get_model(repo) if isinstance(repo, str) else None
    if model_entry is None or model_entry.get("repo") != repo or model_entry.get("metadata_profile") is None:
        raise UnsupportedMetadata("Unverified model metadata profile.")
    expected_type = _MODEL_TYPES.get(model_entry.get("config_class"))
    if expected_type is None:
        raise UnsupportedMetadata("Unsupported metadata architecture.")
    if hf_config.get("model_type") != expected_type:
        raise ConversionError("Model architecture does not match metadata profile.")

    source = Path(source)
    output = Path(output)
    tokenizer_config = _read_object(_file_path(source, model_entry, "tokenizer_config", resolve_file), "tokenizer")
    prompt_template = _chat_template(source, model_entry, tokenizer_config, resolve_file)
    codec_profile = _profile_codec(model_entry["metadata"])
    system_prompt = None
    if "system_prompt" in model_entry["files"]:
        system_prompt = _read_text(_file_path(source, model_entry, "system_prompt", resolve_file), "system-prompt")
    elif model_entry["metadata"].get("inline_sources", {}).get("system_prompt") is not None:
        system_prompt = _inline_text("system_prompt", model_entry["metadata"]["inline_sources"]["system_prompt"])

    bos_token = _optional_text(tokenizer_config.get("bos_token"))
    eos_token = _optional_text(tokenizer_config.get("eos_token"))
    generation_config = _profile_generation_config(source, model_entry, hf_config, resolve_file)

    try:
        tokenizer = Tokenizer.from_file(str(_file_path(source, model_entry, "tokenizer", resolve_file)))
        _add_tokens(tokenizer, tokenizer_config)
        if bos_token is None:
            bos_token = _token_ids_to_text(tokenizer, hf_config.get("bos_token_id"))
        if eos_token is None:
            eos_token = _token_ids_to_text(tokenizer, hf_config.get("eos_token_id"))
    except (UnsupportedMetadata, ConversionError):
        raise
    except Exception:
        # Tokenizers errors can contain source data. Expose only this fixed
        # reason; the CLI must never print the underlying exception context.
        raise ConversionError("Tokenizer normalization failed.") from None

    token_codec_config = {
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
        raise ConversionError("Cannot write tokenizer metadata.") from None
    return {"token_codec_config": token_codec_config, "generation_config": generation_config}
