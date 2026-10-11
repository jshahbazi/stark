"""Import verified Qwen3 metadata without importing Lalamo or JAX.

Adapted from Lalamo's metadata import, tokenizer configuration, generation
configuration, and Qwen model specifications at revision
831accb8252a6b6606c378048caeb44178f1a050.
Copyright (c) 2025 Mirai Tech Inc. MIT licensed; see ../LICENSE.lalamo.

The registry identity is supplied by the caller as ``_stark_model_repo`` in the
in-memory Hugging Face configuration. It is never written into model metadata.
Unknown identities must use the Lalamo backend: architecture alone does not
determine a model's reasoning defaults or output parser.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from tokenizers import AddedToken, Tokenizer

from stark_convert.streaming import ConversionError, read_json

__all__ = ["SUPPORTED_METADATA_REPOS", "UnsupportedMetadata", "write_metadata"]

_OPTIONAL_THINKING_OUTPUT_PARSER_REGEX = (
    r"(?s)(?=(<think>|.*?</think>)?)"
    r"(?(1)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*|\Z))"
    r"(?P<response>.*)\Z"
)
_GENERATION_BLOCK_TAG = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
_QWEN3_BASE_REPOS = frozenset(f"Qwen/Qwen3-{size}" for size in ("0.6B", "1.7B", "4B", "8B", "14B", "32B"))
_QWEN3_INSTRUCT_REPO = "Qwen/Qwen3-4B-Instruct-2507"
_QWEN3_THINKING_REPO = "Qwen/Qwen3-4B-Thinking-2507"
SUPPORTED_METADATA_REPOS = _QWEN3_BASE_REPOS | {_QWEN3_INSTRUCT_REPO, _QWEN3_THINKING_REPO}


class UnsupportedMetadata(ValueError):
    """Metadata needs Lalamo; messages contain no checkpoint content."""


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


def write_metadata(source: Path, output: Path, hf_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tokenizer and return Lalamo-compatible config fragments.

    Supports only registered, dense Qwen3 metadata profiles. All source files
    are local; the caller resolves/downloads the checkpoint before invoking
    this adapter. A checkpoint's chat template and sampling parameters remain
    authoritative. The registered repository supplies only parser/reasoning
    defaults, exactly as Lalamo's model specification does.

    ``output`` is the caller's staging directory. The returned fragments are
    merged with the streaming decoder configuration by the caller. This
    function creates ``tokenizer.json`` only after validating all metadata.
    """
    if hf_config.get("model_type") != "qwen3":
        raise UnsupportedMetadata("Unsupported metadata architecture.")
    repo = hf_config.get("_stark_model_repo")
    if not isinstance(repo, str) or repo not in SUPPORTED_METADATA_REPOS:
        raise UnsupportedMetadata("Unverified model metadata profile; use Lalamo.")

    source = Path(source)
    output = Path(output)
    tokenizer_config = _read_object(source / "tokenizer_config.json", "tokenizer")
    generation_source = _read_object(source / "generation_config.json", "generation")
    prompt_template = tokenizer_config.get("chat_template")
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
    bos_token = _optional_text(tokenizer_config.get("bos_token"))
    eos_token = _optional_text(tokenizer_config.get("eos_token"))
    generation_config = _generation_config(generation_source, hf_config)

    try:
        tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
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

    optional_thinking = repo in _QWEN3_BASE_REPOS
    has_thinking_parser = optional_thinking or repo == _QWEN3_THINKING_REPO
    reasoning_config = (
        {
            "default_reasoning_effort": "medium",
            "field_name": "enable_thinking",
            "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
        }
        if optional_thinking
        else None
    )
    token_codec_config = {
        "type": "ChatCodecConfig",
        "prompt_template": _GENERATION_BLOCK_TAG.sub("", prompt_template),
        "output_parser_regex": _OPTIONAL_THINKING_OUTPUT_PARSER_REGEX if has_thinking_parser else None,
        "system_role_name": "system",
        "user_role_name": "user",
        "assistant_role_name": "assistant",
        "eos_token": eos_token,
        "bos_token": bos_token,
        "end_of_thinking_tag": "\n</think>" if has_thinking_parser else None,
        "default_system_prompt": None,
        "reasoning_config": reasoning_config,
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output / "tokenizer.json"))
    except Exception:
        raise ConversionError("Cannot write tokenizer metadata.") from None
    return {"token_codec_config": token_codec_config, "generation_config": generation_config}
