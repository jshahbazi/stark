"""Shared Hugging Face config normalization ahead of the family planners.

Each family planner admits an explicit set of config keys and rejects anything
else. Many published configs differ from a qualified one only by publisher
annotations, sampling defaults or a newer Transformers serialization of the
same setting. This module rewrites exactly those reviewed forms and nothing
else, so the planner's own checks stay authoritative for every other key.

The same pass runs for header-only catalogue admission and for source-backed
conversion, because each planner applies it on the line that reads its config.
A model is therefore never offered by the catalogue and then refused at
installation over a difference this module removes.

Rules fail closed: a value is only removed or renamed when its form is fully
recognised, no value is ever invented, and malformed annotations are refused.
"""

from __future__ import annotations

import math

from .streaming import ConversionError

# Families whose planners read the config through ``normalize_config``. The
# Gemma 3 image wrapper, Qwen 3.8 external profiles, LLaMBA and ModernBERT are
# deliberately absent: their configs are pinned or validated by separate
# contracts and no rejected public checkpoint needed a rewrite.
_FAMILIES = frozenset(
    {
        "qwen3",
        "qwen2",
        "llama",
        "mistral",
        "gemma2",
        "gemma3_text",
        "gemma4_text",
        "olmo3",
        "phi3",
        "granite",
        "smollm3",
        "qwen3_5",
        "qwen3_5_moe",
        "lfm2",
    }
)

# Planners that read their decoder settings from a nested ``text_config``.
_TEXT_CONFIG_FAMILIES = frozenset({"gemma4_text", "qwen3_5", "qwen3_5_moe"})

# Planners whose single global rotary base is the flat ``rope_theta`` and that
# reject a nested declaration. Qwen3, Llama and LFM2 already validate the plain
# nested form themselves, with their own stricter rules, so they judge it
# unchanged. Qwen3.5 requires ``rope_parameters``, and Gemma 3/4 use
# per-attention-type forms, so they are excluded too.
_FLAT_ROPE_FAMILIES = frozenset({"qwen2", "mistral", "gemma2", "olmo3", "phi3", "granite", "smollm3"})

# Gemma text planners take the local window from ``sliding_window`` (required,
# no default) and never read ``sliding_window_size``.
_SLIDING_WINDOW_FAMILIES = frozenset({"gemma2", "gemma3_text"})


def _string(maximum: int):
    return lambda value: type(value) is str and 0 < len(value) <= maximum


def _boolean(value) -> bool:
    return type(value) is bool


def _object(value) -> bool:
    return isinstance(value, dict)


def _number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


# Publisher annotations and sampling defaults. None of them is read by any
# planner or by the native runtime; generation settings come from the separate
# generation metadata. Each value must have its documented shape before it is
# dropped. The string bounds follow the Qwen3.5 image planner's annotation check.
_TRAINING = "Config has malformed training annotation metadata"
_BROWSER = "Config has malformed browser-export metadata"
_SAMPLING = "Config has malformed sampling metadata"
_INERT_KEYS = {
    "unsloth_fixed": (_boolean, _TRAINING),
    "unsloth_version": (_string(64), _TRAINING),
    "transformers.js_config": (_object, _BROWSER),
    "model_name": (_string(256), _TRAINING),
    "generation_config": (_object, _SAMPLING),
    "do_sample": (_boolean, _SAMPLING),
    "top_p": (lambda value: _number(value) and 0 <= value <= 1, _SAMPLING),
    "top_k": (lambda value: type(value) is int and value >= 0, _SAMPLING),
    "temperature": (lambda value: _number(value) and value >= 0, _SAMPLING),
    "is_llama_config": (_boolean, _TRAINING),
    "init_method": (_string(64), _TRAINING),
}

# Judgement call, isolated so it can be removed on its own: attention dropout
# is inactive at inference, but the planners reject a nonzero value as a
# variant outside qualified coverage. Setting this to False restores that.
ZERO_INACTIVE_ATTENTION_DROPOUT = True


def _drop_inert(level: dict, prefix: str, changes: list[str]) -> None:
    for key in sorted(_INERT_KEYS):
        if key not in level:
            continue
        valid, message = _INERT_KEYS[key]
        if not valid(level[key]):
            raise ConversionError(message)
        del level[key]
        changes.append(f"dropped {prefix}{key}")


def _drop_default_rope_flags(level: dict, prefix: str, changes: list[str]) -> None:
    if "rope_interleaved" in level and level["rope_interleaved"] is False:
        del level["rope_interleaved"]
        changes.append(f"dropped {prefix}rope_interleaved=false")
    if "rope_scaling" in level and level["rope_scaling"] is None:
        del level["rope_scaling"]
        changes.append(f"dropped {prefix}rope_scaling=null")


def _rename_dtype(level: dict, prefix: str, changes: list[str]) -> None:
    # Transformers 5 serializes ``torch_dtype`` as ``dtype``. Two differing
    # declarations are left untouched for the planner to judge.
    if "dtype" in level and "torch_dtype" not in level and type(level["dtype"]) is str:
        level["torch_dtype"] = level.pop("dtype")
        changes.append(f"renamed {prefix}dtype to torch_dtype")


def _flatten_plain_rope(level: dict, prefix: str, changes: list[str]) -> None:
    parameters = level.get("rope_parameters")
    if not isinstance(parameters, dict) or not set(parameters) <= {"rope_theta", "rope_type"}:
        return
    if parameters.get("rope_type", "default") != "default" or "rope_theta" not in parameters:
        return
    theta = parameters["rope_theta"]
    if not _number(theta) or theta <= 0:
        return
    # A separate scaling declaration or a different flat theta is a conflict
    # the planner must see unchanged.
    if level.get("rope_scaling") is not None:
        return
    if "rope_theta" in level and (not _number(level["rope_theta"]) or level["rope_theta"] != theta):
        return
    level["rope_theta"] = theta
    del level["rope_parameters"]
    changes.append(f"flattened {prefix}rope_parameters")


def _drop_duplicate_sliding_window(level: dict, prefix: str, changes: list[str]) -> None:
    window = level.get("sliding_window")
    size = level.get("sliding_window_size")
    if type(window) is int and window > 0 and type(size) is int and size == window:
        del level["sliding_window_size"]
        changes.append(f"dropped {prefix}sliding_window_size")


def _zero_attention_dropout(level: dict, prefix: str, changes: list[str]) -> None:
    dropout = level.get("attention_dropout")
    if _number(dropout) and 0 < dropout <= 1:
        level["attention_dropout"] = 0.0
        changes.append(f"zeroed {prefix}attention_dropout")


def _lfm2_aliases(level: dict, changes: list[str]) -> None:
    # The LFM2 planner reads ``tie_embedding`` (default true); the Transformers
    # name ``tie_word_embeddings`` is an alias of the same setting.
    if level.get("tie_word_embeddings") is True and level.get("tie_embedding", True) is True:
        del level["tie_word_embeddings"]
        changes.append("dropped tie_word_embeddings=true")
    # Transformers' Lfm2Config stores ``block_ff_dim`` as ``intermediate_size``;
    # MLX exports of the same checkpoint carry both with equal values.
    size = level.get("intermediate_size")
    if "block_ff_dim" not in level and type(size) is int and size > 0:
        level["block_ff_dim"] = size
        changes.append("set block_ff_dim from intermediate_size")


def normalize_config(family: str, config: dict) -> tuple[dict, tuple[str, ...]]:
    """Return a normalized copy of ``config`` and the fixed change labels applied.

    The caller's dict is never modified. Families without reviewed rules and
    non-object configs are returned unchanged for the planner to judge.
    """
    if family not in _FAMILIES or not isinstance(config, dict):
        return config, ()
    result = dict(config)
    changes: list[str] = []
    levels = [("", result)]
    if family in _TEXT_CONFIG_FAMILIES and isinstance(result.get("text_config"), dict):
        result["text_config"] = dict(result["text_config"])
        levels.append(("text_config.", result["text_config"]))
    for prefix, level in levels:
        _drop_inert(level, prefix, changes)
        _drop_default_rope_flags(level, prefix, changes)
        _rename_dtype(level, prefix, changes)
        if ZERO_INACTIVE_ATTENTION_DROPOUT:
            _zero_attention_dropout(level, prefix, changes)
    if family in _FLAT_ROPE_FAMILIES:
        _flatten_plain_rope(result, "", changes)
    if family in _SLIDING_WINDOW_FAMILIES:
        _drop_duplicate_sliding_window(result, "", changes)
    if family == "lfm2":
        _lfm2_aliases(result, changes)
    return result, tuple(changes)
