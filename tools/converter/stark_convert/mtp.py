"""Loaded-worker MTP capabilities and request-local toggle resolution."""

from __future__ import annotations

import struct
from typing import Any

_MAX_DRAFTS = 2**32 - 1
_SAMPLING = "exact-rejection-v1"
_LILY_TOPK_POLICY = "lily-topk20-one-draft-v1"


def _same_native_float(value: Any, expected: float) -> bool:
    """Match the generation protocol's existing F32 sampling representation."""
    if type(value) not in (int, float):
        return False
    try:
        return struct.pack(">f", value) == struct.pack(">f", expected)
    except (OverflowError, struct.error):
        return False


def _unsupported() -> dict[str, Any]:
    return {
        "supported": False,
        "max_drafts": 0,
        "recommended_drafts": 0,
        "default_drafts": 0,
        "operations": [],
    }


def mtp_capabilities(readiness: Any) -> dict[str, Any]:
    """Return a fresh, fail-closed descriptor from actual worker readiness.

    Earlier exact-MTP workers omitted recommendation and operations. Their
    toggle uses at most two drafts, bounded by the advertised maximum. Model
    names and unverified checkpoint metadata never establish support.
    """
    if not isinstance(readiness, dict) or (
        ("ready" in readiness and readiness["ready"] is not True)
        or ("mode" in readiness and readiness["mode"] != "text-generation-v1")
    ):
        return _unsupported()
    value = readiness.get("mtp")
    if not isinstance(value, dict) or value.get("supported") is not True:
        return _unsupported()
    maximum = value.get("max_drafts")
    if type(maximum) is not int or not 1 <= maximum <= _MAX_DRAFTS:
        return _unsupported()
    recommended = value.get("recommended_drafts", min(2, maximum))
    default = value.get("default_drafts", 0)
    operations = value.get("operations", ["generate"])
    verification = value.get("verification")
    if (
        type(recommended) is not int
        or not 1 <= recommended <= maximum
        or type(default) is not int
        or default != 0
        or operations != ["generate"]
        or not isinstance(operations, list)
        or value.get("sampling") != _SAMPLING
        or not isinstance(verification, str)
        or not 1 <= len(verification) <= 128
        or any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in verification)
    ):
        return _unsupported()
    current = {
        "supported": True,
        "max_drafts": maximum,
        "recommended_drafts": recommended,
        "default_drafts": 0,
        "sampling": _SAMPLING,
        "verification": verification,
        "operations": ["generate"],
    }
    # Older/future policy extensions safely retain the scalar recommendation.
    # Only a loaded worker can grant this policy; model names are not evidence.
    if value.get("sampling_recommendation_policy") == _LILY_TOPK_POLICY:
        current["sampling_recommendation_policy"] = _LILY_TOPK_POLICY
    return current


def resolve_mtp(enabled: Any, capabilities: Any, sampling: Any = None) -> int:
    """Resolve one boolean toggle without carrying state across requests."""
    if type(enabled) is not bool:
        raise ValueError("invalid_mtp_option")
    if not enabled:
        return 0
    current = mtp_capabilities({"mtp": capabilities})
    if not current["supported"]:
        raise ValueError("mtp_not_supported")
    if (
        current.get("sampling_recommendation_policy") == _LILY_TOPK_POLICY
        and isinstance(sampling, dict)
        and sampling.keys() <= {"mode", "temperature", "top_k", "top_p", "min_p"}
        and sampling.get("mode") == "stochastic"
        and _same_native_float(sampling.get("temperature"), 1.0)
        and type(sampling.get("top_k")) is int
        and sampling["top_k"] == 20
        and _same_native_float(sampling.get("top_p"), 0.95)
        and sampling.get("min_p") is None
    ):
        return 1
    return current["recommended_drafts"]
