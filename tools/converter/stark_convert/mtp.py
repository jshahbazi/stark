"""Loaded-worker MTP capabilities and request-local toggle resolution."""

from __future__ import annotations

from typing import Any

_MAX_DRAFTS = 2**32 - 1
_SAMPLING = "exact-rejection-v1"
_INKLING_SAMPLING = "native-target-draw-match-v1"
_INKLING_VERIFICATION = "inkling-dspark-target-draw-v1"
_INKLING_SHORT_VERIFICATION = "inkling-dspark-short-target-draw-v1"


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
    sampling = value.get("sampling")
    if (
        type(recommended) is not int
        or not 1 <= recommended <= maximum
        or type(default) is not int
        or not (
            default == 0
            # The Inkling multi-row source verifier drafts by default.
            or (
                (sampling, verification, maximum) == (_INKLING_SAMPLING, _INKLING_SHORT_VERIFICATION, 3)
                and default == maximum
            )
        )
        or operations != ["generate"]
        or not isinstance(operations, list)
        or (
            sampling != _SAMPLING
            and (sampling, verification, maximum)
            not in (
                (_INKLING_SAMPLING, _INKLING_VERIFICATION, 1),
                (_INKLING_SAMPLING, _INKLING_SHORT_VERIFICATION, 3),
            )
        )
        or not isinstance(verification, str)
        or not 1 <= len(verification) <= 128
        or any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in verification)
    ):
        return _unsupported()
    return {
        "supported": True,
        "max_drafts": maximum,
        "recommended_drafts": recommended,
        "default_drafts": default,
        "sampling": sampling,
        "verification": verification,
        "operations": ["generate"],
    }


_INKLING_GREEDY_DRAFTS = 3
_INKLING_SAMPLED_DRAFTS = 2


def is_inkling_short_verifier(capabilities: Any) -> bool:
    """Whether the loaded worker advertises the Inkling multi-row verifier."""
    return (
        isinstance(capabilities, dict)
        and capabilities.get("supported") is True
        and capabilities.get("sampling") == _INKLING_SAMPLING
        and capabilities.get("verification") == _INKLING_SHORT_VERIFICATION
    )


def default_mtp_enabled(capabilities: Any) -> bool:
    """Drafting is on by default when the loaded worker drafts by default."""
    current = mtp_capabilities({"mtp": capabilities})
    return current["supported"] and current["default_drafts"] > 0


def _greedy(sampling: Any) -> bool:
    return sampling is None or (isinstance(sampling, dict) and sampling.get("mode") == "greedy")


def resolve_mtp(enabled: Any, capabilities: Any, sampling: Any = None) -> int:
    """Resolve one boolean toggle without carrying state across requests.

    The Inkling multi-row verifier uses its measured best depth: three
    proposals for greedy and two for sampled decoding. `sampling` is the
    request's wire sampling object; omitted means greedy.
    """
    if type(enabled) is not bool:
        raise ValueError("invalid_mtp_option")
    if not enabled:
        return 0
    current = mtp_capabilities({"mtp": capabilities})
    if not current["supported"]:
        raise ValueError("mtp_not_supported")
    if is_inkling_short_verifier(current):
        best = _INKLING_GREEDY_DRAFTS if _greedy(sampling) else _INKLING_SAMPLED_DRAFTS
        return min(best, current["max_drafts"])
    return current["recommended_drafts"]
