"""Fixed catalog entries for native models outside generic conversion."""

from __future__ import annotations

import hashlib

FAMILY_ID = "glm5_next"
CHOICE = "glm53-q2"
FAMILY_CHOICES = {FAMILY_ID: (CHOICE,), "ricursive": ("ripple-q8",)}
FAMILY_IDS = tuple(FAMILY_CHOICES)
_NAMES = {FAMILY_ID: "GLM-5.3 Flash", "ricursive": "Ripple-65B Preview"}
_ITEM_NAMES = {CHOICE: "GLM-5.3-Flash-GGUF", "ripple-q8": "Ripple-65B-Preview-GGUF"}


def _source(choice):
    # Keep source metadata owned by the preparation path and avoid importing its
    # download dependencies unless this native catalog family is requested.
    from .hybrid_prepare import SOURCES

    return SOURCES[choice]


def family_definition(family_id: str = FAMILY_ID) -> dict:
    """Describe pinned native preparation without granting generic admission."""
    choices = FAMILY_CHOICES[family_id]
    return {
        "family_id": family_id,
        "name": _NAMES[family_id],
        "architecture_aliases": [family_id],
        "config_classes": [],
        "model_kind": "language",
        "conversion": {
            "status": "implemented",
            "containers": [".gguf"],
            "source_formats": [_source(choice)["quantization"] for choice in choices],
            "output_dtypes": [],
        },
        "modes": {
            "text": {
                "status": "conditional",
                "message": "Native text generation is implemented; the checkpoint must load successfully.",
                "caveats": ["native_load_required", "checkpoint_specific_execution"],
            },
            "decision": {
                "status": "unsupported",
                "message": "Decision scoring is not enabled for this native hybrid family.",
                "caveats": [],
            },
            "vision-language": {
                "status": "unsupported",
                "message": "Image input is not supported for this model family.",
                "caveats": [],
            },
        },
    }


def catalog_item(choice: str = CHOICE) -> dict:
    """Return a pinned checkpoint prepared by the native path."""
    source = _source(choice)
    repo, revision = source["weights_repo"], source["weights_revision"]
    return {
        "id": hashlib.sha256((repo + "@" + revision).encode()).hexdigest()[:32],
        "repo": repo,
        "revision": revision,
        "name": _ITEM_NAMES[choice],
        "size_label": None,
        "parameter_count": None,
        "weight_bytes": source["weight_bytes"],
        "quantization": source["quantization"],
        "artifact_modes": ["text"],
        "default_artifact_mode": "text",
        "downloadable": True,
        "model_group": None,
    }


def catalog_items(family_id: str) -> list[dict]:
    """Return fresh server-owned choices for a native architecture."""
    return [catalog_item(choice) for choice in FAMILY_CHOICES[family_id]]


def source_choice(family: str, item: object) -> str | None:
    """Return the preparation choice only for the exact fixed catalog item."""
    if family not in FAMILY_CHOICES or not isinstance(item, dict):
        return None
    return next((choice for choice in FAMILY_CHOICES[family] if item == catalog_item(choice)), None)
