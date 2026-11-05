"""Fixed catalog entries for native models outside generic conversion."""

from __future__ import annotations

import hashlib

FAMILY_ID = "glm5_next"
CHOICE = "glm53-q2"
QWEN_FAMILY_ID = "qwen4_exp"
QWEN_CHOICE = "qwen38-mixed"

_NATIVE_CHOICES = {
    FAMILY_ID: (CHOICE,),
    QWEN_FAMILY_ID: (QWEN_CHOICE,),
}


def source_for(choice: str) -> dict | None:
    # Keep source metadata owned by the preparation path and avoid importing its
    # download dependencies unless this native catalog family is requested.
    from .hybrid_prepare import SOURCES

    if choice not in {value for choices in _NATIVE_CHOICES.values() for value in choices}:
        return None
    return SOURCES[choice]


def allowed_choice(choice: str) -> bool:
    """Return whether a fixed source has a native catalog install path."""
    return any(choice in choices for choices in _NATIVE_CHOICES.values())


def family_definition() -> dict:
    """Describe GLM's native preparation and app modes without generic admission."""
    return {
        "family_id": FAMILY_ID,
        "name": "GLM-5.3 Flash",
        "architecture_aliases": [FAMILY_ID],
        "config_classes": [],
        "model_kind": "language",
        "conversion": {
            "status": "implemented",
            "containers": [".gguf"],
            "source_formats": ["UD-Q2_K_XL"],
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
    """Return one pinned native checkpoint, defaulting to the legacy GLM item."""
    source = source_for(choice)
    if source is None:
        raise ValueError("unknown native catalog choice")
    repo, revision = source["weights_repo"], source["weights_revision"]
    if choice == CHOICE:
        name = "GLM-5.3-Flash-GGUF"
    elif choice == QWEN_CHOICE:
        name = "Qwen3.8-Flash-Next-MLX-mixed-4_8bit"
    else:
        raise ValueError("unknown native catalog choice")
    return {
        "id": hashlib.sha256((repo + "@" + revision).encode()).hexdigest()[:32],
        "repo": repo,
        "revision": revision,
        "name": name,
        "size_label": None,
        "parameter_count": None,
        "weight_bytes": source["weight_bytes"],
        "quantization": source["quantization"],
        "artifact_modes": ["text"],
        "default_artifact_mode": "text",
        "downloadable": True,
        "model_group": None,
    }


def catalog_items(family: str) -> list[dict]:
    """Return the fixed native choices for a family, without generic duplicates."""
    return [catalog_item(choice) for choice in _NATIVE_CHOICES.get(family, ())]


def source_choice(family: str, item: object) -> str | None:
    """Return a preparation choice only for an exact fixed catalog item."""
    if not isinstance(item, dict):
        return None
    return next((choice for choice in _NATIVE_CHOICES.get(family, ()) if item == catalog_item(choice)), None)
