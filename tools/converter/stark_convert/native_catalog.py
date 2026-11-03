"""Fixed catalog entries for native models outside generic conversion."""

from __future__ import annotations

import hashlib

FAMILY_ID = "glm5_next"
CHOICE = "glm53-q2"


def _source():
    # Keep source metadata owned by the preparation path and avoid importing its
    # download dependencies unless this native catalog family is requested.
    from .hybrid_prepare import SOURCES

    return SOURCES[CHOICE]


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


def catalog_item() -> dict:
    """Return the one pinned GLM checkpoint prepared by the native path."""
    source = _source()
    repo, revision = source["weights_repo"], source["weights_revision"]
    return {
        "id": hashlib.sha256((repo + "@" + revision).encode()).hexdigest()[:32],
        "repo": repo,
        "revision": revision,
        "name": "GLM-5.3-Flash-GGUF",
        "size_label": None,
        "parameter_count": None,
        "weight_bytes": source["weight_bytes"],
        "quantization": source["quantization"],
        "artifact_modes": ["text"],
        "default_artifact_mode": "text",
        "downloadable": True,
        "model_group": None,
    }


def source_choice(family: str, item: object) -> str | None:
    """Return the preparation choice only for the exact fixed catalog item."""
    if family != FAMILY_ID or not isinstance(item, dict):
        return None
    return CHOICE if item == catalog_item() else None
