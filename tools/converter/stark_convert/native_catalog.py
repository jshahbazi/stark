"""Fixed catalog entries for native models outside generic conversion."""

from __future__ import annotations

import hashlib

FAMILY_ID = "glm5_next"
CHOICE = "glm53-q2"
QWEN_FAMILY_ID = "qwen4_exp"
QWEN_CHOICE = "qwen38-mixed"
FLINT_FAMILY_ID = "flint"
FLINT_CHOICE = "flint-int4"
RIPPLE_FAMILY_ID = "ricursive"

_NATIVE_CHOICES = {
    FAMILY_ID: (CHOICE,),
    QWEN_FAMILY_ID: (QWEN_CHOICE, "qwen38-lily-q4-xl", "qwen38-mlx-serve-mixed"),
    FLINT_FAMILY_ID: (FLINT_CHOICE,),
    RIPPLE_FAMILY_ID: ("ripple-q8", "ripple-q4-k-m"),
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


def family_definition(family: str = FAMILY_ID) -> dict:
    """Describe fixed native preparation and app modes without generic admission."""
    return {
        "family_id": family,
        "name": "Ripple-65B Preview"
        if family == RIPPLE_FAMILY_ID
        else "Flint"
        if family == FLINT_FAMILY_ID
        else "GLM-5.3 Flash",
        "architecture_aliases": [family],
        "config_classes": [],
        "model_kind": "language",
        "conversion": {
            "status": "implemented",
            "containers": [".safetensors"] if family == FLINT_FAMILY_ID else [".gguf"],
            "source_formats": [source_for(choice)["quantization"] for choice in _NATIVE_CHOICES[family]]
            if family == RIPPLE_FAMILY_ID
            else ["MLX affine INT4"]
            if family == FLINT_FAMILY_ID
            else ["UD-Q2_K_XL"],
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
                "message": "Decision scoring is not enabled for this native catalog choice.",
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
    elif choice in {"ripple-q8", "ripple-q4-k-m"}:
        name = "Ripple-65B-Preview-GGUF"
    elif choice == FLINT_CHOICE:
        name = "Flint INT4"
    elif choice == QWEN_CHOICE:
        name = "Qwen3.8-Flash-Next-MLX-mixed-4_8bit"
    elif choice == "qwen38-lily-q4-xl":
        name = "Qwen3.8-Flash-Next-lily-q4-xl"
    elif choice == "qwen38-mlx-serve-mixed":
        name = "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
    else:
        raise ValueError("unknown native catalog choice")
    return {
        "id": hashlib.sha256((repo + "@" + revision).encode()).hexdigest()[:32],
        "repo": repo,
        "revision": revision,
        "name": name,
        "size_label": "65B" if source["architecture"] == RIPPLE_FAMILY_ID else None,
        "parameter_count": 64_782_689_360 if source["architecture"] == RIPPLE_FAMILY_ID else None,
        "weight_bytes": source["weight_bytes"],
        "quantization": source["quantization"],
        "artifact_modes": ["text", "vision-language"]
        if choice in {"qwen38-lily-q4-xl", "qwen38-mlx-serve-mixed"}
        else ["text"],
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
    # A reference score only orders the catalogue; it is not part of the
    # installable identity, which must still match in every other field.
    identity = {key: value for key, value in item.items() if key != "reference_score"}
    return next((choice for choice in _NATIVE_CHOICES.get(family, ()) if identity == catalog_item(choice)), None)


def unavailable_item(family: str, repo: str, revision: str, config, names, sizes) -> dict | None:
    """Describe a discovered family match that has no admitted native installer.

    This is deliberately metadata-only: family/config matches are surfaced to
    users, but never imply that an artifact can be installed or executed.
    """
    if family not in {FAMILY_ID, FLINT_FAMILY_ID}:
        return None
    files = list(names)
    expected = {family, family + "_text"}
    model_type = config.get("model_type") if isinstance(config, dict) else None
    nested = config.get("text_config") if isinstance(config, dict) else None
    if family == FAMILY_ID and isinstance(nested, dict):
        if nested.get("model_type") not in {None, "glm5_next_text"}:
            return None
        model_type = model_type or nested.get("model_type")
    if model_type is not None and model_type not in expected:
        return None
    lowered = [name.lower() for name in files]
    weight_names = [
        name
        for name in lowered
        if name.endswith((".gguf", ".safetensors", ".bin"))
        and not name.rsplit("/", 1)[-1].startswith(("adapter_model.", "tokenizer."))
    ]
    if not weight_names:
        return None
    # Config-less GGUF repositories are discoverable only when their repository
    # name itself identifies the GLM-5.3 Flash family.
    if family == FAMILY_ID and model_type is None:
        if "glm-5.3-flash" not in repo.lower() or not any(name.endswith(".gguf") for name in weight_names):
            return None
    if family == FLINT_FAMILY_ID and model_type is None:
        return None

    if family == FAMILY_ID and any(name.endswith(".safetensors") for name in weight_names):
        reason = (
            "GLM-5.3 Flash Safetensors import is unavailable. "
            "Stark currently imports the pinned UD-Q2_K_XL GGUF export."
        )
        label = "Safetensors/MLX"
    elif family == FAMILY_ID:
        reason = "The GLM-5.3 Flash installer accepts only the pinned UD-Q2_K_XL GGUF layout."
        label = "GGUF"
    elif any(name.endswith(".safetensors") for name in weight_names):
        reason = (
            "This Flint Safetensors export has no supported import path. "
            "Stark currently imports the pinned MLX affine INT4 checkpoint."
        )
        label = "Safetensors/MLX"
    else:
        reason = (
            "This Flint container has no supported import path. Stark currently imports MLX affine INT4 Safetensors."
        )
        label = "GGUF" if any(name.endswith(".gguf") for name in weight_names) else "Unknown weights"

    byte_count = None
    if isinstance(sizes, dict):
        present = [
            (name, sizes.get(name))
            for name in files
            if name.lower().endswith((".gguf", ".safetensors", ".bin"))
            and not name.lower().rsplit("/", 1)[-1].startswith(("adapter_model.", "tokenizer."))
        ]
        if present and all(type(size) is int and size > 0 for _, size in present):
            total = sum(size for _, size in present)
            if total <= 2**63 - 1:
                byte_count = total
    return {
        "id": hashlib.sha256((repo + "@" + revision).encode()).hexdigest()[:32],
        "repo": repo,
        "revision": revision,
        "name": repo.rsplit("/", 1)[-1],
        "size_label": None,
        "parameter_count": None,
        "weight_bytes": byte_count,
        "quantization": label,
        "artifact_modes": ["text"],
        "default_artifact_mode": "text",
        "downloadable": False,
        "model_group": None,
        "unavailable_reason": reason,
    }
