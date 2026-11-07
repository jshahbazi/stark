"""Explicit, content-bound image upgrade for retained original Qwen checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ._build_metadata import file_hash, open_regular
from .vision_family import _PROCESSOR

PROFILE = "qwen38-lily-retained-bf16-vision-v1"
MANIFEST = "stark-image-artifact.json"
# The native processor implements this fixed recipe, with its own smaller
# request limits. Checkpoint sidecars cannot change its operations or budgets.
PROCESSOR_BYTES = (json.dumps(_PROCESSOR, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read(path: Path, maximum: int) -> bytes:
    with open_regular(path) as stream:
        if stream.seek(0, 2) > maximum:
            raise ValueError("native image metadata exceeds its limit")
        stream.seek(0)
        data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("native image metadata exceeds its limit")
        return data


def prepare_image_artifact(directory: Path, choice: str) -> dict:
    """Hash a complete pinned native folder; add no weights or execution claims."""
    from .hybrid_prepare import SOURCES, prepared_identity

    if choice not in {"qwen38-mlx-serve-mixed", "qwen38-lily-q4-xl"}:
        raise ValueError("native image upgrade requires a supported Qwen checkpoint")
    identity = prepared_identity(directory)
    source = SOURCES[choice]
    if (
        identity is None
        or identity["repo"] != source["weights_repo"]
        or identity["revision"] != source["weights_revision"]
    ):
        raise ValueError("native image upgrade requires its complete pinned source receipt")
    config = json.loads(_read(directory / "config.json", 1024 * 1024))
    if not isinstance(config.get("vision_config"), dict):
        raise ValueError("native checkpoint does not declare a vision tower")
    # Header admission independently verifies every visual shape and BF16
    # payload before the native runtime publishes capability.
    index = json.loads(_read(directory / "model.safetensors.index.json", 16 * 1024 * 1024))
    visual = [name for name in index.get("weight_map", {}) if name.startswith(("model.visual.", "vision_tower."))]
    if len(visual) != 9 + 12 * config["vision_config"].get("depth", 0):
        raise ValueError("native checkpoint does not retain a complete vision tower")
    template = config.get("token_codec_config", {}).get("prompt_template")
    if not isinstance(template, str) or not 0 < len(template.encode()) <= 256 * 1024:
        raise ValueError("native image template is unavailable")
    processor = directory / "preprocessor_config.json"
    standalone = directory / "chat_template.jinja"
    if processor.is_symlink() or standalone.is_symlink():
        raise ValueError("native image metadata must be regular local files")
    if processor.exists() and _read(processor, 64 * 1024) != PROCESSOR_BYTES:
        # Serialize only the implemented fixed recipe; never modify an
        # unrecognized local sidecar during an upgrade.
        from .vision_family import normalize_family_processor

        if normalize_family_processor(json.loads(_read(processor, 64 * 1024)), processor.name) != _PROCESSOR:
            raise ValueError("native image processor is outside the fixed profile")
    if standalone.exists() and _read(standalone, 256 * 1024) != template.encode():
        raise ValueError("native image template metadata disagrees")
    processor.write_bytes(PROCESSOR_BYTES)
    if not standalone.exists():
        standalone.write_text(template)
    weights = sorted(directory.glob("*.safetensors"))
    table = directory / "ngram_table.bin"
    if table.exists():
        weights.append(table)
    weights.sort(key=lambda path: path.name)
    if not 1 <= len(weights) <= 256:
        raise ValueError("native image shard inventory exceeds its limit")
    records = {path.name: {"size": path.stat().st_size, "sha256": file_hash(path)} for path in weights}
    shard_names = {name for name in records if name.endswith(".safetensors")}
    if set(index.get("weight_map", {}).values()) != shard_names:
        raise ValueError("native image shard index disagrees with its inventory")
    digest = hashlib.sha256()
    for name, record in records.items():
        digest.update(name.encode() + b"\0" + record["sha256"].encode())
    result = {
        "kind": "Qwen38RetainedVisionArtifact",
        "schema_version": 6,
        "family_profile": PROFILE,
        "family_version": 1,
        "model_source": identity["repo"],
        "model_revision": identity["revision"],
        "config_sha256": file_hash(directory / "config.json"),
        "artifact_sha256": digest.hexdigest(),
        "tokenizer_sha256": file_hash(directory / "tokenizer.json"),
        "processor_sha256": file_hash(processor),
        "template_sha256": file_hash(standalone),
        "weight_files": records,
        "index_sha256": file_hash(directory / "model.safetensors.index.json"),
    }
    path = directory / MANIFEST
    if path.is_symlink():
        raise ValueError("native image manifest must be a regular local file")
    temporary = directory / (MANIFEST + ".tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
    return result
