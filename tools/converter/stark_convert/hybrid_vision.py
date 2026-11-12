"""Explicit, content-bound image upgrade for retained original Qwen checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ._build_metadata import file_hash, open_regular
from .vision_family import _PROCESSOR

PROFILE = "qwen38-lily-retained-bf16-vision-v1"
MANIFEST = "stark-image-artifact.json"
HASH_CACHE = ".stark-image-validation.json"
# The native processor implements this fixed recipe, with its own smaller
# request limits. Checkpoint sidecars cannot change its operations or budgets.
PROCESSOR_BYTES = (json.dumps(_PROCESSOR, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native image metadata key")
        result[key] = value
    return result


def prepared_image_identity(directory: Path) -> dict | None:
    """Inspect a retained-image descriptor without claiming payload validation.

    The native loader verifies every recorded shard hash before loading. This
    bounded host inspection binds metadata and checks the on-disk inventory;
    it never hashes a hundred-gigabyte checkpoint during model-list refreshes.
    """
    from .hybrid_prepare import prepared_identity

    try:
        base = prepared_identity(directory)
        if base is None or base["architecture"] != "qwen4_exp" or not _remote_identity(base):
            return None
        value = json.loads(_read(directory / MANIFEST, 262144), object_pairs_hook=_unique_object)
        hashes = {
            "config_sha256",
            "artifact_sha256",
            "tokenizer_sha256",
            "processor_sha256",
            "template_sha256",
            "index_sha256",
        }
        keys = hashes | {
            "kind",
            "schema_version",
            "family_profile",
            "family_version",
            "model_source",
            "model_revision",
            "weight_files",
        }
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or value["kind"] != "Qwen38RetainedVisionArtifact"
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 6
            or value["family_profile"] != PROFILE
            or type(value["family_version"]) is not int
            or value["family_version"] != 1
            or (value["model_source"], value["model_revision"]) != (base["repo"], base["revision"])
        ):
            return None

        def digest(value):
            return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

        if not all(digest(value[key]) for key in hashes):
            return None
        metadata = {}
        for name, key, maximum in (
            ("config.json", "config_sha256", 1048576),
            ("tokenizer.json", "tokenizer_sha256", 67108864),
            ("preprocessor_config.json", "processor_sha256", 65536),
            ("chat_template.jinja", "template_sha256", 262144),
            ("model.safetensors.index.json", "index_sha256", 16777216),
        ):
            data = _read(directory / name, maximum)
            if hashlib.sha256(data).hexdigest() != value[key]:
                return None
            metadata[name] = data
        config = json.loads(metadata["config.json"], object_pairs_hook=_unique_object)
        if (
            metadata["preprocessor_config.json"] != PROCESSOR_BYTES
            or metadata["chat_template.jinja"].decode() != config.get("token_codec_config", {}).get("prompt_template")
            or not isinstance(config.get("vision_config"), dict)
        ):
            return None
        records = value["weight_files"]
        if not isinstance(records, dict) or not 1 <= len(records) <= 256:
            return None
        aggregate = hashlib.sha256()
        for name, record in sorted(records.items()):
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in {".", ".."}
                or not (name.endswith(".safetensors") or name == "ngram_table.bin")
                or not isinstance(record, dict)
                or set(record) != {"size", "sha256"}
                or type(record["size"]) is not int
                or record["size"] <= 0
                or not digest(record["sha256"])
            ):
                return None
            with open_regular(directory / name) as stream:
                if stream.seek(0, 2) != record["size"]:
                    return None
            aggregate.update(name.encode() + b"\0" + record["sha256"].encode())
        index = json.loads(metadata["model.safetensors.index.json"], object_pairs_hook=_unique_object)
        if (
            aggregate.hexdigest() != value["artifact_sha256"]
            or set(index.get("weight_map", {}).values()) != {name for name in records if name.endswith(".safetensors")}
            or sum(record["size"] for record in records.values()) != base["weight_bytes"]
        ):
            return None
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None


def _read(path: Path, maximum: int) -> bytes:
    with open_regular(path) as stream:
        if stream.seek(0, 2) > maximum:
            raise ValueError("native image metadata exceeds its limit")
        stream.seek(0)
        data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("native image metadata exceeds its limit")
        return data


def _remote_identity(identity: dict) -> bool:
    """Image schema 6 records immutable remote provenance; local imports keep it."""
    from .sources import huggingface_repository

    try:
        return (
            isinstance(identity.get("repo"), str)
            and huggingface_repository(identity["repo"]) == identity["repo"]
            and isinstance(identity.get("revision"), str)
            and len(identity["revision"]) == 40
            and all(character in "0123456789abcdef" for character in identity["revision"])
        )
    except ValueError:
        return False


def _signature(stream) -> list[int]:
    stat = os.fstat(stream.fileno())
    return [
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns // 1_000_000_000,
        stat.st_mtime_ns % 1_000_000_000,
        stat.st_ctime_ns // 1_000_000_000,
        stat.st_ctime_ns % 1_000_000_000,
    ]


def _verified_weight(path: Path) -> tuple[dict, list[int]]:
    with open_regular(path) as stream:
        before = _signature(stream)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if _signature(stream) != before:
            raise ValueError("native image weight changed during preparation")
        return {"size": before[2], "sha256": digest.hexdigest()}, before


def _save_hash_cache(directory: Path, manifest_sha256: str, records: dict, signatures: dict) -> None:
    """Reuse content validation only while descriptor identities stay unchanged."""
    cache = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "files": {
            name: {"sha256": record["sha256"], "signature": signatures[name]} for name, record in records.items()
        },
    }
    temporary = directory / f".stark-image-validation-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            descriptor = -1
            json.dump(cache, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / HASH_CACHE)
    except OSError:
        # Full native validation remains available if this disposable cache
        # cannot be published; preparing the artifact must still work.
        pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def prepare_image_artifact(directory: Path, choice: str | None = None) -> dict:
    """Bind a complete compatible native folder without granting execution claims."""
    from .hybrid_prepare import SOURCES, prepared_identity

    if choice is not None and choice not in {"qwen38-mlx-serve-mixed", "qwen38-lily-q4-xl"}:
        raise ValueError("native image upgrade requires a supported Qwen checkpoint")
    identity = prepared_identity(directory)
    if identity is None or identity["architecture"] != "qwen4_exp" or not _remote_identity(identity):
        raise ValueError("native image upgrade requires its complete immutable source receipt")
    if choice is not None:
        source = SOURCES[choice]
        if identity["repo"] != source["weights_repo"] or identity["revision"] != source["weights_revision"]:
            raise ValueError("native image upgrade requires its complete pinned source receipt")
    config = json.loads(_read(directory / "config.json", 1024 * 1024))
    if not isinstance(config.get("vision_config"), dict):
        raise ValueError("native checkpoint does not declare a vision tower")
    if choice is None:
        from .qwen38_retained import validate_processing_metadata, validate_vision_layout
        from .streaming import load_checkpoint

        validate_processing_metadata(
            [path.name for path in directory.iterdir()], lambda name, maximum: _read(directory / name, maximum)
        )
        validate_vision_layout(config, load_checkpoint(directory))
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
    with ThreadPoolExecutor(max_workers=4) as workers:
        verified = list(workers.map(_verified_weight, weights))
    records = {path.name: record for path, (record, _) in zip(weights, verified, strict=True)}
    signatures = {path.name: signature for path, (_, signature) in zip(weights, verified, strict=True)}
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
    _save_hash_cache(directory, file_hash(path), records, signatures)
    return result
