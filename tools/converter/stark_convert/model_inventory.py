"""Small, read-only display metadata for models already in the local library."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

from ._build_metadata import open_regular
from .hybrid_prepare import prepared_identity
from .model_support import normalize_architecture
from .streaming import authorize_source_path

MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_FLOATS = {"BF16": "BF16", "F16": "FP16", "F32": "FP32"}


def _text(value, maximum=256):
    try:
        return (
            isinstance(value, str)
            and 0 < len(value.encode()) <= maximum
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)
        )
    except UnicodeError:
        return False


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _json(data):
    return json.loads(
        data, object_pairs_hook=_object, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError())
    )


def _signature(value):
    return tuple(getattr(value, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _small_file(root, name):
    with open_regular(authorize_source_path(root, root / name)) as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= MAX_METADATA_BYTES:
            raise ValueError()
        data = stream.read(before.st_size + 1)
        if len(data) != before.st_size or _signature(before) != _signature(os.fstat(stream.fileno())):
            raise ValueError()
        return data


def _weights(root):
    """Read the header only; display inspection never reads tensor payloads."""
    with open_regular(authorize_source_path(root, root / "model.safetensors")) as stream:
        before = os.fstat(stream.fileno())
        size = before.st_size
        if not 10 <= size <= 2**63 - 1:
            raise ValueError()
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError()
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
            return size, None
        payload = stream.read(length)
        if len(payload) != length or _signature(before) != _signature(os.fstat(stream.fileno())):
            raise ValueError()
    try:
        header = _json(payload)
        if not isinstance(header, dict) or not 1 <= len(header) <= 65536:
            return size, None
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict):
            return size, None
        widths, floating = set(), set()
        specs = 0
        for key, raw in metadata.items():
            if not isinstance(key, str) or not key.endswith(".spec"):
                continue
            if not isinstance(raw, str):
                return size, None
            spec = _json(raw)
            tensor = header.get(key.removesuffix(".spec") + ".weights")
            if not isinstance(spec, dict) or not isinstance(tensor, dict):
                return size, None
            shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
            if (
                not isinstance(shape, list)
                or not 1 <= len(shape) <= 8
                or any(type(value) is not int or not 0 < value <= 2**32 - 1 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] < offsets[1] <= size - 8 - length
            ):
                return size, None
            kind, dtype = spec.get("type"), tensor.get("dtype")
            if kind == "FullPrecisionSpec" and dtype in _FLOATS:
                width = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
                if math.prod(shape) * width != offsets[1] - offsets[0]:
                    return size, None
                floating.add(_FLOATS[dtype])
            elif kind in {"MLXSpec", "IntSpec"} and dtype in {"U8", "I8", "U32", "I32"}:
                bits = spec.get("bits")
                group = spec.get("group_size")
                if (
                    type(bits) is not int
                    or bits not in {2, 3, 4, 5, 6, 8}
                    or type(group) is not int
                    or not 0 < group <= 2**32 - 1
                    or spec.get("layout") not in {"input_output", "output_input"}
                    or math.prod(shape) * (4 if dtype in {"U32", "I32"} else 1) != offsets[1] - offsets[0]
                ):
                    return size, None
                widths.add(bits)
            else:
                return size, None
            specs += 1
        if not specs:
            return size, None
        for name, tensor in header.items():
            if name.endswith(".weights") and name.removesuffix(".weights") + ".spec" not in metadata:
                # Floating convolution/control weights do not change matrix quantization.
                if not isinstance(tensor, dict) or tensor.get("dtype") not in _FLOATS:
                    return size, None
        # Small floating-point control weights are normal in quantized models.
        label = " / ".join(f"{bits}-bit" for bits in sorted(widths)) if widths else " / ".join(sorted(floating))
        return size, label or None
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return size, None


def _source(receipt, model):
    sources = receipt.get("sources")
    files = receipt.get("source_files")
    candidates = {}
    if (
        not isinstance(sources, list)
        or not 1 <= len(sources) <= 64
        or not isinstance(files, list)
        or not 1 <= len(files) <= 4096
    ):
        return None, None
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"repo", "revision"}:
            return None, None
        repo, revision = source["repo"], source["revision"]
        if (
            repo is not None
            and (not _text(repo) or _REPO.fullmatch(repo) is None)
            or revision is not None
            and (not isinstance(revision, str) or _REVISION.fullmatch(revision) is None)
            or repo in candidates
        ):
            return None, None
        candidates[repo] = revision
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"repo", "revision", "size", "sha256"}
            or not isinstance(item["repo"], (str, type(None)))
            or item["repo"] not in candidates
            or candidates[item["repo"]] != item["revision"]
            or type(item["size"]) is not int
            or not 0 <= item["size"] <= 2**63 - 1
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            return None, None
    identity = {"sources": sorted(sources, key=lambda item: item["repo"] or ""), "files": files}
    if "conversion_profile" in receipt:
        profile = receipt["conversion_profile"]
        if not isinstance(profile, dict) or set(profile) != {"architecture", "metadata_profile"}:
            return None, None
        identity["conversion_profile"] = profile
    if "artifact_mode" in receipt:
        if receipt["artifact_mode"] != "vision-language":
            return None, None
        identity["artifact_mode"] = receipt["artifact_mode"]
    if (
        receipt.get("source_manifest_sha256")
        != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ):
        return None, None
    declared = model.get("source", "").removeprefix("huggingface:")
    if model.get("source", "").startswith("huggingface:"):
        if declared not in candidates:
            return None, None
        repo = declared
    elif len(candidates) == 1:
        repo = next(iter(candidates))
    else:
        return None, None
    revision = candidates[repo]
    return repo, revision if isinstance(revision, str) and _REVISION.fullmatch(revision) else None


def model_display(model):
    """Describe saved versions; missing metadata never changes runtime admission."""
    if not _text(model.get("id"), 128) or not _text(model.get("name")):
        return None
    mode = model.get("artifact_mode", "text")
    result = {
        "schema_version": 1,
        "group_id": "installed:" + model["id"],
        "group_name": model["name"],
        "family_id": None,
        "quantization": None,
        "weight_bytes": None,
        "source_revision": None,
        "artifact_mode": mode if isinstance(mode, str) and mode in {"text", "vision-language"} else "text",
    }
    root = Path(os.path.abspath(model["path"]))
    hybrid = prepared_identity(root)
    if hybrid is not None:
        repo = hybrid["repo"]
        result.update(
            group_id="repo:" + repo,
            group_name=repo.split("/")[-1],
            family_id=hybrid["architecture"],
            quantization=hybrid["quantization"],
            weight_bytes=hybrid["weight_bytes"],
            source_revision=hybrid["revision"],
        )
        return result
    try:
        result["weight_bytes"], result["quantization"] = _weights(root)
    except Exception:
        pass
    try:
        config = _small_file(root, "config.json")
        parsed = _json(config)
        if isinstance(parsed, dict) and parsed.get("type") == "Qwen35VisionLanguageModelConfig":
            result["artifact_mode"] = "vision-language"
        elif isinstance(parsed, dict) and parsed.get("type") in {"LanguageModelConfig", "ClassifierModelConfig"}:
            result["artifact_mode"] = "text"
        receipt = _json(_small_file(root, "conversion.json"))
        if (
            not isinstance(parsed, dict)
            or parsed.get("type")
            not in {"LanguageModelConfig", "Qwen35VisionLanguageModelConfig", "ClassifierModelConfig"}
            or not isinstance(receipt, dict)
        ):
            return result
        config_hash = receipt.get("config_sha256")
        if config_hash is None and isinstance(receipt.get("outputs"), dict):
            config_hash = receipt["outputs"].get("config.json")
        if config_hash != hashlib.sha256(config).hexdigest():
            return result
        declared_bytes = receipt.get("output_bytes")
        if declared_bytes is not None and (
            type(declared_bytes) is not int or declared_bytes != result["weight_bytes"]
        ):
            return result
        repo, revision = _source(receipt, model)
        if repo is not None:
            result.update(group_id="repo:" + repo, group_name=repo.split("/")[-1], source_revision=revision)
            architecture = normalize_architecture(receipt.get("architecture"))
            profile = receipt.get("conversion_profile")
            if isinstance(profile, dict):
                if normalize_architecture(profile.get("architecture")) == architecture:
                    result["family_id"] = architecture
            else:
                from .registry import get_model

                entry = get_model(repo)
                if entry is not None and entry.get("adapter") == architecture:
                    result["family_id"] = architecture
    except Exception:
        pass
    return result
