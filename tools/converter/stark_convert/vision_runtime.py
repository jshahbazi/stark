"""Exact native image execution qualifications, separate from artifact conversion."""

from __future__ import annotations

import json
import platform
import re
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from ._build_metadata import ConversionError, file_hash
from .vision_contract import REPO, REVISION

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_OS_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))?\Z")
_ARTIFACT = "7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4"
FAMILY_PROFILE = "qwen35-dense-bf16-merged-v1"
_ARTIFACT_HASHES = {
    "artifact_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "template_sha256",
}
_RUNTIME_KEYS = {"backend", "device_name", "apple_gpu_family", "gpu_core_count", "supports_mxu"}
_RECORD_KEYS = _ARTIFACT_HASHES | {
    "repo",
    "revision",
    "worker_sha256",
    "transport",
    "runtime_identity",
    "os_version",
    "passed",
    "evidence_sha256",
}


def _valid_runtime_identity(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _RUNTIME_KEYS
        and value.get("backend") == "metal"
        and isinstance(value.get("device_name"), str)
        and 0 < len(value["device_name"]) <= 128
        and value["device_name"].strip() == value["device_name"]
        and all(ord(char) >= 32 and ord(char) != 127 for char in value["device_name"])
        and type(value.get("apple_gpu_family")) is int
        and 1001 <= value["apple_gpu_family"] <= 1010
        and type(value.get("gpu_core_count")) is int
        and 1 <= value["gpu_core_count"] <= 512
        and type(value.get("supports_mxu")) is bool
    )


def _valid_os_version(value: object) -> bool:
    return isinstance(value, str) and len(value) <= 32 and _OS_VERSION.fullmatch(value) is not None


def _current_os_version() -> str | None:
    try:
        value = platform.mac_ver()[0]
    except Exception:
        return None
    return value if _valid_os_version(value) else None


def image_input_contract(value: object) -> dict | None:
    """Recognize functionality without interpreting it as qualification evidence."""
    if not isinstance(value, dict):
        return None
    operations, formats = value.get("operations"), value.get("formats")
    if (
        value.get("version") != "mediaref-v1"
        or value.get("transport") != "owner-private-file-v1"
        or not _valid_runtime_identity(value.get("runtime_identity"))
        or type(value.get("max_images")) is not int
        or value["max_images"] != 1
        or type(value.get("artifact_schema_version")) is not int
        or value["artifact_schema_version"] not in (2, 3)
        or any(not isinstance(value.get(key), str) or _HASH.fullmatch(value[key]) is None for key in _ARTIFACT_HASHES)
        or not isinstance(operations, list)
        or any(not isinstance(item, str) for item in operations)
        or len(operations) != 3
        or set(operations) != {"prepare_image", "generate_image", "discard_image"}
        or not isinstance(formats, list)
        or any(not isinstance(item, str) for item in formats)
        or len(formats) != 2
        or set(formats) != {"image/jpeg", "image/png"}
        or any(
            type(value.get(key)) is not int or value[key] < 1
            for key in (
                "max_compressed_bytes",
                "max_source_pixels",
                "max_processed_pixels",
                "max_patches",
                "max_image_tokens",
            )
        )
    ):
        return None
    if value["artifact_schema_version"] == 2:
        if (
            value.get("model_source") != REPO
            or value.get("model_revision") != REVISION
            or value.get("artifact_sha256") != _ARTIFACT
        ):
            return None
    elif not family_image_input_contract(value):
        return None
    return dict(value)


def family_image_input_contract(value: object) -> bool:
    """Recognize the native implementation profile, never execution evidence."""
    if not isinstance(value, dict):
        return False
    source, revision = value.get("model_source"), value.get("model_revision")
    return (
        value.get("family_profile") == FAMILY_PROFILE
        and type(value.get("family_version")) is int
        and value["family_version"] == 1
        and isinstance(source, str)
        and 0 < len(source) <= 256
        and not source.startswith(("/", "."))
        and ".." not in source.split("/")
        and all(char.isascii() and (char.isalnum() or char in "_./-") for char in source)
        and (revision is None or isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40,64}", revision) is not None)
        and _valid_runtime_identity(value.get("runtime_identity"))
        and all(
            type(value.get(key)) is int and 0 < value[key] <= maximum
            for key, maximum in (
                ("max_compressed_bytes", 20 * 1024 * 1024),
                ("max_source_pixels", 32_000_000),
                ("max_processed_pixels", 1_048_576),
                ("max_patches", 4096),
                ("max_image_tokens", 1024),
            )
        )
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _records() -> list[dict]:
    try:
        document = json.loads(
            files("stark_convert").joinpath("vision_runtime_qualifications.json").read_text(),
            object_pairs_hook=_unique_object,
        )
        if (
            not isinstance(document, dict)
            or set(document) != {"schema_version", "records"}
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 2
            or not isinstance(document["records"], list)
        ):
            return []
        identities = set()
        for record in document["records"]:
            if (
                not isinstance(record, dict)
                or set(record) != _RECORD_KEYS
                or record.get("repo") != REPO
                or record.get("revision") != REVISION
                or record.get("artifact_sha256") != _ARTIFACT
                or record.get("transport") != "owner-private-file-v1"
                or not _valid_runtime_identity(record.get("runtime_identity"))
                or not _valid_os_version(record.get("os_version"))
                or record.get("passed") is not True
                or any(
                    not isinstance(record.get(key), str) or _HASH.fullmatch(record[key]) is None
                    for key in _ARTIFACT_HASHES | {"worker_sha256", "evidence_sha256"}
                )
            ):
                return []
            identity = (
                *(record[key] for key in sorted(_ARTIFACT_HASHES)),
                record["worker_sha256"],
                json.dumps(record["runtime_identity"], sort_keys=True, separators=(",", ":")),
                record["os_version"],
            )
            if identity in identities:
                return []
            identities.add(identity)
        return document["records"]
    except (OSError, ValueError, TypeError, ConversionError):
        return []


def qualified_image_worker(
    contract: dict | None,
    executable: Path,
    expected_worker_sha256: str | None = None,
) -> bool:
    """Match all five admitted files and the executable to completed image evidence."""
    if image_input_contract(contract) is None or contract.get("artifact_schema_version") != 2:
        return False
    try:
        worker_hash = file_hash(Path(executable))
    except (OSError, ValueError, TypeError, ConversionError):
        return False
    if expected_worker_sha256 is not None and worker_hash != expected_worker_sha256:
        return False
    os_version = _current_os_version()
    if os_version is None:
        return False
    return any(
        record["worker_sha256"] == worker_hash
        and all(record[key] == contract[key] for key in _ARTIFACT_HASHES)
        and record["runtime_identity"] == contract["runtime_identity"]
        and record["os_version"] == os_version
        for record in _records()
    )


def has_recorded_image_runtime() -> bool:
    """Whether any named native build has completed this exact image qualification."""
    return bool(_records())


def image_qualification_for(repo: str, dtype: str, hashes: dict) -> dict | None:
    """Return named-build evidence without inferring the currently installed worker."""
    if repo != REPO or dtype != "bfloat16":
        return None
    for record in _records():
        if all(record[key] == hashes.get(key) for key in _ARTIFACT_HASHES):
            return deepcopy(record)
    return None
