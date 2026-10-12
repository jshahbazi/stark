"""Match exact artifacts to explicitly recorded native loader qualifications.

A record proves loading for the named build only. It never implies execution,
qualification of another engine binary, or family-wide native compatibility.
The packaged document contains public identifiers and aggregate hashes only.
"""

from __future__ import annotations

import copy
import json
import re
from importlib.resources import files

from .registry import get_model
from .streaming import ConversionError

_HASH = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_HASHES = frozenset({"source_manifest_sha256", "output_sha256", "config_sha256", "tokenizer_sha256"})
_RECORD_KEYS = _ARTIFACT_HASHES | {"repo", "dtype", "native_build"}
_BUILD_HASHES = frozenset(
    {"worker_sha256", "native_source_manifest_sha256", "cargo_lock_sha256", "rust_toolchain_sha256"}
)
_BUILD_KEYS = _BUILD_HASHES | {"passed", "native_score_calls", "rust", "profile", "features"}
_CLASSIFIER_HASHES = frozenset({"harness_source_sha256", "harness_manifest_sha256", "harness_lock_sha256"})
_CLASSIFIER_BUILD_KEYS = (
    _BUILD_KEYS | _CLASSIFIER_HASHES | {"loader", "native_classification_calls", "native_generation_calls"}
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _load_qualifications() -> list[dict]:
    """Reject malformed or ambiguous evidence rather than weakening matching."""
    try:
        document = json.loads(
            files("stark_convert").joinpath("native_qualifications.json").read_text(), object_pairs_hook=_unique_object
        )
        if (
            not isinstance(document, dict)
            or set(document) != {"schema_version", "records"}
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 1
            or not isinstance(document["records"], list)
        ):
            raise ValueError()
        identities = set()
        for record in document["records"]:
            entry = get_model(record.get("repo")) if isinstance(record, dict) else None
            classifier = bool(entry and entry["model_kind"] == "ClassifierModelSpec")
            if (
                not isinstance(record, dict)
                or set(record) != _RECORD_KEYS | ({"model_kind"} if classifier else set())
                or not isinstance(record["repo"], str)
                or entry is None
                or classifier
                and record["model_kind"] != "classifier"
                or record["dtype"] != "bfloat16"
                or any(not _valid_hash(record[key]) for key in _ARTIFACT_HASHES)
            ):
                raise ValueError()
            build = record["native_build"]
            if (
                not isinstance(build, dict)
                or set(build) != (_CLASSIFIER_BUILD_KEYS if classifier else _BUILD_KEYS)
                or build["passed"] is not True
                or type(build["native_score_calls"]) is not int
                or build["native_score_calls"] != 0
                or build["rust"] != "1.98.1"
                or build["profile"] != "release"
                or not isinstance(build["features"], list)
                or len(build["features"]) != 2
                or set(build["features"]) != {"default", "optimized"}
                or any(not _valid_hash(build[key]) for key in _BUILD_HASHES)
            ):
                raise ValueError()
            if classifier and (
                build["loader"] != "classifier_load_harness"
                or any(
                    type(build[key]) is not int or build[key] != 0
                    for key in ("native_classification_calls", "native_generation_calls")
                )
                or any(not _valid_hash(build[key]) for key in _CLASSIFIER_HASHES)
            ):
                raise ValueError()
            identity = (record["repo"], record["dtype"], *(record[key] for key in sorted(_ARTIFACT_HASHES)))
            if identity in identities:
                raise ValueError()
            identities.add(identity)
        return document["records"]
    except (OSError, KeyError, TypeError, ValueError, UnicodeError):
        raise ConversionError("Invalid native qualification catalog") from None


def qualification_for(repo: str, dtype: str, hashes: dict) -> dict | None:
    """Return copied evidence only for an exact source/output/config/tokenizer match.

    No current-engine identity is inferred. Consumers must retain the named
    native_build record when presenting the qualification and keep execution
    status separate. Reading evidence never opens a model or invokes a worker.
    """
    records = _load_qualifications()
    if (
        not isinstance(repo, str)
        or dtype != "bfloat16"
        or not isinstance(hashes, dict)
        or any(not _valid_hash(hashes.get(key)) for key in _ARTIFACT_HASHES)
    ):
        return None
    for record in records:
        if (
            record["repo"] == repo
            and record["dtype"] == dtype
            and all(record[key] == hashes[key] for key in _ARTIFACT_HASHES)
        ):
            return copy.deepcopy(record)
    return None
