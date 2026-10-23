"""Comprehensive source and build provenance; no model content in diagnostics."""

import hashlib
import json
import os
from pathlib import Path

from ._build_metadata import ENGINE_UPSTREAM_REVISION as ENGINE_UPSTREAM_REVISION
from ._build_metadata import REFERENCE_REVISION as REFERENCE_REVISION
from ._build_metadata import SOURCE_INPUTS as SOURCE_INPUTS
from ._build_metadata import toolchain_metadata
from .streaming import ConversionError, _signature, open_regular

PROJECT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open_regular(path.resolve(strict=True)) as handle:
        before = _signature(os.fstat(handle.fileno()))
        remaining = before[2]
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ConversionError("Source changed while hashing")
            remaining -= len(chunk)
            digest.update(chunk)
        if handle.read(1) or _signature(os.fstat(handle.fileno())) != before:
            raise ConversionError("Source changed while hashing")
    return digest.hexdigest()


def toolchain_provenance(project: Path | None = None) -> dict:
    return toolchain_metadata(project or PROJECT, file_hash)


def native_build_provenance(binary: Path) -> dict:
    """Bind loader checks to the executable, source tree and build inputs."""
    repository = PROJECT.parent.parent
    files = []
    for folder in ("crates", ".cargo", "vendor"):
        files.extend(
            path
            for path in (repository / folder).rglob("*")
            if path.is_file() and not {".git", "__pycache__", "target", ".venv"}.intersection(path.parts)
        )
    files.extend(repository / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"))
    manifest = {path.relative_to(repository).as_posix(): file_hash(path) for path in sorted(files)}
    return {
        "worker_sha256": file_hash(binary),
        "native_source_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "cargo_lock_sha256": file_hash(repository / "Cargo.lock"),
        "rust_toolchain_sha256": file_hash(repository / "rust-toolchain.toml"),
    }
