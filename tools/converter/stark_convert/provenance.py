"""Comprehensive source and build provenance; no model content in diagnostics."""

import hashlib
import json
from pathlib import Path

from . import _build_metadata
from ._build_metadata import ENGINE_UPSTREAM_REVISION as ENGINE_UPSTREAM_REVISION
from ._build_metadata import REFERENCE_REVISION as REFERENCE_REVISION
from ._build_metadata import SOURCE_INPUTS as SOURCE_INPUTS
from ._build_metadata import ConversionError as ConversionError
from ._build_metadata import _signature as _signature
from ._build_metadata import file_hash as file_hash
from ._build_metadata import open_regular as open_regular

PROJECT = Path(__file__).resolve().parents[1]


def toolchain_provenance(project: Path | None = None) -> dict:
    return _build_metadata.toolchain_provenance(project or PROJECT)


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
