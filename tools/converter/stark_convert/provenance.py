"""Comprehensive source and build provenance; no model content in diagnostics."""

import hashlib
import json
import os
from pathlib import Path

from ._build_metadata import SOURCE_INPUTS
from .streaming import ConversionError, _signature, open_regular

PROJECT = Path(__file__).resolve().parents[1]
ENGINE_UPSTREAM_REVISION = "8d01581474760bd1fc7f4a547ff1d02f7dada437"
REFERENCE_REVISION = "831accb8252a6b6606c378048caeb44178f1a050"


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
    root = project or PROJECT
    package = root / "stark_convert"
    capsule = package / "_build_inputs"
    inputs = capsule if capsule.is_dir() else root
    if capsule.exists():
        if (root / "pyproject.toml").exists():
            raise ValueError("ambiguous_packaged_provenance")
        expected = {*SOURCE_INPUTS, "uv.lock"}
        actual = {
            path.relative_to(capsule).as_posix()
            for path in capsule.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        if actual != expected:
            raise ValueError("invalid_packaged_provenance")
    manifest = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc" or path.is_relative_to(capsule):
            continue
        # Include source, catalogs, profiles, schemas and packaged resources.
        if path.suffix not in {".py", ".json", ".toml", ".txt", ".md"}:
            raise ValueError("unaccounted_package_resource")
        manifest[path.relative_to(root).as_posix()] = file_hash(path)
    for name in SOURCE_INPUTS:
        path = inputs / name
        if not path.is_file():
            raise ValueError("missing_provenance_input")
        manifest[name] = file_hash(path)
    return {
        "converter_sources_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "converter_lock_sha256": file_hash(inputs / "uv.lock"),
        "reference_source_revision": REFERENCE_REVISION,
        "engine_upstream_revision": ENGINE_UPSTREAM_REVISION,
    }


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
