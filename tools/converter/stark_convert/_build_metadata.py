"""Dependency-free source provenance shared by packaging and conversion."""

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import BinaryIO

PROJECT = Path(__file__).resolve().parents[1]
ENGINE_UPSTREAM_REVISION = "8d01581474760bd1fc7f4a547ff1d02f7dada437"
REFERENCE_REVISION = "831accb8252a6b6606c378048caeb44178f1a050"

SOURCE_INPUTS = (
    "pyproject.toml",
    ".python-version",
    "LICENSE.lalamo",
    "setup.py",
    "MANIFEST.in",
    "scripts/export_conversion_catalog.py",
)


class ConversionError(Exception):
    """An invalid checkpoint or failed conversion must not be published."""


def open_regular(path: Path) -> BinaryIO:
    """Open a canonical regular file without blocking on FIFOs or following races."""
    resolved = Path(os.path.abspath(path))
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in resolved.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(resolved.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ConversionError("Source must be a regular file")
    return os.fdopen(descriptor, "rb", buffering=0)


def _signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


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
