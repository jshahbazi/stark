"""Dependency-free source inputs shared by packaging and runtime provenance."""

import hashlib
import json

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


def toolchain_metadata(root, file_hash):
    """Shared manifest construction; callers supply their source-file hash policy."""
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
