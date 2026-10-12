"""Pin and hash exactly the checkpoint files selected by a catalog profile.

Private manifests contain local cache paths for offline replay. Public receipt
fragments contain only repository revisions, file sizes, and content hashes.
FileSpec interpretation follows Lalamo 831accb8, MIT, Copyright (c) 2025 Mirai
Tech Inc.; this module does not import Lalamo or execute checkpoint code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import get_model, required_files
from .streaming import ConversionError, UnsupportedConversion, read_json

_REVISION = re.compile(r"[0-9a-f]{40}")
_HASH = re.compile(r"[0-9a-f]{64}")


def _relative(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ConversionError("Invalid source filename")
    result = Path(name)
    if result.is_absolute() or ".." in result.parts or result.as_posix() != name:
        raise ConversionError("Invalid source filename")
    return result


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, repo: str | None, revision: str | None, filename: str, role: str) -> dict:
    before = _signature(path)
    digest = _hash(path)
    if before != _signature(path):
        raise ConversionError("Source changed while hashing")
    return {
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "role": role,
        "path": str(path.absolute()),
        "size": before[2],
        "sha256": digest,
        "signature": list(before),
    }


def _weights(names: tuple[str, ...], container: str, load: Any) -> tuple[tuple[str, ...], str | None]:
    """Use the declared container and checkpoint index, never mix alternatives."""
    if container == ".safetensors":
        index_name, extensions = "model.safetensors.index.json", (".safetensors",)
    elif container == ".pth":
        index_name, extensions = "pytorch_model.bin.index.json", (".bin", ".pth", ".pt")
    elif container == ".nemo":
        index_name, extensions = None, (".nemo",)
    else:
        raise UnsupportedConversion("Unsupported catalog checkpoint container")
    if index_name in names:
        index = read_json(load(index_name))
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(key, str) for key in mapping):
            raise ConversionError("Invalid checkpoint index")
        if any(not isinstance(name, str) for name in mapping.values()):
            raise ConversionError("Invalid checkpoint index")
        selected = tuple(sorted(set(mapping.values())))
        if any(_relative(name).suffix not in extensions or name not in names for name in selected):
            raise ConversionError("Checkpoint index references an unavailable shard")
        return selected, index_name
    canonical = {".safetensors": "model.safetensors", ".pth": "pytorch_model.bin"}.get(container)
    if canonical in names:
        return (canonical,), None
    for extension in extensions:
        selected = tuple(sorted(name for name in names if name.endswith(extension) and len(Path(name).parts) == 1))
        if selected:
            for name in selected:
                _relative(name)
            if len(selected) != 1:
                raise UnsupportedConversion("Multiple checkpoint files require an explicit shard index")
            return selected, None
    raise FileNotFoundError("missing_checkpoint_weights")


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    repo: str | None
    revision: str | None
    manifest: dict

    @property
    def checkpoint(self) -> Path:
        return self.root

    def resolve_file(self, filename: str, repo: str | None = None) -> Path:
        _relative(filename)
        effective_repo = repo if repo is not None else self.repo
        for record in self.manifest["files"]:
            if record["repo"] == effective_repo and record["filename"] == filename:
                return Path(record["path"])
        raise FileNotFoundError("source_file_not_in_manifest")

    def assert_unchanged(self, *, verify_content: bool = True) -> None:
        """Check bindings and signatures, with full content verification by default.

        Signature-only mode is for a repeated planning check after the initial
        source hashes have been computed. Offline replay and publication must
        retain the default full check, including restored-signature mutations.
        """
        if type(verify_content) is not bool:
            raise ValueError("invalid_content_verification_mode")
        entry = get_model(self.repo) if self.repo else None
        names = tuple(
            sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        )
        selected, index_name = _weights(
            names, entry["container"] if entry else ".safetensors", lambda filename: self.root / _relative(filename)
        )
        expected_weights = tuple(
            sorted(
                record["filename"]
                for record in self.manifest["files"]
                if record["repo"] == self.repo and record["role"] == "weights"
            )
        )
        expected_indices = tuple(
            record["filename"]
            for record in self.manifest["files"]
            if record["repo"] == self.repo and record["role"] == "weight_index"
        )
        if selected != expected_weights or expected_indices != ((index_name,) if index_name else ()):
            raise ConversionError("Source checkpoint selection changed during conversion")
        config_name = entry["files"]["model_config"]["filename"] if entry else "config.json"
        for filename in (*selected, *((index_name,) if index_name else ()), config_name):
            if self.resolve_file(filename).resolve() != (self.root / filename).resolve():
                raise ConversionError("Source file does not match checkpoint root")
        for record in self.manifest["files"]:
            path = Path(record["path"])
            try:
                if (
                    tuple(record["signature"]) != _signature(path)
                    or (verify_content and record["sha256"] != _hash(path))
                    or tuple(record["signature"]) != _signature(path)
                ):
                    raise ConversionError("Source changed during conversion")
            except OSError:
                raise ConversionError("Source unavailable during conversion") from None

    def receipt(self) -> dict:
        """Exclude private paths, roles, and filenames from public evidence."""
        sources = sorted(self.manifest["sources"], key=lambda record: record["repo"] or "")
        files = [
            {
                "repo": record["repo"],
                "revision": record["revision"],
                "size": record["size"],
                "sha256": record["sha256"],
            }
            for record in self.manifest["files"]
        ]
        return {
            "sources": sources,
            "source_files": files,
            "source_manifest_sha256": hashlib.sha256(
                json.dumps({"sources": sources, "files": files}, sort_keys=True).encode()
            ).hexdigest(),
        }

    def write_manifest(self, path: Path) -> None:
        """Create a private replay manifest atomically without overwriting."""
        path = Path(path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".stark-source-", dir=path.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(self.manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _replay(path: Path, *, repo: str | None, revision: str | None) -> SourceSnapshot:
    manifest = read_json(path)
    try:
        if manifest["schema_version"] != 1 or not isinstance(manifest["root"], str):
            raise ValueError()
        primary = manifest["primary_repo"]
        if repo is not None and repo != primary or revision is not None and revision != manifest["primary_revision"]:
            raise ValueError()
        root = Path(manifest["root"])
        if not root.is_absolute() or not root.is_dir() or not isinstance(manifest["files"], list):
            raise ValueError()
        known = set()
        revisions = {}
        for source in manifest["sources"]:
            source_repo, source_revision = source["repo"], source["revision"]
            if source_repo in revisions or source_repo is not None and not isinstance(source_repo, str):
                raise ValueError()
            if source_revision is not None and (
                not isinstance(source_revision, str) or not _REVISION.fullmatch(source_revision)
            ):
                raise ValueError()
            if source_repo != primary and source_revision is None:
                raise ValueError()
            revisions[source_repo] = source_revision
        if primary not in revisions or revisions[primary] != manifest["primary_revision"]:
            raise ValueError()
        for record in manifest["files"]:
            key = record["repo"], record["filename"]
            if key in known or record["repo"] not in revisions or record["revision"] != revisions[record["repo"]]:
                raise ValueError()
            known.add(key)
            _relative(record["filename"])
            if (
                not Path(record["path"]).is_absolute()
                or not isinstance(record["sha256"], str)
                or not _HASH.fullmatch(record["sha256"])
                or type(record["size"]) is not int
                or record["size"] < 0
                or len(record["signature"]) != 4
                or record["size"] != record["signature"][2]
                or any(type(number) is not int for number in record["signature"])
            ):
                raise ValueError()
        entry = get_model(primary) if primary else None
        rules = (
            required_files(entry)
            if entry is not None
            else (
                {"repo": primary, "filename": name}
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
            )
        )
        if any((rule["repo"], rule["filename"]) not in known for rule in rules):
            raise ValueError()
        snapshot = SourceSnapshot(root, primary, manifest["primary_revision"], manifest)
        names = tuple(record["filename"] for record in manifest["files"] if record["repo"] == primary)
        selected, index_name = _weights(names, entry["container"] if entry else ".safetensors", snapshot.resolve_file)
        if any((primary, name) not in known for name in selected):
            raise ValueError()
        # Weights consumed by the planner must be the pinned files under root.
        for name in (*selected, *((index_name,) if index_name else ())):
            if snapshot.resolve_file(name).resolve() != (root / name).resolve():
                raise ValueError()
        for record in manifest["files"]:
            if record["repo"] == primary and record["role"] == "model_config":
                if Path(record["path"]).resolve() != (root / record["filename"]).resolve():
                    raise ValueError()
        snapshot.assert_unchanged()
        return snapshot
    except (KeyError, TypeError, ValueError, OSError):
        raise ConversionError("Invalid or incomplete offline source manifest") from None


def resolve_source(
    value: str | Path,
    *,
    repo: str | None = None,
    revision: str | None = None,
    source_manifest: Path | None = None,
    offline: bool = False,
    hub: Any = None,
) -> SourceSnapshot:
    """Resolve all repositories before any file downloads or conversion starts."""
    if source_manifest is not None:
        return _replay(Path(source_manifest), repo=repo, revision=revision)
    if revision is not None and not isinstance(revision, str):
        raise ConversionError("Invalid source revision")
    path = Path(value).expanduser()
    local = path.is_dir()
    if not local and (path.exists() or str(value).startswith(("/", ".", "~"))):
        raise ValueError("invalid_source")
    if not local:
        if repo is not None and repo != str(value):
            raise ValueError("conflicting_repository")
        repo = str(value)
    entry = get_model(repo) if repo else None
    if repo is not None and entry is None:
        raise UnsupportedConversion("Repository is outside the pinned conversion catalog")
    if not local and offline:
        raise ConversionError("Offline resolution requires a source manifest")
    rules = (
        required_files(entry)
        if entry is not None
        else tuple(
            {"role": role, "repo": repo, "filename": filename}
            for role, filename in (
                ("model_config", "config.json"),
                ("tokenizer", "tokenizer.json"),
                ("tokenizer_config", "tokenizer_config.json"),
                ("generation_config", "generation_config.json"),
            )
        )
    )
    repositories = {repo, *(rule["repo"] for rule in rules)}
    if local and revision is not None and not _REVISION.fullmatch(revision):
        raise ConversionError("Local source revision must be immutable")
    if offline and repositories != {repo}:
        raise ConversionError("External metadata requires a source manifest offline")
    if not local or repositories != {repo}:
        if hub is None:
            import huggingface_hub

            hub = huggingface_hub
        revisions = {repo: revision} if local else {}
        for repository in sorted(repositories - ({repo} if local else set())):
            resolved = (
                hub.HfApi().model_info(repo_id=repository, revision=revision if repository == repo else None).sha
            )
            if not isinstance(resolved, str) or not _REVISION.fullmatch(resolved):
                raise ConversionError("Repository revision is not immutable")
            revisions[repository] = resolved
    else:
        revisions = {repo: revision}
    if local:
        root = path.resolve()
        names = tuple(sorted(file.relative_to(root).as_posix() for file in root.rglob("*") if file.is_file()))

        def loader(filename: str) -> Path:
            return root / _relative(filename)
    else:
        names = tuple(sorted(hub.list_repo_files(repo_id=repo, revision=revisions[repo])))

        def loader(filename: str) -> Path:
            _relative(filename)
            return Path(
                hub.hf_hub_download(repo_id=repo, filename=filename, revision=revisions[repo], local_files_only=False)
            )

        config_rule = next(rule for rule in rules if rule["role"] == "model_config")
        config_path = loader(config_rule["filename"])
        # snapshot files preserve repository-relative placement in the hub cache.
        root = config_path
        for _ in _relative(config_rule["filename"]).parts:
            root = root.parent
    selected, index_name = _weights(names, entry["container"] if entry else ".safetensors", loader)
    descriptors = list(rules)
    if index_name:
        descriptors.append({"role": "weight_index", "repo": repo, "filename": index_name})
    descriptors.extend({"role": "weights", "repo": repo, "filename": name} for name in selected)
    records, known = [], set()
    for descriptor in descriptors:
        effective_repo, filename = descriptor["repo"], descriptor["filename"]
        _relative(filename)
        if (effective_repo, filename) in known:
            continue
        known.add((effective_repo, filename))
        if effective_repo == repo:
            resolved_path = loader(filename)
        else:
            resolved_path = Path(
                hub.hf_hub_download(
                    repo_id=effective_repo,
                    filename=filename,
                    revision=revisions[effective_repo],
                    local_files_only=False,
                )
            )
        records.append(
            _record(
                resolved_path,
                repo=effective_repo,
                revision=revisions[effective_repo],
                filename=filename,
                role=descriptor["role"],
            )
        )
    manifest = {
        "schema_version": 1,
        "root": str(root.absolute()),
        "primary_repo": repo,
        "primary_revision": revisions[repo],
        "sources": [
            {"repo": repository, "revision": revisions[repository]}
            for repository in sorted(revisions, key=lambda item: item or "")
        ],
        "files": records,
    }
    return SourceSnapshot(root, repo, revisions[repo], manifest)
