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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .profiles import GENERIC_OPTIONAL, GENERIC_PROFILE, GENERIC_REQUIRED, generic_files
from .progress import DownloadProgress, ProgressCallback, ProgressReporter, reporter
from .registry import architecture_for_config, get_model
from .registry import required_files as required_files
from .streaming import ConversionError, UnsupportedConversion, authorize_source_path, open_regular, read_json
from .vision_contract import REPO as VISION_REPO
from .vision_contract import REVISION as VISION_REVISION
from .vision_contract import source_rules
from .vision_family import family_source_files, read_family_config

_REVISION = re.compile(r"[0-9a-f]{40}")
_HASH = re.compile(r"[0-9a-f]{64}")
_PARALLEL_HASH_MIN_BYTES = 8 * 1024 * 1024
_HASH_WORKERS = 4


def huggingface_repository(value: str) -> str:
    """Normalize a model repository ID or HTTPS repository page URL."""
    value = value.strip()
    if any(character.isspace() for character in value):
        raise ValueError("invalid_huggingface_repository")
    if "://" in value:
        url = urlsplit(value)
        if url.scheme != "https" or url.netloc.lower() != "huggingface.co":
            raise ValueError("invalid_huggingface_repository")
        value = url.path.removeprefix("/").removesuffix("/")
    if not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)?[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid_huggingface_repository")
    return value


def _relative(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ConversionError("Invalid source filename")
    result = Path(name)
    if result.is_absolute() or ".." in result.parts or result.as_posix() != name:
        raise ConversionError("Invalid source filename")
    return result


def _signature(path: Path) -> tuple[int, int, int, int]:
    with open_regular(path) as handle:
        stat = os.fstat(handle.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _hash(path: Path) -> str:
    # A finite reader uses one 256 KiB buffer and checks source growth.
    with open_regular(path) as handle:
        before = os.fstat(handle.fileno())
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = handle.read(min(256 * 1024, remaining))
            if not chunk:
                raise ConversionError("Source changed while hashing")
            remaining -= len(chunk)
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        if handle.read(1) or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ConversionError("Source changed while hashing")
        return digest.hexdigest()


def _hash_all(paths: list[Path], signatures: list[tuple[int, int, int, int]]) -> list[str]:
    """Hash independent large files with at most 1 MiB of hashing buffers.

    Small metadata files stay on the calling thread. A single large checkpoint
    avoids executor startup. Every submitted job is joined, including failures
    and interruption, so no hash reader survives its verification stage.
    """
    large = [index for index, signature in enumerate(signatures) if signature[2] >= _PARALLEL_HASH_MIN_BYTES]
    if len(large) < 2:
        return [_hash(path) for path in paths]
    # Finish metadata first so the calling thread never adds a fifth digest
    # buffer while four large-file workers are active.
    large_set = set(large)
    hashes = {index: _hash(path) for index, path in enumerate(paths) if index not in large_set}
    pool = ThreadPoolExecutor(max_workers=min(_HASH_WORKERS, len(large)), thread_name_prefix="stark-source-hash")
    try:
        pending = {index: pool.submit(_hash, paths[index]) for index in large}
        for index, future in pending.items():
            hashes[index] = future.result()
        return [hashes[index] for index in range(len(paths))]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _verified_hashes(
    paths: list[Path], *, expected_signatures: list[tuple[int, int, int, int]] | None = None
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Check every file's signature before and after the complete hash batch."""
    message = "Source changed while hashing" if expected_signatures is None else "Source changed during conversion"
    try:
        before = [_signature(path) for path in paths]
        if expected_signatures is not None and before != expected_signatures:
            raise ConversionError(message)
        hashes = _hash_all(paths, before)
        if before != [_signature(path) for path in paths]:
            raise ConversionError(message)
        return before, hashes
    except OSError:
        unavailable = (
            "Source unavailable while hashing"
            if expected_signatures is None
            else "Source unavailable during conversion"
        )
        raise ConversionError(unavailable) from None


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
    def artifact_mode(self) -> str:
        return self.manifest.get("artifact_mode", "text")

    @property
    def conversion_profile(self) -> dict | None:
        return self.manifest.get("conversion_profile")

    @property
    def checkpoint(self) -> Path:
        return self.root

    @property
    def config_path(self) -> Path:
        """Return the declared model config, including external source rules."""
        entry = get_model(self.repo) if self.repo and self.conversion_profile is None else None
        rule = entry["files"]["model_config"] if entry else {"repo": self.repo, "filename": "config.json"}
        return self.resolve_file(rule["filename"], repo=rule["repo"])

    def resolve_file(self, filename: str, repo: str | None = None) -> Path:
        _relative(filename)
        effective_repo = repo if repo is not None else self.repo
        for record in self.manifest["files"]:
            if record["repo"] == effective_repo and record["filename"] == filename:
                path = Path(record["path"])
                if effective_repo == self.repo:
                    return authorize_source_path(self.root, path)
                return path.resolve(strict=True)
        raise FileNotFoundError("source_file_not_in_manifest")

    def _assert_selection(self) -> None:
        if self.artifact_mode not in {"text", "vision-language"}:
            raise ConversionError("Unknown source artifact mode")
        entry = get_model(self.repo) if self.repo and self.conversion_profile is None else None
        names = tuple(
            sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        )
        profile = self.conversion_profile
        if profile is not None:
            if (
                entry is not None
                or not isinstance(profile, dict)
                or set(profile) != {"architecture", "metadata_profile"}
                or profile["metadata_profile"] != GENERIC_PROFILE
                or architecture_for_config(
                    read_family_config(self.resolve_file("config.json"))
                    if self.artifact_mode == "vision-language"
                    else read_json(self.root / "config.json")
                )
                != profile["architecture"]
            ):
                raise ConversionError("Generic source profile disagrees with its checkpoint")
            rules = (
                family_source_files(self.repo, names)
                if self.artifact_mode == "vision-language"
                else generic_files(self.repo, names)
            )
            roles = set(GENERIC_REQUIRED) | set(GENERIC_OPTIONAL)
            if self.artifact_mode == "vision-language":
                roles.add("processor_config")
            actual = {
                (record["role"], record["filename"]) for record in self.manifest["files"] if record["role"] in roles
            }
            if actual != {(rule["role"], rule["filename"]) for rule in rules} or any(
                record["repo"] != self.repo or record["role"] not in roles | {"weights", "weight_index"}
                for record in self.manifest["files"]
            ):
                raise ConversionError("Generic metadata selection changed during conversion")
            for rule in rules:
                if self.resolve_file(rule["filename"]).resolve() != (self.root / rule["filename"]).resolve():
                    raise ConversionError("Generic metadata is outside its checkpoint source")
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
        for filename in (*selected, *((index_name,) if index_name else ())):
            if self.resolve_file(filename).resolve() != (self.root / filename).resolve():
                raise ConversionError("Source file does not match checkpoint root")
        if entry is not None:
            # Sidecars can share a physical file (e.g. a JSON-field template).
            # Resolution keeps the first role in the deterministic rule order.
            descriptors = [*source_rules(entry, self.artifact_mode)]
            if index_name:
                descriptors.append({"repo": self.repo, "filename": index_name, "role": "weight_index"})
            descriptors.extend({"repo": self.repo, "filename": name, "role": "weights"} for name in selected)
            expected = {}
            for descriptor in descriptors:
                expected.setdefault((descriptor["repo"], descriptor["filename"]), descriptor["role"])
            actual = {(record["repo"], record["filename"]): record["role"] for record in self.manifest["files"]}
            if len(actual) != len(self.manifest["files"]) or actual != expected:
                raise ConversionError("Catalog source descriptors disagree with the declared profile")
            config_rule = entry["files"]["model_config"]
        else:
            config_rule = {"repo": self.repo, "filename": "config.json"}
        if (
            config_rule["repo"] == self.repo
            and self.config_path.resolve() != (self.root / config_rule["filename"]).resolve()
        ):
            raise ConversionError("Source config does not match checkpoint root")
        if config_rule["repo"] != self.repo:
            # Preserve the manifest's lexical route even though resolve_file
            # returns a canonical target for safe reads. A primary-tree alias
            # must not become an authorized external-repository descriptor.
            declared_path = next(
                Path(record["path"])
                for record in self.manifest["files"]
                if record["repo"] == config_rule["repo"] and record["filename"] == config_rule["filename"]
            )
            if Path(os.path.abspath(declared_path)).is_relative_to(
                Path(os.path.abspath(self.root))
            ) or self.config_path.resolve().is_relative_to(self.root.resolve()):
                raise ConversionError("External model config must be outside checkpoint root")

    def assert_unchanged(self, *, verify_content: bool = True) -> None:
        """Check bindings and signatures, with full content verification by default.

        Signature-only mode is for a repeated planning check after the initial
        source hashes have been computed. Offline replay and publication must
        retain the default full check, including restored-signature mutations.
        """
        if type(verify_content) is not bool:
            raise ValueError("invalid_content_verification_mode")
        try:
            self._assert_selection()
            records = self.manifest["files"]
            paths = [self.resolve_file(record["filename"], repo=record["repo"]) for record in records]
            expected = [tuple(record["signature"]) for record in records]
            if verify_content:
                _, hashes = _verified_hashes(paths, expected_signatures=expected)
                if hashes != [record["sha256"] for record in records]:
                    raise ConversionError("Source changed during conversion")
            else:
                for path, signature in zip(paths, expected, strict=True):
                    if signature != _signature(path) or signature != _signature(path):
                        raise ConversionError("Source changed during conversion")
            # Recheck selection after the payload reads: a new index cannot
            # redirect the source between the initial binding check and publish.
            self._assert_selection()
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
        result = {
            "sources": sources,
            "source_files": files,
            "source_manifest_sha256": hashlib.sha256(
                json.dumps({"sources": sources, "files": files}, sort_keys=True).encode()
            ).hexdigest(),
        }
        if self.conversion_profile is not None:
            result["conversion_profile"] = self.conversion_profile
            result["source_manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    {"sources": sources, "files": files, "conversion_profile": self.conversion_profile}, sort_keys=True
                ).encode()
            ).hexdigest()
        if self.artifact_mode != "text":
            result["artifact_mode"] = self.artifact_mode
            result["source_manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    {
                        "sources": sources,
                        "files": files,
                        "artifact_mode": self.artifact_mode,
                        **(
                            {"conversion_profile": self.conversion_profile}
                            if self.conversion_profile is not None
                            else {}
                        ),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        return result

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


def _replay(
    path: Path,
    *,
    repo: str | None,
    revision: str | None,
    metadata_profile: str | None,
    architecture: str | None,
    artifact_mode: str,
) -> SourceSnapshot:
    manifest = read_json(path)
    try:
        schema = manifest["schema_version"]
        profile = manifest.get("conversion_profile")
        if type(schema) is not int or schema not in {1, 2, 3} or not isinstance(manifest["root"], str):
            raise ValueError()
        if schema in {1, 2} and ("artifact_mode" in manifest or artifact_mode != "text"):
            raise ValueError()
        if schema == 3 and (
            manifest.get("artifact_mode") != "vision-language"
            or artifact_mode != "vision-language"
            or profile is None
            and (
                metadata_profile is not None
                or manifest.get("primary_repo") != VISION_REPO
                or manifest.get("primary_revision") != VISION_REVISION
            )
        ):
            raise ValueError()
        if schema == 1 and (profile is not None or metadata_profile is not None):
            raise ValueError()
        if (schema == 2 or schema == 3 and profile is not None) and (
            not isinstance(profile, dict)
            or set(profile) != {"architecture", "metadata_profile"}
            or profile["metadata_profile"] != GENERIC_PROFILE
            or metadata_profile is not None
            and metadata_profile != profile["metadata_profile"]
            or architecture is not None
            and architecture != profile["architecture"]
        ):
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
        entry = get_model(primary) if primary and profile is None else None
        if (schema == 2 or schema == 3 and profile is not None) and entry is not None:
            raise ValueError()
        names = tuple(record["filename"] for record in manifest["files"] if record["repo"] == primary)
        rules = (
            (family_source_files if schema == 3 else generic_files)(
                primary, tuple(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
            )
            if profile is not None
            else source_rules(entry, artifact_mode)
            if entry is not None
            else (
                {"repo": primary, "filename": name}
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
            )
        )
        if any((rule["repo"], rule["filename"]) not in known for rule in rules):
            raise ValueError()
        snapshot = SourceSnapshot(root, primary, manifest["primary_revision"], manifest)
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
    metadata_profile: str | None = None,
    architecture: str | None = None,
    artifact_mode: str = "text",
    progress: ProgressCallback | ProgressReporter | None = None,
) -> SourceSnapshot:
    """Resolve all repositories before any file downloads or conversion starts."""
    progress = reporter(progress)
    if repo is not None:
        repo = huggingface_repository(repo)
    if artifact_mode not in {"text", "vision-language"}:
        raise ValueError("invalid_artifact_mode")
    if source_manifest is not None:
        snapshot = _replay(
            Path(source_manifest),
            repo=repo,
            revision=revision,
            metadata_profile=metadata_profile,
            architecture=architecture,
            artifact_mode=artifact_mode,
        )
        if progress is not None:
            progress.emit("download", "skipped", force=True)
        return snapshot
    if revision is not None and not isinstance(revision, str):
        raise ConversionError("Invalid source revision")
    path = Path(value).expanduser()
    local = path.is_dir()
    if not local and (path.exists() or str(value).startswith(("/", ".", "~"))):
        raise ValueError("invalid_source")
    if not local:
        repository = huggingface_repository(str(value))
        if repo is not None and repo != repository:
            raise ValueError("conflicting_repository")
        repo = repository
    entry = get_model(repo) if repo else None
    if artifact_mode == "vision-language":
        if repo == VISION_REPO:
            if metadata_profile is not None or revision not in {None, VISION_REVISION}:
                raise UnsupportedConversion("Vision conversion requires the pinned official checkpoint")
            revision = VISION_REVISION
        else:
            if entry is not None and entry.get("adapter") != "qwen3_5":
                raise UnsupportedConversion("Vision family requires Qwen3.5 architecture")
            # Vision family sources always bind their own metadata, even when
            # the text catalog has external metadata rules for this repository.
            entry = None
            metadata_profile = metadata_profile or GENERIC_PROFILE
            if architecture not in {None, "qwen3_5"}:
                raise UnsupportedConversion("Vision family requires Qwen3.5 architecture")
    if metadata_profile not in {None, GENERIC_PROFILE} or metadata_profile is not None and entry is not None:
        raise UnsupportedConversion("Generic metadata requires an uncatalogued source")
    if repo is not None and entry is None and metadata_profile is None:
        raise UnsupportedConversion("Repository is outside the pinned conversion catalog")
    if not local and offline:
        raise ConversionError("Offline resolution requires a source manifest")
    rules = (
        source_rules(entry, artifact_mode)
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
    remote_repositories = repositories - ({repo} if local else set())
    downloads = DownloadProgress(progress) if progress is not None and remote_repositories else None
    if progress is not None and downloads is None:
        progress.emit("download", "skipped", force=True)
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
            if artifact_mode == "vision-language" and repo == VISION_REPO and resolved != VISION_REVISION:
                raise ConversionError("Vision source resolved to a different checkpoint")
            revisions[repository] = resolved
    else:
        revisions = {repo: revision}
    selected = None
    if local:
        root = path.resolve()
        names = tuple(sorted(file.relative_to(root).as_posix() for file in root.rglob("*") if file.is_file()))

        def loader(filename: str) -> Path:
            return authorize_source_path(root, root / _relative(filename))
    else:
        names = tuple(sorted(hub.list_repo_files(repo_id=repo, revision=revisions[repo])))

        def loader(filename: str) -> Path:
            _relative(filename)
            if downloads is not None:
                return downloads.download(hub, repo, filename, revisions[repo])
            return Path(
                hub.hf_hub_download(repo_id=repo, filename=filename, revision=revisions[repo], local_files_only=False)
            )

        config_rule = next(rule for rule in rules if rule["role"] == "model_config")
        root_filename = config_rule["filename"]
        if config_rule["repo"] != repo:
            # Config-less checkpoint repositories must derive their physical
            # root from a selected primary weight, never an external sidecar.
            selected, index_name = _weights(names, entry["container"] if entry else ".safetensors", loader)
            if downloads is not None:
                downloads.select(
                    hub,
                    [(rule["repo"], rule["filename"], revisions[rule["repo"]]) for rule in rules]
                    + [(repo, name, revisions[repo]) for name in ((index_name,) if index_name else ()) + selected],
                )
            root_filename = selected[0]
        config_path = loader(root_filename)
        # snapshot files preserve repository-relative placement in the hub cache.
        root = config_path
        for _ in _relative(root_filename).parts:
            root = root.parent
    profile = None
    if metadata_profile == GENERIC_PROFILE:
        rules = family_source_files(repo, names) if artifact_mode == "vision-language" else generic_files(repo, names)
        config_path = authorize_source_path(root, loader("config.json"))
        detected = architecture_for_config(
            read_family_config(config_path) if artifact_mode == "vision-language" else read_json(config_path),
            architecture,
        )
        if artifact_mode == "vision-language" and detected != "qwen3_5":
            raise UnsupportedConversion("Vision family requires Qwen3.5 architecture")
        profile = {"architecture": detected, "metadata_profile": GENERIC_PROFILE}
    if selected is None:
        selected, index_name = _weights(names, entry["container"] if entry else ".safetensors", loader)
    descriptors = list(rules)
    if index_name:
        descriptors.append({"role": "weight_index", "repo": repo, "filename": index_name})
    descriptors.extend({"role": "weights", "repo": repo, "filename": name} for name in selected)
    if downloads is not None:
        downloads.select(
            hub,
            [
                (descriptor["repo"], descriptor["filename"], revisions[descriptor["repo"]])
                for descriptor in descriptors
                if not local or descriptor["repo"] != repo
            ],
        )
    records, known = [], set()
    for descriptor in descriptors:
        effective_repo, filename = descriptor["repo"], descriptor["filename"]
        _relative(filename)
        if (effective_repo, filename) in known:
            continue
        known.add((effective_repo, filename))
        if effective_repo == repo:
            resolved_path = authorize_source_path(root, loader(filename))
        else:
            if downloads is not None:
                resolved_path = downloads.download(hub, effective_repo, filename, revisions[effective_repo])
            else:
                resolved_path = Path(
                    hub.hf_hub_download(
                        repo_id=effective_repo,
                        filename=filename,
                        revision=revisions[effective_repo],
                        local_files_only=False,
                    )
                )
            resolved_path = resolved_path.resolve(strict=True)
        records.append(
            {
                "repo": effective_repo,
                "revision": revisions[effective_repo],
                "filename": filename,
                "role": descriptor["role"],
                "path": str(resolved_path.absolute()),
            }
        )
    signatures, hashes = _verified_hashes([Path(record["path"]) for record in records])
    for record, signature, digest in zip(records, signatures, hashes, strict=True):
        record.update(size=signature[2], sha256=digest, signature=list(signature))
    manifest = {
        "schema_version": 3 if artifact_mode == "vision-language" else 2 if profile is not None else 1,
        "root": str(root.absolute()),
        "primary_repo": repo,
        "primary_revision": revisions[repo],
        "sources": [
            {"repo": repository, "revision": revisions[repository]}
            for repository in sorted(revisions, key=lambda item: item or "")
        ],
        "files": records,
    }
    if profile is not None:
        manifest["conversion_profile"] = profile
    if artifact_mode != "text":
        manifest["artifact_mode"] = artifact_mode
    snapshot = SourceSnapshot(root, repo, revisions[repo], manifest)
    if profile is not None:
        snapshot.assert_unchanged(verify_content=False)
    if downloads is not None:
        downloads.complete()
    return snapshot
