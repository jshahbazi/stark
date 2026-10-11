"""Quiet adapter for the separately locked Lalamo environment.

Run this file with the interpreter from tools/converter/lalamo. The streaming
project deliberately does not import Lalamo or its JAX/PyTorch dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LALAMO_REVISION = "831accb8252a6b6606c378048caeb44178f1a050"


class UnsupportedModelError(Exception):
    """The requested repository is outside the pinned upstream registry."""


class OfflineDependencyError(Exception):
    """An external registry dependency needs explicitly authorized resolution."""


class PinnedSnapshotResolver:
    """Serve primary files locally and pin each external repository once."""

    def __init__(self, primary_repo, source, *, allow_downloads=False, source_revision=None, hub=None):
        if hub is None:
            import huggingface_hub

            hub = huggingface_hub
        if source_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", source_revision):
            raise ValueError("Source revision must be immutable")
        self.primary_repo = primary_repo
        self.source = Path(source)
        self.allow_downloads = allow_downloads
        self.hub = hub
        self.revisions = {primary_repo: source_revision}

    def _revision(self, repo):
        if repo not in self.revisions:
            if not self.allow_downloads:
                raise OfflineDependencyError()
            revision = self.hub.HfApi().model_info(repo_id=repo).sha
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Unresolved external repository")
            self.revisions[repo] = revision
        return self.revisions[repo]

    @staticmethod
    def _filename(name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid repository filename")
        return relative

    def resolve_file(self, repo, filename, *, file_repo=None):
        effective_repo = file_repo or repo
        relative = self._filename(filename)
        if effective_repo == self.primary_repo:
            result = self.source / relative
            if not result.is_file():
                raise FileNotFoundError()
            return result
        revision = self._revision(effective_repo)
        return Path(
            self.hub.hf_hub_download(
                repo_id=effective_repo,
                filename=filename,
                revision=revision,
                local_files_only=False,
            )
        )

    def weight_filenames(self, repo, extension):
        if repo != self.primary_repo:
            revision = self._revision(repo)
            return tuple(
                sorted(
                    filename
                    for filename in self.hub.list_repo_files(repo_id=repo, revision=revision)
                    if filename.endswith(extension)
                )
            )
        if extension == ".safetensors":
            indexed = _indexed_weights(self.source, "model.safetensors.index.json", extension)
            names = indexed if indexed is not None else _matching_weights(self.source, extension)
            if not names:
                raise FileNotFoundError()
            return names
        if extension == ".pth":
            # All these containers are decoded by the same upstream Torch reader.
            indexed = _indexed_weights(self.source, "pytorch_model.bin.index.json", ".bin")
            if indexed is not None:
                return indexed
            for torch_extension in (".pth", ".pt", ".bin"):
                names = _matching_weights(self.source, torch_extension)
                if names:
                    return names
            raise FileNotFoundError()
        return _matching_weights(self.source, extension)

    def provenance(self):
        return {"sources": [{"repo": repo, "revision": revision} for repo, revision in sorted(self.revisions.items())]}


@contextmanager
def _patched_origins(origins, resolver):
    """Retain each registered Origin's behavior, including Nemo extraction."""
    original_file, original_weights = origins.hf_resolve_file, origins.hf_resolve_weights

    def resolve_file(repo, file_spec, progress_callback=None):
        origins.report_status(progress_callback, origins.DownloadingFileEvent(file_spec))
        result = resolver.resolve_file(repo, file_spec.filename, file_repo=file_spec.repo)
        origins.report_status(progress_callback, origins.FinishedDownloadingFileEvent(file_spec))
        return result

    def resolve_weights(repo, extension, progress_callback=None):
        return tuple(
            resolve_file(repo, origins.FileSpec(filename), progress_callback)
            for filename in resolver.weight_filenames(repo, extension)
        )

    origins.hf_resolve_file, origins.hf_resolve_weights = resolve_file, resolve_weights
    try:
        yield
    finally:
        origins.hf_resolve_file, origins.hf_resolve_weights = original_file, original_weights


@contextmanager
def _quiet_native_output():
    """Suppress both Python messages and native-library diagnostics."""
    sys.stdout.flush()
    sys.stderr.flush()
    original = (os.dup(1), os.dup(2))
    with open(os.devnull, "wb") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original[0], 1)
            os.dup2(original[1], 2)
            os.close(original[0])
            os.close(original[1])


def _indexed_weights(source, index_name, extension):
    index = source / index_name
    if not index.is_file():
        return None
    with index.open() as handle:
        names = sorted(set(json.load(handle)["weight_map"].values()))
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not name.endswith(extension):
            raise ValueError("Invalid checkpoint index")
        if not (source / relative).is_file():
            raise FileNotFoundError()
    return tuple(names)


def _matching_weights(source, extension):
    return tuple(
        sorted(path.relative_to(source).as_posix() for path in source.rglob(f"*{extension}") if path.is_file())
    )


def _weight_files(source: Path) -> tuple[tuple[str, ...], str]:
    """Prefer the snapshot's index to avoid loading unrelated tensor files."""
    for index_name, extension in (
        ("model.safetensors.index.json", ".safetensors"),
        ("pytorch_model.bin.index.json", ".bin"),
    ):
        names = _indexed_weights(source, index_name, extension)
        if names is not None:
            return names, extension
    for extension in (".safetensors", ".bin", ".pth", ".pt"):
        names = _matching_weights(source, extension)
        if names:
            return names, extension
    raise FileNotFoundError()


def _header_metadata(source: Path, names: tuple[str, ...]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    # Lalamo combines shards with ChainMap; the first shard wins duplicates.
    for name in reversed(names):
        with (source / name).open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Invalid tensor header")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 100_000_000:
                raise ValueError("Invalid tensor header")
            header = json.loads(handle.read(length))
            metadata.update(header.get("__metadata__", {}))
    return metadata


def _tensor_count(output: Path) -> int:
    weights = output / "model.safetensors"
    if not weights.is_file():
        return 0
    with weights.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 100_000_000:
            raise ValueError("Invalid tensor header")
        return sum(key != "__metadata__" for key in json.loads(handle.read(length)))


def _run(args: argparse.Namespace) -> dict[str, object]:
    import jax.numpy as jnp
    from lalamo.model_import import origins
    from lalamo.model_import.common import import_model
    from lalamo.model_import.origins import HuggingFaceOrigin, NemoOrigin
    from lalamo.model_registry import ModelRegistry
    from lalamo.utils.sharding import ShardingConfig

    source = args.source.resolve()
    model_spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model.get(args.repo)
    if model_spec is None:
        raise UnsupportedModelError()
    if not isinstance(model_spec.origin, (HuggingFaceOrigin, NemoOrigin)):
        raise UnsupportedModelError()
    resolver = PinnedSnapshotResolver(
        model_spec.origin.repo,
        source,
        allow_downloads=args.allow_downloads,
        source_revision=args.source_revision,
    )
    if args.output.is_symlink() or (args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir()))):
        raise FileExistsError()

    with _patched_origins(origins, resolver):
        if args.metadata_only:
            from lalamo.model_import.common import _import_chat_codec, _import_generation_config
            from lalamo.model_import.model_spec import LanguageModelSpec

            if not isinstance(model_spec, LanguageModelSpec) or not isinstance(model_spec.origin, HuggingFaceOrigin):
                raise UnsupportedModelError()
            names = resolver.weight_filenames(model_spec.origin.repo, ".safetensors")
            foreign_config = model_spec.config_type.from_json(
                model_spec.origin.resolve_file(model_spec.configs.model_config)
            )
            tokenizer, codec = _import_chat_codec(model_spec)
            generation = _import_generation_config(model_spec, foreign_config, None)
            config = foreign_config.to_lalamo_config(
                metadata_dict=_header_metadata(source, names),
                token_codec_config=codec,
                generation_config=generation,
            )
            args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
            (args.output / "config.json").write_text(json.dumps(config.to_json(), indent=4))
            tokenizer.save(str(args.output / "tokenizer.json"))
        else:
            imported = import_model(
                model_spec,
                sharding_config=ShardingConfig.replicated(),
                dtype=jnp.dtype(args.dtype),
            )
            imported.model.save(args.output)

    provenance = resolver.provenance()
    encoded_provenance = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    descriptor = os.open(args.output / "source-revisions.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded_provenance + b"\n")

    return {
        "status": "ok",
        "backend": "lalamo_metadata" if args.metadata_only else "lalamo",
        "source_revisions_sha256": hashlib.sha256(encoded_provenance).hexdigest(),
        "tensor_count": _tensor_count(args.output),
        "output_bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
    }


def _error_category(error: BaseException) -> str:
    if isinstance(error, OfflineDependencyError):
        return "external_source_unavailable_offline"
    if isinstance(error, UnsupportedModelError):
        return "unsupported_model"
    if isinstance(error, FileExistsError):
        return "output_exists"
    if isinstance(error, FileNotFoundError):
        return "missing_source_file"
    if isinstance(error, PermissionError):
        return "filesystem_permission"
    if isinstance(error, MemoryError):
        return "memory_exhausted"
    if isinstance(error, ImportError):
        return "dependency_error"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return "invalid_model"
    if isinstance(error, OSError):
        return "filesystem_error"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    return "conversion_failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--source-revision", help="Immutable Hugging Face revision of the primary snapshot")
    args = parser.parse_args()
    # Primary files remain authoritative; external downloads require the flag.
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}
    args.allow_downloads = args.allow_downloads and not offline
    if not args.allow_downloads:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("LALAMO_VERBOSE", None)
    started = time.perf_counter()
    try:
        with _quiet_native_output():
            result = _run(args)
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_category": _error_category(error)}, separators=(",", ":")))
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        max_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
