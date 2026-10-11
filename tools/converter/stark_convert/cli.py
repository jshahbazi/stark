"""Private, atomic conversion orchestration with isolated Lalamo fallback."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import LALAMO_REVISION

PROJECT = Path(__file__).resolve().parents[1]
ENGINE_REVISION = "8d01581474760bd1fc7f4a547ff1d02f7dada437"


class FallbackFailure(RuntimeError):
    """Only the fixed runner error category may reach command output."""

    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def toolchain_provenance() -> dict:
    """Bind evidence to exact implementation and dependency files."""
    manifest = {path.name: file_hash(path) for path in sorted((PROJECT / "stark_convert").glob("*.py"))}
    for name in ("pyproject.toml", ".python-version", "lalamo/pyproject.toml", "lalamo/.python-version"):
        manifest[name] = file_hash(PROJECT / name)
    return {
        "converter_sources_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "fast_lock_sha256": file_hash(PROJECT / "uv.lock"),
        "lalamo_lock_sha256": file_hash(PROJECT / "lalamo" / "uv.lock"),
    }


def _source(value: str, repo: str | None, revision: str | None) -> tuple[Path, str | None, str | None]:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve(), repo, revision
    if path.exists() or value.startswith(("/", ".", "~")):
        raise ValueError("invalid_source")
    if repo is not None and repo != value:
        raise ValueError("conflicting_repository")
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    resolved = HfApi().model_info(value, revision=revision).sha
    path = snapshot_download(
        value,
        revision=resolved,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.jinja",
            "*.txt",
            "*.model",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.nemo",
            "*.yaml",
            "*.tiktoken",
        ],
    )
    return Path(path).resolve(), value, resolved


def _lalamo(source: Path, output: Path, repo: str | None, dtype: str) -> dict:
    if not repo:
        raise ValueError("fallback_requires_registry_repository")
    command = [
        "uv",
        "run",
        "--project",
        str(PROJECT / "lalamo"),
        "--locked",
        "python",
        str(PROJECT / "stark_convert" / "lalamo_runner.py"),
        "--source",
        str(source),
        "--repo",
        repo,
        "--output",
        str(output),
        "--dtype",
        dtype,
    ]
    env = os.environ.copy()
    env.update(HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    if env.get("HF_HUB_OFFLINE") != "1":
        command.append("--allow-downloads")
    if (
        source.parent.name == "snapshots"
        and len(source.name) == 40
        and all(c in "0123456789abcdef" for c in source.name)
    ):
        command.extend(("--source-revision", source.name))
    process = subprocess.run(command, capture_output=True, env=env, check=False)
    try:
        result = json.loads(process.stdout)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("lalamo_protocol_failed") from error
    if not isinstance(result, dict):
        raise RuntimeError("lalamo_protocol_failed")
    if process.returncode or result.get("status") != "ok":
        allowed = {
            "unsupported_model",
            "output_exists",
            "missing_source_file",
            "filesystem_permission",
            "memory_exhausted",
            "dependency_error",
            "invalid_model",
            "filesystem_error",
            "interrupted",
            "conversion_failed",
            "external_source_unavailable_offline",
        }
        category = result.get("error_category")
        raise FallbackFailure(category if isinstance(category, str) and category in allowed else "conversion_failed")
    return result


def _publish(staging: Path, output: Path) -> None:
    """Use the host's atomic no-replace rename, including empty directories."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(os.fsencode(staging), os.fsencode(output), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError("unsupported_publication_platform")
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError("output_exists")
        raise OSError(code, "atomic_publication_failed")


def _metadata_signatures(source: Path) -> dict:
    result = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = source / name
        if path.exists():
            stat = path.stat()
            result[name] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        else:
            result[name] = None
    return result


def convert(
    source: Path,
    output: Path,
    *,
    repo: str | None,
    dtype: str,
    backend: str,
    chunk_bytes: int,
    source_revision: str | None = None,
) -> dict:
    """Publish a completed conversion only; malformed input never triggers fallback."""
    from .metadata import UnsupportedMetadata, write_metadata
    from .qwen3 import build_qwen3_plan
    from .streaming import ConversionError, UnsupportedConversion, write_plan

    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("output_exists")
    if output == source or output in source.parents:
        raise ValueError("invalid_output")
    if chunk_bytes < 4096 or chunk_bytes > 1024 * 1024 * 1024:
        raise ValueError("invalid_chunk_size")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stark-convert-", dir=output.parent))
    os.chmod(staging, 0o700)
    started = time.perf_counter()
    signatures = _metadata_signatures(source)
    provenance = toolchain_provenance()
    try:
        selected = backend
        fallback = False
        if backend in {"auto", "streaming"}:
            try:
                plan = build_qwen3_plan(source, dtype=dtype)
                hf_config = json.loads((source / "config.json").read_text())
                hf_config["_stark_model_repo"] = repo
                fragments = write_metadata(source, staging, hf_config)
            except (UnsupportedConversion, UnsupportedMetadata):
                if backend == "streaming":
                    raise
                selected, fallback = "lalamo", True
                shutil.rmtree(staging)
                staging.mkdir(mode=0o700)
            else:
                selected = "streaming"
                config = {"type": "LanguageModelConfig", "decoder_config": plan.decoder_config, **fragments}
                (staging / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
                stats = write_plan(plan, staging / "model.safetensors", chunk_bytes=chunk_bytes)
        if selected == "lalamo":
            stats = _lalamo(source, staging, repo, dtype)
        if signatures != _metadata_signatures(source):
            raise ConversionError("Checkpoint metadata changed during conversion")
        if provenance != toolchain_provenance():
            raise ConversionError("Converter tooling changed during conversion")
        elapsed = time.perf_counter() - started
        if not all((staging / name).is_file() for name in ("model.safetensors", "config.json", "tokenizer.json")):
            raise RuntimeError("incomplete_output")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
        verification_start = time.perf_counter()
        receipt = {
            "status": "ok",
            "backend": selected,
            "fallback": fallback,
            "elapsed_seconds": elapsed,
            "max_rss_bytes": max(rss_bytes, int(stats.get("max_rss_bytes", 0))),
            "tensor_count": int(stats.get("tensor_count", stats.get("tensors", 0))),
            "output_bytes": (staging / "model.safetensors").stat().st_size,
            "lalamo_revision": LALAMO_REVISION,
            "engine_upstream_revision": ENGINE_REVISION,
            "source_revision": source_revision,
            "output_sha256": file_hash(staging / "model.safetensors"),
            "config_sha256": file_hash(staging / "config.json"),
            "tokenizer_sha256": file_hash(staging / "tokenizer.json"),
            **provenance,
        }
        if "source_revisions_sha256" in stats:
            receipt["source_revisions_sha256"] = stats["source_revisions_sha256"]
        receipt["verification_seconds"] = time.perf_counter() - verification_start
        (staging / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        # Same-filesystem rename publishes complete output; existing output is never replaced.
        if output.exists() or output.is_symlink():
            raise FileExistsError("output_exists")
        _publish(staging, output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert an HF model with a streaming prototype or pinned Lalamo.")
    parser.add_argument("source", help="HF repository ID or local checkpoint directory")
    parser.add_argument("--repo", help="Lalamo registry ID for a local checkpoint's model family and chat semantics")
    parser.add_argument("--revision", help="HF source revision; resolves to an immutable commit before downloading")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "streaming", "lalamo"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    try:
        source, repo, revision = _source(args.source, args.repo, args.revision)
        result = convert(
            source,
            args.output,
            repo=repo,
            dtype=args.dtype,
            backend=args.backend,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            source_revision=revision,
        )
    except BaseException as error:
        # Never expose exception text, checkpoint values, credentials, or native dumps.
        if isinstance(error, KeyboardInterrupt):
            category = "interrupted"
        else:
            category = error.category if isinstance(error, FallbackFailure) else type(error).__name__
        print(json.dumps({"status": "error", "error_category": category}), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
