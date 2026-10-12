"""Independent, private, atomic conversion through registered streaming adapters."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import logging
import os
import resource
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import CapabilityResult, ConversionRequest, ConversionResult, PlannedConversion
from .provenance import file_hash, toolchain_provenance
from .registry import adapter_for_model, catalog_sha256, get_model
from .sources import SourceSnapshot, resolve_source
from .streaming import ConversionError, UnsupportedConversion, read_json, write_plan


def capability(request: ConversionRequest) -> CapabilityResult:
    """Probe catalog coverage only; this does not validate checkpoint contents."""
    entry = get_model(request.repo)
    kind = {"LanguageModelSpec": "language", "ClassifierModelSpec": "classifier", "TTSModelSpec": "tts"}
    actual_kind = kind.get(entry["model_kind"], request.model_kind) if entry else request.model_kind
    supported = bool(entry and entry["adapter"] and entry["metadata_profile"])
    supported = supported and actual_kind == request.model_kind
    native = "supported_layout_unverified_tuple" if supported and request.dtype == "bfloat16" else "unverified"
    if supported and request.dtype != "bfloat16":
        native = "runtime_blocked"
    return CapabilityResult(
        adapter=entry["adapter"] if supported else None,
        model_kind=actual_kind,
        conversion="implemented" if supported else "unsupported",
        native_loading=native,
        reason=None if supported else "unsupported_catalog_tuple",
    )


def _validate_request(request: ConversionRequest) -> None:
    if request.backend not in {"auto", "streaming"}:
        raise ValueError("invalid_backend")
    if request.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("invalid_dtype")
    if request.model_kind not in {"language", "classifier", "tts"}:
        raise ValueError("invalid_model_kind")
    if type(request.workspace_bytes) is not int or not 4096 <= request.workspace_bytes <= 1024 * 1024 * 1024:
        raise ValueError("invalid_workspace")
    if request.require_native_support and request.dtype != "bfloat16":
        raise UnsupportedConversion("Native dense loading requires BF16")
    if capability(request).conversion == "unsupported":
        raise UnsupportedConversion("Unsupported catalog tuple")


def plan_conversion(snapshot: SourceSnapshot, request: ConversionRequest) -> PlannedConversion:
    """Validate a complete supported layout and build its single tensor/config plan."""
    from .adapters import build_llama_plan, build_mistral_plan, build_qwen2_plan
    from .qwen3 import build_qwen3_plan

    _validate_request(request)
    if snapshot.repo != request.repo:
        raise ConversionError("Request and source identity disagree")
    # Resolution/replay has already hashed every consumed file. Recheck selection
    # and signatures here; full content verification remains mandatory prepublish.
    snapshot.assert_unchanged(verify_content=False)
    config = read_json(snapshot.resolve_file("config.json"))
    adapter = adapter_for_model(request.repo, config.get("model_type"))
    planners = {
        "qwen3": build_qwen3_plan,
        "qwen2": build_qwen2_plan,
        "llama": build_llama_plan,
        "mistral": build_mistral_plan,
    }
    tensors = planners[adapter](snapshot.checkpoint, dtype=request.dtype)
    # Metadata and final content verification are still pending. Only a
    # successfully published conversion reports complete source validation.
    result = capability(request)
    return PlannedConversion(request, result, tensors, config)


def _publish(staging: Path, output: Path) -> None:
    """Use the host's atomic no-replace rename, including empty directories."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(os.fsencode(staging), os.fsencode(output), 4)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    else:
        raise RuntimeError("unsupported_publication_platform")
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError("output_exists")
        raise OSError(code, "atomic_publication_failed")


def convert(
    source: Path | SourceSnapshot,
    output: Path,
    *,
    repo: str | None,
    dtype: str = "bfloat16",
    backend: str = "auto",
    chunk_bytes: int = 8 * 1024 * 1024,
    source_revision: str | None = None,
    require_native_support: bool = False,
    model_kind: str = "language",
) -> dict:
    """Publish complete independently converted outputs; unsupported inputs fail."""
    from .metadata import write_metadata

    request = ConversionRequest(repo, dtype, model_kind, backend, chunk_bytes, require_native_support)
    _validate_request(request)
    snapshot = (
        source if isinstance(source, SourceSnapshot) else resolve_source(source, repo=repo, revision=source_revision)
    )
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("output_exists")
    if output.resolve() == snapshot.root.resolve() or output.resolve() in snapshot.root.resolve().parents:
        raise ValueError("invalid_output")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stark-convert-", dir=output.parent))
    started = time.perf_counter()
    try:
        os.chmod(staging, 0o700)
        provenance = toolchain_provenance()
        planned = plan_conversion(snapshot, request)
        config = {**planned.model_config, "_stark_model_repo": repo}
        fragments = write_metadata(
            snapshot.root, staging, config, model_entry=get_model(repo), resolve_file=snapshot.resolve_file
        )
        root_config = {"type": "LanguageModelConfig", "decoder_config": planned.tensors.decoder_config, **fragments}
        (staging / "config.json").write_text(json.dumps(root_config, indent=4), encoding="utf-8")
        stats = write_plan(planned.tensors, staging / "model.safetensors", chunk_bytes=chunk_bytes)
        elapsed = time.perf_counter() - started
        if provenance != toolchain_provenance():
            raise ConversionError("Converter tooling changed during conversion")
        if not all((staging / name).is_file() for name in ("model.safetensors", "config.json", "tokenizer.json")):
            raise ConversionError("incomplete_output")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        source_receipt = snapshot.receipt()
        artifact_hashes = {
            "source_manifest_sha256": source_receipt["source_manifest_sha256"],
            "output_sha256": stats["output_sha256"],
            "config_sha256": file_hash(staging / "config.json"),
            "tokenizer_sha256": file_hash(staging / "tokenizer.json"),
        }
        from .qualifications import qualification_for

        qualification = qualification_for(repo, dtype, artifact_hashes)
        if request.require_native_support and qualification is None:
            raise UnsupportedConversion("No recorded named-build native qualification for these exact artifacts")
        converted_capability = replace(
            planned.capability,
            source_validation="complete",
            native_loading="qualified_tuple" if qualification else planned.capability.native_loading,
        )
        result = ConversionResult(
            "streaming",
            converted_capability,
            stats["tensors"],
            stats["output_bytes"],
            stats["peak_buffer_bytes"],
            int(rss if sys.platform == "darwin" else rss * 1024),
            artifact_hashes["output_sha256"],
            source_receipt["source_manifest_sha256"],
        )
        receipt = {
            "status": "ok",
            **asdict(result),
            **source_receipt,
            **provenance,
            "catalog_sha256": catalog_sha256(),
            "elapsed_seconds": elapsed,
            "workspace_bytes": chunk_bytes,
            "source_revision": snapshot.revision,
            "config_sha256": artifact_hashes["config_sha256"],
            "tokenizer_sha256": artifact_hashes["tokenizer_sha256"],
            "native_qualification": qualification,
        }
        (staging / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        snapshot.assert_unchanged()
        if provenance != toolchain_provenance():
            raise ConversionError("Converter tooling changed during qualification")
        _publish(staging, output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps({"status": "error", "error_category": "invalid_arguments"}), file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    parser = _Parser(description="Convert selected HF catalog profiles with independent Stark streaming adapters.")
    parser.add_argument("source", nargs="?", help="HF repository ID or local checkpoint directory")
    parser.add_argument("--repo", help="Catalog identity required for a local checkpoint")
    parser.add_argument("--revision", help="HF revision resolved to an immutable commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("auto", "streaming"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--model-kind", choices=("language", "classifier", "tts"), default="language")
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--capabilities", action="store_true", help="Probe catalog coverage without reading a source")
    parser.add_argument(
        "--require-native-support",
        action="store_true",
        help="Require an exact stored load-only qualification for its named Stark build",
    )
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--write-source-manifest", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        repo = args.repo or (args.source if args.source and not Path(args.source).expanduser().is_dir() else None)
        request = ConversionRequest(
            repo, args.dtype, args.model_kind, args.backend, args.chunk_mib * 1024 * 1024, args.require_native_support
        )
        if args.capabilities:
            print(
                json.dumps(
                    {"status": "ok", "capability": asdict(capability(request)), "catalog_sha256": catalog_sha256()},
                    sort_keys=True,
                )
            )
            return 0
        if not args.output or not args.source:
            raise ValueError("missing_arguments")
        if not args.source_manifest:
            _validate_request(request)
        logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        snapshot = resolve_source(
            args.source, repo=repo, revision=args.revision, source_manifest=args.source_manifest, offline=args.offline
        )
        if args.write_source_manifest:
            snapshot.write_manifest(args.write_source_manifest)
        result = convert(
            snapshot,
            args.output,
            repo=snapshot.repo,
            dtype=args.dtype,
            backend=args.backend,
            chunk_bytes=request.workspace_bytes,
            require_native_support=args.require_native_support,
            model_kind=args.model_kind,
        )
    except BaseException as error:
        category = "interrupted" if isinstance(error, KeyboardInterrupt) else type(error).__name__
        print(json.dumps({"status": "error", "error_category": category}), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
