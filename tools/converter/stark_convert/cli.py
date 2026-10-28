"""Independent, private, atomic conversion through registered streaming adapters."""

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
import sys
import tempfile
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from . import __version__
from .contracts import CapabilityResult, ConversionRequest, ConversionResult, PlannedConversion
from .model_support import MODES, preflight_source, support_matrix
from .profiles import GENERIC_PROFILE, generic_entry
from .progress import FDProgressSink, ProgressCallback, ProgressReporter, reporter
from .provenance import file_hash, toolchain_provenance
from .registry import (
    CONFIG_ADAPTERS,
    adapter_for_model,
    architecture_for_config,
    catalog_sha256,
    get_model,
    model_kind_for_architecture,
)
from .sources import SourceSnapshot, huggingface_repository, resolve_source
from .streaming import ConversionError, UnsupportedConversion, read_json, read_regular_bytes, write_plan
from .vision_contract import REPO as VISION_REPO
from .vision_contract import source_rules, validate_vision_source
from .vision_family import (
    ALTERNATE_OUTPUT_TOKENIZER_SHA256,
    COMPONENT_FILES,
    OUTPUT_TOKENIZER_SHA256,
    family_config_bytes,
    family_processor_filename,
    family_reasoning_profile,
    read_family_config,
    validate_family_output_header,
    validate_family_plan,
)


def capability(request: ConversionRequest) -> CapabilityResult:
    """Probe catalog coverage only; this does not validate checkpoint contents."""
    if request.artifact_mode != "text":
        if request.repo == "google/gemma-4-E2B-it":
            supported = (
                request.artifact_mode == "vision-language"
                and request.dtype == "bfloat16"
                and request.model_kind == "language"
                and request.architecture in {None, "gemma4_text"}
                and request.metadata_profile is None
            )
            return CapabilityResult(
                adapter="gemma4_text" if supported else None,
                model_kind=request.model_kind,
                conversion="implemented" if supported else "unsupported",
                native_loading="supported_family_unverified" if supported else "runtime_blocked",
                reason=None if supported else "unsupported_gemma4_vision_tuple",
            )
        official = request.repo == VISION_REPO
        supported = (
            request.artifact_mode == "vision-language"
            and (official or get_model(request.repo) is None or get_model(request.repo).get("adapter") == "qwen3_5")
            and request.dtype == "bfloat16"
            and request.model_kind == "language"
            and request.architecture in {None, "qwen3_5"}
            and request.metadata_profile in ({None} if official else {None, GENERIC_PROFILE})
        )
        return CapabilityResult(
            adapter="qwen3_5" if supported else None,
            model_kind=request.model_kind,
            conversion="implemented" if supported else "unsupported",
            native_loading=("supported_layout_unverified_tuple" if official else "supported_family_unverified")
            if supported
            else "runtime_blocked",
            reason=None if supported else "unsupported_vision_tuple",
        )
    entry = get_model(request.repo)
    if request.metadata_profile == GENERIC_PROFILE:
        supported = (
            entry is None
            and request.architecture in CONFIG_ADAPTERS.values()
            and request.model_kind == model_kind_for_architecture(request.architecture)
        )
        return CapabilityResult(
            adapter=request.architecture if supported else None,
            model_kind=request.model_kind,
            conversion="implemented" if supported else "unsupported",
            native_loading="runtime_blocked" if supported and request.dtype != "bfloat16" else "unverified",
            reason=None if supported else "checkpoint_architecture_required",
        )
    kind = {"LanguageModelSpec": "language", "ClassifierModelSpec": "classifier", "TTSModelSpec": "tts"}
    actual_kind = kind.get(entry["model_kind"], request.model_kind) if entry else request.model_kind
    supported = bool(entry and entry["adapter"] and entry["metadata_profile"])
    supported = supported and actual_kind == request.model_kind
    supported = supported and (request.architecture is None or entry["adapter"] == request.architecture)
    native = "supported_layout_unverified_tuple" if supported and request.dtype == "bfloat16" else "unverified"
    if supported and entry["adapter"] in {"qwen3_5", "lfm2", "llamba", "modernbert"}:
        native = "unverified"
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
    if request.artifact_mode not in {"text", "vision-language"}:
        raise ValueError("invalid_artifact_mode")
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
    if request.artifact_mode == "vision-language" and request.require_native_support:
        from .vision_runtime import has_recorded_image_runtime

        if request.repo != VISION_REPO:
            raise UnsupportedConversion("Merged family conversion has no exact recorded native qualification")
        if not has_recorded_image_runtime():
            raise UnsupportedConversion("No named-build native image qualification is recorded")
    if request.architecture is not None and request.architecture not in CONFIG_ADAPTERS.values():
        raise UnsupportedConversion("Unsupported requested architecture")
    if request.artifact_mode == "vision-language":
        if capability(request).conversion == "unsupported":
            raise UnsupportedConversion("Unsupported catalog tuple")
        return
    if request.metadata_profile is not None:
        if (
            request.metadata_profile != GENERIC_PROFILE
            or get_model(request.repo) is not None
            or request.model_kind not in {"language", "classifier"}
            or request.architecture is not None
            and request.model_kind != model_kind_for_architecture(request.architecture)
        ):
            raise UnsupportedConversion("Generic metadata requires an uncatalogued model of the matching kind")
        if request.require_native_support:
            raise UnsupportedConversion("Generic conversion has no recorded native qualification")
        return
    if capability(request).conversion == "unsupported":
        raise UnsupportedConversion("Unsupported catalog tuple")


def plan_conversion(snapshot: SourceSnapshot, request: ConversionRequest) -> PlannedConversion:
    """Validate a complete supported layout and build its single tensor/config plan."""
    from .adapters import (
        build_gemma4_plan,
        build_gemma_plan,
        build_granite_plan,
        build_lfm2_plan,
        build_llama_plan,
        build_llamba_plan,
        build_mistral_plan,
        build_modernbert_plan,
        build_olmo3_plan,
        build_phi_plan,
        build_qwen2_plan,
        build_qwen35_plan,
        build_smollm3_plan,
    )
    from .qwen3 import build_qwen3_plan

    if request.artifact_mode == "vision-language" and snapshot.conversion_profile is not None:
        request = replace(
            request,
            metadata_profile=request.metadata_profile or snapshot.conversion_profile["metadata_profile"],
            architecture=request.architecture or snapshot.conversion_profile["architecture"],
        )
    _validate_request(request)
    if snapshot.repo != request.repo:
        raise ConversionError("Request and source identity disagree")
    if snapshot.artifact_mode != request.artifact_mode:
        raise ConversionError("Request and source artifact modes disagree")
    # Resolution/replay has already hashed every consumed file. Recheck selection
    # and signatures here; full content verification remains mandatory prepublish.
    snapshot.assert_unchanged(verify_content=False)
    config = (
        read_family_config(snapshot.config_path)
        if request.artifact_mode == "vision-language"
        else read_json(snapshot.config_path)
    )
    gemma_images = request.artifact_mode == "vision-language" and snapshot.repo == "google/gemma-4-E2B-it"
    if gemma_images:
        from .adapters.gemma4_vision import gemma4_vision_contract

        artifact_contract = gemma4_vision_contract(snapshot.checkpoint, snapshot.repo, snapshot.revision)
        artifact_contract["source"]["hashes"] = {
            record["filename"]: record["sha256"]
            for record in snapshot.manifest["files"]
            if record["role"] not in {"weights", "weight_index"}
        }
    else:
        artifact_contract = (
            validate_vision_source(snapshot, request.dtype) if request.artifact_mode == "vision-language" else None
        )
    if snapshot.conversion_profile is not None:
        if request.metadata_profile != GENERIC_PROFILE:
            raise ConversionError("Request and source metadata profile disagree")
        adapter = architecture_for_config(config, request.architecture)
        if adapter != snapshot.conversion_profile["architecture"]:
            raise ConversionError("Request and source architecture disagree")
        request = replace(request, architecture=adapter)
    else:
        if request.metadata_profile is not None:
            raise ConversionError("Request and source metadata profile disagree")
        adapter = adapter_for_model(request.repo, config.get("model_type"))
        if request.architecture is not None and adapter != request.architecture:
            raise ConversionError("Requested architecture disagrees with catalog profile")
    if request.model_kind != model_kind_for_architecture(adapter):
        raise UnsupportedConversion("Checkpoint architecture disagrees with the requested model kind")
    if adapter in {"qwen3", "llama", "smollm3", "gemma3_text", "gemma4_text", "olmo3", "qwen3_5", "lfm2"}:
        from .compressed import parse_mlx_quantization

        entry = get_model(request.repo)
        packing = entry.get("compressed_format") if entry else None
        packing_config = config
        if adapter == "qwen3_5":
            from .adapters.qwen35 import normalize_qwen35_config

            packing_config = normalize_qwen35_config(config)
        elif adapter == "gemma4_text":
            from .adapters.gemma4 import normalize_gemma4_config

            packing_config = normalize_gemma4_config(config)
        quantization = parse_mlx_quantization(packing_config)
        if packing is not None:
            if quantization is None or quantization.bits != packing["bits"]:
                raise ConversionError("Checkpoint packing disagrees with its catalog profile")
        elif quantization is not None and request.metadata_profile is None:
            raise UnsupportedConversion("Packed checkpoint requires a registered compressed profile")
    llamba_packing = None
    if adapter == "llamba":
        from .compressed import parse_mlx_quantization

        entry = get_model(request.repo)
        llamba_packing = entry.get("compressed_format") if entry else None
        quantization = parse_mlx_quantization(config)
        if llamba_packing is None and quantization is not None and request.metadata_profile is None:
            raise UnsupportedConversion("Packed checkpoint requires a registered compressed profile")
        if llamba_packing is not None and quantization is not None and quantization.bits != llamba_packing["bits"]:
            raise ConversionError("Checkpoint packing disagrees with its catalog profile")
    planners = {
        "qwen3": build_qwen3_plan,
        "qwen2": build_qwen2_plan,
        "llama": build_llama_plan,
        "mistral": build_mistral_plan,
        "gemma2": partial(build_gemma_plan, family="gemma2"),
        "gemma3_text": partial(build_gemma_plan, family="gemma3_text"),
        "gemma4_text": build_gemma4_plan,
        "olmo3": build_olmo3_plan,
        "phi3": build_phi_plan,
        "granite": build_granite_plan,
        "smollm3": build_smollm3_plan,
        "qwen3_5": partial(build_qwen35_plan, include_vision=request.artifact_mode == "vision-language"),
        "lfm2": build_lfm2_plan,
        "llamba": partial(
            build_llamba_plan,
            config=config,
            expected_mlx_bits=llamba_packing["bits"] if llamba_packing else None,
        ),
        "modernbert": build_modernbert_plan,
    }
    if gemma_images:
        from .adapters.gemma4_vision import build_gemma4_vision_plan

        planners["gemma4_text"] = build_gemma4_vision_plan
    tensors = planners[adapter](snapshot.checkpoint, dtype=request.dtype)
    if artifact_contract is not None and artifact_contract["schema_version"] == 3:
        validate_family_plan(tensors)
        if any(
            source.dtype
            not in (
                {"BF16", "F32"}
                if operation.kind == "cast_f32" and operation.name.endswith((".scales", ".a_log", ".dt_bias"))
                else {"BF16"}
            )
            for operation in tensors.operations
            for source in operation.sources
        ):
            raise UnsupportedConversion("Vision family requires merged BF16 matrices and BF16/F32 control weights")
    if artifact_contract is not None and adapter == "qwen3_5":
        for layer in tensors.decoder_config["transformer_config"]["layer_configs"]:
            if layer["rope_config"] is not None:
                layer["rope_config"] = {
                    **layer["rope_config"],
                    "type": "Qwen35MRoPEConfig",
                    **artifact_contract["position_contract"],
                }
    if adapter == "qwen3_5":
        from .adapters.qwen35 import normalize_qwen35_config

        config = normalize_qwen35_config(config)
    elif adapter == "gemma4_text":
        from .adapters.gemma4 import normalize_gemma4_config

        config = normalize_gemma4_config(config)
    # Metadata and final content verification are still pending. Only a
    # successfully published conversion reports complete source validation.
    result = capability(request)
    layers = tensors.decoder_config["transformer_config"]["layer_configs"] if request.model_kind == "language" else ()
    for layer in layers:
        mixer = layer["mixer_config"]
        if mixer["type"] == "DeltaNetConfig" and (mixer["head_dim"] != 128 or mixer["value_head_dim"] != 128):
            result = replace(result, native_loading="runtime_blocked", reason="native_delta_net_dimensions")
        if mixer["type"] in {"DeltaNetConfig", "ShortConvConfig", "Mamba2Config"} and mixer["kernel_size"] < 2:
            result = replace(result, native_loading="runtime_blocked", reason="native_convolution_kernel")
    return PlannedConversion(request, result, tensors, config, artifact_contract)


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
    architecture: str | None = None,
    metadata_profile: str | None = None,
    artifact_mode: str = "text",
    progress: ProgressCallback | ProgressReporter | None = None,
) -> dict:
    """Publish complete independently converted outputs; unsupported inputs fail."""
    from .metadata import write_metadata

    if isinstance(source, SourceSnapshot) and source.conversion_profile is not None:
        profile = source.conversion_profile
        metadata_profile = metadata_profile or profile["metadata_profile"]
        architecture = architecture or profile["architecture"]
    request = ConversionRequest(
        repo,
        dtype,
        model_kind,
        backend,
        chunk_bytes,
        require_native_support,
        architecture,
        metadata_profile,
        artifact_mode,
    )
    _validate_request(request)
    progress = reporter(progress)
    snapshot = (
        source
        if isinstance(source, SourceSnapshot)
        else resolve_source(
            source,
            repo=repo,
            revision=source_revision,
            architecture=architecture,
            metadata_profile=metadata_profile,
            artifact_mode=artifact_mode,
            dtype=dtype,
            model_kind=model_kind,
            **({"progress": progress} if progress is not None else {}),
        )
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
        if progress is not None:
            progress.emit("conversion", "running", force=True)
        provenance = toolchain_provenance()
        planned = plan_conversion(snapshot, request)
        entry = get_model(repo)
        if artifact_mode == "vision-language" and entry is not None and snapshot.conversion_profile is None:
            entry = {
                **entry,
                "files": {
                    rule["role"]: {"repo": rule["repo"], "filename": rule["filename"]}
                    for rule in source_rules(entry, artifact_mode)
                },
                "metadata": {
                    **entry["metadata"],
                    "generation_config": None,
                    "generation_params_overrides": None,
                    "inline_sources": {
                        key: value
                        for key, value in entry["metadata"].get("inline_sources", {}).items()
                        if key not in {"generation_config", "generation_params_overrides"}
                    },
                },
            }
        if snapshot.conversion_profile is not None:
            names = tuple(record["filename"] for record in snapshot.manifest["files"])
            entry = generic_entry(repo, planned.request.architecture, names)
            if (
                planned.artifact_contract is not None
                and planned.artifact_contract.get("family", {}).get("version") == 2
            ):
                entry["metadata"]["reasoning_id"] = family_reasoning_profile(planned.artifact_contract)
        config = {**planned.model_config, "_stark_model_repo": repo}
        if artifact_mode == "vision-language":
            # The exact checkpoint publishes the same template inline and as a
            # sidecar. Pin both, then give the metadata writer one unambiguous
            # source without changing its rules for other catalog profiles.
            tokenizer_config = read_json(snapshot.resolve_file("tokenizer_config.json"))
            inline_template = tokenizer_config.get("chat_template")
            if inline_template is not None:
                if inline_template != read_regular_bytes(snapshot.resolve_file("chat_template.jinja")).decode("utf-8"):
                    raise ConversionError("Conflicting vision chat-template metadata")
                entry = {
                    **entry,
                    "files": {key: value for key, value in entry["files"].items() if key != "chat_template"},
                }
        if planned.request.model_kind == "classifier":
            from .classifier_metadata import write_classifier_metadata

            fragments = write_classifier_metadata(
                snapshot.root, staging, config, model_entry=entry, resolve_file=snapshot.resolve_file
            )
            root_config = {
                "type": "ClassifierModelConfig",
                "classifier_config": planned.tensors.decoder_config,
                **fragments,
            }
        else:
            fragments = write_metadata(
                snapshot.root, staging, config, model_entry=entry, resolve_file=snapshot.resolve_file
            )
            root_config = {
                "type": (
                    "Gemma4VisionLanguageModelConfig"
                    if planned.capability.adapter == "gemma4_text"
                    else "Qwen35VisionLanguageModelConfig"
                )
                if artifact_mode == "vision-language"
                else "LanguageModelConfig",
                "decoder_config": planned.tensors.decoder_config,
                **fragments,
            }
        family = planned.artifact_contract is not None and planned.artifact_contract["schema_version"] == 3
        gemma_image = planned.artifact_contract is not None and planned.artifact_contract["schema_version"] == 4
        if planned.artifact_contract is not None:
            root_config["artifact_contract"] = planned.artifact_contract
            for filename in ("preprocessor_config.json", "chat_template.jinja"):
                hashes = planned.artifact_contract["source"]["hashes"]
                source_filename = (
                    "processor_config.json"
                    if gemma_image and filename == "preprocessor_config.json"
                    else family_processor_filename(hashes)
                    if family and filename == "preprocessor_config.json"
                    else filename
                )
                payload = read_regular_bytes(snapshot.resolve_file(source_filename))
                if hashlib.sha256(payload).hexdigest() != hashes[source_filename]:
                    raise ConversionError("Vision metadata changed during conversion")
                if family and filename == "preprocessor_config.json":
                    payload = json.dumps(planned.artifact_contract["processor_config"], indent=2).encode("utf-8")
                (staging / filename).write_bytes(payload)
        if family:
            if file_hash(staging / "tokenizer.json") not in {
                OUTPUT_TOKENIZER_SHA256,
                ALTERNATE_OUTPUT_TOKENIZER_SHA256,
            }:
                raise UnsupportedConversion("Vision family normalized tokenizer is outside the implemented profile")
            # Hash strings have a fixed size. Reserve their exact serialized
            # footprint before spending work on a complete weights output.
            planned.artifact_contract["component_hashes"] = {name: "0" * 64 for name in COMPONENT_FILES}
            (staging / "config.json").write_bytes(family_config_bytes(root_config))
            validate_family_output_header(planned.tensors)
        elif gemma_image:
            planned.artifact_contract["component_hashes"] = {name: "0" * 64 for name in COMPONENT_FILES}
            (staging / "config.json").write_bytes(family_config_bytes(root_config))
        else:
            (staging / "config.json").write_text(json.dumps(root_config, indent=4), encoding="utf-8")
        stats = write_plan(
            planned.tensors,
            staging / "model.safetensors",
            chunk_bytes=chunk_bytes,
            **({"progress": progress} if progress is not None else {}),
        )
        if family or gemma_image:
            planned.artifact_contract["component_hashes"] = {
                name: file_hash(staging / name) for name in COMPONENT_FILES
            }
            if planned.artifact_contract["component_hashes"]["model.safetensors"] != stats["output_sha256"]:
                raise ConversionError("Vision family weights changed during conversion")
            (staging / "config.json").write_bytes(family_config_bytes(root_config))
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

        if artifact_mode == "vision-language" and planned.artifact_contract["schema_version"] == 2:
            from .vision_runtime import image_qualification_for

            qualification = image_qualification_for(
                repo,
                dtype,
                {
                    "artifact_sha256": stats["output_sha256"],
                    "config_sha256": artifact_hashes["config_sha256"],
                    "tokenizer_sha256": artifact_hashes["tokenizer_sha256"],
                    "processor_sha256": file_hash(staging / "preprocessor_config.json"),
                    "template_sha256": file_hash(staging / "chat_template.jinja"),
                },
            )
        elif artifact_mode == "vision-language":
            qualification = None
        else:
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
            "architecture": planned.capability.adapter,
            "metadata_profile": planned.request.metadata_profile or entry["metadata_profile"],
            "catalogued": get_model(repo) is not None,
            "artifact_contract": planned.artifact_contract
            or {
                "schema_version": 1,
                "input_modalities": ["text"],
                **({"transport": "text-generation-v1"} if request.model_kind == "language" else {}),
            },
        }
        (staging / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        snapshot.assert_unchanged()
        if provenance != toolchain_provenance():
            raise ConversionError("Converter tooling changed during qualification")
        _publish(staging, output)
        if progress is not None:
            progress.emit("conversion", "complete", stats["output_bytes"], stats["output_bytes"], force=True)
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
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.add_argument(
        "source", nargs="?", help="HF repository ID, full repository URL or local checkpoint directory"
    )
    parser.add_argument(
        "--repo", help="Catalog identity for a local checkpoint; omit with --metadata-profile hf-generic"
    )
    parser.add_argument("--revision", help="HF revision resolved to an immutable commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("auto", "streaming"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--model-kind", choices=("language", "classifier", "tts"))
    parser.add_argument(
        "--artifact-mode",
        choices=("text", "vision-language"),
        default="text",
        help="Full vision artifact conversion; image execution requires an exact native qualification record",
    )
    parser.add_argument(
        "--architecture", choices=sorted(set(CONFIG_ADAPTERS.values())), help="Assert the checkpoint architecture"
    )
    parser.add_argument(
        "--metadata-profile",
        choices=(GENERIC_PROFILE,),
        help="Use explicit plain self-contained metadata for an uncatalogued checkpoint",
    )
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--capabilities", action="store_true", help="Probe catalog coverage without reading a source")
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List structural family coverage separately from checkpoint qualification",
    )
    parser.add_argument("--family-models", help="Browse compatible model-family downloads without fetching weights")
    parser.add_argument(
        "--preflight", action="store_true", help="Check bounded model metadata without downloading weights"
    )
    parser.add_argument("--support-mode", choices=MODES, help="Check the requested app mode before weight downloads")
    parser.add_argument(
        "--require-native-support",
        action="store_true",
        help="Require an exact stored native qualification for the output and its named Stark build",
    )
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--write-source-manifest", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--progress-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    progress_sink = None
    progress = None
    try:
        if sum((args.capabilities, args.list_families, args.preflight, args.family_models is not None)) > 1:
            raise ValueError("conflicting_inspection_actions")
        if args.family_models is not None:
            from .model_catalog import family_catalog

            logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            print(json.dumps(family_catalog(args.family_models, offline=args.offline), sort_keys=True))
            return 0
        if args.list_families:
            print(json.dumps(support_matrix(), sort_keys=True))
            return 0
        if args.progress_fd is not None:
            progress_sink = FDProgressSink(args.progress_fd)
            progress = ProgressReporter(progress_sink)
            progress.emit("download", "waiting", force=True)
            progress.emit("conversion", "waiting", force=True)
        repo = args.repo or (args.source if args.source and not Path(args.source).expanduser().is_dir() else None)
        if repo is not None:
            repo = huggingface_repository(repo)
        if args.preflight:
            if not args.source:
                raise ValueError("missing_arguments")
            logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            result = preflight_source(
                args.source,
                repo=repo,
                revision=args.revision,
                mode=args.support_mode or args.artifact_mode,
                dtype=args.dtype,
                model_kind=args.model_kind,
                architecture=args.architecture,
                offline=args.offline,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.support_mode is not None and (args.support_mode == "vision-language") != (
            args.artifact_mode == "vision-language"
        ):
            raise ValueError("support_mode_artifact_mode_mismatch")
        model_kind = args.model_kind or "language"
        request = ConversionRequest(
            repo,
            args.dtype,
            model_kind,
            args.backend,
            args.chunk_mib * 1024 * 1024,
            args.require_native_support,
            args.architecture,
            args.metadata_profile,
            args.artifact_mode,
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
        if (
            Path(args.source).expanduser().is_dir()
            and repo is None
            and args.artifact_mode == "text"
            and not (args.metadata_profile or args.source_manifest)
        ):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_category": "local_profile_required",
                        "hint": "Supply --repo for a catalog model or --metadata-profile hf-generic.",
                    }
                ),
                file=sys.stderr,
            )
            return 1
        # Replay infers identity from its manifest. Native qualification requests
        # still require named-build evidence before any checkpoint reads.
        if not args.source_manifest or request.artifact_mode == "vision-language" and request.require_native_support:
            _validate_request(request)
        logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        snapshot = resolve_source(
            args.source,
            repo=repo,
            revision=args.revision,
            source_manifest=args.source_manifest,
            offline=args.offline,
            architecture=args.architecture,
            metadata_profile=args.metadata_profile,
            artifact_mode=args.artifact_mode,
            support_mode=args.support_mode,
            dtype=args.dtype,
            model_kind=model_kind,
            **({"progress": progress} if progress is not None else {}),
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
            model_kind=model_kind,
            architecture=args.architecture,
            metadata_profile=args.metadata_profile,
            artifact_mode=args.artifact_mode,
            **({"progress": progress} if progress is not None else {}),
        )
    except BaseException as error:
        category = "interrupted" if isinstance(error, KeyboardInterrupt) else type(error).__name__
        print(json.dumps({"status": "error", "error_category": category}), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if progress_sink is not None:
            progress_sink.close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
