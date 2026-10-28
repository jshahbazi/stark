"""Structural family inventory and bounded metadata-first admission.

Family coverage describes implemented code paths. Tensor validation, native
execution, checkpoint qualification and Decision calibration remain separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .streaming import ConversionError, UnsupportedConversion, authorize_source_path, open_regular

SCHEMA_VERSION = 1
MODES = ("text", "decision", "vision-language")
_QUALIFICATION = {"status": "not_assessed", "scope": "checkpoint_and_build", "calibration": "not_assessed"}
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MLX = {"qwen3", "llama", "gemma3_text", "gemma4_text", "olmo3", "smollm3", "qwen3_5", "lfm2", "llamba"}
_FAMILY_DEFINITIONS = (
    ("qwen3", "Qwen3 dense", "HFQwen3Config", ()),
    ("qwen2", "Qwen2 dense", "HFQwen2Config", ()),
    ("llama", "Llama dense", "HFLlamaConfig", ()),
    ("mistral", "Mistral dense", "HFMistralConfig", ()),
    ("gemma2", "Gemma 2 dense", "HFGemma2Config", ()),
    ("gemma3_text", "Gemma 3 text", "HFGemma3TextConfig", ()),
    ("gemma4_text", "Gemma 4 dense PLE", "HFGemma4TextConfig", ("gemma4",)),
    ("olmo3", "OLMo 3 dense", "HFOlmo3Config", ()),
    ("phi3", "Phi-3 dense", "HFPhi3Config", ()),
    ("granite", "Granite dense", "HFGraniteConfig", ()),
    ("smollm3", "SmolLM3 dense", "HFSmolLM3Config", ()),
    ("qwen3_5", "Qwen3.5 dense", "HFQwen35Config", ("qwen3_5_text",)),
    ("lfm2", "LFM2", "HFLFM2Config", ()),
    ("llamba", "LLaMBA", "HFLlambaConfig", ()),
    ("modernbert", "ModernBERT classifier", "ModernBERTConfig", ()),
)
CONFIG_ADAPTERS = {config: architecture for architecture, _name, config, _aliases in _FAMILY_DEFINITIONS}
ARCHITECTURE_ALIASES = {
    alias: architecture for architecture, _name, _config, aliases in _FAMILY_DEFINITIONS for alias in aliases
}


def _family(architecture: str, name: str, config_class: str, aliases: tuple[str, ...]) -> dict:
    classifier = architecture == "modernbert"
    modes = {}
    for mode in MODES:
        supported = not classifier and (mode != "vision-language" or architecture in {"qwen3_5", "gemma4_text"})
        if classifier:
            message = "Classifier conversion is implemented; the app has no text, Decision or image workflow for it."
        elif mode == "vision-language" and not supported:
            message = "Image input requires a supported native vision family and BF16 artifact."
        elif mode == "decision":
            message = "Decision scoring needs a direct-answer template, single-token labels and checkpoint checks."
        elif mode == "vision-language" and architecture == "gemma4_text":
            message = "Image input requires the pinned Gemma 4 E2B BF16 image artifact and local execution checks."
        elif mode == "vision-language":
            message = "Image input requires dense BF16 tensors, reviewed processing metadata and local image checks."
        else:
            message = "Text generation requires supported tensors, metadata and a successful native load."
        caveats = ["tensor_and_operation_validation", "checkpoint_specific_execution"] if supported else []
        if supported and mode == "decision":
            caveats.extend(["direct_answer_template", "unique_single_token_labels", "calibration_not_inherited"])
        if supported and mode == "vision-language" and architecture == "qwen3_5":
            caveats.extend(["qwen35_dense_bf16_geometry", "reviewed_processing_programs", "local_image_checks"])
        if supported and mode == "vision-language" and architecture == "gemma4_text":
            caveats.extend(["gemma4_e2b_bf16_geometry", "pinned_processing_metadata", "local_image_checks"])
        if supported and architecture in {"phi3", "gemma2", "gemma3_text", "gemma4_text"}:
            caveats.append("tied_embeddings_required")
        if supported and architecture == "gemma4_text" and mode != "vision-language":
            caveats.extend(["dense_ple_text_only", "image_and_audio_towers_excluded"])
        modes[mode] = {"status": "conditional" if supported else "unsupported", "message": message, "caveats": caveats}
    formats = ["safetensors-float"]
    if architecture in _MLX:
        formats.extend(["mlx-affine-4bit", "mlx-affine-8bit"])
    return {
        "family_id": architecture,
        "name": name,
        "architecture_aliases": [architecture, *aliases],
        "config_classes": [config_class],
        "model_kind": "classifier" if classifier else "language",
        "conversion": {
            "status": "implemented",
            "containers": [".safetensors"],
            "source_formats": formats,
            "output_dtypes": ["bfloat16", "float32"]
            if architecture == "olmo3"
            else ["bfloat16", "float16", "float32"],
        },
        "modes": modes,
    }


_FAMILIES = tuple(_family(*definition) for definition in _FAMILY_DEFINITIONS)


class MetadataSizeError(ConversionError):
    """Selected metadata exceeds a declared bounded inspection budget."""


def support_matrix() -> dict:
    """Return a fresh displayable inventory without granting model capability."""
    return {
        "schema_version": SCHEMA_VERSION,
        "modes": list(MODES),
        "families": deepcopy(list(_FAMILIES)),
        "qualification": dict(_QUALIFICATION),
    }


def normalize_architecture(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = ARCHITECTURE_ALIASES.get(value, value)
    return normalized if normalized in CONFIG_ADAPTERS.values() else None


def _report(family: dict | None, mode: str | None, status: str, reason: str, message: str, warnings=()) -> dict:
    modes = {
        name: family["modes"][name]["status"]
        if family
        else "inconclusive"
        if status == "inconclusive"
        else "unsupported"
        for name in MODES
    }
    if mode in MODES and status != "conditional":
        modes[mode] = status
    mode_specific = {
        "unsupported_mode",
        "vision_tower_missing",
        "unsupported_vision_wrapper",
        "unsupported_vision_geometry",
        "unsupported_processing_metadata",
        "unsupported_vision_weight_format",
    }
    if status == "unsupported" and reason not in mode_specific:
        modes = dict.fromkeys(MODES, "unsupported")
    if status == "inconclusive" and not reason.startswith(("vision_", "image_")):
        modes = dict.fromkeys(MODES, "inconclusive")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "family_id": family["family_id"] if family else None,
        "family_name": family["name"] if family else None,
        "architecture": family["family_id"] if family else None,
        "mode": mode,
        "model_kind": family["model_kind"] if family else None,
        "message": message,
        "reasons": [reason],
        "warnings": list(warnings),
        "modes": modes,
        "alternatives": [name for name in MODES if name != mode and modes[name] == "conditional"],
        "pending_checks": [
            "tensor_headers",
            "tensor_operations",
            "tokenizer_and_template",
            "native_loading",
            "local_execution",
        ]
        if status == "conditional"
        else [],
        "qualification": dict(_QUALIFICATION),
    }


def preflight_model(
    config: dict | None,
    *,
    mode: str | None = None,
    dtype: str = "bfloat16",
    model_kind: str | None = None,
    expected_architecture: str | None = None,
    container: str = ".safetensors",
) -> dict:
    """Reject metadata-proven exclusions; recognized models remain conditional."""
    if mode is not None and mode not in MODES:
        return _report(
            None, mode, "unsupported", "unsupported_mode", "Choose text, Decision or vision-language support."
        )
    if not isinstance(config, dict):
        return _report(
            None,
            mode,
            "inconclusive",
            "metadata_unavailable",
            "Model metadata is unavailable; support has not been established.",
        )
    if config.get("type") in (
        "LanguageModelConfig",
        "Qwen35VisionLanguageModelConfig",
        "Gemma4VisionLanguageModelConfig",
        "ClassifierModelConfig",
    ):
        return _report(
            None,
            mode,
            "inconclusive",
            "converted_artifact_requires_native_admission",
            "Converted Stark models require native admission and their recorded checks.",
        )
    raw_architecture = config.get("model_type")
    if not isinstance(raw_architecture, str) or not raw_architecture:
        return _report(
            None,
            mode,
            "inconclusive",
            "missing_architecture_metadata",
            "Model metadata does not identify an architecture.",
        )
    architecture = normalize_architecture(raw_architecture)
    if architecture is None:
        return _report(
            None,
            mode,
            "unsupported",
            "unsupported_architecture",
            "This model architecture has no implemented converter.",
        )
    family = next(value for value in _FAMILIES if value["family_id"] == architecture)
    if expected_architecture is not None and normalize_architecture(expected_architecture) != architecture:
        return _report(
            family,
            mode,
            "unsupported",
            "architecture_mismatch",
            "Model metadata disagrees with the requested architecture.",
        )
    if model_kind is not None and model_kind != family["model_kind"]:
        return _report(
            family,
            mode,
            "unsupported",
            "model_kind_mismatch",
            "Model metadata disagrees with the requested model kind.",
        )
    if container not in family["conversion"]["containers"]:
        return _report(
            family,
            mode,
            "unsupported",
            "unsupported_container",
            "This checkpoint container has no implemented conversion path.",
        )
    if dtype not in family["conversion"]["output_dtypes"]:
        return _report(
            family, mode, "unsupported", "unsupported_output_dtype", "The requested output dtype is not implemented."
        )
    if mode is not None and dtype != "bfloat16":
        return _report(
            family,
            mode,
            "unsupported",
            "unsupported_native_dtype",
            "Native app modes require BF16 output; floating-point sources can be converted to BF16.",
        )
    if architecture in {"phi3", "gemma2", "gemma3_text"} and config.get("tie_word_embeddings") is False:
        return _report(
            family,
            mode,
            "unsupported",
            "unsupported_embedding_layout",
            "This family currently implements tied embeddings only.",
        )
    if architecture not in {"qwen3_5", "gemma4_text"} and "text_config" in config:
        return _report(
            family,
            mode,
            "unsupported",
            "unsupported_nested_architecture",
            "This family does not implement a nested text wrapper.",
        )
    text = config.get("text_config", config) if architecture in {"qwen3_5", "gemma4_text"} else config
    if not isinstance(text, dict):
        return _report(family, mode, "unsupported", "invalid_model_metadata", "Nested model metadata is invalid.")
    if architecture == "qwen3_5":
        if text is not config and text.get("model_type") not in (None, "qwen3_5", "qwen3_5_text"):
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_nested_architecture",
                "The nested decoder architecture is outside implemented dense coverage.",
            )
        text = {**text, **{key: value for key, value in config.items() if key != "text_config"}}
    elif architecture == "gemma4_text":
        if text is not config and text.get("model_type") not in (None, "gemma4_text"):
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_nested_architecture",
                "The nested decoder architecture is outside implemented Gemma 4 text coverage.",
            )
        text = {**text, **{key: value for key, value in config.items() if key != "text_config"}}
        if text.get("tie_word_embeddings") is False:
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_embedding_layout",
                "Gemma 4 text currently requires tied embeddings.",
            )
        if text.get("enable_moe_block") is True:
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_moe",
                "Gemma 4 MoE operations are outside implemented dense PLE coverage.",
            )
        if text.get("hidden_size_per_layer_input") == 0:
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_native_geometry",
                "Gemma 4 text currently implements the dense PLE E2B/E4B layouts.",
            )
    annotation = text.get("auto_map")
    phi_annotation = {
        "AutoConfig": "configuration_phi3.Phi3Config",
        "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
        "AutoTokenizer": "Xenova/gpt-4o",
    }
    known_annotation = (
        architecture == "phi3"
        and isinstance(annotation, dict)
        and all(key in phi_annotation and value == phi_annotation[key] for key, value in annotation.items())
    )
    if annotation is not None and not known_annotation:
        return _report(
            family,
            mode,
            "unsupported",
            "custom_model_code",
            "Custom model code is outside implemented conversion coverage.",
        )
    if any(
        value not in (None, 0)
        for key in (
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "n_routed_experts",
        )
        for value in [text.get(key)]
    ):
        return _report(
            family,
            mode,
            "unsupported",
            "unsupported_moe",
            "Mixture-of-experts operations are outside implemented dense coverage.",
        )
    quantized = any(text.get(key) is not None for key in ("quantization", "quantization_config"))
    if quantized:
        if architecture not in _MLX:
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_weight_format",
                "This family and mode do not implement the declared quantized representation.",
            )
        try:
            from .compressed import parse_mlx_quantization

            parse_mlx_quantization(text)
        except (ConversionError, UnsupportedConversion):
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_weight_format",
                "Only supported MLX affine 4/8-bit descriptions can be repacked.",
            )
    warnings = []
    if config.get("vision_config") is not None and mode in {None, "text", "decision"}:
        warnings.append("Text conversion omits this checkpoint's vision tower.")
    if architecture == "gemma4_text" and config.get("audio_config") is not None and mode in {None, "text", "decision"}:
        warnings.append("Text conversion omits this checkpoint's audio tower.")
    if mode is not None and architecture == "qwen3_5":
        from .vision_family import _DELTA_HEAD_DIM

        if any(
            text.get(key, _DELTA_HEAD_DIM) != _DELTA_HEAD_DIM
            for key in ("linear_key_head_dim", "linear_value_head_dim")
        ):
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_native_geometry",
                "The native DeltaNet path requires 128-element key and value heads.",
            )
    # A mode-specific exclusion cannot hide metadata that rules out the source
    # in every mode, or suggest an alternative already known to be impossible.
    if mode is not None and family["modes"][mode]["status"] == "unsupported":
        return _report(family, mode, "unsupported", "unsupported_mode", family["modes"][mode]["message"])
    if quantized and mode == "vision-language":
        return _report(
            family,
            mode,
            "unsupported",
            "unsupported_vision_weight_format",
            "Image input requires merged BF16 weights; the declared MLX format is limited to text paths.",
        )
    if mode == "vision-language" and architecture == "gemma4_text":
        from .adapters.gemma4_vision import GEOMETRY, TOKENS

        vision = config.get("vision_config")
        if not isinstance(vision, dict):
            return _report(family, mode, "unsupported", "vision_tower_missing", "This checkpoint has no vision tower.")
        if raw_architecture != "gemma4" or not isinstance(config.get("text_config"), dict):
            return _report(
                family, mode, "unsupported", "unsupported_vision_wrapper", "Gemma images require the full E2B wrapper."
            )
        if (
            any(vision.get(key) != value for key, value in GEOMETRY.items())
            or text.get("hidden_size") != 1536
            or text.get("num_hidden_layers") != 35
            or any(
                config.get(key) != TOKENS[token]
                for key, token in (
                    ("image_token_id", "image"),
                    ("boi_token_id", "vision_start"),
                    ("eoi_token_id", "vision_end"),
                    ("video_token_id", "video"),
                    ("audio_token_id", "audio"),
                )
            )
        ):
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_vision_geometry",
                "Native Gemma images require the pinned E2B BF16 vision and decoder geometry.",
            )
    elif mode == "vision-language":
        if not isinstance(config.get("vision_config"), dict):
            report = _report(
                family, mode, "unsupported", "vision_tower_missing", "This checkpoint does not declare a vision tower."
            )
            report["modes"]["vision-language"] = "unsupported"
            return report
        if not isinstance(config.get("text_config"), dict) or config.get("architectures") is None:
            return _report(
                family,
                mode,
                "inconclusive",
                "vision_metadata_incomplete",
                "The full vision wrapper metadata is incomplete.",
            )
        if raw_architecture != "qwen3_5" or config["architectures"] != ["Qwen3_5ForConditionalGeneration"]:
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_vision_wrapper",
                "This model wrapper is outside the supported vision family.",
            )
        from .vision_family import _TEXT_BOUNDS, _VISION_BOUNDS, validate_family_geometry

        required_text = set(_TEXT_BOUNDS) | {
            "head_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "tie_word_embeddings",
            "rms_norm_eps",
            "layer_types",
            "rope_parameters",
        }
        required_vision = set(_VISION_BOUNDS) | {
            "model_type",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "out_hidden_size",
        }
        merged_text = {**text, **{key: value for key, value in config.items() if key != "text_config"}}
        if not required_text <= set(merged_text):
            return _report(
                family,
                mode,
                "inconclusive",
                "shared_geometry_incomplete",
                "Decoder geometry metadata is incomplete; support has not been established.",
            )
        if not required_vision <= set(config["vision_config"]):
            return _report(
                family,
                mode,
                "inconclusive",
                "vision_metadata_incomplete",
                "Vision geometry metadata is incomplete; support has not been established.",
            )
        try:
            validate_family_geometry(config)
        except (ConversionError, UnsupportedConversion, KeyError, TypeError, ValueError, OverflowError):
            return _report(
                family,
                mode,
                "unsupported",
                "unsupported_vision_geometry",
                "This checkpoint declares vision operations or geometry outside the implemented family.",
            )
    report = _report(
        family,
        mode,
        "conditional",
        "structural_path_implemented",
        "The model family has an implemented path; tensors, metadata and local execution still need checks.",
        warnings,
    )
    if mode == "decision":
        report["warnings"].append("Family support does not establish calibrated Decision probabilities.")
        report["pending_checks"].extend(["decision_template", "single_token_labels", "checkpoint_calibration"])
    return report


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError()
        value[key] = item
    return value


def read_metadata_bytes(path: Path, *, root: Path, maximum: int = 1024 * 1024) -> bytes:
    """Bound reads under the same local/HF-cache source authority as conversion."""
    try:
        selected = authorize_source_path(root, path)
        with open_regular(selected) as stream:
            before = os.fstat(stream.fileno())
            if before.st_size > maximum:
                raise MetadataSizeError("Model metadata exceeds its size limit")
            if not 0 < before.st_size <= maximum:
                raise ValueError()
            data = stream.read(before.st_size + 1)
            after = os.fstat(stream.fileno())
            if len(data) != before.st_size or any(
                getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            ):
                raise ValueError()
            return data
    except (OSError, ValueError):
        raise ConversionError("Model metadata is unavailable or exceeds its size limit") from None


def read_support_config(path: Path, *, root: Path) -> dict:
    try:
        value = json.loads(
            read_metadata_bytes(path, root=root),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ConversionError("Model metadata is invalid") from None


def _source_info(repo, revision=None):
    return {"repo": repo, "revision": revision, "metadata_sha256": {}, "metadata_sources": []}


def _metadata_root(path: Path, filename: str) -> Path:
    root = path
    for _part in Path(filename).parts:
        root = root.parent
    return root


def check_hub_metadata_size(hub: Any, repo: str, filename: str, revision: str, maximum: int) -> None:
    """HEAD only selected metadata before allowing its payload fetch."""
    try:
        metadata = hub.get_hf_file_metadata(hub.hf_hub_url(repo_id=repo, filename=filename, revision=revision))
        if type(metadata.size) is not int or metadata.size <= 0:
            raise ValueError()
    except Exception:
        raise ConversionError("Model metadata size is unavailable or exceeds its size limit") from None
    if metadata.size > maximum:
        raise MetadataSizeError("Model metadata exceeds its size limit")


def check_vision_processing_metadata(names, read) -> None:
    """Validate the selected small vision sidecars without fetching tensors."""
    from .vision_family import (
        _MAPPING_TEMPLATE_SHA256,
        _TEMPLATE_SHA256,
        _THINKING_TEMPLATE_SHA256,
        family_processor_filenames,
        normalize_family_processor,
    )

    processors = []
    for name in family_processor_filenames(names):
        value = json.loads(read(name, 65536), object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise ConversionError("Vision processing metadata is invalid")
        processors.append(normalize_family_processor(value, name))
    if any(processor != processors[0] for processor in processors[1:]):
        raise UnsupportedConversion("Vision processor metadata disagrees")
    template = read("chat_template.jinja", 262144)
    if hashlib.sha256(template).hexdigest() not in {
        _TEMPLATE_SHA256,
        _MAPPING_TEMPLATE_SHA256,
        _THINKING_TEMPLATE_SHA256,
    }:
        raise UnsupportedConversion("Vision family tokenizer or template is outside the implemented profile")
    tokenizer_config = json.loads(read("tokenizer_config.json", 1024 * 1024), object_pairs_hook=_unique_object)
    if not isinstance(tokenizer_config, dict):
        raise ConversionError("Vision tokenizer metadata is invalid")
    if tokenizer_config.get("auto_map") is not None:
        raise UnsupportedConversion("Vision tokenizer metadata is outside the implemented family")
    if tokenizer_config.get("chat_template") not in (None, template.decode("utf-8")):
        raise ConversionError("Vision family tokenizer or template metadata is invalid")


def preflight_source(
    value: str | Path,
    *,
    repo: str | None = None,
    revision: str | None = None,
    mode: str = "text",
    dtype: str = "bfloat16",
    model_kind: str | None = None,
    architecture: str | None = None,
    hub: Any = None,
    offline: bool = False,
) -> dict:
    """Inspect bounded metadata only; never fetch tensors or tensor byte ranges."""
    from .registry import get_model
    from .sources import huggingface_repository
    from .vision_contract import REPO as VISION_REPO
    from .vision_contract import REVISION as VISION_REVISION

    path = Path(value).expanduser()
    local = path.is_dir()
    if repo is not None:
        repo = huggingface_repository(repo)
    if not local and (path.exists() or str(value).startswith(("/", ".", "~"))):
        report = _report(
            None,
            mode,
            "inconclusive",
            "metadata_unavailable",
            "Select an available model source folder or Hugging Face repository.",
        )
        report["source"] = _source_info(repo)
        return report
    if not local:
        primary = huggingface_repository(str(value))
        if repo is not None and repo != primary:
            raise ValueError("conflicting_repository")
        repo = primary
    source = _source_info(repo, revision if local else None)
    selected_family = None
    selected_metadata = "config.json"

    def inconclusive(reason, message):
        report = _report(selected_family, mode, "inconclusive", reason, message)
        report["source"] = source
        return report

    if local and revision is not None and (not isinstance(revision, str) or _REVISION.fullmatch(revision) is None):
        return inconclusive(
            "metadata_revision_unavailable", "A local source revision must be immutable when supplied."
        )

    if mode not in MODES:
        report = preflight_model(None, mode=mode)
        report["source"] = source
        return report
    if not local and offline:
        return inconclusive(
            "offline_metadata_unavailable", "Remote metadata is unavailable offline; support has not been established."
        )
    entry = get_model(repo) if repo else None
    if mode == "vision-language":
        entry = None
        if repo == VISION_REPO:
            if revision not in (None, VISION_REVISION):
                report = _report(
                    None,
                    mode,
                    "unsupported",
                    "unsupported_pinned_revision",
                    "The exact official image path requires its recorded source revision.",
                )
                report["source"] = source
                return report
            revision = VISION_REVISION
    config_rule = entry["files"].get("model_config") if entry else None
    config_rule = config_rule or {"repo": repo, "filename": "config.json"}
    repositories = {repo, config_rule["repo"]}
    revisions = {repo: revision} if local else {}
    try:
        if (not local or repositories != {repo}) and hub is None:
            import huggingface_hub

            hub = huggingface_hub
        for repository in sorted(repositories - ({repo} if local else set())):
            if offline:
                return inconclusive(
                    "offline_metadata_unavailable",
                    "External metadata is unavailable offline; support has not been established.",
                )
            selected_revision = (
                hub.HfApi().model_info(repo_id=repository, revision=revision if repository == repo else None).sha
            )
            if not isinstance(selected_revision, str) or _REVISION.fullmatch(selected_revision) is None:
                return inconclusive(
                    "metadata_revision_unavailable", "An immutable metadata revision could not be established."
                )
            if (
                repository == repo
                and isinstance(revision, str)
                and _REVISION.fullmatch(revision)
                and revision != selected_revision
            ):
                return inconclusive(
                    "metadata_revision_unavailable", "The requested immutable source revision could not be verified."
                )
            revisions[repository] = selected_revision
        source["revision"] = revisions[repo]
        names = (
            tuple(file.name for file in path.iterdir() if file.is_file())
            if local
            else tuple(hub.list_repo_files(repo_id=repo, revision=revisions[repo]))
        )
        if len(names) > 16384 or any(not isinstance(name, str) or len(name) > 1024 for name in names):
            return inconclusive(
                "metadata_inventory_limit", "The metadata inventory exceeds its supported inspection limit."
            )

        def read(rule, maximum=1024 * 1024):
            nonlocal selected_metadata
            repository, filename = rule["repo"], rule["filename"]
            selected_metadata = filename
            if local and repository == repo:
                selected, authority = path / filename, path.resolve()
            else:
                check_hub_metadata_size(hub, repository, filename, revisions[repository], maximum)
                selected = Path(
                    hub.hf_hub_download(
                        repo_id=repository, filename=filename, revision=revisions[repository], local_files_only=False
                    )
                )
                authority = _metadata_root(selected, filename)
            payload = read_metadata_bytes(selected, root=authority, maximum=maximum)
            digest = hashlib.sha256(payload).hexdigest()
            source["metadata_sha256"][filename] = digest
            source["metadata_sources"].append(
                {
                    "repo": repository,
                    "revision": revisions[repository],
                    "filename": filename,
                    "sha256": digest,
                    "bytes": len(payload),
                }
            )
            return payload

        payload = read(config_rule)
        config = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(config, dict):
            return inconclusive(
                "invalid_model_metadata", "Model metadata is invalid; support has not been established."
            )
        if any(name.endswith(".safetensors") or name == "model.safetensors.index.json" for name in names):
            container = ".safetensors"
        elif any(name.endswith((".bin", ".pth", ".pt")) for name in names):
            container = ".pth"
        elif any(name.endswith(".gguf") for name in names):
            container = ".gguf"
        elif any(name.endswith(".nemo") for name in names):
            container = ".nemo"
        else:
            container = None
        report = preflight_model(
            config,
            mode=mode,
            dtype=dtype,
            model_kind=model_kind,
            expected_architecture=architecture,
            container=container or ".safetensors",
        )
        selected_family = next((family for family in _FAMILIES if family["family_id"] == report["family_id"]), None)
        if report["status"] == "conditional" and container is None:
            report = inconclusive(
                "checkpoint_inventory_incomplete", "Metadata does not list a supported checkpoint container."
            )
        if report["status"] == "conditional" and mode == "vision-language":
            if report["family_id"] == "gemma4_text":
                if repo != "google/gemma-4-E2B-it" or revisions[repo] != "3e22461f65e89153144f8adb70e3b8c2cc9845a7":
                    raise UnsupportedConversion("Gemma images require the pinned official profile")
                for filename, maximum, expected in (
                    (
                        "processor_config.json",
                        65536,
                        "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
                    ),
                    (
                        "chat_template.jinja",
                        262144,
                        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
                    ),
                ):
                    if hashlib.sha256(read({"repo": repo, "filename": filename}, maximum)).hexdigest() != expected:
                        raise UnsupportedConversion("Gemma image processing identity changed")
            else:
                check_vision_processing_metadata(
                    names, lambda filename, maximum: read({"repo": repo, "filename": filename}, maximum)
                )
        report["source"] = source
        return report
    except MetadataSizeError:
        reason = (
            "image_metadata_size_limit"
            if selected_metadata in {"preprocessor_config.json", "processor_config.json", "chat_template.jinja"}
            else "metadata_size_limit"
        )
        return inconclusive(reason, "Model metadata exceeds the supported inspection size limit.")
    except UnsupportedConversion:
        report = _report(
            selected_family,
            mode,
            "unsupported",
            "unsupported_processing_metadata",
            "This checkpoint declares processing metadata outside the implemented family.",
        )
        report["source"] = source
        return report
    except Exception:
        reason = (
            "image_metadata_access_failed"
            if selected_metadata in {"preprocessor_config.json", "processor_config.json", "chat_template.jinja"}
            else "metadata_access_failed"
        )
        return inconclusive(
            reason,
            "Model metadata could not be read. Check the source, access and network, then try again.",
        )


def require_metadata_support(config: dict, **kwargs) -> dict:
    """Resolver backstop: metadata-only results never grant checkpoint support."""
    report = preflight_model(config, **kwargs)
    if report["status"] == "unsupported":
        if report["reasons"] == ["architecture_mismatch"]:
            raise ConversionError("Checkpoint architecture disagrees with the requested architecture")
        if report["reasons"] == ["unsupported_vision_geometry"]:
            from .vision_family import validate_family_geometry

            validate_family_geometry(config)
        raise UnsupportedConversion("Model metadata is outside implemented family support")
    if report["status"] == "inconclusive":
        raise ConversionError("Cannot verify model metadata before downloading weights")
    return report
