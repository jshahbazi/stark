"""Public typed contracts for independent Stark conversion."""

from dataclasses import dataclass
from typing import Literal

from .streaming import ConversionPlan

ModelKind = Literal["language", "classifier", "tts"]


@dataclass(frozen=True)
class ConversionRequest:
    repo: str | None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    model_kind: ModelKind = "language"
    backend: Literal["auto", "streaming"] = "auto"
    workspace_bytes: int = 8 * 1024 * 1024
    require_native_support: bool = False
    architecture: str | None = None
    metadata_profile: Literal["hf-generic"] | None = None


@dataclass(frozen=True)
class CapabilityResult:
    adapter: str | None
    model_kind: ModelKind
    conversion: Literal["implemented", "unsupported"]
    source_validation: Literal["not_performed", "complete"] = "not_performed"
    native_loading: Literal[
        "qualified_tuple", "supported_layout_unverified_tuple", "runtime_blocked", "unverified"
    ] = "unverified"
    execution: Literal["unverified"] = "unverified"
    reason: str | None = None


@dataclass(frozen=True)
class PlannedConversion:
    request: ConversionRequest
    capability: CapabilityResult
    tensors: ConversionPlan
    model_config: dict


@dataclass(frozen=True)
class ConversionResult:
    backend: Literal["streaming"]
    capability: CapabilityResult
    tensor_count: int
    output_bytes: int
    transform_peak_bytes: int
    total_rss_bytes: int
    output_sha256: str
    source_manifest_sha256: str
