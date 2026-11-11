"""Shared planner selection for conversion and header-only catalog admission."""

from functools import partial
from pathlib import Path

from .checkpoint_metadata import CheckpointMetadata
from .streaming import UnsupportedConversion


def planner_for(family: str, *, mode: str = "text"):
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
        build_qwen38_plan,
        build_smollm3_plan,
    )
    from .adapters.gemma3_vision import build_gemma3_vision_plan
    from .adapters.gemma4_vision import build_gemma4_vision_plan
    from .qwen3 import build_qwen3_plan

    planners = {
        "qwen3": build_qwen3_plan,
        "qwen2": build_qwen2_plan,
        "llama": build_llama_plan,
        "mistral": build_mistral_plan,
        "gemma2": partial(build_gemma_plan, family="gemma2"),
        "gemma3_text": partial(build_gemma_plan, family="gemma3_text"),
        "gemma3": build_gemma3_vision_plan,
        "gemma4_text": build_gemma4_vision_plan if mode == "vision-language" else build_gemma4_plan,
        "olmo3": build_olmo3_plan,
        "phi3": build_phi_plan,
        "granite": build_granite_plan,
        "smollm3": build_smollm3_plan,
        "qwen3_5": partial(build_qwen35_plan, include_vision=mode == "vision-language"),
        "qwen3_5_moe": partial(build_qwen35_plan, include_vision=mode == "vision-language"),
        "qwen4_exp": build_qwen38_plan,
        "lfm2": build_lfm2_plan,
        "llamba": build_llamba_plan,
        "modernbert": build_modernbert_plan,
    }
    if family not in planners:
        raise UnsupportedConversion("This model family has no implemented checkpoint planner")
    return planners[family]


def plan_from_headers(family: str, metadata: CheckpointMetadata, *, mode: str = "text"):
    """Reuse full layout rules while deferring duplicate payload byte equality.

    Paths in the supplied TensorInfo records identify remote shards and are
    never opened. Processing metadata and artifact identity are checked by the
    catalogue before this tensor-storage check grants a download.
    """
    if family == "modernbert":
        raise UnsupportedConversion("The app has no download workflow for classifier models")
    return planner_for(family, mode=mode)(Path("."), checkpoint_metadata=metadata)
