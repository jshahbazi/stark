"""Dense Qwen2 and Qwen2.5 share the Hugging Face qwen2 layout."""

from pathlib import Path

from ..checkpoint_metadata import CheckpointMetadata
from ..streaming import ConversionPlan
from .dense import build_dense_plan


def build_qwen2_plan(
    source: Path, *, dtype: str = "bfloat16", checkpoint_metadata: CheckpointMetadata | None = None
) -> ConversionPlan:
    return build_dense_plan(source, family="qwen2", dtype=dtype, checkpoint_metadata=checkpoint_metadata)
