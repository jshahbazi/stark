"""Dense Mistral profiles, including full and sliding attention."""

from pathlib import Path

from ..checkpoint_metadata import CheckpointMetadata
from ..streaming import ConversionPlan
from .dense import build_dense_plan


def build_mistral_plan(
    source: Path, *, dtype: str = "bfloat16", checkpoint_metadata: CheckpointMetadata | None = None
) -> ConversionPlan:
    return build_dense_plan(source, family="mistral", dtype=dtype, checkpoint_metadata=checkpoint_metadata)
