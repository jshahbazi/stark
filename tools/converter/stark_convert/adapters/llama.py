"""Dense Llama profiles with unscaled or Llama 3.1/3.2 rotary embeddings."""

from pathlib import Path

from ..checkpoint_metadata import CheckpointMetadata
from ..streaming import ConversionPlan
from .dense import build_dense_plan


def build_llama_plan(
    source: Path, *, dtype: str = "bfloat16", checkpoint_metadata: CheckpointMetadata | None = None
) -> ConversionPlan:
    return build_dense_plan(source, family="llama", dtype=dtype, checkpoint_metadata=checkpoint_metadata)
