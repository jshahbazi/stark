"""Dense Llama profiles with unscaled or Llama 3.1/3.2 rotary embeddings."""

from pathlib import Path

from ..streaming import ConversionPlan
from .dense import build_dense_plan


def build_llama_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    return build_dense_plan(source, family="llama", dtype=dtype)
