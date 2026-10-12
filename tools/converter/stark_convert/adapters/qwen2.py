"""Dense Qwen2 and Qwen2.5 share the Hugging Face qwen2 layout."""

from pathlib import Path

from ..streaming import ConversionPlan
from .dense import build_dense_plan


def build_qwen2_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    return build_dense_plan(source, family="qwen2", dtype=dtype)
