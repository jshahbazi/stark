"""Dense Mistral profiles, including full and sliding attention."""

from pathlib import Path

from ..streaming import ConversionPlan
from .dense import build_dense_plan


def build_mistral_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    return build_dense_plan(source, family="mistral", dtype=dtype)
