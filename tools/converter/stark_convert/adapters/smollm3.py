"""Independent SmolLM3 layout and per-layer RoPE-mask adapter.

Adapted from Lalamo revision 831accb8252a6b6606c378048caeb44178f1a050,
MIT; Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo.
"""

from pathlib import Path

from ..streaming import ConversionPlan
from .granite import _build_plan


def build_smollm3_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Validate every tensor and preserve the pinned per-layer mask semantics."""
    return _build_plan(source, family="smollm3", dtype=dtype)
