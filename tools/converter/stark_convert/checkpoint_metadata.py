"""Explicit header-only inputs for the production conversion planners.

Header admission verifies topology and storage. Duplicate tensor byte equality
remains mandatory when the ordinary source-backed conversion is planned.
"""

from dataclasses import dataclass

from .streaming import TensorInfo, tensor_sha256


@dataclass(frozen=True)
class CheckpointMetadata:
    config: dict
    tensors: dict[str, TensorInfo]


def tensor_aliases_equal(left: TensorInfo, right: TensorInfo, *, header_only: bool = False) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return header_only or tensor_sha256(left) == tensor_sha256(right)
