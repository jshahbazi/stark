"""Bit-exact reimplementation of MLX affine quantization for 8-bit codes.

This module reproduces the arithmetic of the Metal kernel ``affine_quantize``
in ``mlx/backend/metal/kernels/quantized.h`` and the reference ``affine_quantize``
in ``mlx/ops.cpp`` of https://github.com/ml-explore/mlx (MIT License,
Copyright (c) 2023 Apple Inc.), read at ``main`` resolving to commit
58d91ee81a86dd9710123ec42e7ff91f6d1d21b1. ``mx.quantize`` on the default
(GPU) device runs that Metal kernel, which is what MLX-Serve's
``tests/convert_inkling2.py`` (commit 2fe9be834a601c70ef1a597563026340f0e722fe,
Apache-2.0) used to build its published Inkling 2 Mini 8-bit pack. Nothing is
imported from MLX; this is an independent NumPy implementation.

Per row and per group of ``group_size`` consecutive input elements, in IEEE
float32 throughout:

    w_min, w_max = min(group), max(group)
    scale = max((w_max - w_min) / n_bins, 1e-7)          n_bins = 2**bits - 1
    side  = |w_min| > |w_max|
    scale = scale if side else -scale
    edge  = w_min if side else w_max
    q0    = round_half_away(edge / scale)
    scale = scale if q0 == 0 else edge / q0
    bias  = 0 if q0 == 0 else edge
    code  = min(round_half_away((w - bias) / scale), n_bins)   then uint8

The float32 ``scale`` and ``bias`` are used for the codes; only the stored
planes are rounded to the source dtype (BF16, round to nearest even). Metal's
``round`` rounds halves away from zero, unlike NumPy's ``round``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

SUPPORTED_GROUP_SIZES = (32, 64, 128)
BITS = 8
N_BINS = np.float32((1 << BITS) - 1)
EPS = np.float32(1e-7)


def round_half_away(values: np.ndarray) -> np.ndarray:
    """Round float32 values to integers, halves away from zero (Metal ``round``)."""
    values = np.asarray(values, dtype=np.float32)
    magnitude = np.abs(values)
    whole = np.floor(magnitude)
    # ``magnitude - whole`` is exact in float32, so the half test is exact.
    whole += (magnitude - whole >= np.float32(0.5)).astype(np.float32)
    return np.copysign(whole, values).astype(np.float32, copy=False)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen raw BF16 bit patterns (uint16) to float32 exactly."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 bit patterns, to nearest with ties to even."""
    raw = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    if np.any((raw & np.uint32(0x7F800000)) == np.uint32(0x7F800000)):
        raise ValueError("Non-finite value in affine quantization parameters")
    lsb = (raw >> np.uint32(16)) & np.uint32(1)
    return ((raw + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


@dataclass(frozen=True)
class AffineRows:
    """Quantized rows: ``codes`` uint8 [rows, in]; BF16 bit planes [rows, in / group]."""

    codes: np.ndarray
    scales: np.ndarray
    biases: np.ndarray

    def packed_codes(self) -> np.ndarray:
        """Codes as MLX stores them: uint32 [rows, in / 4], little-endian along input."""
        return np.ascontiguousarray(self.codes).view("<u4")


def quantize_rows(values: np.ndarray, group_size: int = 64) -> AffineRows:
    """Quantize a float32 [rows, in] block with MLX affine 8-bit semantics."""
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError("MLX affine quantization supports group sizes 32, 64 and 128")
    values = np.asarray(values)
    if values.dtype != np.float32 or values.ndim != 2:
        raise ValueError("Affine quantization expects a float32 matrix")
    rows, columns = values.shape
    if columns == 0 or columns % group_size:
        raise ValueError("Input dimension must be a positive multiple of the group size")
    if not np.all(np.isfinite(values)):
        raise ValueError("Affine quantization input must be finite")
    groups = values.reshape(rows, columns // group_size, group_size)
    w_min = groups.min(axis=-1)
    w_max = groups.max(axis=-1)
    scale = np.maximum((w_max - w_min) / N_BINS, EPS)
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale)
    edge = np.where(side, w_min, w_max)
    q0 = round_half_away(edge / scale)
    at_zero = q0 == np.float32(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        refined = edge / q0
    scale = np.where(at_zero, scale, refined).astype(np.float32, copy=False)
    bias = np.where(at_zero, np.float32(0), edge).astype(np.float32, copy=False)
    codes = round_half_away((groups - bias[..., None]) / scale[..., None])
    codes = np.minimum(codes, N_BINS)
    # The Metal kernel has no lower clamp; a rounded value below zero would be
    # undefined there. It cannot occur because ``bias`` is the edge on the
    # side that ``scale`` points away from, but enforce it rather than wrap.
    if np.any(codes < 0):
        raise ValueError("Affine quantization produced a negative code")
    return AffineRows(
        codes=codes.astype(np.uint8).reshape(rows, columns),
        scales=f32_to_bf16_bits(scale),
        biases=f32_to_bf16_bits(bias),
    )


def dequantize_rows(quantized: AffineRows, group_size: int = 64) -> np.ndarray:
    """Reconstruct float32 values as ``code * scale + bias`` with BF16 planes."""
    rows, columns = quantized.codes.shape
    codes = quantized.codes.astype(np.float32).reshape(rows, columns // group_size, group_size)
    scales = bf16_bits_to_f32(quantized.scales)[..., None]
    biases = bf16_bits_to_f32(quantized.biases)[..., None]
    return (codes * scales + biases).reshape(rows, columns)


def iter_quantized_bf16(
    read_rows,
    shape: tuple[int, int],
    group_size: int = 64,
    rows_per_chunk: int = 2048,
) -> Iterator[tuple[int, AffineRows]]:
    """Quantize a BF16 matrix chunk by chunk.

    ``read_rows(start, count)`` returns ``count * columns`` raw BF16 values as
    bytes. Yields ``(first_row, AffineRows)`` so callers never hold the full
    tensor.
    """
    rows, columns = shape
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    for start in range(0, rows, rows_per_chunk):
        count = min(rows_per_chunk, rows - start)
        raw = read_rows(start, count)
        if len(raw) != count * columns * 2:
            raise ValueError("Truncated BF16 tensor data")
        values = bf16_bits_to_f32(np.frombuffer(raw, dtype="<u2")).reshape(count, columns)
        yield start, quantize_rows(values, group_size)
