"""Independent MLX affine checkpoint repacking and bounded parameter export.

Conventions follow Lalamo revision 831accb8252a6b6606c378048caeb44178f1a050,
MIT License, Copyright (c) 2025 Mirai Tech Inc. No quantization is performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from .streaming import (
    DTYPE_BYTES,
    ConversionError,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    _cast_values,
    _open_tensor,
)

_FLOAT_DTYPES = {"BF16", "F16", "F32"}
_GROUP_SIZES = {16, 32, 64, 128}


@dataclass(frozen=True)
class MLXQuantization:
    bits: int
    group_size: int


@dataclass(frozen=True)
class MLXMatrixSource:
    weights: TensorInfo
    scales: TensorInfo
    biases: TensorInfo
    logical_shape: tuple[int, int]

    @property
    def planes(self) -> tuple[TensorInfo, ...]:
        return self.weights, self.scales, self.biases


def parse_mlx_quantization(config: dict) -> MLXQuantization | None:
    """Accept only identical, explicitly described MLX affine representations."""
    descriptions = []
    for key in ("quantization", "quantization_config"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConversionError("MLX quantization configuration must be an object")
        if set(value) - {"bits", "group_size", "mode", "quant_method"}:
            raise UnsupportedConversion("MLX quantization configuration has unimplemented options")
        if value.get("mode", "affine") != "affine" or value.get("quant_method", "mlx") != "mlx":
            raise UnsupportedConversion("Only MLX affine checkpoint repacking is implemented")
        bits, group_size = value.get("bits"), value.get("group_size")
        if type(bits) is not int or type(group_size) is not int:
            raise ConversionError("MLX bits and group size must be integers")
        if bits not in {4, 8} or group_size not in _GROUP_SIZES:
            raise UnsupportedConversion("MLX conversion supports 4/8 bits and groups 16/32/64/128")
        descriptions.append(MLXQuantization(bits, group_size))
    if descriptions and any(value != descriptions[0] for value in descriptions):
        raise ConversionError("MLX quantization descriptions disagree")
    return descriptions[0] if descriptions else None


def read_mlx_matrix(
    tensors: dict[str, TensorInfo], name: str, shape: tuple[int, int], quantization: MLXQuantization
) -> MLXMatrixSource:
    """Validate every plane against the logical dimensions and config."""
    if not name.endswith(".weight"):
        raise ConversionError("Invalid MLX matrix source name")
    prefix = name.removesuffix(".weight")
    names = name, prefix + ".scales", prefix + ".biases"
    if any(key not in tensors for key in names):
        raise ConversionError("MLX checkpoint is missing a matrix plane")
    weights, scales, biases = (tensors[key] for key in names)
    rows, columns = shape
    if columns % quantization.group_size or columns % (32 // quantization.bits):
        raise ConversionError("MLX matrix dimensions are not divisible by its packing and groups")
    if weights.dtype not in {"I32", "U32"}:
        raise UnsupportedConversion("MLX checkpoint weights require packed I32/U32")
    if weights.shape != (rows, columns * quantization.bits // 32):
        raise ConversionError("MLX packed weight shape disagrees with its configuration")
    plane_shape = rows, columns // quantization.group_size
    if scales.shape != plane_shape or biases.shape != plane_shape:
        raise ConversionError("MLX parameter plane shape disagrees with its configuration")
    if scales.dtype not in _FLOAT_DTYPES or biases.dtype not in _FLOAT_DTYPES:
        raise UnsupportedConversion("MLX scales and biases require BF16/F16/F32")
    if scales.dtype != biases.dtype:
        raise UnsupportedConversion("Mixed-dtype MLX matrix parameter planes are outside implemented coverage")
    return MLXMatrixSource(weights, scales, biases, shape)


def plan_mlx_matrix(
    out: str,
    sources: tuple[MLXMatrixSource, ...],
    *,
    dtype: str,
    layout: str,
    quantization: MLXQuantization,
) -> tuple[TensorOperation, ...]:
    if not sources or layout not in {"input_output", "output_input"} or dtype not in _FLOAT_DTYPES:
        raise ConversionError("Invalid MLX matrix plan")
    columns = sources[0].logical_shape[1]
    if any(source.logical_shape[1] != columns for source in sources):
        raise ConversionError("MLX fused matrix columns disagree")
    if len({source.scales.dtype for source in sources}) != 1:
        raise UnsupportedConversion("Mixed-dtype MLX fusion promotion is outside implemented coverage")
    rows = sum(source.logical_shape[0] for source in sources)
    groups = columns // quantization.group_size
    operations = [
        TensorOperation(
            out + ".weights",
            tuple(source.weights for source in sources),
            "U8",
            (rows, columns * quantization.bits // 8),
            "mlx_repack",
        )
    ]
    for name in ("scales", "biases"):
        planes = tuple(getattr(source, name) for source in sources)
        if layout == "output_input":
            shape, kind = (groups, rows + (-rows % 4)), "mlx_params"
        else:
            shape = rows, groups
            kind = "fuse" if len(planes) > 1 else "copy"
            if any(plane.dtype != dtype for plane in planes):
                kind = "fuse_cast" if len(planes) > 1 else "cast"
        operations.append(TensorOperation(out + "." + name, planes, dtype, shape, kind))
    return tuple(operations)


def validate_mlx_operation(op: TensorOperation) -> None:
    """Validate new writer operations before creating any output file."""
    if op.kind not in {"mlx_repack", "mlx_params"}:
        raise ConversionError("Invalid MLX operation kind")
    if (
        not op.sources
        or op.scalar is not None
        or len(op.shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in op.shape)
        or any(len(source.shape) != 2 or any(dimension <= 0 for dimension in source.shape) for source in op.sources)
        or len({source.shape[1] for source in op.sources}) != 1
    ):
        raise ConversionError("Invalid MLX operation shape")
    rows = sum(source.shape[0] for source in op.sources)
    width = op.sources[0].shape[1]
    if op.kind == "mlx_repack":
        if (
            op.dtype != "U8"
            or any(source.dtype not in {"I32", "U32"} for source in op.sources)
            or any(source.nbytes != math.prod(source.shape) * 4 for source in op.sources)
            or op.shape != (rows, width * 4)
            or sum(source.nbytes for source in op.sources) != op.nbytes
        ):
            raise ConversionError("Invalid MLX packed weight operation")
    elif (
        op.dtype not in _FLOAT_DTYPES
        or any(source.dtype not in _FLOAT_DTYPES for source in op.sources)
        or len({source.dtype for source in op.sources}) != 1
        or op.shape != (width, rows + (-rows % 4))
        or any(source.nbytes != math.prod(source.shape) * source.itemsize for source in op.sources)
    ):
        raise ConversionError("Invalid MLX parameter plane operation")


def write_mlx_params(op: TensorOperation, output: BinaryIO, chunk_bytes: int) -> int:
    """Cast and transpose fused planes into padded group/output tiles."""
    import ml_dtypes

    validate_mlx_operation(op)
    if op.kind != "mlx_params":
        raise ConversionError("Invalid MLX parameter writer operation")
    dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
    itemsize = DTYPE_BYTES[op.dtype]
    source_size = op.sources[0].itemsize
    mask_bytes = int("F16" in {op.sources[0].dtype, op.dtype})
    # Reads retain at most an input tile plus a row. Casting reserves both its
    # output and native scratch; the transpose uses that same two-output budget.
    bytes_per_element = max(2 * source_size, source_size + 2 * itemsize + mask_bytes)
    elements = chunk_bytes // bytes_per_element
    if elements < 1:
        raise ConversionError("insufficient_transform_workspace")
    groups, padded_rows = op.shape
    tile_columns = min(groups, max(1, math.isqrt(elements)))
    tile_rows = max(1, elements // tile_columns)
    start, fused_row, peak = output.tell(), 0, 0
    for source in op.sources:
        rows = source.shape[0]
        with _open_tensor(source) as stream:
            for row in range(0, rows, tile_rows):
                height = min(tile_rows, rows - row)
                for group in range(0, groups, tile_columns):
                    width = min(tile_columns, groups - group)
                    tile = np.empty((height, width), dtype=dtypes[source.dtype])
                    for index in range(height):
                        stream.seek(source.offset + ((row + index) * groups + group) * source.itemsize)
                        chunk = stream.read(width * source.itemsize)
                        if len(chunk) != width * source.itemsize:
                            raise ConversionError("Checkpoint tensor was truncated")
                        tile[index] = np.frombuffer(chunk, dtype=dtypes[source.dtype])
                        peak = max(peak, tile.nbytes + len(chunk))
                        del chunk
                    values = _cast_values(tile.reshape(-1), source.dtype, op.dtype).reshape(height, width)
                    peak = max(peak, tile.nbytes + 2 * values.nbytes + height * width * mask_bytes)
                    transposed = np.ascontiguousarray(values.T)
                    peak = max(peak, tile.nbytes + values.nbytes + transposed.nbytes)
                    for index in range(width):
                        offset = ((group + index) * padded_rows + fused_row + row) * itemsize
                        output.seek(start + offset)
                        output.write(memoryview(transposed[index].view(f"u{itemsize}")).cast("B"))
                    del tile, values, transposed
        fused_row += rows
    padding = bytes((padded_rows - fused_row) * itemsize)
    if padding:
        for group in range(groups):
            output.seek(start + (group * padded_rows + fused_row) * itemsize)
            output.write(padding)
        peak = max(peak, len(padding))
    output.seek(start + op.nbytes)
    return peak
