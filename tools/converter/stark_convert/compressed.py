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


def _read_parameter_tile(
    source: TensorInfo,
    stream: BinaryIO,
    row: int,
    height: int,
    group: int,
    width: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Read contiguous complete rows, or a bounded band of strided rows."""
    groups = source.shape[1]
    if width == groups:
        stream.seek(source.offset + row * groups * source.itemsize)
        chunk = stream.read(height * width * source.itemsize)
        if len(chunk) != height * width * source.itemsize:
            raise ConversionError("Checkpoint tensor was truncated")
        tile = np.frombuffer(chunk, dtype=dtype).reshape(height, width)
        return tile, tile.nbytes
    tile = np.empty((height, width), dtype=dtype)
    peak = tile.nbytes
    for index in range(height):
        stream.seek(source.offset + ((row + index) * groups + group) * source.itemsize)
        chunk = stream.read(width * source.itemsize)
        if len(chunk) != width * source.itemsize:
            raise ConversionError("Checkpoint tensor was truncated")
        tile[index] = np.frombuffer(chunk, dtype=dtype)
        peak = max(peak, tile.nbytes + len(chunk))
        del chunk
    return tile, peak


def write_mlx_params(op: TensorOperation, output: BinaryIO, chunk_bytes: int) -> int:
    """Emit padded fused planes in sequential group/output order.

    Complete output group bands permit contiguous source-plane reads when the
    workspace holds all groups. Larger planes use narrower bands. If even one
    complete output group cannot fit, row tiles keep writes sequential under
    the same workspace limit. No destination seek invalidates inline hashing.
    """
    import ml_dtypes

    validate_mlx_operation(op)
    if op.kind != "mlx_params":
        raise ConversionError("Invalid MLX parameter writer operation")
    dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
    itemsize = DTYPE_BYTES[op.dtype]
    source_size = op.sources[0].itemsize
    mask_bytes = int("F16" in {op.sources[0].dtype, op.dtype})
    # Reads retain at most an input tile plus a row. Casting reserves its
    # result and native scratch, including the portable half-cast NaN mask.
    bytes_per_element = max(2 * source_size, source_size + 2 * itemsize + mask_bytes)
    elements = chunk_bytes // bytes_per_element
    if elements < 1:
        raise ConversionError("insufficient_transform_workspace")
    groups, padded_rows = op.shape
    rows = sum(source.shape[0] for source in op.sources)
    max_source_rows = max(source.shape[0] for source in op.sources)
    tile_columns = min(groups, chunk_bytes // (padded_rows * itemsize + max_source_rows * bytes_per_element))
    peak = 0
    if tile_columns:
        for group in range(0, groups, tile_columns):
            width = min(tile_columns, groups - group)
            transposed = np.empty((width, padded_rows), dtype=dtypes[op.dtype])
            transposed[:, rows:] = 0
            fused_row = 0
            for source in op.sources:
                available = chunk_bytes - transposed.nbytes
                # A narrow output band can still read complete source rows in
                # contiguous chunks. Reserve the selected-column copy as well
                # as its cast, so this path does not hide a retained full row.
                contiguous_cost = groups * source_size + width * (source_size + 2 * itemsize + mask_bytes)
                read_columns = groups if width == groups or available >= contiguous_cost else width
                row_cost = contiguous_cost if read_columns != width else width * bytes_per_element
                tile_rows = min(source.shape[0], available // row_cost)
                with _open_tensor(source) as stream:
                    for row in range(0, source.shape[0], tile_rows):
                        height = min(tile_rows, source.shape[0] - row)
                        tile, read_peak = _read_parameter_tile(
                            source,
                            stream,
                            row,
                            height,
                            0 if read_columns == groups else group,
                            read_columns,
                            dtypes[source.dtype],
                        )
                        selected = (
                            tile if read_columns == width else np.ascontiguousarray(tile[:, group : group + width])
                        )
                        selected_bytes = 0 if selected is tile else selected.nbytes
                        peak = max(peak, transposed.nbytes + read_peak)
                        values = _cast_values(selected.reshape(-1), source.dtype, op.dtype).reshape(height, width)
                        peak = max(
                            peak,
                            transposed.nbytes
                            + tile.nbytes
                            + selected_bytes
                            + 2 * values.nbytes
                            + height * width * mask_bytes,
                        )
                        transposed[:, fused_row + row : fused_row + row + height] = values.T
                        del tile, selected, values
                fused_row += source.shape[0]
            output.write(memoryview(transposed.view(f"u{itemsize}")).cast("B"))
            del transposed
        return peak
    # A small workspace cannot retain a whole group. Each group still reaches
    # the output in order, with fusion padding written only after all sources.
    padding_bytes = (padded_rows - rows) * itemsize
    for group in range(groups):
        for source in op.sources:
            with _open_tensor(source) as stream:
                for row in range(0, source.shape[0], elements):
                    height = min(elements, source.shape[0] - row)
                    tile, read_peak = _read_parameter_tile(source, stream, row, height, group, 1, dtypes[source.dtype])
                    values = _cast_values(tile.reshape(-1), source.dtype, op.dtype)
                    peak = max(peak, read_peak, tile.nbytes + 2 * values.nbytes + height * mask_bytes)
                    output.write(memoryview(values.view(f"u{itemsize}")).cast("B"))
                    del tile, values
        if padding_bytes:
            output.write(bytes(padding_bytes))
            peak = max(peak, padding_bytes)
    return peak
