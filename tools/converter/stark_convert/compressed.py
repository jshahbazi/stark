"""Independent MLX affine checkpoint repacking and bounded parameter export.

Conventions follow Lalamo revision 831accb8252a6b6606c378048caeb44178f1a050,
MIT License, Copyright (c) 2025 Mirai Tech Inc. No quantization is performed.

Non-power-of-two affine packing follows the byte layout in MLX's
mlx/backend/metal/kernels/quantized.h at
ba8c005c1b687227ff79a55da5b8aa062d92541f. Native 4/8-bit words remain
byte-identical. Other supported widths are widened to U8 integer codes while
preserving their affine coefficients. No MLX implementation is imported or
executed by this converter.
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
    overrides: tuple[tuple[str, MLXQuantization | None], ...] = ()

    def for_module(self, name: str) -> MLXQuantization | None:
        """Resolve an original checkpoint module name before planning a matrix.

        MLX overrides name modules, not weight tensors. Matching is exact:
        a parent module's override does not implicitly change its children.
        ``None`` represents a module explicitly left in floating point.
        """
        return dict(self.overrides).get(name, MLXQuantization(self.bits, self.group_size))


@dataclass(frozen=True)
class MLXMatrixSource:
    weights: TensorInfo
    scales: TensorInfo
    biases: TensorInfo
    logical_shape: tuple[int, int]
    quantization: MLXQuantization | None = None

    @property
    def planes(self) -> tuple[TensorInfo, ...]:
        return self.weights, self.scales, self.biases


def parse_mlx_quantization(config: dict, *, allow_mixed: bool = False) -> MLXQuantization | None:
    """Accept only identical, explicitly described MLX affine representations."""
    fields = {"bits", "group_size", "mode", "quant_method"}

    def description(value: dict) -> MLXQuantization:
        if value.get("mode", "affine") != "affine" or value.get("quant_method", "mlx") != "mlx":
            raise UnsupportedConversion("Only MLX affine checkpoint repacking is implemented")
        bits, group_size = value.get("bits"), value.get("group_size")
        if type(bits) is not int or type(group_size) is not int:
            raise ConversionError("MLX bits and group size must be integers")
        supported_bits = {4, 5, 6, 8} if allow_mixed else {4, 8}
        if bits not in supported_bits or group_size not in _GROUP_SIZES:
            raise UnsupportedConversion("MLX conversion has an unsupported bit width or group size")
        return MLXQuantization(bits, group_size)

    descriptions = []
    for key in ("quantization", "quantization_config"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConversionError("MLX quantization configuration must be an object")
        override_names = set(value) - fields
        if override_names and not allow_mixed:
            raise UnsupportedConversion("MLX quantization configuration has unimplemented options")
        if any(not isinstance(name, str) for name in override_names):
            raise ConversionError("Invalid MLX quantization module name")
        default = description(value)
        overrides = []
        for name in sorted(override_names):
            override = value[name]
            if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
                raise ConversionError("Invalid MLX quantization module name")
            if override is False:
                overrides.append((name, None))
            elif isinstance(override, dict) and not set(override) - fields:
                overrides.append((name, description(override)))
            else:
                raise UnsupportedConversion("Invalid MLX per-module quantization description")
        descriptions.append(MLXQuantization(default.bits, default.group_size, tuple(overrides)))
    if descriptions and any(value != descriptions[0] for value in descriptions):
        raise ConversionError("MLX quantization descriptions disagree")
    return descriptions[0] if descriptions else None


def read_mlx_matrix(
    tensors: dict[str, TensorInfo], name: str, shape: tuple[int, int], quantization: MLXQuantization
) -> MLXMatrixSource:
    """Validate every plane against the logical dimensions and config."""
    if not name.endswith(".weight"):
        raise ConversionError("Invalid MLX matrix source name")
    if quantization.overrides:
        raise ConversionError("Resolve MLX per-module quantization before reading a matrix")
    prefix = name.removesuffix(".weight")
    names = name, prefix + ".scales", prefix + ".biases"
    if any(key not in tensors for key in names):
        raise ConversionError("MLX checkpoint is missing a matrix plane")
    weights, scales, biases = (tensors[key] for key in names)
    rows, columns = shape
    if quantization.bits not in {4, 5, 6, 8} or quantization.group_size not in _GROUP_SIZES:
        raise UnsupportedConversion("Unsupported MLX affine matrix representation")
    if columns % quantization.group_size or columns * quantization.bits % 32:
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
    return MLXMatrixSource(weights, scales, biases, shape, quantization)


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
    if quantization.overrides:
        raise ConversionError("Resolve MLX per-module quantization before planning a matrix")
    columns = sources[0].logical_shape[1]
    if any(source.logical_shape[1] != columns for source in sources):
        raise ConversionError("MLX fused matrix columns disagree")
    if quantization.bits not in {4, 8} or quantization.group_size not in _GROUP_SIZES:
        raise UnsupportedConversion("Unsupported MLX affine matrix representation")
    if columns % quantization.group_size or columns * quantization.bits % 32:
        raise ConversionError("MLX matrix dimensions are not divisible by its packing and groups")
    for source in sources:
        source_rows = source.logical_shape[0]
        source_quant = source.quantization or quantization
        if (
            source_quant.group_size != quantization.group_size
            or source_quant.bits not in {4, 5, 6, 8}
            or source_quant.bits > quantization.bits
            or source.weights.shape != (source_rows, columns * source_quant.bits // 32)
            or source.scales.shape != (source_rows, columns // quantization.group_size)
            or source.biases.shape != source.scales.shape
        ):
            raise ConversionError("MLX fused matrix quantization representations disagree")
    if len({source.scales.dtype for source in sources}) != 1:
        raise UnsupportedConversion("Mixed-dtype MLX fusion promotion is outside implemented coverage")
    rows = sum(source.logical_shape[0] for source in sources)
    groups = columns // quantization.group_size
    widen = any((source.quantization or quantization).bits != quantization.bits for source in sources)
    operations = [
        TensorOperation(
            out + ".weights",
            tuple(source.weights for source in sources),
            "U8",
            (rows, columns * quantization.bits // 8),
            "mlx_widen" if widen else "mlx_repack",
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
    if op.kind not in {"mlx_repack", "mlx_params", "mlx_widen"}:
        raise ConversionError("Invalid MLX operation kind")
    if (
        not op.sources
        or op.scalar is not None
        or len(op.shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in op.shape)
        or any(len(source.shape) != 2 or any(dimension <= 0 for dimension in source.shape) for source in op.sources)
        or (op.kind != "mlx_widen" and len({source.shape[1] for source in op.sources}) != 1)
    ):
        raise ConversionError("Invalid MLX operation shape")
    rows = sum(source.shape[0] for source in op.sources)
    width = op.sources[0].shape[1]
    if op.kind == "mlx_widen":
        columns = op.shape[1]
        if (
            op.dtype != "U8"
            or op.shape[0] != rows
            or columns % 8
            or any(source.dtype not in {"I32", "U32"} for source in op.sources)
            or any(source.nbytes != math.prod(source.shape) * 4 for source in op.sources)
            or any(source.shape[1] * 32 not in {columns * bits for bits in (4, 5, 6, 8)} for source in op.sources)
        ):
            raise ConversionError("Invalid MLX integer widening operation")
    elif op.kind == "mlx_repack":
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


def write_mlx_widen(op: TensorOperation, output: BinaryIO, chunk_bytes: int) -> int:
    """Expand packed integer codes to U8 without changing affine coefficients.

    Eight logical values occupy exactly ``bits`` bytes at every supported
    source width. Working on these aligned packs also handles codes crossing
    byte and U32 boundaries. Floating-point arithmetic is never involved.
    """
    validate_mlx_operation(op)
    if op.kind != "mlx_widen":
        raise ConversionError("Invalid MLX widening writer operation")
    columns = op.shape[1]
    peak = 0
    for source in op.sources:
        bits = source.shape[1] * 32 // columns
        # Input + result + temporary U16 arrays and output byte copy. Reserve
        # conservatively so tiny workspaces and vectorized large tiles agree.
        workspace_per_pack = bits + 32
        capacity = chunk_bytes // workspace_per_pack
        if not capacity:
            raise ConversionError("insufficient_transform_workspace")
        remaining = source.shape[0] * columns // 8
        with _open_tensor(source) as stream:
            while remaining:
                count = min(remaining, capacity)
                payload = stream.read(count * bits)
                if len(payload) != count * bits:
                    raise ConversionError("Checkpoint tensor was truncated")
                packed = np.frombuffer(payload, dtype=np.uint8).reshape(count, bits)
                codes = np.empty((count, 8), dtype=np.uint8)
                for index in range(8):
                    byte, shift = divmod(index * bits, 8)
                    values = packed[:, byte].astype(np.uint16) >> shift
                    if shift + bits > 8:
                        values |= packed[:, byte + 1].astype(np.uint16) << (8 - shift)
                    codes[:, index] = values & ((1 << bits) - 1)
                    del values
                output.write(codes.tobytes())
                peak = max(peak, count * workspace_per_pack)
                remaining -= count
                del codes, packed, payload
    return peak


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
