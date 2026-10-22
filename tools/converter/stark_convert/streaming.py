"""Bounded-memory safetensors transformations for Stark conversion.

The wire format and metadata conventions follow Lalamo, revision
831accb8252a6b6606c378048caeb44178f1a050 (MIT; Copyright (c) 2025 Mirai Tech Inc.).
This implementation does not import Lalamo, JAX, or PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .progress import ProgressCallback, ProgressReporter, reporter


class UnsupportedConversion(Exception):
    """A representation is outside independently implemented coverage."""


class ConversionError(Exception):
    """An invalid checkpoint or failed conversion must not be published."""


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
}
MAX_HEADER_BYTES = 100_000_000
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError("JSON contains duplicate keys")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    """Read a JSON object while rejecting duplicate keys and nonfinite numbers."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=lambda _: (_ for _ in ()).throw(ConversionError("Nonfinite JSON number")),
            )
    except (OSError, ValueError, UnicodeError) as exc:
        raise ConversionError("Cannot read checkpoint JSON") from exc
    if not isinstance(result, dict):
        raise ConversionError("Checkpoint JSON must be an object")
    return result


def _signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class TensorInfo:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    file_signature: tuple[int, int, int, int]

    @property
    def itemsize(self) -> int:
        return DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class TensorOperation:
    name: str
    sources: tuple[TensorInfo, ...]
    dtype: str
    shape: tuple[int, ...]
    kind: str = "copy"
    scalar: float | None = None

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class ConversionPlan:
    decoder_config: dict
    operations: tuple[TensorOperation, ...]
    metadata: dict[str, str]
    input_bytes: int
    source_tensors: int
    input_tensors: tuple[TensorInfo, ...] = ()


def read_safetensors_header(path: Path) -> tuple[dict[str, str], dict[str, TensorInfo]]:
    """Validate the complete on-disk layout without loading tensor payloads."""
    try:
        with path.open("rb") as stream:
            signature = _signature(os.fstat(stream.fileno()))
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ConversionError("Truncated safetensors header length")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= MAX_HEADER_BYTES or header_size > signature[2] - 8:
                raise ConversionError("Invalid safetensors header size")
            header_bytes = stream.read(header_size)
            if len(header_bytes) != header_size or not header_bytes.startswith(b"{"):
                raise ConversionError("Invalid safetensors JSON header")
            header = json.loads(header_bytes, object_pairs_hook=_unique_object)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ConversionError("Cannot read safetensors header") from exc
    if not isinstance(header, dict):
        raise ConversionError("Safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ConversionError("Safetensors metadata must contain strings")
    data_start = 8 + header_size
    tensors = {}
    spans = []
    for name, info in header.items():
        if not isinstance(name, str) or not name or not isinstance(info, dict):
            raise ConversionError("Invalid safetensors tensor entry")
        if set(info) != {"dtype", "shape", "data_offsets"}:
            raise ConversionError("Invalid safetensors tensor fields")
        dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise UnsupportedConversion("Checkpoint dtype is outside streaming parser coverage")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ConversionError("Invalid safetensors tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ConversionError("Invalid safetensors tensor offsets")
        start, end = offsets
        nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != nbytes or data_start + end > signature[2]:
            raise ConversionError("Safetensors tensor byte size or offset mismatch")
        tensors[name] = TensorInfo(path, dtype, tuple(shape), data_start + start, nbytes, signature)
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ConversionError("Safetensors data has gaps or overlapping tensors")
        cursor = end
    if data_start + cursor != signature[2]:
        raise ConversionError("Safetensors file has unindexed trailing data")
    return metadata, tensors


def load_checkpoint(source: Path) -> dict[str, TensorInfo]:
    """Load one safetensors checkpoint or its explicitly indexed shards."""
    source = source.resolve()
    index_path = source / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in weight_map.items()
            )
        ):
            raise ConversionError("Invalid checkpoint shard index")
        files = []
        for filename in sorted(set(weight_map.values())):
            relative = Path(filename)
            path = source / relative
            if relative.is_absolute() or ".." in relative.parts or path.suffix != ".safetensors":
                raise ConversionError("Invalid checkpoint shard path")
            # HF snapshot files commonly point to blobs outside the snapshot.
            # Restrict the index's lexical path; permit those regular symlinks.
            files.append(path)
    elif (source / "model.safetensors").exists():
        files = [source / "model.safetensors"]
    else:
        files = sorted(source.glob("*.safetensors"))
        if len(files) != 1:
            raise UnsupportedConversion("Streaming requires a safetensors checkpoint or shard index")
    result = {}
    for path in files:
        _, tensors = read_safetensors_header(path)
        for name, tensor in tensors.items():
            if name in result:
                raise ConversionError("Tensor occurs in more than one shard")
            if weight_map is not None and weight_map.get(name) != str(path.relative_to(source)):
                raise ConversionError("Checkpoint shard index disagrees with its tensor headers")
            result[name] = tensor
    if weight_map is not None and set(result) != set(weight_map):
        raise ConversionError("Checkpoint shard index references missing tensors")
    return result


def _open_tensor(tensor: TensorInfo) -> BinaryIO:
    stream = tensor.path.open("rb")
    if _signature(os.fstat(stream.fileno())) != tensor.file_signature:
        stream.close()
        raise ConversionError("Checkpoint changed during conversion")
    stream.seek(tensor.offset)
    return stream


def tensor_sha256(tensor: TensorInfo, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Hash a tensor payload without materializing it or exposing its values."""
    if chunk_bytes < 1:
        raise ConversionError("Chunk size must be positive")
    digest = hashlib.sha256()
    with _open_tensor(tensor) as stream:
        remaining = tensor.nbytes
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ConversionError("Checkpoint tensor was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
            del chunk
    return digest.hexdigest()


def _copy(tensor: TensorInfo, output: BinaryIO, chunk_bytes: int) -> int:
    peak = 0
    with _open_tensor(tensor) as stream:
        remaining = tensor.nbytes
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ConversionError("Checkpoint tensor was truncated")
            output.write(chunk)
            peak = max(peak, len(chunk))
            remaining -= len(chunk)
            del chunk
    return peak


def _cast_values(source: np.ndarray, source_dtype: str, dtype: str) -> np.ndarray:
    """Cast with the frozen CPU reference's portable F16 NaN bit policy.

    Native half conversion kernels differ in NaN quieting and payload handling.
    Reconstruct those exceptional outputs from the source bits; finite values
    retain the native rounded cast. The only extra array is a one-byte mask.
    """
    import ml_dtypes

    dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
    with np.errstate(over="ignore", invalid="ignore"):
        values = source.astype(dtypes[dtype])
        if source_dtype == dtype or "F16" not in {source_dtype, dtype}:
            return values
        nan_mask = np.isnan(source)
    if not nan_mask.any():
        return values
    source_bits = source.view(f"<u{DTYPE_BYTES[source_dtype]}")
    target_bits = values.view(f"<u{DTYPE_BYTES[dtype]}")
    # Scalar reconstruction avoids index/payload arrays for an all-NaN chunk.
    for index, is_nan in enumerate(nan_mask):
        if not is_nan:
            continue
        bits = int(source_bits[index])
        sign = bits >> (31 if source_dtype == "F32" else 15)
        if dtype == "BF16":
            target_bits[index] = (sign << 15) | 0x7FC0
        elif dtype == "F32":
            target_bits[index] = (sign << 31) | 0x7FC00000 | ((bits & 0x03FF) << 13)
        else:
            payload = (bits & 0x007F) << 3 if source_dtype == "BF16" else (bits & 0x007FFFFF) >> 13
            target_bits[index] = (sign << 15) | 0x7E00 | payload
    return values


def _cast_f32(tensor: TensorInfo, output: BinaryIO, chunk_bytes: int) -> int:
    if tensor.dtype == "F32":
        return _copy(tensor, output, chunk_bytes)
    if tensor.dtype not in {"BF16", "F16"}:
        raise UnsupportedConversion("Normalization dtype is unsupported")
    # F16 also needs a one-byte NaN mask while constructing the F32 output.
    mask_bytes = int(tensor.dtype == "F16")
    elements = chunk_bytes // (tensor.itemsize + 4 + mask_bytes)
    if elements < 1:
        raise ConversionError("insufficient_transform_workspace")
    peak = 0
    with _open_tensor(tensor) as stream:
        remaining = tensor.nbytes
        while remaining:
            chunk = stream.read(min(elements * tensor.itemsize, remaining))
            if not chunk or len(chunk) % tensor.itemsize:
                raise ConversionError("Checkpoint tensor was truncated")
            if tensor.dtype == "BF16":
                values = np.frombuffer(chunk, dtype="<u2").astype("<u4")
                np.left_shift(values, 16, out=values)
                values = values.view("<f4")
            else:
                values = _cast_values(np.frombuffer(chunk, dtype="<f2"), "F16", "F32")
            output.write(memoryview(values).cast("B"))
            peak = max(peak, len(chunk) + values.nbytes + len(values) * mask_bytes)
            remaining -= len(chunk)
            del chunk, values
    return peak


def _cast(tensor: TensorInfo, output: BinaryIO, dtype: str, chunk_bytes: int) -> int:
    """Cast a dense payload in bounded chunks using IEEE target semantics."""
    if tensor.dtype == dtype:
        return _copy(tensor, output, chunk_bytes)
    import ml_dtypes

    dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
    if tensor.dtype not in dtypes or dtype not in dtypes:
        raise UnsupportedConversion("unsupported_dense_dtype")
    # Reserve native casting scratch and the portable half-cast NaN mask.
    # This is a transform workspace, not whole-process RSS.
    mask_bytes = int("F16" in {tensor.dtype, dtype})
    elements = chunk_bytes // (tensor.itemsize + 2 * DTYPE_BYTES[dtype] + mask_bytes)
    if elements < 1:
        raise ConversionError("insufficient_transform_workspace")
    peak = 0
    with _open_tensor(tensor) as stream:
        remaining = tensor.nbytes
        while remaining:
            chunk = stream.read(min(elements * tensor.itemsize, remaining))
            if not chunk or len(chunk) % tensor.itemsize:
                raise ConversionError("truncated_tensor")
            source = np.frombuffer(chunk, dtype=dtypes[tensor.dtype])
            values = _cast_values(source, tensor.dtype, dtype)
            output.write(memoryview(values.view(f"u{DTYPE_BYTES[dtype]}")).cast("B"))
            peak = max(peak, len(chunk) + 2 * values.nbytes + len(values) * mask_bytes)
            remaining -= len(chunk)
            del chunk, source, values
    return peak


def _flush_f32_subnormals(values: np.ndarray) -> None:
    """Reproduce the pinned CPU arithmetic backend's signed flush to zero."""
    bits = values.view("<u4")
    subnormal = np.bitwise_and(bits, 0x7F800000) == 0
    np.bitwise_and(bits, 0x80000000, out=bits, where=subnormal)


def _scalar_transform(
    tensor: TensorInfo, output: BinaryIO, dtype: str, scalar: float, kind: str, chunk_bytes: int
) -> int:
    """Scale in the source precision before casting, as the pinned importer does.

    BF16/F16 arithmetic widens to F32, then rounds back to the source dtype.
    CPU F32 subnormal operands/results flush to signed zero. Division by a
    constant uses a rounded reciprocal multiplication: F16 rounds its inverse
    to F16; BF16 widens its divisor before taking an F32 inverse. The subsequent
    cast remains a separate operation, including widening normalization scales.
    Frozen independent reference hashes cover every 16-bit source pattern.
    """
    import ml_dtypes

    dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
    if tensor.dtype not in dtypes or dtype not in dtypes:
        raise UnsupportedConversion("unsupported_dense_dtype")
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
        raise ConversionError("invalid_scalar_transform")
    if kind not in {"scalar_multiply", "scalar_divide"} or (kind == "scalar_divide" and scalar == 0):
        raise ConversionError("invalid_scalar_transform")
    # Peak stages: input + F32 working values + exponent/mask scratch (5B),
    # or input + source-rounded values + target cast. The extra target-sized
    # allowance covers the same casting scratch reserved by _cast.
    mask_bytes = int(tensor.dtype != dtype and "F16" in {tensor.dtype, dtype})
    bytes_per_element = max(tensor.itemsize + 9, 2 * tensor.itemsize + 2 * DTYPE_BYTES[dtype] + mask_bytes)
    elements = chunk_bytes // bytes_per_element
    if elements < 1:
        raise ConversionError("insufficient_transform_workspace")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        operand = np.float32(dtypes[tensor.dtype].type(scalar))
        if operand.view("<u4") & 0x7F800000 == 0:
            operand = np.uint32(operand.view("<u4") & 0x80000000).view("<f4")
        if kind == "scalar_divide":
            operand = np.float32(1) / operand
            if tensor.dtype == "F16":
                operand = np.float32(np.float16(operand))
            if operand.view("<u4") & 0x7F800000 == 0:
                operand = np.uint32(operand.view("<u4") & 0x80000000).view("<f4")
        # Hardware FTZ also flushes an exact tiny result that would otherwise
        # round UP to the smallest normal F32. Comparing input F32 bit patterns
        # to the scalar's upward-rounded threshold captures this without F64
        # payload arrays or changing source precision.
        underflow_limit = None
        if np.isfinite(operand) and operand != 0:
            threshold = float(np.finfo(np.float32).tiny) / abs(float(operand))
            limit = np.float32(threshold)
            if float(limit) < threshold:
                limit = np.nextafter(limit, np.float32(np.inf))
            underflow_limit = limit.view("<u4")
    peak = 0
    with _open_tensor(tensor) as stream:
        remaining = tensor.nbytes
        while remaining:
            chunk = stream.read(min(elements * tensor.itemsize, remaining))
            if not chunk or len(chunk) % tensor.itemsize:
                raise ConversionError("truncated_tensor")
            count = len(chunk) // tensor.itemsize
            source = np.frombuffer(chunk, dtype=dtypes[tensor.dtype])
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                values = _cast_values(source, tensor.dtype, "F32")
                _flush_f32_subnormals(values)
                underflow = None
                if underflow_limit is not None:
                    underflow = np.bitwise_and(values.view("<u4"), 0x7FFFFFFF) < underflow_limit
                np.multiply(values, operand, out=values)
                if underflow is not None:
                    bits = values.view("<u4")
                    np.bitwise_and(bits, 0x80000000, out=bits, where=underflow)
                    del underflow
                _flush_f32_subnormals(values)
                rounded = _cast_values(values, "F32", tensor.dtype)
                del values
                target = _cast_values(rounded, tensor.dtype, dtype)
            output.write(memoryview(target.view(f"u{DTYPE_BYTES[dtype]}")).cast("B"))
            peak = max(peak, count * bytes_per_element)
            remaining -= len(chunk)
            del chunk, source, rounded, target
    return peak


def transpose_tensor(tensor: TensorInfo, output: BinaryIO, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Transpose a dense rank-two tensor into the current output position.

    Use raw integer elements to preserve BF16 and floating-point bit patterns.
    Tiles avoid a full-model or full-tensor mmap and keep heap buffers bounded.
    This primitive is available for future adapters; Qwen3 embeds are copied.
    """
    if len(tensor.shape) != 2 or chunk_bytes < 16:
        raise ConversionError("Transpose needs a matrix and at least 16 chunk bytes")
    rows, cols = tensor.shape
    itemsize = tensor.itemsize
    if rows == 0 or cols == 0:
        return 0
    if chunk_bytes < 3 * itemsize:
        raise ConversionError("Transpose chunk cannot hold a tile and its scratch")
    edge = max(1, int(math.sqrt(chunk_bytes // (3 * itemsize))))
    tile_rows, tile_cols = min(rows, edge), min(cols, edge)
    output_start = output.tell()
    peak = 0
    raw_dtype = np.dtype(f"u{itemsize}")
    with _open_tensor(tensor) as stream:
        for row in range(0, rows, tile_rows):
            height = min(tile_rows, rows - row)
            for col in range(0, cols, tile_cols):
                width = min(tile_cols, cols - col)
                tile = np.empty((height, width), dtype=raw_dtype)
                for i in range(height):
                    stream.seek(tensor.offset + ((row + i) * cols + col) * itemsize)
                    chunk = stream.read(width * itemsize)
                    if len(chunk) != width * itemsize:
                        raise ConversionError("Checkpoint tensor was truncated")
                    tile[i] = np.frombuffer(chunk, dtype=raw_dtype)
                transposed = np.ascontiguousarray(tile.T)
                peak = max(peak, tile.nbytes + transposed.nbytes + len(chunk))
                for i in range(width):
                    output.seek(output_start + ((col + i) * rows + row) * itemsize)
                    output.write(memoryview(transposed[i]).cast("B"))
                del tile, transposed, chunk
    output.seek(output_start + tensor.nbytes)
    return peak


def transpose_f32(tensor: TensorInfo, output: BinaryIO, chunk_bytes: int) -> int:
    """Transpose a dense convolution matrix and widen within bounded tiles.

    This follows normalization widening: BF16 bits widen exactly, F16 uses the
    portable frozen NaN policy, and F32 payload bits are preserved. The output
    uses random tile writes, so the writer hashes the completed file afterward.
    """
    if len(tensor.shape) != 2 or tensor.dtype not in {"BF16", "F16", "F32"}:
        raise ConversionError("Invalid F32 transpose source")
    rows, cols = tensor.shape
    if rows == 0 or cols == 0:
        return 0
    # Include the row read, input tile, cast, contiguous transpose, and NaN mask.
    mask_bytes = int(tensor.dtype == "F16")
    bytes_per_element = 2 * tensor.itemsize + 8 + mask_bytes
    if chunk_bytes < bytes_per_element:
        raise ConversionError("insufficient_transform_workspace")
    edge = max(1, math.isqrt(chunk_bytes // bytes_per_element))
    tile_rows, tile_cols = min(rows, edge), min(cols, edge)
    output_start = output.tell()
    peak = 0
    raw_dtype = np.dtype(f"<u{tensor.itemsize}")
    with _open_tensor(tensor) as stream:
        for row in range(0, rows, tile_rows):
            height = min(tile_rows, rows - row)
            for col in range(0, cols, tile_cols):
                width = min(tile_cols, cols - col)
                tile = np.empty((height, width), dtype=raw_dtype)
                for i in range(height):
                    stream.seek(tensor.offset + ((row + i) * cols + col) * tensor.itemsize)
                    chunk = stream.read(width * tensor.itemsize)
                    if len(chunk) != width * tensor.itemsize:
                        raise ConversionError("Checkpoint tensor was truncated")
                    tile[i] = np.frombuffer(chunk, dtype=raw_dtype)
                if tensor.dtype == "BF16":
                    values = tile.astype("<u4")
                    np.left_shift(values, 16, out=values)
                    values = values.view("<f4")
                elif tensor.dtype == "F16":
                    values = _cast_values(tile.view("<f2").reshape(-1), "F16", "F32").reshape(height, width)
                else:
                    values = tile.view("<f4")
                transposed = np.ascontiguousarray(values.T)
                peak = max(peak, height * width * bytes_per_element)
                for i in range(width):
                    output.seek(output_start + ((col + i) * rows + row) * 4)
                    output.write(memoryview(transposed[i]).cast("B"))
                del tile, values, transposed, chunk
    output.seek(output_start + rows * cols * 4)
    return peak


class _HashingWriter:
    """Hash successful sequential writes without rereading the output payload.

    Random writes invalidate the incremental digest; tiled transpose falls back
    to hashing the completed file. This wrapper never retains payload buffers.
    """

    def __init__(self, stream: BinaryIO, progress: ProgressReporter | None = None, total_bytes: int | None = None):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.sequential = True
        self.progress = progress
        self.total_bytes = total_bytes
        self.written_bytes = 0

    def write(self, data) -> int:
        written = self.stream.write(data)
        if written != len(data):
            raise ConversionError("Incomplete output write")
        if self.sequential:
            self.digest.update(data)
        self.written_bytes += written
        if self.progress is not None:
            self.progress.emit("conversion", "running", self.written_bytes, self.total_bytes)
        return written

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset, whence=0) -> int:
        self.sequential = False
        return self.stream.seek(offset, whence)

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()


def write_plan(
    plan: ConversionPlan,
    path: Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    progress: ProgressCallback | ProgressReporter | None = None,
) -> dict:
    """Write a header-first safetensors file into a caller-owned staging area."""
    if type(chunk_bytes) is not int or chunk_bytes < 16:
        raise ConversionError("Chunk size must be an integer of at least 16 bytes")
    operations = sorted(plan.operations, key=lambda op: (-DTYPE_BYTES[op.dtype], op.name))
    if len({op.name for op in operations}) != len(operations):
        raise ConversionError("Conversion plan contains duplicate output tensors")
    if any(op.kind in {"fuse", "fuse_cast"} and len({source.dtype for source in op.sources}) > 1 for op in operations):
        raise UnsupportedConversion("Mixed-dtype fusion promotion is not implemented")
    if any(
        op.kind in {"scalar_multiply", "scalar_divide"}
        and (
            len(op.sources) != 1
            or isinstance(op.scalar, bool)
            or not isinstance(op.scalar, (int, float))
            or not math.isfinite(op.scalar)
            or (op.kind == "scalar_divide" and op.scalar == 0)
        )
        for op in operations
    ):
        raise ConversionError("invalid_scalar_transform")
    if any(op.kind in {"mlx_repack", "mlx_params"} for op in operations):
        from .compressed import validate_mlx_operation

        for op in operations:
            if op.kind in {"mlx_repack", "mlx_params"}:
                validate_mlx_operation(op)
    for op in operations:
        if op.kind == "transpose_f32" and (
            len(op.sources) != 1
            or op.dtype != "F32"
            or len(op.sources[0].shape) != 2
            or op.shape != op.sources[0].shape[::-1]
            or op.sources[0].dtype not in {"BF16", "F16", "F32"}
        ):
            raise ConversionError("Invalid F32 transpose operation")
    header = {"__metadata__": plan.metadata} if plan.metadata else {}
    offset = 0
    for op in operations:
        header[op.name] = {"dtype": op.dtype, "shape": list(op.shape), "data_offsets": [offset, offset + op.nbytes]}
        offset += op.nbytes
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    progress = reporter(progress)
    total_bytes = 8 + len(encoded) + offset
    if progress is not None:
        progress.emit("conversion", "running", 0, total_bytes, force=True)
    start = time.perf_counter()
    peak = 0
    # Refuse accidental in-place conversion, even when the caller bypasses CLI.
    if path.resolve() in {tensor.path.resolve() for op in operations for tensor in op.sources}:
        raise ConversionError("Output overlaps checkpoint input")
    try:
        with path.open("xb") as stream:
            output = _HashingWriter(stream, progress, total_bytes)
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for op in operations:
                before = output.tell()
                if op.kind in {"copy", "fuse", "mlx_repack"}:
                    for source in op.sources:
                        peak = max(peak, _copy(source, output, chunk_bytes))
                elif op.kind == "cast_f32" and len(op.sources) == 1:
                    peak = max(peak, _cast_f32(op.sources[0], output, chunk_bytes))
                elif op.kind in {"cast", "fuse_cast"}:
                    for source in op.sources:
                        peak = max(peak, _cast(source, output, op.dtype, chunk_bytes))
                elif op.kind in {"scalar_multiply", "scalar_divide"} and len(op.sources) == 1:
                    peak = max(
                        peak, _scalar_transform(op.sources[0], output, op.dtype, op.scalar, op.kind, chunk_bytes)
                    )
                elif op.kind == "transpose" and len(op.sources) == 1:
                    peak = max(peak, transpose_tensor(op.sources[0], output, chunk_bytes))
                elif op.kind == "transpose_f32" and len(op.sources) == 1:
                    peak = max(peak, transpose_f32(op.sources[0], output, chunk_bytes))
                elif op.kind == "mlx_params":
                    from .compressed import write_mlx_params

                    peak = max(peak, write_mlx_params(op, output, chunk_bytes))
                else:
                    raise ConversionError("Conversion plan has an invalid operation")
                if output.tell() - before != op.nbytes:
                    raise ConversionError("Conversion operation byte count mismatch")
            output.flush()
            os.fsync(output.fileno())
        # Recheck every original input after the last read, including accepted
        # tied-head duplicates that intentionally do not produce an output.
        # stat follows the current symlink target, detecting replaced HF blobs.
        signatures = {
            tensor.path: tensor.file_signature
            for tensor in (plan.input_tensors or tuple(source for op in operations for source in op.sources))
        }
        for source_path, signature in signatures.items():
            if _signature(source_path.stat()) != signature:
                raise ConversionError("Checkpoint changed during conversion")
    except OSError as exc:
        raise ConversionError("Cannot write converted checkpoint") from exc
    if output.sequential:
        output_hash = output.digest.hexdigest()
    else:
        with path.open("rb") as stream:
            output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if progress is not None:
        progress.emit("conversion", "running", total_bytes, total_bytes, force=True)
    return {
        "tensors": len(operations),
        "source_tensors": plan.source_tensors,
        "input_bytes": plan.input_bytes,
        "output_tensor_bytes": offset,
        "output_bytes": 8 + len(encoded) + offset,
        "output_sha256": output_hash,
        "peak_buffer_bytes": peak,
        "chunk_bytes": chunk_bytes,
        "write_seconds": time.perf_counter() - start,
    }


def convert_streaming(
    source: Path,
    output: Path,
    *,
    dtype: str = "bfloat16",
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict:
    """Convert supported Qwen3 weights; the caller supplies tokenizer/config."""
    from .qwen3 import build_qwen3_plan

    start = time.perf_counter()
    plan = build_qwen3_plan(source, dtype=dtype)
    planned = time.perf_counter()
    stats = write_plan(plan, output / "model.safetensors", chunk_bytes=chunk_bytes)
    stats.update(
        {
            "plan_seconds": planned - start,
            "total_seconds": time.perf_counter() - start,
            "decoder_config": plan.decoder_config,
        }
    )
    return stats
