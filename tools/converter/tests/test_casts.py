"""Exact dense casts against frozen independent JAX CPU reference bytes.

The fixture was generated in the external pinned Lalamo validation environment.
Normal converter tests do not import JAX or Lalamo. Source patterns cover signed
zero, subnormal boundaries, extrema, infinity, rounding ties, and quiet/signaling
NaNs with positive/negative signs and distinct payloads.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from stark_convert.streaming import (
    ConversionError,
    ConversionPlan,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    _cast,
    _cast_f32,
    _cast_values,
    read_safetensors_header,
    write_plan,
)

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "dense_casts.json").read_text())


def round_right_ties_even(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    midpoint = 1 << (shift - 1)
    return quotient + int(remainder > midpoint or (remainder == midpoint and quotient & 1))


def f16_to_f32_bits(bits):
    sign = (bits & 0x8000) << 16
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x03FF
    if exponent == 0:
        if fraction == 0:
            return sign
        leading_bit = fraction.bit_length() - 1
        return sign | ((leading_bit + 103) << 23) | ((fraction - (1 << leading_bit)) << (23 - leading_bit))
    if exponent == 0x1F:
        return sign | 0x7F800000 | (fraction << 13) | (0x00400000 if fraction else 0)
    return sign | ((exponent + 112) << 23) | (fraction << 13)


def reference_16bit_cast_bits(bits, source_dtype, target):
    """Independent IEEE integer arithmetic; no native floating-point casts."""
    if source_dtype == "F16":
        widened = f16_to_f32_bits(bits)
        if target == "F32":
            return widened
        if (bits & 0x7C00) == 0x7C00 and bits & 0x03FF:
            return (bits & 0x8000) | 0x7FC0
        return (widened + 0x7FFF + ((widened >> 16) & 1)) >> 16
    sign = bits & 0x8000
    exponent = (bits >> 7) & 0xFF
    fraction = bits & 0x007F
    if exponent == 0xFF:
        return sign | 0x7C00 | (fraction << 3) | (0x0200 if fraction else 0)
    if exponent > 142:
        return sign | 0x7C00
    if exponent >= 113:
        return sign | ((exponent - 112) << 10) | (fraction << 3)
    if exponent < 102:
        return sign
    return sign | round_right_ties_even(128 + fraction, 110 - exponent)


def exhaustive_reference_digest(source_dtype, target):
    bits = np.fromiter(
        (reference_16bit_cast_bits(value, source_dtype, target) for value in range(1 << 16)),
        dtype="<u4" if target == "F32" else "<u2",
        count=1 << 16,
    )
    return hashlib.sha256(bits.tobytes()).hexdigest()


def source_tensor(path, dtype, case):
    payload = np.array(case["source_bits"], dtype="<u4" if dtype == "F32" else "<u2").tobytes()
    assert hashlib.sha256(payload).hexdigest() == case["source_sha256"]
    path.write_bytes(payload)
    stat = path.stat()
    return TensorInfo(
        path,
        dtype,
        (len(case["source_bits"]),),
        0,
        len(payload),
        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
    )


@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("target", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("workspace", [16, 96])
@pytest.mark.parametrize("fused", [False, True])
def test_cast_and_fused_cast_match_independent_reference_bytes(tmp_path, dtype, target, workspace, fused):
    case = REFERENCE["cases"][dtype]
    tensor = source_tensor(tmp_path / "synthetic-payload.bin", dtype, case)
    if fused:
        split = len(case["source_bits"]) // 2
        sources = (
            TensorInfo(tensor.path, dtype, (split,), 0, split * tensor.itemsize, tensor.file_signature),
            TensorInfo(
                tensor.path,
                dtype,
                (len(case["source_bits"]) - split,),
                split * tensor.itemsize,
                tensor.nbytes - split * tensor.itemsize,
                tensor.file_signature,
            ),
        )
    else:
        sources = (tensor,)
    operation = TensorOperation("matrix.weights", sources, target, tensor.shape, "fuse_cast" if fused else "cast")
    plan = ConversionPlan({}, (operation,), {}, tensor.nbytes, len(sources), (tensor,))
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=workspace)
    _, tensors = read_safetensors_header(output)
    result = tensors["matrix.weights"]
    with output.open("rb") as handle:
        handle.seek(result.offset)
        digest = hashlib.sha256(handle.read(result.nbytes)).hexdigest()
    assert result.dtype == target
    assert result.shape == tensor.shape
    assert digest == case["output_sha256"][target]
    assert stats["peak_buffer_bytes"] <= workspace
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("workspace", [16, 96])
def test_normalization_cast_matches_independent_reference_bytes(tmp_path, dtype, workspace):
    case = REFERENCE["cases"][dtype]
    tensor = source_tensor(tmp_path / "synthetic-payload.bin", dtype, case)
    operation = TensorOperation("normalization.scales", (tensor,), "F32", tensor.shape, "cast_f32")
    plan = ConversionPlan({}, (operation,), {}, tensor.nbytes, 1, (tensor,))
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=workspace)
    _, tensors = read_safetensors_header(output)
    result = tensors["normalization.scales"]
    with output.open("rb") as handle:
        handle.seek(result.offset)
        digest = hashlib.sha256(handle.read(result.nbytes)).hexdigest()
    assert result.dtype == "F32"
    assert result.shape == tensor.shape
    assert digest == case["output_sha256"]["F32"]
    assert stats["peak_buffer_bytes"] <= workspace
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("dtype,target", [("F16", "F32"), ("F16", "BF16"), ("BF16", "F16")])
def test_every_16bit_pattern_matches_independent_integer_reference(tmp_path, dtype, target):
    payload = np.arange(1 << 16, dtype="<u2").tobytes()
    case = {"source_bits": range(1 << 16), "source_sha256": hashlib.sha256(payload).hexdigest()}
    tensor = source_tensor(tmp_path / "exhaustive-synthetic-payload.bin", dtype, case)
    operation = TensorOperation("matrix.weights", (tensor,), target, tensor.shape, "cast")
    plan = ConversionPlan({}, (operation,), {}, tensor.nbytes, 1, (tensor,))
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=1024)
    result = read_safetensors_header(output)[1]["matrix.weights"]
    with output.open("rb") as handle:
        handle.seek(result.offset)
        digest = hashlib.sha256(handle.read(result.nbytes)).hexdigest()
    assert result.dtype == target
    assert result.shape == tensor.shape
    assert digest == exhaustive_reference_digest(dtype, target)
    assert stats["peak_buffer_bytes"] <= 1024


def test_every_f16_normalization_pattern_matches_independent_integer_reference(tmp_path):
    payload = np.arange(1 << 16, dtype="<u2").tobytes()
    case = {"source_bits": range(1 << 16), "source_sha256": hashlib.sha256(payload).hexdigest()}
    tensor = source_tensor(tmp_path / "exhaustive-synthetic-payload.bin", "F16", case)
    output = io.BytesIO()
    peak = _cast_f32(tensor, output, 1024)
    assert hashlib.sha256(output.getvalue()).hexdigest() == exhaustive_reference_digest("F16", "F32")
    assert peak <= 1024


@pytest.mark.parametrize(
    "dtype,target,minimum", [("F16", "F32", 11), ("F16", "BF16", 7), ("BF16", "F16", 7), ("F32", "F16", 9)]
)
def test_cast_minimum_workspace_includes_nan_mask(tmp_path, dtype, target, minimum):
    tensor = source_tensor(tmp_path / "synthetic-payload.bin", dtype, REFERENCE["cases"][dtype])
    with pytest.raises(ConversionError, match="insufficient_transform_workspace"):
        _cast(tensor, io.BytesIO(), target, minimum - 1)
    output = io.BytesIO()
    peak = _cast(tensor, output, target, minimum)
    assert hashlib.sha256(output.getvalue()).hexdigest() == REFERENCE["cases"][dtype]["output_sha256"][target]
    assert peak == minimum


def test_f16_normalization_minimum_workspace_includes_nan_mask(tmp_path):
    tensor = source_tensor(tmp_path / "synthetic-payload.bin", "F16", REFERENCE["cases"]["F16"])
    with pytest.raises(ConversionError, match="insufficient_transform_workspace"):
        _cast_f32(tensor, io.BytesIO(), 6)
    output = io.BytesIO()
    peak = _cast_f32(tensor, output, 7)
    assert hashlib.sha256(output.getvalue()).hexdigest() == REFERENCE["cases"]["F16"]["output_sha256"]["F32"]
    assert peak == 7


class BrokenNativeNaNCast(np.ndarray):
    """Simulate a native kernel losing the sign, payload, and quiet bit."""

    def astype(self, dtype, *args, **kwargs):
        source = self.view(np.ndarray)
        with np.errstate(over="ignore", invalid="ignore"):
            values = source.astype(dtype, *args, **kwargs)
        bits = values.view("<u4" if values.itemsize == 4 else "<u2")
        exponent = 0x7F800000 if values.itemsize == 4 else 0x7F80 if values.dtype.name == "bfloat16" else 0x7C00
        np.copyto(bits, exponent | 1, where=np.isnan(source))
        return values


@pytest.mark.parametrize("dtype,target", [("F16", "F32"), ("F16", "BF16"), ("BF16", "F16"), ("F32", "F16")])
def test_native_nan_corruption_is_repaired_without_changing_finite_values(dtype, target):
    import ml_dtypes

    dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
    if dtype == "F32":
        raw = np.array(REFERENCE["cases"][dtype]["source_bits"], dtype="<u4")
        expected_digest = REFERENCE["cases"][dtype]["output_sha256"][target]
    else:
        raw = np.arange(1 << 16, dtype="<u2")
        expected_digest = exhaustive_reference_digest(dtype, target)
    source = raw.view(dtypes[dtype]).view(BrokenNativeNaNCast)
    corrupted = source.astype(dtypes[target])
    assert hashlib.sha256(corrupted.tobytes()).hexdigest() != expected_digest
    result = _cast_values(source, dtype, target)
    assert hashlib.sha256(result.tobytes()).hexdigest() == expected_digest


@pytest.mark.parametrize("target", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("kind", ["fuse", "fuse_cast"])
def test_mixed_source_dtype_fusion_fails_before_output_creation(tmp_path, target, kind):
    # Lalamo concatenates before casting, so JAX first promotes mixed dtypes.
    # A source already at target dtype cannot be copied independently: promotion
    # can quiet signaling NaNs and change their bytes. Reject this unqualified
    # operation rather than silently substituting per-source cast semantics.
    first = source_tensor(tmp_path / "f16-payload.bin", "F16", REFERENCE["cases"]["F16"])
    second = source_tensor(tmp_path / "f32-payload.bin", "F32", REFERENCE["cases"]["F32"])
    shape = (first.shape[0] + second.shape[0],)
    operation = TensorOperation("matrix.weights", (first, second), target, shape, kind)
    plan = ConversionPlan({}, (operation,), {}, first.nbytes + second.nbytes, 2, (first, second))
    output = tmp_path / "converted.safetensors"
    with pytest.raises(UnsupportedConversion):
        write_plan(plan, output, chunk_bytes=96)
    assert not output.exists()


def test_transposed_writer_hash_matches_completed_file_after_random_writes(tmp_path):
    raw = source_tensor(tmp_path / "f16-payload.bin", "F16", REFERENCE["cases"]["F16"])
    matrix = TensorInfo(raw.path, raw.dtype, (4, 5), raw.offset, raw.nbytes, raw.file_signature)
    operation = TensorOperation("matrix.weights", (matrix,), "F16", (5, 4), "transpose")
    plan = ConversionPlan({}, (operation,), {"synthetic": "metadata"}, matrix.nbytes, 1, (matrix,))
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=64)
    _, tensors = read_safetensors_header(output)
    result = tensors["matrix.weights"]
    with output.open("rb") as handle:
        handle.seek(result.offset)
        payload = handle.read(result.nbytes)
    expected = np.array(REFERENCE["cases"]["F16"]["source_bits"], dtype="<u2").reshape(4, 5).T.tobytes()
    assert payload == expected
    assert stats["peak_buffer_bytes"] <= 64
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
