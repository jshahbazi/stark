"""Exact dense casts against frozen independent JAX CPU reference bytes.

The fixture was generated in the external pinned Lalamo validation environment.
Normal converter tests do not import JAX or Lalamo. Source patterns cover signed
zero, subnormal boundaries, extrema, infinity, rounding ties, and quiet/signaling
NaNs with positive/negative signs and distinct payloads.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stark_convert.streaming import (
    ConversionPlan,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    read_safetensors_header,
    write_plan,
)

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "dense_casts.json").read_text())


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
