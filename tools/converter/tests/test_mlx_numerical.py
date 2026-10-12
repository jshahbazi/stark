"""Frozen external MLX export and synthetic dequantization checks.

The reference is Lalamo 831accb8252a6b6606c378048caeb44178f1a050.
No external converter is imported or executed by the test suite.
"""

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.compressed import MLXQuantization, plan_mlx_matrix, read_mlx_matrix
from stark_convert.streaming import ConversionPlan, read_safetensors_header, write_plan

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "mlx_export_numerical.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())
ARRAY_DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}


def synthetic_planes(rows, columns, bits, group_size, source_dtype, *, row_offset=0):
    """Make every code observable and vary coefficients across both axes."""
    # Integer codes run through all 16/256 values, including high/sign bits.
    indices = np.arange(row_offset * columns, (row_offset + rows) * columns, dtype=np.uint32).reshape(rows, columns)
    codes = ((indices * 73 + 19) & ((1 << bits) - 1)).astype(np.uint8)
    factor = 32 // bits
    grouped = codes.astype(np.uint32).reshape(rows, columns // factor, factor)
    words = np.zeros(grouped.shape[:2], dtype="<u4")
    for index in range(factor):
        words |= grouped[:, :, index] << (index * bits)
    row = np.arange(row_offset, row_offset + rows, dtype=np.int32)[:, None]
    group = np.arange(columns // group_size, dtype=np.int32)[None, :]
    powers = (row * 3 + group) % 4
    scales = np.power(2.0, powers - 5).astype(ARRAY_DTYPES[source_dtype])
    biases = (((row + group * 3) % 4) - 1).astype(np.float32) / 2
    biases = biases.astype(ARRAY_DTYPES[source_dtype])
    return words, scales, biases, codes


def _native_codes(payload, bits, rows, columns):
    """Decode native bytes with scalar indexing, independent of word packing."""
    result = np.empty((rows, columns), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            bit = (row * columns + column) * bits
            result[row, column] = (payload[bit // 8] >> (bit % 8)) & ((1 << bits) - 1)
    return result


@pytest.mark.parametrize("case", REFERENCE["cases"].values(), ids=REFERENCE["cases"])
def test_mlx_export_and_binary_affine_values_match_frozen_external_reference(tmp_path, case):
    bits, group_size = case["bits"], case["group_size"]
    source_dtype, target_dtype, layout = case["source_dtype"], case["target_dtype"], case["layout"]
    row_counts = (5, 7, 5) if case["fused"] else (17,)
    columns = 128
    tensors, physical_codes, original_scales, original_biases = {}, [], [], []
    offset = 0
    for index, rows in enumerate(row_counts):
        words, scales, biases, codes = synthetic_planes(
            rows, columns, bits, group_size, source_dtype, row_offset=offset
        )
        # Both containers must preserve identical word payloads. Multiple
        # shards deliberately split a fused matrix's three source planes.
        word_dtype = "I32" if index % 2 == 0 else "U32"
        tensors[f"source{index}.weight"] = (word_dtype, words.view("<i4") if word_dtype == "I32" else words)
        tensors[f"source{index}.scales"] = (source_dtype, scales)
        tensors[f"source{index}.biases"] = (source_dtype, biases)
        physical_codes.append(codes)
        original_scales.append(scales)
        original_biases.append(biases)
        offset += rows
    infos = {}
    for shard, contents in enumerate((dict(list(tensors.items())[::2]), dict(list(tensors.items())[1::2]))):
        path = tmp_path / f"source-{shard}.safetensors"
        save_tensors(path, contents)
        infos.update(read_safetensors_header(path)[1])
    quantization = MLXQuantization(bits, group_size)
    sources = tuple(
        read_mlx_matrix(infos, f"source{index}.weight", (rows, columns), quantization)
        for index, rows in enumerate(row_counts)
    )
    operations = plan_mlx_matrix("matrix", sources, dtype=target_dtype, layout=layout, quantization=quantization)
    spec = {"type": "MLXSpec", "bits": bits, "group_size": group_size, "layout": layout}
    plan = ConversionPlan(
        {},
        operations,
        {"matrix.spec": json.dumps(spec)},
        sum(info.nbytes for info in infos.values()),
        len(infos),
        tuple(infos.values()),
    )
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert json.loads(metadata["matrix.spec"]) == case["spec"]
    for name in ("weights", "scales", "biases"):
        info = converted[f"matrix.{name}"]
        assert info.dtype == ("U8" if name == "weights" else target_dtype)
        assert list(info.shape) == case["arrays"][name]["shape"]
        assert hashlib.sha256(tensor_bytes(info)).hexdigest() == case["arrays"][name]["sha256"]
    codes = _native_codes(tensor_bytes(converted["matrix.weights"]), bits, offset, columns)
    assert np.array_equal(codes, np.concatenate(physical_codes))
    assert set(codes.flat) == set(range(1 << bits))
    stored_planes = []
    for name in ("scales", "biases"):
        info = converted[f"matrix.{name}"]
        values = np.frombuffer(tensor_bytes(info), dtype=ARRAY_DTYPES[target_dtype]).reshape(info.shape)
        if layout == "output_input":
            assert np.all(values[:, offset:] == 0)
            values = values[:, :offset].T
        stored_planes.append(values)
    scales, biases = stored_planes
    expected_scales = np.concatenate(original_scales).astype(ARRAY_DTYPES[target_dtype])
    expected_biases = np.concatenate(original_biases).astype(ARRAY_DTYPES[target_dtype])
    assert np.array_equal(scales, expected_scales)
    assert np.array_equal(biases, expected_biases)
    # Binary coefficients make the product exactly representable in all
    # supported dtypes. Final target rounding is explicit; tolerance is zero.
    dequantized = np.empty((offset, columns), dtype=ARRAY_DTYPES[target_dtype])
    for row in range(offset):
        for column in range(columns):
            group = column // group_size
            dequantized[row, column] = int(codes[row, column]) * float(scales[row, group]) + float(biases[row, group])
    assert hashlib.sha256(dequantized.tobytes()).hexdigest() == case["dequantized_sha256"]
