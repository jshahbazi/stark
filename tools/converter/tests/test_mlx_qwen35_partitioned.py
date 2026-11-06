"""Packed affine row partitions preserve source bytes and decoder row order."""

import json

import numpy as np
import pytest
from test_mlx import save_mlx_fixture
from test_mlx_qwen35 import mlx_qwen35_fixture
from test_streaming import tensor_bytes

from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.streaming import read_safetensors_header, write_plan


def unpack(payload, rows, columns, bits):
    row_bytes = columns * bits // 8
    return np.array(
        [
            [
                (int.from_bytes(payload[row * row_bytes : (row + 1) * row_bytes], "little") >> (column * bits))
                & ((1 << bits) - 1)
                for column in range(columns)
            ]
            for row in range(rows)
        ],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("bits", [5, 6])
def test_uniform_packed_width_keeps_head_bytes_and_widens_embedding(tmp_path, bits):
    source = tmp_path / "source"
    _, tensors = mlx_qwen35_fixture(source, bits=bits, group_size=64)
    plan = build_qwen35_plan(source)
    output = tmp_path / "out.safetensors"
    write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    head = "decoder.embedding.output_embedding"
    assert json.loads(metadata[head + ".spec"])["bits"] == bits
    assert tensor_bytes(converted[head + ".weights"]) == tensors["lm_head.weight"][1].tobytes()
    embedding = "decoder.embedding.input_embedding"
    assert json.loads(metadata[embedding + ".spec"])["bits"] == 8
    expected = unpack(tensors["model.embed_tokens.weight"][1].tobytes(), 17, 128, bits)
    assert tensor_bytes(converted[embedding + ".weights"]) == expected.tobytes()


def test_heterogeneous_projection_retains_code_and_coefficient_row_permutations(tmp_path):
    source = tmp_path / "source"
    config, tensors = mlx_qwen35_fixture(source, group_size=64)
    attn = "model.layers.1.self_attn"
    for branch, bits in (("q_proj", 5), ("v_proj", 6)):
        name = attn + "." + branch
        rows = tensors[name + ".weight"][1].shape[0]
        payload = b"".join(
            sum(((col * 7 + row * 11) % (1 << bits)) << (col * bits) for col in range(128)).to_bytes(
                128 * bits // 8, "little"
            )
            for row in range(rows)
        )
        tensors[name + ".weight"] = "U32", np.frombuffer(payload, dtype="<u4").reshape(rows, 4 * bits)
        for key in ("quantization", "quantization_config"):
            config[key][name] = {"bits": bits, "group_size": 64}
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    plan = build_qwen35_plan(source)
    output = tmp_path / "out.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    assert stats["peak_buffer_bytes"] <= 96
    metadata, converted = read_safetensors_header(output)
    prefix = "decoder.transformer.layers.1.mixer.qkvg_projection.weights"
    spec = json.loads(metadata[prefix + ".spec"])
    assert spec["type"] == "MLXRowPartitionedSpec"
    assert [(part["bits"], part["rows"]) for part in spec["segments"]] == [(5, 128), (4, 64), (6, 64), (5, 128)]
    actual = []
    coefficient_rows = {"scales": [], "biases": []}
    for index, part in enumerate(spec["segments"]):
        stem = prefix + f".segment_{index}"
        actual.append(unpack(tensor_bytes(converted[stem + ".weights"]), part["rows"], 128, part["bits"]))
        for plane in coefficient_rows:
            info = converted[stem + "." + plane]
            # GroupOutput planes pad only each segment's output row tail.
            coefficient_rows[plane].append(
                np.frombuffer(tensor_bytes(info), dtype="<u2").reshape(2, -1)[:, : part["rows"]].T
            )
    qcodes = unpack(tensors[attn + ".q_proj.weight"][1].tobytes(), 256, 128, 5)
    kcodes = unpack(tensors[attn + ".k_proj.weight"][1].tobytes(), 64, 128, 4)
    vcodes = unpack(tensors[attn + ".v_proj.weight"][1].tobytes(), 64, 128, 6)
    expected = np.concatenate(
        [
            *(qcodes[h * 64 : h * 64 + 32] for h in range(4)),
            kcodes,
            vcodes,
            *(qcodes[h * 64 + 32 : h * 64 + 64] for h in range(4)),
        ]
    )
    assert np.array_equal(np.concatenate(actual), expected)
    for plane in coefficient_rows:
        q = tensors[attn + ".q_proj." + plane][1].view("<u2")
        expected = np.concatenate(
            [
                *(q[h * 64 : h * 64 + 32] for h in range(4)),
                tensors[attn + ".k_proj." + plane][1].view("<u2"),
                tensors[attn + ".v_proj." + plane][1].view("<u2"),
                *(q[h * 64 + 32 : h * 64 + 64] for h in range(4)),
            ]
        )
        assert np.array_equal(np.concatenate(coefficient_rows[plane]), expected)
