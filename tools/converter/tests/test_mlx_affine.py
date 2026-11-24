"""Model-free checks of the MLX affine 8-bit quantizer and the Inkling pack verifier."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from stark_convert import inkling_affine8
from stark_convert.mlx_affine import (
    AffineRows,
    bf16_bits_to_f32,
    dequantize_rows,
    f32_to_bf16_bits,
    iter_quantized_bf16,
    quantize_rows,
    round_half_away,
)

F = np.float32


def bf16_round(values):
    return bf16_bits_to_f32(f32_to_bf16_bits(np.asarray(values, dtype=F)))


def reference_group(group):
    """Scalar float32 transcription of the Metal kernel, one group at a time."""
    w_min, w_max = F(group.min()), F(group.max())
    scale = max(F(w_max - w_min) / F(255), F(1e-7))
    side = abs(w_min) > abs(w_max)
    scale = scale if side else F(-scale)
    edge = w_min if side else w_max
    q0 = F(round_half_away(np.array([edge / scale], dtype=F))[0])
    if q0 != 0:
        scale = F(edge / q0)
    bias = F(0) if q0 == 0 else edge
    codes = [min(F(round_half_away(np.array([F(v - bias) / scale], dtype=F))[0]), F(255)) for v in group]
    return np.array(codes, dtype=np.uint8), scale, bias


def test_round_half_away_matches_metal_round():
    values = np.array([0.5, -0.5, 1.5, 2.5, -2.5, 0.49999997, 254.5, 255.49998, -0.0, 3.0], dtype=F)
    assert round_half_away(values).tolist() == [1, -1, 2, 3, -3, 0, 255, 255, 0, 3]
    # NumPy's round would give 0, -0, 2, 2, -2 for the ties above.
    assert np.round(values[:5]).tolist() != round_half_away(values[:5]).tolist()


def test_bf16_rounding_is_nearest_even():
    one = 1.0
    tie_down = F(one + 2**-8)  # halfway between 1 and the next BF16; even is 1
    tie_up = F(one + 3 * 2**-8)  # halfway, even neighbour is above
    above = F(one + 2**-8 + 2**-20)
    bits = f32_to_bf16_bits(np.array([tie_down, tie_up, above, -tie_down], dtype=F))
    assert bits.tolist() == [0x3F80, 0x3F82, 0x3F81, 0xBF80]
    assert bf16_bits_to_f32(bits[:1]).tolist() == [1.0]
    with pytest.raises(ValueError):
        f32_to_bf16_bits(np.array([np.inf], dtype=F))


def test_all_zero_group():
    result = quantize_rows(np.zeros((2, 64), dtype=F))
    assert not result.codes.any()
    assert bf16_bits_to_f32(result.biases).tolist() == [[0.0], [0.0]]
    # |w_min| > |w_max| is false, so the clamped scale is negated.
    assert np.all(bf16_bits_to_f32(result.scales) == bf16_round(-1e-7))
    assert not dequantize_rows(result).any()


def test_constant_group_is_exact():
    values = np.full((1, 64), 0.75, dtype=F)
    result = quantize_rows(values)
    assert not result.codes.any()
    assert bf16_bits_to_f32(result.biases)[0, 0] == F(0.75)
    assert np.all(dequantize_rows(result) == values)
    negative = quantize_rows(-values)
    assert not negative.codes.any()
    assert np.all(dequantize_rows(negative) == -values)


def test_minimum_edge_has_positive_scale():
    values = np.linspace(-2.0, 1.0, 64, dtype=F)[None]
    result = quantize_rows(values)
    assert bf16_bits_to_f32(result.scales)[0, 0] > 0
    assert bf16_bits_to_f32(result.biases)[0, 0] == F(-2.0)
    assert result.codes[0, 0] == 0 and result.codes[0, -1] == 255


def test_maximum_edge_has_negative_scale():
    values = np.linspace(-1.0, 2.0, 64, dtype=F)[None]
    result = quantize_rows(values)
    assert bf16_bits_to_f32(result.scales)[0, 0] < 0
    assert bf16_bits_to_f32(result.biases)[0, 0] == F(2.0)
    assert result.codes[0, -1] == 0 and result.codes[0, 0] == 255


def test_ties_in_magnitude_pick_the_maximum():
    values = np.linspace(-1.0, 1.0, 64, dtype=F)[None]
    result = quantize_rows(values)
    assert bf16_bits_to_f32(result.biases)[0, 0] == F(1.0)
    assert result.codes[0, -1] == 0


def test_far_edge_clamps_to_255():
    # q0 rounds up in magnitude, so the refined scale is too small for the far
    # edge, whose unclamped code exceeds 255.5.
    low = F(-0.3965)
    values = np.linspace(low, 1.0, 64, dtype=F)[None]
    scale0 = F(F(1.0) - low) / F(255)
    q0 = round_half_away(np.array([F(1.0) / -scale0], dtype=F))[0]
    refined = F(F(1.0) / q0)
    assert F(low - F(1.0)) / refined > 255.5
    result = quantize_rows(values)
    assert result.codes[0, 0] == 255 and result.codes[0, -1] == 0


def test_matches_scalar_reference_on_random_rows():
    rng = np.random.default_rng(7)
    values = (rng.standard_normal((6, 256)) * rng.uniform(1e-4, 3.0, (6, 1))).astype(F)
    values = bf16_round(values)  # source tensors are BF16
    result = quantize_rows(values)
    for row in range(values.shape[0]):
        for group in range(values.shape[1] // 64):
            codes, scale, bias = reference_group(values[row, group * 64 : (group + 1) * 64])
            assert np.array_equal(result.codes[row, group * 64 : (group + 1) * 64], codes)
            assert result.scales[row, group] == f32_to_bf16_bits(np.array([scale], dtype=F))[0]
            assert result.biases[row, group] == f32_to_bf16_bits(np.array([bias], dtype=F))[0]
    # Half a step, plus BF16 rounding of the scale (times up to 255) and bias.
    error = np.abs(dequantize_rows(result) - values).max(axis=1)
    assert np.all(error <= 0.014 * np.abs(values).max(axis=1))


def test_codes_pack_little_endian_into_u32():
    rows = AffineRows(
        np.arange(1, 9, dtype=np.uint8).reshape(1, 8), np.zeros((1, 1), np.uint16), np.zeros((1, 1), np.uint16)
    )
    assert rows.packed_codes().tolist() == [[0x04030201, 0x08070605]]


def test_rejects_unsupported_shapes_and_values():
    with pytest.raises(ValueError):
        quantize_rows(np.zeros((1, 96), dtype=F))  # MLX requires a multiple of the group
    with pytest.raises(ValueError):
        quantize_rows(np.zeros((1, 64), dtype=F), group_size=16)
    with pytest.raises(ValueError):
        quantize_rows(np.zeros((1, 64), dtype=np.float64))
    with pytest.raises(ValueError):
        quantize_rows(np.full((1, 64), np.nan, dtype=F))


def test_chunked_quantization_equals_one_shot():
    rng = np.random.default_rng(3)
    bits = f32_to_bf16_bits(rng.standard_normal((10, 128)).astype(F))
    raw = bits.astype("<u2").tobytes()
    whole = quantize_rows(bf16_bits_to_f32(bits))
    chunks = list(
        iter_quantized_bf16(lambda start, count: raw[start * 256 : (start + count) * 256], (10, 128), rows_per_chunk=3)
    )
    assert [start for start, _ in chunks] == [0, 3, 6, 9]
    assert np.array_equal(np.concatenate([c.codes for _, c in chunks]), whole.codes)
    assert np.array_equal(np.concatenate([c.scales for _, c in chunks]), whole.scales)


def test_rename_rules_for_quantized_tensors():
    names = {
        "model.llm.embed.weight": "model.language_model.embed_tokens",
        "model.llm.unembed.weight": "lm_head",
        "model.llm.layers.3.attn.wq_du.weight": "model.language_model.layers.3.self_attn.q_proj",
        "model.llm.layers.3.attn.wk_dv.weight": "model.language_model.layers.3.self_attn.k_proj",
        "model.llm.layers.3.attn.wv_dv.weight": "model.language_model.layers.3.self_attn.v_proj",
        "model.llm.layers.3.attn.wr_du.weight": "model.language_model.layers.3.self_attn.r_proj",
        "model.llm.layers.3.attn.wo_ud.weight": "model.language_model.layers.3.self_attn.o_proj",
        "model.llm.layers.3.attn.gate_attn.weight": "model.language_model.layers.3.self_attn.gate_proj",
        "model.llm.layers.3.attn.attn_sink_proj.weight": "model.language_model.layers.3.self_attn.sink_proj",
        "model.llm.layers.5.attn.k_sconv.weight_proj": "model.language_model.layers.5.self_attn.k_sconv.weight_proj",
        "model.llm.layers.5.attn.v_sconv.weight_proj": "model.language_model.layers.5.self_attn.v_sconv.weight_proj",
    }
    others = [
        "model.llm.layers.3.attn.rel_logits_proj.proj",
        "model.llm.layers.3.attn.k_sconv.weight",
        "model.llm.layers.3.mlp.gate.weight",
        "model.llm.layers.3.controllers.k.down_weight",
        "model.llm.layers.3.mlp.experts.w2_weight",
        "model.visual.backbone.layers.0.self_attn.q_proj.weight",
    ]
    mapping = inkling_affine8.quantized_mapping(dict.fromkeys([*names, *others]))
    assert mapping == names
    assert inkling_affine8.pack_name("model.llm.attn_res.attn_queries.0") == (
        "model.language_model.layers.1.attention_input.query"
    )
    assert inkling_affine8.pack_name("model.llm.layers.3.controllers.attn.gain_weight") == (
        "model.language_model.layers.3.q_message_gate.message_gate_proj.weight"
    )


def _save(path: Path, tensors: dict) -> None:
    header, payload, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        payload.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


def test_verify_detects_identity_and_differences(tmp_path, capsys):
    rng = np.random.default_rng(11)
    source_bits = f32_to_bf16_bits(rng.standard_normal((8, 128)).astype(F))
    ours = quantize_rows(bf16_bits_to_f32(source_bits))
    original, pack = tmp_path / "original", tmp_path / "pack"
    original.mkdir()
    pack.mkdir()
    _save(original / "model.safetensors", {"model.llm.layers.0.attn.wq_du.weight": ("BF16", source_bits)})
    codes = ours.codes.copy()
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    pack_tensors = {
        f"{prefix}.weight": ("U32", codes.view("<u4")),
        f"{prefix}.scales": ("BF16", ours.scales),
        f"{prefix}.biases": ("BF16", ours.biases),
    }
    _save(pack / "model.safetensors", pack_tensors)
    assert inkling_affine8.main(["verify", str(original), str(pack)]) == 0
    assert '"byte_identical": 1' in capsys.readouterr().out

    codes[2, 5] = codes[2, 5] + 3 if codes[2, 5] < 128 else codes[2, 5] - 3
    scales = ours.scales.copy()
    scales[1, 0] ^= 1
    pack_tensors[f"{prefix}.weight"] = ("U32", codes.view("<u4"))
    pack_tensors[f"{prefix}.scales"] = ("BF16", scales)
    _save(pack / "model.safetensors", pack_tensors)
    assert inkling_affine8.main(["verify", str(original), str(pack)]) == 1
    record = json.loads(capsys.readouterr().out.splitlines()[1])
    assert record["codes"]["differing"] == 1 and record["codes"]["max_abs_diff"] == 3
    assert record["scales"]["differing"] == 1 and record["biases"]["differing"] == 0
