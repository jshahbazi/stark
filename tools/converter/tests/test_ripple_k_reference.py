"""Independent K-block oracle indexing, container opt-in and row addressing."""

import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stark_convert.gguf_import import build_import_plan, read_gguf
from stark_convert.streaming import UnsupportedConversion

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("ripple_full_reference", SCRIPTS / "ripple_full_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def scalar_decode(block, kind):
    """Element-index oracle; no vectorized decoder helpers are reused."""
    result = []
    for index in range(256):
        if kind == "Q4_K":
            d, minimum = struct.unpack_from("<ee", block)
            group = index // 32
            if group < 4:
                scale, shift = block[4 + group] & 63, block[8 + group] & 63
            else:
                scale = (block[8 + group] & 15) + ((block[group] >> 6) * 16)
                shift = (block[8 + group] >> 4) + ((block[4 + group] >> 6) * 16)
            packed = block[16 + (index // 64) * 32 + index % 32]
            code = (packed >> ((group % 2) * 4)) & 15
            result.append(np.float32(np.float32(d * scale) * code) - np.float32(minimum * shift))
        else:
            d = struct.unpack_from("<e", block, 208)[0]
            half, within = divmod(index, 128)
            group, lane = divmod(within, 32)
            low = block[half * 64 + (group % 2) * 32 + lane]
            high = block[128 + half * 32 + lane]
            code = ((low >> ((group // 2) * 4)) & 15) + (((high >> (group * 2)) & 3) * 16) - 32
            scale = struct.unpack_from("b", block, 192 + half * 8 + group * 2 + lane // 16)[0]
            result.append(np.float32(np.float32(d * scale) * code))
    return np.array(result, dtype=np.float32)


def blocks(kind, count=7):
    size = 144 if kind == "Q4_K" else 210
    rng = np.random.default_rng(18471)
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    for i, block in enumerate(raw):
        offset = 0 if kind == "Q4_K" else 208
        block[offset : offset + 2] = np.frombuffer(struct.pack("<e", (-1) ** i * (i + 1) / 32), dtype=np.uint8)
        if kind == "Q4_K":
            block[2:4] = np.frombuffer(struct.pack("<e", (i + 1) / 64), dtype=np.uint8)
    return raw


@pytest.mark.parametrize("kind", ["Q4_K", "Q6_K"])
def test_k_reference_matches_element_oracle(kind):
    raw = blocks(kind)
    actual = reference.dequantize_k_blocks(raw.tobytes(), kind)
    expected = np.stack([scalar_decode(row.tobytes(), kind) for row in raw])
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert (actual > 0).any() and (actual < 0).any()
    with pytest.raises(ValueError, match="payload"):
        reference.dequantize_k_blocks(raw.tobytes()[:-1], kind)


def text(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


@pytest.mark.parametrize("kind,type_id", [("Q4_K", 12), ("Q6_K", 14)])
def test_k_container_opt_in_and_embedding_row_offsets(tmp_path, monkeypatch, kind, type_id):
    raw = blocks(kind, 4)
    # Two 512-wide rows; each row contains two distinct blocks.
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    header += text("general.architecture") + struct.pack("<I", 8) + text("ricursive")
    header += text("token_embd.weight") + struct.pack("<IQQIQ", 2, 512, 2, type_id, 0)
    header += b"\0" * (-len(header) % 32)
    path = tmp_path / "weights.gguf"
    path.write_bytes(header + raw.tobytes())
    with pytest.raises(UnsupportedConversion):
        read_gguf(path)
    source = read_gguf(path, reference_k_quants=True)
    assert source.tensors["token_embd.weight"].kind == kind
    with pytest.raises(UnsupportedConversion, match="Q8_0, F16 and F32"):
        build_import_plan(source, tmp_path / "no-hf-source")
    weights = reference.Q8Weights(tmp_path)
    # Only the final framework conversion is stubbed: exercise real file/row
    # addressing and NumPy decoding without adding torch to converter deps.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda value: value))
    first = weights._read("token_embd.weight", 0, 1)
    second = weights._read("token_embd.weight", 1, 1)
    expected = np.stack([scalar_decode(row.tobytes(), kind) for row in raw]).reshape(2, 512)
    np.testing.assert_array_equal(first, expected[:1])
    np.testing.assert_array_equal(second, expected[1:])
    assert not np.array_equal(first, second)
