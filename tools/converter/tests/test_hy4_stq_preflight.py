import json
import struct

import pytest

from stark_convert._build_metadata import ConversionError
from stark_convert.hy4_stq_preflight import ENGRAM_FILES, PACK_REVISION, _check_config
from stark_convert.streaming import read_safetensors_header


def _write_padded_safetensors(path, padding: bytes) -> None:
    header = json.dumps(
        {
            "first": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
            "second": {"dtype": "U8", "shape": [1], "data_offsets": [1 + len(padding), 2 + len(padding)]},
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"a" + padding + b"b")


def _compatible_config() -> dict:
    return {
        "model_type": "hy_v4_flash",
        "architectures": ["HYV4FlashForConditionalGeneration"],
        "dtype": "bfloat16",
        "pack_revision": {"id": PACK_REVISION, "unstamped": list(ENGRAM_FILES)},
        "trunk_quant": {
            "format": "stq",
            "k": 6,
            "codebook": "spec1",
            "window": 15,
            "codebook_table": "spec1-w15",
            "codebook_digest": "37f72431",
        },
        "expert_quant": {
            "format": "stq",
            "k": 4,
            "codebook": "pair1",
            "window": 16,
            "codebook_table": "pair1-w16-v1",
            "codebook_digest": "c089b4eb",
            "tile_order": "pair_major",
        },
        "drafter_quant": {
            "format": "stq",
            "k": 4,
            "codebook": "spec1",
            "window": 15,
            "codebook_table": "spec1-w15",
            "codebook_digest": "37f72431",
        },
        "engram_tables": "bf16",
        "text_config": {
            "model_type": "hy_v4_flash",
            "num_hidden_layers": 42,
            "n_routed_experts": 512,
            "num_experts_per_tok": 16,
            "engram_layer_ids": [2, 12, 22, 32],
            "mlp_layer_types": ["dense", *(["sparse"] * 41)],
        },
        "vision_config": {"model_type": "hy_v4_flash", "hidden_size": 1152, "num_hidden_layers": 27},
        "speculative_config": {"method": "dfly", "model": "dfly", "num_speculative_tokens": 5},
        "dfly_num_hidden_layers": 5,
        "dfly_enable_hidden_correction": True,
    }


def test_stq_header_padding_is_opt_in_zero_filled_and_bounded(tmp_path):
    path = tmp_path / "aligned.safetensors"
    _write_padded_safetensors(path, bytes(4))

    with pytest.raises(ConversionError, match="gaps"):
        read_safetensors_header(path)

    _metadata, tensors = read_safetensors_header(path, max_zero_padding_bytes=4)
    assert tensors["first"].nbytes == tensors["second"].nbytes == 1

    _write_padded_safetensors(path, b"\x00\x00\x00\x01")
    with pytest.raises(ConversionError, match="not zero"):
        read_safetensors_header(path, max_zero_padding_bytes=4)

    _write_padded_safetensors(path, bytes(5))
    with pytest.raises(ConversionError, match="gaps"):
        read_safetensors_header(path, max_zero_padding_bytes=4)

    _write_padded_safetensors(path, bytes(4))
    with path.open("ab") as stream:
        stream.write(b"\x00")
    with pytest.raises(ConversionError, match="trailing data"):
        read_safetensors_header(path, max_zero_padding_bytes=4)


def test_stq_pack_config_rejects_a_changed_expert_contract():
    config = _compatible_config()
    _check_config(config)

    config["expert_quant"]["tile_order"] = "expert_major"
    with pytest.raises(ConversionError, match="tile_order"):
        _check_config(config)
