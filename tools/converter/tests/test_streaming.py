"""Synthetic fixtures exercise format validation and bit-preserving transforms."""

import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from stark_convert import streaming
from stark_convert.qwen3 import build_qwen3_plan
from stark_convert.streaming import (
    ConversionError,
    UnsupportedConversion,
    convert_streaming,
    load_checkpoint,
    read_safetensors_header,
    tensor_sha256,
    transpose_tensor,
)


def save_tensors(path: Path, tensors: dict, *, trailing: bytes = b""):
    header, payload, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        payload.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload) + trailing)


def qwen_fixture(path: Path, *, dtype="BF16", tied=False, attention_bias=False, sharded=False):
    path.mkdir()
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 64,
        "tie_word_embeddings": tied,
        "attention_bias": attention_bias,
        "max_window_layers": 0,
        "use_sliding_window": True,
        "sliding_window": 16,
        "eos_token_id": 1,
        "torch_dtype": {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[dtype],
    }
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}
    raw_dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[dtype]

    def add(name, shape):
        base = len(tensors) * 37
        tensors[name] = (dtype, np.arange(base, base + np.prod(shape), dtype=raw_dtype).reshape(shape))

    add("model.embed_tokens.weight", (16, 8))
    add("lm_head.weight", (16, 8))
    if tied:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    add("model.norm.weight", (8,))
    layer = "model.layers.0"
    add(layer + ".input_layernorm.weight", (8,))
    add(layer + ".post_attention_layernorm.weight", (8,))
    add(layer + ".self_attn.q_norm.weight", (2,))
    add(layer + ".self_attn.k_norm.weight", (2,))
    for name, width in (("q", 8), ("k", 4), ("v", 4)):
        add(layer + f".self_attn.{name}_proj.weight", (width, 8))
        if attention_bias:
            add(layer + f".self_attn.{name}_proj.bias", (width,))
    add(layer + ".self_attn.o_proj.weight", (8, 8))
    if attention_bias:
        add(layer + ".self_attn.o_proj.bias", (8,))
    add(layer + ".mlp.up_proj.weight", (12, 8))
    add(layer + ".mlp.gate_proj.weight", (12, 8))
    add(layer + ".mlp.down_proj.weight", (8, 12))
    if sharded:
        # Alternating assignment deliberately places fused projections in different shards.
        shards = [dict(list(tensors.items())[index::2]) for index in range(2)]
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        weight_map = {}
        for name, contents in zip(names, shards, strict=True):
            save_tensors(path / name, contents)
            weight_map.update(dict.fromkeys(contents, name))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_tensors(path / "model.safetensors", tensors)
    return config, tensors


def tensor_bytes(info):
    with info.path.open("rb") as stream:
        stream.seek(info.offset)
        return stream.read(info.nbytes)


@pytest.mark.parametrize("dtype,requested", [("BF16", "bfloat16"), ("F16", "float16"), ("F32", "float32")])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_matches_lalamo_layout_and_fusion(tmp_path, dtype, requested, sharded, tied):
    source, output = tmp_path / "source", tmp_path / "output"
    config, tensors = qwen_fixture(source, dtype=dtype, tied=tied, attention_bias=True, sharded=sharded)
    output.mkdir()
    stats = convert_streaming(source, output, dtype=requested, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output / "model.safetensors")
    assert stats["peak_buffer_bytes"] <= 96
    assert stats["tensors"] == 12 if tied else stats["tensors"] == 13
    decoder = stats["decoder_config"]
    assert decoder["transformer_config"]["layer_configs"][0]["mixer_config"]["sliding_window_size"] == 16
    assert decoder["transformer_config"]["model_dim"] == config["hidden_size"]
    embed_name = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    assert tensor_bytes(converted[embed_name + ".weights"]) == tensors["model.embed_tokens.weight"][1].tobytes()
    assert json.loads(metadata[embed_name + ".spec"]) == {"type": "FullPrecisionSpec", "layout": "input_output"}
    prefix = "decoder.transformer.layers.0"
    qkv_name = prefix + ".mixer.qkvg_projection.weights.weights"
    assert converted[qkv_name].shape == (16, 8)
    assert tensor_bytes(converted[qkv_name]) == b"".join(
        tensors[f"model.layers.0.self_attn.{name}_proj.weight"][1].tobytes() for name in ("q", "k", "v")
    )
    up_name = prefix + ".mlp.up_projection.weights.weights"
    assert tensor_bytes(converted[up_name]) == b"".join(
        tensors[f"model.layers.0.mlp.{name}_proj.weight"][1].tobytes() for name in ("up", "gate")
    )
    bias_name = prefix + ".mixer.qkvg_projection.biases"
    assert tensor_bytes(converted[bias_name]) == b"".join(
        tensors[f"model.layers.0.self_attn.{name}_proj.bias"][1].tobytes() for name in ("q", "k", "v")
    )
    scale = converted["decoder.transformer.output_norm.scales"]
    source_scale = tensors["model.norm.weight"][1]
    expected = (
        (source_scale.astype("<u4") << 16).tobytes() if dtype == "BF16" else source_scale.astype("<f4").tobytes()
    )
    assert scale.dtype == "F32"
    assert tensor_bytes(scale) == expected
    assert list(converted) == sorted(converted, key=lambda name: (-converted[name].itemsize, name))
    assert stats["output_bytes"] == (output / "model.safetensors").stat().st_size


@pytest.mark.parametrize("shape", [(13, 7), (1, 19), (18, 1), (0, 9)])
@pytest.mark.parametrize("dtype,raw_dtype", [("BF16", "<u2"), ("F32", "<u4")])
def test_tiled_transpose_preserves_bits_and_bounds_buffers(tmp_path, shape, dtype, raw_dtype):
    source = tmp_path / "source.safetensors"
    matrix = np.arange(np.prod(shape), dtype=raw_dtype).reshape(shape)
    save_tensors(source, {"matrix": (dtype, matrix)})
    info = read_safetensors_header(source)[1]["matrix"]
    output = io.BytesIO(b"prefix")
    output.seek(6)
    peak = transpose_tensor(info, output, chunk_bytes=64)
    assert output.getvalue() == b"prefix" + matrix.T.tobytes()
    assert peak <= 64


@pytest.mark.parametrize(
    "case", ["truncated_prefix", "short_header", "duplicate", "overlap", "gap", "size", "trailing"]
)
def test_malformed_safetensors_are_errors(tmp_path, case):
    path = tmp_path / "invalid.safetensors"
    info = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
    payload = b"\0" * 4
    if case == "truncated_prefix":
        path.write_bytes(b"\0" * 7)
    elif case == "short_header":
        path.write_bytes(struct.pack("<Q", 1000) + b"{}")
    else:
        header = {"a": info}
        if case == "duplicate":
            encoded = b'{"a":' + json.dumps(info).encode() + b',"a":' + json.dumps(info).encode() + b"}"
        else:
            if case == "overlap":
                header["b"] = info
            elif case == "gap":
                info["data_offsets"] = [2, 6]
                payload += b"\0" * 2
            elif case == "size":
                info["shape"] = [3]
            elif case == "trailing":
                payload += b"\0"
            encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    with pytest.raises(ConversionError):
        read_safetensors_header(path)


def test_unknown_tensors_are_rejected_before_writing(tmp_path):
    source = tmp_path / "source"
    _, tensors = qwen_fixture(source)
    tensors["unexpected.weight"] = ("BF16", np.zeros(2, dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="unexpected tensors"):
        build_qwen3_plan(source)


def test_requested_qwen3_dtype_change_is_unsupported(tmp_path):
    source = tmp_path / "source"
    qwen_fixture(source, dtype="F32")
    with pytest.raises(UnsupportedConversion, match="dense dtype conversion"):
        build_qwen3_plan(source, dtype="bfloat16")


@pytest.mark.parametrize("malformation", ["extra", "shape", "missing"])
def test_dtype_fallback_cannot_hide_malformed_qwen3(tmp_path, malformation):
    source = tmp_path / "source"
    _, tensors = qwen_fixture(source, dtype="F32")
    if malformation == "extra":
        tensors["unexpected.weight"] = ("F32", np.zeros(2, dtype="<f4"))
    elif malformation == "shape":
        tensors["model.layers.0.self_attn.q_proj.weight"] = ("F32", np.zeros((1, 8), dtype="<f4"))
    else:
        del tensors["model.layers.0.self_attn.q_proj.weight"]
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_qwen3_plan(source, dtype="bfloat16")


def test_missing_or_wrong_shape_tensor_does_not_select_fallback(tmp_path):
    source = tmp_path / "source"
    _, tensors = qwen_fixture(source)
    tensors["model.layers.0.self_attn.q_proj.weight"] = ("BF16", np.zeros((1, 8), dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="shape"):
        build_qwen3_plan(source)
    del tensors["model.layers.0.self_attn.q_proj.weight"]
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="missing"):
        build_qwen3_plan(source)


def test_tied_lm_head_must_be_identical(tmp_path):
    source = tmp_path / "source"
    _, tensors = qwen_fixture(source, tied=True)
    tensors["lm_head.weight"] = ("BF16", np.zeros((16, 8), dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="Tied lm_head differs"):
        build_qwen3_plan(source)


def test_unknown_semantic_config_selects_fallback(tmp_path):
    source = tmp_path / "source"
    config, _ = qwen_fixture(source)
    config["rope_scaling"] = {"rope_type": "yarn", "factor": 2.0}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="RoPE"):
        build_qwen3_plan(source)


def test_index_missing_duplicate_and_traversal_rejected(tmp_path):
    source = tmp_path / "source"
    qwen_fixture(source, sharded=True)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    original = index_path.read_bytes()
    index["weight_map"]["missing.weight"] = "model-00001-of-00002.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ConversionError, match="missing"):
        load_checkpoint(source)
    index_path.write_bytes(original)
    index = json.loads(original)
    index["weight_map"]["model.norm.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ConversionError, match="path"):
        load_checkpoint(source)
    index_path.write_bytes(original)
    (source / "model-00002-of-00002.safetensors").write_bytes(
        (source / "model-00001-of-00002.safetensors").read_bytes()
    )
    with pytest.raises(ConversionError):
        load_checkpoint(source)


def test_sharded_hf_snapshot_symlinks_are_accepted(tmp_path):
    source = tmp_path / "source"
    qwen_fixture(source, sharded=True)
    blob = tmp_path / "blob"
    shard = source / "model-00001-of-00002.safetensors"
    shard.rename(blob)
    shard.symlink_to(blob)
    assert len(load_checkpoint(source)) == 14


def test_checkpoint_changed_after_planning_is_rejected(tmp_path):
    source = tmp_path / "source"
    qwen_fixture(source)
    info = load_checkpoint(source)["model.norm.weight"]
    with info.path.open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ConversionError, match="changed"):
        tensor_sha256(info)


def test_checkpoint_changed_during_last_copy_is_rejected(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    qwen_fixture(source)
    output.mkdir()
    plan = build_qwen3_plan(source)
    last_source = sorted(plan.operations, key=lambda op: (-streaming.DTYPE_BYTES[op.dtype], op.name))[-1].sources[-1]
    original_copy = streaming._copy

    def mutate_after_last_copy(tensor, target, chunk_bytes):
        peak = original_copy(tensor, target, chunk_bytes)
        if tensor == last_source:
            with tensor.path.open("ab") as handle:
                handle.write(b"\0")
        return peak

    monkeypatch.setattr(streaming, "_copy", mutate_after_last_copy)
    with pytest.raises(ConversionError, match="changed"):
        streaming.write_plan(plan, output / "model.safetensors")


def test_omitted_tied_head_shard_is_rechecked_after_write(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    _, tensors = qwen_fixture(source, tied=True)
    (source / "model.safetensors").unlink()
    head = {"lm_head.weight": tensors.pop("lm_head.weight")}
    head_path = source / "head.safetensors"
    save_tensors(head_path, head)
    save_tensors(source / "weights.safetensors", tensors)
    index = {"weight_map": {**dict.fromkeys(tensors, "weights.safetensors"), "lm_head.weight": "head.safetensors"}}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    output.mkdir()
    original_copy = streaming._copy
    mutated = False

    def mutate_omitted_head_after_copy(tensor, target, chunk_bytes):
        nonlocal mutated
        peak = original_copy(tensor, target, chunk_bytes)
        if not mutated:
            with head_path.open("ab") as handle:
                handle.write(b"\0")
            mutated = True
        return peak

    monkeypatch.setattr(streaming, "_copy", mutate_omitted_head_after_copy)
    with pytest.raises(ConversionError, match="changed"):
        convert_streaming(source, output)


def test_duplicate_config_keys_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3","model_type":"llama"}')
    with pytest.raises(ConversionError, match="duplicate"):
        build_qwen3_plan(source)


@pytest.mark.parametrize("tied,attention_bias", [(True, False), (False, True)])
def test_decoder_config_matches_frozen_reference(tmp_path, tied, attention_bias):
    source = tmp_path / "source"
    qwen_fixture(source, tied=tied, attention_bias=attention_bias)
    baseline = json.loads((Path(__file__).parent / "fixtures" / "qwen3_decoder_configs.json").read_text())
    assert baseline["reference_revision"] == "831accb8252a6b6606c378048caeb44178f1a050"
    assert build_qwen3_plan(source).decoder_config == baseline["cases"][str(tied)]
