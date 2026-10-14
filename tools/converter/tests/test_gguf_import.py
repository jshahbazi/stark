"""Independent synthetic GGUF block/layout and fail-closed import coverage."""

import json
import struct
from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest
from tokenizers import Tokenizer, models

from stark_convert.gguf_import import (
    _hf_names,
    build_import_plan,
    import_qwen3,
    main,
    read_gguf,
    validate_codes,
    validate_metadata,
    write_import,
)
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header


def _string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _value(value):
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    if type(value) is int:
        return struct.pack("<II", 4, value)
    if type(value) is float:
        return struct.pack("<If", 6, value)
    if isinstance(value, list):
        return struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_string(item) for item in value)
    raise AssertionError("unsupported fixture metadata")


def gguf_fixture(tmp_path, *, tied=False, dense_matrix=None, type_override=None, overlap=False):
    hf = tmp_path / "hf"
    hf.mkdir()
    config = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 33,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "max_position_embeddings": 64,
        "tie_word_embeddings": tied,
        "eos_token_id": 1,
        "bos_token_id": 0,
    }
    (hf / "config.json").write_text(json.dumps(config))
    tokens = [f"token{index}" for index in range(config["vocab_size"])]
    Tokenizer(models.BPE(dict(zip(tokens, range(len(tokens)), strict=True)), [])).save(str(hf / "tokenizer.json"))
    (hf / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"})
    )
    (hf / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    metadata = {
        "general.architecture": "qwen3",
        "qwen3.embedding_length": 32,
        "qwen3.feed_forward_length": 64,
        "qwen3.block_count": 1,
        "qwen3.attention.head_count": 2,
        "qwen3.attention.head_count_kv": 1,
        "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen3.attention.key_length": 16,
        "qwen3.attention.value_length": 16,
        "qwen3.context_length": 64,
        "qwen3.rope.freq_base": 1000000.0,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.bos_token_id": 0,
    }
    tensors = {}
    for name in _hf_names(config):
        if name == "output_norm.weight" or "_norm.weight" in name:
            shape = (16,) if name.endswith(("q_norm.weight", "k_norm.weight")) else (32,)
            values = np.arange(shape[0], dtype="<f4") / 100 + 1
            tensors[name] = (0, shape, values.tobytes())
        else:
            if name in {"token_embd.weight", "output.weight"}:
                shape = (33, 32)
            elif name.endswith(("attn_k.weight", "attn_v.weight")):
                shape = (16, 32)
            elif name.endswith(("ffn_up.weight", "ffn_gate.weight")):
                shape = (64, 32)
            elif name.endswith("ffn_down.weight"):
                shape = (32, 64)
            else:
                shape = (32, 32)
            count = np.prod(shape) // 32
            scales = (0.011 + np.arange(count) % 5 * 0.0013).astype("<f2")
            codes = ((np.arange(count * 32) * 17 + len(tensors) * 7) % 256 - 128).astype("i1").reshape(count, 32)
            blocks = np.empty((count, 34), dtype="u1")
            blocks[:, :2] = scales.view("u1").reshape(count, 2)
            blocks[:, 2:] = codes.view("u1")
            kind, payload = 8, blocks.tobytes()
            if name == dense_matrix:
                kind = 1
                payload = (codes.astype("<f4") * scales[:, None]).astype("<f2").tobytes()
            tensors[name] = (kind, shape, payload)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    header += b"".join(_string(key) + _value(value) for key, value in metadata.items())
    data, offset = b"", 0
    for index, (name, (kind, shape, payload)) in enumerate(tensors.items()):
        data += bytes(-len(data) % 32)
        offset = len(data)
        if overlap and index == 1:
            offset = 0
        header += _string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape[::-1])
        header += struct.pack("<IQ", type_override if type_override is not None else kind, offset)
        data += payload
    header += bytes(-len(header) % 32)
    path = tmp_path / "source.gguf"
    path.write_bytes(header + data)
    return path, hf, config, tensors


def _bytes(info):
    with info.path.open("rb") as stream:
        stream.seek(info.offset)
        return stream.read(info.nbytes)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("chunk_bytes", [4096, 8192, 1024 * 1024])
def test_q8_codes_shapes_fusion_coefficients_and_error(tmp_path, tied, chunk_bytes):
    path, hf, _, originals = gguf_fixture(tmp_path, tied=tied)
    source = read_gguf(path)
    plan, operations, metadata, _ = build_import_plan(source, hf)
    output = tmp_path / "converted.safetensors"
    stats = write_import(source, operations, metadata, output, chunk_bytes=chunk_bytes)
    native_metadata, native = read_safetensors_header(output)
    by_offset = {tensor.offset: originals[name] for name, tensor in source.tensors.items()}
    squared_error = squared_reference = 0.0
    maximum_error = 0.0
    total_codes = 0
    for op in operations:
        if op.kind != "gguf_q8_codes":
            continue
        prefix = op.name.removesuffix(".weights")
        codes, coefficients = [], []
        for info in op.sources:
            blocks = np.frombuffer(by_offset[info.offset][2], dtype="u1").reshape(-1, 34)
            signed = blocks[:, 2:].view("i1").astype("<f8")
            original_scales = blocks[:, :2].copy().view("<f2").reshape(-1).astype("<f8")
            bf16_scales = original_scales.astype(ml_dtypes.bfloat16).astype("<f8")
            codes.append((signed.astype("<i2") + 128).astype("u1").tobytes())
            coefficients.append(bf16_scales.reshape(info.shape[0], info.shape[1] // 32))
            reference = original_scales[:, None] * signed
            adapted = bf16_scales[:, None] * signed
            squared_error += float(np.sum((reference - adapted) ** 2))
            squared_reference += float(np.sum(reference**2))
            maximum_error = max(maximum_error, float(np.max(np.abs(reference - adapted))))
            total_codes += signed.size
        assert _bytes(native[op.name]) == b"".join(codes)
        expected = np.concatenate(coefficients).astype(ml_dtypes.bfloat16)
        spec = json.loads(native_metadata[prefix + ".spec"])
        assert spec["type"] == "IntSpec" and spec["is_symmetric"] and spec["group_size"] == 32
        if spec["layout"] == "output_input":
            expected = np.pad(expected.T, ((0, 0), (0, -expected.shape[0] % 4)))
        assert _bytes(native[prefix + ".scales"]) == expected.tobytes()
    verified = validate_codes(source, operations, output, chunk_bytes)
    assert verified["compared_codes"] == total_codes and verified["mismatched_codes"] == 0
    assert verified["compared_scales"] == total_codes // 32 and verified["mismatched_scale_bits"] == 0
    assert stats["peak_buffer_bytes"] <= chunk_bytes
    assert stats["scale_f16_to_bf16"]["weight_relative_rms_error"] == pytest.approx(
        np.sqrt(squared_error / squared_reference), rel=1e-12
    )
    assert stats["scale_f16_to_bf16"]["weight_max_absolute_error"] == maximum_error
    assert stats["scale_f16_to_bf16"]["weight_count"] == total_codes
    assert len(plan.decoder_config["transformer_config"]["layer_configs"]) == 1


def test_dense_matrix_preserved_and_mixed_fusion_rejected(tmp_path):
    path, hf, _, _ = gguf_fixture(tmp_path, dense_matrix="blk.0.ffn_down.weight")
    source = read_gguf(path)
    _, operations, metadata, _ = build_import_plan(source, hf)
    output = tmp_path / "output.safetensors"
    write_import(source, operations, metadata, output, chunk_bytes=4096)
    prefix = "decoder.transformer.layers.0.mlp.down_projection.weights"
    assert json.loads(metadata[prefix + ".spec"])["type"] == "FullPrecisionSpec"
    original = source.tensors["blk.0.ffn_down.weight"]
    changed = dict(source.tensors)
    changed["blk.0.attn_k.weight"] = replace(source.tensors["blk.0.attn_k.weight"], kind="F16")
    with pytest.raises(UnsupportedConversion, match="Mixed-dtype|Mixed or nonmatrix"):
        build_import_plan(replace(source, tensors=changed), hf)
    assert original.kind == "F16"


@pytest.mark.parametrize(
    "key", ["qwen3.embedding_length", "tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id"]
)
def test_metadata_mismatches_rejected(tmp_path, key):
    path, hf, config, _ = gguf_fixture(tmp_path)
    source = read_gguf(path)
    metadata = dict(source.metadata)
    if key.endswith("tokens"):
        metadata[key] = [*metadata[key]]
        metadata[key][1] = "changed"
    elif key.endswith("merges"):
        metadata[key] = ["changed merge"]
    else:
        metadata[key] += 1
    with pytest.raises(ConversionError):
        validate_metadata(replace(source, metadata=metadata), hf, config)


@pytest.mark.parametrize("variant", ["unknown_type", "overlap", "truncated", "trailing", "wrong_version"])
def test_invalid_gguf_rejected(tmp_path, variant):
    path, _, _, _ = gguf_fixture(
        tmp_path, type_override=9 if variant == "unknown_type" else None, overlap=variant == "overlap"
    )
    if variant == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif variant == "trailing":
        path.write_bytes(path.read_bytes() + b"x")
    elif variant == "wrong_version":
        raw = bytearray(path.read_bytes())
        raw[4:8] = struct.pack("<I", 2)
        path.write_bytes(raw)
    with pytest.raises((ConversionError, UnsupportedConversion)):
        read_gguf(path)


def test_header_budget_duplicate_metadata_and_invalid_boolean(tmp_path, monkeypatch):
    path = tmp_path / "malformed.gguf"
    prefix = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    path.write_bytes(prefix + _string("bool") + struct.pack("<IB", 7, 2))
    with pytest.raises(ConversionError, match="boolean"):
        read_gguf(path)
    repeated = _string("general.architecture") + _value("qwen3")
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + repeated * 2)
    with pytest.raises(ConversionError, match="Duplicate"):
        read_gguf(path)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    path, _, _, _ = gguf_fixture(fixture)
    monkeypatch.setattr("stark_convert.gguf_import._MAX_HEADER", 128)
    with pytest.raises(UnsupportedConversion, match="header"):
        read_gguf(path)


def test_atomic_publication_private_receipt_and_no_overwrite(tmp_path):
    path, hf, _, _ = gguf_fixture(tmp_path)
    output = tmp_path / "native"
    receipt = import_qwen3(path, hf, output, source_revision="a" * 40, hf_revision="b" * 40, chunk_bytes=4096)
    assert receipt["native_execution"] == "unverified"
    assert receipt["stats"]["code_preservation"]["mismatched_codes"] == 0
    assert all(item.stat().st_mode & 0o777 == 0o600 for item in output.iterdir())
    assert output.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ConversionError, match="exists"):
        import_qwen3(path, hf, output, source_revision="a" * 40, hf_revision="b" * 40)


def test_failed_import_removes_staging_and_cli_hides_exceptions(tmp_path, monkeypatch, capsys):
    path, hf, _, _ = gguf_fixture(tmp_path)
    monkeypatch.setattr(
        "stark_convert.gguf_import.write_metadata", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("private"))
    )
    output = tmp_path / "native"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gguf_import",
            "--source",
            str(path),
            "--hf-source",
            str(hf),
            "--output",
            str(output),
            "--source-revision",
            "a" * 40,
            "--hf-revision",
            "b" * 40,
        ],
    )
    assert main() == 1
    assert "private" not in capsys.readouterr().out
    assert not output.exists() and not list(tmp_path.glob(".gguf-staging-*"))


def test_nonfinite_scales_and_tampered_native_codes_fail(tmp_path):
    path, hf, _, _ = gguf_fixture(tmp_path)
    source = read_gguf(path)
    _, operations, metadata, _ = build_import_plan(source, hf)
    output = tmp_path / "converted.safetensors"
    write_import(source, operations, metadata, output, chunk_bytes=4096)
    op = next(item for item in operations if item.kind == "gguf_q8_codes")
    location = read_safetensors_header(output)[1][op.name].offset
    with output.open("r+b") as stream:
        stream.seek(location)
        value = stream.read(1)[0]
        stream.seek(location)
        stream.write(bytes([value ^ 1]))
    with pytest.raises(ConversionError, match="preservation"):
        validate_codes(source, operations, output, 4096)
    with output.open("r+b") as stream:
        stream.seek(location)
        stream.write(bytes([value]))
        coefficient = read_safetensors_header(output)[1][op.name.removesuffix(".weights") + ".scales"]
        stream.seek(coefficient.offset)
        value = stream.read(1)[0]
        stream.seek(coefficient.offset)
        stream.write(bytes([value ^ 1]))
    with pytest.raises(ConversionError, match="coefficient layout"):
        validate_codes(source, operations, output, 4096)
    quantized = next(item for item in source.tensors.values() if item.kind == "Q8_0")
    with path.open("r+b") as stream:
        stream.seek(quantized.offset)
        stream.write(np.array([np.inf], dtype="<f2").tobytes())
    changed = read_gguf(path)
    _, operations, metadata, _ = build_import_plan(changed, hf)
    with pytest.raises(ConversionError, match="nonfinite"):
        write_import(changed, operations, metadata, tmp_path / "bad.safetensors", chunk_bytes=4096)
