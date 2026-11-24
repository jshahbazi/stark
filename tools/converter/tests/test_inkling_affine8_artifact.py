"""Model-free checks of the Inkling 2 Mini affine8 artifact writer."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from stark_convert import inkling_affine8_artifact as artifact
from stark_convert.mlx_affine import bf16_bits_to_f32, f32_to_bf16_bits, quantize_rows
from stark_convert.streaming import read_safetensors_header

PROJECTIONS = ("wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud", "gate_attn", "attn_sink_proj")


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


def _synthetic(tmp_path: Path) -> tuple[Path, dict]:
    rng = np.random.default_rng(5)

    def matrix(rows: int = 2, columns: int = 128):
        return ("BF16", f32_to_bf16_bits(rng.standard_normal((rows, columns)).astype(np.float32)))

    tensors = {"model.llm.embed.weight": matrix(3)}
    tensors["model.llm.norm.weight"] = ("F32", rng.standard_normal(16).astype(np.float32))
    for layer in range(24):
        for name in PROJECTIONS:
            tensors[f"model.llm.layers.{layer}.attn.{name}.weight"] = matrix()
        if layer in (5, 11, 17, 23):
            tensors[f"model.llm.layers.{layer}.attn.k_sconv.weight_proj"] = matrix()
            tensors[f"model.llm.layers.{layer}.attn.v_sconv.weight_proj"] = matrix()
        tensors[f"model.llm.layers.{layer}.attn.rel_logits_proj.proj"] = matrix(1, 64)
        expert = rng.integers(0, 255, (2, 8), dtype=np.uint8)
        tensors[f"model.llm.layers.{layer}.mlp.experts.w2_weight"] = ("U8", expert)
    tensors["model.llm.unembed.weight"] = matrix(3)
    path = tmp_path / "model.safetensors"
    _save(path, tensors)
    return path, tensors


def test_target_replaces_exactly_the_affine_tensors(tmp_path):
    source, tensors = _synthetic(tmp_path)
    _, original = read_safetensors_header(source)
    assert len(artifact.quantized_names(original)) == artifact.QUANTIZED_TENSORS
    output = tmp_path / "out.safetensors"
    artifact.write_target(original, output, {"stark_layout": artifact.LAYOUT}, tensors_only=False)
    metadata, written = read_safetensors_header(output)
    assert metadata["stark_layout"] == artifact.LAYOUT
    quantized = set(artifact.quantized_names(original))
    data = output.read_bytes()

    def payload(info):
        return data[info.offset : info.offset + info.nbytes]

    for name, (_, array) in tensors.items():
        if name not in quantized:
            assert name in written and payload(written[name]) == np.ascontiguousarray(array).tobytes()
            continue
        assert name not in written
        expected = quantize_rows(bf16_bits_to_f32(array))
        codes, scales, biases = (written[f"{name}.{suffix}"] for suffix in artifact.SUFFIXES)
        assert (codes.dtype, codes.shape) == ("U8", array.shape)
        assert (scales.dtype, scales.shape) == ("BF16", (array.shape[0], array.shape[1] // 64))
        assert payload(codes) == expected.codes.tobytes()
        assert payload(scales) == expected.scales.astype("<u2").tobytes()
        assert payload(biases) == expected.biases.astype("<u2").tobytes()
    # Unquantized tensors keep their original relative order.
    order = [name for name in sorted(written, key=lambda n: written[n].offset) if not name.endswith(artifact.SUFFIXES)]
    assert order == [name for name in tensors if name not in quantized]


def test_tensors_only_writes_just_the_affine_planes(tmp_path):
    source, _ = _synthetic(tmp_path)
    _, original = read_safetensors_header(source)
    output = tmp_path / "affine.safetensors"
    artifact.write_target(original, output, {}, tensors_only=True)
    _, written = read_safetensors_header(output)
    assert len(written) == 3 * artifact.QUANTIZED_TENSORS
    assert all(name.endswith(artifact.SUFFIXES) for name in written)


def test_rejects_an_unexpected_tensor_set(tmp_path):
    source, tensors = _synthetic(tmp_path)
    del tensors["model.llm.layers.3.attn.wq_du.weight"]
    _save(source, tensors)
    _, original = read_safetensors_header(source)
    with pytest.raises(ValueError):
        artifact.quantized_names(original)


def test_installed_config_binds_the_layout():
    raw = json.dumps({"model_type": "inkling2_loop"}).encode()
    config = json.loads(artifact._installed_config(raw, b"template"))
    assert config["stark_inkling_affine8"] == {"layout": artifact.LAYOUT, "group_size": 64, "bits": 8}
    assert config["stark_inkling"]["revision"] == "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
    assert config["token_codec_config"] == {"type": "ChatCodecConfig", "prompt_template": "template"}


def _prepared(tmp_path: Path, monkeypatch) -> Path:
    """A synthetic prepared directory whose pinned metadata hashes are patched to match."""
    import hashlib

    from stark_convert import inkling_prepare

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    raw = json.dumps({"model_type": inkling_prepare.FAMILY}).encode()
    template, processor, quant = b"template", b"{}", b"{}\n"
    monkeypatch.setattr(inkling_prepare, "CONFIG_SHA", sha(raw))
    monkeypatch.setattr(inkling_prepare, "TEMPLATE_SHA", sha(template))
    monkeypatch.setattr(inkling_prepare, "PROCESSOR_SHA", sha(processor))
    monkeypatch.setattr(inkling_prepare, "QUANT_SHA", sha(quant))
    monkeypatch.setattr(inkling_prepare, "AFFINE8_TARGET_BYTES", 16)
    monkeypatch.setattr(
        inkling_prepare, "WEIGHTS", {inkling_prepare.TARGET: (32, "a" * 64), inkling_prepare.DRAFT: (8, "b" * 64)}
    )
    root = tmp_path / "prepared"
    root.mkdir()
    encoded = artifact._installed_config(raw, template)
    (root / inkling_prepare.CONFIG).write_bytes(raw)
    (root / "config.json").write_bytes(encoded)
    (root / "chat_template.jinja").write_bytes(template)
    (root / "processor_config.json").write_bytes(processor)
    (root / "hf_quant_config.json").write_bytes(quant)
    (root / "tokenizer.json").write_bytes(b"{}")
    (root / inkling_prepare.TARGET).write_bytes(b"\0" * 16)
    (root / inkling_prepare.DRAFT).write_bytes(b"\0" * 8)
    receipt = artifact._receipt(encoded, 16, inkling_prepare.AFFINE8_TARGET_SHA)
    (root / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return root


def test_prepared_directory_is_recognized_as_the_inkling_image_family(tmp_path, monkeypatch):
    from stark_convert import hybrid_prepare, inkling_prepare, vision_runtime

    assert artifact.LAYOUT == inkling_prepare.AFFINE8_LAYOUT
    root = _prepared(tmp_path, monkeypatch)
    identity = hybrid_prepare.prepared_identity(root)
    assert identity == {
        "repo": inkling_prepare.REPO,
        "revision": inkling_prepare.REVISION,
        "architecture": inkling_prepare.FAMILY,
        "weight_bytes": 24,
        "quantization": artifact.QUANTIZATION,
    }
    config = json.loads((root / "config.json").read_text())
    assert vision_runtime.model_image_family(config, root) == {"profile": inkling_prepare.VISION_PROFILE, "version": 1}


@pytest.mark.parametrize(
    "change", ["target_size", "receipt_hash", "receipt_layout", "config_binding", "draft_missing"]
)
def test_prepared_directory_fails_closed_on_any_identity_change(tmp_path, monkeypatch, change):
    from stark_convert import hybrid_prepare, inkling_prepare

    root = _prepared(tmp_path, monkeypatch)
    receipt = json.loads((root / "native-source.json").read_text())
    if change == "target_size":
        (root / inkling_prepare.TARGET).write_bytes(b"\0" * 17)
    elif change == "receipt_hash":
        receipt["artifact_sha256"] = "0" * 64
    elif change == "receipt_layout":
        receipt["layout"] = inkling_prepare.LAYOUT
    elif change == "config_binding":
        config = json.loads((root / "config.json").read_text())
        del config["stark_inkling_affine8"]
        (root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    else:
        (root / inkling_prepare.DRAFT).unlink()
    (root / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert hybrid_prepare.prepared_identity(root) is None
