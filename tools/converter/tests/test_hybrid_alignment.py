"""Bounded identity recognition for a lossless safetensors container derivative."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from stark_convert import hybrid_prepare as hybrid


@pytest.fixture
def aligned_model(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import align_native_safetensors as alignment

    model = tmp_path / "original"
    model.mkdir()
    mapping = {}
    for index, modulo in enumerate([7, 11]):
        name = f"weight-{index}"
        header = json.dumps({name: {"dtype": "U32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        while (8 + len(header)) % 16 != modulo:
            header += b" "
        filename = f"model-{index}.safetensors"
        (model / filename).write_bytes(struct.pack("<Q", len(header)) + header + b"abcdefgh")
        mapping[name] = filename
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"vocab_size": 8, "eos_token_id": 1},
            }
        )
    )
    (model / "tokenizer.json").write_text("{}")
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0]['content'] }}"}))
    source = {
        **hybrid.SOURCES["qwen38-mixed"],
        "weight_bytes": sum(path.stat().st_size for path in model.glob("*.safetensors")),
    }
    monkeypatch.setitem(hybrid.SOURCES, "qwen38-mixed", source)
    hybrid.finish_metadata(model, "qwen38-mixed")
    output = tmp_path / "aligned"
    alignment.align_model(model, output, lambda _: None)
    return model, output


def rebind_alignment(output, mutate):
    alignment_path = output / "alignment-receipt.json"
    alignment = json.loads(alignment_path.read_bytes())
    mutate(alignment)
    alignment_path.write_text(json.dumps(alignment))
    receipt_path = output / "native-source.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["storage_alignment"]["receipt_sha256"] = hashlib.sha256(alignment_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))


def test_alignment_identity_accepts_bound_padding_and_preserves_validation_status(aligned_model):
    original, output = aligned_model
    original_identity = hybrid.prepared_identity(original)
    derived_identity = hybrid.prepared_identity(output)
    assert original_identity is not None and derived_identity is not None
    assert derived_identity["repo"] == original_identity["repo"]
    assert derived_identity["revision"] == original_identity["revision"]
    assert derived_identity["weight_bytes"] == original_identity["weight_bytes"] + 14
    original_receipt = (original / "native-source.json").read_bytes()
    assert (output / "native-source.original.json").read_bytes() == original_receipt
    derived_receipt = json.loads((output / "native-source.json").read_bytes())
    assert derived_receipt["native_loading"] == derived_receipt["execution"] == "unverified"
    assert (original / "config.json").read_bytes() == (output / "config.json").read_bytes()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt["shards"][0].update(padding_bytes=16),
        lambda receipt: receipt["shards"][0].update(original_header_bytes=1),
        lambda receipt: receipt["shards"][0].update(payload_bytes=-1),
        lambda receipt: receipt["shards"][0].update(tensor_count=99),
        lambda receipt: receipt["shards"][0].update(file="../outside.safetensors"),
        lambda receipt: receipt["shards"][0].update(original_file_sha256="invalid"),
        lambda receipt: receipt["shards"][1].update(file=receipt["shards"][0]["file"]),
        lambda receipt: receipt.update(
            original_native_source={"file": "native-source.original.json", "sha256": "0" * 64}
        ),
    ],
)
def test_alignment_identity_rejects_rebound_but_invalid_manifest(aligned_model, mutate):
    _, output = aligned_model
    rebind_alignment(output, mutate)
    assert hybrid.prepared_identity(output) is None


def test_alignment_identity_rejects_current_size_header_and_original_receipt_changes(aligned_model):
    _, output = aligned_model
    path = next(output.glob("*.safetensors"))
    contents = path.read_bytes()
    path.write_bytes(contents + b"x")
    assert hybrid.prepared_identity(output) is None
    path.write_bytes(contents)
    path.write_bytes(struct.pack("<Q", 1) + contents[8:])
    assert hybrid.prepared_identity(output) is None
    path.write_bytes(contents)
    source_path = output / "native-source.original.json"
    source_path.write_bytes(source_path.read_bytes() + b" ")
    assert hybrid.prepared_identity(output) is None


def test_alignment_identity_rejects_status_upgrade_and_unbound_or_excessive_receipt(aligned_model):
    _, output = aligned_model
    path = output / "native-source.json"
    contents = path.read_bytes()
    receipt = json.loads(contents)
    receipt["execution"] = "verified"
    path.write_text(json.dumps(receipt))
    assert hybrid.prepared_identity(output) is None
    path.write_bytes(contents)
    alignment_path = output / "alignment-receipt.json"
    alignment_path.write_bytes(alignment_path.read_bytes() + b" ")
    assert hybrid.prepared_identity(output) is None
    alignment_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    assert hybrid.prepared_identity(output) is None


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "native-source.json",
        "alignment-receipt.json",
        "native-source.original.json",
    ],
)
def test_alignment_identity_treats_excessive_json_nesting_as_invalid(aligned_model, filename):
    _, output = aligned_model
    path = output / filename
    path.write_text("[" * 2000 + "0" + "]" * 2000)
    assert hybrid.prepared_identity(output) is None


def test_alignment_identity_converts_decoder_recursion_to_invalid_artifact(aligned_model, monkeypatch):
    _, output = aligned_model

    def excessive_nesting(*_args, **_kwargs):
        raise RecursionError("decoder nesting")

    monkeypatch.setattr(hybrid.json, "loads", excessive_nesting)
    assert hybrid.prepared_identity(output) is None
    with pytest.raises(ValueError, match="nesting"):
        hybrid._json(output / "config.json")


def test_alignment_identity_accepts_explicit_reset_from_a_qualified_original(aligned_model):
    import align_native_safetensors as alignment

    original, output = aligned_model
    receipt_path = original / "native-source.json"
    source_receipt = json.loads(receipt_path.read_bytes())
    source_receipt.update(native_loading="verified", execution="verified")
    receipt_path.write_text(json.dumps(source_receipt))
    preserved = receipt_path.read_bytes()
    reset_output = output.with_name("reset-aligned")
    alignment.align_model(original, reset_output, lambda _: None)
    assert hybrid.prepared_identity(reset_output) is not None
    assert (reset_output / "native-source.original.json").read_bytes() == preserved
    derived = json.loads((reset_output / "native-source.json").read_bytes())
    assert derived["native_loading"] == derived["execution"] == "unverified"
