"""Synthetic processing-selection controls, with no model execution claims."""

import hashlib
import json

import pytest
from test_qwen38 import fixture as lily_checkpoint
from test_qwen38_external import external_config as external_config
from test_qwen38_kquant import indexed_config as indexed_config

from stark_convert import qwen38_kquant, qwen38_retained
from stark_convert.hybrid_vision import PROCESSOR_BYTES, PROFILE
from stark_convert.streaming import UnsupportedConversion
from stark_convert.vision_family import _PROCESSOR


@pytest.fixture
def processing_files(monkeypatch):
    # This explicit synthetic program isolates file selection from production
    # template review. The fixture grants no execution or image qualification.
    template = "{{ messages[0].content }}"
    monkeypatch.setattr(
        qwen38_retained,
        "SUPPORTED_TEMPLATE_HASHES",
        qwen38_retained.SUPPORTED_TEMPLATE_HASHES | {hashlib.sha256(template.encode()).hexdigest()},
    )
    return {
        "tokenizer.json": b"{}",
        "tokenizer_config.json": json.dumps({"chat_template": template}).encode(),
        "preprocessor_config.json": json.dumps(_PROCESSOR).encode(),
        "model.safetensors": b"synthetic unopened weight placeholder",
    }


@pytest.mark.parametrize("dialect", ["lily", "external"])
def test_unselected_ple_manifest_does_not_reject_another_retained_dialect(
    tmp_path, processing_files, external_config, dialect
):
    if dialect == "lily":
        config, _ = lily_checkpoint(tmp_path / "lily")
    else:
        config = external_config
        processing_files["ngram_table.bin"] = b"synthetic unopened table placeholder"
    processing_files["config.json"] = json.dumps(config).encode()
    qwen38_retained.validate_processing_metadata(processing_files, lambda name, _: processing_files[name])

    # A repository can hold metadata for another export. This file is outside
    # the selected dialect, and its contents must not affect this source.
    processing_files["ple_manifest.json"] = b'{"unselected_export":true}'
    rules = qwen38_retained.retained_files("example/independent-publisher", processing_files, config)
    assert "ple_manifest.json" not in {rule["filename"] for rule in rules}

    def read_selected(name, _maximum):
        assert name != "ple_manifest.json", "unselected PLE metadata was opened"
        return processing_files[name]

    qwen38_retained.validate_processing_metadata(processing_files, read_selected)


def indexed_manifest(config):
    architecture = qwen38_kquant.architecture(config)
    rows = architecture["ple"]["padded_vocab_size"]
    parts = architecture["split_ngram_parts"]
    width = architecture["ple_embed_dim"] // ((architecture["ngram_size"] - 1) * architecture["heads_per_ngram"])
    return {
        "version": 2,
        "source_root": ".",
        "layout": "safetensors_ranges",
        "row_width": width,
        "row_count": rows,
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "shards": [
            {
                "row_start": part * (rows // parts),
                "row_count": rows // parts,
                **{
                    plane: {
                        "file": "ple.safetensors",
                        "offset": 8 + part * 4096 + offset,
                        "dtype": dtype,
                        "shape": [rows // parts, columns],
                    }
                    for plane, dtype, columns, offset in (
                        ("weight", "U32", width // 8, 0),
                        ("scales", "BF16", width // 32, 2048),
                        ("biases", "BF16", width // 32, 3072),
                    )
                },
            }
            for part in range(parts)
        ],
    }


@pytest.mark.parametrize("manifest_state", ["missing", "malformed", "valid"])
def test_indexed_processing_requires_its_own_ple_manifest(processing_files, indexed_config, manifest_state):
    processing_files["config.json"] = json.dumps(indexed_config).encode()
    if manifest_state == "malformed":
        processing_files["ple_manifest.json"] = b"{}"
    elif manifest_state == "valid":
        processing_files["ple_manifest.json"] = json.dumps(indexed_manifest(indexed_config)).encode()

    if manifest_state == "valid":
        qwen38_retained.validate_processing_metadata(processing_files, lambda name, _: processing_files[name])
    else:
        with pytest.raises(UnsupportedConversion, match="PLE|manifest"):
            qwen38_retained.validate_processing_metadata(processing_files, lambda name, _: processing_files[name])


def image_metadata(config):
    """Add tiny complete image metadata for processing-selection controls."""
    config["text_config"]["rope_parameters"].update(mrope_section=[11, 11, 10], mrope_interleaved=True)
    config.update(image_token_id=2, video_token_id=3, vision_start_token_id=4, vision_end_token_id=5)
    config["vision_config"] = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "depth": 1,
        "num_position_embeddings": 4,
        "patch_size": 16,
        "in_channels": 3,
        "out_hidden_size": config["text_config"]["hidden_size"],
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "hidden_act": "gelu_pytorch_tanh",
    }
    if "lily" in config:
        config["lily"]["vision"] = {"dtype": "bf16", "tensors": 21}
    return config


@pytest.mark.parametrize("dialect", ["lily", "external", "indexed"])
def test_complete_retained_contract_can_select_the_documented_family_processor(
    tmp_path, request, monkeypatch, processing_files, dialect
):
    # These tiny graphs isolate metadata/default policy, not native numerical
    # validation. Separate mutation tests preserve the production geometry gate.
    monkeypatch.setattr(qwen38_retained, "validate_native_geometry", lambda _architecture: None)
    if dialect == "lily":
        config, _ = lily_checkpoint(tmp_path / "lily")
    else:
        config = request.getfixturevalue("indexed_config" if dialect == "indexed" else "external_config")
    config = image_metadata(config)
    processing_files["config.json"] = json.dumps(config).encode()
    processing_files.pop("preprocessor_config.json")
    if dialect == "external":
        processing_files["ngram_table.bin"] = b"unopened synthetic table"
    if dialect == "indexed":
        processing_files["model.safetensors.index.json"] = b"unopened synthetic index"
        processing_files["ple_manifest.json"] = json.dumps(indexed_manifest(config)).encode()

    descriptor = qwen38_retained.validate_processing_metadata(processing_files, lambda name, _: processing_files[name])
    assert descriptor == {
        "origin": "stark-family-default",
        "family_profile": PROFILE,
        "family_version": 1,
        "sha256": hashlib.sha256(PROCESSOR_BYTES).hexdigest(),
        "bytes": len(PROCESSOR_BYTES),
    }
    rules = qwen38_retained.retained_files("independent-publisher/compatible-checkpoint", processing_files, config)
    assert not any(rule["role"] == "processor_config" for rule in rules)


@pytest.mark.parametrize("change", ["unknown_family", "missing_tower", "native_geometry", "missing_config"])
def test_family_default_does_not_bypass_image_or_native_geometry_admission(processing_files, external_config, change):
    config = image_metadata(external_config)
    processing_files.pop("preprocessor_config.json")
    if change == "unknown_family":
        config["model_type"] = "unimplemented_family"
    elif change == "missing_tower":
        config.pop("vision_config")
    if change != "missing_config":
        processing_files["config.json"] = json.dumps(config).encode()
    # The synthetic head_dim=32 fails the real native rotary-64 gate for the
    # native_geometry arm; no production admission check is patched here.
    with pytest.raises(UnsupportedConversion):
        qwen38_retained.validate_processing_metadata(processing_files, lambda name, _: processing_files[name])


@pytest.mark.parametrize("change", ["unknown", "disagreeing"])
def test_family_default_never_overrides_explicit_unsupported_processing(processing_files, external_config, change):
    processing_files["config.json"] = json.dumps(external_config).encode()
    if change == "unknown":
        processing_files["preprocessor_config.json"] = json.dumps({**_PROCESSOR, "patch_size": 32}).encode()
    else:
        processing_files["processor_config.json"] = json.dumps(
            {"processor_class": _PROCESSOR["processor_class"], "image_processor": {**_PROCESSOR, "patch_size": 32}}
        ).encode()
    with pytest.raises(UnsupportedConversion, match="processor recipe|processing metadata"):
        qwen38_retained.validate_processing_metadata(processing_files, lambda name, _: processing_files[name])


@pytest.mark.parametrize("source_recipe", [False, True])
def test_remote_processor_provenance_distinguishes_upstream_from_family_default(
    tmp_path, monkeypatch, external_config, processing_files, source_recipe
):
    from types import SimpleNamespace

    from test_flashnext_image_source import REPO, image_checkpoint
    from test_qwen38_external import PIN, RangeHub

    from stark_convert.cli import convert
    from stark_convert.hybrid_vision import prepared_image_identity
    from stark_convert.model_support import preflight_source
    from stark_convert.sources import resolve_source

    monkeypatch.setattr(qwen38_retained, "validate_native_geometry", lambda _architecture: None)
    root = tmp_path / "source"
    image_checkpoint(root, external_config)
    source_processor = (root / "preprocessor_config.json").read_bytes()
    if not source_recipe:
        (root / "preprocessor_config.json").unlink()
    hub = RangeHub()
    hub.repos[REPO] = root, SimpleNamespace(sha=PIN)
    assert preflight_source(REPO, revision=PIN, mode="vision-language", hub=hub)["status"] == "conditional"
    source = resolve_source(
        REPO, revision=PIN, metadata_profile="hf-generic", artifact_mode="vision-language", hub=hub
    )
    assert any(record["role"] == "processor_config" for record in source.manifest["files"]) == source_recipe
    source_manifest = tmp_path / "source.json"
    source.write_manifest(source_manifest)
    replay = resolve_source(root, source_manifest=source_manifest, artifact_mode="vision-language", offline=True)
    output = tmp_path / "prepared"
    receipt = convert(replay, output, repo=REPO, artifact_mode="vision-language")
    assert (output / "preprocessor_config.json").read_bytes() == PROCESSOR_BYTES
    image = prepared_image_identity(output)
    assert image["processor_sha256"] == hashlib.sha256(PROCESSOR_BYTES).hexdigest()
    if source_recipe:
        assert "derived_metadata" not in receipt
        assert (root / "preprocessor_config.json").read_bytes() == source_processor
        assert any(
            record["sha256"] == hashlib.sha256(source_processor).hexdigest()
            and record["size"] == len(source_processor)
            for record in receipt["source_files"]
        )
    else:
        descriptor = receipt["derived_metadata"]["preprocessor_config.json"]
        assert descriptor["origin"] == "stark-family-default"
        assert descriptor["sha256"] == image["processor_sha256"]
        assert not (root / "preprocessor_config.json").exists()
        assert all(record["sha256"] != descriptor["sha256"] for record in receipt["source_files"])
