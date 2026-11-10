"""Retained source image manifests bind contents; these fixtures cannot execute."""

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from test_hybrid_prepare import metadata

from stark_convert import hybrid_prepare as hybrid
from stark_convert.hybrid_vision import (
    HASH_CACHE,
    MANIFEST,
    PROCESSOR_BYTES,
    prepare_image_artifact,
    prepared_image_identity,
)


def fixture(tmp_path, monkeypatch):
    source = {**hybrid.SOURCES["qwen38-lily-q4-xl"], "weight_bytes": 4, "weight_shards": 1, "required_files": []}
    monkeypatch.setitem(hybrid.SOURCES, "qwen38-lily-q4-xl", source)
    metadata(tmp_path, "qwen4_exp")
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["vision_config"] = {"depth": 1}
    path.write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"model.visual.synthetic_{i}": "model.safetensors" for i in range(21)}})
    )
    hybrid.finish_metadata(tmp_path, "qwen38-lily-q4-xl")


def test_explicit_image_upgrade_preserves_text_receipt_and_binds_every_component(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    old_receipt = (tmp_path / "native-source.json").read_bytes()
    result = prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    assert result["schema_version"] == 6
    assert (tmp_path / "native-source.json").read_bytes() == old_receipt
    assert (tmp_path / "preprocessor_config.json").read_bytes() == PROCESSOR_BYTES
    digest = hashlib.sha256()
    digest.update(b"model.safetensors\0" + hashlib.sha256(b"test").hexdigest().encode())
    assert result["artifact_sha256"] == digest.hexdigest()
    assert json.loads((tmp_path / MANIFEST).read_text()) == result
    assert prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl") == result
    (tmp_path / "chat_template.jinja").write_text("changed")
    with pytest.raises(ValueError, match="template metadata"):
        prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")


def test_image_upgrade_requires_complete_declared_tower_and_regular_metadata(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    index = tmp_path / "model.safetensors.index.json"
    value = json.loads(index.read_text())
    value["weight_map"].popitem()
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete vision tower"):
        prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    with pytest.raises(ValueError, match="supported Qwen"):
        prepare_image_artifact(tmp_path, "qwen38-mixed")


def test_image_upgrade_seeds_private_descriptor_bound_digest_cache(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    manifest = prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    cache_path = tmp_path / HASH_CACHE
    cache = json.loads(cache_path.read_text())
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
    assert cache["schema_version"] == 1
    assert cache["manifest_sha256"] == hashlib.sha256((tmp_path / MANIFEST).read_bytes()).hexdigest()
    weight = (tmp_path / "model.safetensors").stat()
    assert cache["files"] == {
        "model.safetensors": {
            "sha256": manifest["weight_files"]["model.safetensors"]["sha256"],
            "signature": [
                weight.st_dev,
                weight.st_ino,
                weight.st_size,
                weight.st_mtime_ns // 1_000_000_000,
                weight.st_mtime_ns % 1_000_000_000,
                weight.st_ctime_ns // 1_000_000_000,
                weight.st_ctime_ns % 1_000_000_000,
            ],
        }
    }


def test_optional_digest_cache_write_failure_does_not_block_preparation(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    original = os.open

    def fail_cache(path, *args, **kwargs):
        if Path(path).name.startswith(".stark-image-validation-"):
            raise OSError("synthetic cache failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_cache)
    manifest = prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    assert prepared_image_identity(tmp_path) == manifest
    assert not (tmp_path / HASH_CACHE).exists()


def test_retained_image_host_inspection_binds_metadata_and_inventory(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    assert prepared_image_identity(tmp_path) is None
    manifest = prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    assert prepared_image_identity(tmp_path) == manifest
    # Payload admission belongs to the native loader. Host refresh still checks
    # the exact inventory and size, without streaming checkpoint payloads.
    (tmp_path / "model.safetensors").write_bytes(b"shorter")
    assert prepared_image_identity(tmp_path) is None


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", True), ("family_version", True), ("artifact_sha256", "0" * 64), ("model_revision", "0" * 40)],
)
def test_retained_image_host_inspection_rejects_descriptor_drift(tmp_path, monkeypatch, field, value):
    fixture(tmp_path, monkeypatch)
    manifest = prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    manifest[field] = value
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    assert prepared_image_identity(tmp_path) is None


def test_retained_image_host_inspection_rejects_component_changes_and_duplicate_keys(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    manifest = prepare_image_artifact(tmp_path, "qwen38-lily-q4-xl")
    path = tmp_path / MANIFEST
    path.write_text(json.dumps(manifest)[:-1] + ', "family_version":1}')
    assert prepared_image_identity(tmp_path) is None
    path.write_text(json.dumps(manifest))
    (tmp_path / "preprocessor_config.json").write_bytes(b"{}")
    assert prepared_image_identity(tmp_path) is None
