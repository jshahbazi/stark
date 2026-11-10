"""Regression coverage for external-table Qwen3.8 catalog and install routing."""

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest
from test_model_catalog import PIN, Hub

from stark_convert import model_catalog
from stark_convert.cli import convert
from stark_convert.desktop import model_valid
from stark_convert.hybrid_prepare import prepared_identity
from stark_convert.model_inventory import model_display
from stark_convert.model_support import preflight_source
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion


def external_config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "ple_layer_ids": [2],
        },
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64, "calibration": "imatrix-weighted affine"},
        "ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32},
    }


def write_tensors(path, planes):
    header, offset = {}, 0
    for name, dtype, columns in planes:
        width = columns * {"U32": 4, "BF16": 2}[dtype]
        header[name] = {"dtype": dtype, "shape": [1, columns], "data_offsets": [offset, offset + width]}
        offset += width
    payload = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(offset))


@pytest.mark.parametrize("valid_table", [True, False])
def test_external_qwen_catalog_uses_config_and_includes_table_bytes(tmp_path, monkeypatch, valid_table):
    monkeypatch.setattr(model_catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub = Hub()
    repo = "example/weighted-export"
    names = {"model.safetensors": 120, "config.json": 100, "tokenizer.json": 20, "tokenizer_config.json": 20}
    if valid_table:
        names["ngram_table.bin"] = 30
    hub.add(tmp_path, repo, external_config(), names=names)
    hub.discovered["qwen4_exp"] = [repo]
    report = model_catalog.family_catalog("qwen4_exp", hub=hub)
    assert report["status"] == "ready"
    if not valid_table:
        assert report["items"] == []
        return
    (item,) = report["items"]
    assert item["repo"] == repo and item["revision"] == PIN
    assert item["weight_bytes"] == 150
    assert item["artifact_modes"] == ["text"]
    assert not any(event[-1] == "ngram_table.bin" for event in hub.events)


@pytest.mark.parametrize("remote", [False, True])
def test_generic_external_qwen_import_preserves_indexed_source_and_binds_table(tmp_path, remote):
    root = tmp_path / "checkpoint"
    root.mkdir()
    config_path = root / "config.json"
    config_path.write_text(json.dumps(external_config()))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}"}))
    (root / "LICENSE").write_text("Synthetic license fixture")
    name = "language_model.model.embed_tokens.weight"
    shard = root / "model-00001-of-00001.safetensors"
    write_tensors(shard, [(name, "U32", 8)])
    table = root / "ngram_table.bin"
    write_tensors(table, [("weight", "U32", 20), ("scales", "BF16", 5), ("biases", "BF16", 5)])
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {name: shard.name}}))

    assert preflight_source(root)["status"] == "conditional"
    assert preflight_source(root, mode="decision")["status"] == "unsupported"
    assert preflight_source(root, mode="vision-language")["status"] == "unsupported"
    if remote:

        class SourceHub(Hub):
            def list_repo_files(self, *, repo_id, revision):
                assert repo_id == repo and revision == PIN
                return [path.name for path in root.iterdir()]

            def hf_hub_download(self, *, repo_id, filename, revision, **kwargs):
                assert repo_id == repo and revision == PIN
                return str(root / filename)

        repo = "example/external-export"
        hub = SourceHub()
        hub.repos[repo] = (root, SimpleNamespace(sha=PIN))
        assert preflight_source(repo, hub=hub)["source"]["revision"] == PIN
        snapshot = resolve_source(repo, revision=PIN, metadata_profile="hf-generic", hub=hub)
    else:
        repo = None
        snapshot = resolve_source(root, metadata_profile="hf-generic", offline=True)
    manifest = tmp_path / "source.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(root, source_manifest=manifest, offline=True)
    output = tmp_path / "prepared"
    receipt = convert(replay, output, repo=repo)
    assert receipt["status"] == "prepared"
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert receipt["weight_bytes"] == table.stat().st_size + shard.stat().st_size
    assert "lily" not in json.loads((output / "config.json").read_text())
    for file in (shard, table, index, root / "LICENSE"):
        assert (output / file.name).read_bytes() == file.read_bytes()
    assert receipt["weight_files"][table.name]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    identity = prepared_identity(output)
    assert (identity["repo"], identity["revision"]) == (repo, PIN if remote else None)
    assert identity["architecture"] == "qwen4_exp" and identity["weight_bytes"] == receipt["weight_bytes"]
    assert model_valid(output)
    display = model_display({"id": "test", "name": "External Qwen", "path": str(output)})
    assert display["source_revision"] == identity["revision"]
    assert display["weight_bytes"] == identity["weight_bytes"]
    (output / index.name).write_bytes(index.read_bytes().replace(b"embed_tokens", b"other_tokens"))
    assert prepared_identity(output) is None

    write_tensors(table, [("weight", "U32", 20), ("scales", "BF16", 5)])
    invalid = tmp_path / "invalid"
    with pytest.raises(ConversionError, match="n-gram table planes"):
        convert(root, invalid, repo=None, metadata_profile="hf-generic")
    assert not invalid.exists()

    config = external_config()
    config["quantization"]["bits"] = 3
    config_path.write_text(json.dumps(config))
    assert preflight_source(root)["status"] == "unsupported"
    with pytest.raises(UnsupportedConversion):
        resolve_source(root, metadata_profile="hf-generic", offline=True)
