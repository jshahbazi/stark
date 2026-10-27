"""Saved-model variants derive from bounded metadata, without reading weights."""

import contextlib
import hashlib
import io
import json
import os
import struct

import pytest

from stark_convert import model_inventory as inventory
from stark_convert.desktop import Desktop


def artifact(tmp_path, *, repo="owner/Model", revision="a" * 40, specs=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = b'{"type":"LanguageModelConfig","decoder_config":{}}'
    (tmp_path / "config.json").write_bytes(config)
    specs = specs or [("FullPrecisionSpec", "BF16", None)]
    header, metadata, offset = {}, {}, 0
    for index, (kind, dtype, bits) in enumerate(specs):
        prefix = f"matrix{index}"
        width = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "U32": 4}[dtype]
        header[prefix + ".weights"] = {"dtype": dtype, "shape": [2, 4], "data_offsets": [offset, offset + 8 * width]}
        metadata[prefix + ".spec"] = json.dumps(
            {"type": kind, "layout": "output_input", **({"bits": bits, "group_size": 64} if bits is not None else {})}
        )
        offset += 8 * width
    header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)
    receipt = {
        "architecture": "qwen3",
        "sources": [{"repo": repo, "revision": revision}],
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "output_bytes": weights.stat().st_size,
        "conversion_profile": {"architecture": "qwen3", "metadata_profile": "hf-generic"},
        "source_files": [{"repo": repo, "revision": revision, "size": 4, "sha256": "0" * 64}],
    }
    bind_receipt(receipt)
    (tmp_path / "conversion.json").write_text(json.dumps(receipt))
    return {
        "id": "saved-model",
        "name": "My exact local name",
        "path": str(tmp_path),
        "source": "local:" + str(tmp_path),
    }


def change_receipt(model, update):
    from pathlib import Path

    path = Path(model["path"]) / "conversion.json"
    receipt = json.loads(path.read_bytes())
    update(receipt)
    bind_receipt(receipt)
    path.write_text(json.dumps(receipt))


def bind_receipt(receipt):
    value = {
        "sources": sorted(receipt["sources"], key=lambda item: item["repo"] or ""),
        "files": receipt["source_files"],
    }
    for key in ("conversion_profile", "artifact_mode"):
        if key in receipt:
            value[key] = receipt[key]
    receipt["source_manifest_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize(
    "specs,label",
    [
        ([("FullPrecisionSpec", "BF16", None)], "BF16"),
        ([("FullPrecisionSpec", "F16", None)], "FP16"),
        ([("FullPrecisionSpec", "F32", None)], "FP32"),
        ([("MLXSpec", "U8", 4), ("FullPrecisionSpec", "BF16", None)], "4-bit"),
        ([("IntSpec", "U8", 8)], "8-bit"),
        ([("MLXSpec", "U8", 4), ("MLXSpec", "U8", 8)], "4-bit / 8-bit"),
        ([("UnknownSpec", "BF16", None)], None),
        ([("MLXSpec", "U8", True)], None),
    ],
)
def test_storage_comes_from_actual_matrix_specs_not_names(tmp_path, specs, label):
    model = artifact(tmp_path, specs=specs)
    model["name"] = "Misleading-name-Q4-8bit"
    result = inventory.model_display(model)
    assert result["quantization"] == label
    assert result["group_name"] == "Model" and result["group_id"] == "repo:owner/Model"
    assert result["family_id"] == "qwen3" and result["source_revision"] == "a" * 40
    assert result["weight_bytes"] == (tmp_path / "model.safetensors").stat().st_size
    assert "path" not in result and "qualification" not in result


def test_revisions_and_quantizations_group_same_repo_but_fine_tunes_remain_distinct(tmp_path):
    first = inventory.model_display(artifact(tmp_path / "first"))
    second = inventory.model_display(artifact(tmp_path / "second", revision="b" * 40, specs=[("MLXSpec", "U8", 4)]))
    fine = inventory.model_display(artifact(tmp_path / "fine", repo="other/FineTune"))
    assert first["group_id"] == second["group_id"] != fine["group_id"]
    assert first["source_revision"] != second["source_revision"]
    assert first["quantization"] != second["quantization"]


@pytest.mark.parametrize("mode", [[], {}, None, True, "unknown"])
def test_malformed_optional_artifact_mode_falls_back_without_blocking(tmp_path, mode):
    model = artifact(tmp_path / "model")
    model["artifact_mode"] = mode
    assert inventory.model_display(model)["artifact_mode"] == "text"
    output = io.StringIO()
    host = Desktop(tmp_path / "unused-worker", tmp_path / "data", output)
    host.models = [model]
    host.library()
    event = json.loads(output.getvalue())
    assert event["models"][0]["artifact_mode"] == "text"


def test_native_config_determines_image_variant_despite_bad_registry_mode(tmp_path):
    model = artifact(tmp_path / "model")
    model["artifact_mode"] = []
    (tmp_path / "model/config.json").write_text('{"type":"Qwen35VisionLanguageModelConfig"}')
    assert inventory.model_display(model)["artifact_mode"] == "vision-language"


def test_native_text_config_overrides_stale_image_registry_mode(tmp_path):
    model = artifact(tmp_path / "model")
    model["artifact_mode"] = "vision-language"
    assert inventory.model_display(model)["artifact_mode"] == "text"


def test_floating_convolutions_without_matrix_specs_do_not_hide_bf16(tmp_path):
    artifact(tmp_path)
    path = tmp_path / "model.safetensors"
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    payload = raw[8 + length :]
    header["mixer.conv.weights"] = {"dtype": "F32", "shape": [1], "data_offsets": [len(payload), len(payload) + 4]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + b"\0" * 4)
    assert inventory._weights(tmp_path)[1] == "BF16"


@pytest.mark.parametrize(
    "reason", ["missing", "bad-config-hash", "bad-weight-size", "ambiguous-source", "source-mismatch"]
)
def test_unbound_source_metadata_leaves_a_standalone_saved_model(tmp_path, reason):
    model = artifact(tmp_path)
    if reason == "missing":
        (tmp_path / "conversion.json").unlink()
    elif reason == "bad-config-hash":
        change_receipt(model, lambda receipt: receipt.update(config_sha256="0" * 64))
    elif reason == "bad-weight-size":
        change_receipt(model, lambda receipt: receipt.update(output_bytes=1))
    elif reason == "ambiguous-source":
        change_receipt(
            model, lambda receipt: receipt["sources"].append({"repo": "other/Metadata", "revision": "b" * 40})
        )
    else:
        model["source"] = "huggingface:other/Model"
    result = inventory.model_display(model)
    assert result["group_id"] == "installed:saved-model" and result["group_name"] == model["name"]
    assert result["source_revision"] is None
    assert result["quantization"] == "BF16"  # This comes from the actual local header.


def test_primary_source_is_selected_by_explicit_saved_repo_among_external_metadata(tmp_path):
    model = artifact(tmp_path)
    model["source"] = "huggingface:owner/Model"
    change_receipt(model, lambda receipt: receipt["sources"].append({"repo": "other/Metadata", "revision": "b" * 40}))
    assert inventory.model_display(model)["group_id"] == "repo:owner/Model"


def test_local_primary_with_external_metadata_is_not_grouped_under_the_metadata_repo(tmp_path):
    model = artifact(tmp_path)
    change_receipt(model, lambda receipt: receipt["sources"].append({"repo": None, "revision": None}))
    assert inventory.model_display(model)["group_id"] == "installed:saved-model"


@pytest.mark.parametrize("reason", ["edited-source", "conflicting-duplicate", "malformed-source", "bad-manifest"])
def test_source_grouping_rejects_inconsistent_receipt_identity(tmp_path, reason):
    model = artifact(tmp_path)
    receipt = json.loads((tmp_path / "conversion.json").read_bytes())
    if reason == "edited-source":
        receipt["sources"][0]["repo"] = "other/Replaced"
    elif reason == "conflicting-duplicate":
        receipt["sources"].append({"repo": "owner/Model", "revision": "b" * 40})
        bind_receipt(receipt)
    elif reason == "malformed-source":
        receipt["sources"].append({"repo": "invalid", "revision": None})
        bind_receipt(receipt)
    else:
        receipt["source_manifest_sha256"] = "0" * 64
    (tmp_path / "conversion.json").write_text(json.dumps(receipt))
    assert inventory.model_display(model)["group_id"] == "installed:saved-model"


@pytest.mark.parametrize("profile", [True, False])
def test_family_label_must_match_the_bound_profile_or_catalog(tmp_path, monkeypatch, profile):
    model = artifact(tmp_path)
    if profile:
        change_receipt(model, lambda receipt: receipt.update(architecture="qwen2"))
    else:
        from stark_convert import registry

        monkeypatch.setattr(registry, "get_model", lambda _repo: {"adapter": "qwen2"})
        change_receipt(model, lambda receipt: receipt.pop("conversion_profile"))
    result = inventory.model_display(model)
    assert result["family_id"] is None and result["group_name"] == "Model"


def test_gigabyte_sparse_weights_read_only_header(tmp_path, monkeypatch):
    model = artifact(tmp_path)
    header = {
        "__metadata__": {"m.spec": json.dumps({"type": "FullPrecisionSpec"})},
        "m.weights": {"dtype": "BF16", "shape": [1024, 524288], "data_offsets": [0, 1024**3]},
    }
    encoded = json.dumps(header).encode()
    with (tmp_path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + 1024**3)
    original = inventory.open_regular
    reads = []

    @contextlib.contextmanager
    def guard(path):
        with original(path) as stream:

            class HeaderOnly:
                def fileno(self):
                    return stream.fileno()

                def read(self, count):
                    assert 0 < count <= inventory.MAX_HEADER_BYTES
                    reads.append(count)
                    return stream.read(count)

            yield HeaderOnly()

    monkeypatch.setattr(inventory, "open_regular", guard)
    size, label = inventory._weights(tmp_path)
    assert label == "BF16" and size > 1024**3
    assert reads == [8, len(encoded)]
    assert model["name"] == "My exact local name"


@pytest.mark.parametrize("kind", ["large-header", "large-receipt", "fifo", "escaping-link", "malformed-json"])
def test_unreadable_or_unbounded_metadata_falls_back_without_blocking(tmp_path, kind):
    model = artifact(tmp_path / "model")
    root = tmp_path / "model"
    if kind == "large-header":
        (root / "model.safetensors").write_bytes(struct.pack("<Q", inventory.MAX_HEADER_BYTES + 1) + b"x" * 4)
    elif kind == "large-receipt":
        with (root / "conversion.json").open("wb") as stream:
            stream.truncate(inventory.MAX_METADATA_BYTES + 1)
    elif kind == "fifo":
        (root / "conversion.json").unlink()
        os.mkfifo(root / "conversion.json")
    elif kind == "escaping-link":
        (tmp_path / "outside.json").write_text("{}")
        (root / "conversion.json").unlink()
        (root / "conversion.json").symlink_to(tmp_path / "outside.json")
    else:
        (root / "conversion.json").write_text('{"architecture":NaN}')
    result = inventory.model_display(model)
    assert result["group_id"] == "installed:saved-model"
    if kind == "large-header":
        assert result["quantization"] is None


def test_library_publishes_optional_display_without_loading_or_mutating_models(tmp_path):
    model = artifact(tmp_path / "model")
    output = io.StringIO()
    host = Desktop(tmp_path / "unused-worker", tmp_path / "data", output)
    host.models = [model]
    before = json.dumps(model, sort_keys=True)
    host.library()
    event = json.loads(output.getvalue())
    assert event["event"] == "models" and event["models"][0]["display"]["quantization"] == "BF16"
    assert host.worker is None and json.dumps(model, sort_keys=True) == before
