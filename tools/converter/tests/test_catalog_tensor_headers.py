"""Real converter layouts are admitted from pinned header ranges alone."""

import json

import pytest
from test_mlx_qwen35 import mlx_qwen35_fixture
from test_qwen35_vision import vision_fixture
from test_qwen38_external import RangeHub
from test_streaming import qwen_fixture, save_tensors

from stark_convert import hub_headers, vision_family
from stark_convert import model_catalog as catalog
from stark_convert.checkpoint_metadata import CheckpointMetadata
from stark_convert.hub_headers import checkpoint_headers
from stark_convert.planning import plan_from_headers
from stark_convert.streaming import ConversionError, UnsupportedConversion


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _: [])
    monkeypatch.setattr(catalog, "get_model", lambda _: None)


def add_checkpoint(tmp_path, source, *, family="qwen3"):
    hub = RangeHub()
    repo = "example/Model"
    config = json.loads((source / "config.json").read_bytes())
    files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    hub.add(
        tmp_path / "hub",
        repo,
        config,
        names={name: len(data) for name, data in files.items()},
        sidecars={name: data for name, data in files.items() if name != "config.json"},
    )
    hub.discovered[family] = [repo]
    return hub, repo, config


@pytest.mark.parametrize("sharded", [False, True])
def test_downloadable_qwen_headers_use_only_exact_bounded_ranges(tmp_path, sharded):
    source = tmp_path / "source"
    qwen_fixture(source, tied=True, sharded=sharded)
    hub, repo, _ = add_checkpoint(tmp_path, source)

    result = catalog.family_catalog("qwen3", hub=hub)

    assert result["status"] == "ready"
    assert len(result["items"]) == 1 and result["items"][0]["downloadable"] is True
    ranges = [event for event in hub.events if event[0] == "range"]
    assert len(ranges) == (4 if sharded else 2)
    assert all(event[1] == repo for event in ranges)
    assert not [event for event in hub.events if event[0] == "download" and event[-1].endswith(".safetensors")]


@pytest.mark.parametrize("case", ["missing_tensor", "wrong_shape", "wrong_index", "ignored_ranges", "network_error"])
def test_config_compatible_bad_layout_is_visible_and_cannot_be_downloaded(tmp_path, monkeypatch, case):
    source = tmp_path / "source"
    _, tensors = qwen_fixture(source, sharded=case == "wrong_index")
    if case == "missing_tensor":
        del tensors["model.norm.weight"]
        save_tensors(source / "model.safetensors", tensors)
    elif case == "wrong_shape":
        dtype, values = tensors["model.layers.0.self_attn.q_proj.weight"]
        tensors["model.layers.0.self_attn.q_proj.weight"] = dtype, values.reshape(4, 16)
        save_tensors(source / "model.safetensors", tensors)
    elif case == "wrong_index":
        path = source / "model.safetensors.index.json"
        index = json.loads(path.read_bytes())
        index["weight_map"]["model.embed_tokens.weight"] = "model-00002-of-00002.safetensors"
        path.write_text(json.dumps(index))
    hub, _, _ = add_checkpoint(tmp_path, source)
    if case == "ignored_ranges":
        hub.ignores_ranges = True
    elif case == "network_error":

        def fail():
            raise RuntimeError("secret credential or private exception sentinel")

        monkeypatch.setattr(hub.utils, "get_session", fail)

    result = catalog.family_catalog("qwen3", hub=hub)

    assert result["status"] == "ready" and len(result["items"]) == 1
    item = result["items"][0]
    assert item["downloadable"] is False and item["unavailable_reason"]
    assert "sentinel" not in json.dumps(result) and "credential" not in json.dumps(result)
    assert not [event for event in hub.events if event[0] == "download" and event[-1].endswith(".safetensors")]


@pytest.mark.parametrize("bad_packing", [False, True])
def test_actual_affine_packing_is_checked_by_the_same_qwen35_planner(tmp_path, monkeypatch, bad_packing):
    source = tmp_path / "source"
    mlx_qwen35_fixture(source)
    if bad_packing:
        config_path = source / "config.json"
        config = json.loads(config_path.read_bytes())
        # The bytes remain W4/group32 while both declarations claim W8/group32.
        for field in ("quantization", "quantization_config"):
            if field in config:
                config[field]["bits"] = 8
        config_path.write_text(json.dumps(config))
    hub, repo, config = add_checkpoint(tmp_path, source, family="qwen3_5")
    monkeypatch.setattr(vision_family, "_DELTA_HEAD_DIM", 32)

    result = catalog.family_catalog("qwen3_5", hub=hub)

    assert result["status"] == "ready" and len(result["items"]) == 1
    assert result["items"][0]["downloadable"] is (not bad_packing)
    inventory = {path.name: path.stat().st_size for path in source.iterdir()}
    tensors = checkpoint_headers(
        hub, repo, result["items"][0]["revision"], ["model.safetensors"], None, inventory, lambda: None
    )
    if bad_packing:
        with pytest.raises((ConversionError, UnsupportedConversion)):
            plan_from_headers("qwen3_5", CheckpointMetadata(config, tensors))
    else:
        assert plan_from_headers("qwen3_5", CheckpointMetadata(config, tensors)).operations


def test_oversized_header_has_a_permanent_budget_reason_without_reading_its_body(tmp_path, monkeypatch):
    source = tmp_path / "source"
    qwen_fixture(source)
    hub, _, _ = add_checkpoint(tmp_path, source)
    original = hub_headers.read_header

    def bounded(*args, **kwargs):
        return original(*args, **kwargs, maximum=8)

    monkeypatch.setattr(hub_headers, "read_header", bounded)

    result = catalog.family_catalog("qwen3", hub=hub)

    item = result["items"][0]
    assert item["downloadable"] is False
    assert "verification limit" in item["unavailable_reason"]
    assert len([event for event in hub.events if event[0] == "range"]) == 1


@pytest.mark.parametrize("vision_layout", ["valid", "missing", "wrong_shape"])
def test_image_header_failure_cannot_offer_a_text_only_download(tmp_path, monkeypatch, vision_layout):
    source = tmp_path / "source"
    _, tensors, _ = vision_fixture(source)
    name = "model.visual.patch_embed.proj.bias"
    if vision_layout == "missing":
        del tensors[name]
    elif vision_layout == "wrong_shape":
        dtype, values = tensors[name]
        tensors[name] = dtype, values.reshape(2, 2)
    save_tensors(source / "model.safetensors", tensors)
    hub, _, _ = add_checkpoint(tmp_path, source, family="qwen3_5")
    monkeypatch.setattr(vision_family, "_DELTA_HEAD_DIM", 2)
    # Processing admission is exercised separately. This test composes the real
    # text and vision tensor planners after the source has passed that gate.
    monkeypatch.setattr(catalog, "_catalog_image_metadata", lambda *args: True)

    result = catalog.family_catalog("qwen3_5", hub=hub)

    assert result["status"] == "ready" and len(result["items"]) == 1
    item = result["items"][0]
    assert item["artifact_modes"] == ["text", "vision-language"]
    assert item["default_artifact_mode"] == "vision-language"
    assert item["downloadable"] is (vision_layout == "valid")
    if vision_layout != "valid":
        assert "tensor" in item["unavailable_reason"]
