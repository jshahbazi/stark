import io
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from stark_convert import desktop as desktop_module
from stark_convert import hybrid_prepare, inkling_prepare, model_catalog, native_catalog
from stark_convert.desktop import Desktop


def test_legacy_glm_catalog_api_remains_compatible():
    item = native_catalog.catalog_item()
    assert native_catalog.catalog_item(native_catalog.CHOICE) == item
    assert native_catalog.catalog_items(native_catalog.FAMILY_ID) == [item]
    assert native_catalog.source_choice(native_catalog.FAMILY_ID, item) == native_catalog.CHOICE
    assert native_catalog.allowed_choice(native_catalog.CHOICE)


def test_inkling_catalog_binds_original_target_and_draft_to_default_vision_mode(tmp_path, monkeypatch):
    family = native_catalog.INKLING_FAMILY_ID
    choice = native_catalog.INKLING_CHOICE
    [item] = native_catalog.catalog_items(family)
    assert item["artifact_modes"] == ["vision-language"]
    assert item["default_artifact_mode"] == "vision-language"
    assert item["repo"] == "thinkingmachines/Inkling-2-Mini"
    assert item["revision"] == inkling_prepare.REVISION
    assert item["download_revision"] == inkling_prepare.DOWNLOAD_REVISION
    assert hybrid_prepare.SOURCES[choice]["required_weight_files"] == [
        "model-00001-of-00001.safetensors",
        "mtp.safetensors",
    ]
    assert native_catalog.source_choice(family, item) == choice
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    report = model_catalog.family_catalog(family, hub=object())
    host = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    try:
        [published] = host.catalog_report(report, family)["items"]
        assert published["download_revision"] == inkling_prepare.DOWNLOAD_REVISION
        report["items"][0]["download_revision"] = "main"
        with pytest.raises(ValueError):
            host.catalog_report(report, family)
    finally:
        host.stop(quiet=True)


def test_hy4_catalog_binds_exact_split_source_and_image_projector():
    family = native_catalog.HY4_FAMILY_ID
    choice = native_catalog.HY4_CHOICE
    [item] = native_catalog.catalog_items(family)
    source = hybrid_prepare.SOURCES[choice]
    assert item["name"] == "Hy4-flash AngelSlim Q3_K"
    assert item["repo"] == "AngelSlim/Hy4-flash-GGUF"
    assert item["revision"] == "c50d4dd30be42f2b91138e5731de0e2f32ad56b2"
    assert item["quantization"] == "Q3_K"
    assert item["weight_bytes"] == source["weight_bytes"] == 156_415_403_072
    assert item["artifact_modes"] == ["vision-language"]
    assert item["default_artifact_mode"] == "vision-language"
    assert source["required_weight_files"] == [
        "Hy4-flash-Q3_K-00001-of-00002.gguf",
        "Hy4-flash-Q3_K-00002-of-00002.gguf",
        "mmproj-Hy4-flash-bf16.gguf",
    ]
    assert source["metadata_repo"] == "tencent/Hy4-flash"
    assert source["metadata_revision"] == "eccd3123eec73f2040f4c9ed0de1883593470510"
    assert native_catalog.source_choice(family, item) == choice
    assert native_catalog.allowed_choice(choice)


def test_hy4_family_is_visible_as_a_conditional_native_image_family():
    from stark_convert.model_support import support_matrix

    definition = next(
        family for family in support_matrix()["families"] if family["family_id"] == native_catalog.HY4_FAMILY_ID
    )
    assert definition["modes"]["text"]["status"] == "conditional"
    assert definition["modes"]["vision-language"]["status"] == "conditional"


def test_hy4_family_catalog_returns_the_pinned_downloadable_choice_without_hub_search(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    report = model_catalog.family_catalog(native_catalog.HY4_FAMILY_ID, hub=object())
    assert report["status"] == "ready"
    assert report["family_id"] == native_catalog.HY4_FAMILY_ID
    assert report["items"] == native_catalog.catalog_items(native_catalog.HY4_FAMILY_ID)


def test_qwen_native_catalog_preserves_text_source_and_adds_retained_image_choices():
    items = native_catalog.catalog_items(native_catalog.QWEN_FAMILY_ID)
    assert len(items) == 3
    item = items[0]
    source = hybrid_prepare.SOURCES[native_catalog.QWEN_CHOICE]
    assert item["repo"] == source["weights_repo"]
    assert item["revision"] == source["weights_revision"]
    assert item["weight_bytes"] == source["weight_bytes"]
    assert item["artifact_modes"] == ["text"]
    assert item["downloadable"] is True
    assert native_catalog.catalog_item(native_catalog.QWEN_CHOICE) == item
    assert native_catalog.source_choice(native_catalog.QWEN_FAMILY_ID, item) == native_catalog.QWEN_CHOICE
    assert native_catalog.allowed_choice(native_catalog.QWEN_CHOICE)
    for item in items[1:]:
        assert item["artifact_modes"] == ["text", "vision-language"]
        assert item["default_artifact_mode"] == "text"
        assert native_catalog.source_choice(native_catalog.QWEN_FAMILY_ID, item) in {
            "qwen38-lily-q4-xl",
            "qwen38-mlx-serve-mixed",
        }


def test_native_catalog_rejects_other_families_and_mutated_items():
    item = native_catalog.catalog_item(native_catalog.QWEN_CHOICE)
    mutated = deepcopy(item)
    mutated["revision"] = "f" * 40
    assert native_catalog.catalog_items("unknown") == []
    assert native_catalog.source_choice("unknown", item) is None
    assert native_catalog.source_choice(native_catalog.QWEN_FAMILY_ID, mutated) is None
    assert native_catalog.source_choice(native_catalog.FAMILY_ID, item) is None
    assert native_catalog.source_for("unregistered") is None
    assert not native_catalog.allowed_choice("unregistered")


def test_unavailable_native_catalog_rows_surface_unqualified_native_layouts():
    glm = native_catalog.unavailable_item(
        native_catalog.FAMILY_ID,
        "someone/GLM-5.3-Flash-MLX",
        "a" * 40,
        {"model_type": "glm5_next"},
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        {"model-00001-of-00002.safetensors": 12, "model-00002-of-00002.safetensors": 15},
    )
    assert glm["downloadable"] is False
    assert glm["artifact_modes"] == ["text"]
    assert glm["weight_bytes"] == 27
    assert "Safetensors import is unavailable" in glm["unavailable_reason"]

    flint = native_catalog.unavailable_item(
        native_catalog.FLINT_FAMILY_ID,
        "someone/flint-affine-export",
        "b" * 40,
        {"model_type": "flint"},
        ["weights.safetensors"],
        {"weights.safetensors": 99},
    )
    assert flint["downloadable"] is False
    assert flint["quantization"] == "Safetensors/MLX"
    assert "pinned MLX affine INT4 checkpoint" in flint["unavailable_reason"]


def test_unavailable_native_catalog_rows_require_matching_family_and_weights():
    assert native_catalog.unavailable_item("other", "owner/repo", "a" * 40, {}, ["weights.gguf"], {}) is None
    assert (
        native_catalog.unavailable_item(
            native_catalog.FAMILY_ID,
            "owner/GLM-5.3-Flash",
            "a" * 40,
            {"model_type": "glm_moe_dsa"},
            ["weights.gguf"],
            {},
        )
        is None
    )
    assert (
        native_catalog.unavailable_item(
            native_catalog.FAMILY_ID,
            "owner/GLM-5.3-Flash",
            "a" * 40,
            {},
            ["config.json"],
            {},
        )
        is None
    )
    gguf = native_catalog.unavailable_item(
        native_catalog.FAMILY_ID,
        "owner/GLM-5.3-Flash-GGUF",
        "a" * 40,
        {},
        ["model.gguf"],
        {"model.gguf": 123},
    )
    assert gguf["quantization"] == "GGUF"
    assert gguf["weight_bytes"] == 123
    assert gguf["downloadable"] is False


def test_unavailable_native_catalog_weight_bytes_are_bounded_and_exclude_sidecars():
    config = {"model_type": "flint"}
    base = native_catalog.unavailable_item(
        "flint",
        "owner/flint-export",
        "a" * 40,
        config,
        ["weights.safetensors"],
        {"weights.safetensors": 2**63},
    )
    assert base["weight_bytes"] is None
    zero = native_catalog.unavailable_item(
        "flint",
        "owner/flint-export",
        "a" * 40,
        config,
        ["weights.safetensors"],
        {"weights.safetensors": 0},
    )
    assert zero["weight_bytes"] is None
    sidecars = native_catalog.unavailable_item(
        "flint",
        "owner/flint-export",
        "a" * 40,
        config,
        ["tokenizer.bin", "adapter_model.safetensors"],
        {"tokenizer.bin": 512, "adapter_model.safetensors": 1024},
    )
    assert sidecars is None


def test_qwen_native_install_prepares_and_registers_the_pinned_source(tmp_path, monkeypatch):
    from test_hybrid_prepare import metadata

    choice = native_catalog.QWEN_CHOICE
    source = {**hybrid_prepare.SOURCES[choice], "weight_bytes": 4}
    monkeypatch.setitem(hybrid_prepare.SOURCES, choice, source)
    calls = []

    def snapshot(repo, **options):
        assert repo == source["weights_repo"]
        assert options["revision"] == source["weights_revision"]
        root = Path(options["local_dir"])
        calls.append((repo, options["revision"]))
        (root / "model.safetensors").write_bytes(b"test")
        metadata(root, architecture="qwen4_exp")

    monkeypatch.setattr(hybrid_prepare, "snapshot_download", snapshot)
    host = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    host.begin_install_progress(host.epoch, "native")

    def prepare(command, _environment, _epoch):
        assert command[:6] == [sys.executable, "-I", "-B", "-m", "stark_convert.hybrid_prepare", choice]
        assert command[6] == "--output" and len(command) == 8
        hybrid_prepare.prepare(choice, Path(command[7]))
        return 0

    monkeypatch.setattr(host, "run_converter", prepare)
    try:
        host.install(host.epoch, "native", choice, "text", "text", "install", source["weights_revision"])

        assert calls == [(source["weights_repo"], source["weights_revision"])] * 2
        assert len(host.models) == 1 and host.worker is None
        model = host.models[0]
        assert model["source"] == "huggingface:" + source["weights_repo"]
        path = Path(model["path"])
        assert path.parent == host.model_dir and desktop_module.model_valid(path)
        assert desktop_module.prepared_identity(path) == {
            "repo": source["weights_repo"],
            "revision": source["weights_revision"],
            "architecture": source["architecture"],
            "weight_bytes": source["weight_bytes"],
            "quantization": source["quantization"],
        }
        assert json.loads(host.registry.read_bytes()) == host.models
        assert list(host.model_dir.iterdir()) == [path]
        installed = next(
            event for event in map(json.loads, host.output.getvalue().splitlines()) if event["event"] == "installed"
        )
        assert installed["support_request_id"] == "install"
        assert host.install_progress["download"]["state"] == "complete"
    finally:
        host.stop(quiet=True)


def test_failed_native_image_check_restores_owned_download_for_retry(tmp_path, monkeypatch):
    from test_hybrid_prepare import metadata

    choice = native_catalog.QWEN_CHOICE
    source = {**hybrid_prepare.SOURCES[choice], "weight_bytes": 4}
    monkeypatch.setitem(hybrid_prepare.SOURCES, choice, source)

    def snapshot(repo, **options):
        assert repo == source["weights_repo"]
        root = Path(options["local_dir"])
        (root / "model.safetensors").write_bytes(b"test")
        metadata(root, architecture="qwen4_exp")

    monkeypatch.setattr(hybrid_prepare, "snapshot_download", snapshot)
    host = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    host.begin_install_progress(host.epoch, "native")

    def prepare(command, _environment, _epoch):
        hybrid_prepare.prepare(choice, Path(command[7]))
        return 0

    monkeypatch.setattr(host, "run_converter", prepare)
    checks = 0

    def fail_once(_epoch, _path):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise desktop_module.DesktopError("image validation needs a retry")

    monkeypatch.setattr(host, "check_family_install", fail_once)
    try:
        try:
            host.install(host.epoch, "native", choice, "text", "text", "first", source["weights_revision"])
        except desktop_module.DesktopError:
            pass
        else:
            raise AssertionError("expected the first image validation attempt to fail")

        [owned] = (host.data_dir / "native-downloads").glob(choice + "-*")
        staged_model = owned / "model"
        assert desktop_module.model_valid(staged_model)
        assert host.models == []

        host.install(host.epoch, "native", choice, "text", "text", "retry", source["weights_revision"])
        assert checks == 2
        assert len(host.models) == 1
        assert desktop_module.model_valid(Path(host.models[0]["path"]))
    finally:
        host.stop(quiet=True)
