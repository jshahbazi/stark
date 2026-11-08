import io
import json
import sys
from copy import deepcopy
from pathlib import Path

from stark_convert import desktop as desktop_module
from stark_convert import hybrid_prepare, native_catalog
from stark_convert.desktop import Desktop


def test_legacy_glm_catalog_api_remains_compatible():
    item = native_catalog.catalog_item()
    assert native_catalog.catalog_item(native_catalog.CHOICE) == item
    assert native_catalog.catalog_items(native_catalog.FAMILY_ID) == [item]
    assert native_catalog.source_choice(native_catalog.FAMILY_ID, item) == native_catalog.CHOICE
    assert native_catalog.allowed_choice(native_catalog.CHOICE)


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
