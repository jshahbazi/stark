"""Pinned native Ripple preparation, provenance and app catalog admission."""

import json
from pathlib import Path

import pytest

from stark_convert import hybrid_prepare as hybrid
from stark_convert import model_catalog, native_catalog
from stark_convert.model_support import preflight_model, support_matrix


def ripple_metadata(root):
    (root / "config.json").write_text(
        json.dumps({"model_type": "ricursive", "vocab_size": 100352, "eos_token_id": [100265, 100257]})
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "plain fallback", "eos_token": "<|endoftext|>"})
    )
    (root / "tokenizer.json").write_text("{}")
    (root / "chat_template.jinja").write_text("tools-aware {{ messages[0]['content'] }}")
    (root / "LICENSE").write_text("Apache License 2.0")


@pytest.mark.parametrize("choice", ["ripple-q8", "ripple-bf16", "ripple-q4-k-m"])
def test_ripple_preparation_preserves_exact_selected_shards_and_official_metadata(tmp_path, monkeypatch, choice):
    original = hybrid.SOURCES[choice]
    source = {**original, "weight_bytes": original["weight_shards"]}
    monkeypatch.setitem(hybrid.SOURCES, choice, source)
    calls = []

    def download(repo, **options):
        calls.append((repo, options))
        root = Path(options["local_dir"])
        if options["allow_patterns"] == source["weights_patterns"]:
            assert repo == source["weights_repo"]
            assert options["revision"] == source["weights_revision"]
            for name in source["required_weight_files"]:
                (root / name).write_bytes(b"q")
        else:
            assert repo == "ricursiveAI/Ripple-65B-Preview"
            assert options["revision"] == "e27e171792f303d74c048184e8259ba663cfa0d5"
            ripple_metadata(root)

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    output = tmp_path / choice
    receipt = hybrid.prepare(choice, output, offline=True)
    assert receipt["architecture"] == "ricursive"
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert receipt["weights_source"] == {
        "repo": source["weights_repo"],
        "revision": source["weights_revision"],
    }
    assert receipt["metadata_source"] == {
        "repo": source["metadata_repo"],
        "revision": source["metadata_revision"],
    }
    assert hybrid.prepared_identity(output)["architecture"] == "ricursive"
    assert all(call[1]["local_files_only"] for call in calls)
    assert not any(pattern.endswith(".py") for _, options in calls for pattern in options["allow_patterns"])
    config = json.loads((output / "config.json").read_text())
    assert config["eos_token_id"] == [100265, 100257]
    assert config["token_codec_config"]["prompt_template"] == "tools-aware {{ messages[0]['content'] }}"
    assert config["token_codec_config"]["eos_token"] == "<|endoftext|>"
    # A different variant or shard set cannot impersonate this pinned recipe,
    # even if its count and aggregate byte size match.
    replacement = output / ("replacement" + source.get("weight_extension", ".safetensors"))
    (output / source["required_weight_files"][0]).rename(replacement)
    assert hybrid.prepared_identity(output) is None


def test_ripple_template_file_is_required_and_bounded(tmp_path):
    ripple_metadata(tmp_path)
    template = tmp_path / "chat_template.jinja"
    template.unlink()
    with pytest.raises(ValueError, match="missing pinned chat template"):
        hybrid.finish_metadata(tmp_path, "ripple-q8")
    template.write_text("x" * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="invalid chat template"):
        hybrid.finish_metadata(tmp_path, "ripple-q8")
    template.unlink()
    template.symlink_to(tmp_path / "tokenizer_config.json")
    with pytest.raises(ValueError, match="invalid chat template"):
        hybrid.finish_metadata(tmp_path, "ripple-q8")


def test_ripple_catalog_is_pinned_text_only_and_uses_native_preparation(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    class NoHub:
        def __getattr__(self, name):
            pytest.fail(f"native Ripple catalog unexpectedly accessed hub.{name}")

    family = next(item for item in support_matrix()["families"] if item["family_id"] == "ricursive")
    assert family["name"] == "Ripple-65B Preview"
    assert family["conversion"]["source_formats"] == ["Q8_0", "Q4_K_M"]
    assert family["modes"]["text"]["status"] == "conditional"
    assert family["modes"]["decision"]["status"] == "unsupported"
    assert family["modes"]["vision-language"]["status"] == "unsupported"
    report = model_catalog.family_catalog("ricursive", hub=NoHub())
    assert report["status"] == "ready"
    item = report["items"][0]
    assert item == native_catalog.catalog_item("ripple-q8")
    assert native_catalog.source_choice("ricursive", item) == "ripple-q8"
    assert native_catalog.source_choice("ricursive", {**item, "revision": "f" * 40}) is None
    assert native_catalog.source_choice("glm5_next", item) is None
    assert item["weight_bytes"] == 68_840_113_568
    assert item["size_label"] == "65B"
    assert item["parameter_count"] == 64_782_689_360
    assert item["artifact_modes"] == ["text"]
    q4 = report["items"][1]
    assert q4 == native_catalog.catalog_item("ripple-q4-k-m")
    assert q4["id"] != item["id"]
    assert q4["weight_bytes"] == 38_798_820_576
    assert native_catalog.source_choice("ricursive", q4) == "ripple-q4-k-m"
    # Generic conversion must not treat custom HF code as an implementation.
    assert preflight_model({"model_type": "ricursive"}, mode="text")["reasons"] == ["unsupported_architecture"]
    assert model_catalog.family_catalog("ricursive", offline=True, hub=NoHub())["status"] == "offline"


def test_ripple_catalog_pagination_preserves_both_same_repository_artifacts(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    emitted = []
    first = model_catalog.family_catalog("ricursive", limit=1, on_item=emitted.append)
    assert first["items"] == [native_catalog.catalog_item("ripple-q8")]
    assert emitted == first["items"]
    assert first["next_offset"] == 1
    second = model_catalog.family_catalog("ricursive", limit=1, offset=first["next_offset"])
    assert second["items"] == [native_catalog.catalog_item("ripple-q4-k-m")]
    assert second["next_offset"] is None
    assert model_catalog.family_catalog("ricursive", offset=2)["items"] == []


def test_ripple_q8_recipe_selects_only_q8_and_keeps_publisher_provenance():
    source = hybrid.SOURCES["ripple-q8"]
    assert source["weights_repo"] == "AtomicChat/Ripple-65B-Preview-GGUF"
    assert source["weights_revision"] == "57066fdb2348a53f2261f096e0a793b9554b4106"
    assert source["weights_patterns"] == ["Ripple-65B-Preview-Q8_0-*.gguf", "README.md"]
    assert source["metadata_repo"] == "ricursiveAI/Ripple-65B-Preview"
    assert source["required_files"] == ["LICENSE", "chat_template.jinja"]
