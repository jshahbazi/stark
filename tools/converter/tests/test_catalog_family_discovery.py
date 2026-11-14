"""Family discovery reaches alternate exports and explains rejected variants."""

import pytest
from test_model_catalog import PIN, Hub

from stark_convert import model_catalog as catalog
from stark_convert import native_catalog


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)
    monkeypatch.setattr(catalog, "_verify_hub_layout", lambda *args: dict.fromkeys(args[-1]))


def test_name_search_finds_untagged_model_and_still_checks_architecture(tmp_path):
    hub = Hub()
    good, unrelated = "someone/Qwen3-Untagged", "someone/Qwen3-Name-Only"
    hub.add(tmp_path, good, {"model_type": "qwen3"})
    hub.add(tmp_path, unrelated, {"model_type": "llama"})
    hub.searched["Qwen3"] = [good, unrelated]

    result = catalog.family_catalog("qwen3", hub=hub)

    assert result["status"] == "ready"
    assert [item["repo"] for item in result["items"]] == [good]
    assert result["items"][0]["revision"] == PIN


def test_large_alias_cannot_starve_other_aliases_or_name_search(tmp_path):
    hub = Hub()
    for index in range(50):
        repo = f"someone/First-Alias-{index}"
        hub.add(tmp_path, repo, {"model_type": "gemma4_text"})
    hub.discovered["gemma4_text"] = list(hub.repos)
    alternate, untagged = "someone/Unified-Export", "someone/gemma-4-Untagged"
    hub.add(tmp_path, alternate, {"model_type": "gemma4_unified_text"})
    hub.add(tmp_path, untagged, {"model_type": "gemma4_text"})
    hub.discovered["gemma4_unified_text"] = [alternate]
    hub.searched["gemma-4"] = [untagged]

    result = catalog.family_catalog("gemma4_text", hub=hub)

    assert result["status"] == "ready"
    assert {alternate, untagged} <= {item["repo"] for item in result["items"]}
    assert len(result["items"]) == 30 and result["next_offset"] == 30


def test_interleaved_pagination_visits_each_repository_once(tmp_path, monkeypatch):
    hub = Hub()
    for index in range(8):
        hub.add(tmp_path, f"someone/Model-{index}", {"model_type": "qwen3_5"})
    repos = list(hub.repos)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _: repos[:1])
    hub.discovered["qwen3_5"] = repos[:5]
    hub.discovered["qwen3_5_text"] = repos[2:6]
    hub.searched["Qwen3.5"] = repos[3:]
    hub.searched["Qwen3.6"] = repos[4:]
    offset, items = 0, []
    while offset is not None:
        result = catalog.family_catalog("qwen3_5", hub=hub, offset=offset, limit=2)
        assert result["status"] == "ready"
        items.extend(result["items"])
        offset = result["next_offset"]
    assert sorted(item["repo"] for item in items) == sorted(repos)


@pytest.mark.parametrize("family", [native_catalog.FAMILY_ID, native_catalog.FLINT_FAMILY_ID])
def test_native_family_preserves_pinned_download_and_explains_alternate_export(tmp_path, family):
    hub = Hub()
    repo = "someone/Alternate"
    hub.add(tmp_path, repo, {"model_type": family})
    hub.discovered[family] = [repo]

    result = catalog.family_catalog(family, hub=hub)

    assert result["status"] == "ready"
    rows = {item["repo"]: item for item in result["items"]}
    pinned = native_catalog.catalog_items(family)[0]
    assert rows[pinned["repo"]] == pinned
    alternate = rows[repo]
    assert alternate["downloadable"] is False and alternate["unavailable_reason"]
    assert native_catalog.source_choice(family, alternate) is None
    assert all(not event[-1].endswith(".safetensors") for event in hub.events if event[0] == "download")


def test_gemma4_quantized_no_ple_is_visible_with_bf16_requirement(tmp_path):
    hub = Hub()
    repo = "someone/gemma-4-12B-4bit"
    config = {
        "model_type": "gemma4_unified",
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size_per_layer_input": 0,
            "attention_k_eq_v": True,
        },
        "quantization": {"bits": 4, "group_size": 64},
    }
    hub.add(tmp_path, repo, config)
    hub.discovered["gemma4_unified"] = [repo]

    result = catalog.family_catalog("gemma4_text", hub=hub)

    assert result["status"] == "ready"
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["downloadable"] is False
    assert "BF16" in item["unavailable_reason"]
    assert item["quantization"] == "4-bit"


@pytest.mark.parametrize("architecture", ["Gemma4TextModel", "Gemma4UnifiedTextModel"])
def test_gemma4_text_export_architecture_hints_match_the_converter(architecture):
    assert catalog._generation_architecture({"architectures": [architecture]}, "gemma4_text")


def test_native_pinned_download_survives_discovery_service_failure():
    class BrokenHub:
        def HfApi(self):
            raise RuntimeError("private connection sentinel")

    result = catalog.family_catalog(native_catalog.FLINT_FAMILY_ID, hub=BrokenHub())

    assert result["status"] == "ready"
    assert result["items"] == native_catalog.catalog_items(native_catalog.FLINT_FAMILY_ID)
    assert result["has_more"] is True


def test_known_gemma_image_source_uses_the_supported_revision_after_upstream_moves(tmp_path, monkeypatch):
    from stark_convert.adapters.gemma4_vision import PROFILES

    hub = Hub()
    repo = "google/gemma-4-E2B-it"
    pinned = PROFILES[repo]["revision"]
    hub.add(tmp_path, repo, {"model_type": "gemma4_text"}, revision=pinned)
    hub.discovered["gemma4_text"] = [repo]
    model_info = hub.model_info

    def moved_head(*, repo_id, revision=None, **kwargs):
        if revision is None:
            raise AssertionError("The latest repository revision has no bound image import contract")
        assert revision == pinned
        return model_info(repo_id=repo_id, revision=revision, **kwargs)

    monkeypatch.setattr(hub, "model_info", moved_head)
    monkeypatch.setattr(catalog, "_catalog_image_metadata", lambda *args: True)

    result = catalog.family_catalog("gemma4_text", hub=hub)

    assert result["status"] == "ready" and len(result["items"]) == 1
    item = result["items"][0]
    assert item["revision"] == pinned
    assert item["artifact_modes"] == ["text", "vision-language"]
    assert item["downloadable"] is True
