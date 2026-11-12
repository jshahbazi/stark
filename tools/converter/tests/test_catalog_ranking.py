"""Reference scores order a family catalogue without benchmarking or admitting any row."""

import io
from types import SimpleNamespace

import pytest
from test_catalog_api_reuse import ListingHub, listed, requests
from test_desktop_model_browser import report as page_report
from test_model_catalog import PIN

from stark_convert import catalog_ranking, native_catalog
from stark_convert import model_catalog as catalog
from stark_convert.catalog_ranking import ReferenceScores
from stark_convert.desktop import Desktop
from stark_convert.recommendation_leaderboard import GIB, LeaderboardSnapshot

CONFIG = {"model_type": "qwen3", "torch_dtype": "float16"}
PACKED = {"model_type": "qwen3", "quantization": {"group_size": 64, "bits": 4}}


def scores():
    def bucket(task, ram, rows):
        return {
            "task": task,
            "ram_gb": ram,
            "url": "https://whatmodel.sh/?fixture",
            "sha256": "0" * 64,
            "models": rows,
        }

    snapshot = LeaderboardSnapshot(
        {
            "schema_version": 1,
            "captured": "2026-11-11",
            "model_repositories": {
                "big-70b": "maker/Big-70B",
                "small-4b": "maker/Small-4B",
                "coder-7b": "maker/Coder-7B",
                "twin-one": "one/Twin",
                "twin-two": "two/Twin",
            },
            "buckets": [
                bucket("general", 16, [{"id": "small-4b", "score": 50.5}]),
                bucket(
                    "general",
                    128,
                    [
                        {"id": "big-70b", "score": 70},
                        {"id": "small-4b", "score": 50.5},
                        {"id": "twin-one", "score": 40},
                        {"id": "twin-two", "score": 41},
                    ],
                ),
                bucket("coding", 128, [{"id": "coder-7b", "score": 60}, {"id": "small-4b", "score": 30}]),
            ],
        }
    )
    return ReferenceScores(snapshot, {("exporter/Reviewed", PIN): "maker/Big-70B"})


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)
    monkeypatch.setattr(catalog, "_verify_hub_layout", lambda *args: dict.fromkeys(args[-1]))
    monkeypatch.setattr(catalog_ranking, "reference_scores", scores)


def test_score_belongs_to_the_exact_model_or_a_reviewed_export():
    ranking = scores()

    exact = ranking.reference("maker/Small-4B", PIN, "BF16", [], "Small-4B")
    assert exact == {
        "model_id": "maker/Small-4B",
        "relationship": "benchmarked_model",
        "general": 50.5,
        "coding": 30,
        "source_url": "https://whatmodel.sh/model/small-4b",
        "captured": "2026-11-11",
    }
    coding_only = ranking.reference("maker/Coder-7B", PIN, "BF16", [], "Coder-7B")
    assert (coding_only["general"], coding_only["coding"]) == (None, 60)
    assert coding_only["source_url"].endswith("/coder-7b")

    reviewed = ranking.reference("exporter/Reviewed", PIN, "4-bit", [], "Reviewed")
    assert (reviewed["model_id"], reviewed["relationship"]) == ("maker/Big-70B", "reviewed_export")
    # A reviewed binding covers one revision only.
    assert ranking.reference("exporter/Reviewed", "b" * 40, "4-bit", [], "Reviewed") is None


def test_quantized_export_shows_its_declared_or_named_parent_score():
    ranking = scores()
    declared = ["base_model:quantized:maker/Big-70B", "base_model:maker/Big-70B", "mlx"]

    by_tag = ranking.reference("exporter/Anything-q4", PIN, "4-bit", declared, "Anything-q4")
    assert (by_tag["model_id"], by_tag["relationship"], by_tag["general"]) == (
        "maker/Big-70B",
        "declared_quantization",
        70,
    )
    by_name = ranking.reference("exporter/Small-4B-4bit", PIN, "4-bit", [], "Small-4B")
    assert (by_name["model_id"], by_name["relationship"]) == ("maker/Small-4B", "named_quantization")
    # An unscored declared parent leaves the exact name as the only evidence.
    chained = ["base_model:quantized:exporter/Small-4B-bf16"]
    assert (
        ranking.reference("exporter/Small-4B-4bit", PIN, "4-bit", chained, "Small-4B")["model_id"] == "maker/Small-4B"
    )
    # The export's own publisher settles a name that two scored models share.
    assert ranking.reference("two/Twin-8bit", PIN, "8-bit", [], "Twin")["model_id"] == "two/Twin"
    assert ranking.reference("exporter/Twin-8bit", PIN, "8-bit", [], "Twin") is None
    assert (
        ranking.reference("exporter/Other-4bit", PIN, "4-bit", ["base_model:quantized:x/A", declared[0]], "Other")
        is None
    )


def test_full_precision_derivatives_never_inherit_a_score():
    ranking = scores()
    tags = ["base_model:quantized:maker/Big-70B", "base_model:finetune:maker/Big-70B"]

    assert ranking.reference("tuner/Big-70B-Tuned", PIN, "BF16", tags, "Big-70B-Tuned") is None
    # A re-upload under another owner is not the benchmarked repository.
    assert ranking.reference("mirror/Small-4B", PIN, "BF16", [], "Small-4B") is None
    assert ranking.reference("exporter/Small-4B-4bit", PIN, "4-bit", [], "Small-4B-4bit") is None


def test_priority_prefers_scored_models_from_a_list_this_machine_can_use():
    ranking = scores()

    assert ranking.priority("maker/Small-4B", [], 16 * GIB) == (True, 50.5)
    assert ranking.priority("maker/Big-70B", [], 16 * GIB) == (False, 70)
    assert ranking.priority("maker/Big-70B", [], 128 * GIB) == (True, 70)
    assert ranking.priority("maker/Big-70B", [], None) == (False, 70)
    assert ranking.priority("exporter/Small-4B-MLX-4bit", [], 16 * GIB) == (True, 50.5)
    assert ranking.priority("exporter/X", ["base_model:quantized:maker/Big-70B"], 128 * GIB) == (True, 70)
    assert ranking.priority("maker/Coder-7B", [], 128 * GIB) == (False, 60)
    assert ranking.priority("someone/Unknown", ["base_model:finetune:maker/Big-70B"], 128 * GIB) is None


def ranked_hub(tmp_path):
    hub = ListingHub()
    files = ["config.json", "model.safetensors"]
    order = ["someone/Popular", "maker/Small-4B", "someone/Second", "exporter/Small-4B-4bit", "maker/Big-70B"]
    for repo in order:
        packed = repo.endswith("-4bit")
        hub.add(tmp_path / "hub", repo, PACKED if packed else CONFIG, parameters={"U32": 10} if packed else None)
        hub.listing[repo] = listed(files)
    hub.discovered["qwen3"] = order
    return hub


def test_scored_sources_are_inspected_and_reported_first(tmp_path):
    hub = ranked_hub(tmp_path)

    first = catalog.family_catalog("qwen3", hub=hub, limit=2, physical_memory=128 * GIB)
    assert sorted(item["repo"] for item in first["items"]) == ["maker/Big-70B", "maker/Small-4B"]
    assert first["next_offset"] == 2
    second = catalog.family_catalog("qwen3", hub=hub, limit=2, offset=2, physical_memory=128 * GIB)
    assert sorted(item["repo"] for item in second["items"]) == ["exporter/Small-4B-4bit", "someone/Popular"]
    third = catalog.family_catalog("qwen3", hub=hub, limit=2, offset=second["next_offset"], physical_memory=128 * GIB)
    assert [item["repo"] for item in third["items"]] == ["someone/Second"] and third["next_offset"] is None

    # A smaller machine sees the model from its own RAM list, and that model's export, first.
    small = catalog.family_catalog("qwen3", hub=hub, limit=2, physical_memory=16 * GIB)
    assert sorted(item["repo"] for item in small["items"]) == ["exporter/Small-4B-4bit", "maker/Small-4B"]

    rows = {item["repo"]: item.get("reference_score") for report in (first, second, third) for item in report["items"]}
    assert rows["maker/Big-70B"]["relationship"] == "benchmarked_model" and rows["maker/Big-70B"]["general"] == 70
    assert rows["exporter/Small-4B-4bit"]["relationship"] == "named_quantization"
    assert rows["exporter/Small-4B-4bit"]["model_id"] == "maker/Small-4B"
    assert rows["someone/Popular"] is None and rows["someone/Second"] is None


def test_score_does_not_prioritize_a_source_without_safetensors_or_change_admission(tmp_path):
    hub = ranked_hub(tmp_path)
    hub.listing["maker/Big-70B"] = listed(["config.json", "model.gguf"])
    hub.repos["maker/Big-70B"][1].siblings = [
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="model.gguf", size=10),
    ]

    first = catalog.family_catalog("qwen3", hub=hub, limit=2, physical_memory=128 * GIB)
    assert sorted(item["repo"] for item in first["items"]) == ["exporter/Small-4B-4bit", "maker/Small-4B"]
    everything = catalog.family_catalog("qwen3", hub=hub, physical_memory=128 * GIB)
    (unavailable,) = [item for item in everything["items"] if item["repo"] == "maker/Big-70B"]
    # The score is shown, but the row stays unavailable for its own reason.
    assert unavailable["downloadable"] is False and unavailable["reference_score"]["general"] == 70


def test_failed_listing_keeps_scored_sources_found_before_the_failure(tmp_path, monkeypatch):
    hub = ranked_hub(tmp_path)
    original = hub.list_models

    def failing(**kwargs):
        entries = original(**kwargs)
        yield from entries[:2]
        if entries:
            raise OSError("private Hub error")

    monkeypatch.setattr(hub, "list_models", failing)
    result = catalog.family_catalog("qwen3", hub=hub, physical_memory=128 * GIB)

    assert result["status"] == "ready" and result["next_offset"] is not None
    assert sorted(item["repo"] for item in result["items"]) == ["maker/Small-4B", "someone/Popular"]


def test_guided_planning_scan_is_not_reordered_or_scored(tmp_path):
    hub = ranked_hub(tmp_path)

    planned = catalog.family_catalog("qwen3", hub=hub, limit=2, planning=True)

    assert sorted(item["repo"] for item in planned["items"]) == ["maker/Small-4B", "someone/Popular"]
    assert all("reference_score" not in item for item in planned["items"])
    assert all(event[2]["limit"] == 100 for event in requests(hub, "list"))


def reference(**changes):
    return {
        "model_id": "maker/Small-4B",
        "relationship": "named_quantization",
        "general": 50.5,
        "coding": None,
        "source_url": "https://whatmodel.sh/model/small-4b",
        "captured": "2026-11-11",
        **changes,
    }


@pytest.fixture
def host(tmp_path):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    yield value
    value.stop(quiet=True)


def with_reference(value):
    page = page_report()
    page["items"][0]["reference_score"] = value
    return page


def test_backend_passes_a_well_formed_reference_score(host):
    assert host.catalog_report(with_reference(reference()), "qwen3")["items"][0]["reference_score"] == reference()
    coding = reference(general=None, coding=44, relationship="benchmarked_model")
    assert host.catalog_report(with_reference(coding), "qwen3")["items"][0]["reference_score"] == coding


@pytest.mark.parametrize(
    "value",
    [
        "50.5",
        reference(general=None, coding=None),
        reference(general="50"),
        reference(general=True),
        reference(general=float("inf")),
        reference(coding=5000),
        reference(relationship="similar_name"),
        reference(model_id="not a repository"),
        reference(source_url="https://example.com/model/small-4b"),
        reference(source_url="https://whatmodel.sh/model/../admin"),
        reference(captured="yesterday"),
        reference(extra=1),
        {key: value for key, value in reference().items() if key != "captured"},
    ],
)
def test_backend_rejects_a_malformed_reference_score(host, value):
    with pytest.raises(ValueError):
        host.catalog_report(with_reference(value), "qwen3")


def test_native_install_identity_ignores_only_the_reference_score():
    family = native_catalog.FAMILY_ID
    item = native_catalog.catalog_items(family)[0]
    choice = native_catalog.source_choice(family, item)

    assert choice is not None
    assert native_catalog.source_choice(family, {**item, "reference_score": reference()}) == choice
    assert native_catalog.source_choice(family, {**item, "reference_score": reference(), "note": "x"}) is None
    assert native_catalog.source_choice(family, {**item, "revision": "b" * 40}) is None
