"""Cached source rankings remain separate from Stark artifact admission."""

import socket
import urllib.request
from dataclasses import replace

import pytest

from stark_convert import guided_models as guided
from stark_convert import recommendation_leaderboard as module
from stark_convert.recommendation_benchmarks import BenchmarkRegistry


def host(**changes):
    return {
        "supported": True,
        "physical_memory_bytes": 128 * guided.GIB,
        "available_memory_bytes": 100 * guided.GIB,
        "free_disk_bytes": 512 * guided.GIB,
        "download_free_disk_bytes": 512 * guided.GIB,
        **changes,
    }


def recipe(name, parameters, *, weights=guided.GIB):
    return replace(
        guided.RECIPES[1],
        id=name,
        name=name,
        repo="example/" + name,
        logical_parameters=parameters,
        estimated_weight_bytes=weights,
    )


def registry(*, bindings=()):
    return BenchmarkRegistry({"schema_version": 1, "cohorts": [], "bindings": list(bindings)})


def snapshot(general, coding=None, *, repositories=None, sizes=(8, 24, 32, 128), speed=None):
    """Use synthetic scores and plans; these fixtures never qualify a model."""
    repositories = repositories or {
        name: "example/" + name for name in set(general) | set(coding or general)
    }
    return module.LeaderboardSnapshot(
        {
            "schema_version": 1,
            "captured": "2026-11-11",
            "model_repositories": repositories,
            "buckets": [
                {
                    "task": task,
                    "ram_gb": ram,
                    "url": f"https://whatmodel.sh/?gpu=apple&max=0&ram={ram}"
                    + ("&task=coding" if task == "coding" else ""),
                    "sha256": "a" * 64,
                    "models": [
                        {"id": name, "score": score, "memory_gb": 0.001, "speed": (speed or {}).get(name)}
                        for name, score in scores.items()
                    ],
                }
                for task, scores in [("general", general), ("coding", coding if coding is not None else general)]
                for ram in sizes
            ],
        }
    )


def report(candidates, leaderboard, *, task="general", preference="quality", machine=None, benchmarks=None):
    return guided.recommendation_report(
        host() if machine is None else machine,
        {},
        candidates=candidates,
        task=task,
        preference=preference,
        benchmarks=registry() if benchmarks is None else benchmarks,
        leaderboard=leaderboard,
    )


@pytest.mark.parametrize("task, expected", [("general", ["large", "small"]), ("coding", ["small", "large"])])
def test_quality_uses_the_selected_tasks_scores_after_admission(task, expected):
    candidates = [recipe("large", 10), recipe("small", 1)]
    cached = snapshot({"large": 95, "small": 15}, {"large": 10, "small": 99})
    for order in (candidates, list(reversed(candidates))):
        result = report(order, cached, task=task)
        assert [item["id"] for item in result["items"]] == expected
        assert result["leaderboard_ranking"]["applied"]
        assert result["leaderboard_ranking"]["task"] == task
        assert all(item["benchmarks"][0]["benchmark"] == "whatmodel_" + task for item in result["items"])
        assert all(item["benchmarks"][0]["used_for_ranking"] for item in result["items"])


@pytest.mark.parametrize("task", ["general", "coding"])
def test_speed_keeps_lighter_weights_despite_better_score_and_source_speed(task):
    candidates = [recipe("large", 10, weights=2 * guided.GIB), recipe("small", 1)]
    speeds = {
        "large": {"value": 9999, "unit": "tok/s", "how": "elsewhere", "on": "Other runtime"},
        "small": {"value": 1, "unit": "tok/s", "how": "elsewhere", "on": "Other runtime"},
    }
    result = report(candidates, snapshot({"large": 100, "small": 0}, speed=speeds), task=task, preference="speed")
    assert [item["id"] for item in result["items"]] == ["small", "large"]
    assert not result["leaderboard_ranking"]["applied"]
    assert not any(item["benchmarks"][0]["used_for_ranking"] for item in result["items"])
    assert result["items"][1]["benchmarks"][0]["source_speed"] == speeds["large"]
    assert "response speed has not been measured" in result["items"][0]["recommendation_reason"]


def test_high_source_score_and_tiny_source_plan_cannot_override_stark_memory_exclusion():
    large = recipe("large", 10, weights=256 * guided.GIB)
    small = recipe("small", 1)
    cached = snapshot({"large": 100, "small": 0})
    # Even a source list that includes the high-scoring model cannot admit its
    # independently planned 256 GiB weights to this synthetic 128 GiB machine.
    result = report([large, small], cached)
    assert result["status"] == "ready"
    assert [item["id"] for item in result["items"]] == ["small"]
    assert result["items"][0]["benchmarks"][0]["score"] == "0"
    assert result["leaderboard_ranking"]["applied"]


def test_missing_score_stays_unknown_after_scored_options_and_all_unknown_falls_back():
    unknown, scored = recipe("unknown", 100), recipe("scored", 1)
    result = report([unknown, scored], snapshot({"unknown": None, "scored": 0}))
    assert [item["id"] for item in result["items"]] == ["scored", "unknown"]
    assert result["items"][1]["benchmarks"] == []
    assert "quality is unknown" in result["items"][1]["recommendation_reason"]
    fallback = report([scored, unknown], snapshot({"unknown": None, "scored": None}), task="coding")
    assert [item["id"] for item in fallback["items"]] == ["unknown", "scored"]
    assert not fallback["leaderboard_ranking"]["applied"]
    assert all(item["benchmarks"] == [] for item in fallback["items"])
    assert "resource-based fallback; coding quality is unknown" in fallback["items"][0]["recommendation_reason"]


def test_quantized_parent_score_requires_exact_reviewed_revision():
    binding = {
        "repo": "export/quantized",
        "revision": "b" * 40,
        "model_id": "example/parent",
        "source": {
            "url": "https://example.com/source",
            "publisher": "Synthetic fixture",
            "revision": "fixture",
            "sha256": "a" * 64,
        },
        "relationship": "reviewed_quantized_export_reference_only",
    }
    quantized = replace(recipe("quantized", 1), repo=binding["repo"], revision=binding["revision"])
    other = recipe("other", 10)
    cached = snapshot({"parent": 99, "other": 10})
    bound = registry(bindings=[binding])
    result = report([other, quantized], cached, benchmarks=bound)
    assert [item["id"] for item in result["items"]] == ["quantized", "other"]
    evidence = result["items"][0]["benchmarks"][0]
    assert evidence["model_id"] == "example/parent"
    assert evidence["candidate_revision"] == binding["revision"]
    changed = replace(quantized, revision="c" * 40)
    result = report([changed, other], cached, benchmarks=bound)
    assert [item["id"] for item in result["items"]] == ["other", "quantized"]
    assert result["items"][1]["benchmarks"] == []
    assert "quality is unknown" in result["items"][1]["recommendation_reason"]


@pytest.mark.parametrize("task", ["general", "coding"])
@pytest.mark.parametrize("physical_gib, expected", [(24, 24), (36, 32), (7, None)])
def test_ram_selection_uses_the_largest_bucket_that_does_not_exceed_physical_ram(task, physical_gib, expected):
    cached = snapshot({"small": 20})
    bucket, records = cached.for_host(task, physical_gib * guided.GIB)
    assert (bucket["ram_gb"] if bucket else None) == expected
    assert bool(records) == (expected is not None)


def test_negative_scores_are_recorded_and_null_scores_remain_unknown():
    cached = snapshot({"negative": -3.25, "unknown": None})
    _, records = cached.for_host("general", 128 * guided.GIB)
    assert records["example/negative"]["score"] == -3.25
    assert "example/unknown" not in records
    result = report([recipe("unknown", 100), recipe("negative", 1)], cached)
    assert [item["id"] for item in result["items"]] == ["negative", "unknown"]
    assert result["items"][0]["benchmarks"][0]["score"] == "-3.25"
    assert result["items"][1]["benchmarks"] == []


def test_packaged_snapshot_loads_both_tasks_all_ram_buckets_and_known_reference_scores():
    cached = module.load_snapshot()
    sizes = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512}
    assert len(cached.buckets) == 22
    assert set(cached.buckets) == {(task, size) for task in ("general", "coding") for size in sizes}
    for bucket, records in cached.buckets.values():
        assert bucket["models"] and records
        assert module.HEX64.fullmatch(bucket["sha256"])
    _, general = cached.for_host("general", 128 * guided.GIB)
    _, coding = cached.for_host("coding", 128 * guided.GIB)
    assert general["Qwen/Qwen3.8-Flash-Next"]["score"] == 73.4
    assert coding["Qwen/Qwen3.8-Flash-Next"]["score"] == 76.1
    assert general["Qwen/Qwen3-4B"]["score"] == 51.2


@pytest.mark.parametrize("payload", ["{", "[]", "{}", '{"schema_version": 999}'])
def test_malformed_optional_snapshot_falls_back_without_network_or_disabling_admission(monkeypatch, tmp_path, payload):
    def forbidden_network(*_args, **_kwargs):
        pytest.fail("A cached recommendation must never contact the network")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "module.py"))
    (tmp_path / "recommendation_leaderboard.json").write_text(payload)
    module.load_snapshot.cache_clear()
    try:
        cached = module.load_snapshot()
        assert cached.buckets == {}
        result = report([recipe("small", 1)], cached)
        assert result["status"] == "ready"
        assert not result["leaderboard_ranking"]["applied"]
        assert [item["id"] for item in result["items"]] == ["small"]
    finally:
        module.load_snapshot.cache_clear()
