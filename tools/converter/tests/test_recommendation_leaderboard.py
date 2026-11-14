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
    repositories = repositories or {name: "example/" + name for name in set(general) | set(coding or general)}
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


@pytest.mark.parametrize("task, expected", [("general", ["large"]), ("coding", ["small"])])
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
def test_speed_selects_score_within_payload_budget_without_using_source_speed(task):
    candidates = [recipe("large", 10, weights=2 * guided.GIB), recipe("small", 1)]
    speeds = {
        "large": {"value": 1, "unit": "tok/s", "how": "elsewhere", "on": "Other runtime"},
        "small": {"value": 9999, "unit": "tok/s", "how": "elsewhere", "on": "Other runtime"},
    }
    result = report(candidates, snapshot({"large": 90, "small": 85}, speed=speeds), task=task, preference="speed")
    assert [item["id"] for item in result["items"]] == ["large", "small"]
    assert result["leaderboard_ranking"]["applied"]
    assert all(item["benchmarks"][0]["used_for_ranking"] for item in result["items"])
    assert result["items"][0]["benchmarks"][0]["source_speed"] == speeds["large"]
    assert result["speed_policy"]["applied"]
    assert result["speed_policy"]["weight_budget_bytes"] == 32 * guided.GIB


@pytest.mark.parametrize("task, floor", [("general", 40), ("coding", 35)])
def test_speed_requires_task_floor_for_every_recommended_alternative(task, floor):
    candidates = [recipe(name, 1) for name in ("below", "edge", "above", "unknown")]
    cached = snapshot({"below": floor - 0.1, "edge": floor, "above": floor + 1, "unknown": None})
    result = report(candidates, cached, task=task, preference="speed")
    assert [item["id"] for item in result["items"]] == ["above", "edge"]
    assert result["speed_policy"]["minimum_score"] == floor
    assert all(float(item["benchmarks"][0]["score"]) >= floor for item in result["items"])


def test_speed_applies_different_general_and_coding_floors_to_the_same_score():
    cached = snapshot({"small": 37})
    candidate = recipe("small", 1)
    general = report([candidate], cached, preference="speed")
    coding = report([candidate], cached, task="coding", preference="speed")
    assert general["status"] == "unavailable" and general["items"] == []
    assert [item["id"] for item in coding["items"]] == [candidate.id]


@pytest.mark.parametrize("ram, expected", [(16, "medium"), (32, "large"), (64, "largest")])
def test_speed_payload_budget_scales_model_selection_with_ram(ram, expected):
    candidates = [
        recipe("small", 1),
        recipe("medium", 4, weights=4 * guided.GIB),
        recipe("large", 8, weights=8 * guided.GIB),
        recipe("largest", 16, weights=16 * guided.GIB),
    ]
    cached = snapshot({"small": 50, "medium": 60, "large": 70, "largest": 80}, sizes=(8, 16, 32, 64, 128))
    machine = host(physical_memory_bytes=ram * guided.GIB, available_memory_bytes=(ram - 2) * guided.GIB)
    result = report(candidates, cached, preference="speed", machine=machine)
    assert result["items"][0]["id"] == expected
    assert result["speed_policy"]["weight_budget_bytes"] == ram * guided.GIB // 4


def test_speed_payload_boundary_is_inclusive_and_quality_can_use_larger_model():
    edge = recipe("edge", 8, weights=8 * guided.GIB)
    over = recipe("over", 9, weights=8 * guided.GIB + 1)
    cached = snapshot({"edge": 50, "over": 99})
    machine = host(physical_memory_bytes=32 * guided.GIB, available_memory_bytes=30 * guided.GIB)
    speed = report([over, edge], cached, preference="speed", machine=machine)
    quality = report([over, edge], cached, preference="quality", machine=machine)
    assert [item["id"] for item in speed["items"]] == [edge.id]
    assert quality["items"][0]["id"] == over.id
    assert not quality["speed_policy"]["applied"]


def test_speed_budgets_actual_export_weights_without_claiming_quantized_artifact_quality():
    parent = "example/parent"
    quantized = replace(recipe("quantized", 8_000_000_000, weights=4 * guided.GIB), quantization="4-bit")
    bf16 = replace(recipe("bf16", 8_000_000_000, weights=16 * guided.GIB), quantization="BF16")
    bindings = [
        {
            "repo": candidate.repo,
            "revision": candidate.revision,
            "model_id": parent,
            "source": {
                "url": "https://example.com/source",
                "publisher": "Synthetic fixture",
                "revision": "fixture",
                "sha256": "a" * 64,
            },
            "relationship": "reviewed_quantized_export_reference_only",
        }
        for candidate in (quantized, bf16)
    ]
    cached = snapshot({"parent": 70})
    machine = host(physical_memory_bytes=32 * guided.GIB, available_memory_bytes=30 * guided.GIB)
    result = report(
        [bf16, quantized], cached, preference="speed", machine=machine, benchmarks=registry(bindings=bindings)
    )
    assert [item["id"] for item in result["items"]] == [quantized.id]
    assert result["items"][0]["quantization"] == "4-bit"
    evidence = result["items"][0]["benchmarks"][0]
    assert evidence["model_id"] == parent
    assert evidence["candidate_repo"] == quantized.repo
    assert evidence["quantization_quality_evaluated"] is False
    assert evidence["relationship"] == "parent_model_reference_only"


def test_speed_score_ties_prefer_lower_runtime_then_lighter_weights_deterministically():
    high_runtime = replace(recipe("high-runtime", 1), estimated_runtime_bytes=20 * guided.GIB)
    heavier = replace(recipe("heavier", 2, weights=2 * guided.GIB), estimated_runtime_bytes=10 * guided.GIB)
    lighter = replace(recipe("lighter", 1), estimated_runtime_bytes=10 * guided.GIB)
    cached = snapshot({"high-runtime": 70, "heavier": 70, "lighter": 70})
    for candidates in ([high_runtime, heavier, lighter], [lighter, heavier, high_runtime]):
        result = report(candidates, cached, preference="speed")
        assert [item["id"] for item in result["items"]] == [lighter.id, heavier.id, high_runtime.id]


@pytest.mark.parametrize("task, tiny_score", [("general", 38.4), ("coding", 32.8)])
def test_speed_does_not_recommend_below_floor_half_billion_model_on_high_ram(task, tiny_score):
    tiny = recipe("tiny", 500_000_000, weights=guided.GIB // 2)
    useful = recipe("useful", 8_000_000_000, weights=8 * guided.GIB)
    cached = snapshot({"tiny": tiny_score, "useful": 60}, sizes=(8, 128, 512))
    result = report(
        [tiny, useful],
        cached,
        task=task,
        preference="speed",
        machine=host(physical_memory_bytes=512 * guided.GIB, available_memory_bytes=500 * guided.GIB),
    )
    assert [item["id"] for item in result["items"]] == [useful.id]


def test_speed_recommendation_uses_capacity_and_startup_still_checks_current_memory():
    large = replace(recipe("large", 16, weights=16 * guided.GIB), estimated_runtime_bytes=101 * guided.GIB)
    small = recipe("small", 1)
    machine = host()
    result = report([large, small], snapshot({"large": 99, "small": 95}), preference="speed", machine=machine)
    assert [item["id"] for item in result["items"]] == [large.id, small.id]
    assert not guided.execution_allowed(large, machine)


@pytest.mark.parametrize("scores", [{"small": None}, {"small": 34}])
def test_speed_without_qualifying_evidence_is_unavailable_with_recovery_guidance(scores):
    result = report([recipe("small", 1)], snapshot(scores), preference="speed")
    assert result["status"] == "unavailable" and result["items"] == []
    assert "quality" in result["message"].lower()
    assert "advanced" in result["message"].lower()


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


def test_missing_score_is_not_a_quality_alternative_and_all_unknown_offers_one_fallback():
    unknown, scored = recipe("unknown", 100), recipe("scored", 1)
    result = report([unknown, scored], snapshot({"unknown": None, "scored": 0}))
    assert [item["id"] for item in result["items"]] == ["scored"]
    fallback = report([scored, unknown], snapshot({"unknown": None, "scored": None}), task="coding")
    assert [item["id"] for item in fallback["items"]] == ["unknown"]
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
    assert [item["id"] for item in result["items"]] == ["quantized"]
    evidence = result["items"][0]["benchmarks"][0]
    assert evidence["model_id"] == "example/parent"
    assert evidence["candidate_revision"] == binding["revision"]
    changed = replace(quantized, revision="c" * 40)
    result = report([changed, other], cached, benchmarks=bound)
    assert [item["id"] for item in result["items"]] == ["other"]


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
    assert [item["id"] for item in result["items"]] == ["negative"]
    assert result["items"][0]["benchmarks"][0]["score"] == "-3.25"


@pytest.mark.parametrize("task", ["general", "coding"])
@pytest.mark.parametrize("preference", ["quality", "speed"])
def test_alternatives_have_an_inclusive_ten_point_gap(task, preference):
    candidates = [recipe(name, 1) for name in ("main", "near", "edge", "distant", "unknown")]
    cached = snapshot({"main": 73.7, "near": 73.6, "edge": 63.7, "distant": 63.69, "unknown": None})
    for order in (candidates, list(reversed(candidates))):
        result = report(order, cached, task=task, preference=preference)
        assert [item["id"] for item in result["items"]] == ["main", "near", "edge"]


@pytest.mark.parametrize("preference", ["quality", "speed"])
def test_screenshot_coding_gap_does_not_fill_alternative_slots(preference):
    candidates = [recipe(name, 1) for name in ("qwen38", "coder14", "qwen14")]
    result = report(
        candidates, snapshot({"qwen38": 73.7, "coder14": 51, "qwen14": 48.4}), task="coding", preference=preference
    )
    assert [item["id"] for item in result["items"]] == ["qwen38"]


@pytest.mark.parametrize("preference", ["quality", "speed"])
def test_modes_do_not_offer_multiple_exports_of_the_same_parent(preference):
    primary = recipe("primary", 1)
    duplicate = replace(primary, id="duplicate")
    alternative = recipe("alternative", 1)
    result = report(
        [primary, duplicate, alternative], snapshot({"primary": 70, "alternative": 69}), preference=preference
    )
    assert [item["id"] for item in result["items"]] == ["duplicate", "alternative"]


@pytest.mark.parametrize("preference", ["quality", "speed"])
def test_reviewed_exports_of_one_parent_cannot_crowd_out_another_model(preference):
    exports = [recipe("export-one", 1), recipe("export-two", 1)]
    alternative = recipe("alternative", 1)
    bindings = [
        {
            "repo": candidate.repo,
            "revision": candidate.revision,
            "model_id": "example/parent",
            "source": {
                "url": "https://example.com/source",
                "publisher": "Synthetic fixture",
                "revision": "fixture",
                "sha256": "a" * 64,
            },
            "relationship": "reviewed_quantized_export_reference_only",
        }
        for candidate in exports
    ]
    result = report(
        exports + [alternative],
        snapshot({"parent": 70, "alternative": 69}),
        preference=preference,
        benchmarks=registry(bindings=bindings),
    )
    assert len(result["items"]) == 2
    assert [item["benchmarks"][0]["model_id"] for item in result["items"]] == ["example/parent", "example/alternative"]


@pytest.mark.parametrize("preference", ["quality", "speed"])
@pytest.mark.parametrize("available_gib", [None, 0, 1, 72, 100])
def test_scored_offers_are_stable_across_current_memory_changes(preference, available_gib):
    strong = replace(recipe("strong", 100), estimated_runtime_bytes=87 * guided.GIB)
    small = recipe("small", 1)
    machine = host(available_memory_bytes=None if available_gib is None else available_gib * guided.GIB)
    result = report(
        [strong, small],
        snapshot({"strong": 76.1, "small": 73.7}),
        task="coding",
        preference=preference,
        machine=machine,
    )
    assert [item["id"] for item in result["items"]] == ["strong", "small"]
    assert guided.execution_allowed(strong, machine) is (available_gib is not None and available_gib >= 87)


@pytest.mark.parametrize("blocker", ["gpu", "disk", "excluded", "unknown-score"])
def test_capacity_policy_preserves_other_resource_and_score_gates(blocker):
    strong = replace(recipe("strong", 100, weights=20 * guided.GIB), estimated_runtime_bytes=87 * guided.GIB)
    small = recipe("small", 1)
    machine = host(available_memory_bytes=72 * guided.GIB)
    if blocker == "gpu":
        machine.update(gpu_available=True, unified_memory=True, gpu_memory_budget_bytes=60 * guided.GIB)
    if blocker == "disk":
        machine["free_disk_bytes"] = 10 * guided.GIB
    cached = snapshot({"strong": None if blocker == "unknown-score" else 76.1, "small": 73.7})
    result = guided.recommendation_report(
        machine,
        {},
        [strong.id] if blocker == "excluded" else (),
        candidates=[strong, small],
        task="coding",
        preference="quality",
        benchmarks=registry(),
        leaderboard=cached,
    )
    assert [item["id"] for item in result["items"]] == ["small"]


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
        speed = report([recipe("small", 1)], cached, preference="speed")
        assert speed["status"] == "unavailable" and speed["items"] == []
        assert "quality" in speed["message"].lower()
    finally:
        module.load_snapshot.cache_clear()
