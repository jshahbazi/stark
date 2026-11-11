"""Comparable published references never replace artifact/resource admission."""

from copy import deepcopy
from dataclasses import replace

import pytest

from stark_convert import guided_models as guided
from stark_convert import recommendation_benchmarks as module
from stark_convert.guided_memory_profiles import source_profile


def cohort(identity="test", *, benchmark="livecodebench", scores=None):
    return {
        "id": identity,
        "benchmark": benchmark,
        "label": "Synthetic benchmark cohort",
        "metric": "accuracy",
        "settings": "Synthetic shared harness and split",
        "source": {
            "url": "https://example.com/scores",
            "publisher": "Fixture",
            "revision": "test",
            "sha256": "a" * 64,
        },
        "authority": "benchmark_publisher",
        "results": [
            {"model_id": model, "score": score}
            for model, score in (scores or {"example/large": "10", "example/small": "90"}).items()
        ],
    }


def registry(*cohorts, bindings=()):
    return module.BenchmarkRegistry({"schema_version": 1, "cohorts": list(cohorts), "bindings": list(bindings)})


def recipe(name, parameters, *, weights=guided.GIB):
    return replace(
        guided.RECIPES[1],
        id=name,
        name=name,
        repo="example/" + name,
        logical_parameters=parameters,
        estimated_weight_bytes=weights,
    )


def host():
    return {
        "supported": True,
        "physical_memory_bytes": 128 * guided.GIB,
        "available_memory_bytes": 100 * guided.GIB,
        "free_disk_bytes": 256 * guided.GIB,
        "download_free_disk_bytes": 256 * guided.GIB,
    }


def test_scores_reorder_known_slots_and_preserve_unknown_position():
    large, unknown, small = recipe("large", 10), recipe("unknown", 5), recipe("small", 1)
    report = guided.recommendation_report(
        host(), {}, candidates=[small, unknown, large], task="files", benchmarks=registry(cohort(benchmark="bfcl"))
    )
    assert [item["id"] for item in report["items"]] == ["small", "unknown", "large"]
    assert report["benchmark_ranking"]["applied"]
    assert report["items"][0]["benchmarks"][0]["used_for_ranking"]
    assert "Stark artifact has not been benchmarked" in report["items"][0]["recommendation_reason"]
    assert report["items"][1]["benchmarks"] == []
    assert "unknown" in report["items"][1]["benchmark_details"][0]


def test_coding_evidence_displaces_unknown_models_including_first_place():
    large, unknown, small = recipe("large", 10), recipe("unknown", 50), recipe("small", 1)
    report = guided.recommendation_report(
        host(), {}, candidates=[small, unknown, large], task="code", benchmarks=registry(cohort())
    )
    assert [item["id"] for item in report["items"]] == ["small", "large", "unknown"]
    assert report["benchmark_ranking"]["applied"]
    assert "Prioritizes published coding benchmark evidence" in report["items"][0]["recommendation_reason"]
    assert "It follows models with that evidence" in report["items"][2]["recommendation_reason"]
    assert "missing scores remain unknown" in report["message"]


@pytest.mark.parametrize("score", ["0", "90"])
def test_unsure_promotes_single_instruction_reference_over_unbenchmarked_capacity(score):
    unknown, scored = recipe("unknown", 125_000_000_000), recipe("small", 1)
    report = guided.recommendation_report(
        host(),
        {},
        candidates=[unknown, scored],
        task="unsure",
        benchmarks=registry(cohort(benchmark="ifeval", scores={"example/small": score})),
    )
    assert [item["id"] for item in report["items"]] == ["small", "unknown"]
    assert report["items"][0]["benchmarks"][0]["used_for_ranking"]
    assert "Stark artifact has not been benchmarked" in report["items"][0]["recommendation_reason"]
    assert "general-purpose answer quality is unknown" in report["items"][1]["recommendation_reason"]


def test_unsure_prioritizes_workflow_checks_within_instruction_evidence():
    checked = guided.RECIPES[0]
    unchecked = recipe("unchecked", 125_000_000_000)
    scores = registry(cohort(benchmark="ifeval", scores={checked.repo: "50", unchecked.repo: "99"}))
    report = guided.recommendation_report(host(), {}, candidates=[unchecked, checked], benchmarks=scores)
    assert [item["id"] for item in report["items"]] == [checked.id, unchecked.id]
    assert "recorded bounded native chat/tool workflow checks" in report["items"][0]["recommendation_reason"]


def test_unsure_without_benchmarks_prefers_recorded_starters_over_flint():
    machine = {**host(), "gpu_available": True, "unified_memory": True, "gpu_memory_budget_bytes": 115_448_725_504}
    flint = next(candidate for candidate in guided.native_recipes() if candidate.native_choice == "flint-int4")
    report = guided.recommendation_report(
        machine, {}, candidates=[flint, *reversed(guided.RECIPES)], benchmarks=module.load_registry()
    )
    assert [item["id"] for item in report["items"]] == [r.id for r in (*guided.RECIPES, flint)]
    assert not report["benchmark_ranking"]["applied"]
    assert "No eligible instruction-following references" in report["message"]


def test_unsure_128gb_mac_promotes_packaged_instruction_evidence_over_flint():
    # Synthetic memory geometry isolates policy; no live model fit/quality claim.
    instructions = replace(recipe("instructions", 14_000_000_000), repo="Qwen/Qwen2.5-14B-Instruct")
    machine = {
        **host(),
        "gpu_available": True,
        "unified_memory": True,
        "gpu_memory_budget_bytes": 115_448_725_504,
    }
    report = guided.recommendation_report(
        machine, {}, candidates=[*guided.native_recipes(), *guided.RECIPES, instructions]
    )
    assert report["items"][0]["repo"] == instructions.repo
    assert report["items"][0]["selection_basis"] == "instruction_reference"
    assert report["items"][0]["benchmarks"][0]["used_for_ranking"]
    assert [item["id"] for item in report["items"]][1:] == [r.id for r in guided.RECIPES]


def test_unsure_unknown_fallback_uses_cost_instead_of_parameter_count():
    large = recipe("large", 125_000_000_000, weights=2 * guided.GIB)
    small = recipe("small", 1)
    report = guided.recommendation_report(host(), {}, candidates=[large, small], benchmarks=registry())
    assert [item["id"] for item in report["items"]] == ["small", "large"]


def test_unsure_checked_native_fallback_precedes_unchecked_source():
    flint = next(candidate for candidate in guided.native_recipes() if candidate.native_choice == "flint-int4")
    unknown = recipe("unknown", 125_000_000_000)
    machine = {**host(), "gpu_available": True, "unified_memory": True, "gpu_memory_budget_bytes": 115_448_725_504}
    report = guided.recommendation_report(machine, {}, candidates=[unknown, flint], benchmarks=registry())
    assert [item["id"] for item in report["items"]] == [flint.id, unknown.id]


@pytest.mark.parametrize("preference, expected", [("speed", "small"), ("answers", "large")])
def test_unsure_preserves_explicit_resource_preferences(preference, expected):
    large, small = recipe("large", 100, weights=2 * guided.GIB), recipe("small", 1)
    scores = registry(cohort(benchmark="ifeval"))
    report = guided.recommendation_report(
        host(), {}, candidates=[large, small], preference=preference, benchmarks=scores
    )
    assert report["items"][0]["id"] == expected
    assert not report["benchmark_ranking"]["applied"]


def test_unsure_evidence_never_bypasses_exclusion_or_resource_admission():
    scored, fallback = recipe("small", 1), guided.RECIPES[0]
    scores = registry(cohort(benchmark="ifeval", scores={scored.repo: "99"}))
    for candidates, excluded in [
        ([scored, fallback], [scored.id]),
        ([replace(scored, estimated_weight_bytes=256 * guided.GIB), fallback], []),
    ]:
        report = guided.recommendation_report(host(), {}, excluded, candidates=candidates, benchmarks=scores)
        assert [item["id"] for item in report["items"]] == [fallback.id]
        assert not report["benchmark_ranking"]["applied"]


def test_one_coding_reference_including_zero_is_enough_to_prioritize_evidence():
    unknown, scored = recipe("unknown", 50), recipe("small", 1)
    scores = registry(cohort(scores={"example/small": "0"}))
    report = guided.recommendation_report(host(), {}, candidates=[unknown, scored], task="code", benchmarks=scores)
    assert [item["id"] for item in report["items"]] == ["small", "unknown"]
    assert report["items"][0]["benchmarks"][0]["score"] == "0"
    assert report["items"][0]["benchmarks"][0]["used_for_ranking"]
    assert "Used to prioritize coding recommendations" in report["items"][0]["benchmark_details"][0]


def test_coding_scores_override_soft_fit_tiers_after_hard_admission():
    large, small = recipe("large", 10), recipe("small", 1)
    scores = registry(cohort(scores={"example/small": "90"}))
    ranked, selected, used = scores.rank([large, small], "code", "balanced", lambda candidate: candidate.id == "large")
    assert ranked == [small, large] and selected is not None and used == {small.id}


def test_all_unknown_coding_models_use_explicit_resource_fallback():
    large, small = recipe("large", 10), recipe("small", 1)
    report = guided.recommendation_report(host(), {}, candidates=[small, large], task="code", benchmarks=registry())
    assert [item["id"] for item in report["items"]] == ["large", "small"]
    assert not report["benchmark_ranking"]["applied"]
    assert "No eligible coding benchmark references" in report["message"]
    assert "resource-based fallback; coding quality is unknown" in report["items"][0]["recommendation_reason"]


def test_resource_admission_precedes_scores_and_lighter_preference_is_preserved():
    large, small = recipe("large", 10, weights=2 * guided.GIB), recipe("small", 1)
    scores = registry(cohort(scores={"example/large": "100", "example/small": "0"}))
    report = guided.recommendation_report(
        host(), {}, candidates=[large, small], task="code", preference="speed", benchmarks=scores
    )
    assert report["items"][0]["id"] == "small"
    assert not report["benchmark_ranking"]["applied"]
    opposite = registry(cohort())
    report = guided.recommendation_report(
        host(), {}, candidates=[large, small], task="code", preference="answers", benchmarks=opposite
    )
    assert report["items"][0]["id"] == "large"
    assert not report["benchmark_ranking"]["applied"]
    blocked = replace(large, estimated_weight_bytes=256 * guided.GIB)
    report = guided.recommendation_report(host(), {}, candidates=[blocked, small], task="code", benchmarks=scores)
    assert [item["id"] for item in report["items"]] == ["small"]
    assert report["benchmark_ranking"]["applied"]
    assert report["items"][0]["benchmarks"][0]["score"] == "0"


def test_cohorts_are_not_combined_or_cherry_picked_and_ties_preserve_fallback():
    large, small = recipe("large", 10), recipe("small", 1)
    scores = registry(
        cohort("b", scores={"example/large": "0", "example/small": "100"}),
        cohort("a", scores={"example/large": "60", "example/small": "60"}),
    )
    for candidates in ([small, large], [large, small]):
        report = guided.recommendation_report(host(), {}, candidates=candidates, task="code", benchmarks=scores)
        assert report["benchmark_ranking"]["cohort"] == "a"
        assert [item["id"] for item in report["items"]] == ["large", "small"]
    separate = registry(cohort("a", scores={"example/large": "0"}), cohort("b", scores={"example/small": "100"}))
    report = guided.recommendation_report(host(), {}, candidates=[small, large], task="code", benchmarks=separate)
    assert report["benchmark_ranking"]["cohort"] == "a"
    assert [item["id"] for item in report["items"]] == ["large", "small"]
    assert not report["items"][1]["benchmarks"][0]["used_for_ranking"]


def test_comparison_cannot_cross_resource_tiers_or_count_duplicate_parent_as_coverage():
    large, small = recipe("large", 10), recipe("small", 1)
    scores = registry(cohort(benchmark="bfcl"))
    ranked, selected, used = scores.rank(
        [large, small], "files", "balanced", lambda candidate: candidate.id == "large"
    )
    assert ranked == [large, small] and selected is None and not used
    duplicate = replace(large, id="duplicate")
    ranked, selected, used = scores.rank([large, duplicate], "files", "balanced", lambda candidate: 1)
    assert selected is None and not used


def test_cohort_selection_counts_only_models_within_comparable_memory_tiers():
    large, unknown, small = recipe("large", 10), recipe("unknown", 5), recipe("small", 1)
    scores = registry(
        cohort("a", benchmark="bfcl", scores={"example/large": "10", "example/small": "90"}),
        cohort("b", benchmark="bfcl", scores={"example/large": "10", "example/unknown": "90"}),
    )
    ranked, selected, used = scores.rank(
        [large, unknown, small], "files", "balanced", lambda candidate: candidate.id != "small"
    )
    assert selected.id == "b"
    assert ranked == [unknown, large, small] and used == {large.id, unknown.id}


def test_coding_cohort_coverage_counts_parents_not_quantized_aliases():
    large, small = recipe("large", 10), recipe("small", 1)
    duplicate = replace(large, id="duplicate")
    other = recipe("other", 5)
    scores = registry(
        cohort("a", scores={"example/large": "99"}),
        cohort("b", scores={"example/small": "80", "example/other": "40"}),
    )
    ranked, selected, used = scores.rank([large, duplicate, other, small], "code", "balanced", lambda candidate: 1)
    assert selected.id == "b" and used == {small.id, other.id}
    assert ranked == [small, other, large, duplicate]


def test_native_coding_on_128gb_mac_recommends_fresh_profiled_qwen_references_over_flint():
    native = guided.native_recipes()
    qwen = [recipe for recipe in native if recipe.native_choice in {"qwen38-lily-q4-xl", "qwen38-mlx-serve-mixed"}]
    machine = {
        **host(),
        "gpu_available": True,
        "unified_memory": True,
        "gpu_memory_budget_bytes": 115_448_725_504,
        "free_disk_bytes": 256 * guided.GIB,
    }
    installed = {recipe.id: ["exact-source-installed-" + recipe.native_choice] for recipe in qwen}
    assert all(recipe.resident_profile is not None for recipe in qwen)
    assert all(
        recipe.resident_profile == source_profile(recipe.native_choice, recipe.native_profile) for recipe in qwen
    )
    # Beginner can recommend a fresh pinned source at 100 GiB when the
    # resident profile is source-bound and its setup space is available.
    report = guided.recommendation_report(machine, {}, candidates=native, task="code")
    assert [item["repo"] for item in report["items"]] == [
        "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
        "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
        "RecursiveSI/flint-int4",
    ]
    for item in report["items"][:2]:
        assert not item["installed_ids"]
        assert item["benchmarks"][0]["score"] == "91.9" and item["benchmarks"][0]["used_for_ranking"]
    assert report["benchmark_ranking"]["cohort"] == "qwen38-card-livecodebench-v6"
    assert report["items"][2]["benchmarks"] == []
    machine["gpu_memory_budget_bytes"] = 80 * guided.GIB
    blocked = guided.recommendation_report(machine, installed, candidates=native, task="code")
    assert [item["repo"] for item in blocked["items"]] == ["RecursiveSI/flint-int4"]
    assert not blocked["benchmark_ranking"]["applied"]


def test_quantized_reference_bindings_require_exact_reviewed_source_revision():
    source = cohort()["source"]
    binding = {
        "repo": "export/quantized",
        "revision": "b" * 40,
        "model_id": "example/small",
        "source": source,
        "relationship": "reviewed_quantized_export_reference_only",
    }
    scores = registry(cohort(), bindings=[binding])
    quant = replace(recipe("small", 1), repo=binding["repo"], revision=binding["revision"])
    assert scores.model_id(quant) == "example/small"
    assert scores.model_id(replace(quant, revision="c" * 40)) == "export/quantized"
    assert scores.model_id(replace(quant, repo="export/finetune")) == "export/finetune"


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-1", "101", True, 90, "not-a-score"])
def test_invalid_scores_are_rejected(score):
    value = cohort()
    value["results"][0]["score"] = score
    with pytest.raises(ValueError):
        registry(value)


def test_invalid_optional_snapshot_falls_back_without_disabling_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "module.py"))
    (tmp_path / "recommendation_benchmarks.json").write_text('{"schema_version":1,"cohorts":[{}]}')
    module.load_registry.cache_clear()
    try:
        assert module.load_registry().cohorts == []
    finally:
        module.load_registry.cache_clear()


def test_task_mapping_separates_instruction_tool_and_coding_scores():
    candidates = [recipe("large", 10), recipe("small", 1)]
    scores = registry(cohort("coding"), cohort("tools", benchmark="bfcl"), cohort("instructions", benchmark="ifeval"))
    for task, expected in [
        ("code", "coding"),
        ("files", "tools"),
        ("text", "instructions"),
        ("everyday", "instructions"),
        ("unsure", "instructions"),
    ]:
        report = guided.recommendation_report(host(), {}, candidates=candidates, task=task, benchmarks=scores)
        assert report["benchmark_ranking"]["cohort"] == expected


def test_optional_evidence_respects_swift_display_bounds_and_keeps_selected_cohort():
    scores = registry(*(cohort(f"cohort-{index}", benchmark="bfcl") for index in range(20)))
    selected = scores.cohorts[-1]
    evidence, details = scores.evidence(recipe("small", 1), "files", selected, {"small"})
    assert len(details) == 16 and len(evidence) == 7
    assert evidence[0]["cohort"] == selected.id and evidence[0]["used_for_ranking"]
    assert all(len(value.encode()) <= 2_000 for value in details)
    invalid = cohort()
    invalid["label"] = "\u00e9" * 100
    with pytest.raises(ValueError):
        registry(invalid)


def test_packaged_scores_keep_sources_variants_and_reference_limits():
    scores = module.load_registry()
    assert {value.benchmark for value in scores.cohorts} == module.BENCHMARKS
    assert len(scores.bindings) == 4
    bfcl = next(value for value in scores.cohorts if value.id == "bfcl-v4-2026-04-12-fc")
    assert bfcl.results["Qwen/Qwen3-8B"][0] == module.Decimal("42.57")
    assert "Qwen/Qwen3-4B" not in bfcl.results
    instructions = next(value for value in scores.cohorts if value.benchmark == "ifeval")
    assert instructions.results["Qwen/Qwen2.5-14B-Instruct"][1] == "f55224c616ca27d4bcf28969a156de12c98981cf"
    assert "Qwen/Qwen2.5-Coder-14B-Instruct" not in instructions.results
    coding = next(value for value in scores.cohorts if value.benchmark == "livecodebench")
    assert coding.authority == "model_developer" and "undisclosed" in coding.settings
    native = next(value for value in guided.native_recipes() if value.native_choice == "qwen38-lily-q4-xl")
    evidence, details = scores.evidence(native, "code", None, set())
    assert evidence[0]["model_id"] == "Qwen/Qwen3.8-Flash-Next"
    assert evidence[0]["score"] == "91.9" and evidence[0]["evidence_kind"] == "upstream_reference"
    assert any("were not benchmarked" in detail for detail in details)
    assert "maxxbench" not in str(deepcopy(scores.cohorts)).lower()


def test_published_bfcl_can_promote_8b_over_14b_capacity_fallback():
    # Synthetic resource plans isolate ranking; no model execution or fit claim.
    large = replace(recipe("14b", 14_000_000_000), repo="Qwen/Qwen3-14B")
    small = replace(recipe("8b", 8_000_000_000), repo="Qwen/Qwen3-8B")
    baseline = guided.recommendation_report(host(), {}, candidates=[small, large], task="files", benchmarks=registry())
    assert baseline["items"][0]["id"] == "14b"
    measured = guided.recommendation_report(host(), {}, candidates=[large, small], task="files")
    assert [value["id"] for value in measured["items"]] == ["8b", "14b"]
    assert measured["benchmark_ranking"]["cohort"] == "bfcl-v4-2026-04-12-fc"
    assert measured["items"][0]["benchmarks"][0]["score"] == "42.57"
