"""Bounded guided model discovery and local hardware envelope regressions."""

import io
import json
import threading
import time

import pytest

from stark_convert import guided_discovery
from stark_convert import guided_models as guided
from stark_convert.desktop import Desktop, DesktopError

GIB = guided.GIB


def hardware(**changes):
    return {
        "gpu_available": True,
        "gpu_name": "Apple GPU",
        "unified_memory": True,
        "gpu_memory_budget_bytes": 12 * GIB,
        "cpu_cores": 10,
        **changes,
    }


def resources(**changes):
    return {
        "supported": True,
        "physical_memory_bytes": 32 * GIB,
        "available_memory_bytes": 24 * GIB,
        "free_disk_bytes": 100 * GIB,
        "download_free_disk_bytes": 100 * GIB,
        **changes,
    }


def catalog_item(*, identifier="a" * 32, revision="b" * 40, repo="example/model", weight=2 * GIB):
    return {
        "id": identifier,
        "name": "Example model",
        "repo": repo,
        "revision": revision,
        "weight_bytes": weight,
        "parameter_count": 1_000_000_000,
        "quantization": "BF16",
        "artifact_modes": ["text"],
        "downloadable": True,
        "planning": {
            "status": "available",
            "architecture": "llama",
            "context_limit": 32768,
            "attention_layers": 8,
            "kv_heads": 2,
            "query_heads": 8,
            "head_dim": 64,
            "hidden_size": 1024,
            "download_bytes": weight + 32 * 1024**2,
            "task_tags": [],
            "chat_template_available": True,
            "supports_tools": True,
            "tool_parser": "qwen3-hermes-v1",
        },
    }


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO(), timeout=2)
    monkeypatch.setattr("stark_convert.desktop.host_resources", lambda _path: resources())
    yield value
    value.stop(quiet=True)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"gpu_available": True},
        hardware(gpu_available=1),
        hardware(unified_memory="yes"),
        hardware(gpu_memory_budget_bytes=True),
        hardware(gpu_memory_budget_bytes=-1),
        hardware(gpu_memory_budget_bytes=2**63),
        hardware(cpu_cores=0),
        hardware(cpu_cores=4097),
        hardware(gpu_name=""),
        hardware(gpu_name="bad\nname"),
    ],
)
def test_hardware_envelope_rejects_malformed_values(desktop, bad):
    with pytest.raises(DesktopError):
        desktop.guided_device(bad)


def test_hardware_budget_is_clamped_to_independently_probed_physical_memory(desktop):
    report = hardware(gpu_memory_budget_bytes=48 * GIB)
    assert desktop.guided_device(report) == report
    constrained = desktop.guided_resources(report)
    assert constrained["gpu_memory_budget_bytes"] == 32 * GIB
    assert constrained["physical_memory_bytes"] == 32 * GIB
    no_host_memory = desktop.guided_resources(report)
    assert no_host_memory["gpu_memory_budget_bytes"] == min(
        report["gpu_memory_budget_bytes"], no_host_memory["physical_memory_bytes"]
    )


def test_discover_uses_capped_concurrent_family_windows_and_hides_exceptions(monkeypatch):
    families = [{"family_id": f"family-{index}", "modes": {"text": {"status": "conditional"}}} for index in range(7)]
    monkeypatch.setattr(guided_discovery, "support_matrix", lambda: {"families": families})
    lock = threading.Lock()
    active = peak = 0
    calls = []

    def catalog(family, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((family, kwargs))
        try:
            if family == "family-2":
                raise RuntimeError("private path /secret/token=do-not-leak")
            time.sleep(0.01)
            return {
                "status": "ready",
                "has_more": True,
                "items": [{"id": "duplicate", "family": family}, {"id": family}],
            }
        finally:
            with lock:
                active -= 1

    result = guided_discovery.discover(catalog=catalog)
    assert peak <= 3
    assert len(calls) == len(families)
    assert all(kwargs == {"planning": True, "deadline_seconds": 10, "limit": 30} for _, kwargs in calls)
    assert result["bounded_scan"] is True and result["incomplete"] is True
    assert result["families_checked"] == len(families) - 1
    assert result["failed_families"] == 1
    assert len(result["items"]) == len(families)
    assert "secret" not in repr(result) and "do-not-leak" not in repr(result)


def test_catalog_recipe_binds_dynamic_revision_and_rejects_forged_or_unplannable_candidates():
    item = catalog_item(revision="c" * 40, repo="some-org/custom-model")
    recipe = guided.recipe_from_catalog(item)
    assert recipe is not None
    assert recipe.repo == item["repo"] and recipe.revision == item["revision"]
    assert recipe.catalog_item["revision"] == "c" * 40
    assert guided.recipe_from_catalog({**item, "revision": "forged"}) is None
    assert guided.recipe_from_catalog({**item, "downloadable": False}) is None
    assert guided.recipe_from_catalog({**item, "planning": {**item["planning"], "status": "unavailable"}}) is None


def test_dynamic_install_uses_bound_revision_and_rejects_forged_offer(desktop, monkeypatch):
    item = catalog_item(revision="d" * 40, repo="another-org/private-name")
    desktop.publish_guided_models(
        "everyday",
        "balanced",
        hardware(),
        {"excluded_model_ids": set(), "excluded_recipe_ids": set()},
        "offer",
        {"items": [item], "incomplete": False},
    )
    bound = desktop.guided_reports["offer"][item["id"]]
    assert bound.revision == item["revision"]
    assert bound.repo == item["repo"]

    def install(*args, **kwargs):
        assert args[8] == item["revision"]
        assert args[9] == item["id"]

    monkeypatch.setattr(desktop, "start", install)
    desktop.install_guided(
        {
            "command": "install_guided",
            "recommendation_request_id": "offer",
            "recipe_id": item["id"],
            "support_request_id": "install",
        }
    )
    with pytest.raises(DesktopError):
        desktop.guided_recipe({"recommendation_request_id": "offer", "recipe_id": "e" * 32})


def test_background_discovery_adds_candidate_and_stale_refresh_cannot_publish(desktop, monkeypatch):
    candidate = catalog_item()
    started, release = threading.Event(), threading.Event()

    def discover(_generation):
        started.set()
        release.wait(2)
        return {"items": [candidate], "incomplete": False}

    monkeypatch.setattr(desktop, "run_guided_discovery", discover)
    desktop.guided_models(
        {"command": "guided_models", "request_id": "first", "task": "everyday", "hardware": hardware()}
    )
    assert started.wait(1)
    # A newer request invalidates the first worker before its catalog result lands.
    desktop.guided_models(
        {"command": "guided_models", "request_id": "second", "task": "everyday", "hardware": hardware()}
    )
    release.set()
    deadline = time.monotonic() + 3
    while (
        desktop.guided_discovery_job is not None
        and desktop.guided_discovery_job.is_alive()
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    events = [json.loads(line) for line in desktop.output.getvalue().splitlines()]
    offers = [event for event in events if event.get("event") == "guided_models"]
    assert offers
    assert all(event["request_id"] != "first" for event in offers)
    assert offers[-1]["request_id"] == "second"
    ids = {item["id"] for item in offers[-1]["report"]["items"]}
    assert candidate["id"] in ids
    assert offers[-1]["report"]["catalog_search"]["bounded"] is True


def test_failed_partial_discovery_is_retried_instead_of_cached(desktop, monkeypatch):
    calls, published = [], []
    partial = {"items": [catalog_item()], "incomplete": True, "failed_families": 1}
    monkeypatch.setattr(desktop, "run_guided_discovery", lambda generation: calls.append(generation) or partial)
    monkeypatch.setattr(desktop, "publish_guided_models", lambda *args: published.append(args[4]))
    generation = desktop.guided_discovery_generation
    args = ("everyday", "balanced", hardware(), {"excluded_model_ids": set(), "excluded_recipe_ids": set()})
    desktop.load_guided_models(*args, "first", generation)
    desktop.load_guided_models(*args, "second", generation)
    assert calls == [generation, generation]
    assert published == ["first", "second"]
    assert desktop.guided_catalog_cache is None


def test_normal_bounded_discovery_without_failed_families_is_cached(desktop, monkeypatch):
    calls, published = [], []
    bounded = {"items": [catalog_item()], "incomplete": True, "failed_families": 0}

    def discover(generation):
        calls.append(generation)
        return bounded

    monkeypatch.setattr(desktop, "run_guided_discovery", discover)
    monkeypatch.setattr(desktop, "publish_guided_models", lambda *args: published.append(args[4]))
    generation = desktop.guided_discovery_generation
    args = ("everyday", "balanced", hardware(), {"excluded_model_ids": set(), "excluded_recipe_ids": set()})
    desktop.load_guided_models(*args, "first", generation)
    desktop.load_guided_models(*args, "second", generation)
    assert calls == [generation]
    assert published == ["first", "second"]
    assert desktop.guided_catalog_cache[1] is bounded


def test_stale_failed_discovery_does_not_publish_fallback(desktop, monkeypatch):
    published = []

    def stale_failure(_generation):
        desktop.cancel_guided_discovery()
        return {"items": [catalog_item()], "incomplete": True, "failed_families": 1}

    monkeypatch.setattr(desktop, "run_guided_discovery", stale_failure)
    monkeypatch.setattr(desktop, "publish_guided_models", lambda *args: published.append(args))
    generation = desktop.guided_discovery_generation
    desktop.load_guided_models(
        "everyday",
        "balanced",
        hardware(),
        {"excluded_model_ids": set(), "excluded_recipe_ids": set()},
        "stale",
        generation,
    )
    assert published == []
    assert desktop.guided_catalog_cache is None
