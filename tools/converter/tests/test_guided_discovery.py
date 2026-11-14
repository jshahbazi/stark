"""Bounded guided model discovery and local hardware envelope regressions."""

import io
import json
import os
import threading
import time

import pytest

from stark_convert import desktop as desktop_module
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
                "items": [
                    {"id": "duplicate", "family": family, "downloadable": True},
                    {"id": family, "downloadable": True},
                ],
            }
        finally:
            with lock:
                active -= 1

    result = guided_discovery.discover(catalog=catalog)
    assert peak <= 3
    assert len(calls) == len(families)
    assert all(
        kwargs == {"planning": True, "deadline_seconds": 10, "limit": 30, "ranking_task": "general"}
        for _, kwargs in calls
    )
    assert result["bounded_scan"] is True and result["incomplete"] is True
    assert result["families_checked"] == len(families) - 1
    assert result["failed_families"] == 1
    assert len(result["items"]) == len(families)
    assert "secret" not in repr(result) and "do-not-leak" not in repr(result)


def test_discover_reaches_downloadable_model_after_unavailable_page(monkeypatch):
    monkeypatch.setattr(
        guided_discovery,
        "support_matrix",
        lambda: {"families": [{"family_id": "llama", "modes": {"text": {"status": "conditional"}}}]},
    )
    ticks = iter([100, 106])
    monkeypatch.setattr(guided_discovery.time, "monotonic", lambda: next(ticks))
    calls = []
    unavailable = {**catalog_item(identifier="c" * 32), "downloadable": False}
    available = catalog_item()

    def catalog(family, **options):
        calls.append((family, options))
        if "offset" not in options:
            return {"status": "ready", "has_more": True, "next_offset": 30, "items": [unavailable]}
        return {"status": "ready", "has_more": True, "next_offset": 60, "items": [available]}

    result = guided_discovery.discover(catalog=catalog, task="coding")

    assert calls == [
        ("llama", {"planning": True, "deadline_seconds": 10, "limit": 30, "ranking_task": "coding"}),
        (
            "llama",
            {"planning": True, "deadline_seconds": 4, "limit": 30, "ranking_task": "coding", "offset": 30},
        ),
    ]
    assert result["items"] == [available]
    assert guided.recipe_from_catalog(result["items"][0]) is not None
    assert result["incomplete"] is True and result["failed_families"] == 0


def test_family_discovery_stops_at_shared_deadline(monkeypatch):
    ticks = iter([100, 111])
    monkeypatch.setattr(guided_discovery.time, "monotonic", lambda: next(ticks))
    calls = []

    def catalog(family, **options):
        calls.append(options)
        return {"status": "ready", "has_more": True, "next_offset": 30, "items": []}

    report = guided_discovery._discover_family(catalog, "llama")

    assert len(calls) == 1
    assert report["has_more"] is True


@pytest.mark.parametrize("next_offset", [None, False, 0, -1, "30", 10_000_001])
def test_family_discovery_stops_at_invalid_frontier(next_offset):
    calls = []

    def catalog(family, **options):
        calls.append(options)
        return {"status": "ready", "has_more": True, "next_offset": next_offset, "items": []}

    report = guided_discovery._discover_family(catalog, "llama")

    assert len(calls) == 1
    assert report["has_more"] is True


def test_family_discovery_retains_rows_when_later_page_fails():
    unavailable = {**catalog_item(), "downloadable": False}

    def catalog(family, **options):
        if "offset" in options:
            raise RuntimeError("private exception sentinel")
        return {"status": "ready", "has_more": True, "next_offset": 30, "items": [unavailable]}

    report = guided_discovery._discover_family(catalog, "llama")

    assert report == {"status": "error", "has_more": True, "items": []}
    assert "sentinel" not in repr(report)


def test_discovery_item_and_timeout_bounds_cover_all_conditional_family_waves(monkeypatch):
    families = [{"family_id": f"family-{index}", "modes": {"text": {"status": "conditional"}}} for index in range(19)]
    monkeypatch.setattr(guided_discovery, "support_matrix", lambda: {"families": families})

    maximum_items, timeout_seconds = guided_discovery.discovery_bounds()

    assert maximum_items == 19 * 30
    assert timeout_seconds == 7 * guided_discovery.FAMILY_SECONDS + 10


def test_discovery_accepts_thirty_downloadable_rows_from_all_nineteen_families(monkeypatch):
    families = [{"family_id": f"family-{index}", "modes": {"text": {"status": "conditional"}}} for index in range(19)]
    monkeypatch.setattr(guided_discovery, "support_matrix", lambda: {"families": families})

    def catalog(family, **_options):
        family_index = int(family.removeprefix("family-"))
        return {
            "status": "ready",
            "has_more": True,
            "next_offset": 30,
            "items": [
                catalog_item(
                    identifier=f"{family_index * 30 + item_index:032x}",
                    repo=f"example/model-{family_index}-{item_index}",
                )
                for item_index in range(30)
            ],
        }

    report = guided_discovery.discover(catalog=catalog)

    maximum_items, _timeout_seconds = guided_discovery.discovery_bounds()
    assert report["families_checked"] == 19
    assert len(report["items"]) == maximum_items == 570
    assert report["failed_families"] == 0
    assert report["incomplete"] is True


def test_unavailable_pages_do_not_fill_the_cross_family_payload(monkeypatch):
    families = [{"family_id": f"family-{index}", "modes": {"text": {"status": "conditional"}}} for index in range(19)]
    monkeypatch.setattr(guided_discovery, "support_matrix", lambda: {"families": families})
    calls = []

    def catalog(family, *, offset=0, **_options):
        calls.append((family, offset))
        family_index = int(family.removeprefix("family-"))
        return {
            "status": "ready",
            "has_more": True,
            "next_offset": offset + 30,
            "items": [
                {
                    **catalog_item(
                        identifier=f"{family_index * 300 + offset + item_index:032x}",
                        repo=f"example/model-{family_index}-{offset + item_index}",
                    ),
                    "downloadable": False,
                }
                for item_index in range(30)
            ],
        }

    report = guided_discovery.discover(catalog=catalog)

    assert len(calls) == 19 * guided_discovery.MAX_FAMILY_PAGES
    assert report["items"] == []
    assert report["families_checked"] == 19
    assert report["failed_families"] == 0
    assert report["incomplete"] is True
    assert len(json.dumps(report).encode()) < 2048


def test_family_discovery_caps_metadata_pages_even_with_immediate_responses(monkeypatch):
    monkeypatch.setattr(guided_discovery.time, "monotonic", lambda: 100)
    calls = []

    def catalog(family, **options):
        calls.append(options)
        return {"status": "ready", "has_more": True, "next_offset": 30 * len(calls), "items": []}

    report = guided_discovery._discover_family(catalog, "llama")

    assert len(calls) == guided_discovery.MAX_FAMILY_PAGES
    assert report["has_more"] is True


def test_catalog_recipe_binds_dynamic_revision_and_rejects_forged_or_unplannable_candidates():
    item = catalog_item(revision="c" * 40, repo="some-org/custom-model")
    recipe = guided.recipe_from_catalog(item)
    assert recipe is not None
    assert recipe.repo == item["repo"] and recipe.revision == item["revision"]
    assert recipe.catalog_item["revision"] == "c" * 40
    assert guided.recipe_from_catalog({**item, "revision": "forged"}) is None
    assert guided.recipe_from_catalog({**item, "downloadable": False}) is None
    assert guided.recipe_from_catalog({**item, "planning": {**item["planning"], "status": "unavailable"}}) is None


@pytest.mark.parametrize(
    "downloadable,partial_failure",
    [(True, False), (True, True), (False, True)],
)
def test_discovery_keeps_partial_failure_state_without_unavailable_rows(monkeypatch, downloadable, partial_failure):
    monkeypatch.setattr(
        guided_discovery,
        "support_matrix",
        lambda: {"families": [{"family_id": "qwen3", "modes": {"text": {"status": "conditional"}}}]},
    )
    candidate = {**catalog_item(), "downloadable": downloadable}

    def partial_catalog(_family, **_kwargs):
        return {
            "status": "ready",
            "items": [candidate],
            "has_more": True,
            "incomplete": True,
            "partial_failure": partial_failure,
        }

    report = guided_discovery.discover(catalog=partial_catalog)
    assert report["items"] == ([candidate] if downloadable else [])
    assert report["incomplete"] is True
    # Normal bounded windows may be cached. Failed metadata within that window
    # must set failed_families so the desktop's next Check again retries it.
    assert report["failed_families"] == int(partial_failure)


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

    def discover(_generation, _task):
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
    monkeypatch.setattr(
        desktop, "run_guided_discovery", lambda generation, task: calls.append((generation, task)) or partial
    )
    monkeypatch.setattr(desktop, "publish_guided_models", lambda *args: published.append(args[4]))
    generation = desktop.guided_discovery_generation
    args = ("everyday", "balanced", hardware(), {"excluded_model_ids": set(), "excluded_recipe_ids": set()})
    desktop.load_guided_models(*args, "first", generation)
    desktop.load_guided_models(*args, "second", generation)
    assert calls == [(generation, "general"), (generation, "general")]
    assert published == ["first", "second"]
    assert desktop.guided_catalog_cache == {}


def test_normal_bounded_discovery_without_failed_families_is_cached(desktop, monkeypatch):
    calls, published = [], []
    bounded = {"items": [catalog_item()], "incomplete": True, "failed_families": 0}

    def discover(generation, task):
        calls.append((generation, task))
        return bounded

    monkeypatch.setattr(desktop, "run_guided_discovery", discover)
    monkeypatch.setattr(desktop, "publish_guided_models", lambda *args: published.append(args[4]))
    generation = desktop.guided_discovery_generation
    args = ("everyday", "balanced", hardware(), {"excluded_model_ids": set(), "excluded_recipe_ids": set()})
    desktop.load_guided_models(*args, "first", generation)
    desktop.load_guided_models(*args, "second", generation)
    assert calls == [(generation, "general")]
    assert published == ["first", "second"]
    assert desktop.guided_catalog_cache["general"][1] is bounded


def test_discovery_subprocess_receives_task_argument(desktop, monkeypatch):
    calls = []
    wait_timeouts = []
    items = [{"id": f"{index:032x}"} for index in range(570)]

    class Child:
        pid = 1

        def __init__(self, descriptor, writer):
            self.stdout = os.fdopen(descriptor, "rb", buffering=0)
            self.writer = writer

        def wait(self, timeout=None):
            wait_timeouts.append(timeout)
            self.writer.join(timeout)
            assert not self.writer.is_alive()
            return 0

        def poll(self):
            return 0

    def popen(command, **_kwargs):
        calls.append(command)
        read_fd, write_fd = os.pipe()

        def send():
            with os.fdopen(write_fd, "wb") as output:
                output.write(json.dumps({"schema_version": 1, "items": items}).encode())

        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        return Child(read_fd, writer)

    monkeypatch.setattr(desktop_module.subprocess, "Popen", popen)
    monkeypatch.setattr(desktop_module.time, "monotonic", lambda: 100)
    result = desktop.run_guided_discovery(desktop.guided_discovery_generation, "coding")

    assert calls[0][-3:] == ["stark_convert.guided_discovery", "--task", "coding"]
    assert result == {"schema_version": 1, "items": items}
    assert wait_timeouts == [guided_discovery.discovery_bounds()[1]]


def test_discovery_cache_is_separate_for_general_and_coding_tasks(desktop, monkeypatch):
    calls = []
    reports = {
        task: {"items": [catalog_item(identifier=task[0] * 32)], "failed_families": 0}
        for task in ("general", "coding")
    }
    monkeypatch.setattr(desktop, "run_guided_discovery", lambda generation, task: calls.append(task) or reports[task])
    monkeypatch.setattr(desktop, "publish_guided_models", lambda *args: None)
    generation = desktop.guided_discovery_generation
    excluded = {"excluded_model_ids": set(), "excluded_recipe_ids": set()}

    for task in ("general", "coding", "general", "coding"):
        desktop.load_guided_models(task, "quality", hardware(), excluded, task, generation)

    assert calls == ["general", "coding"]
    assert set(desktop.guided_catalog_cache) == {"general", "coding"}


def test_stale_failed_discovery_does_not_publish_fallback(desktop, monkeypatch):
    published = []

    def stale_failure(_generation):
        desktop.cancel_guided_discovery()
        return {"items": [catalog_item()], "incomplete": True, "failed_families": 1}

    monkeypatch.setattr(desktop, "run_guided_discovery", lambda generation, _task: stale_failure(generation))
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
    assert desktop.guided_catalog_cache == {}


def test_installed_model_checks_do_not_hold_generation_lock_or_publish_after_cancellation(desktop, monkeypatch):
    started, release = threading.Event(), threading.Event()
    desktop.models = [{"id": "installed", "name": "Fixture", "path": "/synthetic/unused", "source": "fixture"}]

    def slow_validity(_path):
        started.set()
        assert release.wait(2)
        return False

    monkeypatch.setattr("stark_convert.desktop.model_valid", slow_validity)
    generation = desktop.guided_discovery_generation
    thread = threading.Thread(
        target=desktop.publish_guided_models,
        args=(
            "everyday",
            "balanced",
            hardware(),
            {"excluded_model_ids": set(), "excluded_recipe_ids": set()},
            "offer",
            None,
            generation,
        ),
    )
    thread.start()
    try:
        assert started.wait(1)
        acquired = desktop.lock.acquire(timeout=0.2)
        if acquired:
            desktop.lock.release()
        assert acquired, "Template/header inspection must not stall generation or Stop."
        desktop.cancel_guided_discovery()
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert "offer" not in desktop.guided_reports
    assert desktop.output.getvalue() == ""
