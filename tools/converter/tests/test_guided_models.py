"""Beginner recipes, source binding, scoped cancellation, and context policy."""

import hashlib
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from stark_convert import desktop as module
from stark_convert import guided_models as guided
from stark_convert.desktop import Desktop


def resources(**changes):
    return {
        "supported": True,
        "physical_memory_bytes": 16 * guided.GIB,
        "available_memory_bytes": 12 * guided.GIB,
        "free_disk_bytes": 32 * guided.GIB,
        "download_free_disk_bytes": 32 * guided.GIB,
        **changes,
    }


def artifact(path, recipe):
    """Synthetic output plus a source receipt; never real model payloads."""
    path.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"type": "LanguageModelConfig", "decoder_config": {}}).encode()
    (path / "config.json").write_bytes(config)
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic output")
    sources = [{"repo": recipe.repo, "revision": recipe.revision}]
    files = [
        {"repo": recipe.repo, "revision": recipe.revision, "size": size, "sha256": digest}
        for _name, size, digest in recipe.files
    ]
    identity = {"sources": sources, "files": files}
    if recipe.metadata_profile is not None:
        identity["conversion_profile"] = {
            "architecture": recipe.architecture,
            "metadata_profile": recipe.metadata_profile,
        }
    (path / "conversion.json").write_text(
        json.dumps(
            {
                **identity,
                "source_files": files,
                "source_manifest_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "config_sha256": hashlib.sha256(config).hexdigest(),
            }
        )
    )
    return path


class ExistingWorker:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def host(tmp_path, monkeypatch):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO(), timeout=2)
    monkeypatch.setattr(module, "host_resources", lambda _path: resources())
    monkeypatch.setattr(module, "model_valid", lambda path: (path / "config.json").is_file())
    yield value
    value.stop(quiet=True)


def events(host):
    return [json.loads(line) for line in host.output.getvalue().splitlines()]


def finish(host):
    job = host.job
    if job is not None:
        job.join(3)
        assert not job.is_alive()
    assert host.job is None


def offer(host, *, request_id="offer", **changes):
    host.dispatch({"command": "guided_models", "request_id": request_id, "task": "everyday", **changes})
    return events(host)[-1]["report"]


def install_command(recipe, *, offer_id="offer", request_id="install"):
    return {
        "command": "install_guided",
        "recommendation_request_id": offer_id,
        "recipe_id": recipe.id,
        "support_request_id": request_id,
    }


def add_model(host, recipe, identifier="installed"):
    path = artifact(host.model_dir / identifier, recipe)
    model = {"id": identifier, "name": recipe.name, "path": str(path), "source": "huggingface:" + recipe.repo}
    host.models.append(model)
    return model


def test_fixed_recipes_bind_complete_real_source_manifests():
    assert len({recipe.id for recipe in guided.RECIPES}) == 2
    for recipe in guided.RECIPES:
        assert len(recipe.revision) == 40 and int(recipe.revision, 16)
        assert recipe.download_bytes == sum(file[1] for file in recipe.files)
        assert recipe.setup_bytes > recipe.download_bytes + recipe.installed_bytes
        assert len({file[0] for file in recipe.files}) == len(recipe.files)
        assert all(len(file[2]) == 64 and int(file[2], 16) for file in recipe.files)


def test_resources_select_starting_choices_and_installed_reuse():
    normal = guided.recommendation_report(resources(), {})
    assert [item["id"] for item in normal["items"]] == [recipe.id for recipe in guided.RECIPES]
    small = guided.recommendation_report(resources(physical_memory_bytes=8 * guided.GIB), {})
    assert small["items"] == [] and "16 GB" in small["message"]
    assert [item["context_tokens"] for item in normal["items"]] == [32768, 32768]
    assert [item["runtime_memory_bytes_estimate"] for item in normal["items"]] == [10 * guided.GIB, 9 * guided.GIB]
    pressure = resources(available_memory_bytes=0, free_disk_bytes=0, download_free_disk_bytes=0)
    assert guided.recommendation_report(pressure, {})["items"] == []
    reused = guided.recommendation_report(pressure, {guided.RECIPES[0].id: ["same-copy"]})
    assert reused["items"][0]["installed_ids"] == ["same-copy"]
    assert "planning estimates" in reused["items"][0]["caveats"][0]


@pytest.mark.parametrize(
    "changes",
    [
        {"supported": False},
        {"physical_memory_bytes": None},
        {"physical_memory_bytes": True},
        {"physical_memory_bytes": 4 * guided.GIB},
        {"free_disk_bytes": 1},
        {"download_free_disk_bytes": 1},
        {"available_memory_bytes": None},
    ],
)
def test_unknown_or_insufficient_resources_fail_closed(changes):
    report = guided.recommendation_report(resources(**changes), {})
    assert report["status"] == "unavailable" and report["items"] == []


def test_exact_receipt_match_rejects_changed_config_source_and_profile(tmp_path):
    recipe = guided.RECIPES[0]
    path = artifact(tmp_path / "model", recipe)
    original = (path / "conversion.json").read_text()
    assert guided.matches_recipe(path, recipe)
    for key, value in [("conversion_profile", {}), ("sources", []), ("source_files", [])]:
        receipt = json.loads(original)
        receipt[key] = value
        (path / "conversion.json").write_text(json.dumps(receipt))
        assert not guided.matches_recipe(path, recipe)
    (path / "conversion.json").write_text(original)
    (path / "config.json").write_text("{}")
    assert not guided.matches_recipe(path, recipe)


def test_recommendation_excludes_the_rejected_exact_model_and_keeps_worker(host):
    recipe = guided.RECIPES[0]
    add_model(host, recipe)
    worker = host.worker = ExistingWorker()
    messages = host.messages = [{"role": "user", "content": "synthetic retained draft"}]
    report = offer(host, excluded_model_ids=["installed"])
    assert [item["id"] for item in report["items"]] == [guided.RECIPES[1].id]
    assert host.worker is worker and not worker.closed and host.messages is messages
    report = offer(host, excluded_recipe_ids=[item.id for item in guided.RECIPES])
    assert report["items"] == [] and report["status"] == "unavailable"


@pytest.mark.parametrize("change", [{"task": []}, {"task": "other"}, {"excluded_model_ids": "bad"}, {"extra": True}])
def test_malformed_chooser_requests_are_tagged_errors(host, change):
    host.dispatch({"command": "guided_models", "request_id": "request", "task": "everyday", **change})
    assert events(host)[-2]["event"] == "error"
    assert events(host)[-2]["request_id"] == "request"
    assert events(host)[-1]["state"] == "idle"


def test_installed_exact_recipe_is_reused_without_download(host, monkeypatch):
    recipe = guided.RECIPES[0]
    model = add_model(host, recipe)
    report = offer(host)
    assert report["items"][0]["installed_ids"] == [model["id"]]
    monkeypatch.setattr(host, "start", lambda *_args, **_kwargs: pytest.fail("redownloaded an exact installed copy"))
    host.dispatch(install_command(recipe))
    installed = next(event for event in events(host) if event["event"] == "installed")
    assert installed["model"]["id"] == model["id"]
    assert installed["support_request_id"] == installed["request_id"] == "install"


@pytest.mark.parametrize("recipe", guided.RECIPES)
def test_guided_download_pins_own_metadata_and_checks_exact_receipt(host, monkeypatch, recipe):
    worker = host.worker = ExistingWorker()
    retained = host.messages = [{"role": "user", "content": "synthetic retained draft"}]
    offer(host)
    commands = []

    def preflight(command, _environment, _epoch):
        commands.append(command)
        return {"mode": "text", "status": "conditional", "source": {"repo": recipe.repo, "revision": recipe.revision}}

    def convert(command, _environment, _epoch):
        commands.append(command)
        artifact(Path(command[command.index("--output") + 1]), recipe)
        return 0

    monkeypatch.setattr(host, "run_support_check", preflight)
    monkeypatch.setattr(host, "run_converter", convert)
    host.dispatch(install_command(recipe))
    finish(host)
    assert len(commands) == 2
    assert all(command[command.index("--revision") + 1] == recipe.revision for command in commands)
    if recipe.metadata_profile is not None:
        assert all(command[command.index("--metadata-profile") + 1] == recipe.metadata_profile for command in commands)
    else:
        assert all("--metadata-profile" not in command for command in commands)
    assert host.worker is worker and not worker.closed and host.messages is retained
    assert len(host.models) == 1 and guided.matches_recipe(host.models[0]["path"], recipe)
    assert all(event.get("request_id") == "install" for event in events(host) if event["event"] == "progress")
    assert events(host)[-1]["state"] == "ready"


def test_stale_offer_and_changed_resources_do_not_stop_running_model(host, monkeypatch):
    recipe = guided.RECIPES[0]
    worker = host.worker = ExistingWorker()
    host.dispatch(install_command(recipe, offer_id="stale"))
    assert events(host)[-2]["event"] == "error"
    offer(host)
    monkeypatch.setattr(module, "host_resources", lambda _path: resources(free_disk_bytes=1))
    host.dispatch(install_command(recipe))
    assert events(host)[-2]["event"] == "error" and events(host)[-2]["request_id"] == "install"
    assert host.worker is worker and not worker.closed and host.job is None


def test_report_refresh_cannot_evict_the_current_install_binding(host):
    recipe = guided.RECIPES[0]
    add_model(host, recipe)
    offer(host)
    host.dispatch(install_command(recipe))
    for index in range(20):
        offer(host, request_id=f"refresh-{index}")
    assert recipe.id in host.guided_reports["offer"]
    assert len(host.guided_reports) == 8


def test_scoped_cancel_reaps_child_and_preserves_current_model_and_conversation(host):
    worker = host.worker = ExistingWorker()
    retained = host.messages = [{"role": "user", "content": "synthetic retained draft"}]
    host.active_model, host.active_mode, host.api_context = "old-model", "chat", 8192
    script = "import time;time.sleep(30)"

    def install(epoch, _kind):
        host.run_converter([sys.executable, "-c", script], {}, epoch)

    host.start(install, "installing", "Installing", "huggingface", request_id="install", guided_install=True)
    deadline = time.monotonic() + 2
    while host.converter is None and time.monotonic() < deadline:
        time.sleep(0.005)
    child = host.converter
    assert child is not None
    host.dispatch({"command": "cancel_install", "request_id": "wrong"})
    assert events(host)[-1]["event"] == "install_cancel_error" and child.poll() is None
    host.dispatch({"command": "cancel_install", "request_id": "install"})
    finish(host)
    assert child.poll() is not None and host.converter is None
    assert host.worker is worker and not worker.closed and host.messages is retained
    assert host.active_model == "old-model" and host.api_context == 8192 and host.epoch == 0
    assert not any(event["event"] in {"error", "installed"} for event in events(host))
    assert any(event["event"] == "install_cancelled" and event["request_id"] == "install" for event in events(host))
    assert events(host)[-1]["state"] == "ready"
    host.dispatch({"command": "cancel_install", "request_id": "install"})
    assert events(host)[-1]["event"] == "install_cancel_error"


@pytest.mark.parametrize("exception", [RuntimeError("synthetic failure"), module.DesktopError("Setup failed.")])
def test_candidate_failure_preserves_loaded_worker(host, exception):
    worker = host.worker = ExistingWorker()
    retained = host.messages = [{"role": "user", "content": "synthetic retained draft"}]

    def fail(_epoch, _kind):
        raise exception

    host.start(fail, "installing", "Installing", "huggingface", request_id="install", guided_install=True)
    finish(host)
    assert host.worker is worker and not worker.closed and host.messages is retained
    assert events(host)[-2]["event"] == "error" and events(host)[-2]["request_id"] == "install"
    assert events(host)[-1]["state"] == "ready"


@pytest.mark.parametrize("maximum,known", [(40960, True), (0xFFFFFFFF, False)])
def test_guided_context_applies_to_playground_and_api_without_leaking_to_advanced(host, monkeypatch, maximum, known):
    recipe = guided.RECIPES[0]
    model = add_model(host, recipe)
    offer(host)
    worker = host.worker = ExistingWorker()
    host.context_override = 8192

    class ReadyWorker:
        images = False

        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def read(self):
            return {"ready": True, "mode": "text-generation-v1", "max_context_tokens": maximum, "max_new_tokens": 8192}

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "Worker", ReadyWorker)
    monkeypatch.setattr(module, "load_template", lambda _path: ({}, object()))
    host.dispatch(
        {
            "command": "run_guided",
            "request_id": "start",
            "recommendation_request_id": "offer",
            "recipe_id": recipe.id,
            "model_id": model["id"],
        }
    )
    finish(host)
    assert worker.closed
    assert host.context == host.api_context == recipe.context_tokens
    assert host.context_limit == maximum
    assert host.api_models()[0]["context_length"] == (recipe.context_tokens if known else None)
    assert host.context_capacity_known is known
    ready = [event for event in events(host) if event["event"] == "ready"][-1]
    assert ready["request_id"] == "start" and ready["model_id"] == model["id"]
    assert ready["api_context_tokens"] == recipe.context_tokens and ready["max_context_tokens"] == maximum
    if known:
        host.dispatch(
            {
                "command": "configure",
                "context_tokens": 40000,
                "max_new_tokens": 32,
                "system_prompt": None,
                "thinking": False,
            }
        )
        assert host.context == 40000 and host.api_context == recipe.context_tokens
    host.dispatch({"command": "run", "model_id": model["id"]})
    finish(host)
    assert host.context_limit == host.api_context == maximum and host.context == (40000 if known else 8192)


def test_wrong_installed_model_cannot_use_a_guided_offer_or_stop_current_worker(host):
    recipe = guided.RECIPES[0]
    other = add_model(host, guided.RECIPES[1])
    offer(host)
    worker = host.worker = ExistingWorker()
    host.dispatch(
        {
            "command": "run_guided",
            "request_id": "start",
            "recommendation_request_id": "offer",
            "recipe_id": recipe.id,
            "model_id": other["id"],
        }
    )
    assert events(host)[-2]["event"] == "error" and events(host)[-2]["request_id"] == "start"
    assert host.worker is worker and not worker.closed and host.epoch == 0


@pytest.mark.parametrize("running", [False, True])
def test_rejected_guided_operations_echo_authoritative_preserved_state(host, running):
    if running:
        host.worker = ExistingWorker()
        host.active_model = "existing"
        host.status("ready", "Ready")
    host.dispatch(install_command(guided.RECIPES[0], offer_id="stale"))
    error, status = events(host)[-2:]
    assert error["event"] == "error" and error["request_id"] == "install"
    assert error["operation_started"] is False
    assert error["preserved_model_id"] == ("existing" if running else None)
    assert status["state"] == ("ready" if running else "idle")


def test_rejected_candidate_echoes_the_actual_inflight_operation(host):
    offer(host)
    entered, release = threading.Event(), threading.Event()

    def operation(epoch):
        entered.set()
        assert release.wait(2)
        host.status("idle", "Done", epoch)

    host.start(operation, "generating", "Generating", request_id="other")
    assert entered.wait(2)
    try:
        host.dispatch(install_command(guided.RECIPES[0]))
        assert events(host)[-2]["operation_started"] is False
        assert events(host)[-1]["state"] == "generating"
    finally:
        release.set()
        finish(host)


def test_source_manifest_mismatch_fails_before_publication_and_keeps_worker(host, monkeypatch):
    recipe = guided.RECIPES[0]
    worker = host.worker = ExistingWorker()
    offer(host)
    monkeypatch.setattr(host, "check_install_support", lambda *_args, **_kwargs: recipe.revision)

    def convert(command, _environment, _epoch):
        path = artifact(Path(command[command.index("--output") + 1]), recipe)
        (path / "conversion.json").write_text("{}")
        return 0

    monkeypatch.setattr(host, "run_converter", convert)
    host.dispatch(install_command(recipe))
    finish(host)
    assert host.models == [] and not host.registry.exists() and list(host.model_dir.iterdir()) == []
    assert host.worker is worker and not worker.closed
    assert events(host)[-2]["event"] == "error" and events(host)[-1]["state"] == "ready"


def test_cancel_acknowledgement_is_visible_only_after_the_operation_slot_is_released(host):
    acknowledgements = []

    class ObservedOutput(io.StringIO):
        def write(self, text):
            if json.loads(text).get("event") == "install_cancelled":
                acknowledgements.append(host.job is None)
            return super().write(text)

    host.output = ObservedOutput()
    entered = threading.Event()

    def operation(epoch, _kind):
        entered.set()
        assert host.install_cancelled.wait(2)
        host.check_install_cancelled(epoch)

    host.start(operation, "installing", "Installing", "huggingface", request_id="install", guided_install=True)
    assert entered.wait(2)
    host.dispatch({"command": "cancel_install", "request_id": "install"})
    finish(host)
    assert acknowledgements == [True]
