"""Catalogue request fencing, pinned downloads and reversible library removal."""

import io
import json
import sys
import threading
import time

import pytest

from stark_convert import cli, model_catalog, model_removal
from stark_convert import desktop as module
from stark_convert.desktop import Desktop, DesktopError

PIN = "a" * 40


def report():
    return {
        "schema_version": 1,
        "family_id": "qwen3",
        "status": "ready",
        "message": "Choose a model.",
        "has_more": False,
        "items": [
            {
                "id": "b" * 32,
                "repo": "owner/Model",
                "revision": PIN,
                "name": "Model",
                "size_label": "0.6B",
                "parameter_count": 600000000,
                "weight_bytes": 1200000000,
                "quantization": "BF16",
                "artifact_modes": ["text"],
                "default_artifact_mode": "text",
                "downloadable": True,
            }
        ],
    }


class Worker:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def host(tmp_path):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    yield value
    value.stop(quiet=True)


def events(host):
    return [json.loads(line) for line in host.output.getvalue().splitlines()]


def source(tmp_path, identifier, *, managed, host):
    path = host.model_dir / identifier if managed else tmp_path / "external"
    path.mkdir()
    (path / "payload").write_bytes(b"synthetic model bytes")
    return {
        "id": identifier,
        "name": "Fixture model",
        "path": str(path),
        "source": "huggingface:owner/Model" if managed else "local:" + str(path),
    }


def test_catalogue_is_read_only_and_stale_family_results_cannot_publish(host, monkeypatch):
    host.worker = Worker()
    host.catalog_request_id, host.catalog_family_id = "request", "qwen3"
    monkeypatch.setattr(host, "run_catalog_check", lambda *_args: report())
    host.load_catalog("qwen3", "request", host.catalog_generation)
    assert events(host)[-1]["event"] == "model_catalog"
    assert events(host)[-1]["report"]["items"][0]["installed_ids"] == []
    assert not host.worker.closed and host.job is None
    before = len(events(host))
    host.cancel_catalog("request")
    host.load_catalog("qwen3", "request", host.catalog_generation - 1)
    assert len(events(host)) == before and host.catalog_entries == {}


@pytest.mark.parametrize(
    "kind",
    [
        "schema",
        "family",
        "revision",
        "repository",
        "mode",
        "boolean-size",
        "duplicate",
        "status",
        "group-id",
        "group-owner",
        "group-extra",
    ],
)
def test_malformed_catalogue_is_a_local_error_without_stopping_loaded_worker(host, monkeypatch, kind):
    value = report()
    if kind == "schema":
        value["schema_version"] = True
    elif kind == "family":
        value["family_id"] = "llama"
    elif kind == "revision":
        value["items"][0]["revision"] = "main"
    elif kind == "repository":
        value["items"][0]["repo"] = "https://example.com/model"
    elif kind == "mode":
        value["items"][0]["default_artifact_mode"] = "vision-language"
    elif kind == "boolean-size":
        value["items"][0]["weight_bytes"] = True
    elif kind == "duplicate":
        value["items"].append(dict(value["items"][0]))
    elif kind.startswith("group-"):
        group = {"id": "c" * 32, "name": "Model", "publisher": "owner"}
        if kind == "group-id":
            group["id"] = "invalid"
        elif kind == "group-owner":
            group["publisher"] = "owner/name"
        elif kind == "group-extra":
            group["unexpected"] = "field"
        value["items"][0]["model_group"] = group
    else:
        value["status"] = "offline"
    host.worker = Worker()
    host.catalog_request_id = "request"
    monkeypatch.setattr(host, "run_catalog_check", lambda *_args: value)
    host.load_catalog("qwen3", "request", host.catalog_generation)
    assert events(host)[-1]["report"]["status"] == "error"
    assert host.catalog_entries == {} and not host.worker.closed


def test_catalogue_accepts_optional_model_group_descriptor(host):
    value = report()
    value["items"][0]["model_group"] = {"id": "c" * 32, "name": "Model", "publisher": "owner"}
    assert host.catalog_report(value, "qwen3")["items"][0]["model_group"] == value["items"][0]["model_group"]


def test_catalogue_download_resolves_server_cache_and_pins_mode_and_revision(host, monkeypatch):
    item = report()["items"][0]
    host.catalog_request_id, host.catalog_entries = "request", {item["id"]: item}
    called = []
    monkeypatch.setattr(host, "start", lambda *args, **kwargs: called.append((args, kwargs)))
    payload = {
        "command": "install_catalog",
        "catalog_request_id": "request",
        "entry_id": item["id"],
        "support_request_id": "install",
    }
    assert host.dispatch(payload)
    args, kwargs = called[0]
    assert args[3:] == ("huggingface", "owner/Model", "text", "text", "install", PIN)
    assert kwargs == {"request_id": "install"}
    payload["catalog_request_id"] = "stale"
    host.dispatch(payload)
    assert len(called) == 1 and events(host)[-1]["request_id"] == "install"


def test_catalogue_rejects_unoffered_artifact_mode(host, monkeypatch):
    item = report()["items"][0]
    host.catalog_request_id, host.catalog_entries = "request", {item["id"]: item}
    monkeypatch.setattr(host, "start", lambda *_args, **_kwargs: pytest.fail("started unoffered mode"))
    with pytest.raises(DesktopError):
        host.install_catalog(
            {
                "catalog_request_id": "request",
                "entry_id": item["id"],
                "support_request_id": "install",
                "artifact_mode": "vision-language",
            }
        )


def test_installed_markers_match_source_version_and_artifact_type(host, monkeypatch):
    host.models = [{"id": "text"}, {"id": "image"}, {"id": "old"}]
    displays = {
        "text": {"group_id": "repo:owner/Model", "source_revision": PIN, "artifact_mode": "text"},
        "image": {"group_id": "repo:owner/Model", "source_revision": PIN, "artifact_mode": "vision-language"},
        "old": {"group_id": "repo:owner/Model", "source_revision": "c" * 40, "artifact_mode": "text"},
    }
    monkeypatch.setattr(module, "model_display", lambda model: displays[model["id"]])
    assert host.catalog_installed_ids(report()["items"][0]) == ["text", "image"]
    assert host.catalog_installed_ids(report()["items"][0], "vision-language") == ["image"]


def test_selected_revision_is_in_preflight_and_changed_report_cannot_start_download(host, monkeypatch):
    commands = []

    def check(command, _env, _epoch):
        commands.append(command)
        return {
            "schema_version": 1,
            "status": "conditional",
            "mode": "text",
            "message": "Compatible.",
            "source": {"repo": "owner/Model", "revision": "c" * 40},
        }

    monkeypatch.setattr(host, "run_support_check", check)
    monkeypatch.setattr(host, "run_converter", lambda *_args: pytest.fail("downloaded changed revision"))
    with pytest.raises(DesktopError, match="version changed"):
        host.install(host.epoch, "huggingface", "owner/Model", "text", "text", "install", PIN)
    assert commands[0][commands[0].index("--revision") + 1] == PIN
    assert list(host.model_dir.iterdir()) == []


def test_bounded_catalogue_child_and_cancellation_reap_process(host):
    value = report()
    command = [sys.executable, "-c", "import json; print(" + repr(json.dumps(value)) + ")"]
    assert host.run_catalog_check(command, {}, host.catalog_generation) == value
    with pytest.raises(ValueError):
        host.run_catalog_check([sys.executable, "-c", "print('x'*300000)"], {}, host.catalog_generation)
    assert host.catalog_process is None
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            host.run_catalog_check([sys.executable, "-c", "import time;time.sleep(60)"], {}, host.catalog_generation)
        )
    )
    thread.start()
    deadline = time.monotonic() + 2
    while host.catalog_process is None and time.monotonic() < deadline:
        time.sleep(0.01)
    process = host.catalog_process
    assert process is not None
    host.cancel_catalog()
    thread.join(timeout=3)
    assert not thread.is_alive() and result == [None]
    assert process.poll() is not None and process.stdout.closed


def test_cli_family_catalogue_has_no_source_or_output_and_no_weights(monkeypatch, capsys):
    monkeypatch.setattr(
        model_catalog, "family_catalog", lambda family, **kwargs: report() if family == "qwen3" else None
    )
    monkeypatch.setattr(sys, "argv", ["stark-convert", "--family-models", "qwen3", "--offline"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


@pytest.mark.parametrize("managed", [True, False])
def test_removal_moves_only_owned_files_and_preserves_an_unrelated_worker(host, tmp_path, monkeypatch, managed):
    model = source(tmp_path, "a" * 32, managed=managed, host=host)
    host.models = [model]
    host.persist()
    host.worker, host.active_model = Worker(), "different-model"
    trash = tmp_path / "trash"
    trash.mkdir()
    actual = model_removal.remove_model
    moves = []

    def move(path):
        target = trash / path.name
        path.rename(target)
        moves.append(target)
        return target

    monkeypatch.setattr(module, "remove_model", lambda plan, commit: actual(plan, commit, trash=move))
    action = "trash" if managed else "unregister"
    host.remove_saved_model(host.epoch, model["id"], action, "remove")
    assert host.models == [] and json.loads(host.registry.read_bytes()) == []
    assert not host.worker.closed and host.active_model == "different-model"
    assert bool(moves) == managed
    assert events(host)[-2]["event"] == "model_removed" and events(host)[-2]["request_id"] == "remove"
    assert (tmp_path / "external").exists() != managed


def test_failed_removal_restores_registry_and_folder_and_action_mismatch_cannot_delete(host, tmp_path, monkeypatch):
    model = source(tmp_path, "a" * 32, managed=True, host=host)
    host.models = [model]
    host.persist()
    before = host.registry.read_bytes()
    host.remove_saved_model(host.epoch, model["id"], "unregister", "old-action")
    assert events(host)[-2]["event"] == "model_remove_error"
    assert host.models == [model]
    actual = model_removal.remove_model
    trash = tmp_path / "trash"
    trash.mkdir()

    def move(path):
        result = trash / path.name
        path.rename(result)
        return result

    monkeypatch.setattr(module, "remove_model", lambda plan, commit: actual(plan, commit, trash=move))
    monkeypatch.setattr(host, "persist", lambda: (_ for _ in ()).throw(OSError()))
    host.remove_saved_model(host.epoch, model["id"], "trash", "remove")
    assert host.models == [model] and host.registry.read_bytes() == before
    assert (host.model_dir / model["id"] / "payload").read_bytes() == b"synthetic model bytes"
    assert list(trash.iterdir()) == []


def test_removing_loaded_model_closes_only_that_worker(host, tmp_path):
    model = source(tmp_path, "a" * 32, managed=False, host=host)
    host.models = [model]
    worker = Worker()
    host.worker, host.active_model = worker, model["id"]
    host.remove_saved_model(host.epoch, model["id"], "unregister", "remove")
    assert worker.closed and host.worker is None and host.active_model is None
    assert events(host)[-2]["stopped_model_id"] == model["id"]


def test_concurrent_library_projection_cannot_resurrect_a_removed_row(host, tmp_path, monkeypatch):
    model = source(tmp_path, "a" * 32, managed=False, host=host)
    host.models = [model]
    entered, proceed = threading.Event(), threading.Event()

    def display(_model):
        entered.set()
        assert proceed.wait(timeout=3)
        return None

    monkeypatch.setattr(module, "model_display", display)
    listing = threading.Thread(target=host.library)
    listing.start()
    assert entered.wait(timeout=2)
    removing = threading.Thread(
        target=lambda: host.remove_saved_model(host.epoch, model["id"], "unregister", "remove")
    )
    removing.start()
    proceed.set()
    listing.join(timeout=3)
    removing.join(timeout=3)
    assert not listing.is_alive() and not removing.is_alive()
    assert [event for event in events(host) if event["event"] == "models"][-1]["models"] == []
