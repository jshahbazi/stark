"""Catalogue request fencing, pinned downloads and reversible library removal."""

import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from stark_convert import cli, hybrid_prepare, model_catalog, model_removal, native_catalog
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


def test_legacy_catalogue_more_flag_without_offset_is_not_pageable(host):
    value = report()
    value["has_more"] = True
    normalized = host.catalog_report(value, "qwen3")
    assert normalized["has_more"] is False and normalized["next_offset"] is None


def test_catalogue_page_offsets_are_correlated_and_bounded(host):
    value = report()
    value.update(offset=30, next_offset=60, has_more=True)
    assert host.catalog_report(value, "qwen3", 30)["next_offset"] == 60
    for change in (
        {"offset": 29},
        {"offset": True},
        {"next_offset": 30},
        {"next_offset": True},
        {"next_offset": 10_000_001},
        {"next_offset": None},
    ):
        invalid = {**value, **change}
        with pytest.raises(ValueError):
            host.catalog_report(invalid, "qwen3", 30)
    with pytest.raises(ValueError):
        host.catalog_report(value, "qwen3", 31)
    with pytest.raises(ValueError):
        host.catalog_report({**value, "offset": None}, "qwen3")


def test_catalogue_append_keeps_earlier_rows_installable(host, monkeypatch):
    first = report()["items"][0]
    second = {**first, "id": "d" * 32, "repo": "owner/NextModel", "name": "NextModel"}
    host.catalog_request_id, host.catalog_family_id = "request", "qwen3"
    host.catalog_next_offset, host.catalog_active_offset = 30, 30
    host.catalog_entries = {first["id"]: dict(first)}
    value = report()
    value.update(items=[second], offset=30, next_offset=60, has_more=True)
    monkeypatch.setattr(host, "run_catalog_check", lambda *_args: value)

    host.load_catalog("qwen3", "request", host.catalog_generation, 30)

    assert host.catalog_entries.keys() == {first["id"], second["id"]}
    assert host.catalog_next_offset == 60 and host.catalog_active_offset is None
    emitted = events(host)[-1]["report"]
    assert emitted["offset"] == 30 and emitted["items"][0]["id"] == second["id"]
    called = []
    monkeypatch.setattr(host, "start", lambda *args, **kwargs: called.append((args, kwargs)))
    host.install_catalog(
        {
            "command": "install_catalog",
            "catalog_request_id": "request",
            "entry_id": first["id"],
            "artifact_mode": "text",
            "support_request_id": "support",
        }
    )
    assert called and called[0][0][0] == host.install


def test_catalogue_failed_append_retains_retry_offset_and_previous_rows(host, monkeypatch):
    first = report()["items"][0]
    host.catalog_request_id, host.catalog_family_id = "request", "qwen3"
    host.catalog_next_offset, host.catalog_active_offset = 30, 30
    host.catalog_entries = {first["id"]: dict(first)}
    value = report()
    value.update(status="error", items=[], offset=30, next_offset=None, has_more=False)
    monkeypatch.setattr(host, "run_catalog_check", lambda *_args: value)

    host.load_catalog("qwen3", "request", host.catalog_generation, 30)

    assert host.catalog_next_offset == 30
    assert host.catalog_entries == {first["id"]: first}
    assert events(host)[-1]["report"]["offset"] == 30


def test_catalogue_rejects_unrequested_offsets_and_stale_page_ids(host):
    with pytest.raises(DesktopError):
        host.start_catalog("qwen3", "request", 30)
    host.catalog_request_id, host.catalog_family_id = "request", "qwen3"
    host.catalog_next_offset = 30
    host.catalog_active_offset = None
    with pytest.raises(DesktopError):
        host.start_catalog("qwen3", "stale", 30)
    with pytest.raises(DesktopError):
        host.start_catalog("qwen3", "request", 60)
    with pytest.raises(DesktopError):
        host.start_catalog("qwen3", "request", True)
    host.catalog_active_offset = 30
    with pytest.raises(DesktopError):
        host.start_catalog("qwen3", "request", 30)


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


def test_glm_catalogue_download_uses_only_the_offered_pinned_native_choice(host, monkeypatch):
    item = native_catalog.catalog_item()
    host.catalog_request_id, host.catalog_family_id = "request", native_catalog.FAMILY_ID
    host.catalog_entries = {item["id"]: item}
    called = []
    monkeypatch.setattr(host, "start", lambda *args, **kwargs: called.append((args, kwargs)))
    payload = {
        "catalog_request_id": "request",
        "entry_id": item["id"],
        "support_request_id": "install",
        "source": "untrusted/Replacement",
        "revision": "f" * 40,
    }
    host.install_catalog(payload)
    args, kwargs = called[0]
    assert args[3:] == ("native", "glm53-q2", "text", "text", "install", item["revision"])
    assert kwargs == {"request_id": "install"}
    for change in ({"revision": "f" * 40}, {"repo": "untrusted/Replacement"}, {"artifact_modes": ["vision-language"]}):
        host.catalog_entries[item["id"]] = {**item, **change}
        with pytest.raises(DesktopError):
            host.install_catalog(payload)
    host.catalog_entries[item["id"]] = item
    for change in ({"catalog_request_id": "stale"}, {"entry_id": "unknown"}, {"artifact_mode": "vision-language"}):
        with pytest.raises(DesktopError):
            host.install_catalog({**payload, **change})
    assert len(called) == 1


def test_qwen_mixed_catalogue_download_uses_the_offered_native_choice(host, monkeypatch):
    item = native_catalog.catalog_item(native_catalog.QWEN_CHOICE)
    host.catalog_request_id, host.catalog_family_id = "request", native_catalog.QWEN_FAMILY_ID
    host.catalog_entries = {item["id"]: item}
    called = []
    monkeypatch.setattr(host, "start", lambda *args, **kwargs: called.append((args, kwargs)))

    host.install_catalog(
        {
            "catalog_request_id": "request",
            "entry_id": item["id"],
            "support_request_id": "install",
            "artifact_mode": "text",
        }
    )

    args, kwargs = called[0]
    assert args[3:] == ("native", native_catalog.QWEN_CHOICE, "text", "text", "install", item["revision"])
    assert kwargs == {"request_id": "install"}
    host.catalog_entries[item["id"]] = {**item, "repo": "untrusted/Replacement"}
    with pytest.raises(DesktopError):
        host.install_catalog(
            {
                "catalog_request_id": "request",
                "entry_id": item["id"],
                "support_request_id": "install",
                "artifact_mode": "text",
            }
        )


@pytest.fixture
def native_download(host, monkeypatch):
    from test_hybrid_prepare import metadata

    source = {**hybrid_prepare.SOURCES["glm53-q2"], "weight_bytes": 4}
    monkeypatch.setitem(hybrid_prepare.SOURCES, "glm53-q2", source)
    monkeypatch.setitem(module.NATIVE_SOURCES, "glm53-q2", source)

    def snapshot(repo, **options):
        root = Path(options["local_dir"])
        if repo == source["weights_repo"]:
            assert options["revision"] == source["weights_revision"]
            shards = root / "UD-Q2_K_XL"
            shards.mkdir(exist_ok=True)
            for index in range(4):
                (shards / f"model-{index}.gguf").write_bytes(b"x")
        else:
            assert repo == source["metadata_repo"]
            assert options["revision"] == source["metadata_revision"]
            metadata(root)

    monkeypatch.setattr(hybrid_prepare, "snapshot_download", snapshot)
    monkeypatch.setattr(host, "run_support_check", lambda *_args: pytest.fail("generic preflight used for GLM"))

    def prepare(command, _environment, _epoch):
        assert command[:6] == [sys.executable, "-I", "-B", "-m", "stark_convert.hybrid_prepare", "glm53-q2"]
        assert command[6] == "--output" and len(command) == 8
        hybrid_prepare.prepare("glm53-q2", Path(command[7]))
        return 0

    monkeypatch.setattr(host, "run_converter", prepare)
    return source


def test_glm_preparation_registers_verified_managed_folder_without_loading(host, native_download):
    host.worker = Worker()
    host.begin_install_progress(host.epoch, "native")
    assert host.install_progress["download"]["state"] == "waiting"
    host.install(host.epoch, "native", "glm53-q2", "text", "text", "install", native_download["weights_revision"])
    assert len(host.models) == 1 and not host.worker.closed
    model = host.models[0]
    assert model["source"] == "huggingface:" + native_download["weights_repo"]
    assert Path(model["path"]).parent == host.model_dir
    assert module.model_valid(Path(model["path"]))
    assert json.loads(host.registry.read_bytes()) == host.models
    assert list(host.model_dir.iterdir()) == [Path(model["path"])]
    assert next(event for event in events(host) if event["event"] == "installed")["support_request_id"] == "install"
    assert host.install_progress["download"]["state"] == "complete"


def test_glm_download_retries_reuse_owned_partial_shards(host, native_download, monkeypatch):
    snapshot = hybrid_prepare.snapshot_download
    attempts = []

    def interrupted(repo, **options):
        if repo == native_download["weights_repo"]:
            root = Path(options["local_dir"])
            attempts.append(root)
            if len(attempts) == 1:
                shards = root / "UD-Q2_K_XL"
                shards.mkdir()
                (shards / "model-0.gguf").write_bytes(b"x")
                raise OSError("synthetic interrupted download")
            assert (root / "UD-Q2_K_XL" / "model-0.gguf").read_bytes() == b"x"
        return snapshot(repo, **options)

    monkeypatch.setattr(hybrid_prepare, "snapshot_download", interrupted)
    args = (host.epoch, "native", "glm53-q2", "text", "text", "install", native_download["weights_revision"])
    with pytest.raises(OSError):
        host.install(*args)
    assert host.models == []
    host.install(*args)
    assert attempts[0] == attempts[1] and len(host.models) == 1


def test_glm_registry_failure_keeps_completed_download_for_retry(host, native_download, monkeypatch):
    persist = host.persist
    monkeypatch.setattr(host, "persist", lambda: (_ for _ in ()).throw(OSError("synthetic registry failure")))
    args = (host.epoch, "native", "glm53-q2", "text", "text", "install", native_download["weights_revision"])
    with pytest.raises(OSError):
        host.install(*args)
    assert host.models == [] and list(host.model_dir.iterdir()) == []
    cached = list((host.data_dir / "native-downloads").glob("*/model"))
    assert len(cached) == 1 and module.model_valid(cached[0])
    monkeypatch.setattr(host, "persist", persist)
    monkeypatch.setattr(host, "run_converter", lambda *_args: pytest.fail("redownloaded complete pinned model"))
    host.install(*args)
    assert len(host.models) == 1 and not cached[0].exists()


def test_glm_download_stage_rejects_concurrent_installers_and_symlinks(host, tmp_path, native_download):
    with module.native_download_stage(host.data_dir, "glm53-q2"):
        with pytest.raises(DesktopError, match="already being downloaded"):
            host.install(
                host.epoch, "native", "glm53-q2", "text", "text", "install", native_download["weights_revision"]
            )
    assert host.models == []
    other = tmp_path / "other-data"
    other.mkdir()
    (other / "native-downloads").symlink_to(host.data_dir / "native-downloads")
    with pytest.raises(DesktopError, match="unavailable"):
        with module.native_download_stage(other, "glm53-q2"):
            pytest.fail("followed native download directory symlink")


@pytest.mark.parametrize("failure", ["identity", "persist", "cancel", "child"])
def test_glm_failed_or_cancelled_preparation_never_publishes_a_library_entry(
    host, native_download, monkeypatch, failure
):
    actual = host.run_converter

    def prepare(*args):
        result = actual(*args)
        if failure == "cancel":
            host.epoch += 1
        return 1 if failure == "child" else result

    monkeypatch.setattr(host, "run_converter", prepare)
    if failure == "identity":
        identity = module.prepared_identity
        monkeypatch.setattr(module, "prepared_identity", lambda path: {**identity(path), "revision": "f" * 40})
    elif failure == "persist":
        monkeypatch.setattr(host, "persist", lambda: (_ for _ in ()).throw(OSError("synthetic registry failure")))
    args = (host.epoch, "native", "glm53-q2", "text", "text", "install", native_download["weights_revision"])
    if failure == "cancel":
        host.install(*args)
    else:
        with pytest.raises((DesktopError, OSError)):
            host.install(*args)
    assert host.models == [] and list(host.model_dir.iterdir()) == []
    assert not host.registry.exists()
    assert all(event["event"] != "installed" for event in events(host))


@pytest.mark.parametrize(
    "choice,revision,artifact_mode,support_mode",
    [
        ("glm53-q2", None, "text", "text"),
        ("glm53-q2", "a" * 40, "text", "text"),
        ("qwen38-mixed", "a" * 40, "text", "text"),
        ("glm53-q2", None, "vision-language", "vision-language"),
        ("glm53-q2", None, "text", "decision"),
    ],
)
def test_native_installer_rejects_unpinned_or_unoffered_choices(
    host, monkeypatch, choice, revision, artifact_mode, support_mode
):
    monkeypatch.setattr(host, "run_converter", lambda *_args: pytest.fail("started unoffered native download"))
    with pytest.raises(DesktopError):
        host.install(host.epoch, "native", choice, artifact_mode, support_mode, "install", revision)
    assert list(host.model_dir.iterdir()) == []


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
    called = []
    monkeypatch.setattr(
        model_catalog,
        "family_catalog",
        lambda family, **kwargs: called.append((family, kwargs)) or report() if family == "qwen3" else None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["stark-convert", "--family-models", "qwen3", "--catalog-offset", "30", "--offline"],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert called == [("qwen3", {"offset": 30, "offline": True})]


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
