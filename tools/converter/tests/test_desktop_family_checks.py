"""Automatic family installation admission, lifecycle and evidence display."""

import io
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_image_checks import family_contract

from stark_convert.chat import ChatError, Worker
from stark_convert.desktop import Desktop, DesktopError
from stark_convert.image_checks import CheckStore
from stark_convert.vision_runtime import CONFIGURED_FAMILY_PROFILE, FAMILY_PROFILE


def artifact(path, *, schema=3, vision=True):
    path.mkdir(parents=True, exist_ok=True)
    value = {
        "type": "Qwen35VisionLanguageModelConfig" if vision else "LanguageModelConfig",
        "decoder_config": {},
        "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": "synthetic template"},
    }
    if vision:
        value["artifact_contract"] = {
            "schema_version": schema,
            "family": {"profile": FAMILY_PROFILE, "version": 1},
        }
    (path / "config.json").write_text(json.dumps(value))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic merged tensor fixture")
    if vision:
        (path / "preprocessor_config.json").write_text("{}")
        (path / "chat_template.jinja").write_text("synthetic template")
    return path


class StubWorker:
    def __init__(self, path):
        self.model_path = path
        self.process = SimpleNamespace(closed=False)
        self.closed = False
        self.images = False
        self.image_verification = "unverified"
        self.image_input = family_contract()
        self.cancellation = True
        self.streaming = True
        self.stop_sequences = True

    def close(self):
        self.closed = True
        self.process.closed = True


@pytest.fixture
def host(tmp_path, monkeypatch):
    attachments = tmp_path / "attachments"
    attachments.mkdir(mode=0o700)
    desktop = Desktop(tmp_path / "worker", tmp_path / "library", io.StringIO(), attachment_root=attachments)
    instances = []
    state = {"fail": False, "after_read": None, "before_create": None, "images": True}

    def factory(_executable, path, _timeout):
        if state["before_create"] is not None:
            state["before_create"]()
        instance = StubWorker(path)
        instances.append(instance)

        def read():
            assert instance.image_check_dir == desktop.data_dir / "image-checks"
            if state["fail"]:
                raise ChatError("The model did not pass local image checks.")
            instance.images = state["images"]
            instance.image_verification = "locally_tested" if instance.images else "unverified"
            if state["after_read"] is not None:
                state["after_read"]()
            return {
                "ready": True,
                "mode": "text-generation-v1",
                "max_context_tokens": 8192,
                "max_new_tokens": 4096,
                "streaming": "text-delta-v1",
                "stop_sequences": True,
                "vocab_size": 248320,
            }

        instance.read = read
        return instance

    monkeypatch.setattr("stark_convert.desktop.Worker", factory)
    monkeypatch.setattr(
        "stark_convert.desktop.terminate", lambda process: setattr(process, "closed", True) if process else None
    )
    monkeypatch.setattr("stark_convert.desktop.load_template", lambda *_args, **_kwargs: ({}, SimpleNamespace()))
    yield desktop, instances, state
    desktop.stop(quiet=True)
    desktop.attachments.close()


def events(desktop):
    return [json.loads(line) for line in desktop.output.getvalue().splitlines()]


@pytest.mark.parametrize(
    "family",
    [
        {"profile": FAMILY_PROFILE, "version": 1},
        {"profile": CONFIGURED_FAMILY_PROFILE, "version": 2},
    ],
)
def test_family_install_checks_before_publication_and_releases_temporary_worker(host, tmp_path, family):
    desktop, instances, _state = host
    model = artifact(tmp_path / "merged")
    config = json.loads((model / "config.json").read_text())
    config["artifact_contract"]["family"] = family
    (model / "config.json").write_text(json.dumps(config))
    desktop.install(desktop.epoch, "local", str(model))
    assert len(instances) == 1 and instances[0].closed
    assert desktop.worker is None and desktop.active_model is None
    assert len(desktop.models) == 1 and desktop.models[0]["image_verification"] == "locally_tested"
    assert desktop.models[0]["artifact_mode"] == "vision-language"
    saved = json.loads(desktop.registry.read_text())
    assert saved == desktop.models
    messages = events(desktop)
    check_index = next(index for index, value in enumerate(messages) if value.get("event") == "image_check")
    installed_index = next(index for index, value in enumerate(messages) if value.get("event") == "installed")
    assert check_index < installed_index


def test_family_install_unloads_previous_model_before_starting_check(host, tmp_path):
    desktop, instances, state = host
    previous = StubWorker(tmp_path / "old")
    previous.images = True
    desktop.worker, desktop.active_model, desktop.active_mode = previous, "old", "chat"

    def before_create():
        assert previous.closed and previous.process.closed
        assert desktop.worker is None and desktop.active_model is None

    state["before_create"] = before_create
    desktop.install(desktop.epoch, "local", str(artifact(tmp_path / "merged")))
    assert instances[0].closed and len(desktop.models) == 1


def test_failed_family_install_does_not_register_or_delete_user_source(host, tmp_path):
    desktop, instances, state = host
    state["fail"] = True
    source = artifact(tmp_path / "merged")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(DesktopError, match="local image checks"):
        desktop.install(desktop.epoch, "local", str(source))
    assert instances[0].closed and desktop.worker is None
    assert desktop.models == [] and not desktop.registry.exists()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert all(value.get("event") != "installed" for value in events(desktop))


def test_failed_converted_family_install_cleans_only_owned_output(host, tmp_path):
    desktop, instances, state = host
    state["fail"] = True
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "config.json").write_text("{}")

    def convert(command, _environment, _epoch):
        artifact(Path(command[command.index("--output") + 1]))
        return 0

    desktop.run_converter = convert
    with pytest.raises(DesktopError, match="local image checks"):
        desktop.install(desktop.epoch, "local", str(source), "vision-language")
    assert instances[0].closed and desktop.models == []
    assert list(desktop.model_dir.iterdir()) == []
    assert (source / "config.json").read_text() == "{}"


def test_deduplicated_family_install_rechecks_without_publishing_second_model(host, tmp_path):
    desktop, instances, _state = host
    source = artifact(tmp_path / "merged")
    desktop.install(desktop.epoch, "local", str(source))
    identifier = desktop.models[0]["id"]
    desktop.install(desktop.epoch, "local", str(source))
    assert len(instances) == 2 and all(value.closed for value in instances)
    assert len(desktop.models) == 1 and desktop.models[0]["id"] == identifier


def test_install_fenced_by_epoch_does_not_publish_success(host, tmp_path):
    desktop, instances, state = host
    state["after_read"] = lambda: setattr(desktop, "epoch", desktop.epoch + 1)
    with pytest.raises(DesktopError, match="interrupted"):
        desktop.install(desktop.epoch, "local", str(artifact(tmp_path / "merged")))
    assert instances[0].closed and desktop.worker is None
    assert desktop.models == [] and not desktop.registry.exists()
    assert all(value.get("event") != "installed" for value in events(desktop))


def test_lost_worker_identity_does_not_publish_unchecked_model_or_close_replacement(host, tmp_path):
    desktop, instances, state = host
    replacement = StubWorker(tmp_path / "replacement")

    def replace():
        desktop.worker = replacement
        desktop.active_model = "replacement"
        desktop.active_mode = "chat"

    state["after_read"] = replace
    try:
        desktop.install(desktop.epoch, "local", str(artifact(tmp_path / "merged")))
    except DesktopError:
        pass
    assert instances[0].closed and not replacement.closed
    assert desktop.worker is replacement and desktop.active_model == "replacement"
    assert desktop.models == [] and not desktop.registry.exists()


@pytest.mark.parametrize(
    "family", [{"profile": FAMILY_PROFILE, "version": 2}, {"profile": FAMILY_PROFILE, "version": True}, {}]
)
def test_unknown_family_rejected_before_current_model_is_unloaded(host, tmp_path, family):
    desktop, instances, _state = host
    source = artifact(tmp_path / "unsupported")
    config = json.loads((source / "config.json").read_text())
    config["artifact_contract"]["family"] = family
    (source / "config.json").write_text(json.dumps(config))
    previous = StubWorker(tmp_path / "previous")
    desktop.worker, desktop.active_model, desktop.active_mode = previous, "previous", "chat"
    with pytest.raises(DesktopError, match="not supported"):
        desktop.install(desktop.epoch, "local", str(source))
    assert instances == [] and not previous.closed
    assert desktop.worker is previous and desktop.active_model == "previous"
    assert desktop.models == [] and not desktop.registry.exists()


@pytest.mark.parametrize("schema", [None, 2])
def test_non_family_install_preserves_existing_admission_path(host, tmp_path, schema):
    desktop, instances, _state = host
    source = artifact(tmp_path / "legacy", schema=schema, vision=schema is not None)
    desktop.install(desktop.epoch, "local", str(source))
    assert instances == [] and len(desktop.models) == 1
    assert "image_verification" not in desktop.models[0]


@pytest.mark.parametrize("contract", [None, {}, [], "missing"])
def test_vision_install_requires_a_supported_artifact_contract(host, tmp_path, contract):
    desktop, instances, _state = host
    source = artifact(tmp_path / "missing-contract")
    config = json.loads((source / "config.json").read_text())
    if contract == "missing":
        del config["artifact_contract"]
    else:
        config["artifact_contract"] = contract
    (source / "config.json").write_text(json.dumps(config))
    desktop.run_converter = lambda *_args: pytest.fail("incomplete Stark artifact reached conversion")
    with pytest.raises(DesktopError, match="incomplete"):
        desktop.install(desktop.epoch, "local", str(source))
    assert not instances and not desktop.models and not desktop.registry.exists()


@pytest.mark.parametrize("kind", ["oversized", "fifo", "valid"])
def test_local_install_never_uses_unbounded_config_reopens(host, tmp_path, monkeypatch, kind):
    desktop, _instances, _state = host
    source = artifact(tmp_path / "bounded-config")
    config = source / "config.json"
    if kind == "oversized":
        config.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    elif kind == "fifo":
        config.unlink()
        os.mkfifo(config)
    original = Path.read_text

    def read_text(path, *_args, **_kwargs):
        if path == config:
            pytest.fail("unbounded configuration read")
        return original(path, *_args, **_kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    desktop.run_converter = lambda *_args: 1
    if kind == "valid":
        desktop.install(desktop.epoch, "local", str(source))
        assert len(desktop.models) == 1
    else:
        with pytest.raises(DesktopError, match="installation failed"):
            desktop.install(desktop.epoch, "local", str(source))
        assert not desktop.models and not desktop.registry.exists()


@pytest.mark.parametrize("unexpected_exit", [False, True])
def test_desktop_real_worker_shutdown_preserves_only_successful_live_receipts(tmp_path, unexpected_exit):
    source = artifact(tmp_path / "text-model", vision=False)
    executable = tmp_path / "worker"
    executable.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        "print(json.dumps({'ready':True}),flush=True)\n"
        "for _line in sys.stdin: pass\n"
    )
    executable.chmod(0o700)
    attachments = tmp_path / "attachments"
    attachments.mkdir(mode=0o700)
    desktop = Desktop(executable, tmp_path / "library", io.StringIO(), attachment_root=attachments)
    worker = Worker(executable, source, 5)
    worker.image_check_dir = desktop.data_dir / "image-checks"
    identity = {"synthetic_lifecycle_identity": unexpected_exit}
    store = CheckStore(worker.image_check_dir)
    try:
        assert worker.read()["ready"] is True
        store.save(identity)
        worker._image_check_identity = identity
        desktop.worker, desktop.active_model, desktop.active_mode = worker, "synthetic", "chat"
        if unexpected_exit:
            os.killpg(worker.process.pid, signal.SIGKILL)
            worker.process.wait(timeout=5)
        desktop.close_worker()
        assert worker.process.poll() is not None
        assert worker.process.stdout.closed and worker.process.stdin.closed
        assert desktop.worker is None and desktop.active_model is None
        assert store.load(identity) is (not unexpected_exit)
    finally:
        worker.close()
        store.close()
        desktop.attachments.close()


def test_running_family_reports_local_evidence_in_app_and_api(host, tmp_path):
    desktop, _instances, _state = host
    source = artifact(tmp_path / "merged")
    desktop.install(desktop.epoch, "local", str(source))
    identifier = desktop.models[0]["id"]
    desktop.run(desktop.epoch, identifier)
    assert desktop.worker.images and desktop.active_model == identifier
    ready = [value for value in events(desktop) if value.get("event") == "ready"][-1]
    assert ready["images"] is True and ready["image_verification"] == "locally_tested"
    status = desktop.api_status()
    assert status["capabilities"]["images"] is True and status["image_verification"] == "locally_tested"
    assert ready["image_verification"] != "reference_verified"


def test_registry_claim_cannot_enable_runtime_images_or_reference_parity(host, tmp_path):
    desktop, _instances, state = host
    source = artifact(tmp_path / "merged")
    desktop.models.append(
        {
            "id": "forged",
            "name": "synthetic",
            "path": str(source),
            "source": "local:fixture",
            "image_verification": "reference_verified",
        }
    )
    state["images"] = False
    desktop.run(desktop.epoch, "forged")
    assert desktop.settings_summary()["images"] is False
    assert desktop.api_status()["image_verification"] == "unverified"
    assert desktop.api_status()["capabilities"]["images"] is False
