"""Metadata-first app admission, revision binding and cancellation boundaries."""

import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark_convert.desktop import Desktop, DesktopError

REVISION = "a" * 40


def support_report(*, status="conditional", mode="text", repo="example/model", revision=REVISION):
    return {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "family_id": "llama",
        "family_name": "Llama",
        "architecture": "llama",
        "model_kind": "language",
        "message": {
            "conditional": "Recognized family; validation is still required.",
            "unsupported": "This family is not supported for image input.",
            "inconclusive": "Could not verify this model's family. Check access and retry.",
        }[status],
        "reasons": [],
        "warnings": [],
        "pending_checks": ["tensor_layout", "native_execution"],
        "modes": {"text": "conditional", "decision": "conditional", "vision-language": "unsupported"},
        "alternatives": ["text", "decision"],
        "qualification": {"status": "not_assessed"},
        "source": {"repo": repo, "revision": revision, "metadata_sha256": "b" * 64, "metadata_sources": []},
    }


def text_artifact(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "LanguageModelConfig",
                "decoder_config": {},
                "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": "{{ messages[0].content }}"},
            }
        )
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic model fixture")
    return path


@pytest.fixture
def host(tmp_path):
    controller = Desktop(tmp_path / "worker", tmp_path / "library", io.StringIO(), timeout=2)
    yield controller
    controller.stop(quiet=True)


def emitted(host):
    return [json.loads(line) for line in host.output.getvalue().splitlines()]


@pytest.mark.parametrize("mode", ["text", "decision", "vision-language"])
def test_conditional_family_checks_before_converter_and_pins_primary_revision(host, mode):
    calls = []

    def preflight(command, _environment, epoch):
        assert epoch == host.epoch
        assert "--preflight" in command and command[command.index("--support-mode") + 1] == mode
        assert not calls
        calls.append("preflight")
        return support_report(mode=mode)

    def convert(command, _environment, _epoch):
        assert calls == ["preflight"]
        assert command[command.index("--revision") + 1] == REVISION
        assert command[command.index("--support-mode") + 1] == mode
        calls.append("weights")
        text_artifact(Path(command[command.index("--output") + 1]))
        return 0

    host.run_support_check, host.run_converter = preflight, convert
    host.check_family_install = lambda *_args: None
    host.install(
        host.epoch,
        "huggingface",
        "example/model",
        "vision-language" if mode == "vision-language" else None,
        mode,
        "fixture-request",
    )
    assert calls == ["preflight", "weights"] and len(host.models) == 1
    report = next(event for event in emitted(host) if event["event"] == "model_support")
    assert report["request_id"] == "fixture-request" and report["report"]["mode"] == mode


@pytest.mark.parametrize("status", ["unsupported", "inconclusive"])
def test_blocked_support_never_downloads_or_disturbs_running_model(host, status):
    existing = SimpleNamespace(images=True, image_verification="locally_tested")
    host.worker, host.active_model, host.active_mode = existing, "existing", "chat"
    host.messages = [{"role": "user", "content": "synthetic existing conversation"}]
    before = list(host.messages)
    host.run_support_check = lambda *_args: support_report(status=status)
    host.run_converter = lambda *_args: pytest.fail("blocked family reached the weight downloader")
    try:
        with pytest.raises(DesktopError, match="not supported|Could not verify"):
            host.install(host.epoch, "huggingface", "example/model", support_request_id="fixture-request")
        assert host.worker is existing and host.active_model == "existing" and host.messages == before
        assert host.models == [] and not host.registry.exists() and list(host.model_dir.iterdir()) == []
        assert (
            next(event for event in emitted(host) if event["event"] == "model_support")["report"]["status"] == status
        )
    finally:
        host.worker = None


@pytest.mark.parametrize("change", ["epoch", "mode", "revision", "repo", "empty"])
def test_stale_or_mismatched_support_cannot_start_download(host, change):
    def preflight(*_args):
        report = support_report()
        if change == "epoch":
            host.epoch += 1
        elif change == "mode":
            report["mode"] = "decision"
        elif change == "revision":
            report["source"]["revision"] = None
        elif change == "repo":
            report["source"]["repo"] = "other/model"
        elif change == "empty":
            return None
        return report

    host.run_support_check = preflight
    host.run_converter = lambda *_args: pytest.fail("stale or mismatched support reached weights")
    try:
        host.install(host.epoch, "huggingface", "example/model", support_request_id="fixture-request")
    except DesktopError:
        assert change != "epoch"
    assert host.models == [] and not host.registry.exists()
    assert all(event.get("event") != "model_support" for event in emitted(host))


def test_converted_image_artifact_is_importable_for_chat_without_conversion_checkbox(host, tmp_path):
    model = text_artifact(tmp_path / "converted-image")
    config = json.loads((model / "config.json").read_text())
    config["type"] = "Qwen35VisionLanguageModelConfig"
    config["artifact_contract"] = {
        "schema_version": 3,
        "family": {"profile": "qwen35-dense-bf16-merged-v2", "version": 2},
    }
    for name in ("preprocessor_config.json", "chat_template.jinja"):
        (model / name).write_text("synthetic fixture")
    (model / "config.json").write_text(json.dumps(config))
    checks = []
    host.check_family_install = lambda _epoch, path: checks.append(path) or "locally_tested"
    host.run_support_check = lambda *_args: pytest.fail("converted import ran a source downloader")
    host.install(host.epoch, "local", str(model), support_mode="text")
    assert checks == [model] and host.models[0]["artifact_mode"] == "vision-language"
    with pytest.raises(DesktopError, match="Chat mode only"):
        host.install(host.epoch, "local", str(model), support_mode="decision")


@pytest.mark.parametrize("body", [b"not json", b"{}", b"x" * 65537])
def test_support_child_malformed_or_oversized_output_is_bounded(host, body):
    command = [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({body!r})"]
    with pytest.raises(DesktopError, match="support|family"):
        host.run_support_check(command, dict(os.environ), host.epoch)
    assert host.converter is None


def test_real_support_child_reply_and_epoch_cancellation_reap_process(host):
    command = [sys.executable, "-c", f"import json; print(json.dumps({support_report()!r}))"]
    assert host.run_support_check(command, dict(os.environ), host.epoch)["status"] == "conditional"
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            host.run_support_check(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                dict(os.environ),
                host.epoch,
            )
        )
    )
    thread.start()
    deadline = time.monotonic() + 2
    while host.converter is None and time.monotonic() < deadline:
        time.sleep(0.01)
    process = host.converter
    assert process is not None
    host.epoch += 1
    thread.join(timeout=3)
    assert not thread.is_alive() and results == [None]
    assert process.poll() is not None and process.stdout.closed and host.converter is None


def test_family_matrix_is_discoverable_without_loading_a_model(host):
    assert host.dispatch({"command": "support"}) is True
    event = emitted(host)[-1]
    assert event["event"] == "model_families" and event["report"]["schema_version"] == 1
    assert host.worker is None and host.active_model is None
