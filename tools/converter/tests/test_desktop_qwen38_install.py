"""Qwen3.8 source-to-canonical installation routing with synthetic files."""

import io
import json
from pathlib import Path

import pytest
from test_desktop import conditional_support_report

import stark_convert.desktop as desktop_module
from stark_convert.desktop import Desktop, DesktopError, model_valid


def source_export(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text", "vocab_size": 8},
                "stark_native_hybrid": {"schema_version": 1, "source": "synthetic/qwen38", "revision": "a" * 40},
                "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": "{{ messages[0]['content'] }}"},
            }
        )
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic source quantized bytes")
    (path / "native-source.json").write_text("{}")
    return path


def canonical_artifact(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "Qwen38LanguageModelConfig",
                "decoder_config": {"architecture": {"vocab_size": 8}},
                "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": "{{ messages[0]['content'] }}"},
            }
        )
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic canonical fixture; execution is not asserted")
    return path


@pytest.fixture
def desktop(tmp_path):
    instance = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO(), timeout=1)
    yield instance
    instance.stop(quiet=True)


@pytest.fixture
def converter(desktop, monkeypatch):
    commands = []
    preflights = []

    def preflight(command, environment, epoch):
        preflights.append(command)
        result = conditional_support_report(command, environment, epoch)
        return {**result, "family_id": "qwen4_exp", "family_name": "Qwen3.8", "architecture": "qwen4_exp"}

    def convert(command, environment, epoch):
        commands.append(command)
        canonical_artifact(Path(command[command.index("--output") + 1]))
        return 0

    monkeypatch.setattr(desktop, "run_support_check", preflight)
    monkeypatch.setattr(desktop, "run_converter", convert)
    return commands, preflights


def register_legacy(desktop, source):
    model = {"id": "old-source", "name": "Qwen3.8 source", "path": str(source), "source": "local:" + str(source)}
    desktop.models.append(model)
    desktop.persist()
    return model


def test_prepared_qwen_source_receipt_does_not_grant_runtime_admission(tmp_path, monkeypatch):
    source = source_export(tmp_path / "source")
    monkeypatch.setattr(desktop_module, "prepared_identity", lambda _path: {"architecture": "qwen4_exp"})
    assert not model_valid(source)


def test_canonical_qwen_artifact_uses_ordinary_ready_import(desktop, tmp_path, monkeypatch):
    artifact = canonical_artifact(tmp_path / "canonical")
    assert model_valid(artifact)

    def unexpected_converter(*_args):
        pytest.fail("an existing canonical artifact must not be converted again")

    monkeypatch.setattr(desktop, "run_converter", unexpected_converter)
    desktop.install(desktop.epoch, "local", str(artifact))
    assert len(desktop.models) == 1
    assert desktop.models[0]["path"] == str(artifact)
    desktop.install(desktop.epoch, "local", str(artifact))
    assert len(desktop.models) == 1


def test_legacy_local_qwen_source_runs_ordinary_converter(desktop, tmp_path, converter):
    source = source_export(tmp_path / "source")
    original = (source / "model.safetensors").read_bytes()
    desktop.install(desktop.epoch, "local", str(source))
    commands, preflights = converter
    assert len(commands) == len(preflights) == 1
    assert commands[0][commands[0].index("stark_convert.cli") + 1] == str(source)
    assert "--offline" in commands[0]
    assert commands[0][commands[0].index("--metadata-profile") + 1] == "hf-generic"
    installed = Path(desktop.models[0]["path"])
    assert installed.parent == desktop.model_dir
    assert model_valid(installed)
    assert (source / "model.safetensors").read_bytes() == original
    assert json.loads((source / "config.json").read_text())["model_type"] == "qwen4_exp"


def test_reimport_replaces_legacy_registration_only_after_canonical_publication(desktop, tmp_path, converter):
    source = source_export(tmp_path / "source")
    previous = register_legacy(desktop, source)
    desktop.install(desktop.epoch, "local", str(source))
    assert len(converter[0]) == 1
    assert len(desktop.models) == 1
    assert desktop.models[0]["id"] != previous["id"]
    assert model_valid(Path(desktop.models[0]["path"]))
    assert source.is_dir()
    assert (source / "model.safetensors").read_bytes() == b"synthetic source quantized bytes"


def test_failed_migration_preserves_legacy_registration_and_source(desktop, tmp_path, converter, monkeypatch):
    source = source_export(tmp_path / "source")
    previous = register_legacy(desktop, source)
    monkeypatch.setattr(desktop, "run_converter", lambda *_args: 1)
    with pytest.raises(DesktopError, match="installation failed"):
        desktop.install(desktop.epoch, "local", str(source))
    assert desktop.models == [previous]
    assert source.is_dir()
    assert list(desktop.model_dir.iterdir()) == []


def test_failed_registry_commit_restores_legacy_registration(desktop, tmp_path, converter, monkeypatch):
    source = source_export(tmp_path / "source")
    previous = register_legacy(desktop, source)

    def fail():
        raise OSError("synthetic persistence failure")

    monkeypatch.setattr(desktop, "persist", fail)
    with pytest.raises(OSError):
        desktop.install(desktop.epoch, "local", str(source))
    assert desktop.models == [previous]
    assert source.is_dir()
    assert list(desktop.model_dir.iterdir()) == []


def test_legacy_source_run_explains_conversion_before_worker_launch(desktop, tmp_path):
    source = source_export(tmp_path / "source")
    previous = register_legacy(desktop, source)
    with pytest.raises(DesktopError, match="Import.*again"):
        desktop.run(desktop.epoch, previous["id"])
    assert desktop.worker is None


def test_huggingface_qwen_download_retains_canonical_converter_path(desktop, converter):
    desktop.install(desktop.epoch, "huggingface", "synthetic/qwen38-source")
    commands, preflights = converter
    assert len(commands) == len(preflights) == 1
    assert commands[0][commands[0].index("stark_convert.cli") + 1] == "synthetic/qwen38-source"
    assert commands[0][commands[0].index("--revision") + 1] == "a" * 40
    assert "--offline" not in commands[0]
    assert model_valid(Path(desktop.models[0]["path"]))
