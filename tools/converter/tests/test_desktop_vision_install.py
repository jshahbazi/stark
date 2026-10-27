"""Desktop admission and install-mode routing for synthetic vision artifacts."""

import io
import json
from pathlib import Path

import pytest

from stark_convert.desktop import Desktop, DesktopError, model_valid


def make_artifact(path: Path, *, vision: bool, processor: bool = True, template: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "decoder_config": {},
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "default_system_prompt": "synthetic system",
            "prompt_template": "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}",
        },
    }
    if vision:
        config["type"] = "Qwen35VisionLanguageModelConfig"
        # These legacy install-routing fixtures use the schema-2 path. Native
        # admission and qualification use separate complete artifact fixtures.
        config["artifact_contract"] = {"type": "Qwen35VisionLanguageArtifact", "schema_version": 2}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic weights")
    if processor:
        (path / "preprocessor_config.json").write_text("{}")
    if template:
        (path / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    return path


@pytest.fixture
def desktop(tmp_path):
    instance = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO(), timeout=1)
    yield instance
    instance.stop(quiet=True)


def test_local_vision_artifact_admitted_deduplicated_and_mode_conflict_rejected(desktop, tmp_path):
    folder = make_artifact(tmp_path / "vision-model", vision=True)
    assert model_valid(folder)

    desktop.install(desktop.epoch, "local", str(folder))
    assert len(desktop.models) == 1
    first = desktop.models[0]
    assert first["artifact_mode"] == "vision-language"
    assert first["source"] == "local:" + str(folder.resolve())

    desktop.install(desktop.epoch, "local", str(folder), "vision-language")
    assert len(desktop.models) == 1

    with pytest.raises(DesktopError, match="does not match"):
        desktop.install(desktop.epoch, "local", str(folder), "text")
    assert len(desktop.models) == 1


@pytest.mark.parametrize("missing", ["processor", "template"])
def test_local_vision_artifact_requires_processor_and_template(desktop, tmp_path, missing):
    folder = make_artifact(
        tmp_path / "incomplete-vision",
        vision=True,
        processor=missing != "processor",
        template=missing != "template",
    )
    assert not model_valid(folder)
    with pytest.raises(DesktopError, match="incomplete"):
        desktop.install(desktop.epoch, "local", str(folder), "vision-language")
    assert desktop.models == []


def test_local_vision_requires_explicit_mode_when_not_already_converted(desktop, tmp_path):
    folder = tmp_path / "source-vision"
    make_artifact(folder, vision=True)
    # A Stark artifact is admitted as converted; an upstream folder needs the
    # converter path. Remove the Stark decoder marker to model that source.
    config = json.loads((folder / "config.json").read_text())
    config.pop("decoder_config")
    (folder / "config.json").write_text(json.dumps(config))

    commands = []

    def convert(command, environment, epoch):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        make_artifact(output, vision=True)
        return 0

    desktop.run_converter = convert
    desktop.install(desktop.epoch, "local", str(folder), "vision-language")
    assert len(desktop.models) == 1
    assert desktop.models[0]["artifact_mode"] == "vision-language"
    assert "--artifact-mode" in commands[0]
    assert commands[0][commands[0].index("--artifact-mode") + 1] == "vision-language"


def test_huggingface_text_and_vision_installs_coexist_and_vision_flag_is_forwarded(desktop):
    commands = []

    def convert(command, environment, epoch):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        vision = "--artifact-mode" in command
        make_artifact(output, vision=vision)
        return 0

    desktop.run_converter = convert
    source = "synthetic-owner/synthetic-qwen"
    desktop.install(desktop.epoch, "huggingface", source)
    desktop.install(desktop.epoch, "huggingface", source, "vision-language")

    assert len(desktop.models) == 2
    assert {model.get("artifact_mode", "text") for model in desktop.models} == {"text", "vision-language"}
    assert commands[0][0:3] == commands[1][0:3]
    assert "--artifact-mode" not in commands[0]
    assert commands[1][commands[1].index("--artifact-mode") + 1] == "vision-language"


def test_dispatch_rejects_invalid_install_mode_without_starting_job(desktop):
    with pytest.raises(DesktopError, match="Choose text or vision-language"):
        desktop.dispatch(
            {"command": "install", "kind": "huggingface", "source": "owner/model", "artifact_mode": "image"}
        )
    assert desktop.job is None


def test_vision_artifact_is_rejected_for_decision_before_score_worker(desktop, tmp_path, monkeypatch):
    folder = make_artifact(tmp_path / "decision-vision", vision=True)
    desktop.models.append({"id": "synthetic-vision", "name": "synthetic", "path": str(folder)})
    monkeypatch.setattr(
        "stark_convert.desktop.ScoreWorker",
        lambda *args, **kwargs: pytest.fail("vision decision must be rejected before creating ScoreWorker"),
    )

    with pytest.raises(DesktopError, match="Chat mode only"):
        desktop.run(desktop.epoch, "synthetic-vision", "decision")
    assert desktop.worker is None
