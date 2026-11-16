"""Desktop native-Decision routing and private attachment lifecycle controls."""

import io
import json
import math
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer, models, pre_tokenizers

from stark_convert.chat import ChatCancelled
from stark_convert.decider import SOURCE_REPO, SOURCE_REVISION
from stark_convert.decider_install import FAMILY_ID
from stark_convert.desktop import Desktop, DesktopError, model_valid
from stark_convert.model_catalog import family_catalog

PNG = b"\x89PNG\r\n\x1a\nsynthetic image fixture"


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stark_convert.desktop_decider.IsolatedTemplate",
        lambda source: SimpleNamespace(
            render_with_cancellation=lambda _cancelled, **arguments: (
                ImmutableSandboxedEnvironment().from_string(source).render(**arguments)
            )
        ),
    )
    path = tmp_path / "prepared-decider"
    path.mkdir()
    codes = ["A", "B", *(f"C{i}" for i in range(253))]
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{code: i + 1 for i, code in enumerate(codes)}}, "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(
        [
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
        ]
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "chat_template.jinja").write_text(
        "<|im_start|>system\n{{messages[0].content}}<|im_end|>\n<|im_start|>user\n"
        "{% for part in messages[1].content %}"
        "{% if part.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{part.text}}{% endif %}{% endfor %}"
    )
    config = {
        "type": "DeciderModelConfig",
        "decoder_config": {
            "vocab_size": 4000,
            "transformer_config": {
                "layer_configs": [
                    {
                        "mixer_config": {"type": "AttentionConfig", "is_causal": False},
                        "rope_config": {
                            "type": "Qwen35MRoPEConfig",
                            "mrope_section": [11, 11, 10],
                            "interleaved": True,
                        },
                    }
                ]
            },
        },
        "vision_config": {"hidden_size": 1024},
        "decision_config": {
            "codes": codes,
            "token_ids": list(range(1, 256)),
            "temperature": 2.0,
            "protocol_version": 4,
            "num_options": 255,
            "max_pixels": 1_638_400,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"synthetic native-loader fixture")
    return path


def fake_worker(tmp_path, behavior="normal"):
    path = tmp_path / f"decide-worker-{behavior}"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json,sys,time\nfrom pathlib import Path\n"
        "print(json.dumps({'ready':True,'mode':'decider','images':True,'max_images':1,"
        "'options':255,'max_input_tokens':4096,'mtp':{'supported':False},"
        "'prefix_cache':{'supported':False}}),flush=True)\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " if r['operation']=='reset_cache':\n"
        "  print(json.dumps({'ok':True,'cache_cleared':True}),flush=True);continue\n"
        f" if {behavior!r}=='wait':time.sleep(30)\n"
        f" if {behavior!r}=='fail':raise SystemExit(3)\n"
        " images=r.get('images',[])\n"
        " if images and (len(images)!=1 or not Path(images[0]).is_file() or "
        "Path(images[0]).read_bytes()!=b'\\x89PNG\\r\\n\\x1a\\nsynthetic image fixture'):\n"
        "  print(json.dumps({'ok':False,'error':'invalid_image_request'}),flush=True);continue\n"
        f" if {behavior!r}=='overflow' and images:\n"
        "  print(json.dumps({'ok':False,'error':'decider_context_overflow'}),flush=True);continue\n"
        " n=r['option_counts'][0]\n"
        " result={'ok':True,'logits':[[float(i*2) for i in range(n)]],"
        "'input_tokens':len(r['input_token_ids'])+(63 if images else 0),'cache_hit':False}\n"
        f" if {behavior!r}=='malformed':result['logits']=[[float('nan')]*n]\n"
        " print(json.dumps(result),flush=True)\n"
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def desktop(tmp_path, artifact):
    attachments = (tmp_path / "attachments").resolve()
    attachments.mkdir(mode=0o700)
    worker = fake_worker(tmp_path)
    instance = Desktop(
        tmp_path / "unused-generator",
        tmp_path / "library",
        io.StringIO(),
        timeout=2,
        decide_executable=worker,
        attachment_root=attachments,
    )
    instance.install(instance.epoch, "local", str(artifact))
    yield instance, attachments
    instance.stop(quiet=True)


def events(instance):
    return [json.loads(line) for line in instance.output.getvalue().splitlines()]


def finish(instance):
    thread = instance.job
    if thread is not None:
        thread.join(timeout=5)
        assert not thread.is_alive()


def run(instance):
    identifier = instance.models[0]["id"]
    instance.dispatch({"command": "run", "mode": "decision", "model_id": identifier})
    finish(instance)
    assert instance.worker is not None
    return identifier


def stage(root, identifier="a" * 32):
    path = root / identifier
    path.write_bytes(PNG)
    path.chmod(0o600)
    return {"id": identifier, "media_type": "image/png", "byte_length": len(PNG)}


def request(image=None, text="State", request_id="request-1"):
    payload = {
        "command": "decide",
        "text": text,
        "question": "Which option?",
        "answers": ["first", "second"],
        "request_id": request_id,
    }
    if image is not None:
        payload["image"] = image
    return payload


def test_prepared_local_import_and_decision_native_ready(desktop):
    instance, _ = desktop
    assert instance.models[0]["artifact_mode"] == "decision"
    assert instance.models[0]["name"] == "Decider Flash v1.3"
    run(instance)
    ready = next(x for x in events(instance) if x["event"] == "ready")
    assert ready["mode"] == "decision" and ready["backend"] == "decision-native"
    assert ready["decision_model"] and ready["images"] and ready["decision_images"]
    assert ready["image_attachment_transport"] == "private-file-v1"
    assert ready["max_context_tokens"] == 4096
    assert ready["image_limits"]["max_processed_pixels"] == 1_638_400
    assert not ready["tools"] and not ready["mtp"]["supported"]


def test_add_source_chat_default_imports_prepared_decider_as_native_decision(desktop, artifact):
    instance, _ = desktop
    instance.models.clear()
    instance.dispatch(
        {
            "command": "install",
            "kind": "local",
            "source": str(artifact),
            "support_mode": "text",
            "support_request_id": "gui-add-source",
        }
    )
    finish(instance)
    assert len(instance.models) == 1 and instance.models[0]["artifact_mode"] == "decision"
    assert any(x["event"] == "installed" and x.get("support_request_id") == "gui-add-source" for x in events(instance))
    assert not [x for x in events(instance) if x["event"] == "error"]


@pytest.mark.parametrize(
    "artifact_mode,support_mode", [("text", "text"), ("vision-language", "vision-language"), ("decision", "text")]
)
def test_explicit_conflicting_decider_install_modes_remain_rejected(desktop, artifact, artifact_mode, support_mode):
    instance, _ = desktop
    with pytest.raises(DesktopError):
        instance.install(instance.epoch, "local", str(artifact), artifact_mode, support_mode)


def test_decider_cannot_run_as_chat(desktop):
    instance, _ = desktop
    with pytest.raises(DesktopError, match="Decision mode only"):
        instance.run(instance.epoch, instance.models[0]["id"], "chat")
    assert instance.worker is None


def test_ordered_choices_use_checkpoint_temperature(desktop):
    instance, _ = desktop
    run(instance)
    instance.dispatch(request())
    finish(instance)
    reply = next(x for x in events(instance) if x["event"] == "decision")
    assert reply["request_id"] == "request-1"
    assert reply["answers"] == ["first", "second"]
    assert reply["choice_index"] == 1
    assert reply["probabilities"] == pytest.approx([1 / (1 + math.e), math.e / (1 + math.e)])
    assert reply["calibration"] == "checkpoint_temperature" and reply["temperature"] == 2.0


def test_image_only_state_is_claimed_and_worker_lease_is_cleaned(desktop):
    instance, attachments = desktop
    run(instance)
    media = instance.worker.media.root
    instance.dispatch(request(stage(attachments), text=""))
    finish(instance)
    assert not list(attachments.iterdir()) and not list(media.iterdir())
    replies = [x for x in events(instance) if x["event"] == "decision"]
    assert len(replies) == 1 and replies[0]["request_id"] == "request-1"
    assert [x["phase"] for x in events(instance) if x["event"] == "image_progress"] == ["preparing", "deciding"]


def test_validation_error_consumes_attachment_without_worker_call(desktop):
    instance, attachments = desktop
    run(instance)
    payload = request(stage(attachments))
    payload["answers"] = ["same", "same"]
    with pytest.raises(DesktopError, match="distinct"):
        instance.dispatch(payload)
    assert not list(attachments.iterdir())
    assert instance.worker.process.poll() is None


def test_reset_clears_unclaimed_attachment_and_keeps_worker(desktop):
    instance, attachments = desktop
    run(instance)
    stage(attachments)
    instance.dispatch({"command": "reset"})
    assert not list(attachments.iterdir())
    assert instance.worker.process.poll() is None


@pytest.mark.parametrize("behavior", ["fail", "malformed"])
def test_failed_native_decision_clears_worker_and_attachment(desktop, tmp_path, behavior):
    instance, attachments = desktop
    instance.decide_executable = fake_worker(tmp_path, behavior)
    run(instance)
    child, media = instance.worker.process, instance.worker.media.root
    instance.dispatch(request(stage(attachments)))
    finish(instance)
    assert child.poll() is not None and instance.worker is None
    assert not list(attachments.iterdir()) and not media.exists()
    assert not [x for x in events(instance) if x["event"] == "decision"]


def test_expanded_image_context_rejection_recovers_same_worker(desktop, tmp_path):
    instance, attachments = desktop
    instance.decide_executable = fake_worker(tmp_path, "overflow")
    run(instance)
    worker = instance.worker
    instance.dispatch(request(stage(attachments)))
    finish(instance)
    assert instance.worker is worker and not list(attachments.iterdir())
    assert not list(worker.media.root.iterdir())
    instance.dispatch(request(request_id="recovered"))
    finish(instance)
    assert [x["request_id"] for x in events(instance) if x["event"] == "decision"] == ["recovered"]


def test_stop_during_image_decision_reaps_child_and_suppresses_stale_reply(desktop, tmp_path):
    instance, attachments = desktop
    instance.decide_executable = fake_worker(tmp_path, "wait")
    run(instance)
    child, media = instance.worker.process, instance.worker.media.root
    instance.dispatch(request(stage(attachments)))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not list(media.iterdir()):
        time.sleep(0.01)
    assert list(media.iterdir())
    instance.stop(quiet=True)
    assert child.poll() is not None and instance.worker is None
    assert not media.exists() and not list(attachments.iterdir())
    assert not [x for x in events(instance) if x["event"] == "decision"]


def test_busy_decision_consumes_new_attachment_and_correlates_rejection(desktop, tmp_path):
    instance, attachments = desktop
    instance.decide_executable = fake_worker(tmp_path, "wait")
    run(instance)
    media = instance.worker.media.root
    instance.dispatch(request(stage(attachments)))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not list(media.iterdir()):
        time.sleep(0.01)
    instance.dispatch(request(stage(attachments, "b" * 32), request_id="busy-second"))
    assert not list(attachments.iterdir())
    assert any(x["event"] == "error" and x.get("request_id") == "busy-second" for x in events(instance))
    instance.stop(quiet=True)
    assert not media.exists()


def test_stop_interrupts_template_encoding_and_clears_claimed_image(desktop):
    instance, attachments = desktop
    run(instance)
    worker = instance.worker
    entered = threading.Event()

    class SlowTemplate:
        def render(self, **_arguments):
            entered.set()
            assert worker._decision_cancelled.wait(timeout=2)
            raise ChatCancelled("Request cancelled.")

    worker.encoder.template = SlowTemplate()
    instance.dispatch(request(stage(attachments)))
    assert entered.wait(timeout=2)
    instance.stop(quiet=True)
    assert worker.process.poll() is not None
    assert not list(attachments.iterdir()) and not worker.media
    assert not [x for x in events(instance) if x["event"] == "decision"]


def test_images_are_not_advertised_without_private_attachment_transport(tmp_path, artifact):
    worker = fake_worker(tmp_path)
    instance = Desktop(
        tmp_path / "unused-generator",
        tmp_path / "library-no-attachments",
        io.StringIO(),
        timeout=2,
        decide_executable=worker,
    )
    try:
        instance.install(instance.epoch, "local", str(artifact))
        run(instance)
        ready = next(x for x in events(instance) if x["event"] == "ready")
        assert ready["decision_model"] and not ready["decision_images"] and not ready["images"]
    finally:
        instance.stop(quiet=True)


def test_scalar_rope_decider_is_not_importable(artifact):
    config = json.loads((artifact / "config.json").read_text())
    config["decoder_config"]["transformer_config"]["layer_configs"][0]["rope_config"]["type"] = "UnscaledRoPEConfig"
    (artifact / "config.json").write_text(json.dumps(config))
    assert not model_valid(artifact)


@pytest.mark.parametrize("kind", ["local", "huggingface"])
@pytest.mark.parametrize("mode,purpose", [("decision", "decision"), (None, "text")])
def test_raw_decider_install_uses_pinned_native_helper(desktop, artifact, tmp_path, monkeypatch, kind, mode, purpose):
    instance, _ = desktop
    raw = tmp_path / "raw-decider"
    raw.mkdir()
    (raw / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5Model"]}))
    source = SOURCE_REPO if kind == "huggingface" else str(raw)
    calls = []
    monkeypatch.setattr("stark_convert.decider_install.is_decider_source", lambda *args, **kwargs: True)

    def preflight(command, environment, epoch):
        assert "stark_convert.decider_install" in command and "preflight" in command
        assert command[command.index("--kind") + 1] == kind
        assert command[command.index("--mode") + 1] == "decision"
        calls.append("preflight")
        return {
            "schema_version": 1,
            "status": "conditional",
            "mode": "decision",
            "message": "Synthetic pinned-source admission.",
            "source": {"kind": kind, "repo": SOURCE_REPO, "revision": SOURCE_REVISION},
        }

    def convert(command, environment, epoch):
        assert "stark_convert.decider_install" in command and "prepare" in command
        assert command[command.index("--kind") + 1] == kind
        assert command[command.index("--revision") + 1] == SOURCE_REVISION
        assert "--metadata-profile" not in command and "--dtype" not in command
        shutil.copytree(artifact, command[command.index("--output") + 1])
        calls.append("prepare")
        return 0, None

    monkeypatch.setattr(instance, "run_support_check", preflight)
    monkeypatch.setattr(instance, "run_install_support_check", preflight)
    monkeypatch.setattr(instance, "run_converter", convert)
    monkeypatch.setattr(instance, "run_download_converter", convert)
    instance.install(instance.epoch, kind, source, mode, purpose)
    assert calls == ["preflight", "prepare"]
    assert instance.models[-1]["artifact_mode"] == "decision"
    assert instance.models[-1]["name"] == "Decider Flash v1.3"


def test_decider_family_browses_its_pinned_native_decision_row(desktop, monkeypatch):
    instance, _ = desktop
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(instance, "run_catalog_check", lambda *_args: family_catalog(FAMILY_ID))
    instance.dispatch({"command": "catalog", "family_id": FAMILY_ID, "request_id": "catalog-decider"})
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not any(x["event"] == "model_catalog" for x in events(instance)):
        time.sleep(0.01)
    result = next(x for x in events(instance) if x["event"] == "model_catalog")
    assert result["report"]["status"] == "ready"
    row = result["report"]["items"][0]
    assert row["repo"] == SOURCE_REPO and row["revision"] == SOURCE_REVISION
    assert row["artifact_modes"] == ["decision"] and row["default_artifact_mode"] == "decision"
    assert row["downloadable"] and row["images"] and row["max_images"] == 1


def test_other_non_language_catalog_families_remain_rejected(desktop):
    instance, _ = desktop
    with pytest.raises(DesktopError, match="family to browse"):
        instance.dispatch({"command": "catalog", "family_id": "modernbert", "request_id": "catalog-other"})
