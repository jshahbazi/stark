import hashlib
import json
from copy import deepcopy

import pytest

from stark_convert import glm_vision


def config():
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "vocab_size": 154880,
            "num_nextn_predict_layers": 1,
        },
        "vision_config": dict(glm_vision._VISION),
        **glm_vision._TOKENS,
    }


def plan(value):
    return glm_vision.plan_image_source(
        json.dumps(value).encode(),
        metadata_repo=glm_vision.METADATA_REPO,
        metadata_revision=glm_vision.METADATA_REVISION,
    )


def test_glm_source_plan_retains_head_declaration_without_granting_runtime_capability():
    value = config()
    original = deepcopy(value)
    result = plan(value)
    assert result["declared_mtp_layers"] == 1
    assert result["image_execution"] == "not_checked"
    assert result["weight_admission"] == "not_checked"
    assert result["required_vision_sidecar"]["filename"] == "mmproj-BF16.gguf"
    assert result["required_vision_sidecar"]["size"] == 1_164_010_176
    assert result["config_sha256"] == hashlib.sha256(json.dumps(value).encode()).hexdigest()
    assert value == original
    result["required_vision_sidecar"]["sha256"] = "0" * 64
    assert plan(value)["required_vision_sidecar"]["sha256"] == glm_vision.SIDECAR["sha256"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "model_type", "qwen4_exp"),
        (None, "image_token_id", 154855),
        (None, "video_end_token_id", None),
        ("vision_config", "patch_size", 16),
        ("vision_config", "depth", 27),
        ("vision_config", "attention_bias", 1),
        ("vision_config", "swiglu_limit", 0.0),
        ("text_config", "hidden_size", 1024),
        ("text_config", "num_nextn_predict_layers", True),
        ("text_config", "num_nextn_predict_layers", 0),
    ],
)
def test_glm_source_plan_rejects_changed_tower_tokens_and_head(section, key, value):
    candidate = config()
    (candidate if section is None else candidate[section])[key] = value
    with pytest.raises(ValueError, match="unsupported GLM"):
        plan(candidate)


@pytest.mark.parametrize("payload", [b"{}", b"[]", b'{"a":1,"a":2}', b"\xff", b"x" * (1024 * 1024 + 1)])
def test_glm_source_plan_rejects_ambiguous_or_excessive_metadata(payload):
    with pytest.raises(ValueError):
        glm_vision.plan_image_source(
            payload,
            metadata_repo=glm_vision.METADATA_REPO,
            metadata_revision=glm_vision.METADATA_REVISION,
        )


def test_glm_source_plan_does_not_accept_another_revision():
    with pytest.raises(ValueError, match="metadata source"):
        glm_vision.plan_image_source(
            json.dumps(config()).encode(), metadata_repo=glm_vision.METADATA_REPO, metadata_revision="f" * 40
        )


@pytest.fixture
def local_glm_source(tmp_path, monkeypatch):
    from stark_convert import hybrid_prepare

    directory = tmp_path / "model"
    directory.mkdir()
    records = {}
    for index, name in enumerate(glm_vision.DECODER_FILES):
        payload = bytes([index + 1])
        (directory / name).write_bytes(payload)
        records[name] = {"size": 1, "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(glm_vision, "DECODER_FILES", records)
    monkeypatch.setitem(hybrid_prepare.SOURCES, "glm53-q2", {**hybrid_prepare.SOURCES["glm53-q2"], "weight_bytes": 4})
    value = config()
    value["text_config"]["eos_token_id"] = 154820
    (directory / "config.json").write_text(json.dumps(value))
    (directory / "tokenizer.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0]['content'] }}"}))
    hybrid_prepare.finish_metadata(directory, "glm53-q2")
    monkeypatch.setattr(
        glm_vision, "CONFIG_SHA256", hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        glm_vision, "TOKENIZER_SHA256", hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        glm_vision,
        "TEMPLATE_SHA256",
        hashlib.sha256(
            json.loads((directory / "config.json").read_bytes())["token_codec_config"]["prompt_template"].encode()
        ).hexdigest(),
    )
    sidecar = tmp_path / "mmproj-BF16.gguf"
    sidecar.write_bytes(b"tower")
    monkeypatch.setattr(
        glm_vision, "SIDECAR", {**glm_vision.SIDECAR, "size": 5, "sha256": hashlib.sha256(b"tower").hexdigest()}
    )
    return directory, sidecar


def test_glm_image_upgrade_preserves_decoder_and_shares_sidecar_outside_split_set(local_glm_source):
    directory, sidecar = local_glm_source
    before = {path.name: (path.stat().st_ino, path.read_bytes()) for path in directory.iterdir() if path.is_file()}
    value = glm_vision.prepare_image_artifact(directory, sidecar)
    assert value["schema_version"] == 7
    assert glm_vision.prepared_image_identity(directory) == value
    assert (directory / glm_vision.SIDECAR_PATH).stat().st_ino == sidecar.stat().st_ino
    assert len(list(directory.glob("*.gguf"))) == 4
    assert len(value["weight_files"]) == 5
    for name, expected in before.items():
        path = directory / name
        assert (path.stat().st_ino, path.read_bytes()) == expected
    assert glm_vision.prepare_image_artifact(directory, sidecar) == value


def test_glm_image_prepare_downloads_only_pinned_nested_sidecar(local_glm_source, monkeypatch):
    from stark_convert import hybrid_prepare

    directory, sidecar = local_glm_source
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        (kwargs["local_dir"] / glm_vision.SIDECAR["filename"]).write_bytes(sidecar.read_bytes())

    monkeypatch.setattr(hybrid_prepare, "snapshot_download", download)
    value = hybrid_prepare._prepare_images(directory, "glm53-q2", offline=True)
    assert glm_vision.prepared_image_identity(directory) == value
    assert len(list(directory.glob("*.gguf"))) == 4
    assert calls == [
        (
            glm_vision.SIDECAR["repo"],
            {
                "revision": glm_vision.SIDECAR["revision"],
                "allow_patterns": [glm_vision.SIDECAR["filename"]],
                "local_dir": directory / "vision",
                "local_files_only": True,
                "max_workers": 1,
            },
        )
    ]


def test_glm_image_prepare_local_sidecar_has_no_network_fallback(local_glm_source, monkeypatch):
    from stark_convert import hybrid_prepare

    directory, sidecar = local_glm_source
    monkeypatch.setattr(
        hybrid_prepare, "snapshot_download", lambda *args, **kwargs: pytest.fail("unexpected download")
    )
    value = hybrid_prepare._prepare_images(directory, "glm53-q2", sidecar=sidecar)
    assert glm_vision.prepared_image_identity(directory) == value
    (directory / glm_vision.SIDECAR_PATH).unlink()
    (directory / glm_vision.SIDECAR_PATH).symlink_to(sidecar)
    with pytest.raises(ValueError, match="sidecar file"):
        hybrid_prepare._prepare_images(directory, "glm53-q2")


@pytest.mark.parametrize("mutation", ["decoder", "sidecar", "processor", "template", "symlink"])
def test_glm_image_upgrade_rejects_payload_or_metadata_changes(local_glm_source, mutation):
    directory, sidecar = local_glm_source
    glm_vision.prepare_image_artifact(directory, sidecar)
    if mutation == "decoder":
        (directory / next(iter(glm_vision.DECODER_FILES))).write_bytes(b"x")
    elif mutation == "sidecar":
        sidecar.write_bytes(b"wrong")
    elif mutation == "processor":
        (directory / "preprocessor_config.json").write_text("{}")
    elif mutation == "template":
        (directory / "chat_template.jinja").write_text("changed")
    else:
        (directory / "vision").rename(directory / "saved-vision")
        (directory / "vision").symlink_to(directory / "saved-vision", target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        glm_vision.prepare_image_artifact(directory, sidecar)
    if mutation not in {"decoder", "sidecar"}:
        assert glm_vision.prepared_image_identity(directory) is None


@pytest.mark.parametrize("mutation", ["unknown_key", "schema_bool", "weight_size_float", "weight_path"])
def test_glm_manifest_inspection_rejects_ambiguous_contracts(local_glm_source, mutation):
    directory, sidecar = local_glm_source
    value = glm_vision.prepare_image_artifact(directory, sidecar)
    if mutation == "unknown_key":
        value["execute_python"] = True
    elif mutation == "schema_bool":
        value["schema_version"] = True
    elif mutation == "weight_size_float":
        value["weight_files"][next(iter(glm_vision.DECODER_FILES))]["size"] = 1.0
    else:
        value["weight_files"]["vision/../mmproj-BF16.gguf"] = value["weight_files"].pop(glm_vision.SIDECAR_PATH)
    (directory / glm_vision.MANIFEST).write_text(json.dumps(value))
    assert glm_vision.prepared_image_identity(directory) is None


def test_glm_image_upgrade_keeps_native_image_config_checks_after_hash_rebinding(local_glm_source):
    directory, sidecar = local_glm_source
    glm_vision.prepare_image_artifact(directory, sidecar)
    value = json.loads((directory / "config.json").read_bytes())
    value["vision_config"]["patch_size"] = 16
    encoded = json.dumps(value).encode()
    (directory / "config.json").write_bytes(encoded)
    source = json.loads((directory / "native-source.json").read_bytes())
    source["config_sha256"] = hashlib.sha256(encoded).hexdigest()
    (directory / "native-source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="vision profile"):
        glm_vision.prepare_image_artifact(directory, sidecar)
    assert glm_vision.prepared_image_identity(directory) is None


@pytest.mark.parametrize("mutation", ["text_graph", "tokenizer", "template"])
def test_glm_finished_profile_rejects_coherently_rehashed_alternate_metadata(local_glm_source, mutation):
    directory, sidecar = local_glm_source
    glm_vision.prepare_image_artifact(directory, sidecar)
    value = json.loads((directory / "config.json").read_bytes())
    if mutation == "text_graph":
        value["text_config"]["num_hidden_layers"] = 1
    elif mutation == "tokenizer":
        (directory / "tokenizer.json").write_text('{"changed":true}')
    else:
        value["token_codec_config"]["prompt_template"] = "changed"
    encoded = json.dumps(value).encode()
    (directory / "config.json").write_bytes(encoded)
    source = json.loads((directory / "native-source.json").read_bytes())
    source["config_sha256"] = hashlib.sha256(encoded).hexdigest()
    (directory / "native-source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="finished source identity"):
        glm_vision.prepare_image_artifact(directory, sidecar)
    assert glm_vision.prepared_image_identity(directory) is None
