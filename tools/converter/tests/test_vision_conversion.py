"""Synthetic end-to-end evidence for the conversion-only vision artifact boundary."""

import hashlib
import json
from dataclasses import replace

import pytest
from test_cli import source_tooling  # noqa: F401
from test_qwen35_vision import vision_fixture
from tokenizers import Tokenizer, models

from stark_convert import cli, vision_contract
from stark_convert.contracts import ConversionRequest
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header
from stark_convert.vision_contract import REPO, REVISION


@pytest.fixture
def vision_source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    config, _, _ = vision_fixture(root)
    tokens = {"vision_start": 10, "vision_end": 11, "image_pad": 12, "video_pad": 13}
    vocab = {f"[SYNTHETIC_{i}]": i for i in range(16)}
    for name, value in tokens.items():
        del vocab[f"[SYNTHETIC_{value}]"]
        vocab[f"<|{name}|>"] = value
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[SYNTHETIC_0]"))
    tokenizer.add_special_tokens([f"<|{name}|>" for name in tokens])
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": None,
                "added_tokens_decoder": {
                    str(value): {
                        "content": f"<|{name}|>",
                        "single_word": False,
                        "normalized": False,
                        "special": True,
                        "lstrip": False,
                        "rstrip": False,
                    }
                    for name, value in tokens.items()
                },
            }
        )
    )
    (root / "chat_template.jinja").write_text("{% for message in messages %}{{ message.content }}{% endfor %}")
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "patch_size": 2,
                "temporal_patch_size": 1,
                "merge_size": 1,
                "processor_class": "Qwen3VLProcessor",
                "image_mean": [0.5] * 3,
                "image_std": [0.5] * 3,
            }
        )
    )
    config.update(image_token_id=12, video_token_id=13, vision_start_token_id=10, vision_end_token_id=11)
    config["text_config"]["rope_parameters"].update(mrope_section=[1, 0, 0], mrope_interleaved=True)
    (root / "config.json").write_text(json.dumps(config))
    # Small generated tensors exercise orchestration. These deliberately do not
    # establish correctness or qualification for the official model weights.
    monkeypatch.setattr(vision_contract, "_TOKENS", tokens)
    monkeypatch.setattr(
        vision_contract,
        "_HASHES",
        {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in vision_contract._HASHES},
    )
    weights = root / "model.safetensors"
    monkeypatch.setattr(
        vision_contract,
        "_WEIGHT_FILES",
        {
            weights.name: ("weights", weights.stat().st_size, hashlib.sha256(weights.read_bytes()).hexdigest()),
        },
    )
    return root


@pytest.mark.parametrize("inline_template", [False, True])
def test_vision_conversion_publishes_complete_runtime_supported_artifact(
    vision_source,
    tmp_path,
    monkeypatch,
    inline_template,
):
    if inline_template:
        path = vision_source / "tokenizer_config.json"
        metadata = json.loads(path.read_text())
        metadata["chat_template"] = (vision_source / "chat_template.jinja").read_text()
        path.write_text(json.dumps(metadata))
        monkeypatch.setitem(vision_contract._HASHES, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    output = tmp_path / "output"
    receipt = cli.convert(vision_source, output, repo=REPO, artifact_mode="vision-language", chunk_bytes=4096)
    config = json.loads((output / "config.json").read_text())
    assert config["type"] == "Qwen35VisionLanguageModelConfig"
    assert config["artifact_contract"] == receipt["artifact_contract"]
    assert receipt["capability"]["native_loading"] == "runtime_blocked"
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["native_qualification"] is None
    assert receipt["artifact_contract"]["schema_version"] == 2
    assert receipt["artifact_contract"]["runtime_supported"] is True
    assert receipt["artifact_contract"]["qualification"] == "unverified"
    assert receipt["artifact_contract"]["transport"] == "mediaref-v1"
    assert receipt["artifact_contract"]["source"]["revision"] == REVISION
    assert receipt["transform_peak_bytes"] <= 4096
    assert all(source["repo"] == REPO for source in receipt["sources"])
    assert config["generation_config"]["stop_token_ids"] == [1]
    assert config["generation_config"]["temperature"] is None
    assert (
        config["decoder_config"]["transformer_config"]["layer_configs"][1]["rope_config"]["type"]
        == "Qwen35MRoPEConfig"
    )
    _, tensors = read_safetensors_header(output / "model.safetensors")
    assert any(name.startswith("vision.") for name in tensors)
    for name in ("preprocessor_config.json", "chat_template.jinja"):
        assert (output / name).read_bytes() == (vision_source / name).read_bytes()
    assert str(tmp_path) not in json.dumps(receipt)
    assert json.loads((output / "conversion.json").read_text()) == receipt


def test_vision_source_manifest_replay_preserves_mode_and_all_metadata(vision_source, tmp_path):
    snapshot = resolve_source(vision_source, repo=REPO, artifact_mode="vision-language")
    assert snapshot.revision == REVISION
    assert snapshot.manifest["schema_version"] == 3
    assert {record["role"] for record in snapshot.manifest["files"]} == {
        "model_config",
        "tokenizer",
        "tokenizer_config",
        "processor_config",
        "chat_template",
        "weights",
    }
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(
        REPO, source_manifest=manifest, artifact_mode="vision-language", offline=True, hub=object()
    )
    assert replay.receipt() == snapshot.receipt()
    with pytest.raises(ConversionError):
        resolve_source(REPO, source_manifest=manifest, offline=True, hub=object())


@pytest.mark.parametrize("mutation", ["missing_processor", "missing_template", "mode", "schema", "revision"])
def test_vision_manifest_rejects_missing_metadata_and_mode_downgrades(vision_source, tmp_path, mutation):
    snapshot = resolve_source(vision_source, repo=REPO, artifact_mode="vision-language")
    data = snapshot.manifest
    if mutation.startswith("missing_"):
        role = "processor_config" if mutation == "missing_processor" else "chat_template"
        data["files"] = [record for record in data["files"] if record["role"] != role]
    elif mutation == "mode":
        data["artifact_mode"] = "text"
    elif mutation == "schema":
        data["schema_version"] = 1
    else:
        data["primary_revision"] = "0" * 40
    manifest = tmp_path / "bad-manifest.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ConversionError):
        resolve_source(REPO, source_manifest=manifest, artifact_mode="vision-language", offline=True, hub=object())


@pytest.mark.parametrize(
    "change",
    [
        {"repo": "Qwen/Qwen3-0.6B"},
        {"dtype": "float16"},
        {"model_kind": "classifier"},
        {"metadata_profile": "hf-generic"},
        {"require_native_support": True},
    ],
)
def test_unsupported_vision_tuple_fails_before_source_resolution(monkeypatch, tmp_path, change):
    # A native-support request has no qualifying build in this scenario.
    monkeypatch.setattr("stark_convert.vision_runtime._records", lambda: [])
    request = replace(ConversionRequest(REPO, artifact_mode="vision-language"), **change)
    monkeypatch.setattr(cli, "resolve_source", lambda *a, **kw: pytest.fail("resolved an unsupported tuple"))
    with pytest.raises(UnsupportedConversion):
        cli.convert(
            tmp_path / "missing",
            tmp_path / "output",
            **{
                "repo": request.repo,
                "dtype": request.dtype,
                "model_kind": request.model_kind,
                "metadata_profile": request.metadata_profile,
                "require_native_support": request.require_native_support,
                "artifact_mode": request.artifact_mode,
            },
        )


def test_vision_request_cannot_use_text_source_snapshot(vision_source, tmp_path):
    snapshot = resolve_source(vision_source, repo=REPO, artifact_mode="vision-language")
    with pytest.raises(ConversionError, match="artifact modes disagree"):
        cli.convert(snapshot, tmp_path / "output", repo=REPO)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_probe_separates_conversion_from_runtime_support():
    cap = cli.capability(ConversionRequest(REPO, artifact_mode="vision-language"))
    assert cap.conversion == "implemented"
    assert cap.source_validation == "not_performed"
    assert cap.native_loading == "supported_layout_unverified_tuple"
    assert cap.reason is None


@pytest.mark.parametrize("replay", [False, True])
def test_cli_native_vision_without_records_fails_before_source_resolution(tmp_path, monkeypatch, capsys, replay):
    monkeypatch.setattr("stark_convert.vision_runtime._records", lambda: [])
    args = [
        "stark-convert",
        REPO,
        "--artifact-mode",
        "vision-language",
        "--require-native-support",
        "--output",
        str(tmp_path / "output"),
    ]
    if replay:
        args += ["--source-manifest", str(tmp_path / "manifest.json")]
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(cli, "resolve_source", lambda *a, **kw: pytest.fail("resolved a native vision request"))
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "error", "error_category": "UnsupportedConversion"}
    assert not (tmp_path / "output").exists()


def test_cli_vision_manifest_infers_repository_from_local_source(vision_source, tmp_path, monkeypatch, capsys):
    snapshot = resolve_source(vision_source, repo=REPO, artifact_mode="vision-language")
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "stark-convert",
            str(vision_source),
            "--artifact-mode",
            "vision-language",
            "--source-manifest",
            str(manifest),
            "--offline",
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert receipt["status"] == "ok"
    assert receipt["sources"] == [{"repo": REPO, "revision": REVISION}]
    assert receipt["artifact_mode"] == "vision-language"
    assert (output / "conversion.json").is_file()


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
def test_metadata_drift_fails_without_publishing(vision_source, tmp_path, filename):
    path = vision_source / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ConversionError, match="^Vision source does not match the pinned artifact$"):
        cli.convert(vision_source, tmp_path / "output", repo=REPO, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_same_shape_modified_weights_cannot_claim_official_checkpoint(vision_source, tmp_path):
    path = vision_source / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ConversionError, match="^Vision source does not match the pinned artifact$"):
        cli.convert(vision_source, tmp_path / "output", repo=REPO, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["bad_json", "bad_id", "duplicate_content", "conflicting_template"])
def test_invalid_tokenizer_metadata_is_content_free_and_atomic(vision_source, tmp_path, monkeypatch, mutation):
    path = vision_source / "tokenizer_config.json"
    metadata = json.loads(path.read_text())
    if mutation == "bad_json":
        path = vision_source / "tokenizer.json"
        path.write_text("private malformed tokenizer")
    elif mutation == "bad_id":
        metadata["added_tokens_decoder"]["10"]["id"] = 9
        path.write_text(json.dumps(metadata))
    elif mutation == "duplicate_content":
        metadata["added_tokens_decoder"]["11"]["content"] = "<|vision_start|>"
        path.write_text(json.dumps(metadata))
    else:
        metadata["chat_template"] = "private conflicting template"
        path.write_text(json.dumps(metadata))
    monkeypatch.setitem(vision_contract._HASHES, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ConversionError) as error:
        cli.convert(vision_source, tmp_path / "output", repo=REPO, artifact_mode="vision-language")
    assert str(error.value) in {
        "Vision source does not match the pinned artifact",
        "Conflicting vision chat-template metadata",
    }
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_vision_resolution_rejects_wrong_pin_before_download():
    from types import SimpleNamespace

    class WrongPinHub:
        def HfApi(self):
            return self

        def model_info(self, *, repo_id, revision):
            assert repo_id == REPO and revision == REVISION
            return SimpleNamespace(sha="0" * 40)

        def hf_hub_download(self, **kwargs):
            pytest.fail("downloaded from a wrong revision")

    with pytest.raises(ConversionError, match="different checkpoint"):
        resolve_source(REPO, artifact_mode="vision-language", hub=WrongPinHub())


def test_changed_metadata_copy_cannot_be_published(vision_source, tmp_path, monkeypatch):
    read = cli.read_regular_bytes

    def changed_copy(path):
        if path.name == "preprocessor_config.json":
            return b"private substituted metadata"
        return read(path)

    monkeypatch.setattr(cli, "read_regular_bytes", changed_copy)
    with pytest.raises(ConversionError, match="^Vision metadata changed during conversion$"):
        cli.convert(vision_source, tmp_path / "output", repo=REPO, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))
