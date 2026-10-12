"""Classifier roots, model-kind routing and private source replay."""

import json
from dataclasses import replace

import pytest
from test_generic import future_source
from test_modernbert import modernbert_fixture
from tokenizers import Tokenizer, models

from stark_convert import cli
from stark_convert.contracts import ConversionRequest
from stark_convert.metadata import SUPPORTED_METADATA_REPOS
from stark_convert.registry import get_model
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion

FUTURE = "future/classifier"
ROUTERS = ("trymirai/chat-moderation-router", "trymirai/health-related-router")


def test_classifier_sources_do_not_inherit_language_generation_defaults():
    for repo in ROUTERS:
        assert "generation_config" not in get_model(repo)["files"]
        assert repo not in SUPPORTED_METADATA_REPOS


def classifier_source(path, *, sharded=False, optional_stops=False):
    modernbert_fixture(path, sharded=sharded)
    vocab = {f"[SYNTHETIC_{index}]": index for index in range(16)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[SYNTHETIC_0]"))
    tokenizer.save(str(path / "tokenizer.json"))
    token_config = {"cls_token": "[SYNTHETIC_1]", "sep_token": "[SYNTHETIC_2]", "pad_token": "[SYNTHETIC_0]"}
    (path / "tokenizer_config.json").write_text(json.dumps(token_config))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    if optional_stops:
        config = json.loads((path / "config.json").read_text())
        for field in ("bos_token_id", "eos_token_id"):
            config.pop(field)
        (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("repo", ROUTERS)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_registered_classifier_root_kind_and_runtime_boundary(tmp_path, repo, dtype):
    source = classifier_source(tmp_path / "source", sharded=True)
    receipt = cli.convert(
        source, tmp_path / "output", repo=repo, model_kind="classifier", dtype=dtype, chunk_bytes=4096
    )
    root = json.loads((tmp_path / "output/config.json").read_text())
    assert set(root) == {"type", "classifier_config", "token_codec_config"}
    assert root["type"] == "ClassifierModelConfig"
    assert root["classifier_config"]["output_labels"] == ["label_0", "label_1"]
    assert receipt["architecture"] == "modernbert"
    assert receipt["capability"]["model_kind"] == "classifier"
    assert receipt["capability"]["native_loading"] == ("unverified" if dtype == "bfloat16" else "runtime_blocked")
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["native_qualification"] is None
    assert receipt["transform_peak_bytes"] <= 4096


@pytest.mark.parametrize("repo", [FUTURE, None])
@pytest.mark.parametrize("optional_stops", [False, True])
def test_future_classifier_live_and_offline_replay(tmp_path, repo, optional_stops):
    source = classifier_source(tmp_path / "source", optional_stops=optional_stops)
    # Classifiers bind unused optional files without interpreting a generation policy.
    (source / "generation_config.json").write_text("invalid unused generation metadata")
    selected = resolve_source(source, repo=repo, metadata_profile="hf-generic")
    manifest = tmp_path / "manifest.json"
    selected.write_manifest(manifest)
    first = cli.convert(selected, tmp_path / "first", repo=repo, model_kind="classifier", chunk_bytes=4096)
    replay = resolve_source(source, repo=repo, source_manifest=manifest, offline=True, hub=object())
    second = cli.convert(replay, tmp_path / "second", repo=repo, model_kind="classifier", chunk_bytes=4096)
    for key in ("source_manifest_sha256", "output_sha256", "config_sha256", "tokenizer_sha256"):
        assert first[key] == second[key]
    assert first["catalogued"] is False and first["native_qualification"] is None
    assert first["capability"]["native_loading"] == "unverified"
    assert first["metadata_profile"] == "hf-generic"
    assert "generation_config" not in json.loads((tmp_path / "first/config.json").read_text())


@pytest.mark.parametrize(
    "architecture,kind", [("modernbert", "language"), ("qwen3", "classifier"), ("modernbert", "tts")]
)
def test_inferred_generic_architecture_cannot_change_requested_root_kind(tmp_path, architecture, kind):
    source = (
        classifier_source(tmp_path / "source") if architecture == "modernbert" else future_source(tmp_path / "source")
    )
    selected = resolve_source(source, repo=FUTURE, metadata_profile="hf-generic")
    with pytest.raises(UnsupportedConversion):
        cli.convert(selected, tmp_path / "output", repo=FUTURE, model_kind=kind)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_generic_classifier_probe_has_explicit_kind_and_no_native_promotion():
    request = ConversionRequest(
        FUTURE, architecture="modernbert", metadata_profile="hf-generic", model_kind="classifier"
    )
    assert cli.capability(request).conversion == "implemented"
    assert cli.capability(request).native_loading == "unverified"
    assert cli.capability(replace(request, model_kind="language")).conversion == "unsupported"
    assert cli.capability(replace(request, architecture="qwen3")).conversion == "unsupported"


def test_generic_classifier_cannot_request_stored_native_qualification(tmp_path):
    source = classifier_source(tmp_path / "source")
    selected = resolve_source(source, repo=FUTURE, metadata_profile="hf-generic")
    with pytest.raises(UnsupportedConversion, match="recorded native qualification"):
        cli.convert(selected, tmp_path / "output", repo=FUTURE, model_kind="classifier", require_native_support=True)
    assert not (tmp_path / "output").exists()


def test_registered_classifier_cannot_override_metadata_profile(tmp_path):
    source = classifier_source(tmp_path / "source")
    with pytest.raises(UnsupportedConversion):
        resolve_source(source, repo=ROUTERS[0], metadata_profile="hf-generic")


def test_forged_offline_architecture_does_not_reinterpret_classifier(tmp_path):
    source = classifier_source(tmp_path / "source")
    selected = resolve_source(source, repo=FUTURE, metadata_profile="hf-generic")
    manifest = tmp_path / "manifest.json"
    selected.write_manifest(manifest)
    forged = json.loads(manifest.read_text())
    forged["conversion_profile"]["architecture"] = "qwen3"
    manifest.write_text(json.dumps(forged))
    with pytest.raises(ConversionError):
        resolve_source(source, repo=FUTURE, source_manifest=manifest, offline=True, hub=object())
