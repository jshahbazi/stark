"""Future checkpoints reuse adapters without borrowing model metadata or qualification."""

import json
from dataclasses import replace

import pytest
from test_cli import REPOS, source_fixture, source_tooling  # noqa: F401
from test_lfm2 import lfm2_fixture
from test_mlx import mlx_qwen_fixture
from test_mlx_lfm2 import mlx_lfm2_fixture
from test_mlx_qwen35 import mlx_qwen35_fixture
from test_qwen35 import qwen35_fixture
from test_sources import Hub
from tokenizers import Tokenizer, models

from stark_convert import cli
from stark_convert.contracts import ConversionRequest
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion

FUTURE = "future/checkpoint"


def future_source(path, architecture="qwen3"):
    if architecture in REPOS:
        return source_fixture(path, architecture)
    if architecture == "qwen3_5":
        qwen35_fixture(path, wrapper=True)
    else:
        lfm2_fixture(path)
    tokenizer = Tokenizer(models.WordLevel({f"[SYNTHETIC_{i}]": i for i in range(32)}, unk_token="[SYNTHETIC_0]"))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    return path


def snapshot(path, **kwargs):
    return resolve_source(path, repo=FUTURE, metadata_profile="hf-generic", **kwargs)


@pytest.mark.parametrize("architecture", [*REPOS, "qwen3_5", "lfm2"])
def test_future_models_reuse_every_supported_architecture(tmp_path, architecture):
    source = future_source(tmp_path / "source", architecture)
    receipt = cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE, chunk_bytes=4096)
    assert receipt["architecture"] == architecture
    assert receipt["metadata_profile"] == "hf-generic"
    assert receipt["catalogued"] is False
    assert receipt["native_qualification"] is None
    assert receipt["capability"]["native_loading"] == (
        "runtime_blocked" if architecture == "qwen3_5" else "unverified"
    )
    assert receipt["capability"]["source_validation"] == "complete"
    config = json.loads((tmp_path / "output/config.json").read_text())
    assert config["token_codec_config"]["reasoning_config"] is None
    assert config["token_codec_config"]["output_parser_regex"] is None
    assert config["token_codec_config"]["default_system_prompt"] is None


def test_generic_probe_needs_explicit_architecture_and_never_qualifies_source():
    request = ConversionRequest(FUTURE, metadata_profile="hf-generic")
    assert cli.capability(request).conversion == "unsupported"
    result = cli.capability(replace(request, architecture="qwen3"))
    assert result.conversion == "implemented"
    assert result.native_loading == "unverified"
    assert result.source_validation == "not_performed"
    assert cli.capability(ConversionRequest(REPOS["qwen3"], architecture="llama")).conversion == "unsupported"


@pytest.mark.parametrize("bits", [4, 8])
def test_future_mlx_checkpoint_uses_its_declared_packing(tmp_path, bits):
    source = tmp_path / "source"
    mlx_qwen_fixture(source, bits=bits)
    tokenizer = Tokenizer(models.WordLevel({f"[SYNTHETIC_{i}]": i for i in range(17)}, unk_token="[SYNTHETIC_0]"))
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    receipt = cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE, chunk_bytes=4096)
    assert receipt["architecture"] == "qwen3"
    assert receipt["native_qualification"] is None
    assert receipt["transform_peak_bytes"] <= 4096


@pytest.mark.parametrize("architecture", ["qwen3_5", "lfm2"])
@pytest.mark.parametrize("bits", [4, 8])
def test_future_hybrid_mlx_checkpoint_converts_and_replays_offline(tmp_path, architecture, bits):
    source = tmp_path / "source"
    if architecture == "qwen3_5":
        mlx_qwen35_fixture(source, bits=bits, wrapper=True, vl_wrapper=True, layout="grouped", sharded=True, mtp=True)
    else:
        mlx_lfm2_fixture(source, bits=bits, variant="flat", sharded=True)
    tokenizer = Tokenizer(models.WordLevel({f"[SYNTHETIC_{i}]": i for i in range(17)}, unk_token="[SYNTHETIC_0]"))
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (source / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    selected = snapshot(source)
    manifest = tmp_path / "manifest.json"
    selected.write_manifest(manifest)
    replay = resolve_source(FUTURE, source_manifest=manifest, offline=True, hub=object())
    receipt = cli.convert(replay, tmp_path / "output", repo=FUTURE, chunk_bytes=4096)
    assert receipt["architecture"] == architecture
    assert receipt["metadata_profile"] == "hf-generic"
    assert receipt["catalogued"] is False
    assert receipt["native_qualification"] is None
    assert receipt["capability"]["native_loading"] == (
        "runtime_blocked" if architecture == "qwen3_5" else "unverified"
    )
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["transform_peak_bytes"] <= 4096


@pytest.mark.parametrize("architecture", ["qwen3_5", "lfm2"])
@pytest.mark.parametrize("bits", [4, 8])
def test_hybrid_catalog_packing_must_match_config_before_conversion(tmp_path, architecture, bits):
    source = tmp_path / "source"
    if architecture == "qwen3_5":
        mlx_qwen35_fixture(source, bits=bits, wrapper=True)
        repo = f"mlx-community/Qwen3.5-0.8B-MLX-{12 - bits}bit"
        external = tmp_path / "external"
        external.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            (external / name).write_text("{}")
        selected = resolve_source(
            source, repo=repo, hub=Hub({"Qwen/Qwen3.5-0.8B": external, "Qwen/Qwen3.5-27B": external})
        )
    else:
        mlx_lfm2_fixture(source, bits=bits)
        repo = f"mlx-community/LFM2-350M-{12 - bits}bit"
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (source / name).write_text("{}")
        selected = resolve_source(source, repo=repo)
    with pytest.raises(ConversionError, match="packing disagrees"):
        cli.plan_conversion(selected, ConversionRequest(repo))
    assert not (tmp_path / "output").exists()


def test_generic_remote_pins_before_download_and_replays_without_network(tmp_path):
    source = future_source(tmp_path / "source")
    hub = Hub({FUTURE: source})
    selected = resolve_source(FUTURE, metadata_profile="hf-generic", hub=hub)
    assert hub.events[0][0] == "pin"
    manifest = tmp_path / "manifest.json"
    selected.write_manifest(manifest)
    assert selected.manifest["schema_version"] == 2
    assert manifest.stat().st_mode & 0o777 == 0o600
    replay = resolve_source(FUTURE, source_manifest=manifest, offline=True, hub=object())
    assert replay.receipt() == selected.receipt()
    assert replay.conversion_profile == {"architecture": "qwen3", "metadata_profile": "hf-generic"}
    receipt = cli.convert(replay, tmp_path / "output", repo=FUTURE)
    assert receipt["capability"]["native_loading"] == "unverified"


@pytest.mark.parametrize("optional", ["generation_config.json", "chat_template.jinja"])
def test_generic_optional_sidecar_presence_is_pinned(tmp_path, optional):
    source = future_source(tmp_path / "source")
    (source / optional).unlink(missing_ok=True)
    selected = snapshot(source)
    manifest = tmp_path / "manifest.json"
    selected.write_manifest(manifest)
    (source / optional).write_text("{}")
    with pytest.raises(ConversionError):
        selected.assert_unchanged()
    with pytest.raises(ConversionError):
        resolve_source(FUTURE, source_manifest=manifest, offline=True)


def test_generic_can_use_model_eos_without_generation_sidecar(tmp_path):
    source = future_source(tmp_path / "source")
    (source / "generation_config.json").unlink()
    receipt = cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE)
    assert receipt["capability"]["source_validation"] == "complete"


@pytest.mark.parametrize("template", ["{{ messages }}", "different synthetic template"])
def test_generic_duplicate_templates_must_match(tmp_path, template):
    source = future_source(tmp_path / "source")
    (source / "chat_template.jinja").write_text(template)
    if template == "{{ messages }}":
        cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE)
    else:
        with pytest.raises(ConversionError):
            cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE)
        assert not (tmp_path / "output").exists()
        assert not list(tmp_path.glob(".stark-convert-*"))


@pytest.mark.parametrize("eos", [None, [], True])
def test_generic_invalid_stop_metadata_fails_without_publication(tmp_path, eos):
    source = future_source(tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    config["eos_token_id"] = eos
    (source / "config.json").write_text(json.dumps(config))
    (source / "generation_config.json").write_text("{}")
    with pytest.raises(ConversionError):
        cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE)
    assert not (tmp_path / "output").exists()


def test_generic_requires_native_qualification_and_known_profiles_remain_authoritative(tmp_path):
    source = future_source(tmp_path / "source")
    with pytest.raises(UnsupportedConversion):
        cli.convert(snapshot(source), tmp_path / "output", repo=FUTURE, require_native_support=True)
    with pytest.raises(UnsupportedConversion):
        resolve_source(source, repo=REPOS["qwen3"], metadata_profile="hf-generic")


def test_generic_architecture_assertion_rejects_mismatch(tmp_path):
    source = future_source(tmp_path / "source")
    with pytest.raises(ConversionError):
        snapshot(source, architecture="llama")
    selected = snapshot(source)
    with pytest.raises(ConversionError):
        cli.convert(selected, tmp_path / "output", repo=FUTURE, architecture="llama")


@pytest.mark.parametrize("mutation", ["schema", "known_repo", "external", "metadata_path", "profile", "role"])
def test_generic_replay_rejects_spoofing(tmp_path, mutation):
    source = future_source(tmp_path / "source")
    selected = snapshot(source)
    manifest = tmp_path / "manifest.json"
    selected.write_manifest(manifest)
    data = json.loads(manifest.read_text())
    if mutation == "schema":
        data["schema_version"] = True
    elif mutation == "known_repo":
        data["primary_repo"] = REPOS["qwen3"]
    elif mutation == "external":
        data["files"][0]["repo"] = "external/model"
    elif mutation == "metadata_path":
        record = next(item for item in data["files"] if item["role"] == "tokenizer_config")
        other = tmp_path / "other.json"
        other.write_bytes((source / "tokenizer_config.json").read_bytes())
        record["path"] = str(other)
    elif mutation == "profile":
        data["conversion_profile"]["architecture"] = "llama"
    else:
        data["files"][0]["role"] = "unreviewed"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ConversionError):
        resolve_source(FUTURE, source_manifest=manifest, offline=True)
