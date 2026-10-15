"""Pinned public Empero config/header coverage without storing model payloads.

The fixture records public shape descriptors and sidecar hashes. Tokenizer and
template behavior use synthetic metadata; no prompt or vocabulary is copied.
"""

import hashlib
import json
import runpy
from dataclasses import asdict
from pathlib import Path

import pytest
from test_cli import source_tooling  # noqa: F401
from test_qwen35 import qwen35_fixture
from tokenizers import Tokenizer, models

from stark_convert import cli
from stark_convert.adapters import qwen35
from stark_convert.contracts import ConversionRequest
from stark_convert.metadata import SUPPORTED_METADATA_REPOS, write_metadata
from stark_convert.qualifications import qualification_for
from stark_convert.registry import adapter_for_model, get_model, load_catalog, required_files
from stark_convert.streaming import ConversionError, TensorInfo, UnsupportedConversion

REPO = "empero-ai/Qwen3.8-4B-Distill"
FIXTURE = Path(__file__).parent / "fixtures" / "empero_qwen38_distill.json"


def public_fixture():
    fixture = json.loads(FIXTURE.read_text())
    for role in ("config", "generation_config"):
        fixture[role] = json.loads(FIXTURE.with_name(f"empero_qwen38_distill_{role}.json").read_text())
    return fixture


def synthetic_metadata(source: Path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}", "bos_token": None, "eos_token": "[EOS]"})
    )
    (source / "generation_config.json").write_text(json.dumps(public_fixture()["generation_config"]))


def test_empero_is_an_attributed_extension_with_its_own_sidecars():
    entry = get_model(REPO)
    assert entry["catalog_origin"] == "stark_extension"
    assert entry["profile_reference"] == "Qwen/Qwen3.5-4B"
    assert entry["profile_source"] == public_fixture()["source"]
    assert entry["profile_source"]["revision"] == "c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e"
    assert adapter_for_model(REPO, "qwen3_5") == "qwen3_5"
    assert REPO in SUPPORTED_METADATA_REPOS
    assert {rule["repo"] for rule in required_files(entry)} == {REPO}
    assert {rule["filename"] for rule in required_files(entry)} == {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert entry["metadata"]["inline_sources"] == {}
    assert entry["status"] == {
        "conversion": "implemented",
        "runtime": "not_assessed",
        "access": "not_assessed",
        "qualification": "pending",
    }
    assert not any(entry["repo"] == REPO for entry in load_catalog()["models"])


def test_exporter_reproduces_the_packaged_extension():
    script = Path(__file__).parents[1] / "scripts" / "export_conversion_catalog.py"
    exporter = runpy.run_path(str(script))
    assert exporter["qwen38_distill_extension"](load_catalog()["models"]) == get_model(REPO)


def test_actual_sidecar_fixtures_match_the_reviewed_source_hashes():
    source = public_fixture()["source"]
    for role in ("config", "generation_config"):
        content = FIXTURE.with_name(f"empero_qwen38_distill_{role}.json").read_bytes()
        assert hashlib.sha256(content).hexdigest() == source["files_sha256"][f"{role}.json"]


def test_actual_4b_config_and_every_public_tensor_header_plan_without_payloads(tmp_path, monkeypatch):
    fixture = public_fixture()
    (tmp_path / "config.json").write_text(json.dumps(fixture["config"]))
    descriptors = fixture["tensor_descriptors"]
    total_bytes = max(info["data_offsets"][1] for info in descriptors.values())
    tensors = {
        name: TensorInfo(
            tmp_path / "header-only.safetensors",
            info["dtype"],
            tuple(info["shape"]),
            info["data_offsets"][0],
            info["data_offsets"][1] - info["data_offsets"][0],
            (0, 0, total_bytes, 0),
        )
        for name, info in descriptors.items()
    }
    monkeypatch.setattr(qwen35, "load_checkpoint", lambda source: tensors)
    plan = qwen35.build_qwen35_plan(tmp_path)
    assert plan.source_tensors == 738
    assert plan.input_bytes == total_bytes == 9_319_730_176
    decoder = plan.decoder_config
    assert decoder["vocab_size"] == 248320
    assert decoder["embedding_config"]["type"] == "TiedEmbeddingConfig"
    transformer = decoder["transformer_config"]
    assert transformer["model_dim"] == 2560 and transformer["hidden_dim"] == 9216
    layers = transformer["layer_configs"]
    assert len(layers) == 32
    assert sum(layer["mixer_config"]["type"] == "DeltaNetConfig" for layer in layers) == 24
    attention = layers[3]["mixer_config"]
    assert (attention["num_heads"], attention["num_groups"], attention["head_dim"]) == (16, 4, 256)
    delta = layers[0]["mixer_config"]
    assert (delta["num_heads"], delta["num_groups"], delta["head_dim"], delta["value_head_dim"]) == (32, 16, 128, 128)
    assert layers[3]["rope_config"]["head_dim"] == 64
    operations = {operation.name: operation for operation in plan.operations}
    assert operations["decoder.transformer.layers.0.mixer.in_proj.weights.weights"].shape == (12352, 2560)
    assert operations["decoder.transformer.layers.3.mixer.qkvg_projection.weights.weights"].shape == (10240, 2560)
    assert operations["decoder.embedding.embedding.weights"].shape == (248320, 2560)
    consumed = {info for operation in plan.operations for info in operation.sources}
    ignored = [info for name, info in tensors.items() if name.startswith(("model.visual.", "mtp."))]
    assert all(not (aux.offset <= info.offset < aux.offset + aux.nbytes) for aux in ignored for info in consumed)
    # Auxiliary towers remain in verification even though this is text import.
    assert any(name.startswith("model.visual.") for name in tensors)
    assert any(name.startswith("mtp.") for name in tensors)
    assert set(plan.input_tensors) == set(tensors.values())
    tensors["unexpected.weight"] = next(iter(tensors.values()))
    with pytest.raises(ConversionError, match="unexpected tensors"):
        qwen35.build_qwen35_plan(tmp_path)


def test_own_generation_stops_and_thinking_profile_do_not_inherit_sampling_defaults(tmp_path):
    fixture = public_fixture()
    synthetic_metadata(tmp_path)
    result = write_metadata(tmp_path, tmp_path / "output", {**fixture["config"], "_stark_model_repo": REPO})
    assert result["generation_config"]["stop_token_ids"] == [248044, 248046]
    assert all(value is None for key, value in result["generation_config"].items() if key != "stop_token_ids")
    codec = result["token_codec_config"]
    assert codec["output_parser_regex"] is not None
    assert codec["end_of_thinking_tag"] == "\n</think>"
    assert codec["reasoning_config"] == {
        "default_reasoning_effort": "medium",
        "field_name": "enable_thinking",
        "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
    }
    summary = fixture["metadata_summary"]
    assert summary["inline_template_matches_sidecar"] is True
    assert summary["uses_enable_thinking"] is True and summary["has_thinking_end"] is True
    assert summary["thinking_enabled_when_unspecified"] is True
    assert summary["inline_template_sha256"] == fixture["source"]["files_sha256"]["chat_template.jinja"]


@pytest.mark.parametrize("text_only", [False, True])
def test_named_profile_converts_without_claiming_native_qualification(tmp_path, source_tooling, text_only):  # noqa: F811
    source = tmp_path / "source"
    config, _ = qwen35_fixture(source, wrapper=not text_only, mtp=True)
    if text_only:
        config["model_type"] = "qwen3_5_text"
        (source / "config.json").write_text(json.dumps(config))
    synthetic_metadata(source)
    output = tmp_path / "converted"
    receipt = cli.convert(source, output, repo=REPO, chunk_bytes=4096)
    assert receipt["capability"]["conversion"] == "implemented"
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["capability"]["native_loading"] == "runtime_blocked"
    assert receipt["capability"]["reason"] == "native_delta_net_dimensions"
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["native_qualification"] is None
    assert qualification_for(REPO, "bfloat16", receipt) is None
    capability = asdict(cli.capability(ConversionRequest(REPO)))
    assert capability["source_validation"] == "not_performed"
    assert capability["native_loading"] == capability["execution"] == "unverified"
    with pytest.raises(UnsupportedConversion, match="No recorded named-build native qualification"):
        cli.convert(source, tmp_path / "required-native", repo=REPO, chunk_bytes=4096, require_native_support=True)
    assert not (tmp_path / "required-native").exists()
