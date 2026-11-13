"""Public configuration controls and synthetic tensors for reviewed converter gaps."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_dense_adapters import PLANNERS, dense_fixture
from test_qwen35 import qwen35_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.metadata import _generation_profile_value, _profile_generation_config
from stark_convert.registry import adapter_for_model, get_model
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

CONFIGS = json.loads((Path(__file__).parent / "fixtures" / "catalog_family_configs.json").read_text())["cases"]
SAFE_CONFIGS = [case for case in CONFIGS if case["config"]["model_type"] in {"llama", "qwen2"}]


@pytest.mark.parametrize("case", SAFE_CONFIGS, ids=lambda case: case["repo"])
def test_retained_dense_config_annotations_convert_with_synthetic_dimensions(tmp_path, case):
    source = tmp_path / "source"
    family = case["config"]["model_type"]
    small, _ = dense_fixture(source, family)
    dimensions = {
        key: small[key]
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        )
    }
    config = {**case["config"], **dimensions, "head_dim": 2}
    (source / "config.json").write_text(json.dumps(config))
    plan = PLANNERS[family](source)
    assert plan.decoder_config["transformer_config"]["model_dim"] == 8
    assert len(plan.decoder_config["transformer_config"]["layer_configs"]) == 2


@pytest.mark.parametrize(
    "family,changes,error",
    [
        ("qwen2", {"use_mrope": True}, UnsupportedConversion),
        ("qwen2", {"use_mrope": 0}, ConversionError),
        ("llama", {"embd_pdrop": 0.1}, UnsupportedConversion),
        ("llama", {"resid_pdrop": float("nan")}, ConversionError),
        ("llama", {"classifier_dropout": False}, ConversionError),
        ("llama", {"id2label": []}, ConversionError),
        ("llama", {"label2id": {"LABEL_0": True}}, ConversionError),
        ("llama", {"unreviewed_decoder_flag": False}, UnsupportedConversion),
    ],
)
def test_annotation_admission_keeps_semantic_and_malformed_variants_closed(tmp_path, family, changes, error):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, family)
    (source / "config.json").write_text(json.dumps({**config, **changes}))
    with pytest.raises(error):
        PLANNERS[family](source)


@pytest.mark.parametrize("repo", ["EssentialAI/rnj-1-instruct", "POLARIS-Project/Polaris-4B-Preview"])
def test_unrepresented_yarn_variants_are_not_advertised_as_convertible(repo):
    entry = get_model(repo)
    assert entry["adapter"] is None
    assert entry["metadata_profile"] is None
    assert entry["status"]["conversion"] == "planned"
    with pytest.raises(UnsupportedConversion):
        adapter_for_model(repo)


@pytest.mark.parametrize("size,penalty", [("2B", 1.15), ("4B", 1.10)])
@pytest.mark.parametrize("packing", [None, 4, 8])
def test_qwen35_size_profiles_preserve_source_temperature_and_matching_overrides(tmp_path, size, penalty, packing):
    repo = f"Qwen/Qwen3.5-{size}" if packing is None else f"mlx-community/Qwen3.5-{size}-MLX-{packing}bit"
    entry = get_model(repo)
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"do_sample": True, "temperature": 0.6, "top_k": 20, "eos_token_id": 1}))
    config = _profile_generation_config(tmp_path, entry, {"eos_token_id": 1}, lambda *_args, **_kwargs: path)
    assert config["temperature"] == 0.6
    assert config["top_k"] == 40
    assert config["repetition_penalty"] == penalty
    assert config["suffix_repetition_length"] == 1024
    assert config["stop_token_ids"] == [1]


def test_every_implemented_catalog_inline_generation_digest_resolves_and_rechecks_identity():
    from stark_convert.registry import models

    for entry in models():
        if entry["metadata_profile"] is None:
            continue
        for role in ("generation_config", "generation_params_overrides"):
            if role in entry["metadata"].get("inline_sources", {}):
                assert isinstance(_generation_profile_value(entry["metadata"], role), dict)


@pytest.mark.parametrize("layout,offset", [("hf", 1.0), ("mlx", None)])
def test_unquantized_qwen35_norm_storage_tracks_saved_convolution_layout(tmp_path, layout, offset):
    source = tmp_path / "source"
    _config, tensors = qwen35_fixture(source, dtype="F32")
    conv_name = "model.layers.0.linear_attn.conv1d.weight"
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for name, (dtype, value) in list(tensors.items()):
        if name.endswith(norm_suffixes):
            tensors[name] = dtype, np.linspace(-0.25, 0.25, value.size, dtype="<f4").reshape(value.shape)
    original_gain = tensors["model.norm.weight"][1] + np.float32(1.0)
    if layout == "mlx":
        dtype, conv = tensors[conv_name]
        tensors[conv_name] = dtype, np.moveaxis(conv, 2, 1).copy()
        for name, (dtype, value) in list(tensors.items()):
            if name.endswith(norm_suffixes):
                tensors[name] = dtype, value + np.float32(1.0)
    save_tensors(source / "model.safetensors", tensors)
    plan = build_qwen35_plan(source, dtype="float32")
    transformer = plan.decoder_config["transformer_config"]
    assert transformer["output_norm_config"]["scale_offset"] == offset
    assert all(layer["pre_mixer_norm_config"]["scale_offset"] == offset for layer in transformer["layer_configs"])
    attention = transformer["layer_configs"][1]["mixer_config"]
    assert attention["query_norm_config"]["scale_offset"] == offset
    assert attention["key_norm_config"]["scale_offset"] == offset
    # The direct DeltaNet norm never uses the zero-centered offset.
    assert transformer["layer_configs"][0]["mixer_config"]["norm_config"]["scale_offset"] is None
    output = tmp_path / "converted.safetensors"
    write_plan(plan, output)
    _, converted = read_safetensors_header(output)
    assert (
        tensor_bytes(converted["decoder.transformer.output_norm.scales"]) == tensors["model.norm.weight"][1].tobytes()
    )
    effective_gain = np.frombuffer(tensor_bytes(converted["decoder.transformer.output_norm.scales"]), dtype="<f4")
    assert np.array_equal(effective_gain + (offset or 0.0), original_gain)


def test_unquantized_qwen35_flat_convolution_does_not_guess_norm_storage(tmp_path):
    source = tmp_path / "source"
    _config, tensors = qwen35_fixture(source)
    name = "model.layers.0.linear_attn.conv1d.weight"
    dtype, conv = tensors[name]
    tensors[name] = dtype, conv.reshape(16, 4)
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="ambiguous normalization storage"):
        build_qwen35_plan(source)


def test_unquantized_qwen35_mixed_raw_and_baked_layers_are_rejected(tmp_path):
    source = tmp_path / "source"
    config, tensors = qwen35_fixture(source)
    config.update(num_hidden_layers=4, layer_types=["linear_attention", "full_attention"] * 2)
    for name, value in list(tensors.items()):
        if name.startswith("model.layers."):
            layer = int(name.split(".")[2])
            tensors[name.replace(f".layers.{layer}.", f".layers.{layer + 2}.")] = value
    name = "model.layers.2.linear_attn.conv1d.weight"
    dtype, conv = tensors[name]
    tensors[name] = dtype, np.moveaxis(conv, 2, 1).copy()
    (source / "config.json").write_text(json.dumps(config))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="mixes normalization storage"):
        build_qwen35_plan(source)
