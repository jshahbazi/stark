"""Packed decoder and retained BF16 tower conversion, using synthetic payloads."""

import json

import numpy as np
import pytest
from test_cli import source_tooling  # noqa: F401
from test_mlx_qwen35 import mlx_qwen35_fixture
from test_streaming import save_tensors, tensor_bytes
from test_vision_family import FINE_TUNE, pinned_vision_source  # noqa: F401
from test_vision_family import family_source as _family_source

from stark_convert import cli, vision_family
from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.streaming import ConversionError, read_safetensors_header

family_source = _family_source


def affine_image_source(family_source, tmp_path, monkeypatch, vl_wrapper):
    original = json.loads((family_source / "config.json").read_text())
    _, old = read_safetensors_header(family_source / "model.safetensors")
    source = tmp_path / "affine"
    config, tensors = mlx_qwen35_fixture(source, wrapper=True, vl_wrapper=vl_wrapper, group_size=64)
    config["vision_config"] = original["vision_config"]
    config["vision_config"]["out_hidden_size"] = 128
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        config[key] = original[key]
    config["text_config"]["rope_parameters"].update(mrope_section=[6, 5, 5], mrope_interleaved=True)
    prefix = "vision_tower." if vl_wrapper else "model.visual."
    tensors = {key: value for key, value in tensors.items() if not key.startswith(prefix)}
    for key, info in old.items():
        if key.startswith("model.visual."):
            shape = info.shape
            if key.endswith("merger.linear_fc2.weight"):
                shape = (128, shape[1])
            elif key.endswith("merger.linear_fc2.bias"):
                shape = (128,)
            tensors[prefix + key.removeprefix("model.visual.")] = (
                "BF16",
                np.arange(np.prod(shape), dtype="<u2").reshape(shape).transpose(0, 2, 3, 4, 1).copy()
                if vl_wrapper and key.endswith("patch_embed.proj.weight")
                else np.arange(np.prod(shape), dtype="<u2").reshape(shape),
            )
    save_tensors(source / "model.safetensors", tensors)
    (source / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json"):
        (source / name).write_bytes((family_source / name).read_bytes())
    monkeypatch.setattr(vision_family, "_ATTENTION_HEAD_DIM", 32)
    monkeypatch.setattr(vision_family, "_DELTA_HEAD_DIM", 32)
    monkeypatch.setattr(vision_family, "_ROPE", config["text_config"]["rope_parameters"])
    return source, config, tensors


@pytest.mark.parametrize("vl_wrapper", [False, True])
def test_affine_image_artifact_keeps_decoder_codes_and_complete_bf16_tower(
    family_source, tmp_path, monkeypatch, vl_wrapper
):
    source, _config, tensors = affine_image_source(family_source, tmp_path, monkeypatch, vl_wrapper)
    prefix = "vision_tower." if vl_wrapper else "model.visual."
    output = tmp_path / "converted"
    receipt = cli.convert(source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["family"] == {"profile": vision_family.AFFINE_FAMILY_PROFILE, "version": 3}
    published = json.loads((output / "config.json").read_text())
    assert published["decoder_config"]["transformer_config"]["output_norm_config"]["scale_offset"] is None
    _, infos = read_safetensors_header(output / "model.safetensors")
    assert infos["decoder.embedding.input_embedding.weights"].dtype == "U8"
    vision = {name: info for name, info in infos.items() if name.startswith("vision.")}
    expected = {
        "vision." + name.removeprefix(prefix): value for name, value in tensors.items() if name.startswith(prefix)
    }
    assert set(vision) == set(expected)
    for name, (_, array) in expected.items():
        if vl_wrapper and name.endswith("patch_embed.proj.weight"):
            array = array.transpose(0, 4, 1, 2, 3).copy()
        assert tensor_bytes(vision[name]) == array.tobytes()
    assert all(info.dtype == "BF16" for info in vision.values())
    assert not any(name.startswith("mtp.") for name in infos)


def test_moe_image_conversion_keeps_sparse_banks_and_a_separate_shared_expert(family_source, tmp_path, monkeypatch):
    source, config, tensors = affine_image_source(family_source, tmp_path, monkeypatch, True)
    config["model_type"] = "qwen3_5_moe"
    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    config["vision_config"]["model_type"] = "qwen3_5_moe"
    text = config["text_config"]
    text.update(
        model_type="qwen3_5_moe_text",
        architectures=["Qwen3_5MoeForCausalLM"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )
    text.pop("intermediate_size")
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}.mlp"
        for name in ("up_proj", "gate_proj", "down_proj"):
            for plane in ("weight", "scales", "biases"):
                key = f"{prefix}.{name}.{plane}"
                dtype, array = tensors.pop(key)
                tensors[f"{prefix}.shared_expert.{name}.{plane}"] = (dtype, array)
                tensors[f"{prefix}.switch_mlp.{name}.{plane}"] = (dtype, np.stack([array.copy() for _ in range(4)]))
        for name, rows in (("gate", 4), ("shared_expert_gate", 1)):
            for plane in ("weight", "scales", "biases"):
                dtype, array = tensors[f"{prefix}.shared_expert.up_proj.{plane}"]
                tensors[f"{prefix}.{name}.{plane}"] = (dtype, array[:rows].copy())
    save_tensors(source / "model.safetensors", tensors)
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "converted-moe"
    receipt = cli.convert(source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["family"] == {"profile": vision_family.MOE_FAMILY_PROFILE, "version": 4}
    config = json.loads((output / "config.json").read_text())
    mlp = config["decoder_config"]["transformer_config"]["layer_configs"][0]["mlp_config"]
    assert mlp["type"] == "MixtureOfExpertsConfig" and mlp["num_routed_experts"] == 4
    assert mlp["num_active_routed_experts"] == 2 and mlp["num_shared_experts"] == 1
    _, infos = read_safetensors_header(output / "model.safetensors")
    prefix = "decoder.transformer.layers.0.mlp"
    assert infos[prefix + ".experts.up_projection.weights.weights"].shape == (1024, 64)
    assert infos[prefix + ".experts.down_projection.weights.weights"].shape == (512, 64)
    assert infos[prefix + ".shared_expert.up_projection.weights.weights"].shape == (256, 64)
    assert infos[prefix + ".shared_gate.weights.weights"].shape == (1, 64)
    assert all(info.dtype == "BF16" for name, info in infos.items() if name.startswith("vision."))


def test_vision_namespace_cannot_hide_missing_or_opposite_tower(tmp_path):
    source = tmp_path / "source"
    mlx_qwen35_fixture(source, wrapper=True, vl_wrapper=True, group_size=64)
    with pytest.raises(ConversionError):
        build_qwen35_plan(source, include_vision=True)
