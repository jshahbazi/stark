"""Pinned public config bytes and independent canonical decoder controls."""

import copy
import json
import math
from pathlib import Path

import pytest

from stark_convert.adapters import gemma4, gemma4_vision
from stark_convert.registry import get_model
from stark_convert.streaming import TensorInfo, UnsupportedConversion

FIXTURES = Path(__file__).parent / "fixtures" / "gemma4_image_profiles"
SIZES = ("E2B", "E4B", "12B", "26B-A4B", "31B")


def header_tensors(config):
    """Describe official decoder shapes without allocating or opening payloads."""
    text = config["text_config"]
    dim, hidden, count, vocab, ple = (
        text[key]
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "vocab_size",
            "hidden_size_per_layer_input",
        )
    )
    prefix = "model.language_model"
    tensors = {}

    def add(name, shape):
        tensors[prefix + name] = TensorInfo(
            Path("header-only.safetensors"), "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0)
        )

    add(".embed_tokens.weight", (vocab, dim))
    add(".norm.weight", (dim,))
    if ple:
        add(".embed_tokens_per_layer.weight", (vocab, count * ple))
        add(".per_layer_model_projection.weight", (count * ple, dim))
        add(".per_layer_projection_norm.weight", (ple,))
    for index, kind in enumerate(text["layer_types"]):
        src = f".layers.{index}"
        global_attention = kind == "full_attention"
        width = text["global_head_dim"] if global_attention else text["head_dim"]
        groups = (
            text["num_global_key_value_heads"]
            if global_attention and text["attention_k_eq_v"]
            else text["num_key_value_heads"]
        )
        shared = index >= count - text["num_kv_shared_layers"]
        intermediate = hidden * (2 if shared and text["use_double_wide_mlp"] else 1)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            add(src + "." + name + ".weight", (dim,))
        add(src + ".layer_scalar", (1,))
        add(src + ".self_attn.q_norm.weight", (width,))
        add(src + ".self_attn.q_proj.weight", (text["num_attention_heads"] * width, dim))
        if not shared:
            add(src + ".self_attn.k_norm.weight", (width,))
            add(src + ".self_attn.k_proj.weight", (groups * width, dim))
            if not (global_attention and text["attention_k_eq_v"]):
                add(src + ".self_attn.v_proj.weight", (groups * width, dim))
        add(src + ".self_attn.o_proj.weight", (dim, text["num_attention_heads"] * width))
        for name, shape in (
            ("up_proj", (intermediate, dim)),
            ("gate_proj", (intermediate, dim)),
            ("down_proj", (dim, intermediate)),
        ):
            add(src + ".mlp." + name + ".weight", shape)
        if ple:
            add(src + ".per_layer_input_gate.weight", (ple, dim))
            add(src + ".per_layer_projection.weight", (dim, ple))
            add(src + ".post_per_layer_input_norm.weight", (dim,))
        if text["enable_moe_block"]:
            experts, expert_hidden = text["num_experts"], text["moe_intermediate_size"]
            for name in (
                "post_feedforward_layernorm_1",
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm_2",
            ):
                add(src + "." + name + ".weight", (dim,))
            add(src + ".router.proj.weight", (experts, dim))
            add(src + ".router.scale", (dim,))
            add(src + ".router.per_expert_scale", (experts,))
            add(src + ".experts.gate_up_proj", (experts, 2 * expert_hidden, dim))
            add(src + ".experts.down_proj", (experts, dim, expert_hidden))
    return tensors


def decoder_control(control):
    decoder = copy.deepcopy(control["decoder"])
    decoder["transformer_config"]["layer_configs"] = [
        copy.deepcopy(control["layer_templates"][index]) for index in control["layer_order"]
    ]
    return decoder


@pytest.mark.parametrize("size", SIZES)
def test_official_config_bytes_admit_each_named_image_profile(tmp_path, size):
    repo = "google/gemma-4-" + size + "-it"
    profile = gemma4_vision.PROFILES[repo]
    payload = (FIXTURES / (size.lower() + ".config.json")).read_bytes()
    (tmp_path / "config.json").write_bytes(payload)
    (tmp_path / "processor_config.json").write_bytes((FIXTURES / (size.lower() + ".processor.json")).read_bytes())
    contract = gemma4_vision.gemma4_vision_contract(tmp_path, repo, profile["revision"])
    assert contract["family"] == profile["family"]
    assert (
        gemma4_vision._CONFIG_HASHES[profile["model_dim"]]
        == get_model(repo)["profile_source"]["files_sha256"]["config.json"]
    )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("mutation", ["rotary", "layers"])
def test_named_image_profiles_reject_changed_decoder_before_checkpoint_access(tmp_path, size, mutation):
    repo = "google/gemma-4-" + size + "-it"
    config = json.loads((FIXTURES / (size.lower() + ".config.json")).read_bytes())
    if mutation == "rotary":
        config["text_config"]["rope_parameters"]["full_attention"]["rope_theta"] = 10000.0
    else:
        config["text_config"]["num_hidden_layers"] = 2
        config["text_config"]["layer_types"] = ["sliding_attention", "full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="pinned official source"):
        gemma4_vision.gemma4_vision_contract(tmp_path, repo, gemma4_vision.PROFILES[repo]["revision"])


@pytest.mark.parametrize("size", SIZES)
def test_official_header_only_planner_matches_native_decoder_control(tmp_path, monkeypatch, size):
    payload = (FIXTURES / (size.lower() + ".config.json")).read_bytes()
    config = json.loads(payload)
    (tmp_path / "config.json").write_bytes(payload)
    monkeypatch.setattr(gemma4, "load_checkpoint", lambda _path: header_tensors(config))
    plan = gemma4.build_gemma4_plan(tmp_path)
    decoder = plan.decoder_config
    decoder.update(gemma4_image_embeddings=config["text_config"]["hidden_size_per_layer_input"] > 0)
    if not decoder["gemma4_image_embeddings"]:
        decoder.update(scalar_image_embeddings=True, bidirectional_image_attention=True)
    controls = json.loads((FIXTURES / "decoder_controls.json").read_bytes())
    assert decoder == decoder_control(controls[size])
