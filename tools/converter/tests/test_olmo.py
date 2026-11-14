"""OLMo 3 layout and fail-closed semantic tests using synthetic public fixtures."""

import json
import math

import numpy as np
import pytest
from test_dense_adapters import dense_fixture, update_config
from test_mlx import _float_values, save_mlx_fixture
from test_mlx_dense import mlx_dense_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.olmo import build_olmo3_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan


def olmo_fixture(path, *, dtype="BF16", groups=2, tied=False):
    config, tensors = dense_fixture(path, "llama", dtype=dtype, layers=4, tied=tied)
    config.update(
        model_type="olmo3",
        architectures=["Olmo3ForCausalLM"],
        sliding_window=16,
        num_key_value_heads=groups,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_scaling={
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 0.1 * math.log(2.0) + 1,
        },
    )
    for layer in range(4):
        src = f"model.layers.{layer}"
        tensors[src + ".post_feedforward_layernorm.weight"] = tensors.pop(src + ".input_layernorm.weight")
        for name, width in (("q", 8), ("k", groups * 2), ("v", groups * 2)):
            # Store whole-projection Q/K weights. Per-head norm shape would be 2.
            if name != "v":
                values = np.linspace(0.5, 1.5, width, dtype="<f4")
                tensors[src + f".self_attn.{name}_norm.weight"] = (
                    dtype,
                    (values.view("<u4") >> 16).astype("<u2") if dtype == "BF16" else values,
                )
            if groups != 2 and name != "q":
                values = np.arange(width * 8, dtype="<f4").reshape(width, 8) / 100
                tensors[src + f".self_attn.{name}_proj.weight"] = (
                    dtype,
                    (values.view("<u4") >> 16).astype("<u2") if dtype == "BF16" else values,
                )
    (path / "config.json").write_text(json.dumps(config))
    save_tensors(path / "model.safetensors", tensors)
    return config, tensors


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype,requested", [("BF16", "bfloat16"), ("F32", "float32")])
def test_olmo_preserves_projection_norms_postnorm_and_bounded_payloads(tmp_path, groups, dtype, requested):
    source = tmp_path / "source"
    _, tensors = olmo_fixture(source, dtype=dtype, groups=groups)
    plan = build_olmo3_plan(source, dtype=requested)
    configs = plan.decoder_config["transformer_config"]["layer_configs"]
    assert [layer["mixer_config"]["sliding_window_size"] for layer in configs] == [16, 16, 16, None]
    assert [layer["rope_config"]["type"] for layer in configs] == ["UnscaledRoPEConfig"] * 3 + ["YARNRoPEConfig"]
    assert "separate_prefill_last_token" not in plan.decoder_config
    assert configs[-1]["rope_config"]["truncate"] is True
    for layer in configs:
        assert layer["post_norm"] is True
        assert layer["pre_mixer_norm_config"] is None
        assert layer["pre_mlp_norm_config"]["has_scale"] is False
        assert layer["mixer_config"]["qk_norm_full_width"] is True
        assert layer["post_mixer_norm_config"]["upcast_mode"] == "full_layer"
        assert "materialize_sigmoid" not in layer["mlp_config"]["activation"]
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    assert stats["peak_buffer_bytes"] <= 96
    _, converted = read_safetensors_header(output)
    for layer in range(4):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        for name, width in (("q", 8), ("k", groups * 2)):
            source_values = tensors[src + f".self_attn.{name}_norm.weight"][1]
            expected = (source_values.astype("<u4") << 16).view("<f4") if dtype == "BF16" else source_values
            destination = out + ".mixer." + ("query_norm" if name == "q" else "key_norm") + ".scales"
            assert converted[destination].shape == (width,)
            assert tensor_bytes(converted[destination]) == expected.tobytes()
        assert tensor_bytes(converted[out + ".mixer.qkvg_projection.weights.weights"]) == b"".join(
            tensors[src + f".self_attn.{name}_proj.weight"][1].tobytes() for name in ("q", "k", "v")
        )
    assert not any("pre_mlp_norm" in name or "pre_mixer_norm" in name for name in converted)


def test_olmo_mha_and_tied_embedding_layout(tmp_path):
    source = tmp_path / "source"
    _, _ = olmo_fixture(source, groups=4, tied=True)
    plan = build_olmo3_plan(source)
    assert plan.decoder_config["embedding_config"]["type"] == "TiedEmbeddingConfig"
    assert plan.decoder_config["transformer_config"]["layer_configs"][0]["mixer_config"]["num_groups"] == 4


@pytest.mark.parametrize("case", ["missing", "shape", "extra", "per_head_norm"])
def test_olmo_rejects_corrupt_tensor_layout_before_unknown_options(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = olmo_fixture(source)
    update_config(source, config, unknown_option=True)
    if case == "missing":
        del tensors["model.layers.3.post_feedforward_layernorm.weight"]
    elif case == "shape":
        tensors["model.layers.3.self_attn.k_proj.weight"] = ("BF16", np.zeros((1, 8), dtype="<u2"))
    elif case == "extra":
        tensors["model.layers.0.input_layernorm.weight"] = ("BF16", np.ones(8, dtype="<u2"))
    else:
        tensors["model.layers.0.self_attn.q_norm.weight"] = ("BF16", np.ones(2, dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_olmo3_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": True},
        {"num_attention_heads": 3},
        {"sliding_window": 0},
        {"layer_types": ["full_attention"]},
        {"layer_types": ["sparse_attention"] * 4},
        {"rms_norm_eps": -1},
        {"attention_dropout": "none"},
    ],
)
def test_olmo_rejects_malformed_config(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = olmo_fixture(source)
    update_config(source, config, **changes)
    with pytest.raises(ConversionError):
        build_olmo3_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_act": "gelu"},
        {"attention_bias": True},
        {"unknown_option": True},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
        {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32,
                "attention_factor": 2.0,
            }
        },
    ],
)
def test_olmo_rejects_unimplemented_semantics(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = olmo_fixture(source)
    update_config(source, config, **changes)
    with pytest.raises(UnsupportedConversion):
        build_olmo3_plan(source)


def test_olmo_default_attention_pattern_and_unscaled_rope(tmp_path):
    source = tmp_path / "source"
    config, _ = olmo_fixture(source)
    del config["layer_types"]
    update_config(source, config, rope_scaling=None)
    layers = build_olmo3_plan(source).decoder_config["transformer_config"]["layer_configs"]
    assert [layer["mixer_config"]["sliding_window_size"] for layer in layers] == [16, 16, 16, None]
    assert all(layer["rope_config"]["type"] == "UnscaledRoPEConfig" for layer in layers)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_olmo_affine_mlx_planes_preserve_packing_and_whole_projection_norms(tmp_path, bits, group_size):
    source = tmp_path / "source"
    config, tensors = mlx_dense_fixture(source, "llama", bits=bits, group_size=group_size)
    config.update(
        model_type="olmo3", architectures=["Olmo3ForCausalLM"], sliding_window=16, layer_types=["full_attention"]
    )
    src = "model.layers.0"
    tensors[src + ".post_feedforward_layernorm.weight"] = tensors.pop(src + ".input_layernorm.weight")
    tensors[src + ".self_attn.q_norm.weight"] = "BF16", _float_values((128,), "BF16")
    tensors[src + ".self_attn.k_norm.weight"] = "BF16", _float_values((64,), "BF16")
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    plan = build_olmo3_plan(source)
    source_q4 = bits == 4 and group_size == 64
    assert plan.decoder_config.get("separate_prefill_last_token", False) is source_q4
    layer = plan.decoder_config["transformer_config"]["layer_configs"][0]
    upcast = "only_normalization" if source_q4 else "full_layer"
    assert layer["post_mixer_norm_config"]["upcast_mode"] == upcast
    assert layer["post_mlp_norm_config"]["upcast_mode"] == upcast
    assert layer["mixer_config"]["query_norm_config"]["upcast_mode"] == upcast
    assert layer["mixer_config"]["key_norm_config"]["upcast_mode"] == upcast
    assert plan.decoder_config["transformer_config"]["output_norm_config"]["upcast_mode"] == upcast
    for name in ("post_mixer_norm_config", "post_mlp_norm_config"):
        assert layer[name].get("mlx_bf16_reduction", False) is source_q4
    assert (
        plan.decoder_config["transformer_config"]["output_norm_config"].get("mlx_bf16_reduction", False) is source_q4
    )
    # Packed Q/K uses the independently qualified source reduction policy.
    for name in ("query_norm_config", "key_norm_config"):
        assert layer["mixer_config"][name].get("mlx_bf16_reduction", False) is source_q4
    assert not layer["pre_mlp_norm_config"].get("mlx_bf16_reduction", False)
    assert (
        plan.decoder_config["transformer_config"]["layer_configs"][0]["mlp_config"]["activation"].get(
            "materialize_sigmoid", False
        )
        is source_q4
    )
    destination = tmp_path / "output.safetensors"
    stats = write_plan(plan, destination, chunk_bytes=96)
    assert stats["peak_buffer_bytes"] <= 96
    metadata, converted = read_safetensors_header(destination)
    prefix = "decoder.transformer.layers.0.mixer.qkvg_projection.weights"
    assert json.loads(metadata[prefix + ".spec"])["bits"] == bits
    assert json.loads(metadata[prefix + ".spec"]).get("materialize_affine_input_sum", False) is source_q4
    assert tensor_bytes(converted[prefix + ".weights"]) == b"".join(
        tensors[src + f".self_attn.{name}_proj.weight"][1].tobytes() for name in ("q", "k", "v")
    )
    assert converted["decoder.transformer.layers.0.mixer.query_norm.scales"].shape == (128,)
    assert converted["decoder.transformer.layers.0.mixer.key_norm.scales"].shape == (64,)
