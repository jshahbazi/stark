"""Independent tensor references and semantic edge cases for dense adapters."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters import build_llama_plan, build_mistral_plan, build_qwen2_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

PLANNERS = {"qwen2": build_qwen2_plan, "llama": build_llama_plan, "mistral": build_mistral_plan}
BASELINES = json.loads((Path(__file__).parent / "fixtures" / "dense_decoder_configs.json").read_text())


def dense_fixture(path: Path, family: str, *, tied=False, dtype="BF16", sharded=False, layers=2):
    path.mkdir()
    config = {
        "model_type": family,
        "architectures": [
            {"qwen2": "Qwen2ForCausalLM", "llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM"}[family]
        ],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": tied,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "eos_token_id": 1,
        "bos_token_id": 0,
        "torch_dtype": {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[dtype],
        "initializer_range": 0.02,
        "transformers_version": "4.45.0",
        "use_cache": True,
    }
    if family == "qwen2":
        config.update({"use_sliding_window": True, "sliding_window": 16, "max_window_layers": 1})
    elif family == "llama":
        config.update({"attention_bias": False, "mlp_bias": False, "pretraining_tp": 1, "rope_scaling": None})
    else:
        config["sliding_window"] = 16
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def add(name, shape):
        # Distinct finite numeric values make q/k/v and up/gate order observable.
        values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
        raw = {
            "BF16": lambda x: (x.view("<u4") >> 16).astype("<u2"),
            "F16": lambda x: x.astype("<f2"),
            "F32": lambda x: x,
        }[dtype](values)
        tensors[name] = (dtype, raw)

    add("model.embed_tokens.weight", (16, 8))
    if not tied:
        add("lm_head.weight", (16, 8))
    add("model.norm.weight", (8,))
    for layer in range(layers):
        src = f"model.layers.{layer}"
        add(src + ".input_layernorm.weight", (8,))
        add(src + ".post_attention_layernorm.weight", (8,))
        for name, width in (("q", 8), ("k", 4), ("v", 4)):
            add(src + f".self_attn.{name}_proj.weight", (width, 8))
            if family == "qwen2":
                add(src + f".self_attn.{name}_proj.bias", (width,))
        add(src + ".self_attn.o_proj.weight", (8, 8))
        add(src + ".mlp.up_proj.weight", (12, 8))
        add(src + ".mlp.gate_proj.weight", (12, 8))
        add(src + ".mlp.down_proj.weight", (8, 12))
    if sharded:
        weights = {}
        for shard in range(2):
            filename = f"model-{shard}.safetensors"
            contents = dict(list(tensors.items())[shard::2])
            save_tensors(path / filename, contents)
            weights.update(dict.fromkeys(contents, filename))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    else:
        save_tensors(path / "model.safetensors", tensors)
    return config, tensors


def update_config(source, config, **changes):
    config.update(changes)
    (source / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("case", BASELINES["cases"].values(), ids=BASELINES["cases"])
def test_complete_decoder_configs_match_frozen_external_reference(tmp_path, case):
    source = tmp_path / "source"
    dense_fixture(source, case["family"], tied=case["config"]["tie_word_embeddings"])
    (source / "config.json").write_text(json.dumps(case["config"]))
    assert PLANNERS[case["family"]](source).decoder_config == case["decoder_config"]


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize("dtype,requested", [("BF16", "bfloat16"), ("F16", "float16"), ("F32", "float32")])
@pytest.mark.parametrize("tied,sharded", [(False, False), (True, True)])
def test_dense_payloads_match_independent_reference(tmp_path, family, dtype, requested, tied, sharded):
    source = tmp_path / "source"
    _, tensors = dense_fixture(source, family, tied=tied, dtype=dtype, sharded=sharded)
    plan = PLANNERS[family](source, dtype=requested)
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    expected = {}
    embedding_out = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    expected[embedding_out + ".weights"] = tensors["model.embed_tokens.weight"][1].tobytes()
    if not tied:
        expected["decoder.embedding.output_embedding.weights"] = tensors["lm_head.weight"][1].tobytes()

    def normalization(out, src):
        data = tensors[src][1]
        expected[out + ".scales"] = (
            (data.astype("<u4") << 16).view("<f4") if dtype == "BF16" else data.astype("<f4")
        ).tobytes()

    normalization("decoder.transformer.output_norm", "model.norm.weight")
    for layer in range(2):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        normalization(out + ".pre_mixer_norm", src + ".input_layernorm.weight")
        normalization(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight")
        expected[out + ".mixer.qkvg_projection.weights.weights"] = b"".join(
            tensors[src + f".self_attn.{name}_proj.weight"][1].tobytes() for name in ("q", "k", "v")
        )
        expected[out + ".mixer.out_projection.weights.weights"] = tensors[src + ".self_attn.o_proj.weight"][
            1
        ].tobytes()
        expected[out + ".mlp.up_projection.weights.weights"] = b"".join(
            tensors[src + f".mlp.{name}_proj.weight"][1].tobytes() for name in ("up", "gate")
        )
        expected[out + ".mlp.down_projection.weights.weights"] = tensors[src + ".mlp.down_proj.weight"][1].tobytes()
        if family == "qwen2":
            expected[out + ".mixer.qkvg_projection.biases"] = b"".join(
                tensors[src + f".self_attn.{name}_proj.bias"][1].tobytes() for name in ("q", "k", "v")
            )
    assert set(converted) == set(expected)
    assert all(tensor_bytes(converted[name]) == payload for name, payload in expected.items())
    assert len(metadata) == 9 if tied else len(metadata) == 10
    assert all(json.loads(spec)["type"] == "FullPrecisionSpec" for spec in metadata.values())
    assert json.loads(metadata[embedding_out + ".spec"])["layout"] == "input_output"


def test_qwen2_sliding_window_is_earlier_layers(tmp_path):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "qwen2")
    plan = build_qwen2_plan(source)
    layers = plan.decoder_config["transformer_config"]["layer_configs"]
    assert [layer["mixer_config"]["sliding_window_size"] for layer in layers] == [16, None]
    assert all(layer["mixer_config"]["query_norm_config"] is None for layer in layers)
    assert all(layer["mixer_config"]["has_qkvg_biases"] for layer in layers)
    assert all(not layer["mixer_config"]["has_out_biases"] for layer in layers)
    update_config(source, config, use_sliding_window=False, max_window_layers=28, sliding_window=None)
    plan = build_qwen2_plan(source)
    assert all(
        layer["mixer_config"]["sliding_window_size"] is None
        for layer in plan.decoder_config["transformer_config"]["layer_configs"]
    )


@pytest.mark.parametrize("factor", [8.0, 32.0])
def test_llama31_and_32_rope_semantics(tmp_path, factor):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    update_config(
        source,
        config,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": factor,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    plan = build_llama_plan(source)
    actual = plan.decoder_config["transformer_config"]["layer_configs"][0]["rope_config"]
    assert actual == {
        "type": "LlamaRoPEConfig",
        "base": 500000.0,
        "max_sequence_length": 131072,
        "head_dim": 2,
        "scaling_factor": factor,
        "original_context_length": 8192,
        "low_frequency_factor": 1.0,
        "high_frequency_factor": 4.0,
    }


@pytest.mark.parametrize("rope_type", ["default", "llama3"])
@pytest.mark.parametrize("duplicate_legacy", [False, True])
def test_modern_llama_rope_matches_complete_legacy_plan(tmp_path, rope_type, duplicate_legacy):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    scaling = {"type": rope_type}
    if rope_type == "llama3":
        scaling.update(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=16)
    update_config(source, config, rope_theta=5000000.0, rope_scaling=scaling)
    legacy = build_llama_plan(source)
    if not duplicate_legacy:
        del config["rope_theta"]
        del config["rope_scaling"]
    update_config(
        source,
        config,
        rope_parameters={
            "rope_theta": 5000000,
            **{key: value for key, value in scaling.items() if key != "type"},
            "rope_type": rope_type,
        },
    )
    modern = build_llama_plan(source)
    assert modern.decoder_config == legacy.decoder_config
    assert modern.operations == legacy.operations
    assert all(
        layer["rope_config"]["base"] == 5000000.0
        for layer in modern.decoder_config["transformer_config"]["layer_configs"]
    )


@pytest.mark.parametrize(
    "parameters",
    [
        [],
        {},
        {"rope_type": "default"},
        {"rope_type": "default", "rope_theta": True},
        {"rope_type": "default", "rope_theta": -1},
        {"rope_type": "default", "rope_theta": float("inf")},
        {"rope_type": "default", "rope_theta": "5000000"},
        {"rope_type": True, "rope_theta": 5000000},
        {"rope_type": "default", "type": "llama3", "rope_theta": 5000000},
        {"full_attention": {"rope_type": "default", "rope_theta": 5000000}},
    ],
)
def test_modern_llama_rope_malformed_metadata_fails_closed(tmp_path, parameters):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    del config["rope_theta"]
    update_config(source, config, rope_parameters=parameters)
    with pytest.raises(ConversionError):
        build_llama_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"rope_theta": 10000.0},
        {"rope_scaling": {"rope_type": "llama3", "factor": 8.0}},
        {"rope_scaling": {"rope_type": "default", "factor": 1.0}},
    ],
)
def test_modern_llama_rope_contradictory_legacy_metadata_rejected(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    del config["rope_theta"]
    update_config(source, config, rope_parameters={"rope_type": "default", "rope_theta": 5000000}, **changes)
    with pytest.raises(ConversionError, match="contradictory RoPE"):
        build_llama_plan(source)


@pytest.mark.parametrize(
    "parameters",
    [
        {"rope_type": "default", "rope_theta": 5000000, "partial_rotary_factor": 0.5},
        {"rope_type": "default", "rope_theta": 5000000, "factor": 1.0},
        {"rope_type": "dynamic", "rope_theta": 5000000, "factor": 2.0},
    ],
)
def test_modern_llama_rope_unimplemented_layouts_rejected(tmp_path, parameters):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    del config["rope_theta"]
    update_config(source, config, rope_parameters=parameters)
    with pytest.raises(UnsupportedConversion):
        build_llama_plan(source)


def test_llama_mirror_packaging_annotation_does_not_change_plan(tmp_path):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    before = build_llama_plan(source)
    update_config(source, config, unsloth_fixed=True)
    after = build_llama_plan(source)
    assert before.decoder_config == after.decoder_config
    assert before.operations == after.operations


@pytest.mark.parametrize("grouped", [False, True])
def test_llama_training_annotations_preserve_dense_plan(tmp_path, grouped):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    before = build_llama_plan(source)
    update_config(source, config, fp8=False, use_grouped_mm=grouped)
    after = build_llama_plan(source)
    assert before.decoder_config == after.decoder_config
    assert before.operations == after.operations


@pytest.mark.parametrize("field", ["fp8", "use_grouped_mm"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_llama_training_annotations_require_booleans(tmp_path, field, value):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    update_config(source, config, **{field: value})
    with pytest.raises(ConversionError):
        build_llama_plan(source)


def test_llama_fp8_annotation_refuses_unsupported_representation(tmp_path):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    update_config(source, config, fp8=True)
    with pytest.raises(UnsupportedConversion, match="^FP8 checkpoint"):
        build_llama_plan(source)


@pytest.mark.parametrize("window", [None, 16])
def test_mistral_full_or_sliding_attention(tmp_path, window):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "mistral")
    update_config(source, config, sliding_window=window)
    plan = build_mistral_plan(source)
    assert all(
        layer["mixer_config"]["sliding_window_size"] == window
        for layer in plan.decoder_config["transformer_config"]["layer_configs"]
    )


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize("case", ["missing", "shape", "extra", "tied_head"])
def test_dense_malformed_tensors_fail_before_unknown_semantics(tmp_path, family, case):
    source = tmp_path / "source"
    config, tensors = dense_fixture(source, family, tied=case == "tied_head")
    update_config(source, config, unknown_semantics=True)
    if case == "missing":
        del tensors["model.layers.0.self_attn.q_proj.weight"]
    elif case == "shape":
        tensors["model.layers.0.self_attn.q_proj.weight"] = ("BF16", np.zeros((4, 8), dtype="<u2"))
    elif case == "extra":
        tensors["extra.weight"] = ("BF16", np.zeros(2, dtype="<u2"))
    else:
        tensors["lm_head.weight"] = ("BF16", np.zeros((16, 8), dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": True},
        {"num_attention_heads": 3},
        {"rms_norm_eps": -1},
        {"tie_word_embeddings": "false"},
        {"attention_dropout": "zero"},
        {"pretraining_tp": 0},
    ],
)
def test_dense_malformed_config_rejected_before_unknown_semantics(tmp_path, family, changes):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, family)
    update_config(source, config, unknown_semantics=True, **changes)
    with pytest.raises(ConversionError):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_act": "gelu"},
        {"unknown_semantics": 1},
        {"attention_dropout": 0.1},
        {"mlp_bias": True},
        {"rope_scaling": {"rope_type": "yarn", "factor": 2.0}},
    ],
)
def test_dense_unqualified_semantics_fail_closed(tmp_path, family, changes):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, family)
    update_config(source, config, **changes)
    with pytest.raises(UnsupportedConversion):
        PLANNERS[family](source)


@pytest.mark.parametrize(
    "changes", [{"factor": -1}, {"high_freq_factor": 1.0}, {"original_max_position_embeddings": 0}, {"factor": True}]
)
def test_llama_rope_malformed_boundaries(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 16,
        **changes,
    }
    update_config(source, config, rope_scaling=scaling)
    with pytest.raises(ConversionError):
        build_llama_plan(source)


def promote_fixture_tensor_to_f32(tensors, name):
    dtype, array = tensors[name]
    assert dtype == "BF16"
    tensors[name] = ("F32", (array.astype("<u4") << 16).view("<f4"))


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize("mixed_tensor", ["self_attn.q_proj.weight", "mlp.up_proj.weight"])
def test_mixed_dense_projection_fusion_is_explicitly_unsupported(tmp_path, family, mixed_tensor):
    source = tmp_path / "source"
    _, tensors = dense_fixture(source, family)
    promote_fixture_tensor_to_f32(tensors, "model.layers.0." + mixed_tensor)
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="^Mixed-dtype fusion promotion"):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize("mixed_tensor", ["self_attn.q_proj.weight", "mlp.up_proj.weight"])
@pytest.mark.parametrize("missing_tensor", ["model.layers.1.mlp.down_proj.weight", "model.norm.weight"])
def test_missing_required_tensor_is_rejected_before_mixed_dense_fusion(tmp_path, family, mixed_tensor, missing_tensor):
    source = tmp_path / "source"
    _, tensors = dense_fixture(source, family)
    promote_fixture_tensor_to_f32(tensors, "model.layers.0." + mixed_tensor)
    # Remove a tensor visited after the mixed first-layer projection. An early
    # unsupported return would hide this recognized-layout corruption.
    del tensors[missing_tensor]
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="^Dense checkpoint is missing a required tensor$"):
        PLANNERS[family](source)


def test_mixed_qwen2_bias_fusion_is_explicitly_unsupported(tmp_path):
    source = tmp_path / "source"
    _, tensors = dense_fixture(source, "qwen2")
    promote_fixture_tensor_to_f32(tensors, "model.layers.0.self_attn.q_proj.bias")
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="^Mixed-dtype bias fusion promotion"):
        build_qwen2_plan(source)


@pytest.mark.parametrize("missing_tensor", ["model.layers.1.mlp.down_proj.weight", "model.norm.weight"])
def test_missing_required_tensor_is_rejected_before_mixed_qwen2_bias_fusion(tmp_path, missing_tensor):
    source = tmp_path / "source"
    _, tensors = dense_fixture(source, "qwen2")
    promote_fixture_tensor_to_f32(tensors, "model.layers.0.self_attn.q_proj.bias")
    del tensors[missing_tensor]
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="^Dense checkpoint is missing a required tensor$"):
        build_qwen2_plan(source)


def test_llama_browser_export_annotation_preserves_the_plan(tmp_path):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    before = build_llama_plan(source).decoder_config
    config["transformers.js_config"] = {"dtype": "q4f16", "use_external_data_format": True}
    (source / "config.json").write_text(json.dumps(config))
    assert build_llama_plan(source).decoder_config == before


@pytest.mark.parametrize("value", [False, None, "q4f16", []])
def test_malformed_browser_export_annotation_is_rejected(tmp_path, value):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama")
    config["transformers.js_config"] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="malformed browser-export metadata"):
        build_llama_plan(source)
