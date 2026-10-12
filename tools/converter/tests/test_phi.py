"""Independent fused-payload and frozen pinned-reference Phi configuration checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.phi import build_phi_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

BASELINES = json.loads((Path(__file__).parent / "fixtures" / "phi_decoder_configs.json").read_text())


def phi_fixture(path: Path, *, dtype="BF16", sharded=False, partial_factor=0.75, duplicate_head=False):
    path.mkdir()
    config = {
        "model_type": "phi3",
        "architectures": ["Phi3ForCausalLM"],
        "hidden_size": 16,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 32,
        "partial_rotary_factor": partial_factor,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0] * (int(8 * partial_factor) // 2),
            "long_factor": [2.0 + pair / 4 for pair in range(int(8 * partial_factor) // 2)],
        },
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "attention_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "embd_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "full_attn_mod": 1,
        "interpolate_factor": 1,
        "sliding_window": 256,
        "eos_token_id": 1,
        "bos_token_id": 0,
        "torch_dtype": {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[dtype],
        "initializer_range": 0.02,
        "transformers_version": "4.45.0",
        "use_cache": True,
        "auto_map": {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
            "AutoTokenizer": "Xenova/gpt-4o",
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def add(name, shape):
        values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
        raw = {
            "BF16": lambda x: (x.view("<u4") >> 16).astype("<u2"),
            "F16": lambda x: x.astype("<f2"),
            "F32": lambda x: x,
        }[dtype](values)
        tensors[name] = (dtype, raw)

    add("model.embed_tokens.weight", (32, 16))
    if duplicate_head:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    add("model.norm.weight", (16,))
    for layer in range(2):
        src = f"model.layers.{layer}"
        add(src + ".input_layernorm.weight", (16,))
        add(src + ".post_attention_layernorm.weight", (16,))
        add(src + ".self_attn.qkv_proj.weight", (32, 16))
        add(src + ".self_attn.o_proj.weight", (16, 16))
        add(src + ".mlp.gate_up_proj.weight", (24, 16))
        add(src + ".mlp.down_proj.weight", (16, 12))
    if sharded:
        weights = {}
        for shard in range(3):
            filename = f"model-{shard}.safetensors"
            contents = dict(list(tensors.items())[shard::3])
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
def test_complete_phi_decoder_config_matches_frozen_external_reference(tmp_path, case):
    source = tmp_path / "source"
    phi_fixture(source)
    (source / "config.json").write_text(json.dumps(case["config"]))
    assert build_phi_plan(source).decoder_config == case["decoder_config"]


@pytest.mark.parametrize("dtype,requested", [("BF16", "bfloat16"), ("F16", "float16"), ("F32", "float32")])
@pytest.mark.parametrize("sharded,duplicate_head", [(False, False), (True, True)])
def test_phi_fused_payloads_match_independent_reference(tmp_path, dtype, requested, sharded, duplicate_head):
    source = tmp_path / "source"
    _, tensors = phi_fixture(source, dtype=dtype, sharded=sharded, duplicate_head=duplicate_head)
    plan = build_phi_plan(source, dtype=requested)
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert stats["input_bytes"] == sum(data.nbytes for _, data in tensors.values())
    assert stats["source_tensors"] == len(tensors)
    expected = {"decoder.embedding.embedding.weights": tensors["model.embed_tokens.weight"][1].tobytes()}

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
        expected[out + ".mixer.qkvg_projection.weights.weights"] = tensors[src + ".self_attn.qkv_proj.weight"][
            1
        ].tobytes()
        expected[out + ".mixer.out_projection.weights.weights"] = tensors[src + ".self_attn.o_proj.weight"][
            1
        ].tobytes()
        gate_up = tensors[src + ".mlp.gate_up_proj.weight"][1]
        expected[out + ".mlp.up_projection.weights.weights"] = gate_up[12:].tobytes() + gate_up[:12].tobytes()
        expected[out + ".mlp.down_projection.weights.weights"] = tensors[src + ".mlp.down_proj.weight"][1].tobytes()
    assert set(converted) == set(expected)
    assert all(tensor_bytes(converted[name]) == payload for name, payload in expected.items())
    assert len(metadata) == 9
    assert json.loads(metadata["decoder.embedding.embedding.spec"])["layout"] == "input_output"
    assert all(json.loads(spec)["type"] == "FullPrecisionSpec" for spec in metadata.values())
    assert len(plan.input_tensors) == len(tensors)


@pytest.mark.parametrize("source_dtype,requested", [("BF16", "float32"), ("F16", "bfloat16"), ("F32", "float16")])
def test_phi_casts_preserve_up_then_gate_order(tmp_path, source_dtype, requested):
    source = tmp_path / "source"
    _, tensors = phi_fixture(source, dtype=source_dtype)
    output = tmp_path / "converted.safetensors"
    stats = write_plan(build_phi_plan(source, dtype=requested), output, chunk_bytes=96)
    _, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    source_values = tensors["model.layers.0.mlp.gate_up_proj.weight"][1]
    if source_dtype == "BF16":
        source_values = (source_values.astype("<u4") << 16).view("<f4")
    else:
        source_values = source_values.astype("<f4")
    if requested == "bfloat16":
        import ml_dtypes

        expected = source_values.astype(ml_dtypes.bfloat16).view("<u2")
    else:
        expected = source_values.astype("<f4" if requested == "float32" else "<f2")
    actual = converted["decoder.transformer.layers.0.mlp.up_projection.weights.weights"]
    assert tensor_bytes(actual) == expected[12:].tobytes() + expected[:12].tobytes()


def test_phi_partial_rotary_dimension_and_full_attention(tmp_path):
    source = tmp_path / "source"
    config, _ = phi_fixture(source)
    plan = build_phi_plan(source)
    layers = plan.decoder_config["transformer_config"]["layer_configs"]
    assert all(layer["mixer_config"]["head_dim"] == 8 for layer in layers)
    assert all(layer["rope_config"]["head_dim"] == 6 for layer in layers)
    # Phi-4-mini's sliding_window is ignored with full_attn_mod == 1 in the
    # pinned HFPhi3Config; emitting that window would diverge from reference.
    assert all(layer["mixer_config"]["sliding_window_size"] is None for layer in layers)
    update_config(source, config, original_max_position_embeddings=256)
    assert (
        build_phi_plan(source).decoder_config["transformer_config"]["layer_configs"][0]["rope_config"][
            "scaling_factor"
        ]
        == 0.5
    )


@pytest.mark.parametrize("case", ["missing", "shape", "extra", "unfused", "tied_head"])
def test_phi_malformed_tensors_rejected_before_unqualified_semantics(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = phi_fixture(source)
    update_config(source, config, unknown_semantics=True)
    if case == "missing":
        del tensors["model.layers.1.mlp.down_proj.weight"]
    elif case == "shape":
        tensors["model.layers.0.self_attn.qkv_proj.weight"] = ("BF16", np.zeros((16, 16), dtype="<u2"))
    elif case == "extra":
        tensors["extra.weight"] = ("BF16", np.zeros(2, dtype="<u2"))
    elif case == "unfused":
        tensors["model.layers.0.self_attn.q_proj.weight"] = tensors.pop("model.layers.0.self_attn.qkv_proj.weight")
    else:
        tensors["lm_head.weight"] = ("BF16", np.zeros((32, 16), dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_phi_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": True},
        {"num_attention_heads": 3},
        {"num_key_value_heads": 3},
        {"partial_rotary_factor": 0},
        {"partial_rotary_factor": 1.1},
        {"partial_rotary_factor": 0.5 / 8},
        {"partial_rotary_factor": 0.625},
        {"original_max_position_embeddings": 1},
        {"rms_norm_eps": -1},
        {"rope_theta": True},
        {"tie_word_embeddings": "true"},
        {"full_attn_mod": True},
        {"attention_dropout": "zero"},
        {"sliding_window": 0},
        {"auto_map": []},
    ],
)
def test_phi_malformed_config_rejected_before_unqualified_semantics(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = phi_fixture(source)
    update_config(source, config, unknown_semantics=True, **changes)
    with pytest.raises(ConversionError):
        build_phi_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_act": "gelu"},
        {"unknown_semantics": True},
        {"full_attn_mod": 2},
        {"interpolate_factor": 2},
        {"attention_dropout": 0.1},
        {"embd_pdrop": 0.1},
        {"resid_pdrop": 0.1},
        {"mlp_bias": True},
        {"attention_bias": True},
        {"lm_head_bias": True},
        {"auto_map": {"AutoConfig": "different.Architecture"}},
        {"architectures": ["PhiForCausalLM"]},
    ],
)
def test_phi_unqualified_semantics_fail_closed(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = phi_fixture(source)
    update_config(source, config, **changes)
    with pytest.raises(UnsupportedConversion):
        build_phi_plan(source)


def test_untied_phi3_is_explicitly_unsupported(tmp_path):
    source = tmp_path / "source"
    config, tensors = phi_fixture(source)
    update_config(source, config, tie_word_embeddings=False)
    tensors["lm_head.weight"] = ("BF16", np.zeros((32, 16), dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="^Untied Phi3 embeddings"):
        build_phi_plan(source)


@pytest.mark.parametrize(
    "changes", [{"short_factor": [1.0]}, {"long_factor": [1.0, 0, 1.0]}, {"short_factor": [True] * 3}]
)
def test_phi_malformed_longrope_factors(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = phi_fixture(source)
    update_config(source, config, rope_scaling={**config["rope_scaling"], **changes})
    with pytest.raises(ConversionError):
        build_phi_plan(source)


@pytest.mark.parametrize("changes", [{"type": "linear"}, {"attention_factor": 1.0}])
def test_phi_unqualified_longrope_options(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = phi_fixture(source)
    update_config(source, config, rope_scaling={**config["rope_scaling"], **changes})
    with pytest.raises(UnsupportedConversion):
        build_phi_plan(source)


def test_phi_row_views_retain_original_input_mutation_checks(tmp_path):
    source = tmp_path / "source"
    phi_fixture(source, sharded=True)
    plan = build_phi_plan(source)
    gate_up = next(op for op in plan.operations if op.name.endswith("mlp.up_projection.weights.weights"))
    assert gate_up.sources[0].offset > gate_up.sources[1].offset
    assert gate_up.sources[0].file_signature == gate_up.sources[1].file_signature
    path = gate_up.sources[0].path
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ConversionError, match="^Checkpoint changed during conversion$"):
        write_plan(plan, tmp_path / "converted.safetensors", chunk_bytes=96)
