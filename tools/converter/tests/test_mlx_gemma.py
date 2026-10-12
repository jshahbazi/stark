"""MLX Gemma 3 text planes, runtime scaling semantics, and bounded exports."""

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_mlx import _cast_bytes, _decoded_values, _float_values, save_mlx_fixture
from test_streaming import tensor_bytes

from stark_convert.adapters.gemma import build_gemma_plan
from stark_convert.streaming import (
    ConversionError,
    UnsupportedConversion,
    load_checkpoint,
    read_safetensors_header,
    write_plan,
)

TARGETS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def mlx_gemma_fixture(
    path,
    *,
    bits=4,
    group_size=32,
    dtype="BF16",
    attention_bias=False,
    sharded=False,
    packed_dtype="U32",
    dense_embedding=False,
    dense_down=False,
    hidden_dim=128,
    duplicate_head=False,
):
    path.mkdir()
    config = {
        "model_type": "gemma3_text",
        "architectures": ["Gemma3ForCausalLM"],
        "hidden_size": 128,
        "intermediate_size": hidden_dim,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 17,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": attention_bias,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
        "query_pre_attn_scalar": 32.0,
        "sliding_window": 16,
        "sliding_window_pattern": 3,
        "rope_scaling": None,
        "cache_implementation": "hybrid",
        "eos_token_id": 1,
        "bos_token_id": 0,
        "pad_token_id": 0,
        "torch_dtype": TARGETS[dtype],
        "initializer_range": 0.02,
        "transformers_version": "4.50.0",
        "use_cache": True,
        "quantization": {"bits": bits, "group_size": group_size},
        "quantization_config": {"bits": bits, "group_size": group_size},
    }
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def floating(name, shape):
        tensors[name] = dtype, _float_values(shape, dtype, base=len(tensors))

    def matrix(prefix, shape, dense=False):
        if dense:
            floating(prefix + ".weight", shape)
            return
        rows, columns = shape
        packed_shape = rows, columns * bits // 32
        array = np.arange(np.prod(packed_shape), dtype="<u4").reshape(packed_shape)
        array = array * np.uint32(2654435761) + np.uint32(len(tensors))
        if packed_dtype == "I32":
            array = array.view("<i4")
        tensors[prefix + ".weight"] = packed_dtype, array
        floating(prefix + ".scales", (rows, columns // group_size))
        floating(prefix + ".biases", (rows, columns // group_size))

    matrix("model.embed_tokens", (17, 128), dense_embedding)
    floating("model.norm.weight", (128,))
    for layer in range(3):
        src = f"model.layers.{layer}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            floating(src + "." + name + ".weight", (128,))
        floating(src + ".self_attn.q_norm.weight", (32,))
        floating(src + ".self_attn.k_norm.weight", (32,))
        for name, width in (("q", 128), ("k", 64), ("v", 64)):
            matrix(src + f".self_attn.{name}_proj", (width, 128))
            if attention_bias:
                floating(src + f".self_attn.{name}_proj.bias", (width,))
        matrix(src + ".self_attn.o_proj", (128, 128))
        if attention_bias:
            floating(src + ".self_attn.o_proj.bias", (128,))
        matrix(src + ".mlp.up_proj", (hidden_dim, 128))
        matrix(src + ".mlp.gate_proj", (hidden_dim, 128))
        matrix(src + ".mlp.down_proj", (128, hidden_dim), dense_down)
    if duplicate_head:
        for plane in ("weight",) if dense_embedding else ("weight", "scales", "biases"):
            tensors["lm_head." + plane] = tensors["model.embed_tokens." + plane]
    save_mlx_fixture(path, tensors, sharded=sharded)
    return config, tensors


def update_config(source, config, **changes):
    config.update(changes)
    (source / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.parametrize("source_dtype,target", [(a, b) for a in TARGETS for b in TARGETS])
def test_complete_mlx_gemma_export(tmp_path, bits, group_size, source_dtype, target):
    source = tmp_path / "source"
    _, tensors = mlx_gemma_fixture(source, bits=bits, group_size=group_size, dtype=source_dtype)
    plan = build_gemma_plan(source, family="gemma3_text", dtype=TARGETS[target])
    names = {(info.path, info.offset): name for name, info in load_checkpoint(source).items()}
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    metadata, converted = read_safetensors_header(tmp_path / "output.safetensors")
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(array.nbytes for _, array in tensors.values())
    assert {(info.path, info.offset) for info in plan.input_tensors} == set(names)
    assert set(converted) == {op.name for op in plan.operations}
    assert len(metadata) == 13
    for op in plan.operations:
        arrays = [tensors[names[(info.path, info.offset)]][1] for info in op.sources]
        if op.kind == "mlx_repack":
            assert converted[op.name].dtype == "U8"
            assert tensor_bytes(converted[op.name]) == b"".join(array.tobytes() for array in arrays)
            spec = json.loads(metadata[op.name.removesuffix(".weights") + ".spec"])
            assert spec == {"type": "MLXSpec", "bits": bits, "group_size": group_size, "layout": spec["layout"]}
        elif op.kind == "mlx_params":
            values = np.concatenate([_decoded_values(array, source_dtype) for array in arrays], axis=0)
            values = values.astype({"BF16": ml_dtypes.bfloat16, "F16": "<f2", "F32": "<f4"}[target])
            padded = np.zeros(op.shape, dtype=values.dtype)
            padded[:, : values.shape[0]] = values.T
            assert tensor_bytes(converted[op.name]) == padded.tobytes()
        else:
            wanted_dtype = "F32" if op.kind == "cast_f32" else target
            assert tensor_bytes(converted[op.name]) == b"".join(
                _cast_bytes(array, source_dtype, wanted_dtype) for array in arrays
            )
    decoder = plan.decoder_config
    assert decoder["embedding_config"]["input_scale"] == 128**0.5
    for layer in decoder["transformer_config"]["layer_configs"]:
        for norm in ("pre_mixer_norm_config", "post_mixer_norm_config", "pre_mlp_norm_config", "post_mlp_norm_config"):
            assert layer[norm]["scale_offset"] == 1.0
        assert layer["mixer_config"]["query_norm_config"]["scale_offset"] == 1.0


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dense_embedding", [False, True])
def test_mlx_gemma_biases_shards_padding_and_separate_dense_matrix(tmp_path, bits, dense_embedding):
    source = tmp_path / "source"
    _, tensors = mlx_gemma_fixture(
        source,
        bits=bits,
        attention_bias=True,
        sharded=True,
        packed_dtype="I32",
        dense_embedding=dense_embedding,
        dense_down=True,
        hidden_dim=131,
    )
    for plane in ("weight",) if dense_embedding else ("weight", "scales", "biases"):
        tensors["lm_head." + plane] = tensors["model.embed_tokens." + plane]
    save_mlx_fixture(source, tensors, sharded=True)
    plan = build_gemma_plan(source, family="gemma3_text")
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    metadata, converted = read_safetensors_header(tmp_path / "output.safetensors")
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert json.loads(metadata["decoder.embedding.embedding.spec"])["type"] == (
        "FullPrecisionSpec" if dense_embedding else "MLXSpec"
    )
    for layer in range(3):
        out, src = f"decoder.transformer.layers.{layer}", f"model.layers.{layer}"
        assert tensor_bytes(converted[out + ".mixer.qkvg_projection.biases"]) == b"".join(
            _cast_bytes(tensors[src + f".self_attn.{name}_proj.bias"][1], "BF16", "BF16") for name in ("q", "k", "v")
        )
        assert converted[out + ".mlp.up_projection.weights.scales"].shape == (4, 264)
        values = np.frombuffer(tensor_bytes(converted[out + ".mlp.up_projection.weights.scales"]), dtype="<u2")
        assert np.all(values.reshape(4, 264)[:, 262:] == 0)
        assert json.loads(metadata[out + ".mlp.down_projection.weights.spec"])["type"] == "FullPrecisionSpec"


@pytest.mark.parametrize("plane", ["weight", "scales", "biases"])
def test_mlx_gemma_tied_head_requires_identical_complete_triplet(tmp_path, plane):
    source = tmp_path / "source"
    _, tensors = mlx_gemma_fixture(source)
    for key in ("weight", "scales", "biases"):
        tensors["lm_head." + key] = tensors["model.embed_tokens." + key]
    dtype, value = tensors["lm_head." + plane]
    value = value.copy()
    value.view("u1").flat[0] ^= 1
    tensors["lm_head." + plane] = dtype, value
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError, match="Tied MLX Gemma"):
        build_gemma_plan(source, family="gemma3_text")


@pytest.mark.parametrize(
    "case",
    [
        "missing_weight",
        "missing_scales",
        "missing_biases",
        "packed_shape",
        "coefficient_shape",
        "unknown_tensor",
        "tied_incomplete",
        "bias_instead_of_biases",
        "norm_packed",
    ],
)
def test_mlx_gemma_incomplete_or_inconsistent_layout_fails(tmp_path, case):
    source = tmp_path / "source"
    _, tensors = mlx_gemma_fixture(source)
    prefix = "model.layers.0.self_attn.q_proj"
    if case.startswith("missing_"):
        del tensors[prefix + "." + case.removeprefix("missing_")]
    elif case == "packed_shape":
        tensors[prefix + ".weight"] = "U32", np.zeros((128, 15), dtype="<u4")
    elif case == "coefficient_shape":
        tensors[prefix + ".scales"] = "BF16", np.zeros((128, 3), dtype="<u2")
    elif case == "unknown_tensor":
        tensors["vision.weight"] = "F32", np.zeros(1, dtype="<f4")
    elif case == "tied_incomplete":
        tensors["lm_head.scales"] = tensors["model.embed_tokens.scales"]
    elif case == "bias_instead_of_biases":
        tensors[prefix + ".bias"] = tensors.pop(prefix + ".biases")
    else:
        tensors["model.norm.weight"] = "U32", np.zeros(128, dtype="<u4")
    save_mlx_fixture(source, tensors)
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_gemma_plan(source, family="gemma3_text")


@pytest.mark.parametrize("case", ["qkv_dense", "gate_dense", "fusion_dtype", "planes_dtype", "missing_config"])
def test_mlx_gemma_mixed_fusion_or_parameters_fail_closed(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = mlx_gemma_fixture(source)
    prefix = "model.layers.0." + ("mlp.gate_proj" if case == "gate_dense" else "self_attn.q_proj")
    if case in {"qkv_dense", "gate_dense"}:
        for plane in ("scales", "biases"):
            del tensors[prefix + "." + plane]
        tensors[prefix + ".weight"] = "BF16", _float_values((128, 128), "BF16")
    elif case == "fusion_dtype":
        for plane in ("scales", "biases"):
            tensors[prefix + "." + plane] = "F32", _float_values((128, 4), "F32")
    elif case == "planes_dtype":
        tensors[prefix + ".scales"] = "F32", _float_values((128, 4), "F32")
    else:
        del config["quantization"]
        del config["quantization_config"]
        update_config(source, config)
    save_mlx_fixture(source, tensors)
    with pytest.raises(UnsupportedConversion):
        build_gemma_plan(source, family="gemma3_text")


@pytest.mark.parametrize(
    "changes",
    [
        {"quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}},
        {"quantization": {"bits": 4, "group_size": 32, "unknown": True}},
        {"layer_types": ["sliding_attention"] * 3},
        {"_sliding_window_pattern": 6},
        {"rope_scaling": {"rope_type": "longrope", "factor": 2}},
        {"rope_parameters": {"rope_type": "default"}},
        {"model_type": "gemma3"},
        {"model_type": "gemma4"},
        {"tie_word_embeddings": False},
    ],
)
def test_mlx_gemma_unqualified_semantics_rejected(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = mlx_gemma_fixture(source)
    update_config(source, config, **changes)
    with pytest.raises(UnsupportedConversion):
        build_gemma_plan(source, family="gemma3_text")


def test_mlx_gemma2_remains_explicitly_unsupported(tmp_path):
    source = tmp_path / "source"
    config, _ = mlx_gemma_fixture(source)
    update_config(
        source,
        config,
        model_type="gemma2",
        architectures=["Gemma2ForCausalLM"],
        rope_theta=10000.0,
        sliding_window_pattern=None,
        rope_local_base_freq=None,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
    )
    with pytest.raises(UnsupportedConversion, match="Quantized Gemma"):
        build_gemma_plan(source, family="gemma2")


def test_mlx_gemma_config_requires_at_least_one_packed_matrix(tmp_path):
    from test_gemma import gemma_fixture

    source = tmp_path / "source"
    config, _ = gemma_fixture(source, "gemma3_text")
    update_config(source, config, quantization={"bits": 4, "group_size": 32})
    with pytest.raises(ConversionError, match="no packed"):
        build_gemma_plan(source, family="gemma3_text")


def test_mlx_gemma_complete_export_matches_frozen_external_reference(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "mlx_gemma_export.json").read_text())
    for name, case in fixture["cases"].items():
        source = tmp_path / name
        mlx_gemma_fixture(source, **case["parameters"])
        (source / "config.json").write_text(json.dumps(case["config"]))
        plan = build_gemma_plan(source, family="gemma3_text", dtype=case["target"])
        output = tmp_path / (name + ".safetensors")
        stats = write_plan(plan, output, chunk_bytes=96)
        metadata, tensors = read_safetensors_header(output)
        assert stats["peak_buffer_bytes"] <= 96
        assert plan.decoder_config == case["decoder_config"]
        assert metadata == case["metadata"]
        actual = {
            key: {
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor_bytes(tensor)).hexdigest(),
            }
            for key, tensor in sorted(tensors.items())
        }
        assert actual == case["tensors"]
