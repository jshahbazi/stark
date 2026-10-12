"""Packed MLX checkpoint contracts, complete planning, and bounded export."""

import dataclasses
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.compressed import (
    parse_mlx_quantization,
    validate_mlx_operation,
)
from stark_convert.qwen3 import build_qwen3_plan
from stark_convert.streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_safetensors_header,
    write_plan,
)


def _float_values(shape, dtype, base=0):
    values = np.linspace(-0.75, 0.75, int(np.prod(shape)), dtype="<f4").reshape(shape) + base / 128
    if dtype == "BF16":
        return values.astype(ml_dtypes.bfloat16).view("<u2")
    return values.astype({"F16": "<f2", "F32": "<f4"}[dtype])


def _decoded_values(array, dtype):
    return array.view(ml_dtypes.bfloat16) if dtype == "BF16" else array


def _cast_bytes(array, source_dtype, target):
    values = _decoded_values(array, source_dtype)
    return values.astype({"BF16": ml_dtypes.bfloat16, "F16": "<f2", "F32": "<f4"}[target]).tobytes()


def save_mlx_fixture(path, tensors, *, sharded=False):
    if sharded:
        entries = list(tensors.items())
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        weight_map = {}
        for index, name in enumerate(names):
            contents = dict(entries[index::2])
            save_tensors(path / name, contents)
            weight_map.update(dict.fromkeys(contents, name))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_tensors(path / "model.safetensors", tensors)


def mlx_qwen_fixture(
    path,
    *,
    bits=4,
    group_size=32,
    dtype="BF16",
    tied=False,
    attention_bias=False,
    sharded=False,
    packed_dtype="U32",
):
    path.mkdir()
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 17,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 64,
        "tie_word_embeddings": tied,
        "attention_bias": attention_bias,
        "qkv_bias": attention_bias,
        "use_qk_norm": True,
        "max_window_layers": 1,
        "use_sliding_window": False,
        "sliding_window": None,
        "eos_token_id": 1,
        "torch_dtype": {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[dtype],
        "quantization": {"bits": bits, "group_size": group_size},
        "quantization_config": {"bits": bits, "group_size": group_size},
    }
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def add_float(name, shape):
        tensors[name] = dtype, _float_values(shape, dtype, base=len(tensors))

    def add_matrix(prefix, shape):
        rows, columns = shape
        packed_shape = rows, columns * bits // 32
        array = np.arange(np.prod(packed_shape), dtype="<u4").reshape(packed_shape)
        array = array * np.uint32(2654435761) + np.uint32(len(tensors))
        if packed_dtype == "I32":
            array = array.view("<i4")
        tensors[prefix + ".weight"] = packed_dtype, array
        add_float(prefix + ".scales", (rows, columns // group_size))
        add_float(prefix + ".biases", (rows, columns // group_size))

    add_matrix("model.embed_tokens", (17, 128))
    add_matrix("lm_head", (17, 128))
    if tied:
        for plane in ("weight", "scales", "biases"):
            tensors["lm_head." + plane] = tensors["model.embed_tokens." + plane]
    add_float("model.norm.weight", (128,))
    layer = "model.layers.0"
    add_float(layer + ".input_layernorm.weight", (128,))
    add_float(layer + ".post_attention_layernorm.weight", (128,))
    add_float(layer + ".self_attn.q_norm.weight", (32,))
    add_float(layer + ".self_attn.k_norm.weight", (32,))
    for name, width in (("q", 128), ("k", 64), ("v", 64)):
        add_matrix(layer + f".self_attn.{name}_proj", (width, 128))
        if attention_bias:
            add_float(layer + f".self_attn.{name}_proj.bias", (width,))
    add_matrix(layer + ".self_attn.o_proj", (128, 128))
    if attention_bias:
        add_float(layer + ".self_attn.o_proj.bias", (128,))
    add_matrix(layer + ".mlp.up_proj", (128, 128))
    add_matrix(layer + ".mlp.gate_proj", (128, 128))
    add_matrix(layer + ".mlp.down_proj", (128, 128))
    save_mlx_fixture(path, tensors, sharded=sharded)
    return config, tensors


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "source_dtype,target", [(a, b) for a in ("BF16", "F16", "F32") for b in ("BF16", "F16", "F32")]
)
def test_complete_mlx_qwen3_export(tmp_path, bits, group_size, source_dtype, target):
    source = tmp_path / "source"
    _, tensors = mlx_qwen_fixture(source, bits=bits, group_size=group_size, dtype=source_dtype)
    plan = build_qwen3_plan(source, dtype={"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[target])
    names = {(info.path, info.offset): name for name, info in load_checkpoint(source).items()}
    result = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=8192)
    metadata, converted = read_safetensors_header(tmp_path / "output.safetensors")
    assert result["peak_buffer_bytes"] <= 8192
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(array.nbytes for _, array in tensors.values())
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
            assert converted[op.name].shape == padded.shape
            assert tensor_bytes(converted[op.name]) == padded.tobytes()
        else:
            wanted_dtype = "F32" if op.kind == "cast_f32" else target
            assert tensor_bytes(converted[op.name]) == b"".join(
                _cast_bytes(array, source_dtype, wanted_dtype) for array in arrays
            )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_mlx_biases_tied_triplets_and_cross_shard_fusion(tmp_path, bits, tied):
    source = tmp_path / "source"
    _, tensors = mlx_qwen_fixture(source, bits=bits, tied=tied, attention_bias=True, sharded=True, packed_dtype="I32")
    plan = build_qwen3_plan(source)
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    converted = read_safetensors_header(tmp_path / "output.safetensors")[1]
    assert stats["peak_buffer_bytes"] <= 96
    prefix = "decoder.transformer.layers.0"
    for suffix, names in (
        (".mixer.qkvg_projection.weights.weights", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        (".mlp.up_projection.weights.weights", ("mlp.up_proj", "mlp.gate_proj")),
    ):
        assert tensor_bytes(converted[prefix + suffix]) == b"".join(
            tensors["model.layers.0." + name + ".weight"][1].tobytes() for name in names
        )
    assert tensor_bytes(converted[prefix + ".mixer.qkvg_projection.biases"]) == b"".join(
        tensors[f"model.layers.0.self_attn.{name}_proj.bias"][1].tobytes() for name in ("q", "k", "v")
    )
    assert ("decoder.embedding.output_embedding.weights" in converted) is not tied


@pytest.mark.parametrize("plane", ["weight", "scales", "biases"])
def test_mlx_tied_duplicate_requires_all_planes_identical(tmp_path, plane):
    source = tmp_path / "source"
    _, tensors = mlx_qwen_fixture(source, tied=True)
    name = "lm_head." + plane
    dtype, array = tensors[name]
    array = array.copy()
    array.view("u1").flat[0] ^= 1
    tensors[name] = dtype, array
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError, match="Tied MLX"):
        build_qwen3_plan(source)


def test_mlx_tied_head_may_be_absent(tmp_path):
    source = tmp_path / "source"
    _, tensors = mlx_qwen_fixture(source, tied=True)
    for plane in ("weight", "scales", "biases"):
        del tensors["lm_head." + plane]
    save_mlx_fixture(source, tensors)
    assert build_qwen3_plan(source).source_tensors == len(tensors)


@pytest.mark.parametrize(
    "quantization,error",
    [
        ({"bits": 1, "group_size": 32}, UnsupportedConversion),
        ({"bits": 4, "group_size": 8}, UnsupportedConversion),
        ({"bits": 4, "group_size": -1}, UnsupportedConversion),
        ({"bits": True, "group_size": 32}, ConversionError),
        ({"bits": 4, "group_size": True}, ConversionError),
        ({"bits": 4}, ConversionError),
        ({"bits": 4, "group_size": 32, "mode": "mxfp4"}, UnsupportedConversion),
        ({"bits": 4, "group_size": 32, "quant_method": "awq"}, UnsupportedConversion),
        ({"bits": 4, "group_size": 32, "model.layers.0": {"bits": 8}}, UnsupportedConversion),
        ("mlx", ConversionError),
    ],
)
def test_mlx_unimplemented_or_invalid_config_fails(quantization, error):
    with pytest.raises(error):
        parse_mlx_quantization({"quantization_config": quantization})


def test_mlx_duplicate_config_fields_compare_canonical_semantics():
    assert (
        parse_mlx_quantization(
            {
                "quantization": {"bits": 4, "group_size": 32},
                "quantization_config": {"bits": 4, "group_size": 32, "mode": "affine", "quant_method": "mlx"},
            }
        ).bits
        == 4
    )
    with pytest.raises(ConversionError, match="disagree"):
        parse_mlx_quantization(
            {"quantization": {"bits": 4, "group_size": 32}, "quantization_config": {"bits": 8, "group_size": 32}}
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing_scales",
        "missing_biases",
        "packed_shape",
        "group_shape",
        "packed_dtype",
        "plane_dtype",
        "config_bits",
        "config_groups",
        "no_config",
        "extra",
        "mixed_fusion",
        "mixed_plane_dtype",
        "mixed_fusion_dtype",
    ],
)
def test_mlx_malformed_checkpoint_is_rejected(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = mlx_qwen_fixture(source)
    prefix = "model.layers.0.self_attn.k_proj"
    if case in {"missing_scales", "missing_biases"}:
        del tensors[prefix + "." + case.removeprefix("missing_")]
    elif case == "packed_shape":
        tensors[prefix + ".weight"] = "U32", np.zeros((64, 8), dtype="<u4")
    elif case == "group_shape":
        tensors[prefix + ".scales"] = "BF16", np.zeros((64, 3), dtype="<u2")
    elif case == "packed_dtype":
        tensors[prefix + ".weight"] = "U16", np.zeros((64, 16), dtype="<u2")
    elif case == "plane_dtype":
        tensors[prefix + ".scales"] = "U32", np.zeros((64, 4), dtype="<u4")
    elif case in {"config_bits", "config_groups"}:
        key, value = ("bits", 8) if case == "config_bits" else ("group_size", 64)
        for name in ("quantization", "quantization_config"):
            config[name][key] = value
    elif case == "no_config":
        del config["quantization"]
        del config["quantization_config"]
    elif case == "extra":
        tensors["unexpected.scales"] = "BF16", np.zeros(1, dtype="<u2")
    elif case == "mixed_fusion":
        tensors[prefix + ".weight"] = "BF16", _float_values((64, 128), "BF16")
        del tensors[prefix + ".scales"]
        del tensors[prefix + ".biases"]
    elif case in {"mixed_plane_dtype", "mixed_fusion_dtype"}:
        tensors[prefix + ".scales"] = "F16", _float_values((64, 4), "F16")
        if case == "mixed_fusion_dtype":
            tensors[prefix + ".biases"] = "F16", _float_values((64, 4), "F16")
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_qwen3_plan(source)


def test_mlx_full_precision_projection_is_allowed_outside_fusion(tmp_path):
    source = tmp_path / "source"
    _, tensors = mlx_qwen_fixture(source)
    prefix = "model.layers.0.self_attn.o_proj"
    tensors[prefix + ".weight"] = "BF16", _float_values((128, 128), "BF16")
    del tensors[prefix + ".scales"]
    del tensors[prefix + ".biases"]
    save_mlx_fixture(source, tensors)
    plan = build_qwen3_plan(source)
    spec = json.loads(plan.metadata["decoder.transformer.layers.0.mixer.out_projection.weights.spec"])
    assert spec == {"type": "FullPrecisionSpec", "layout": "output_input"}
    assert any(operation.kind == "mlx_repack" for operation in plan.operations)


@pytest.mark.parametrize("chunk_bytes", [16, 17, 31, 32, 96, 4096])
def test_mlx_parameter_fusion_pads_after_global_row_count(tmp_path, chunk_bytes):
    source = tmp_path / "source.safetensors"
    first = np.arange(6, dtype="<f4").reshape(3, 2)
    second = np.arange(6, 10, dtype="<f4").reshape(2, 2)
    save_tensors(source, {"a": ("F32", first), "b": ("F32", second)})
    tensors = read_safetensors_header(source)[1]
    operation = TensorOperation("test.scales", (tensors["a"], tensors["b"]), "F32", (2, 8), "mlx_params")
    plan = ConversionPlan({}, (operation,), {}, 40, 2)
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=chunk_bytes)
    result = read_safetensors_header(tmp_path / "output.safetensors")[1][operation.name]
    expected = np.zeros((2, 8), dtype="<f4")
    expected[:, :5] = np.concatenate((first, second)).T
    assert tensor_bytes(result) == expected.tobytes()
    assert stats["peak_buffer_bytes"] <= chunk_bytes


@pytest.mark.parametrize("source_dtype", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("target", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("workspace", [16, 96])
def test_mlx_parameter_casts_preserve_frozen_exceptional_float_patterns(tmp_path, source_dtype, target, workspace):
    reference = json.loads((Path(__file__).parent / "fixtures" / "dense_casts.json").read_text())["cases"][
        source_dtype
    ]
    values = np.array(reference["source_bits"], dtype="<u4" if source_dtype == "F32" else "<u2").reshape(-1, 2)
    source = tmp_path / "source.safetensors"
    save_tensors(source, {"params": (source_dtype, values)})
    info = read_safetensors_header(source)[1]["params"]
    rows = info.shape[0]
    operation = TensorOperation("matrix.scales", (info,), target, (2, rows + (-rows % 4)), "mlx_params")
    plan = ConversionPlan({}, (operation,), {}, info.nbytes, 1)
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=workspace)
    result = read_safetensors_header(output)[1][operation.name]
    stored = np.frombuffer(tensor_bytes(result), dtype="<u4" if target == "F32" else "<u2").reshape(result.shape)
    assert np.all(stored[:, rows:] == 0)
    restored = np.ascontiguousarray(stored[:, :rows].T)
    assert hashlib.sha256(restored.tobytes()).hexdigest() == reference["output_sha256"][target]
    assert stats["peak_buffer_bytes"] <= workspace
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "case",
    ["bad_weight_dtype", "bad_output_dtype", "bad_width", "empty", "bad_padding", "mixed_param_dtype", "scalar"],
)
def test_invalid_mlx_writer_operations_fail_before_publication(tmp_path, case):
    source = tmp_path / "source.safetensors"
    save_tensors(
        source,
        {
            "weights": ("U32", np.arange(8, dtype="<u4").reshape(2, 4)),
            "params": ("F32", np.arange(8, dtype="<f4").reshape(2, 4)),
        },
    )
    inputs = read_safetensors_header(source)[1]
    op = TensorOperation("weights", (inputs["weights"],), "U8", (2, 16), "mlx_repack")
    if case == "bad_weight_dtype":
        op = dataclasses.replace(op, sources=(inputs["params"],))
    elif case == "bad_output_dtype":
        op = dataclasses.replace(op, dtype="F32")
    elif case == "bad_width":
        op = dataclasses.replace(op, shape=(2, 15))
    elif case == "empty":
        op = dataclasses.replace(op, sources=())
    elif case == "bad_padding":
        op = TensorOperation("params", (inputs["params"],), "F32", (4, 5), "mlx_params")
    elif case == "mixed_param_dtype":
        op = TensorOperation("params", (inputs["params"], inputs["weights"]), "F32", (4, 4), "mlx_params")
    elif case == "scalar":
        op = dataclasses.replace(op, scalar=1.0)
    with pytest.raises(ConversionError):
        validate_mlx_operation(op)
    output = tmp_path / "output.safetensors"
    with pytest.raises(ConversionError):
        write_plan(ConversionPlan({}, (op,), {}, 32, 1), output)
    assert not output.exists()


@pytest.mark.parametrize(
    "key,value", [("qkv_bias", "false"), ("qkv_bias", True), ("use_qk_norm", False), ("use_qk_norm", 1)]
)
def test_mlx_qwen_aliases_cannot_change_model_semantics(tmp_path, key, value):
    source = tmp_path / "source"
    config, _ = mlx_qwen_fixture(source)
    config[key] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_qwen3_plan(source)
