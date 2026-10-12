"""Frozen full LFM2 affine exports and strict source-plane contracts."""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from test_lfm2 import REFERENCE as DENSE_REFERENCE
from test_lfm2 import lfm2_fixture
from test_mlx import _float_values, save_mlx_fixture
from test_streaming import tensor_bytes

from stark_convert.adapters.lfm2 import build_lfm2_plan
from stark_convert.streaming import (
    ConversionError,
    UnsupportedConversion,
    load_checkpoint,
    read_safetensors_header,
    write_plan,
)

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
REFERENCE_PATH = Path(__file__).parent / "fixtures" / "mlx_lfm2_exports.json"
VARIANTS = (
    "trailing",
    "pytorch",
    "flat",
    "untied_bias",
    "all_conv",
    "all_attention",
    "partial_dense",
    "dense_embedding",
)


def mlx_lfm2_fixture(source, *, bits=4, group_size=32, dtype="BF16", variant="trailing", sharded=False):
    source.mkdir()
    config = copy.deepcopy(DENSE_REFERENCE["cases"]["mixed_tied"]["config"])
    config.update(
        block_dim=128,
        block_ff_dim=128,
        block_multiple_of=128,
        conv_dim=128,
        hidden_size=128,
        vocab_size=17,
        torch_dtype=DTYPES[dtype],
        quantization={"bits": bits, "group_size": group_size},
        quantization_config={"bits": bits, "group_size": group_size},
    )
    if variant == "untied_bias":
        config.update(tie_embedding=False, conv_bias=True)
    if variant in ("all_conv", "all_attention"):
        config["layer_types"] = ["conv" if variant == "all_conv" else "full_attention"] * 2
    tensors = {}

    def add_float(name, shape, source_dtype=dtype):
        tensors[name] = source_dtype, _float_values(shape, source_dtype, base=len(tensors))

    def add_matrix(prefix, shape, *, dense=False):
        if dense:
            add_float(prefix + ".weight", shape, "F32")
            return
        rows, columns = shape
        packed_shape = rows, columns * bits // 32
        packed = np.arange(np.prod(packed_shape), dtype="<u4").reshape(packed_shape)
        packed = packed * np.uint32(2654435761) + np.uint32(len(tensors))
        tensors[prefix + ".weight"] = "U32", packed
        add_float(prefix + ".scales", (rows, columns // group_size))
        add_float(prefix + ".biases", (rows, columns // group_size))

    add_matrix("model.embed_tokens", (17, 128), dense=variant == "dense_embedding")
    if not config.get("tie_embedding", True):
        add_matrix("lm_head", (17, 128))
    else:
        for plane in ("weight", "scales", "biases"):
            if "model.embed_tokens." + plane in tensors:
                tensors["lm_head." + plane] = tensors["model.embed_tokens." + plane]
    add_float("model.embedding_norm.weight", (128,))
    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"model.layers.{layer}"
        add_float(prefix + ".operator_norm.weight", (128,))
        add_float(prefix + ".ffn_norm.weight", (128,))
        if kind == "conv":
            add_matrix(prefix + ".conv.in_proj", (384, 128))
            add_matrix(prefix + ".conv.out_proj", (128, 128), dense=variant == "partial_dense")
            shape = (3, 128) if variant == "flat" else (128, 1, 3) if variant == "pytorch" else (128, 3, 1)
            if variant == "dense_embedding":
                shape = (128, 3)
            alias = "conv1d.weight" if variant == "pytorch" else "conv_weight" if variant == "flat" else "conv.weight"
            add_float(prefix + ".conv." + alias, shape)
            if config["conv_bias"]:
                add_float(prefix + ".conv." + alias.replace("weight", "bias"), (128,))
        else:
            for projection, width in (("q", 128), ("k", 64), ("v", 64)):
                add_matrix(prefix + f".self_attn.{projection}_proj", (width, 128))
            add_matrix(prefix + ".self_attn.o_proj", (128, 128), dense=variant == "partial_dense")
            for side in ("q", "k"):
                add_float(prefix + f".self_attn.{side}_norm.weight", (32,))
        for side in ("w3", "w1"):
            add_matrix(prefix + ".feed_forward." + side, (128, 128))
        add_matrix(prefix + ".feed_forward.w2", (128, 128))
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors, sharded=sharded)
    return config, tensors


def case_name(variant, bits, group_size, source_dtype, target):
    return f"{variant}-w{bits}-g{group_size}-{source_dtype}-{target}"


EXPORT_CASES = list(itertools.product(("trailing", "pytorch", "flat"), (4, 8), (16, 32, 64, 128), DTYPES, DTYPES))
EXPORT_CASES += list(itertools.product(VARIANTS[3:], (4, 8), (32,), DTYPES, DTYPES))


@pytest.mark.parametrize("variant,bits,group_size,source_dtype,target", EXPORT_CASES)
def test_complete_lfm2_affine_exports_match_pinned_actual_loader(
    tmp_path, variant, bits, group_size, source_dtype, target
):
    reference = json.loads(REFERENCE_PATH.read_text())
    case = reference["cases"][case_name(variant, bits, group_size, source_dtype, target)]
    source = tmp_path / "source"
    _, inputs = mlx_lfm2_fixture(source, variant=variant, bits=bits, group_size=group_size, dtype=source_dtype)
    plan = build_lfm2_plan(source, dtype=DTYPES[target])
    assert plan.decoder_config == reference["decoder_configs"][variant]
    assert plan.source_tensors == len(inputs)
    assert plan.input_bytes == sum(array.nbytes for _, array in inputs.values())
    checked_duplicates = {
        info
        for name, info in load_checkpoint(source).items()
        if name.startswith("lm_head.") and variant != "untied_bias"
    }
    assert {info for op in plan.operations for info in op.sources} | checked_duplicates == set(plan.input_tensors)
    for name, (dtype, raw) in inputs.items():
        assert case["inputs"][name] == {
            "dtype": dtype,
            "shape": list(raw.shape),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }
    destination = tmp_path / "output.safetensors"
    stats = write_plan(plan, destination, chunk_bytes=96)
    assert stats["peak_buffer_bytes"] <= 96
    assert stats["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata, tensors = read_safetensors_header(destination)
    assert {name: json.loads(value) for name, value in metadata.items()} == reference["metadata"][case["metadata_key"]]
    assert set(tensors) == set(case["arrays"])
    for name, info in tensors.items():
        expected = case["arrays"][name]
        assert (info.dtype, list(info.shape), hashlib.sha256(tensor_bytes(info)).hexdigest()) == (
            expected["dtype"],
            expected["shape"],
            expected["sha256"],
        )


@pytest.mark.parametrize("bits,sharded", list(itertools.product((4, 8), (False, True))))
def test_lfm2_affine_fusions_preserve_planes_across_shards(tmp_path, bits, sharded):
    source = tmp_path / "source"
    _, inputs = mlx_lfm2_fixture(source, bits=bits, sharded=sharded)
    plan = build_lfm2_plan(source)
    destination = tmp_path / "output.safetensors"
    assert write_plan(plan, destination, chunk_bytes=96)["peak_buffer_bytes"] <= 96
    _, output = read_safetensors_header(destination)
    for suffix, names in (
        ("mixer.qkvg_projection.weights.weights", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("mlp.up_projection.weights.weights", ("feed_forward.w3", "feed_forward.w1")),
    ):
        assert tensor_bytes(output["decoder.transformer.layers.1." + suffix]) == b"".join(
            inputs["model.layers.1." + name + ".weight"][1].tobytes() for name in names
        )


@pytest.mark.parametrize("plane", ("weight", "scales", "biases"))
def test_lfm2_tied_affine_duplicate_requires_identical_complete_planes(tmp_path, plane):
    source = tmp_path / "source"
    _, inputs = mlx_lfm2_fixture(source)
    dtype, raw = inputs["lm_head." + plane]
    raw = raw.copy()
    raw.view("u1").flat[0] ^= 1
    inputs["lm_head." + plane] = dtype, raw
    save_mlx_fixture(source, inputs)
    with pytest.raises(ConversionError, match="Tied MLX"):
        build_lfm2_plan(source)


@pytest.mark.parametrize("partial", (False, True))
def test_lfm2_tied_affine_duplicate_may_be_absent_but_not_partial(tmp_path, partial):
    source = tmp_path / "source"
    _, inputs = mlx_lfm2_fixture(source)
    for plane in ("weight", "scales", "biases"):
        if partial and plane == "scales":
            continue
        del inputs["lm_head." + plane]
    save_mlx_fixture(source, inputs)
    if partial:
        with pytest.raises(ConversionError, match="matrix plane"):
            build_lfm2_plan(source)
    else:
        assert build_lfm2_plan(source).source_tensors == len(inputs)


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "shape",
        "plane_dtype",
        "fusion_dtype",
        "fusion_dense",
        "packed_norm",
        "packed_conv",
        "dense_conv_shape",
        "extra",
        "no_config",
        "awq",
        "config_disagreement",
    ),
)
def test_lfm2_affine_malformed_or_unimplemented_sources_fail_closed(tmp_path, case):
    source = tmp_path / "source"
    config, inputs = mlx_lfm2_fixture(source)
    prefix = "model.layers.1.self_attn.q_proj"
    error = ConversionError
    if case == "missing":
        del inputs[prefix + ".biases"]
    elif case == "shape":
        inputs[prefix + ".weight"] = "U32", np.zeros((128, 15), dtype="<u4")
    elif case in ("plane_dtype", "fusion_dtype"):
        inputs[prefix + ".scales"] = "F32", _float_values((128, 4), "F32")
        if case == "fusion_dtype":
            inputs[prefix + ".biases"] = "F32", _float_values((128, 4), "F32")
        error = UnsupportedConversion
    elif case == "fusion_dense":
        inputs[prefix + ".weight"] = "BF16", _float_values((128, 128), "BF16")
        for plane in ("scales", "biases"):
            del inputs[prefix + "." + plane]
        error = UnsupportedConversion
    elif case in ("packed_norm", "packed_conv"):
        name, shape = (
            ("model.embedding_norm.weight", (128,))
            if case == "packed_norm"
            else ("model.layers.0.conv.conv.weight", (128, 3, 1))
        )
        inputs[name] = "U32", np.zeros(shape, dtype="<u4")
        error = UnsupportedConversion
    elif case == "dense_conv_shape":
        inputs["model.layers.0.conv.conv.weight"] = "BF16", _float_values((128, 3), "BF16")
    elif case == "extra":
        inputs["model.layers.0.conv.conv.scales"] = "BF16", _float_values((128, 3), "BF16")
    elif case == "no_config":
        del config["quantization"]
        del config["quantization_config"]
        error = UnsupportedConversion
    elif case == "awq":
        config["quantization_config"]["quant_method"] = "awq"
        error = UnsupportedConversion
    else:
        config["quantization_config"]["bits"] = 8
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, inputs)
    with pytest.raises(error):
        build_lfm2_plan(source)


def test_lfm2_quantization_annotation_requires_packed_matrices(tmp_path):
    source = tmp_path / "source"
    config, _ = lfm2_fixture(source)
    config["quantization"] = {"bits": 4, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="no packed"):
        build_lfm2_plan(source)
