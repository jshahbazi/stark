"""Frozen Lalamo exports and complete MLX Llama/SmolLM3 source contracts."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from test_dense_adapters import dense_fixture
from test_granite_smollm import fixture as granite_fixture
from test_mlx import _float_values, mlx_qwen_fixture, save_mlx_fixture
from test_streaming import tensor_bytes

from stark_convert.adapters import build_llama_plan, build_mistral_plan, build_qwen2_plan
from stark_convert.adapters.granite import build_granite_plan
from stark_convert.adapters.smollm3 import build_smollm3_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
PLANNERS = {"llama": build_llama_plan, "smollm3": build_smollm3_plan}
REFERENCE_PATH = Path(__file__).parent / "fixtures" / "mlx_dense_exports.json"


def mlx_dense_fixture(source, family, *, bits=4, group_size=32, dtype="BF16", tied=False, sharded=False, biased=False):
    """Reuse synthetic plane generation; provide each family its own config."""
    config, tensors = mlx_qwen_fixture(
        source, bits=bits, group_size=group_size, dtype=dtype, tied=tied, attention_bias=biased, packed_dtype="I32"
    )
    for key in ("qkv_bias", "use_qk_norm", "max_window_layers", "use_sliding_window", "sliding_window"):
        config.pop(key)
    config.update(
        model_type=family,
        architectures=["LlamaForCausalLM" if family == "llama" else "SmolLM3ForCausalLM"],
        attention_dropout=0.0,
        initializer_range=0.02,
        mlp_bias=biased,
        bos_token_id=0,
        pretraining_tp=1,
        rope_scaling=None,
        transformers_version="4.45.0",
        use_cache=True,
    )
    for name in ("q_norm", "k_norm"):
        del tensors[f"model.layers.0.self_attn.{name}.weight"]
    if family == "smollm3":
        config.update(no_rope_layers=[0], no_rope_layer_interval=1, pretraining_tp=2, hidden_act="silu_glu")
        # The reviewed importer defaults omitted SmolLM3 tie_word_embeddings
        # to true in actual quantized profiles. Tests below exercise omission.
        if biased:
            for name, size in (("up", 128), ("gate", 128), ("down", 128)):
                tensors[f"model.layers.0.mlp.{name}_proj.bias"] = dtype, _float_values((size,), dtype, len(tensors))
    (source / "config.json").write_text(json.dumps(config))
    if sharded:
        (source / "model.safetensors").unlink()
    save_mlx_fixture(source, tensors, sharded=sharded)
    return config, tensors


def case_name(family, bits, group_size, source_dtype, target):
    return f"{family}-w{bits}-g{group_size}-{source_dtype}-{target}"


@pytest.mark.parametrize(
    "family,bits,group_size,source_dtype,target",
    list(itertools.product(PLANNERS, (4, 8), (16, 32, 64, 128), DTYPES, DTYPES)),
)
def test_complete_dense_mlx_exports_match_frozen_actual_loader(
    tmp_path, family, bits, group_size, source_dtype, target
):
    reference = json.loads(REFERENCE_PATH.read_text())
    case = reference["cases"][case_name(family, bits, group_size, source_dtype, target)]
    source = tmp_path / "source"
    _, tensors = mlx_dense_fixture(source, family, bits=bits, group_size=group_size, dtype=source_dtype)
    plan = PLANNERS[family](source, dtype=DTYPES[target])
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(array.nbytes for _, array in tensors.values())
    assert {info for operation in plan.operations for info in operation.sources} == set(plan.input_tensors)
    assert plan.decoder_config == reference["decoder_configs"][family]
    assert {key: json.loads(value) for key, value in metadata.items()} == reference["metadata"][case["metadata_key"]]
    assert set(converted) == set(case["arrays"])
    for name, info in converted.items():
        expected = case["arrays"][name]
        assert list(info.shape) == expected["shape"]
        assert info.dtype == expected["dtype"]
        assert hashlib.sha256(tensor_bytes(info)).hexdigest() == expected["sha256"]


@pytest.mark.parametrize("family,bits", list(itertools.product(PLANNERS, (4, 8))))
def test_tied_mlx_head_all_planes_and_cross_shard_fusion(tmp_path, family, bits):
    source = tmp_path / "source"
    _, tensors = mlx_dense_fixture(source, family, bits=bits, tied=True, sharded=True)
    plan = PLANNERS[family](source)
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    metadata, converted = read_safetensors_header(tmp_path / "output.safetensors")
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert "decoder.embedding.output_embedding.weights" not in converted
    assert json.loads(metadata["decoder.embedding.embedding.spec"])["layout"] == "input_output"
    prefix = "decoder.transformer.layers.0"
    for suffix, names in (
        (".mixer.qkvg_projection.weights.weights", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        (".mlp.up_projection.weights.weights", ("mlp.up_proj", "mlp.gate_proj")),
    ):
        assert tensor_bytes(converted[prefix + suffix]) == b"".join(
            tensors["model.layers.0." + name + ".weight"][1].tobytes() for name in names
        )


@pytest.mark.parametrize("family,plane", list(itertools.product(PLANNERS, ("weight", "scales", "biases"))))
def test_tied_mlx_duplicate_difference_in_any_plane_is_rejected(tmp_path, family, plane):
    source = tmp_path / "source"
    _, tensors = mlx_dense_fixture(source, family, tied=True)
    dtype, array = tensors["lm_head." + plane]
    array = array.copy()
    array.view("u1").flat[0] ^= 1
    tensors["lm_head." + plane] = dtype, array
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError, match="Tied MLX"):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize("partial", (False, True))
def test_tied_mlx_head_absent_or_incomplete(tmp_path, family, partial):
    source = tmp_path / "source"
    config, tensors = mlx_dense_fixture(source, family, tied=True)
    for plane in ("weight", "scales", "biases"):
        if partial and plane == "scales":
            continue
        del tensors["lm_head." + plane]
    if family == "smollm3":
        del config["tie_word_embeddings"]
        (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    if partial:
        with pytest.raises(ConversionError, match="missing a matrix plane"):
            PLANNERS[family](source)
    else:
        assert PLANNERS[family](source).source_tensors == len(tensors)


@pytest.mark.parametrize("family", PLANNERS)
def test_independent_dense_matrix_in_mlx_checkpoint_preserves_precision(tmp_path, family):
    source = tmp_path / "source"
    _, tensors = mlx_dense_fixture(source, family)
    prefix = "model.layers.0.self_attn.o_proj"
    tensors[prefix + ".weight"] = "F32", _float_values((128, 128), "F32")
    for plane in ("scales", "biases"):
        del tensors[prefix + "." + plane]
    save_mlx_fixture(source, tensors)
    plan = PLANNERS[family](source)
    assert any(op.kind == "mlx_repack" for op in plan.operations)
    operation = next(op for op in plan.operations if op.name.endswith(".mixer.out_projection.weights.weights"))
    assert (operation.kind, operation.sources[0].dtype, operation.dtype) == ("cast", "F32", "BF16")


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize(
    "case",
    (
        "missing_plane",
        "shape",
        "parameter_dtype",
        "fusion_dtype",
        "fusion_dense",
        "norm_packed",
        "extra",
        "no_config",
        "awq",
        "config_disagreement",
    ),
)
def test_malformed_or_unsupported_mlx_sources_fail_closed(tmp_path, family, case):
    source = tmp_path / "source"
    config, tensors = mlx_dense_fixture(source, family)
    prefix = "model.layers.0.self_attn.q_proj"
    error = ConversionError
    if case == "missing_plane":
        del tensors[prefix + ".biases"]
    elif case == "shape":
        tensors[prefix + ".weight"] = "I32", np.zeros((128, 15), dtype="<i4")
    elif case in ("parameter_dtype", "fusion_dtype"):
        tensors[prefix + ".scales"] = "F32", _float_values((128, 4), "F32")
        if case == "fusion_dtype":
            tensors[prefix + ".biases"] = "F32", _float_values((128, 4), "F32")
        error = UnsupportedConversion
    elif case == "fusion_dense":
        tensors[prefix + ".weight"] = "BF16", _float_values((128, 128), "BF16")
        for plane in ("scales", "biases"):
            del tensors[prefix + "." + plane]
        error = UnsupportedConversion
    elif case == "norm_packed":
        tensors["model.norm.weight"] = "I32", np.zeros(128, dtype="<i4")
        error = UnsupportedConversion
    elif case == "extra":
        tensors["model.unused.scales"] = "BF16", np.zeros(128, dtype="<u2")
    elif case == "no_config":
        del config["quantization"]
        del config["quantization_config"]
        error = UnsupportedConversion
    elif case == "awq":
        config["quantization_config"]["quant_method"] = "awq"
        error = UnsupportedConversion
    else:
        config["quantization_config"]["bits"] = 8
    save_mlx_fixture(source, tensors)
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(error):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", PLANNERS)
def test_quantization_metadata_requires_actual_packed_matrix(tmp_path, family):
    source = tmp_path / "source"
    config, _ = dense_fixture(source, "llama") if family == "llama" else granite_fixture(source, "smollm3")
    config["quantization"] = {"bits": 4, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="no packed checkpoint matrices"):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", ("qwen2", "mistral", "granite"))
def test_other_dense_families_retain_quantized_coverage_rejection(tmp_path, family):
    source = tmp_path / "source"
    config, _ = granite_fixture(source) if family == "granite" else dense_fixture(source, family)
    config["quantization"] = {"bits": 4, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="Quantized checkpoint"):
        {"qwen2": build_qwen2_plan, "mistral": build_mistral_plan, "granite": build_granite_plan}[family](source)


def test_smollm3_biases_are_separate_from_dequantization_biases(tmp_path):
    source = tmp_path / "source"
    _, tensors = mlx_dense_fixture(source, "smollm3", biased=True)
    plan = build_smollm3_plan(source)
    write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    converted = read_safetensors_header(tmp_path / "output.safetensors")[1]
    prefix = "decoder.transformer.layers.0"
    for suffix, names in (
        (".mixer.qkvg_projection.biases", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        (".mlp.up_projection.biases", ("mlp.up_proj", "mlp.gate_proj")),
    ):
        assert tensor_bytes(converted[prefix + suffix]) == b"".join(
            tensors["model.layers.0." + name + ".bias"][1].tobytes() for name in names
        )
