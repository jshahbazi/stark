"""Affine MLX Qwen3.5 row permutations and frozen full-decoder exports."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from test_mlx import _float_values, save_mlx_fixture
from test_streaming import tensor_bytes

from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
REFERENCE = Path(__file__).parent / "fixtures" / "mlx_qwen35_exports.json"
VL_REFERENCE = Path(__file__).parent / "fixtures" / "mlx_qwen35_vl_exports.json"


def mlx_qwen35_fixture(
    path,
    *,
    bits=4,
    group_size=32,
    dtype="BF16",
    layout="split",
    tied=False,
    wrapper=False,
    biased=False,
    explicit_gate=False,
    sharded=False,
    mtp=False,
    conv_mlx=False,
    conv_2d=False,
    dense_out=False,
    packed_dtype="I32",
    vl_wrapper=False,
    nested_head=False,
):
    """Synthetic affine planes make every output-channel reordering observable."""
    path.mkdir()
    text = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 17,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": tied,
        "hidden_act": "silu",
        "attention_bias": biased,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        "torch_dtype": DTYPES[dtype],
        "eos_token_id": 1,
        "quantization": {"bits": bits, "group_size": group_size},
        "quantization_config": {"bits": bits, "group_size": group_size},
    }
    config = (
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "tie_word_embeddings": tied,
            "text_config": {**text, "model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_5_vision"},
        }
        if wrapper
        else text
    )
    if mtp:
        config["mtp_num_hidden_layers"] = 1
    if vl_wrapper:
        if not wrapper:
            raise ValueError("VL namespace fixture requires a declared wrapper")
        config.update(quantization=text["quantization"], quantization_config=text["quantization_config"])
        for key in ("quantization", "quantization_config"):
            config["text_config"].pop(key)
    tensors = {}
    prefix = "language_model.model" if vl_wrapper else "model.language_model" if wrapper else "model"
    head_prefix = "language_model.lm_head" if nested_head else "lm_head"

    def add_float(name, shape):
        tensors[name] = dtype, _float_values(shape, dtype, len(tensors))

    def add_matrix(name, shape):
        rows, columns = shape
        packed_shape = rows, columns * bits // 32
        values = np.arange(np.prod(packed_shape), dtype="<u4").reshape(packed_shape)
        values = values * np.uint32(2654435761) + np.uint32(len(tensors))
        tensors[name + ".weight"] = packed_dtype, values.view("<i4") if packed_dtype == "I32" else values
        add_float(name + ".scales", (rows, columns // group_size))
        add_float(name + ".biases", (rows, columns // group_size))

    add_matrix(prefix + ".embed_tokens", (17, 128))
    add_matrix(head_prefix, (17, 128))
    if tied:
        for plane in ("weight", "scales", "biases"):
            tensors[head_prefix + "." + plane] = tensors[prefix + ".embed_tokens." + plane]
    add_float(prefix + ".norm.weight", (128,))
    for layer in range(2):
        src = f"{prefix}.layers.{layer}"
        add_float(src + ".input_layernorm.weight", (128,))
        add_float(src + ".post_attention_layernorm.weight", (128,))
        add_matrix(src + ".mlp.up_proj", (128, 128))
        add_matrix(src + ".mlp.gate_proj", (128, 128))
        add_matrix(src + ".mlp.down_proj", (128, 128))
        if layer == 0:
            attn = src + ".linear_attn"
            branches = {
                "split": (("qkv", 256), ("z", 128), ("b", 4), ("a", 4)),
                "direct": (("", 392),),
                "grouped": (("qkvz", 384), ("ba", 8)),
            }[layout]
            for name, rows in branches:
                add_matrix(attn + ".in_proj" + ("_" + name if name else ""), (rows, 128))
            add_matrix(attn + ".out_proj", (128, 128))
            if dense_out:
                add_float(attn + ".out_proj.weight", (128, 128))
                for plane in ("scales", "biases"):
                    del tensors[attn + ".out_proj." + plane]
            conv_shape = (256, 4) if conv_2d else (256, 4, 1) if conv_mlx else (256, 1, 4)
            add_float(attn + ".conv1d.weight", conv_shape)
            add_float(attn + ".A_log", (4,))
            add_float(attn + ".dt_bias", (4,))
            add_float(attn + ".norm.weight", (32,))
        else:
            attn = src + ".self_attn"
            for name, rows in (("q", 128 if explicit_gate else 256), ("k", 64), ("v", 64)):
                add_matrix(attn + f".{name}_proj", (rows, 128))
                if biased:
                    add_float(attn + f".{name}_proj.bias", (rows,))
            if explicit_gate:
                add_matrix(attn + ".gate_proj", (128, 128))
                if biased:
                    add_float(attn + ".gate_proj.bias", (128,))
            add_matrix(attn + ".o_proj", (128, 128))
            if biased:
                add_float(attn + ".o_proj.bias", (128,))
            add_float(attn + ".q_norm.weight", (32,))
            add_float(attn + ".k_norm.weight", (32,))
    if wrapper:
        vision_prefix = "vision_tower" if vl_wrapper else "model.visual"
        add_float(vision_prefix + ".patch_embed.proj.weight", (2, 2))
    if mtp:
        add_float("mtp.layers.0.self_attn.q_proj.weight", (2, 2))
    (path / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(path, tensors, sharded=sharded)
    return config, tensors


def case_name(bits, group_size, source_dtype, target, layout, variant="plain"):
    return f"w{bits}-g{group_size}-{source_dtype}-{target}-{layout}-{variant}"


CASES = [
    (*values, {})
    for values in itertools.product((4, 8), (16, 32, 64, 128), DTYPES, DTYPES, ("split", "direct", "grouped"))
]
VARIANTS = {
    "tied-wrapper": {"tied": True, "wrapper": True},
    "explicit-gate-bias": {"explicit_gate": True, "biased": True},
    "interleaved-gate-bias": {"biased": True},
    "wrapper-mtp-conv": {"wrapper": True, "mtp": True, "conv_mlx": True},
    "independent-dense": {"dense_out": True},
    "conv-2d": {"conv_2d": True},
}


@pytest.mark.parametrize("bits,group_size,source_dtype,target,layout,options", CASES)
def test_all_mlx_projection_layouts_match_actual_loader(
    tmp_path, bits, group_size, source_dtype, target, layout, options
):
    assert_reference_export(tmp_path, bits, group_size, source_dtype, target, layout, "plain", options)


@pytest.mark.parametrize("bits,source_dtype,target,variant", list(itertools.product((4, 8), DTYPES, DTYPES, VARIANTS)))
def test_mlx_optional_layouts_match_actual_loader(tmp_path, bits, source_dtype, target, variant):
    assert_reference_export(tmp_path, bits, 32, source_dtype, target, "split", variant, VARIANTS[variant])


def assert_reference_export(
    tmp_path, bits, group_size, source_dtype, target, layout, variant, options, reference_path=REFERENCE
):
    reference = json.loads(reference_path.read_text())
    case = reference["cases"][case_name(bits, group_size, source_dtype, target, layout, variant)]
    source = tmp_path / "source"
    _, tensors = mlx_qwen35_fixture(
        source, bits=bits, group_size=group_size, dtype=source_dtype, layout=layout, sharded=True, **options
    )
    plan = build_qwen35_plan(source, dtype=DTYPES[target])
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(array.nbytes for _, array in tensors.values())
    assert plan.decoder_config == reference["decoder_configs"][case["config_key"]]
    assert {key: json.loads(value) for key, value in metadata.items()} == reference["metadata"][case["metadata_key"]]
    assert set(converted) == set(case["arrays"])
    for name, info in converted.items():
        expected = case["arrays"][name]
        assert list(info.shape) == expected["shape"]
        assert info.dtype == expected["dtype"]
        assert hashlib.sha256(tensor_bytes(info)).hexdigest() == expected["sha256"]


@pytest.mark.parametrize("bits,packed_dtype", list(itertools.product((4, 8), ("I32", "U32"))))
def test_packed_word_sign_does_not_change_output_bytes(tmp_path, bits, packed_dtype):
    source = tmp_path / "source"
    _, tensors = mlx_qwen35_fixture(source, bits=bits, packed_dtype=packed_dtype, layout="grouped")
    plan = build_qwen35_plan(source)
    output = tmp_path / "output.safetensors"
    write_plan(plan, output, chunk_bytes=8192)
    _, converted = read_safetensors_header(output)
    name = "decoder.embedding.input_embedding.weights"
    assert tensor_bytes(converted[name]) == tensors["model.embed_tokens.weight"][1].tobytes()


@pytest.mark.parametrize("plane", ("weight", "scales", "biases"))
def test_tied_duplicate_mlx_head_requires_identical_planes(tmp_path, plane):
    source = tmp_path / "source"
    _, tensors = mlx_qwen35_fixture(source, tied=True)
    dtype, values = tensors["lm_head." + plane]
    values = values.copy()
    values.view("u1").flat[0] ^= 1
    tensors["lm_head." + plane] = dtype, values
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError, match="tied lm_head differs"):
        build_qwen35_plan(source)


@pytest.mark.parametrize("partial", (False, True))
def test_tied_head_absent_or_incomplete(tmp_path, partial):
    source = tmp_path / "source"
    _, tensors = mlx_qwen35_fixture(source, tied=True)
    for plane in ("weight", "scales", "biases"):
        if partial and plane == "scales":
            continue
        del tensors["lm_head." + plane]
    save_mlx_fixture(source, tensors)
    if partial:
        with pytest.raises(ConversionError, match="missing a matrix plane"):
            build_qwen35_plan(source)
    else:
        assert build_qwen35_plan(source).source_tensors == len(tensors)


@pytest.mark.parametrize(
    "case",
    (
        "missing_plane",
        "packing_shape",
        "plane_shape",
        "plane_dtype",
        "fusion_dtype",
        "fusion_dense",
        "norm_packed",
        "extra",
        "no_config",
        "alias_only",
        "awq",
        "disagree",
        "one_bit",
        "unsupported_group",
        "non_affine",
    ),
)
def test_malformed_and_unsupported_affine_checkpoint_fails_closed(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = mlx_qwen35_fixture(source)
    prefix = "model.layers.0.linear_attn.in_proj_qkv"
    error = ConversionError
    if case == "missing_plane":
        del tensors[prefix + ".biases"]
    elif case == "packing_shape":
        tensors[prefix + ".weight"] = "I32", np.zeros((256, 15), dtype="<i4")
    elif case == "plane_shape":
        tensors[prefix + ".scales"] = "BF16", np.zeros((256, 3), dtype="<u2")
    elif case == "plane_dtype":
        tensors[prefix + ".scales"] = "F32", _float_values((256, 4), "F32")
        error = UnsupportedConversion
    elif case == "fusion_dtype":
        for plane in ("scales", "biases"):
            tensors[prefix + "." + plane] = "F32", _float_values((256, 4), "F32")
        error = UnsupportedConversion
    elif case == "fusion_dense":
        tensors[prefix + ".weight"] = "BF16", _float_values((256, 128), "BF16")
        for plane in ("scales", "biases"):
            del tensors[prefix + "." + plane]
        error = UnsupportedConversion
    elif case == "norm_packed":
        tensors["model.norm.weight"] = "I32", np.zeros(128, dtype="<i4")
        error = UnsupportedConversion
    elif case == "extra":
        tensors["model.unused.scales"] = "BF16", np.zeros(128, dtype="<u2")
    elif case == "no_config":
        for key in ("quantization", "quantization_config"):
            config.pop(key)
        error = UnsupportedConversion
    elif case == "alias_only":
        config.pop("quantization")
        error = UnsupportedConversion
    elif case == "disagree":
        config["quantization_config"]["bits"] = 8
    else:
        key, value = {
            "awq": ("quant_method", "awq"),
            "one_bit": ("bits", 1),
            "unsupported_group": ("group_size", 256),
            "non_affine": ("mode", "mxfp4"),
        }[case]
        for quant_key in ("quantization", "quantization_config"):
            config[quant_key][key] = value
        error = UnsupportedConversion
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    with pytest.raises(error):
        build_qwen35_plan(source)


def test_quantization_description_requires_packed_text_matrix(tmp_path):
    from test_qwen35 import qwen35_fixture

    source = tmp_path / "source"
    config, _ = qwen35_fixture(source)
    config["quantization"] = {"bits": 4, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="no packed checkpoint matrices"):
        build_qwen35_plan(source)


def test_sliced_mlx_plane_source_change_is_rejected(tmp_path):
    source = tmp_path / "source"
    mlx_qwen35_fixture(source, layout="grouped", sharded=True)
    plan = build_qwen35_plan(source)
    assert all(
        info.file_signature == original.file_signature
        for op in plan.operations
        for info in op.sources
        for original in plan.input_tensors
        if info.path == original.path
    )
    with (source / "model-00001-of-00002.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)


@pytest.mark.parametrize(
    "namespace,vl_wrapper", (("model.visual", False), ("mtp", False), ("vision_tower", True), ("mtp", True))
)
def test_omitted_quantized_wrapper_auxiliary_source_is_verified(tmp_path, namespace, vl_wrapper):
    from test_streaming import save_tensors

    source = tmp_path / "source"
    _, tensors = mlx_qwen35_fixture(source, wrapper=True, mtp=True, vl_wrapper=vl_wrapper)
    auxiliary = {name: value for name, value in tensors.items() if name.startswith(namespace + ".")}
    text = {name: value for name, value in tensors.items() if name not in auxiliary}
    save_tensors(source / "model.safetensors", text)
    save_tensors(source / "auxiliary.safetensors", auxiliary)
    weight_map = {**dict.fromkeys(text, "model.safetensors"), **dict.fromkeys(auxiliary, "auxiliary.safetensors")}
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    plan = build_qwen35_plan(source)
    assert plan.source_tensors == len(tensors)
    assert not any(info.path.name == "auxiliary.safetensors" for op in plan.operations for info in op.sources)
    with (source / "auxiliary.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)


@pytest.mark.parametrize("namespace", ("vision", "mtp"))
def test_mlx_wrapper_omission_requires_declaration(tmp_path, namespace):
    source = tmp_path / "source"
    config, _ = mlx_qwen35_fixture(source, wrapper=True, mtp=True)
    config.pop("vision_config" if namespace == "vision" else "mtp_num_hidden_layers")
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="unexpected"):
        build_qwen35_plan(source)


def test_mlx_norm_uses_primary_quantization_field_without_alias(tmp_path):
    source = tmp_path / "source"
    config, _ = mlx_qwen35_fixture(source)
    config.pop("quantization_config")
    (source / "config.json").write_text(json.dumps(config))
    plan = build_qwen35_plan(source)
    transformer = plan.decoder_config["transformer_config"]
    assert transformer["output_norm_config"]["scale_offset"] is None
    assert all(layer["pre_mixer_norm_config"]["scale_offset"] is None for layer in transformer["layer_configs"])


@pytest.mark.parametrize(
    "bits,source_dtype,target,layout,tied",
    list(itertools.product((4, 8), DTYPES, DTYPES, ("split", "direct", "grouped"), (False, True))),
)
def test_actual_mlx_vl_wrapper_namespace_matches_pinned_loader(tmp_path, bits, source_dtype, target, layout, tied):
    options = {"wrapper": True, "vl_wrapper": True, "conv_mlx": True, "tied": tied, "nested_head": not tied}
    variant = "vl-tied" if tied else "vl-untied-nested-head"
    assert_reference_export(tmp_path, bits, 64, source_dtype, target, layout, variant, options, VL_REFERENCE)


@pytest.mark.parametrize("vl_wrapper", (False, True))
@pytest.mark.parametrize(
    "case", ("other_vision_alias", "no_vision_declaration", "other_text_root", "undeclared_mtp", "other_mtp_alias")
)
def test_wrapper_namespaces_remain_exclusive_and_declared(tmp_path, vl_wrapper, case):
    source = tmp_path / "source"
    config, tensors = mlx_qwen35_fixture(source, wrapper=True, mtp=True, vl_wrapper=vl_wrapper)
    if case == "other_vision_alias":
        opposite = "model.visual" if vl_wrapper else "vision_tower"
        tensors[opposite + ".unrelated.weight"] = "BF16", np.zeros((2, 2), dtype="<u2")
    elif case == "no_vision_declaration":
        config.pop("vision_config")
    elif case == "other_text_root":
        original = "language_model.model" if vl_wrapper else "model.language_model"
        opposite = "model.language_model" if vl_wrapper else "language_model.model"
        for plane in ("weight", "scales", "biases"):
            tensors[opposite + ".embed_tokens." + plane] = tensors[original + ".embed_tokens." + plane]
    elif case == "undeclared_mtp":
        config.pop("mtp_num_hidden_layers")
    else:
        tensors["language_model.mtp.unrelated.weight"] = "BF16", np.zeros((2, 2), dtype="<u2")
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError):
        build_qwen35_plan(source)
