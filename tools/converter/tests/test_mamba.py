"""Frozen actual Mamba2 mixer exports and strict bounded input validation."""

import copy
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.mamba import mamba2_config, plan_mamba2
from stark_convert.compressed import MLXQuantization
from stark_convert.streaming import (
    ConversionError,
    ConversionPlan,
    UnsupportedConversion,
    load_checkpoint,
    read_safetensors_header,
    write_plan,
)

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "mamba_exports.json").read_text())
DTYPES = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}


def mamba_fixture(source, case="grouped", *, dtype="BF16", sharded=False):
    source.mkdir()
    reference = REFERENCE["cases"][case]
    ssm = copy.deepcopy(reference["ssm_config"])
    model = reference["model_dim"]
    inner = model * ssm["expand"]
    conv = inner + 2 * ssm["n_qk_heads"] * ssm["d_state"]
    proj = conv + inner + ssm["n_v_heads"]
    tensors = {}

    def add(name, shape):
        raw = (np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 32 + len(tensors)).astype(DTYPES[dtype])
        tensors["source.mixer." + name] = (dtype, raw)

    for name, shape in (("in_proj", (proj, model)), ("out_proj", (model, inner))):
        if name in reference.get("packed", []):
            quant = reference["quantization"]
            packed = np.arange(shape[0] * shape[1] * quant["bits"] // 32, dtype="<u4").reshape(
                shape[0], shape[1] * quant["bits"] // 32
            )
            packed = packed * np.uint32(17) + np.uint32(0xA98B4321)
            code = "I32" if reference.get("signed") else "U32"
            if code == "I32":
                packed = packed.view("<i4")
            tensors["source.mixer." + name + ".weight"] = (code, packed)
            add(name + ".scales", (shape[0], shape[1] // quant["group_size"]))
            add(name + ".biases", (shape[0], shape[1] // quant["group_size"]))
        else:
            add(name + ".weight", shape)
        if ssm["bias"]:
            add(name + ".bias", (shape[0],))
    kernel = ssm.get("d_conv", 4)
    shape = {"flat": (conv, kernel), "pytorch": (conv, 1, kernel), "trailing": (conv, kernel, 1)}[
        reference["conv_shape"]
    ]
    add(reference["conv_alias"], shape)
    if ssm.get("conv_bias", True):
        add(reference["conv_alias"].replace("weight", "bias"), (conv,))
    add("D", (ssm["n_v_heads"],))
    add("z_bias", (inner,))
    if sharded:
        mapping = {}
        for index in range(2):
            name = f"model-{index}.safetensors"
            contents = dict(list(tensors.items())[index::2])
            save_tensors(source / name, contents)
            mapping.update(dict.fromkeys(contents, name))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_tensors(source / "model.safetensors", tensors)
    return model, ssm, tensors


def build_mixer_plan(source, model, ssm, dtype="BF16", quantization=None):
    tensors = load_checkpoint(source)
    expected, unsupported = set(), []
    operations, metadata, config = plan_mamba2(
        tensors,
        "source.mixer",
        "decoder.mixer",
        model,
        ssm,
        dtype,
        expected,
        unsupported,
        quantization=quantization,
    )
    if set(tensors) != expected:
        raise ConversionError("Mixer fixture has unconsumed tensors")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    return ConversionPlan(
        config,
        tuple(operations),
        metadata,
        sum(t.nbytes for t in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )


@pytest.mark.parametrize("case", REFERENCE["cases"])
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
@pytest.mark.parametrize("sharded", [False, True])
def test_actual_pinned_mamba2_exports_match_with_bounded_workspace(
    tmp_path, case, source_dtype, target_dtype, sharded
):
    source = tmp_path / "source"
    model, ssm, inputs = mamba_fixture(source, case, dtype=source_dtype, sharded=sharded)
    reference = REFERENCE["cases"][case]
    frozen_inputs = reference["inputs"][source_dtype]
    assert set(inputs) == set(frozen_inputs)
    for name, (dtype, raw) in inputs.items():
        expected_input = frozen_inputs[name]
        assert (dtype, list(raw.shape)) == (expected_input["dtype"], expected_input["shape"])
        assert hashlib.sha256(raw.tobytes()).hexdigest() == expected_input["sha256"]
    quantization = MLXQuantization(**reference["quantization"]) if "quantization" in reference else None
    plan = build_mixer_plan(source, model, ssm, target_dtype, quantization)
    assert plan.decoder_config == reference["mixer_config"]
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, tensors = read_safetensors_header(output)
    frozen = reference["exports"][source_dtype + "_" + target_dtype]
    assert set(tensors) == set(frozen["arrays"])
    for name, info in tensors.items():
        expected = frozen["arrays"][name]
        assert (info.dtype, list(info.shape)) == (expected["dtype"], expected["shape"])
        assert hashlib.sha256(tensor_bytes(info)).hexdigest() == expected["sha256"]
    assert {key: json.loads(value) for key, value in metadata.items()} == frozen["metadata"]
    assert {tensor for op in plan.operations for tensor in op.sources} == set(plan.input_tensors)
    assert stats["peak_buffer_bytes"] <= 96
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["in_proj.weight", "out_proj.weight", "conv1d.weight", "conv1d.bias", "D", "z_bias"])
@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_mamba2_required_weights_fail_closed(tmp_path, name, corruption):
    source = tmp_path / "source"
    model, ssm, tensors = mamba_fixture(source)
    name = "source.mixer." + name
    if corruption == "missing":
        tensors.pop(name)
    else:
        tensors[name] = ("BF16", np.zeros((1,), dtype=ml_dtypes.bfloat16))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_mixer_plan(source, model, ssm)


@pytest.mark.parametrize("suffix", ["conv_weight", "conv.weight", "conv_bias", "conv.bias"])
def test_mamba2_ambiguous_convolution_aliases_fail_closed(tmp_path, suffix):
    source = tmp_path / "source"
    model, ssm, tensors = mamba_fixture(source)
    original = "source.mixer.conv1d." + ("bias" if "bias" in suffix else "weight")
    tensors["source.mixer." + suffix] = tensors[original]
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_mixer_plan(source, model, ssm)


@pytest.mark.parametrize("name", ["in_proj.bias", "out_proj.bias", "A_log", "dt_bias", "extra.weight"])
def test_mamba1_and_later_generation_tensors_are_not_silently_ignored(tmp_path, name):
    source = tmp_path / "source"
    model, ssm, tensors = mamba_fixture(source)
    tensors["source.mixer." + name] = ("F32", np.zeros((8,), dtype="<f4"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_mixer_plan(source, model, ssm)


@pytest.mark.parametrize("key", ["d_state", "n_v_heads", "n_qk_heads", "expand", "activation", "bias"])
def test_mamba2_structural_fields_are_explicit(key):
    config = copy.deepcopy(REFERENCE["cases"]["grouped"]["ssm_config"])
    config.pop(key)
    with pytest.raises(ConversionError):
        mamba2_config(32, config, [])


@pytest.mark.parametrize(
    "changes",
    [
        {"d_state": True},
        {"n_v_heads": 0},
        {"n_qk_heads": 3},
        {"n_v_heads": 3},
        {"expand": 1.5},
        {"d_conv": 0},
        {"bias": 1},
        {"conv_bias": None},
        {"chunk_size": False},
        {"activation": []},
        {"activation": ""},
    ],
)
def test_mamba2_malformed_config_is_error(changes):
    config = {**REFERENCE["cases"]["grouped"]["ssm_config"], **changes}
    with pytest.raises(ConversionError):
        mamba2_config(32, config, [])


@pytest.mark.parametrize("changes", [{"activation": "relu"}, {"time_step_limit": [0, 10]}, {"use_norm": True}])
def test_mamba2_unknown_ssm_semantics_are_reported(changes):
    config = {**REFERENCE["cases"]["grouped"]["ssm_config"], **changes}
    unsupported = []
    mamba2_config(32, config, unsupported)
    assert unsupported


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "mixed_planes", "bit_mismatch"])
def test_mamba2_mlx_planes_are_strictly_validated(tmp_path, corruption):
    source = tmp_path / "source"
    model, ssm, tensors = mamba_fixture(source, "mlx_4_32")
    quantization = MLXQuantization(4, 32)
    name = "source.mixer.in_proj.scales"
    if corruption == "missing":
        tensors.pop(name)
    elif corruption == "shape":
        tensors[name] = ("BF16", np.zeros((1, 4), dtype=ml_dtypes.bfloat16))
    elif corruption == "dtype":
        tensors["source.mixer.in_proj.weight"] = ("F32", tensors["source.mixer.in_proj.weight"][1].astype("<f4"))
    elif corruption == "mixed_planes":
        tensors[name] = ("F32", tensors[name][1].astype("<f4"))
    else:
        quantization = MLXQuantization(8, 32)
    save_tensors(source / "model.safetensors", tensors)
    expected = UnsupportedConversion if corruption in {"dtype", "mixed_planes"} else ConversionError
    with pytest.raises(expected):
        build_mixer_plan(source, model, ssm, quantization=quantization)


def test_mamba2_packed_projection_requires_quantization_representation(tmp_path):
    source = tmp_path / "source"
    model, ssm, _ = mamba_fixture(source, "mlx_4_32")
    with pytest.raises(UnsupportedConversion):
        build_mixer_plan(source, model, ssm)


def test_mamba2_original_source_signatures_remain_bound(tmp_path):
    source = tmp_path / "source"
    model, ssm, tensors = mamba_fixture(source)
    plan = build_mixer_plan(source, model, ssm)
    name = "source.mixer.z_bias"
    tensors[name] = ("F32", tensors[name][1].astype("<f4"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
