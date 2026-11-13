"""MLX affine Qwen2 and Mistral planner contracts.

These model-free fixtures cover structural planning, preflight and catalogue
admission only. They do not qualify any checkpoint's native execution.
"""

import itertools
import json

import numpy as np
import pytest
from test_catalog_tensor_headers import add_checkpoint
from test_mlx import _float_values, mlx_qwen_fixture, save_mlx_fixture
from test_streaming import tensor_bytes

from stark_convert import model_catalog as catalog
from stark_convert.adapters import build_mistral_plan, build_qwen2_plan
from stark_convert.model_support import preflight_model
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

PLANNERS = {"qwen2": build_qwen2_plan, "mistral": build_mistral_plan}
PREFIX = "decoder.transformer.layers.0"


def mlx_family_fixture(
    source, family, *, bits=4, group_size=32, dtype="BF16", tied=False, sharded=False, sliding_window=None
):
    """Qwen2 keeps q/k/v projection biases and has no o_proj bias or Q/K norms."""
    config, tensors = mlx_qwen_fixture(
        source,
        bits=bits,
        group_size=group_size,
        dtype=dtype,
        tied=tied,
        attention_bias=family == "qwen2",
        packed_dtype="I32",
    )
    for key in ("qkv_bias", "use_qk_norm"):
        config.pop(key)
    for name in ("q_norm", "k_norm"):
        del tensors[f"model.layers.0.self_attn.{name}.weight"]
    tensors.pop("model.layers.0.self_attn.o_proj.bias", None)
    if family == "qwen2":
        config.update(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    else:
        for key in ("max_window_layers", "use_sliding_window"):
            config.pop(key)
        config.update(
            model_type="mistral", architectures=["MistralForCausalLM"], attention_bias=False, sliding_window=None
        )
    if sliding_window is not None:
        config["sliding_window"] = sliding_window
        if family == "qwen2":
            config["use_sliding_window"] = True
    (source / "config.json").write_text(json.dumps(config))
    if sharded:
        (source / "model.safetensors").unlink()
    save_mlx_fixture(source, tensors, sharded=sharded)
    return config, tensors


@pytest.mark.parametrize(
    "family,bits,group_size,tied", list(itertools.product(PLANNERS, (4, 8), (32, 64), (False, True)))
)
def test_complete_mlx_export_repacks_without_requantizing(tmp_path, family, bits, group_size, tied):
    source = tmp_path / "source"
    _, tensors = mlx_family_fixture(source, family, bits=bits, group_size=group_size, tied=tied, sharded=tied)
    plan = PLANNERS[family](source)
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    metadata, converted = read_safetensors_header(tmp_path / "output.safetensors")
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    # A tied lm_head is checked against the embedding and never copied.
    read = {info for operation in plan.operations for info in operation.sources}
    assert read <= set(plan.input_tensors)
    assert len(set(plan.input_tensors) - read) == (3 if tied else 0)
    specs = {key: json.loads(value) for key, value in metadata.items() if key.endswith(".spec")}
    assert specs and all(
        spec["type"] == "MLXSpec" and spec["bits"] == bits and spec["group_size"] == group_size
        for spec in specs.values()
    )
    if tied:
        assert "decoder.embedding.output_embedding.weights" not in converted
        assert specs["decoder.embedding.embedding.spec"]["layout"] == "input_output"
    else:
        assert specs["decoder.embedding.output_embedding.spec"]["layout"] == "output_input"
        assert specs["decoder.embedding.input_embedding.spec"]["layout"] == "input_output"
    for suffix, names in (
        (".mixer.qkvg_projection.weights.weights", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        (".mlp.up_projection.weights.weights", ("mlp.up_proj", "mlp.gate_proj")),
    ):
        assert tensor_bytes(converted[PREFIX + suffix]) == b"".join(
            tensors["model.layers.0." + name + ".weight"][1].tobytes() for name in names
        )
    mixer = plan.decoder_config["transformer_config"]["layer_configs"][0]["mixer_config"]
    assert mixer["has_qkvg_biases"] is (family == "qwen2") and mixer["has_out_biases"] is False
    assert (PREFIX + ".mixer.qkvg_projection.biases" in converted) is (family == "qwen2")
    assert PREFIX + ".mixer.out_projection.biases" not in converted


def test_qwen2_attention_biases_are_separate_from_dequantization_biases(tmp_path):
    source = tmp_path / "source"
    _, tensors = mlx_family_fixture(source, "qwen2")
    write_plan(build_qwen2_plan(source), tmp_path / "output.safetensors", chunk_bytes=96)
    converted = read_safetensors_header(tmp_path / "output.safetensors")[1]
    names = ("q_proj", "k_proj", "v_proj")
    biases = converted[PREFIX + ".mixer.qkvg_projection.biases"]
    assert (biases.dtype, list(biases.shape)) == ("BF16", [256])
    assert tensor_bytes(biases) == b"".join(tensors[f"model.layers.0.self_attn.{n}.bias"][1].tobytes() for n in names)
    # The affine correction planes are repacked in the group-output layout,
    # under the matrix's own subtree, never in place of the attention biases.
    correction = converted[PREFIX + ".mixer.qkvg_projection.weights.biases"]
    assert list(correction.shape) == [128 // 32, 256]


@pytest.mark.parametrize("family", PLANNERS)
def test_sliding_window_layout_is_independent_of_weight_representation(tmp_path, family):
    source = tmp_path / "source"
    mlx_family_fixture(source, family, sliding_window=16)
    layers = PLANNERS[family](source).decoder_config["transformer_config"]["layer_configs"]
    assert [layer["mixer_config"]["sliding_window_size"] for layer in layers] == [16]


def test_qwen2_mlx_checkpoint_still_requires_attention_biases(tmp_path):
    source = tmp_path / "source"
    _, tensors = mlx_family_fixture(source, "qwen2")
    del tensors["model.layers.0.self_attn.k_proj.bias"]
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError, match="missing a required tensor"):
        build_qwen2_plan(source)


@pytest.mark.parametrize("family,plane", list(itertools.product(PLANNERS, ("weight", "scales", "biases"))))
def test_tied_mlx_duplicate_difference_in_any_plane_is_rejected(tmp_path, family, plane):
    source = tmp_path / "source"
    _, tensors = mlx_family_fixture(source, family, tied=True)
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
    _, tensors = mlx_family_fixture(source, family, tied=True)
    for plane in ("weight", "scales", "biases"):
        if partial and plane == "scales":
            continue
        del tensors["lm_head." + plane]
    save_mlx_fixture(source, tensors)
    if partial:
        with pytest.raises(ConversionError, match="missing a matrix plane"):
            PLANNERS[family](source)
    else:
        assert PLANNERS[family](source).source_tensors == len(tensors)


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize(
    "case",
    (
        "missing_plane",
        "shape",
        "parameter_dtype",
        "fusion_dense",
        "extra",
        "no_config",
        "awq",
        "config_disagreement",
        "five_bit",
        "six_bit",
        "module_override",
    ),
)
def test_malformed_or_unsupported_mlx_sources_fail_closed(tmp_path, family, case):
    source = tmp_path / "source"
    config, tensors = mlx_family_fixture(source, family)
    prefix = "model.layers.0.self_attn.q_proj"
    error = ConversionError
    match = None
    if case == "missing_plane":
        del tensors[prefix + ".biases"]
        match = "missing a matrix plane"
    elif case == "shape":
        tensors[prefix + ".weight"] = "I32", np.zeros((128, 15), dtype="<i4")
        match = "packed weight shape"
    elif case == "parameter_dtype":
        tensors[prefix + ".scales"] = "F32", _float_values((128, 4), "F32")
        error = UnsupportedConversion
    elif case == "fusion_dense":
        tensors[prefix + ".weight"] = "BF16", _float_values((128, 128), "BF16")
        for plane in ("scales", "biases"):
            del tensors[prefix + "." + plane]
        error, match = UnsupportedConversion, "Mixed full-precision/MLX fusion"
    elif case == "extra":
        tensors["model.unused.scales"] = "BF16", np.zeros(128, dtype="<u2")
    elif case == "no_config":
        del config["quantization"]
        del config["quantization_config"]
        error, match = UnsupportedConversion, "requires explicit quantization"
    elif case == "awq":
        config["quantization_config"]["quant_method"] = "awq"
        error = UnsupportedConversion
    elif case == "config_disagreement":
        config["quantization_config"]["bits"] = 8
    elif case in ("five_bit", "six_bit"):
        for key in ("quantization", "quantization_config"):
            config[key]["bits"] = 5 if case == "five_bit" else 6
        error, match = UnsupportedConversion, "unsupported bit width"
    else:
        for key in ("quantization", "quantization_config"):
            config[key]["model.layers.0.mlp.down_proj"] = {"bits": 8, "group_size": 32}
        error, match = UnsupportedConversion, "unimplemented options"
    save_mlx_fixture(source, tensors)
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(error, match=match):
        PLANNERS[family](source)


@pytest.mark.parametrize("family", PLANNERS)
def test_quantization_metadata_requires_actual_packed_matrix(tmp_path, family):
    from test_dense_adapters import dense_fixture

    source = tmp_path / "source"
    config, _ = dense_fixture(source, family)
    config["quantization"] = {"bits": 4, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="no packed checkpoint matrices"):
        PLANNERS[family](source)


@pytest.mark.parametrize("family,bits", list(itertools.product(PLANNERS, (4, 8))))
def test_mlx_preflight_is_implemented_but_unqualified(family, bits):
    config = {"model_type": family, "quantization": {"bits": bits, "group_size": 64}}
    report = preflight_model(config, mode="text")
    assert report["status"] == "conditional" and report["qualification"]["status"] == "not_assessed"


@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize(
    "quantization",
    ({"bits": 6, "group_size": 64}, {"bits": 4, "group_size": 64, "lm_head": {"bits": 8, "group_size": 64}}),
)
def test_mlx_preflight_keeps_unimplemented_widths_and_overrides_unsupported(family, quantization):
    report = preflight_model({"model_type": family, "quantization": quantization}, mode="text")
    assert report["reasons"] == ["unsupported_weight_format"]


@pytest.fixture
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _: [])
    monkeypatch.setattr(catalog, "get_model", lambda _: None)


@pytest.mark.usefixtures("isolated_catalog")
@pytest.mark.parametrize("family", PLANNERS)
@pytest.mark.parametrize("bad_packing", (False, True))
def test_catalogue_admits_mlx_rows_from_pinned_headers(tmp_path, family, bad_packing):
    source = tmp_path / "source"
    mlx_family_fixture(source, family, bits=4, group_size=32)
    if bad_packing:
        # The bytes remain W4/group32 while both declarations claim W8/group32.
        path = source / "config.json"
        config = json.loads(path.read_bytes())
        for key in ("quantization", "quantization_config"):
            config[key]["bits"] = 8
        path.write_text(json.dumps(config))
    hub, _, _ = add_checkpoint(tmp_path, source, family=family)

    result = catalog.family_catalog(family, hub=hub)

    assert result["status"] == "ready" and len(result["items"]) == 1
    item = result["items"][0]
    assert item["downloadable"] is (not bad_packing)
    assert bool(item.get("unavailable_reason")) is bad_packing
    assert not [event for event in hub.events if event[0] == "download" and event[-1].endswith(".safetensors")]
