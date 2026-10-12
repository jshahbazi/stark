"""Actual pinned LLaMBA decoder exports and strict independent planning."""

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.llamba import build_llamba_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
CASES = {
    "backbone_tied": {},
    "backbone_untied": {"tied": False},
    "backbone_biased": {"tied": False, "biased": True},
    "alternating_tied": {"alternating": True},
    "alternating_untied": {"alternating": True, "tied": False},
    "flat_conv": {"conv_shape": "flat", "conv_alias": "conv_weight"},
    "trailing_conv": {"conv_shape": "trailing", "conv_alias": "conv.weight"},
    "grouped_expansion": {"expand": 2, "groups": 1},
    "mlx4_group16": {"alternating": True, "bits": 4, "group_size": 16},
    "mlx4_group32": {"alternating": True, "bits": 4, "group_size": 32, "tied": False},
    "mlx8_group16": {"alternating": True, "bits": 8, "group_size": 16},
    "mlx4_group128": {"alternating": True, "bits": 4, "group_size": 128, "model": 128},
}
REFERENCE_PATH = Path(__file__).parent / "fixtures" / "llamba_exports.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())


def llamba_fixture(source, case="backbone_tied", *, dtype="BF16", sharded=False):
    source.mkdir()
    flags = CASES[case]
    model, vocab = flags.get("model", 32), 64
    hidden = 2 * model
    tied, alternating, biased = flags.get("tied", True), flags.get("alternating", False), flags.get("biased", False)
    config = {
        "model_type": "llamba",
        "vocab_size": vocab,
        "d_model": model,
        "n_layer": 2,
        "tie_embeddings": tied,
        "pad_vocab_size_multiple": 8,
        "lm_head_bias": False,
        "resid_dropout": 0.0,
        "norm_epsilon": 1e-5,
        "torch_dtype": DTYPES[dtype],
        "mlp_cfg": {"intermediate_size": hidden, "bias": biased, "act_fn": "silu"},
        "ssm_cfg": {
            "d_state": 4,
            "n_v_heads": 4,
            "n_qk_heads": flags.get("groups", 2),
            "expand": flags.get("expand", 1),
            "activation": "identity",
            "bias": biased,
            "chunk_size": 128,
        },
    }
    bits = flags.get("bits")
    if bits:
        config["quantization"] = {"bits": bits, "group_size": flags["group_size"]}
    tensors = {}

    def add(name, shape):
        values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
        raw = {
            "BF16": lambda x: x.astype(ml_dtypes.bfloat16).view("<u2"),
            "F16": lambda x: x.astype("<f2"),
            "F32": lambda x: x,
        }[dtype](values)
        tensors[name] = (dtype, raw)

    def matrix(name, shape, *, packed=True):
        if not bits or not packed:
            add(name + ".weight", shape)
            return
        rows, columns = shape
        values = np.arange(rows * columns, dtype="<u4").reshape(rows, columns) % (1 << bits)
        weights = np.zeros((rows, columns * bits // 32), dtype="<u4")
        for slot in range(32 // bits):
            weights |= values[:, slot :: 32 // bits] << (slot * bits)
        tensors[name + ".weight"] = ("U32", weights)
        add(name + ".scales", (rows, columns // flags["group_size"]))
        add(name + ".biases", (rows, columns // flags["group_size"]))

    embedding = "embedding.encoder" if alternating else "backbone.embedding"
    matrix(embedding, (vocab, model), packed=False)
    if not tied or bits:
        matrix("head.linear" if alternating else "lm_head", (vocab, model))
    add("model.norm.weight" if alternating else "backbone.final_layernorm.weight", (model,))
    state, heads, groups = (
        config["ssm_cfg"]["d_state"],
        config["ssm_cfg"]["n_v_heads"],
        config["ssm_cfg"]["n_qk_heads"],
    )
    inner = config["ssm_cfg"]["expand"] * model
    conv = inner + 2 * groups * state
    for index in range(config["n_layer"]):
        if alternating:
            mixer_layer, mlp_layer = f"model.layers.{2 * index}", f"model.layers.{2 * index + 1}"
            mixer, mlp = mixer_layer + ".layer", mlp_layer + ".layer"
            add(mixer_layer + ".norm.weight", (model,))
            add(mlp_layer + ".norm.weight", (model,))
        else:
            mixer_layer = f"backbone.layers.{index}"
            mixer, mlp = mixer_layer + ".mixer", mixer_layer + ".mlp"
            add(mixer_layer + ".input_layernorm.weight", (model,))
            add(mixer_layer + ".post_attention_layernorm.weight", (model,))
        matrix(mixer + ".in_proj", (conv + inner + heads, model))
        matrix(mixer + ".out_proj", (model, inner))
        if biased:
            add(mixer + ".in_proj.bias", (conv + inner + heads,))
            add(mixer + ".out_proj.bias", (model,))
        alias = flags.get("conv_alias", "conv1d.weight")
        conv_shape = {
            "flat": (conv, 4),
            "pytorch": (conv, 1, 4),
            "trailing": (conv, 4, 1),
        }[flags.get("conv_shape", "flat" if bits else "pytorch")]
        add(mixer + "." + alias, conv_shape)
        add(mixer + "." + alias.replace("weight", "bias"), (conv,))
        add(mixer + ".D", (heads,))
        add(mixer + ".z_bias", (inner,))
        up_keys, down = (
            (("gate_proj", "in_proj"), "out_proj") if alternating else (("up_proj", "gate_proj"), "down_proj")
        )
        for key in up_keys:
            matrix(mlp + "." + key, (hidden, model))
            if biased:
                add(mlp + "." + key + ".bias", (hidden,))
        matrix(mlp + "." + down, (model, hidden))
        if biased:
            add(mlp + "." + down + ".bias", (model,))
    (source / "config.json").write_text(json.dumps(config))
    if sharded:
        mapping = {}
        for shard in range(2):
            filename = f"model-{shard}.safetensors"
            contents = dict(list(tensors.items())[shard::2])
            save_tensors(source / filename, contents)
            mapping.update(dict.fromkeys(contents, filename))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_tensors(source / "model.safetensors", tensors)
    return config, tensors


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
@pytest.mark.parametrize("sharded", [False, True])
def test_llamba_actual_decoder_exports_match_all_precision_pairs(tmp_path, case, source_dtype, target_dtype, sharded):
    source = tmp_path / "source"
    _, inputs = llamba_fixture(source, case, dtype=source_dtype, sharded=sharded)
    frozen = REFERENCE["cases"][f"{case}_{source_dtype}_{target_dtype}"]
    plan = build_llamba_plan(source, dtype=DTYPES[target_dtype])
    assert plan.decoder_config == frozen["decoder_config"]
    assert set(inputs) == set(frozen["inputs"])
    for name, (dtype, raw) in inputs.items():
        assert {
            "dtype": dtype,
            "shape": list(raw.shape),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        } == frozen["inputs"][name]
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=128)
    metadata, tensors = read_safetensors_header(output)
    assert set(tensors) == set(frozen["arrays"])
    for name, info in tensors.items():
        assert {
            "dtype": info.dtype,
            "shape": list(info.shape),
            "sha256": hashlib.sha256(tensor_bytes(info)).hexdigest(),
        } == frozen["arrays"][name]
    assert {name: json.loads(value) for name, value in metadata.items()} == frozen["metadata"]
    assert plan.source_tensors == len(inputs)
    assert len(plan.input_tensors) == len(inputs)
    assert stats["peak_buffer_bytes"] <= 128
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def rewrite(source, config, tensors=None):
    (source / "config.json").write_text(json.dumps(config))
    if tensors is not None:
        save_tensors(source / "model.safetensors", tensors)


@pytest.mark.parametrize("corruption", ["D", "z_bias", "conv", "alias", "shape", "extra", "head", "bias"])
def test_llamba_malformed_tensor_contract_precedes_unknown_semantics(tmp_path, corruption):
    source = tmp_path / "source"
    config, tensors = llamba_fixture(source)
    config["unknown_semantics"] = True
    prefix = "backbone.layers.0.mixer"
    if corruption in {"D", "z_bias"}:
        tensors.pop(prefix + "." + corruption)
    elif corruption == "conv":
        tensors.pop(prefix + ".conv1d.weight")
    elif corruption == "alias":
        tensors[prefix + ".conv_weight"] = tensors[prefix + ".conv1d.weight"]
    elif corruption == "shape":
        tensors[prefix + ".in_proj.weight"] = ("BF16", np.zeros((2, 2), dtype="<u2"))
    elif corruption == "extra":
        tensors["extra.weight"] = ("BF16", np.zeros(2, dtype="<u2"))
    elif corruption == "head":
        tensors["lm_head.weight"] = ("BF16", np.zeros((64, 32), dtype="<u2"))
    else:
        tensors[prefix + ".in_proj.bias"] = ("BF16", np.zeros(84, dtype="<u2"))
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError):
        build_llamba_plan(source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("d_model", True),
        ("n_layer", 0),
        ("vocab_size", -1),
        ("pad_vocab_size_multiple", False),
        ("norm_epsilon", 0),
        ("resid_dropout", float("inf")),
        ("resid_dropout", -1),
        ("resid_dropout", 1),
        ("tie_embeddings", "true"),
        ("lm_head_bias", 1),
        ("mlp_cfg", []),
        ("ssm_cfg", []),
    ],
)
def test_llamba_structural_fields_are_validated(tmp_path, key, value):
    source = tmp_path / "source"
    config, _ = llamba_fixture(source)
    config[key] = value
    rewrite(source, config)
    with pytest.raises(ConversionError):
        build_llamba_plan(source)


@pytest.mark.parametrize(
    "section,key",
    [("mlp_cfg", "bias"), ("mlp_cfg", "act_fn"), ("ssm_cfg", "bias"), ("ssm_cfg", "d_state"), (None, "resid_dropout")],
)
def test_llamba_required_structural_fields_are_explicit(tmp_path, section, key):
    source = tmp_path / "source"
    config, _ = llamba_fixture(source)
    (config if section is None else config[section]).pop(key)
    rewrite(source, config)
    with pytest.raises(ConversionError):
        build_llamba_plan(source)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "unimplemented", True),
        ("mlp_cfg", "act_fn", "gelu"),
        ("mlp_cfg", "unimplemented", True),
        ("ssm_cfg", "activation", "silu"),
        ("ssm_cfg", "unimplemented", True),
        (None, "lm_head_bias", True),
    ],
)
def test_llamba_unimplemented_semantics_fail_closed(tmp_path, section, key, value):
    source = tmp_path / "source"
    config, _ = llamba_fixture(source)
    (config if section is None else config[section])[key] = value
    rewrite(source, config)
    with pytest.raises(UnsupportedConversion):
        build_llamba_plan(source)


@pytest.mark.parametrize("case", ["backbone_tied", "alternating_tied", "mlx4_group16"])
def test_llamba_tied_auxiliary_head_is_retained_in_source_mutation_checks(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = llamba_fixture(source, case)
    head = "head.linear.weight" if CASES[case].get("alternating") else "lm_head.weight"
    if not CASES[case].get("bits"):
        tensors[head] = tensors[
            "embedding.encoder.weight" if CASES[case].get("alternating") else "backbone.embedding.weight"
        ]
        rewrite(source, config, tensors)
    plan = build_llamba_plan(source)
    tensors[head] = (tensors[head][0], np.zeros_like(tensors[head][1]))
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "converted.safetensors", chunk_bytes=128)


def test_llamba_external_config_does_not_read_unselected_primary_config(tmp_path):
    source = tmp_path / "source"
    config, _ = llamba_fixture(source)
    plan = build_llamba_plan(source)
    (source / "config.json").unlink()
    assert build_llamba_plan(source, config=config) == plan


def test_llamba_catalog_mlx_bit_width_infers_and_validates_planes_without_primary_config(tmp_path):
    source = tmp_path / "source"
    config, _ = llamba_fixture(source, "mlx4_group16")
    expected = build_llamba_plan(source)
    config.pop("quantization")
    (source / "config.json").unlink()
    assert build_llamba_plan(source, config=config, expected_mlx_bits=4) == expected
    with pytest.raises(ConversionError):
        build_llamba_plan(source, config=config, expected_mlx_bits=8)
    with pytest.raises(UnsupportedConversion, match="explicit quantization"):
        build_llamba_plan(source, config=config)


def test_llamba_generic_packed_auxiliary_tied_head_requires_explicit_quantization(tmp_path):
    source = tmp_path / "source"
    config, _ = llamba_fixture(source, "mlx4_group16")
    assert build_llamba_plan(source).source_tensors > 0
    config.pop("quantization")
    rewrite(source, config)
    with pytest.raises(UnsupportedConversion, match="explicit quantization"):
        build_llamba_plan(source)


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "group", "bits", "extra"])
def test_llamba_mlx_auxiliary_planes_are_not_ignored(tmp_path, corruption):
    source = tmp_path / "source"
    config, tensors = llamba_fixture(source, "mlx4_group16")
    if corruption == "missing":
        tensors.pop("head.linear.biases")
    elif corruption == "shape":
        tensors["head.linear.scales"] = ("BF16", np.zeros((64, 1), dtype="<u2"))
    elif corruption == "dtype":
        tensors["head.linear.scales"] = ("U32", np.zeros((64, 2), dtype="<u4"))
    elif corruption == "group":
        config["quantization"]["group_size"] = 32
    elif corruption == "bits":
        config["quantization"]["bits"] = 8
    else:
        tensors["head.linear.extra"] = ("BF16", np.zeros(2, dtype="<u2"))
    rewrite(source, config, tensors)
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_llamba_plan(source)


def test_llamba_quantized_auxiliary_head_alone_does_not_qualify_model_as_mlx(tmp_path):
    source = tmp_path / "source"
    config, tensors = llamba_fixture(source, "mlx4_group16")
    for name in tuple(tensors):
        if name.endswith(".scales") and not name.startswith("head.linear."):
            prefix = name.removesuffix(".scales")
            _, raw = tensors[prefix + ".weight"]
            tensors[prefix + ".weight"] = ("BF16", np.zeros((raw.shape[0], raw.shape[1] * 8), dtype="<u2"))
            tensors.pop(prefix + ".scales")
            tensors.pop(prefix + ".biases")
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError, match="no packed checkpoint matrices"):
        build_llamba_plan(source)


def test_llamba_lossy_packed_tied_head_omission_requires_alternating_layout(tmp_path):
    source = tmp_path / "source"
    config, tensors = llamba_fixture(source, "mlx4_group16")
    renamed = {}
    for name, value in tensors.items():
        if name.startswith("embedding.encoder."):
            destination = name.replace("embedding.encoder.", "backbone.embedding.")
        elif name.startswith("head.linear."):
            destination = name.replace("head.linear.", "lm_head.")
        elif name.startswith("model.norm."):
            destination = name.replace("model.norm.", "backbone.final_layernorm.")
        else:
            _, _, index, *suffix = name.split(".")
            layer = int(index)
            rest = ".".join(suffix)
            if layer % 2:
                rest = (
                    rest.replace("norm.", "post_attention_layernorm.", 1)
                    if rest.startswith("norm.")
                    else rest.replace("layer.gate_proj.", "mlp.up_proj.")
                    .replace("layer.in_proj.", "mlp.gate_proj.")
                    .replace("layer.out_proj.", "mlp.down_proj.")
                )
            else:
                rest = (
                    rest.replace("norm.", "input_layernorm.", 1)
                    if rest.startswith("norm.")
                    else rest.replace("layer.", "mixer.", 1)
                )
            destination = f"backbone.layers.{layer // 2}.{rest}"
        renamed[destination] = value
    rewrite(source, config, renamed)
    with pytest.raises(UnsupportedConversion, match="declared alternating layout"):
        build_llamba_plan(source)
