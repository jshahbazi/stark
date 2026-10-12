"""Independent frozen LFM2 decoder exports and explicit unsupported semantics."""

import copy
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.lfm2 import build_lfm2_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "lfm2_exports.json").read_text())
DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def lfm2_fixture(source, case="mixed_tied", *, dtype="BF16", sharded=False):
    source.mkdir()
    reference = REFERENCE["cases"][case]
    config = copy.deepcopy(reference["config"])
    config["torch_dtype"] = DTYPES[dtype]
    model = config["hidden_size"]
    vocab = config["vocab_size"]
    head = model // config["num_attention_heads"]
    kv = head * config["num_key_value_heads"]
    hidden = reference["decoder_config"]["transformer_config"]["hidden_dim"]
    layer_types = config.get("layer_types") or [
        "full_attention" if i in config["full_attn_idxs"] else "conv" for i in range(config["num_hidden_layers"])
    ]
    tensors = {}

    def add(name, shape):
        values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
        raw = {
            "BF16": lambda x: x.astype(ml_dtypes.bfloat16).view("<u2"),
            "F16": lambda x: x.astype("<f2"),
            "F32": lambda x: x,
        }[dtype](values)
        tensors[name] = (dtype, raw)

    add("model.embed_tokens.weight", (vocab, model))
    if not config.get("tie_embedding", True):
        add("lm_head.weight", (vocab, model))
    add("model.embedding_norm.weight", (model,))
    for i, layer_type in enumerate(layer_types):
        src = f"model.layers.{i}"
        add(src + ".operator_norm.weight", (model,))
        add(src + ".ffn_norm.weight", (model,))
        if layer_type == "conv":
            add(src + ".conv.in_proj.weight", (3 * model, model))
            add(src + ".conv.out_proj.weight", (model, model))
            shape = {
                "flat": (model, config["conv_L_cache"]),
                "pytorch": (model, 1, config["conv_L_cache"]),
                "trailing": (model, config["conv_L_cache"], 1),
            }[reference["conv_shape"]]
            add(src + ".conv." + reference["conv_alias"], shape)
            if config["conv_bias"]:
                add(src + ".conv." + reference["conv_alias"].replace("weight", "bias"), (model,))
        else:
            for projection, width in (("q", model), ("k", kv), ("v", kv)):
                add(src + f".self_attn.{projection}_proj.weight", (width, model))
            add(src + ".self_attn." + reference["out_alias"] + ".weight", (model, model))
            for projection in ("q", "k"):
                add(src + f".self_attn.{projection}_{reference['norm_alias']}.weight", (head,))
        add(src + ".feed_forward.w3.weight", (hidden, model))
        add(src + ".feed_forward.w1.weight", (hidden, model))
        add(src + ".feed_forward.w2.weight", (model, hidden))
    (source / "config.json").write_text(json.dumps(config))
    if sharded:
        mapping = {}
        for index in range(2):
            filename = f"model-{index}.safetensors"
            contents = dict(list(tensors.items())[index::2])
            save_tensors(source / filename, contents)
            mapping.update(dict.fromkeys(contents, filename))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_tensors(source / "model.safetensors", tensors)
    return config, tensors


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_lfm2_complete_decoder_config_matches_actual_external_importer(tmp_path, case):
    source = tmp_path / "source"
    lfm2_fixture(source, case)
    assert build_lfm2_plan(source).decoder_config == REFERENCE["cases"][case]["decoder_config"]


@pytest.mark.parametrize("case", REFERENCE["cases"])
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
@pytest.mark.parametrize("sharded", [False, True])
def test_lfm2_actual_external_decoder_exports_match_with_bounded_workspace(
    tmp_path, case, source_dtype, target_dtype, sharded
):
    source = tmp_path / "source"
    _, inputs = lfm2_fixture(source, case, dtype=source_dtype, sharded=sharded)
    frozen_inputs = REFERENCE["cases"][case]["inputs"][source_dtype]
    assert set(inputs) == set(frozen_inputs)
    for name, (dtype, raw) in inputs.items():
        expected_input = frozen_inputs[name]
        assert (DTYPES[dtype], list(raw.shape)) == (expected_input["dtype"], expected_input["shape"])
        assert hashlib.sha256(raw.tobytes()).hexdigest() == expected_input["sha256"]
    plan = build_lfm2_plan(source, dtype=DTYPES[target_dtype])
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, tensors = read_safetensors_header(output)
    frozen = REFERENCE["cases"][case]["exports"][source_dtype + "_" + target_dtype]
    assert set(tensors) == set(frozen["arrays"])
    for name, tensor in tensors.items():
        reference = frozen["arrays"][name]
        assert (DTYPES[tensor.dtype], list(tensor.shape)) == (reference["dtype"], reference["shape"])
        assert hashlib.sha256(tensor_bytes(tensor)).hexdigest() == reference["sha256"]
    assert {key: json.loads(value) for key, value in metadata.items()} == frozen["metadata"]
    assert plan.source_tensors == len(inputs)
    assert {tensor for op in plan.operations for tensor in op.sources} == set(plan.input_tensors)
    assert stats["peak_buffer_bytes"] <= 96
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def rewrite(source, config, tensors=None):
    (source / "config.json").write_text(json.dumps(config))
    if tensors is not None:
        save_tensors(source / "model.safetensors", tensors)


@pytest.mark.parametrize("corruption", ["missing", "shape", "extra", "head", "alias"])
def test_lfm2_malformed_recognized_layout_precedes_unknown_semantics(tmp_path, corruption):
    source = tmp_path / "source"
    config, tensors = lfm2_fixture(source)
    config["unsupported_semantics"] = True
    if corruption == "missing":
        tensors.pop("model.layers.1.feed_forward.w2.weight")
    elif corruption == "shape":
        tensors["model.layers.0.conv.conv.weight"] = ("BF16", np.zeros((8, 2), dtype="<u2"))
    elif corruption == "extra":
        tensors["extra.weight"] = ("BF16", np.zeros(2, dtype="<u2"))
    elif corruption == "head":
        tensors["lm_head.weight"] = ("BF16", np.zeros((16, 8), dtype="<u2"))
    else:
        tensors["model.layers.0.conv.conv_weight"] = tensors["model.layers.0.conv.conv.weight"]
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError):
        build_lfm2_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown_option": True},
        {"architectures": ["DifferentForCausalLM"]},
        {"block_use_swiglu": False},
        {"use_pos_enc": False},
        {"block_dim": 16},
        {"conv_dim": 16},
        {"conv_dim_out": 16},
        {"full_attn_idxs": [0]},
        {"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "linear"}},
        {"rope_parameters": {"rope_theta": 1000000.0, "factor": 2.0}},
    ],
)
def test_lfm2_semantics_outside_explicit_dense_coverage_fail_closed(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = lfm2_fixture(source)
    config.update(changes)
    rewrite(source, config)
    with pytest.raises(UnsupportedConversion):
        build_lfm2_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": True},
        {"num_heads": 2},
        {"num_attention_heads": 3},
        {"num_key_value_heads": 3},
        {"conv_L_cache": 0},
        {"block_multiple_of": 0},
        {"block_ff_dim": 0},
        {"intermediate_size": False},
        {"block_ffn_dim_multiplier": float("nan")},
        {"norm_eps": 0},
        {"block_norm_eps": float("inf")},
        {"block_auto_adjust_ff_dim": "true"},
        {"block_use_swiglu": None},
        {"block_use_swiglu": False, "use_pos_enc": None},
        {"conv_bias": 1},
        {"layer_types": ["conv"]},
        {"layer_types": ["conv", "mamba"]},
        {"layer_types": None, "full_attn_idxs": None},
        {"full_attn_idxs": [True]},
        {"full_attn_idxs": [1, 1]},
        {"full_attn_idxs": [-1]},
        {"full_attn_idxs": [2]},
        {"full_attn_idxs": "1"},
        {"rope_theta": None},
        {"rope_theta": -1},
        {"rope_parameters": []},
        {"rope_parameters": {}},
        {"rope_parameters": {"rope_theta": 500000.0}},
        {"quantization": {}},
        {"initializer_range": float("inf")},
    ],
)
def test_lfm2_malformed_configs_are_errors(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = lfm2_fixture(source)
    config.update(changes)
    rewrite(source, config)
    with pytest.raises(ConversionError):
        build_lfm2_plan(source)


def test_lfm2_required_structural_flags_are_explicit(tmp_path):
    source = tmp_path / "source"
    config, _ = lfm2_fixture(source)
    config.pop("block_use_swiglu")
    rewrite(source, config)
    with pytest.raises(ConversionError, match="structural"):
        build_lfm2_plan(source)


def test_lfm2_tied_head_duplicate_is_verified_and_preserved_in_input_checks(tmp_path):
    source = tmp_path / "source"
    config, tensors = lfm2_fixture(source)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    rewrite(source, config, tensors)
    plan = build_lfm2_plan(source)
    assert plan.source_tensors == len(tensors)
    assert any(tensor.shape == (16, 8) for tensor in plan.input_tensors)
    head_sources = [tensor for op in plan.operations for tensor in op.sources]
    assert len(head_sources) == len(plan.input_tensors) - 1


@pytest.mark.parametrize("name", ["model.layers.1.self_attn.q_proj.weight", "model.layers.0.feed_forward.w3.weight"])
def test_lfm2_mixed_fused_dtypes_fail_after_complete_validation(tmp_path, name):
    source = tmp_path / "source"
    config, tensors = lfm2_fixture(source)
    _, raw = tensors[name]
    tensors[name] = ("F32", (raw.astype("<u4") << 16).view("<f4"))
    rewrite(source, config, tensors)
    with pytest.raises(UnsupportedConversion, match="Mixed-dtype fusion"):
        build_lfm2_plan(source)
    tensors.pop("model.layers.1.feed_forward.w2.weight")
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError, match="missing"):
        build_lfm2_plan(source)


def test_lfm2_consistent_attention_indices_have_membership_semantics(tmp_path):
    source = tmp_path / "source"
    config, _ = lfm2_fixture(source, "all_attention")
    before = build_lfm2_plan(source).decoder_config
    config["full_attn_idxs"] = [1, 0]
    rewrite(source, config)
    assert build_lfm2_plan(source).decoder_config == before


def test_lfm2_native_minimum_kernel_fails_closed_with_valid_checkpoint(tmp_path):
    source = tmp_path / "source"
    config, tensors = lfm2_fixture(source)
    config["conv_L_cache"] = 1
    tensors["model.layers.0.conv.conv.weight"] = ("BF16", np.zeros((8, 1, 1), dtype="<u2"))
    rewrite(source, config, tensors)
    with pytest.raises(UnsupportedConversion, match="at least two"):
        build_lfm2_plan(source)


def test_lfm2_original_sources_rechecked_after_a_plan_is_created(tmp_path):
    source = tmp_path / "source"
    config, tensors = lfm2_fixture(source)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    rewrite(source, config, tensors)
    plan = build_lfm2_plan(source)
    tensors["lm_head.weight"] = ("BF16", np.ones((16, 8), dtype="<u2"))
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "converted.safetensors", chunk_bytes=96)
