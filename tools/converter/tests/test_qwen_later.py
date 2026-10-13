"""Pinned actual-loader export parity for later dense Qwen and Qwen3.5 MoE."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from test_qwen35 import DTYPES, qwen35_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.qwen35 import build_qwen35_moe_plan, build_qwen35_plan
from stark_convert.streaming import (
    ConversionError,
    TensorInfo,
    UnsupportedConversion,
    read_safetensors_header,
    write_plan,
)

REFERENCE = Path(__file__).parent / "fixtures" / "qwen_later_exports.json"


def qwen_later_fixture(path, *, dtype="BF16", profile="later-dense", suffix=False, sharded=False):
    """Scaled declared wrapper/MTP inputs with distinguishable expert and gate rows."""
    config, tensors = qwen35_fixture(path, dtype=dtype, wrapper=True, mtp=True)
    text = config["text_config"]
    if profile == "later-dense":
        config["language_model_only"] = False
        text["output_gate_type"] = "swish"
        text["partial_rotary_factor"] = 1.0
    else:
        config["model_type"] = "qwen3_5_moe"
        config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        text["model_type"] = "qwen3_5_moe_text"
        text.pop("intermediate_size")
        text.update(num_experts=3, num_experts_per_tok=2, moe_intermediate_size=12, shared_expert_intermediate_size=12)
        text.update(output_router_logits=False, router_aux_loss_coef=0.001)
        if profile == "later-moe":
            config["language_model_only"] = True
            text["partial_rotary_factor"] = 1.0
        for layer in range(2):
            src = f"model.language_model.layers.{layer}.mlp"
            for branch in ("up", "gate", "down"):
                del tensors[src + f".{branch}_proj.weight"]

            def add(name, shape):
                values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
                raw = {
                    "BF16": lambda x: (x.view("<u4") >> 16).astype("<u2"),
                    "F16": lambda x: x.astype("<f2"),
                    "F32": lambda x: x,
                }[dtype](values)
                tensors[name] = (dtype, raw)

            add(src + ".gate.weight", (3, 8))
            add(src + ".shared_expert_gate.weight", (1, 8))
            add(src + ".experts.gate_up_proj" + (".weight" if suffix else ""), (3, 24, 8))
            add(src + ".experts.down_proj" + (".weight" if suffix else ""), (3, 8, 12))
            for branch, shape in (("up", (12, 8)), ("gate", (12, 8)), ("down", (8, 12))):
                add(src + f".shared_expert.{branch}_proj.weight", shape)
    (path / "config.json").write_text(json.dumps(config))
    if sharded:
        (path / "model.safetensors").unlink()
        weight_map = {}
        for shard in range(3):
            name = f"model-{shard}.safetensors"
            contents = dict(list(tensors.items())[shard::3])
            save_tensors(path / name, contents)
            weight_map.update(dict.fromkeys(contents, name))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_tensors(path / "model.safetensors", tensors)
    return config, tensors


def later_case_name(source, target, profile, suffix):
    return f"{source}-{target}-{profile}-s{int(suffix)}"


CASES = [
    (source, target, profile, suffix)
    for source, target in itertools.product(DTYPES, DTYPES)
    for profile, suffix in (("later-dense", False), ("moe", False), ("moe", True), ("later-moe", False))
]


@pytest.mark.parametrize("source_dtype,target,profile,suffix", CASES)
def test_complete_exports_match_pinned_lalamo(tmp_path, source_dtype, target, profile, suffix):
    source = tmp_path / "source"
    _, tensors = qwen_later_fixture(source, dtype=source_dtype, profile=profile, suffix=suffix, sharded=True)
    expected = json.loads(REFERENCE.read_text())["cases"][later_case_name(source_dtype, target, profile, suffix)]
    builder = build_qwen35_plan if profile == "later-dense" else build_qwen35_moe_plan
    plan = builder(source, dtype=DTYPES[target])
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(array.nbytes for _, array in tensors.values())
    assert plan.input_tensors and len(plan.input_tensors) == len(tensors)
    assert plan.decoder_config == expected["decoder_config"]
    assert {key: json.loads(value) for key, value in metadata.items()} == expected["metadata"]
    assert set(converted) == set(expected["arrays"])
    for name, info in converted.items():
        assert list(info.shape) == expected["arrays"][name]["shape"]
        assert info.dtype == expected["arrays"][name]["dtype"]
        assert hashlib.sha256(tensor_bytes(info)).hexdigest() == expected["arrays"][name]["sha256"]


@pytest.mark.parametrize("model", ["Qwen3.5-35B-A3B", "Qwen3.6-27B", "Qwen3.6-35B-A3B", "Qwen3.8-27B"])
def test_actual_pinned_model_decoder_config_matches_lalamo(tmp_path, monkeypatch, model):
    """Full dimensions/layer lists match without allocating model tensors."""
    pinned = json.loads(REFERENCE.read_text())["pinned_models"][model]
    config = pinned["config"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    text = config["text_config"]
    dim, vocab = text["hidden_size"], text["vocab_size"]
    tensors = {}

    def header(name, shape):
        tensors[name] = TensorInfo(
            tmp_path / "unused.safetensors", "BF16", shape, 0, 2 * int(np.prod(shape)), (0, 0, 0, 0)
        )

    prefix = "model.language_model"
    header(prefix + ".embed_tokens.weight", (vocab, dim))
    header("lm_head.weight", (vocab, dim))
    header(prefix + ".norm.weight", (dim,))
    q_width = text["num_attention_heads"] * text["head_dim"]
    kv_width = text["num_key_value_heads"] * text["head_dim"]
    key_width = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_width = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    conv_width = 2 * key_width + value_width
    hidden = text.get("intermediate_size", text.get("moe_intermediate_size"))
    for index, kind in enumerate(text["layer_types"]):
        src = prefix + f".layers.{index}"
        header(src + ".input_layernorm.weight", (dim,))
        header(src + ".post_attention_layernorm.weight", (dim,))
        if kind == "full_attention":
            for branch, width in (("q", 2 * q_width), ("k", kv_width), ("v", kv_width)):
                header(src + f".self_attn.{branch}_proj.weight", (width, dim))
            header(src + ".self_attn.o_proj.weight", (dim, q_width))
            for branch in ("q", "k"):
                header(src + f".self_attn.{branch}_norm.weight", (text["head_dim"],))
        else:
            for branch, width in (
                ("qkv", conv_width),
                ("z", value_width),
                ("b", text["linear_num_value_heads"]),
                ("a", text["linear_num_value_heads"]),
            ):
                header(src + f".linear_attn.in_proj_{branch}.weight", (width, dim))
            header(src + ".linear_attn.out_proj.weight", (dim, value_width))
            header(src + ".linear_attn.conv1d.weight", (conv_width, 1, text["linear_conv_kernel_dim"]))
            header(src + ".linear_attn.norm.weight", (text["linear_value_head_dim"],))
            for name in ("A_log", "dt_bias"):
                header(src + ".linear_attn." + name, (text["linear_num_value_heads"],))
        if config["model_type"] == "qwen3_5_moe":
            experts = text["num_experts"]
            header(src + ".mlp.gate.weight", (experts, dim))
            header(src + ".mlp.shared_expert_gate.weight", (1, dim))
            header(src + ".mlp.experts.gate_up_proj", (experts, 2 * hidden, dim))
            header(src + ".mlp.experts.down_proj", (experts, dim, hidden))
            mlp = src + ".mlp.shared_expert"
        else:
            mlp = src + ".mlp"
        for branch in ("up", "gate"):
            header(mlp + f".{branch}_proj.weight", (hidden, dim))
        header(mlp + ".down_proj.weight", (dim, hidden))
    monkeypatch.setattr("stark_convert.adapters.qwen35.load_checkpoint", lambda _: tensors)
    builder = build_qwen35_moe_plan if config["model_type"] == "qwen3_5_moe" else build_qwen35_plan
    plan = builder(tmp_path)
    digest = hashlib.sha256(json.dumps(plan.decoder_config, sort_keys=True).encode()).hexdigest()
    assert digest == pinned["decoder_config_sha256"]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("language_model_only", 1, ConversionError),
        ("output_gate_type", "unknown", UnsupportedConversion),
        ("output_gate_type", {}, UnsupportedConversion),
        ("num_experts", True, ConversionError),
        ("num_experts_per_tok", 4, ConversionError),
        ("shared_expert_intermediate_size", 13, UnsupportedConversion),
        ("quantization", {"bits": 4, "group_size": 8}, UnsupportedConversion),
        ("intermediate_size", 12, UnsupportedConversion),
        ("partial_rotary_factor", 0.25, UnsupportedConversion),
    ],
)
def test_invalid_moe_config_fails_closed(tmp_path, field, value, error):
    source = tmp_path / "source"
    config, _ = qwen_later_fixture(source, profile="moe")
    config[field] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(error):
        build_qwen35_moe_plan(source)


@pytest.mark.parametrize("case", ["missing", "wrong-shape", "ambiguous", "packed", "extra", "shared-width"])
def test_invalid_moe_tensors_fail_closed(tmp_path, case):
    source = tmp_path / "source"
    _, tensors = qwen_later_fixture(source, profile="moe")
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    error = ConversionError
    if case == "missing":
        del tensors[name]
    elif case == "wrong-shape":
        tensors[name] = "BF16", np.zeros((2, 24, 8), dtype="<u2")
    elif case == "ambiguous":
        tensors[name + ".weight"] = tensors[name]
    elif case == "packed":
        tensors[name] = "U32", np.zeros((3, 24, 8), dtype="<u4")
        error = UnsupportedConversion
    elif case == "extra":
        tensors[name + ".scales"] = "F32", np.zeros((3, 24, 1), dtype="<f4")
    else:
        tensors["model.language_model.layers.0.mlp.shared_expert.up_proj.weight"] = (
            "BF16",
            np.zeros((13, 8), dtype="<u2"),
        )
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(error):
        build_qwen35_moe_plan(source)


def test_moe_expert_slice_change_is_rejected(tmp_path):
    source = tmp_path / "source"
    qwen_later_fixture(source, profile="moe", sharded=True)
    plan = build_qwen35_moe_plan(source)
    expert = next(info for info in plan.input_tensors if info.shape == (3, 24, 8))
    with expert.path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)


def test_moe_adapter_rejects_dense_architecture(tmp_path):
    source = tmp_path / "source"
    qwen_later_fixture(source)
    with pytest.raises(UnsupportedConversion):
        build_qwen35_moe_plan(source)
