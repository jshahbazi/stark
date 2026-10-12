"""Frozen actual-loader exports for dense Qwen3.5 text and DeltaNet layouts."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.qwen35 import build_qwen35_plan, normalize_qwen35_config
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
REFERENCE = Path(__file__).parent / "fixtures" / "qwen35_exports.json"


def qwen35_fixture(
    path,
    *,
    dtype="BF16",
    layout="split",
    tied=False,
    wrapper=False,
    biased=False,
    explicit_gate=False,
    sharded=False,
    mtp=False,
):
    """Small synthetic inputs expose every projection branch and row order."""
    path.mkdir()
    text = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 16,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": tied,
        "hidden_act": "silu",
        "attention_bias": biased,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        "torch_dtype": DTYPES[dtype],
        "eos_token_id": 1,
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
    prefix = "model.language_model" if wrapper else "model"
    if mtp:
        config["mtp_num_hidden_layers"] = 1
    tensors = {}

    def add(name, shape, *, recurrent=False):
        values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
        if recurrent:
            # Portable exceptional cast behavior, signed zero and subnormals.
            if dtype == "F32":
                patterns = np.array(
                    [
                        0,
                        0x80000000,
                        1,
                        0x007FFFFF,
                        0x00800000,
                        0x7F800000,
                        0xFF800000,
                        0x7F800001,
                        0x7FC01234,
                        0xFFC04321,
                    ],
                    dtype="<u4",
                )
                raw = np.resize(patterns, np.prod(shape)).reshape(shape).view("<f4")
            else:
                patterns = np.array(
                    [
                        0,
                        0x8000,
                        1,
                        0x007F if dtype == "BF16" else 0x03FF,
                        0x0080 if dtype == "BF16" else 0x0400,
                        0x7F80 if dtype == "BF16" else 0x7C00,
                        0xFF80 if dtype == "BF16" else 0xFC00,
                        0x7F81 if dtype == "BF16" else 0x7C01,
                        0x7FC1 if dtype == "BF16" else 0x7E01,
                        0xFFC3 if dtype == "BF16" else 0xFE03,
                    ],
                    dtype="<u2",
                )
                raw = np.resize(patterns, np.prod(shape)).reshape(shape)
                if dtype == "F16":
                    raw = raw.view("<f2")
        else:
            raw = {
                "BF16": lambda x: (x.view("<u4") >> 16).astype("<u2"),
                "F16": lambda x: x.astype("<f2"),
                "F32": lambda x: x,
            }[dtype](values)
        tensors[name] = (dtype, raw)

    add(prefix + ".embed_tokens.weight", (16, 8))
    if not tied:
        add("lm_head.weight", (16, 8))
    add(prefix + ".norm.weight", (8,), recurrent=True)
    for layer in range(2):
        src = f"{prefix}.layers.{layer}"
        add(src + ".input_layernorm.weight", (8,))
        add(src + ".post_attention_layernorm.weight", (8,))
        add(src + ".mlp.up_proj.weight", (12, 8))
        add(src + ".mlp.gate_proj.weight", (12, 8))
        add(src + ".mlp.down_proj.weight", (8, 12))
        if layer == 0:
            attn = src + ".linear_attn"
            branches = {
                "split": (("qkv", 16), ("z", 8), ("b", 4), ("a", 4)),
                "direct": (("", 32),),
                "grouped": (("qkvz", 24), ("ba", 8)),
            }[layout]
            for name, rows in branches:
                add(attn + ".in_proj" + ("_" + name if name else "") + ".weight", (rows, 8))
            add(attn + ".out_proj.weight", (8, 8))
            add(attn + ".conv1d.weight", (16, 1, 4), recurrent=True)
            add(attn + ".A_log", (4,), recurrent=True)
            add(attn + ".dt_bias", (4,), recurrent=True)
            add(attn + ".norm.weight", (2,), recurrent=True)
        else:
            attn = src + ".self_attn"
            for name, rows in (("q", 8 if explicit_gate else 16), ("k", 4), ("v", 4)):
                add(attn + f".{name}_proj.weight", (rows, 8))
                if biased:
                    add(attn + f".{name}_proj.bias", (rows,))
            if explicit_gate:
                add(attn + ".gate_proj.weight", (8, 8))
                if biased:
                    add(attn + ".gate_proj.bias", (8,))
            add(attn + ".o_proj.weight", (8, 8))
            if biased:
                add(attn + ".o_proj.bias", (8,))
            add(attn + ".q_norm.weight", (2,), recurrent=True)
            add(attn + ".k_norm.weight", (2,), recurrent=True)
    if wrapper:
        add("model.visual.patch_embed.proj.weight", (2, 2))
    if mtp:
        add("mtp.layers.0.self_attn.q_proj.weight", (2, 2))
    (path / "config.json").write_text(json.dumps(config))
    if sharded:
        weight_map = {}
        for shard in range(2):
            filename = f"model-{shard}.safetensors"
            contents = dict(list(tensors.items())[shard::2])
            save_tensors(path / filename, contents)
            weight_map.update(dict.fromkeys(contents, filename))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_tensors(path / "model.safetensors", tensors)
    return config, tensors


def _case_name(dtype, target, layout, tied=False, wrapper=False, biased=False, explicit_gate=False, mtp=False):
    return f"{dtype}-{target}-{layout}-t{int(tied)}-w{int(wrapper)}-b{int(biased)}-g{int(explicit_gate)}" + (
        "-mtp" if mtp else ""
    )


CASES = [
    (source, target, layout, False, False, False, False)
    for source, target, layout in itertools.product(DTYPES, DTYPES, ("split", "direct", "grouped"))
]
CASES += [
    (source, target, "split", tied, wrapper, biased, gate)
    for source, target in itertools.product(DTYPES, DTYPES)
    for tied, wrapper, biased, gate in (
        (True, True, False, False),
        (False, False, True, True),
        (False, False, True, False),
    )
]


CASES = [(*case, False) for case in CASES]
CASES += [
    (source, target, "split", False, True, False, False, True) for source, target in itertools.product(DTYPES, DTYPES)
]


@pytest.mark.parametrize("source_dtype,target,layout,tied,wrapper,biased,gate,mtp", CASES)
def test_complete_exports_match_frozen_actual_loader(
    tmp_path, source_dtype, target, layout, tied, wrapper, biased, gate, mtp
):
    source = tmp_path / "source"
    _, tensors = qwen35_fixture(
        source,
        dtype=source_dtype,
        layout=layout,
        tied=tied,
        wrapper=wrapper,
        biased=biased,
        explicit_gate=gate,
        sharded=True,
        mtp=mtp,
    )
    case = json.loads(REFERENCE.read_text())["cases"][
        _case_name(source_dtype, target, layout, tied, wrapper, biased, gate, mtp)
    ]
    plan = build_qwen35_plan(source, dtype=DTYPES[target])
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    metadata, converted = read_safetensors_header(tmp_path / "output.safetensors")
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(array.nbytes for _, array in tensors.values())
    assert plan.decoder_config == case["decoder_config"]
    assert {key: json.loads(value) for key, value in metadata.items()} == case["metadata"]
    assert set(converted) == set(case["arrays"])
    for name, info in converted.items():
        expected = case["arrays"][name]
        assert list(info.shape) == expected["shape"]
        assert info.dtype == expected["dtype"]
        assert hashlib.sha256(tensor_bytes(info)).hexdigest() == expected["sha256"]


def test_wrapper_config_merge_uses_top_level_overrides():
    config = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "eos_token_id": 9,
        "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": False, "eos_token_id": 1},
    }
    assert normalize_qwen35_config(config) == {"model_type": "qwen3_5", "tie_word_embeddings": True, "eos_token_id": 9}
    with pytest.raises(ConversionError, match="text_config"):
        normalize_qwen35_config({"text_config": []})


@pytest.mark.parametrize("case", ["missing", "shape", "extra", "partial", "duplicate", "integer", "tied"])
def test_malformed_tensor_layouts_fail_closed(tmp_path, case):
    source = tmp_path / "source"
    config, tensors = qwen35_fixture(source, tied=case == "tied")
    name = "model.layers.0.linear_attn.in_proj_qkv.weight"
    error = ConversionError
    if case == "missing":
        del tensors[name]
    elif case == "shape":
        tensors[name] = "BF16", np.zeros((15, 8), dtype="<u2")
    elif case == "extra":
        tensors["unexpected.weight"] = "BF16", np.zeros((2, 2), dtype="<u2")
    elif case == "partial":
        del tensors["model.layers.0.linear_attn.in_proj_b.weight"]
    elif case == "duplicate":
        tensors["model.layers.0.linear_attn.in_proj.weight"] = "BF16", np.zeros((32, 8), dtype="<u2")
    elif case == "integer":
        tensors[name] = "I32", np.zeros((16, 8), dtype="<i4")
        error = UnsupportedConversion
    else:
        tensors["lm_head.weight"] = "BF16", np.zeros((16, 8), dtype="<u2")
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(error):
        build_qwen35_plan(source)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("model_type", "qwen3_5_moe", UnsupportedConversion),
        ("model_type", "qwen3_5_text", UnsupportedConversion),
        ("layer_types", ["linear_attention"], ConversionError),
        ("layer_types", ["linear_attention", "unknown"], ConversionError),
        ("layer_types", ["linear_attention", {}], ConversionError),
        ("linear_num_value_heads", 3, ConversionError),
        ("linear_conv_kernel_dim", 1, ConversionError),
        ("rope_parameters", {"rope_theta": 1.0, "partial_rotary_factor": 0.1}, ConversionError),
        (
            "rope_parameters",
            {"rope_theta": 1.0, "partial_rotary_factor": 1.0, "rope_type": "linear"},
            UnsupportedConversion,
        ),
        ("attn_output_gate", False, UnsupportedConversion),
        ("mlp_only_layers", [0], UnsupportedConversion),
        ("quantization", {"bits": 4}, ConversionError),
        ("unknown", True, UnsupportedConversion),
        ("mamba_ssm_dtype", "float16", UnsupportedConversion),
        ("hidden_act", "relu", UnsupportedConversion),
        ("attention_dropout", 0.1, UnsupportedConversion),
        ("attention_dropout", True, ConversionError),
        ("attention_dropout", "0", ConversionError),
        ("use_bias", True, UnsupportedConversion),
        ("use_qk_norm", False, UnsupportedConversion),
        ("pretraining_tp", 2, UnsupportedConversion),
    ],
)
def test_unsupported_and_malformed_config_variants(tmp_path, field, value, error):
    source = tmp_path / "source"
    config, _ = qwen35_fixture(source)
    config[field] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(error):
        build_qwen35_plan(source)


def test_declared_vision_extraction_preserves_input_signatures(tmp_path):
    source = tmp_path / "source"
    config, tensors = qwen35_fixture(source, wrapper=True, sharded=True)
    plan = build_qwen35_plan(source)
    assert len(plan.input_tensors) == len(tensors)
    assert not any("visual" in operation.name for operation in plan.operations)
    config.pop("vision_config")
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="unexpected"):
        build_qwen35_plan(source)


def test_input_change_after_projection_slice_planning_is_rejected(tmp_path):
    source = tmp_path / "source"
    qwen35_fixture(source, layout="grouped")
    plan = build_qwen35_plan(source)
    file = source / "model.safetensors"
    with file.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)


@pytest.mark.parametrize("declaration", (None, 0, True, -1, "1"))
def test_mtp_omission_requires_a_positive_integer_declaration(tmp_path, declaration):
    source = tmp_path / "source"
    config, _ = qwen35_fixture(source, wrapper=True, mtp=True)
    if declaration is None:
        config.pop("mtp_num_hidden_layers")
    else:
        config["mtp_num_hidden_layers"] = declaration
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError):
        build_qwen35_plan(source)


@pytest.mark.parametrize("namespace", ("mtp", "model.visual"))
def test_omitted_auxiliary_shard_change_is_rejected_after_text_writes(tmp_path, namespace):
    source = tmp_path / "source"
    _, tensors = qwen35_fixture(source, wrapper=True, mtp=True)
    auxiliary = {name: value for name, value in tensors.items() if name.startswith(namespace + ".")}
    text = {name: value for name, value in tensors.items() if name not in auxiliary}
    save_tensors(source / "model.safetensors", text)
    save_tensors(source / "auxiliary.safetensors", auxiliary)
    weight_map = {**dict.fromkeys(text, "model.safetensors"), **dict.fromkeys(auxiliary, "auxiliary.safetensors")}
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    plan = build_qwen35_plan(source)
    assert len(plan.input_tensors) == len(tensors)
    assert not any(info.path.name == "auxiliary.safetensors" for op in plan.operations for info in op.sources)
    with (source / "auxiliary.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)


def test_declared_mtp_does_not_authorize_other_namespaces(tmp_path):
    source = tmp_path / "source"
    _, tensors = qwen35_fixture(source, wrapper=True, mtp=True)
    tensors["model.mtp.weight"] = "BF16", np.zeros((2,), dtype="<u2")
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="unexpected"):
        build_qwen35_plan(source)
