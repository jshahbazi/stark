"""Synthetic Qwen3.8 standard storage, typed metadata and corruption rejection."""

import json
import shutil

import numpy as np
import pytest
from test_streaming import save_tensors
from tokenizers import Tokenizer, models

from stark_convert import cli, model_support
from stark_convert.adapters.qwen38 import (
    architecture_config,
    build_qwen38_plan,
    validate_qwen38_artifact,
)
from stark_convert.streaming import ConversionError, read_safetensors_header, write_plan


def fixture(path, *, sharded=False):
    """Small mixed-Q4/Q8 graph with both mixers, expert banks and paged PLE."""
    path.mkdir()
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "eos_token_id": [1, 2],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 3,
        "output_gate_type": "sigmoid",
        "hidden_act": "silu",
        "hc_count": 2,
        "hc_lowrank": 64,
        "ple_layer_ids": [1],
        "ple_embed_dim": 128,
        "ple_conv_kernel_size": 3,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "split_ngram_parts": 2,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
    }
    config = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "lily": {
            "format": "qwen4_exp-affine-v1",
            "quantization": {
                "default": {"bits": 4, "group_size": 64, "mode": "affine"},
                "ngram_embedding": {"bits": 4, "group_size": 32, "mode": "affine"},
                "q8_groups": ["attn", "shared", "head", "embed"],
            },
            "ple": {
                "layer_multipliers": [9007199254740993, 9007199254740995, 9007199254740997],
                "ngram_heads_vocab_sizes": [3, 3, 3, 3],
                "ngram_heads_offsets": [0, 3, 6, 9],
                "padded_vocab_size": 16,
            },
        },
    }
    tensors = {}

    def control(name, shape, dtype="BF16"):
        raw_dtype = {"BF16": "<u2", "F32": "<u4", "I64": "<i8", "U64": "<u8"}[dtype]
        values = np.arange(np.prod(shape), dtype=raw_dtype).reshape(shape)
        if dtype == "BF16":
            # Preserve signed zeros, infinities and NaN payloads bit-for-bit.
            values = (values * 1877 + 0x8000).astype("<u2")
        elif dtype in {"I64", "U64"}:
            values += 2**53 + 1
        tensors[name] = dtype, values

    def matrix(name, shape, bits=8, group=64):
        packed = (*shape[:-1], shape[-1] * bits // 32)
        values = np.arange(np.prod(packed), dtype="<u4").reshape(packed)
        tensors[name + ".weight"] = "U32", values * np.uint32(2654435761) + np.uint32(len(tensors))
        control(name + ".scales", (*shape[:-1], shape[-1] // group))
        control(name + ".biases", (*shape[:-1], shape[-1] // group))

    def hyper(p, combine=True):
        matrix(p + ".input_mix_weight_down", (64, 128))
        matrix(p + ".input_mix_weight_up", (128, 64))
        if combine:
            matrix(p + ".block_inject_weight", (2, 128))
        control(p + ".hc_norm.weight", (128,))

    base = "model.language_model"
    matrix(base + ".embed_tokens", (16, 64))
    matrix("lm_head", (16, 64))
    hyper(base + ".hyper_connection_mixer", False)
    for layer in range(2):
        p = f"{base}.layers.{layer}"
        hyper(p + ".attn_hyper_connection")
        hyper(p + ".mlp_hyper_connection")
        if layer == 0:
            for name, rows in (("in_proj_qkv", 128), ("in_proj_z", 64), ("in_proj_a", 2), ("in_proj_b", 2)):
                matrix(p + ".linear_attn." + name, (rows, 64), 4)
            matrix(p + ".linear_attn.out_proj", (64, 64), 4)
            control(p + ".linear_attn.conv1d.weight", (128, 1, 3))
            control(p + ".linear_attn.A_log", (2,), "F32")
            control(p + ".linear_attn.dt_bias", (2,))
            control(p + ".linear_attn.norm.weight", (32,))
        else:
            for name, rows in (("q_proj", 128), ("k_proj", 32), ("v_proj", 32)):
                matrix(p + ".self_attn." + name, (rows, 64))
            matrix(p + ".self_attn.o_proj", (64, 64))
            matrix(p + ".self_attn.indexer.index_qk_proj", (32, 64))
            for name, width in (
                ("q_norm", 32),
                ("k_norm", 32),
                ("indexer.q_layernorm", 16),
                ("indexer.k_layernorm", 16),
            ):
                control(p + ".self_attn." + name + ".weight", (width,))
        matrix(p + ".mlp.gate", (4, 64))
        matrix(p + ".mlp.shared_expert_gate", (1, 64))
        for name in ("gate_proj", "up_proj", "down_proj"):
            matrix(p + ".mlp.experts." + name, (4, 64, 64), 4)
            matrix(p + ".mlp.shared_expert." + name, (64, 64))
    p = base + ".layers.0.ple"
    for shard in range(2):
        matrix(f"{p}.ple_embedding.ngram_embedding.shard_{shard}", (8, 32), 4, 32)
    matrix(p + ".key_proj", (128, 128))
    matrix(p + ".value_proj", (64, 128))
    for name in ("norm_key", "norm_query", "norm_conv"):
        control(p + "." + name + ".weight", (128,))
    control(p + ".conv1d.weight", (128, 1, 3))
    for name, count, dtype in (
        ("layer_multipliers", 3, "I64"),
        ("ngram_heads_vocab_sizes", 4, "U64"),
        ("ngram_heads_offsets", 4, "I64"),
    ):
        control(p + ".ple_embedding." + name, (count,), dtype)
        values = np.asarray(config["lily"]["ple"][name], dtype="<u8")
        tensors[p + ".ple_embedding." + name] = dtype, values.view("<i8") if dtype == "I64" else values
    control("mtp.layers.0.unused", (2,))
    control("model.visual.unused", (3,))
    (path / "config.json").write_text(json.dumps(config))
    if sharded:
        weight_map = {}
        for index in range(2):
            name = f"part-{index}.safetensors"
            subset = dict(list(tensors.items())[index::2])
            save_tensors(path / name, subset)
            weight_map.update(dict.fromkeys(subset, name))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_tensors(path / "model.safetensors", tensors)
    tokenizer = Tokenizer(models.WordLevel({f"S{i}": i for i in range(16)}, unk_token="S0"))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))
    return config, tensors


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("workspace", [4096, 65536])
def test_mixed_storage_complete_roundtrip_and_integer_bits(tmp_path, sharded, workspace):
    source = tmp_path / "source"
    fixture(source, sharded=sharded)
    plan = build_qwen38_plan(source)
    target = tmp_path / "model.safetensors"
    stats = write_plan(plan, target, chunk_bytes=workspace)
    receipt = validate_qwen38_artifact(plan, target, chunk_bytes=workspace)
    assert receipt["quantization_preserved"] and receipt["packed_code_planes"] > 30
    assert receipt["inverse_bit_exact_parameter_planes"] > 50
    assert receipt["omitted"] == {"vision": {"bytes": 6, "tensors": 1}, "mtp": {"bytes": 4, "tensors": 1}}
    assert stats["peak_buffer_bytes"] <= workspace
    specs, tensors = read_safetensors_header(target)
    bindings = plan.decoder_config["tensor_bindings"]
    assert len({b["name"] for b in bindings.values()}) == len(tensors) == len(bindings)
    prefix = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    assert tensors[prefix + ".weights"].shape == (256, 32)
    assert tensors[prefix + ".scales"].shape == (1, 256)
    assert json.loads(specs[prefix + ".spec"])["layout"] == "output_input"
    ngram = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_0"
    assert tensors[ngram + ".scales"].shape == (8, 1)
    assert json.loads(specs[ngram + ".spec"])["layout"] == "input_output"
    assert {b["dtype"] for b in bindings.values()} >= {"I64", "U64", "F32", "BF16", "U32"}


def test_corruption_rejected_in_codes_parameters_and_padding(tmp_path):
    source = tmp_path / "source"
    fixture(source)
    plan = build_qwen38_plan(source)
    target = tmp_path / "model.safetensors"
    write_plan(plan, target, chunk_bytes=4096)
    _, tensors = read_safetensors_header(target)
    stem = "language_model.model.layers.0.mlp.shared_expert_gate"
    for key, byte in ((stem + ".weights", 0), (stem + ".scales", 0), (stem + ".biases", 2)):
        damaged = tmp_path / "damaged.safetensors"
        shutil.copyfile(target, damaged)
        with damaged.open("r+b") as output:
            output.seek(tensors[key].offset + byte)
            value = output.read(1)
            output.seek(-1, 1)
            output.write(bytes([value[0] ^ 1]))
        with pytest.raises(ConversionError):
            validate_qwen38_artifact(plan, damaged, chunk_bytes=4096)


def test_cli_publishes_self_contained_typed_artifact_unqualified(tmp_path):
    source = tmp_path / "source"
    fixture(source)
    target = tmp_path / "converted"
    receipt = cli.convert(
        source,
        target,
        repo="example/qwen38",
        architecture="qwen4_exp",
        metadata_profile="hf-generic",
        chunk_bytes=4096,
    )
    config = json.loads((target / "config.json").read_text())
    assert config["type"] == "Qwen38LanguageModelConfig"
    assert set(config["decoder_config"]) == {"architecture", "tensor_bindings"}
    assert "model_type" not in json.dumps(config)
    assert "vision_config" not in json.dumps(config) and "mtp" not in json.dumps(config)
    assert receipt["capability"]["native_loading"] == "unverified"
    assert receipt["storage_validation"]["quantization_preserved"]
    assert {p.name for p in target.iterdir()} == {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "conversion.json",
    }
    assert target.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in target.iterdir())


def test_metadata_rejects_unimplemented_precision_and_conflicting_policy(tmp_path):
    source = tmp_path / "source"
    config, _ = fixture(source)
    report = model_support.preflight_model(config, mode="text")
    assert report["status"] == "conditional" and report["qualification"]["status"] == "not_assessed"
    assert model_support.preflight_model(config, mode="vision-language")["status"] == "unsupported"
    config["lily"]["quantization"]["q8_dense"] = True
    with pytest.raises(ConversionError):
        architecture_config(config)


def test_unexpected_text_tensor_is_not_silently_dropped(tmp_path):
    source = tmp_path / "source"
    _, tensors = fixture(source)
    tensors["model.language_model.unknown.weight"] = "BF16", np.zeros((2,), dtype="<u2")
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="Unexpected Qwen3.8 text tensor"):
        build_qwen38_plan(source)


@pytest.mark.parametrize("dtype", ["I64", "U64"])
def test_exact_unsigned_hash_bits_and_contradictory_config_rejection(tmp_path, dtype):
    source = tmp_path / "source"
    config, tensors = fixture(source)
    values = [2**64 - 3, 2**63 + 1, 2**53 + 1]
    config["lily"]["ple"]["layer_multipliers"] = values
    bits = np.asarray(values, dtype="<u8")
    key = "model.language_model.layers.0.ple.ple_embedding.layer_multipliers"
    tensors[key] = dtype, bits.view("<i8") if dtype == "I64" else bits
    (source / "config.json").write_text(json.dumps(config))
    save_tensors(source / "model.safetensors", tensors)
    plan = build_qwen38_plan(source)
    target = tmp_path / "model.safetensors"
    write_plan(plan, target, chunk_bytes=4096)
    assert validate_qwen38_artifact(plan, target, chunk_bytes=4096)["quantization_preserved"]
    plan.decoder_config["architecture"]["ple"]["layer_multipliers"][0] -= 2
    with pytest.raises(ConversionError, match="PLE control disagrees"):
        validate_qwen38_artifact(plan, target, chunk_bytes=4096)
