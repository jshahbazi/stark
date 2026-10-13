"""LFM2.5 source profiles and bounded exports against the pinned Lalamo loader."""

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_mlx import _float_values, save_mlx_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters import lfm2
from stark_convert.adapters.lfm2 import build_lfm2_plan
from stark_convert.streaming import (
    DTYPE_BYTES,
    ConversionError,
    TensorInfo,
    UnsupportedConversion,
    read_safetensors_header,
    write_plan,
)

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "lfm25_exports.json"
DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
PROFILES = ("dense_nested", "dense_flat", "mlx4", "mlx8")


def lfm25_fixture(source, profile="dense_nested", *, dtype="BF16", sharded=False):
    """Preserve the public sixteen-layer topology at small test dimensions."""
    reference = json.loads(REFERENCE_PATH.read_text())
    config = copy.deepcopy(reference["profiles"][profile]["config"])
    config["dtype"] = DTYPES[dtype]
    source.mkdir()
    model, vocab = config["hidden_size"], config["vocab_size"]
    head = model // config["num_attention_heads"]
    kv = head * config["num_key_value_heads"]
    hidden = 128 if profile == "dense_nested" else 256
    packed = config.get("quantization")
    tensors = {}

    def add(name, shape):
        tensors[name] = dtype, _float_values(shape, dtype, base=len(tensors))

    def matrix(prefix, shape):
        if packed is None:
            add(prefix + ".weight", shape)
            return
        rows, columns = shape
        weight_shape = rows, columns * packed["bits"] // 32
        raw = np.arange(np.prod(weight_shape), dtype="<u4").reshape(weight_shape)
        raw = raw * np.uint32(2654435761) + np.uint32(len(tensors))
        tensors[prefix + ".weight"] = "U32", raw
        for plane in ("scales", "biases"):
            add(prefix + "." + plane, (rows, columns // packed["group_size"]))

    matrix("model.embed_tokens", (vocab, model))
    add("model.embedding_norm.weight", (model,))
    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"model.layers.{layer}"
        add(prefix + ".operator_norm.weight", (model,))
        add(prefix + ".ffn_norm.weight", (model,))
        if kind == "conv":
            matrix(prefix + ".conv.in_proj", (3 * model, model))
            matrix(prefix + ".conv.out_proj", (model, model))
            shape = (model, config["conv_L_cache"], 1) if packed else (model, 1, config["conv_L_cache"])
            add(prefix + ".conv.conv.weight", shape)
        else:
            for projection, width in (("q", model), ("k", kv), ("v", kv)):
                matrix(prefix + f".self_attn.{projection}_proj", (width, model))
            matrix(prefix + ".self_attn.out_proj", (model, model))
            for side in ("q", "k"):
                add(prefix + f".self_attn.{side}_layernorm.weight", (head,))
        for projection in ("w3", "w1"):
            matrix(prefix + ".feed_forward." + projection, (hidden, model))
        matrix(prefix + ".feed_forward.w2", (model, hidden))
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors, sharded=sharded)
    return config, tensors


@pytest.mark.parametrize("profile,source_dtype,target_dtype", list(itertools.product(PROFILES, DTYPES, DTYPES)))
def test_lfm25_full_sixteen_layer_exports_match_pinned_loader(tmp_path, profile, source_dtype, target_dtype):
    reference = json.loads(REFERENCE_PATH.read_text())
    frozen = reference["profiles"][profile]
    source = tmp_path / "source"
    _, inputs = lfm25_fixture(source, profile, dtype=source_dtype)
    for name, (dtype, raw) in inputs.items():
        assert frozen["inputs"][source_dtype][name] == {
            "dtype": dtype,
            "shape": list(raw.shape),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }
    plan = build_lfm2_plan(source, dtype=DTYPES[target_dtype])
    assert plan.decoder_config == frozen["decoder_config"]
    assert plan.source_tensors == len(inputs)
    assert {info for op in plan.operations for info in op.sources} == set(plan.input_tensors)
    destination = tmp_path / "output.safetensors"
    stats = write_plan(plan, destination, chunk_bytes=96)
    assert stats["peak_buffer_bytes"] <= 96
    assert stats["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata, outputs = read_safetensors_header(destination)
    exported = frozen["exports"][source_dtype + "_" + target_dtype]
    assert {name: json.loads(value) for name, value in metadata.items()} == frozen["metadata"]
    assert set(outputs) == set(exported)
    for name, info in outputs.items():
        assert exported[name] == {
            "dtype": info.dtype,
            "shape": list(info.shape),
            "sha256": hashlib.sha256(tensor_bytes(info)).hexdigest(),
        }


@pytest.mark.parametrize("profile", PROFILES)
def test_lfm25_sharded_sources_preserve_complete_export(tmp_path, profile):
    outputs = []
    for sharded in (False, True):
        source = tmp_path / str(sharded)
        lfm25_fixture(source, profile, sharded=sharded)
        destination = tmp_path / (str(sharded) + ".safetensors")
        write_plan(build_lfm2_plan(source), destination, chunk_bytes=96)
        outputs.append(destination.read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("source_index", range(7))
def test_lfm25_pinned_public_config_and_header_shapes_match_reference(tmp_path, monkeypatch, source_index):
    """Header planning only; reduced exports separately verify actual bytes."""
    reference = json.loads(REFERENCE_PATH.read_text())
    public = reference["sources"][source_index]
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(public["config"]))
    tensors = {
        name: TensorInfo(
            source / name,
            info["dtype"],
            tuple(info["shape"]),
            0,
            math.prod(info["shape"]) * DTYPE_BYTES[info["dtype"]],
            (0, 0, 0, 0),
        )
        for name, info in public["source_tensor_layout"].items()
    }
    monkeypatch.setattr(lfm2, "load_checkpoint", lambda _: tensors)
    plan = build_lfm2_plan(source)
    assert plan.decoder_config == reference["source_decoder_configs"][public["decoder_config_key"]]
    assert plan.source_tensors == len(public["source_tensor_layout"])
    assert {info for op in plan.operations for info in op.sources} == set(plan.input_tensors)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("corruption", ("missing", "extra", "alias", "head", "topology", "rope"))
def test_lfm25_malformed_source_profiles_fail_closed(tmp_path, profile, corruption):
    source = tmp_path / "source"
    config, tensors = lfm25_fixture(source, profile)
    if corruption == "missing":
        del tensors["model.layers.15.feed_forward.w2.weight"]
    elif corruption == "extra":
        tensors["model.layers.0.conv.conv.bias"] = "BF16", _float_values((128,), "BF16")
    elif corruption == "alias":
        tensors["model.layers.2.self_attn.q_norm.weight"] = tensors["model.layers.2.self_attn.q_layernorm.weight"]
    elif corruption == "head":
        raw = _float_values((17, 128), "BF16")
        raw.view("u1").flat[0] ^= 1
        tensors["lm_head.weight"] = "BF16", raw
    elif corruption == "topology":
        config["layer_types"][15] = "full_attention"
    else:
        config["rope_theta"] = 1000000.0
        config["rope_parameters"] = {"rope_theta": 500000.0}
    (source / "config.json").write_text(json.dumps(config))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_lfm2_plan(source)


@pytest.mark.parametrize("profile", ("mlx4", "mlx8"))
def test_lfm25_affine_description_disagreement_fails_closed(tmp_path, profile):
    source = tmp_path / "source"
    config, _ = lfm25_fixture(source, profile)
    config["quantization_config"]["bits"] = 12 - config["quantization"]["bits"]
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="disagree"):
        build_lfm2_plan(source)


@pytest.mark.parametrize("profile", PROFILES)
def test_lfm25_new_semantics_are_not_inferred_from_family_name(tmp_path, profile):
    source = tmp_path / "source"
    config, _ = lfm25_fixture(source, profile)
    config["_name_or_path"] = "LiquidAI/LFM2.5-1.2B-Thinking"
    config["sliding_window"] = 1024
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="config options"):
        build_lfm2_plan(source)


@pytest.mark.parametrize("profile", PROFILES)
def test_lfm25_sources_are_rechecked_after_planning(tmp_path, profile):
    source = tmp_path / "source"
    _, tensors = lfm25_fixture(source, profile)
    plan = build_lfm2_plan(source)
    name = "model.layers.15.feed_forward.w2.weight"
    dtype, raw = tensors[name]
    raw = raw.copy()
    raw.view("u1").flat[0] ^= 1
    tensors[name] = dtype, raw
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="changed"):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
