"""Frozen independent config references and complete Granite/SmolLM3 layouts."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_dense_adapters import dense_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.granite import build_granite_plan
from stark_convert.adapters.smollm3 import build_smollm3_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "granite_smollm_configs.json").read_text())
PLANNERS = {"granite": build_granite_plan, "smollm3": build_smollm3_plan}


def fixture(source, case="granite", *, dtype="BF16", sharded=False):
    config = dict(REFERENCE["cases"][case]["config"])
    _, tensors = dense_fixture(source, "llama", tied=config["tie_word_embeddings"], dtype=dtype)
    if config["attention_bias"] or config["mlp_bias"]:
        # Explicit biases use distinct representable finite values, preserving
        # Q/K/V and up/gate concatenation ordering in expected byte assertions.
        def add(name, size):
            values = np.arange(size, dtype="<f4") / 8 + len(tensors)
            if dtype == "BF16":
                values = (values.view("<u4") >> 16).astype("<u2")
            elif dtype == "F16":
                values = values.astype("<f2")
            tensors[name] = (dtype, values)

        for layer in range(2):
            src = f"model.layers.{layer}"
            if config["attention_bias"]:
                for key, width in (("q", 8), ("k", 4), ("v", 4)):
                    add(src + f".self_attn.{key}_proj.bias", width)
                if config["model_type"] == "smollm3":
                    add(src + ".self_attn.o_proj.bias", 8)
            if config["mlp_bias"]:
                for key, width in (("up", 12), ("gate", 12), ("down", 8)):
                    add(src + f".mlp.{key}_proj.bias", width)
    config["torch_dtype"] = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[dtype]
    (source / "config.json").write_text(json.dumps(config))
    if sharded:
        (source / "model.safetensors").unlink()
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
def test_decoder_config_matches_frozen_external_importer(tmp_path, case):
    source = tmp_path / "source"
    config, _ = fixture(source, case)
    assert PLANNERS[config["model_type"]](source).decoder_config == REFERENCE["cases"][case]["decoder_config"]


@pytest.mark.parametrize("case", ["granite", "granite_biases", "smollm3", "smollm3_untied_biases"])
@pytest.mark.parametrize("dtype,requested", [("BF16", "bfloat16"), ("F16", "float16"), ("F32", "float32")])
@pytest.mark.parametrize("sharded", [False, True])
def test_every_tensor_and_fused_bias_is_consumed_and_bounded(tmp_path, case, dtype, requested, sharded):
    source = tmp_path / "source"
    config, tensors = fixture(source, case, dtype=dtype, sharded=sharded)
    plan = PLANNERS[config["model_type"]](source, dtype=requested)
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert {tensor for op in plan.operations for tensor in op.sources} == set(plan.input_tensors)
    assert all(tensor.dtype == ("F32" if name.endswith(".scales") else dtype) for name, tensor in converted.items())
    expected_count = (
        (14 if config["tie_word_embeddings"] else 15)
        + (2 if config["attention_bias"] else 0)
        + (4 if config["mlp_bias"] else 0)
    )
    if config["model_type"] == "smollm3" and config["attention_bias"]:
        expected_count += 2
    assert len(converted) == expected_count
    for layer in range(2):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        assert tensor_bytes(converted[out + ".mixer.qkvg_projection.weights.weights"]) == b"".join(
            tensors[src + f".self_attn.{key}_proj.weight"][1].tobytes() for key in ("q", "k", "v")
        )
        assert tensor_bytes(converted[out + ".mlp.up_projection.weights.weights"]) == b"".join(
            tensors[src + f".mlp.{key}_proj.weight"][1].tobytes() for key in ("up", "gate")
        )
        if config["attention_bias"]:
            assert tensor_bytes(converted[out + ".mixer.qkvg_projection.biases"]) == b"".join(
                tensors[src + f".self_attn.{key}_proj.bias"][1].tobytes() for key in ("q", "k", "v")
            )
        if config["mlp_bias"]:
            assert tensor_bytes(converted[out + ".mlp.up_projection.biases"]) == b"".join(
                tensors[src + f".mlp.{key}_proj.bias"][1].tobytes() for key in ("up", "gate")
            )
    assert all(json.loads(value)["type"] == "FullPrecisionSpec" for value in metadata.values())


def test_granite_scale_order_and_config_are_explicit(tmp_path):
    source = tmp_path / "source"
    config, _ = fixture(source, "granite_biases")
    plan = build_granite_plan(source, dtype="float32")
    by_name = {op.name: op for op in plan.operations}
    final = by_name["decoder.transformer.output_norm.scales"]
    assert (final.kind, final.scalar, final.dtype, final.sources[0].dtype) == ("scalar_divide", 16.0, "F32", "BF16")
    for layer in range(2):
        prefix = f"decoder.transformer.layers.{layer}"
        for suffix in (
            ".mixer.out_projection.weights.weights",
            ".mlp.down_projection.weights.weights",
            ".mlp.down_projection.biases",
        ):
            op = by_name[prefix + suffix]
            assert (op.kind, op.scalar, op.dtype) == ("scalar_multiply", config["residual_multiplier"], "F32")
    assert plan.decoder_config["embedding_config"]["input_scale"] == config["embedding_multiplier"]
    assert all(
        layer["mixer_config"]["scale"] == config["attention_multiplier"]
        for layer in plan.decoder_config["transformer_config"]["layer_configs"]
    )


@pytest.mark.parametrize("case", ["granite", "smollm3"])
@pytest.mark.parametrize("corruption", ["missing", "shape", "extra", "tied"])
def test_malformed_recognized_checkpoint_precedes_unsupported_config(tmp_path, case, corruption):
    source = tmp_path / "source"
    config, tensors = fixture(source, case)
    config["unknown_semantics"] = True
    (source / "config.json").write_text(json.dumps(config))
    if corruption == "missing":
        tensors.pop("model.layers.1.mlp.down_proj.weight")
    elif corruption == "shape":
        tensors["model.norm.weight"] = ("BF16", np.zeros(4, dtype="<u2"))
    elif corruption == "extra":
        tensors["unexpected.weight"] = ("BF16", np.zeros(4, dtype="<u2"))
    else:
        tensors["lm_head.weight"] = ("BF16", np.zeros((16, 8), dtype="<u2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        PLANNERS[case](source)


@pytest.mark.parametrize("case", ["granite_biases", "smollm3_untied_biases"])
@pytest.mark.parametrize(
    "name", ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "mlp.up_proj.weight", "mlp.up_proj.bias"]
)
def test_mixed_fused_dtypes_fail_closed_after_complete_layout_validation(tmp_path, case, name):
    source = tmp_path / "source"
    config, tensors = fixture(source, case)
    key = "model.layers.0." + name
    tensors[key] = ("F32", (tensors[key][1].astype("<u4") << 16).view("<f4"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="Mixed-dtype fusion"):
        PLANNERS[config["model_type"]](source)
    tensors.pop("model.layers.1.mlp.down_proj.weight")
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="missing"):
        PLANNERS[config["model_type"]](source)


@pytest.mark.parametrize("mask", [[1], None, [False, True], [1, "0"]])
def test_smollm3_malformed_mask_is_rejected(tmp_path, mask):
    source = tmp_path / "source"
    config, _ = fixture(source, "smollm3")
    config["no_rope_layers"] = mask
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="per-layer"):
        build_smollm3_plan(source)


def test_smollm3_consistent_full_attention_annotations_preserve_decoder(tmp_path):
    source = tmp_path / "source"
    config, _ = fixture(source, "smollm3")
    expected = build_smollm3_plan(source).decoder_config
    config.update(
        layer_types=["full_attention"] * 2,
        max_window_layers=1,
        no_rope_layer_interval=2,
        sliding_window=None,
        use_sliding_window=False,
        pretraining_tp=2,
    )
    (source / "config.json").write_text(json.dumps(config))
    assert build_smollm3_plan(source).decoder_config == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"no_rope_layer_interval": 4},
        {"layer_types": ["full_attention", "sliding_attention"]},
        {"use_sliding_window": True},
        {"sliding_window": 16},
    ],
)
def test_smollm3_contradictory_attention_annotations_fail_closed(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = fixture(source, "smollm3")
    config.update(changes)
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion):
        build_smollm3_plan(source)


@pytest.mark.parametrize(
    "changes",
    [
        {"no_rope_layer_interval": 0},
        {"no_rope_layer_interval": True},
        {"layer_types": ["full_attention"]},
        {"layer_types": ["full_attention", False]},
        {"layer_types": "full_attention"},
        {"max_window_layers": -1},
        {"max_window_layers": 3},
        {"max_window_layers": True},
        {"use_sliding_window": "false"},
        {"sliding_window": 0},
        {"pretraining_tp": 0},
        {"pretraining_tp": True},
    ],
)
def test_smollm3_malformed_attention_annotations_are_errors(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = fixture(source, "smollm3")
    config.update(changes)
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError):
        build_smollm3_plan(source)


@pytest.mark.parametrize(
    "field", ["embedding_multiplier", "residual_multiplier", "attention_multiplier", "logits_scaling"]
)
@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_granite_malformed_scale_is_rejected(tmp_path, field, value):
    source = tmp_path / "source"
    config, _ = fixture(source)
    config[field] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="positive"):
        build_granite_plan(source)


@pytest.mark.parametrize(
    "field", ["embedding_multiplier", "residual_multiplier", "attention_multiplier", "logits_scaling"]
)
def test_granite_required_scales_cannot_silently_default(tmp_path, field):
    source = tmp_path / "source"
    config, _ = fixture(source)
    del config[field]
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="required scale"):
        build_granite_plan(source)


@pytest.mark.parametrize("case", ["granite", "smollm3"])
def test_unsupported_semantics_and_quantized_sources_fail_closed(tmp_path, case):
    source = tmp_path / "source"
    config, _ = fixture(source, case)
    config["rope_scaling"] = {"rope_type": "linear", "factor": 2}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="RoPE scaling"):
        PLANNERS[case](source)
    config["rope_scaling"] = None
    config["quantization_config"] = {"bits": 4, "group_size": 64}
    (source / "config.json").write_text(json.dumps(config))
    error = ConversionError if case == "smollm3" else UnsupportedConversion
    message = "no packed checkpoint matrices" if case == "smollm3" else "Quantized"
    with pytest.raises(error, match=message):
        PLANNERS[case](source)
