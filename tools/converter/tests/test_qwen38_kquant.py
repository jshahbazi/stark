"""Synthetic indexed affine admission controls; no model execution claims."""

import copy

import pytest
from test_qwen38_external import external_config as external_config

from stark_convert import qwen38_external, qwen38_kquant
from stark_convert.adapters.qwen38 import describe_text_weights
from stark_convert.streaming import UnsupportedConversion


def source_name(name):
    if name.startswith("model.language_model."):
        name = "language_model.model." + name.removeprefix("model.language_model.")
    elif name == "lm_head":
        name = "language_model.lm_head"
    return name.replace(".ngram_embedding.shard_", ".ngram_embedding.shards.")


@pytest.fixture
def indexed_config(external_config, monkeypatch):
    config = copy.deepcopy(external_config)
    config.pop("ngram_table")
    config["text_config"].update(
        shared_expert_intermediate_size=128,
        split_ngram_parts=2,
        ple_storage={"manifest": "ple_manifest.json", "cache_rows": 0},
    )
    monkeypatch.setitem(qwen38_external._NATIVE_TEXT_GEOMETRY, "shared_expert_intermediate_size", 128)
    config["quantization"] = {"mode": "affine", "bits": 4, "group_size": 64}
    _, specs = qwen38_kquant._descriptors(config)
    default = config["quantization"].copy()
    config["quantization"].update({source_name(name): spec for name, spec in specs.items() if spec != default})
    config["quantization_config"] = copy.deepcopy(config["quantization"])
    return config


def tensors_for(config):
    architecture, specs = qwen38_kquant._descriptors(config)
    tensors = {}

    def control(name, shape, integers=False):
        tensors[source_name(name)] = ("I64" if integers else "BF16", shape)

    def matrix(stem, shape, **_options):
        spec = specs[stem]
        if spec is False:
            control(stem + ".weight", shape)
            return
        *leading, columns = shape
        for plane, dtype, count in (
            ("weight", "U32", columns * spec["bits"] // 32),
            ("scales", "BF16", columns // spec["group_size"]),
            ("biases", "BF16", columns // spec["group_size"]),
        ):
            tensors[source_name(stem) + "." + plane] = (dtype, (*leading, count))

    describe_text_weights(architecture, {}, matrix, control, include_mtp=False)
    return tensors


def test_indexed_affine_admits_q8_embedding_gate_and_group128_shared(indexed_config):
    assert qwen38_kquant.is_kquant_affine(indexed_config)
    architecture = qwen38_kquant.validate_config(indexed_config)
    assert architecture["split_ngram_parts"] == 2
    tensors = tensors_for(indexed_config)
    qwen38_kquant.validate_layout(indexed_config, tensors)
    assert tensors["language_model.model.embed_tokens.weight"] == ("U32", (16, 32))
    assert tensors["language_model.model.layers.0.mlp.shared_expert.gate_proj.scales"] == ("BF16", (128, 1))
    assert tensors["language_model.model.layers.0.mlp.shared_expert_gate.weight"] == ("U32", (1, 32))


@pytest.mark.parametrize("change", ["bits", "group", "missing", "norm", "conflict"])
def test_precision_metadata_cannot_change_the_fixed_indexed_contract(indexed_config, change):
    config = copy.deepcopy(indexed_config)
    key = "language_model.model.embed_tokens"
    if change == "bits":
        config["quantization"][key]["bits"] = 6
    elif change == "group":
        config["quantization"][key]["group_size"] = 32
    elif change == "missing":
        del config["quantization"][key]
    elif change == "norm":
        config["qwen4_norm_convention"] = "folded"
    else:
        config["quantization"]["model.language_model.embed_tokens"] = config["quantization"][key]
    config["quantization_config"] = copy.deepcopy(config["quantization"])
    with pytest.raises(UnsupportedConversion):
        qwen38_kquant.validate_config(config)


def test_actual_header_planes_must_match_declared_packing(indexed_config):
    tensors = tensors_for(indexed_config)
    name = "language_model.model.layers.0.mlp.shared_expert_gate.weight"
    tensors[name] = ("BF16", (1, 128))
    with pytest.raises(UnsupportedConversion, match="affine tensor"):
        qwen38_kquant.validate_layout(indexed_config, tensors)
    tensors = tensors_for(indexed_config)
    del tensors["language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.scales"]
    with pytest.raises(UnsupportedConversion, match="affine tensor"):
        qwen38_kquant.validate_layout(indexed_config, tensors)


def test_ple_manifest_requires_contiguous_exact_rows_and_local_tensor_file(indexed_config):
    manifest = {
        "version": 2,
        "source_root": ".",
        "layout": "safetensors_ranges",
        "row_width": 32,
        "row_count": 128,
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "shards": [
            {
                "row_start": index * 64,
                "row_count": 64,
                **{
                    plane: {
                        "file": "ple.safetensors",
                        "offset": 8 + index * 1536 + offset,
                        "dtype": dtype,
                        "shape": [64, width],
                    }
                    for plane, dtype, width, offset in (
                        ("weight", "U32", 4, 0),
                        ("scales", "BF16", 1, 1024),
                        ("biases", "BF16", 1, 1152),
                    )
                },
            }
            for index in range(2)
        ],
    }
    qwen38_kquant.validate_manifest(indexed_config, manifest)
    for field, value in (("row_start", 0), ("row_count", 63)):
        broken = copy.deepcopy(manifest)
        broken["shards"][1][field] = value
        with pytest.raises(UnsupportedConversion, match="missing or overlapping"):
            qwen38_kquant.validate_manifest(indexed_config, broken)
    manifest["shards"][0]["weight"]["file"] = "../ple.safetensors"
    with pytest.raises(UnsupportedConversion, match="supported tensor file"):
        qwen38_kquant.validate_manifest(indexed_config, manifest)
