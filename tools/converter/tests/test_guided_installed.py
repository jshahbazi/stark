import hashlib
import json
from pathlib import Path

import pytest

from stark_convert import guided_installed as guided
from stark_convert.hybrid_prepare import SOURCES
from stark_convert.streaming import TensorInfo


def qwen_config(context=262144):
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "vocab_size": 248320,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "num_hidden_layers": 48,
        "max_position_embeddings": context,
        "layer_types": guided._EXPECTED_LAYERS,
        "ple_layer_ids": [2],
        "mtp": {"layer_types": ["full_attention"], "num_hidden_layers": 1},
        "split_ngram_parts": 128,
    }
    return {
        "model_type": "qwen4_exp",
        "text_config": text,
        "lily": {
            "format": "qwen4_exp-affine-v1",
            "ple": {"padded_vocab_size": 320_001_536},
            "quantization": {"ngram_embedding": {"bits": 4, "group_size": 32, "mode": "affine"}},
        },
    }


@pytest.fixture
def candidate_setup(tmp_path, monkeypatch):
    config_bytes = json.dumps(qwen_config()).encode()
    (tmp_path / "config.json").write_bytes(config_bytes)
    source = SOURCES["qwen38-lily-q4-xl"]
    monkeypatch.setattr(
        guided,
        "prepared_identity",
        lambda _root: {
            "repo": source["weights_repo"],
            "revision": source["weights_revision"],
            "architecture": source["architecture"],
            "weight_bytes": source["weight_bytes"],
            "quantization": source["quantization"],
        },
    )
    monkeypatch.setattr(
        guided,
        "model_tool_support",
        lambda _root: {"status": "available", "parser": "qwen38-xml-v1"},
    )
    monkeypatch.setattr(guided, "_paged_table_weight_bytes", lambda _root, _config: 32_000_153_600)
    return tmp_path


def test_pinned_installed_candidate_returns_conservative_capacity_record(candidate_setup):
    item = guided.installed_qwen38_candidate(candidate_setup)
    assert item is not None
    assert item["repo"] == "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
    assert item["revision"] == "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
    assert item["architecture"] == "qwen4_exp"
    assert item["weight_bytes"] == 106_905_155_192
    assert item["context_tokens"] == 32768
    assert item["runtime_memory"] == 112 * guided.GIB
    assert item["gpu_runtime_memory"] == 83 * guided.GIB
    assert item["startup_memory"] == 83 * guided.GIB
    assert item["startup_memory"] < item["runtime_memory"]
    assert item["paged_table_bytes"] == 32_000_153_600
    assert item["runtime_memory"] > item["weight_bytes"]
    assert item["logical_parameters"] == 120_795_955_200
    assert "lower bound" in item["logical_parameters_basis"]
    assert item["config_sha256"] == hashlib.sha256((candidate_setup / "config.json").read_bytes()).hexdigest()


def test_installed_entrypoint_marks_capacity_lower_bound(candidate_setup):
    item = guided.inspect_installed(candidate_setup)
    assert item is not None
    assert item["capacity_is_lower_bound"] is True


def test_installed_entrypoint_falls_back_to_dense_planner(candidate_setup, monkeypatch):
    expected = {"architecture": "qwen3"}
    monkeypatch.setattr(guided, "installed_qwen38_candidate", lambda _root: None)
    monkeypatch.setattr("stark_convert.guided_dense_installed.inspect_dense", lambda _root: expected)
    assert guided.inspect_installed(candidate_setup) == {**expected, "capacity_is_lower_bound": True}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update(model_type="other"),
        lambda config: config.pop("lily"),
        lambda config: config["text_config"].update(hidden_size=2561),
        lambda config: config["text_config"].update(max_position_embeddings=511),
        lambda config: config["text_config"].update(max_position_embeddings=16384),
        lambda config: config["text_config"].update(layer_types=[]),
    ],
)
def test_candidate_rejects_unpinned_or_undersized_config(candidate_setup, mutation):
    config = qwen_config()
    mutation(config)
    (candidate_setup / "config.json").write_text(json.dumps(config))
    assert guided.installed_qwen38_candidate(candidate_setup) is None


def test_candidate_rejects_missing_config_and_tool_incompatible_artifact(candidate_setup, monkeypatch):
    (candidate_setup / "config.json").unlink()
    assert guided.installed_qwen38_candidate(candidate_setup) is None
    (candidate_setup / "config.json").write_text(json.dumps(qwen_config()))
    monkeypatch.setattr(guided, "model_tool_support", lambda _root: {"status": "unavailable"})
    assert guided.installed_qwen38_candidate(candidate_setup) is None


def test_malformed_native_tensor_headers_do_not_discard_other_recommendations(candidate_setup, monkeypatch):
    def malformed(_root, _config):
        raise guided.ConversionError("Invalid header")

    monkeypatch.setattr(guided, "_paged_table_weight_bytes", malformed)
    assert guided.installed_qwen38_candidate(candidate_setup) is None


def test_candidate_rejects_a_different_pinned_source(candidate_setup, monkeypatch):
    source = SOURCES["qwen38-mixed"]
    monkeypatch.setattr(
        guided,
        "prepared_identity",
        lambda _root: {
            "repo": source["weights_repo"],
            "revision": source["weights_revision"],
            "architecture": source["architecture"],
            "weight_bytes": source["weight_bytes"],
            "quantization": source["quantization"],
        },
    )
    assert guided.installed_qwen38_candidate(candidate_setup) is None


def test_paged_table_plane_accounting_matches_native_shapes():
    config = {
        "text_config": {"ngram_size": 3, "heads_per_ngram": 1, "ple_embed_dim": 64, "split_ngram_parts": 2},
        "lily": {"ple": {"padded_vocab_size": 128}, "quantization": {"ngram_embedding": {"group_size": 32}}},
    }
    tensors = {}
    for shard in range(2):
        base = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}"
        tensors[base + ".weight"] = TensorInfo(Path("unused"), "U32", (64, 4), 0, 1024, (0, 0, 0, 0))
        tensors[base + ".scales"] = TensorInfo(Path("unused"), "BF16", (64, 1), 0, 128, (0, 0, 0, 0))
        tensors[base + ".biases"] = TensorInfo(Path("unused"), "BF16", (64, 1), 0, 128, (0, 0, 0, 0))
    assert guided._count_paged_table_bytes(tensors, config) == 2560

    tensors.pop(next(name for name in tensors if name.endswith(".biases")))
    with pytest.raises(ValueError):
        guided._count_paged_table_bytes(tensors, config)
