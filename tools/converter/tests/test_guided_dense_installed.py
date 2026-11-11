"""Local dense plans use exact artifact geometry independently of discovery."""

import hashlib
import json
import struct

import pytest

from stark_convert import guided_dense_installed as local


@pytest.fixture
def dense(tmp_path, monkeypatch):
    config = {
        "type": "LanguageModelConfig",
        "decoder_config": {
            "transformer_config": {
                "model_dim": 1024,
                "hidden_dim": 4096,
                "layer_configs": [
                    {
                        "mixer_config": {"type": "AttentionConfig", "num_heads": 16, "num_groups": 2, "head_dim": 64},
                        "mlp_config": {"type": "DenseMLPConfig"},
                        "rope_config": {"max_sequence_length": 32768},
                    }
                ]
                * 8,
            }
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    header = {
        "matrix.weights": {"dtype": "U32", "shape": [2, 8], "data_offsets": [0, 64]},
        "__metadata__": {
            "matrix.spec": json.dumps(
                {
                    "type": "MLXSpec",
                    "layout": "output_input",
                    "bits": 4,
                    "group_size": 64,
                }
            )
        },
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(64))
    monkeypatch.setattr(
        local,
        "model_display",
        lambda _model: {
            "family_id": "qwen3",
            "source_revision": "a" * 40,
            "group_id": "repo:example/pinned",
        },
    )
    monkeypatch.setattr(
        local,
        "model_tool_support",
        lambda _root: {
            "status": "available",
            "parser": "qwen3-hermes-v1",
        },
    )
    return tmp_path, config


def test_local_dense_plan_expands_quantized_shapes_and_charges_context(dense):
    root, config = dense
    plan = local.inspect_dense(root)
    assert plan is not None
    assert plan["logical_parameters"] == 128  # 16 packed U32 words, 8 parameters each.
    assert plan["capacity_is_lower_bound"]
    assert plan["context_tokens"] == 32768
    assert plan["config_sha256"] == hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    original = plan["runtime_memory"]
    for layer in config["decoder_config"]["transformer_config"]["layer_configs"]:
        layer["mixer_config"]["num_groups"] = 16
    (root / "config.json").write_text(json.dumps(config))
    assert local.inspect_dense(root)["runtime_memory"] > original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(type="ClassifierModelConfig"),
        lambda c: c["decoder_config"]["transformer_config"].update(layer_configs=[]),
        lambda c: c["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"].update(
            type="GatedDeltaNetConfig"
        ),
        lambda c: c["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"].update(num_heads=True),
        lambda c: c["decoder_config"]["transformer_config"]["layer_configs"][0].pop("rope_config"),
        lambda c: c["decoder_config"]["transformer_config"]["layer_configs"][0]["rope_config"].update(
            max_sequence_length=511
        ),
    ],
)
def test_local_dense_unknown_or_malformed_memory_layout_is_not_admitted(dense, mutation):
    root, config = dense
    mutation(config)
    (root / "config.json").write_text(json.dumps(config))
    assert local.inspect_dense(root) is None


def test_local_dense_requires_source_identity_tools_and_valid_tensor_layout(dense, monkeypatch):
    root, _ = dense
    with monkeypatch.context() as patch:
        patch.setattr(local, "model_display", lambda _model: {"family_id": "qwen3"})
        assert local.inspect_dense(root) is None
    with monkeypatch.context() as patch:
        patch.setattr(local, "model_tool_support", lambda _root: {"status": "unavailable"})
        assert local.inspect_dense(root) is None
    (root / "model.safetensors").write_bytes(b"broken header")
    assert local.inspect_dense(root) is None
