"""Complete predictor preservation; no frozen result is inherited by this path."""
import json

import numpy as np
import pytest
from test_mlx import save_mlx_fixture
from test_mlx_qwen35 import mlx_qwen35_fixture

from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.streaming import ConversionError, read_safetensors_header, write_plan


def complete_mtp_fixture(source):
    config, tensors = mlx_qwen35_fixture(source, wrapper=True, vl_wrapper=True)
    config["text_config"]["mtp_num_hidden_layers"] = 1
    head = "language_model.mtp"
    trunk = "language_model.model"
    for name, value in list(tensors.items()):
        if name.startswith(trunk + ".layers.1."):
            tensors[name.replace(trunk + ".layers.1.", head + ".layers.0.")] = value
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        tensors[head + "." + name + ".weight"] = tensors[trunk + ".norm.weight"]
    # Fusion remains full precision, as in the released oQ4e checkpoint.
    tensors[head + ".fc.weight"] = "BF16", np.zeros((128, 256), dtype="<u2")
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    return config, tensors


def test_complete_mtp_head_is_preserved_with_explicit_graph(tmp_path):
    source = tmp_path / "source"
    _, tensors = complete_mtp_fixture(source)
    plan = build_qwen35_plan(source)
    assert plan.mtp_config["target_layer"] == 1
    assert plan.mtp_config["model_dim"] == 128
    assert plan.mtp_config["transformer_config"]["layer_configs"][0]["mixer_config"]["type"] == "AttentionConfig"
    assert any(op.name == "mtp.fusion.weights.weights" for op in plan.operations)
    output = tmp_path / "converted.safetensors"
    write_plan(plan, output, chunk_bytes=4096)
    _, written = read_safetensors_header(output)
    assert "mtp.fusion.weights.weights" in written
    assert "mtp.transformer.layers.0.mixer.query_norm.scales" in written


@pytest.mark.parametrize("mutation", ["missing_fusion", "bad_fusion", "undeclared", "multiple_layers", "extra"])
def test_partial_or_unsupported_predictor_fails_closed(tmp_path, mutation):
    source = tmp_path / "source"
    config, tensors = complete_mtp_fixture(source)
    if mutation == "missing_fusion":
        del tensors["language_model.mtp.fc.weight"]
    elif mutation == "bad_fusion":
        tensors["language_model.mtp.fc.weight"] = "BF16", np.zeros((128, 128), dtype="<u2")
    elif mutation == "undeclared":
        config["text_config"].pop("mtp_num_hidden_layers")
    elif mutation == "multiple_layers":
        config["text_config"]["mtp_num_hidden_layers"] = 2
    else:
        tensors["language_model.mtp.unexpected.weight"] = "BF16", np.zeros((2,), dtype="<u2")
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError):
        build_qwen35_plan(source)


def test_unqualified_unquantized_predictor_is_not_enabled(tmp_path):
    source = tmp_path / "source"
    config, tensors = complete_mtp_fixture(source)
    config.pop("quantization")
    config.pop("quantization_config")
    tensors = {name: value for name, value in tensors.items() if not name.endswith((".scales", ".biases"))}
    for name, (dtype, values) in list(tensors.items()):
        if dtype in {"I32", "U32"}:
            tensors[name] = "BF16", np.zeros((values.shape[0], values.shape[1] * 8), dtype="<u2")
    # Original raw HF exports use the explicitly declared root MTP namespace.
    tensors = {name.replace("language_model.mtp.", "mtp."): value for name, value in tensors.items()}
    (source / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(source, tensors)
    plan = build_qwen35_plan(source)
    assert plan.mtp_config is None
    assert not any(op.name.startswith("mtp.") for op in plan.operations)


def test_quantized_root_namespace_partial_head_is_rejected(tmp_path):
    source = tmp_path / "source"
    config, tensors = complete_mtp_fixture(source)
    for key in ("quantization", "quantization_config"):
        config[key]["mtp.fc"] = {"bits": 4, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    tensors = {name.replace("language_model.mtp.", "mtp."): value for name, value in tensors.items()}
    del tensors["mtp.fc.weight"]
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError, match="missing its fusion matrix"):
        build_qwen35_plan(source)
