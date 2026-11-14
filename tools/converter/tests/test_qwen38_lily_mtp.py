"""Native retained Lily heads receive their own metadata/header admission."""

import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest
from test_qwen38 import fixture as lily_checkpoint

from stark_convert import qwen38_lily, qwen38_retained
from stark_convert.adapters.qwen38 import architecture_config, build_qwen38_plan, describe_text_weights
from stark_convert.checkpoint_metadata import CheckpointMetadata
from stark_convert.streaming import DTYPE_BYTES, TensorInfo, UnsupportedConversion, load_checkpoint


def info(dtype, shape):
    size = math.prod(shape) * DTYPE_BYTES[dtype]
    return TensorInfo(Path("unopened.safetensors"), dtype, shape, 0, size, (0, 0, size, 0))


@pytest.fixture
def lily_head(tmp_path):
    config, _ = lily_checkpoint(tmp_path / "source")
    config["lily"]["mtp"] = {"layers": 1, "layer_types": ["full_attention"], "rope_theta": 10_000_000}
    return config


def head_tensors(config):
    architecture = architecture_config(config)[0]
    groups = set(architecture["quantization"]["q8_groups"])
    draft_q4 = config["lily"]["quantization"].get("draft_q4", False)
    tensors = {}

    def control(name, shape, integers=False):
        if name.startswith("mtp."):
            tensors[name] = info("BF16", shape)

    def matrix(stem, shape, *, group=None, fixed_q8=False, **_options):
        if not stem.startswith("mtp."):
            return
        bits = 8 if fixed_q8 or group in groups and not draft_q4 else 4
        *leading, columns = shape
        for plane, dtype, width in (
            ("weight", "U32", columns * bits // 32),
            ("scales", "BF16", columns // 64),
            ("biases", "BF16", columns // 64),
        ):
            tensors[stem + "." + plane] = info(dtype, (*leading, width))

    describe_text_weights(architecture, {}, matrix, control, include_mtp=True)
    if draft_q4 and "head" in groups:
        matrix("mtp.lm_head", (architecture["vocab_size"], architecture["hidden_size"]), group="head")
    return tensors


@pytest.mark.parametrize("draft_q4", [False, True])
def test_retained_lily_head_supports_native_draft_precision_and_flat_expert_rows(lily_head, draft_q4):
    lily_head["lily"]["quantization"]["draft_q4"] = draft_q4
    tensors = head_tensors(lily_head)
    # Header validation does not open or read the declared tensor payload path.
    for name, tensor in list(tensors.items()):
        if ".experts." in name:
            tensors[name] = info(tensor.dtype, (math.prod(tensor.shape[:-1]), tensor.shape[-1]))
    qwen38_lily.validate_mtp_config(lily_head)
    qwen38_lily.validate_mtp_layout(lily_head, tensors)
    assert ("mtp.lm_head.weight" in tensors) is draft_q4


@pytest.mark.parametrize(
    "patch",
    [
        {"layers": 2},
        {"layers": True},
        {"layer_types": ["linear_attention"]},
        {"layer_types": []},
        {"rope_theta": 0},
        {"rope_theta": True},
        {"rope_theta": float("nan")},
        {"rope_theta": float("inf")},
        {"rope_theta": 1e50},
        {"rope_theta": 1e-50},
    ],
)
def test_declared_lily_head_metadata_must_match_native_one_layer_contract(lily_head, patch):
    lily_head["lily"]["mtp"].update(patch)
    with pytest.raises(UnsupportedConversion, match="one-layer full-attention"):
        qwen38_lily.validate_mtp_config(lily_head)


@pytest.mark.parametrize("draft_q4", [0, "true", None])
def test_draft_precision_flag_is_native_boolean(lily_head, draft_q4):
    lily_head["lily"]["quantization"]["draft_q4"] = draft_q4
    with pytest.raises(UnsupportedConversion, match="draft precision"):
        qwen38_lily.validate_mtp_config(lily_head)


def test_draft_q4_requires_a_native_q8_draft_group(lily_head):
    lily_head["lily"]["quantization"].update(draft_q4=True, q8_groups=["embed"])
    with pytest.raises(UnsupportedConversion, match="draft precision"):
        qwen38_lily.validate_mtp_config(lily_head)


@pytest.mark.parametrize("flag", ["q8_dense", "q8_embed"])
@pytest.mark.parametrize("value", [0, None])
def test_lily_q8_storage_flags_match_native_boolean_types(lily_head, flag, value):
    lily_head["lily"]["quantization"][flag] = value
    with pytest.raises(UnsupportedConversion, match="native booleans"):
        qwen38_lily.validate_mtp_config(lily_head)


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "norm_dtype", "width", "extra"])
def test_lily_head_headers_cannot_omit_or_change_required_native_planes(lily_head, change):
    tensors = head_tensors(lily_head)
    name = "mtp.fc_embedding.weight"
    if change == "missing":
        del tensors[name]
    elif change == "shape":
        tensors[name] = info("U32", (63, 16))
    elif change == "dtype":
        tensors["mtp.fc_embedding.scales"] = info("F32", (64, 1))
    elif change == "norm_dtype":
        tensors["mtp.pre_fc_norm_embedding.weight"] = info("F32", (64,))
    elif change == "width":
        tensors[name] = info("U32", (64, 8))
    else:
        tensors["mtp.layers.1.unused.weight"] = info("BF16", (64,))
    with pytest.raises(UnsupportedConversion, match="draft"):
        qwen38_lily.validate_mtp_layout(lily_head, tensors)


def test_lily_draft_q4_head_copy_is_required_only_for_q8_target_readout(lily_head):
    lily_head["lily"]["quantization"]["draft_q4"] = True
    tensors = head_tensors(lily_head)
    del tensors["mtp.lm_head.biases"]
    with pytest.raises(UnsupportedConversion, match="missing required draft"):
        qwen38_lily.validate_mtp_layout(lily_head, tensors)
    lily_head["lily"]["quantization"]["q8_groups"].remove("head")
    tensors = head_tensors(lily_head)
    qwen38_lily.validate_mtp_layout(lily_head, tensors)
    tensors["mtp.lm_head.weight"] = info("U32", (16, 8))
    with pytest.raises(UnsupportedConversion, match="outside the declared"):
        qwen38_lily.validate_mtp_layout(lily_head, tensors)


def test_undeclared_lily_head_tensors_are_not_silently_skipped(lily_head):
    tensors = head_tensors(lily_head)
    lily_head["lily"].pop("mtp")
    qwen38_lily.validate_mtp_layout(lily_head, {})
    with pytest.raises(UnsupportedConversion, match="undeclared draft"):
        qwen38_lily.validate_mtp_layout(lily_head, tensors)


def test_retained_images_validate_declared_head_after_typed_text_plan(tmp_path, monkeypatch):
    root = tmp_path / "source"
    config, _ = lily_checkpoint(root)
    config["lily"]["mtp"] = {"layers": 1, "layer_types": ["full_attention"], "rope_theta": 10_000_000}
    tensors = {name: tensor for name, tensor in load_checkpoint(root).items() if not name.startswith("mtp.")}
    # Isolate draft admission from the unrelated tower/native-shape fixture.
    monkeypatch.setattr(qwen38_retained, "validate_vision_layout", lambda *_args: None)
    with pytest.raises(UnsupportedConversion, match="missing required draft"):
        qwen38_retained.validate_layout(copy.deepcopy(config), tensors)


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.1.self_attn.q_norm.weight",
        "model.language_model.layers.1.self_attn.indexer.q_layernorm.weight",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.ple.norm_key.weight",
        "model.language_model.layers.0.ple.conv1d.weight",
    ],
)
def test_retained_trunk_rejects_f32_controls_that_typed_reference_conversion_preserves(tmp_path, monkeypatch, name):
    source = tmp_path / "source"
    config, _ = lily_checkpoint(source)
    tensors = {key: value for key, value in load_checkpoint(source).items() if not key.startswith("mtp.")}
    tensors[name] = replace(tensors[name], dtype="F32", nbytes=tensors[name].nbytes * 2)
    # Header-only reference planning intentionally preserves F32 controls. The
    # retained native route must additionally check the consuming kernel.
    build_qwen38_plan(source, checkpoint_metadata=CheckpointMetadata(config, tensors))
    monkeypatch.setattr(qwen38_retained, "validate_vision_layout", lambda *_args: None)
    with pytest.raises(UnsupportedConversion, match="trunk control.*dtype"):
        qwen38_retained.validate_layout(config, tensors)


@pytest.mark.parametrize("dtype", ["BF16", "F32"])
@pytest.mark.parametrize("suffix", ["A_log", "norm.weight"])
def test_retained_gdn_controls_keep_the_native_bf16_to_f32_upcast_contract(tmp_path, dtype, suffix):
    source = tmp_path / "source"
    config, _ = lily_checkpoint(source)
    tensors = load_checkpoint(source)
    name = "model.language_model.layers.0.linear_attn." + suffix
    tensors[name] = info(dtype, tensors[name].shape)
    qwen38_lily.validate_trunk_control_dtypes(config, tensors)
