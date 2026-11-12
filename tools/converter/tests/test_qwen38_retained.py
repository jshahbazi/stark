"""Generic retained image admission is structural and repository independent."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_model_catalog import PIN
from test_qwen38_external import RangeHub, tensor_bytes, tensor_files
from test_qwen38_external import external_config as external_config

from stark_convert import qwen38_external, qwen38_retained
from stark_convert.hybrid_prepare import prepared_identity, stage_retained_qwen
from stark_convert.hybrid_vision import prepare_image_artifact, prepared_image_identity
from stark_convert.sources import resolve_source
from stark_convert.streaming import UnsupportedConversion
from stark_convert.vision_family import _PROCESSOR


@pytest.fixture(autouse=True)
def synthetic_reviewed_template(monkeypatch):
    template = "{{ messages[0].content }}"
    monkeypatch.setattr(
        qwen38_retained, "SUPPORTED_TEMPLATE_HASHES", frozenset({hashlib.sha256(template.encode()).hexdigest()})
    )


@pytest.fixture
def image_config(external_config, monkeypatch):
    # These small structural fixtures never claim native execution coverage.
    monkeypatch.setattr(qwen38_retained, "validate_native_geometry", lambda _architecture: None)
    config = copy.deepcopy(external_config)
    config["vision_config"] = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "depth": 1,
        "num_position_embeddings": 4,
        "patch_size": 16,
        "in_channels": 3,
        "out_hidden_size": config["text_config"]["hidden_size"],
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "hidden_act": "gelu_pytorch_tanh",
    }
    config["text_config"]["rope_parameters"].update(mrope_section=[11, 11, 10], mrope_interleaved=True)
    config.update(image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6)
    return config


def test_image_metadata_requires_native_geometry_and_exact_rotary_fusion():
    architecture = {
        **qwen38_external._NATIVE_TEXT_GEOMETRY,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    }
    qwen38_retained.validate_native_geometry(architecture)
    for patch in (
        {"hidden_size": 64},
        {"ple_layer_ids": [1]},
        {"rope_parameters": {**architecture["rope_parameters"], "partial_rotary_factor": 0.5}},
        {"layer_types": ["full_attention"] * 4},
    ):
        with pytest.raises(UnsupportedConversion, match="native Flash Next image kernels"):
            qwen38_retained.validate_native_geometry({**architecture, **patch})


def image_tensors(config):
    return {name: ("BF16", shape) for name, shape in qwen38_retained.vision_tensor_shapes(config).items()}


@pytest.mark.parametrize("case", ["valid", "alias", "missing", "extra", "quantized", "shape", "duplicate"])
def test_image_header_admission_requires_the_complete_exact_bf16_tower(image_config, case):
    tensors = image_tensors(image_config)
    name = "model.visual.blocks.0.attn.qkv.weight"
    if case == "alias":
        tensors = {name.replace("model.visual.", "vision_tower."): value for name, value in tensors.items()}
    elif case == "missing":
        del tensors[name]
    elif case == "extra":
        tensors["model.visual.unimplemented.weight"] = ("BF16", (8,))
    elif case == "quantized":
        tensors[name] = ("U32", tensors[name][1])
    elif case == "shape":
        tensors[name] = ("BF16", (8, 8))
    elif case == "duplicate":
        tensors[name.replace("model.visual.", "vision_tower.")] = tensors[name]
    if case in {"valid", "alias"}:
        qwen38_retained.validate_vision_layout(image_config, tensors)
    else:
        with pytest.raises(UnsupportedConversion, match="image tensors|image tower"):
            qwen38_retained.validate_vision_layout(image_config, tensors)


def test_channels_last_patch_shape_is_admitted_only_with_indexed_native_adapter(image_config, monkeypatch):
    # Header admission selects a storage adapter; the tiny graph is not loaded.
    monkeypatch.setattr(qwen38_retained, "validate_vision_config", lambda _config: None)
    monkeypatch.setattr(
        qwen38_retained.qwen38_kquant, "is_kquant_affine", lambda config: config.get("test_indexed", False)
    )
    tensors = image_tensors(image_config)
    name = "model.visual.patch_embed.proj.weight"
    out, channels, frames, height, width = tensors[name][1]
    tensors[name] = ("BF16", (out, frames, height, width, channels))
    with pytest.raises(UnsupportedConversion, match="BF16 shapes"):
        qwen38_retained.validate_vision_layout(image_config, tensors)
    qwen38_retained.validate_vision_layout({**image_config, "test_indexed": True}, tensors)


def test_image_processing_requires_source_recipe_and_template():
    files = {
        "preprocessor_config.json": json.dumps(_PROCESSOR).encode(),
        "tokenizer_config.json": json.dumps({"chat_template": "{{ messages[0].content }}"}).encode(),
    }
    qwen38_retained.validate_processing_metadata(files, lambda name, _maximum: files[name])
    with pytest.raises(Exception, match="processor metadata"):
        qwen38_retained.retained_files(
            "example/model",
            ("config.json", "tokenizer.json", "tokenizer_config.json"),
            {"model_type": "qwen4_exp", "lily": {"format": "qwen4_exp-affine-v1"}},
        )
    files["preprocessor_config.json"] = json.dumps({**_PROCESSOR, "patch_size": 32}).encode()
    with pytest.raises(UnsupportedConversion, match="processor recipe|processing metadata"):
        qwen38_retained.validate_processing_metadata(files, lambda name, _maximum: files[name])
    files["preprocessor_config.json"] = json.dumps(_PROCESSOR).encode()
    files["tokenizer_config.json"] = b"{}"
    with pytest.raises(UnsupportedConversion, match="chat-template source"):
        qwen38_retained.validate_processing_metadata(files, lambda name, _maximum: files[name])


def checkpoint(root, config, *, invalid_tower=False, indexed=True):
    tensors, planes = tensor_files(config)
    visual = image_tensors(config)
    if invalid_tower:
        name = next(iter(visual))
        visual[name] = ("F32", visual[name][1])
    tensors.update(visual)
    for name, data in {
        "config.json": json.dumps(config).encode(),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b'{"chat_template": "{{ messages[0].content }}"}',
        "preprocessor_config.json": json.dumps(_PROCESSOR).encode(),
        "model.safetensors": tensor_bytes(tensors),
        "ngram_table.bin": tensor_bytes(planes),
        "LICENSE": b"Synthetic license fixture",
    }.items():
        (root / name).write_bytes(data)
    if indexed:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
        )
    return tensors


@pytest.mark.parametrize("invalid_tower", [False, True])
def test_remote_tower_admission_reads_headers_before_payloads(tmp_path, image_config, invalid_tower):
    tensors = checkpoint(tmp_path, image_config, invalid_tower=invalid_tower)
    hub = RangeHub()
    repo = "example/future-flash-next"
    hub.repos[repo] = (tmp_path, SimpleNamespace(sha=PIN))
    args = (
        hub,
        repo,
        PIN,
        image_config,
        ("model.safetensors",),
        dict.fromkeys(tensors, "model.safetensors"),
        lambda: None,
    )
    if invalid_tower:
        with pytest.raises(UnsupportedConversion, match="BF16 shapes"):
            qwen38_retained.verify_hub_layout(*args)
    else:
        qwen38_retained.verify_hub_layout(*args)
    assert all(event[0] == "range" for event in hub.events)


@pytest.mark.parametrize("indexed", [False, True])
def test_generic_retained_preparation_preserves_payloads_and_binds_new_remote_source(tmp_path, image_config, indexed):
    root = tmp_path / "source"
    root.mkdir()
    checkpoint(root, image_config, indexed=indexed)
    hub = RangeHub()
    repo = "example/future-flash-next"
    hub.repos[repo] = (root, SimpleNamespace(sha=PIN))
    snapshot = resolve_source(
        repo, metadata_profile="hf-generic", artifact_mode="vision-language", revision=PIN, hub=hub
    )
    output = tmp_path / "prepared"
    output.mkdir()
    receipt = stage_retained_qwen(snapshot, output)
    for name in ("model.safetensors", "ngram_table.bin", "LICENSE"):
        assert (output / name).read_bytes() == (root / name).read_bytes()
    identity = prepared_identity(output)
    assert identity["repo"] == repo and identity["revision"] == PIN
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert receipt["artifact_mode"] == "vision-language"
    manifest = prepared_image_identity(output)
    assert manifest["model_source"] == repo and manifest["model_revision"] == PIN
    assert prepare_image_artifact(output) == manifest
    assert (output / "model.safetensors.index.json").is_file()
    config_path = output / "config.json"
    config = json.loads(config_path.read_bytes())
    config["stark_native_hybrid"]["source"] = "different/source"
    config_path.write_text(json.dumps(config))
    assert prepared_identity(output) is None
    assert prepared_image_identity(output) is None
