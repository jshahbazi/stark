"""Synthetic Flash Next images share source, preparation and host admission paths.

These fixtures exercise routing and provenance. They do not qualify a model's
native execution, image quality, numerical parity or performance.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_qwen38 import fixture as lily_checkpoint
from test_qwen38_external import PIN, RangeHub, tensor_bytes, tensor_files
from test_qwen38_external import external_config as external_config
from test_streaming import save_tensors
from test_vision_runtime import contract

from stark_convert import qwen38_external, qwen38_retained, sources, vision_runtime
from stark_convert.adapters.qwen38 import describe_text_weights
from stark_convert.cli import convert
from stark_convert.desktop import model_valid
from stark_convert.hybrid_prepare import prepared_identity
from stark_convert.hybrid_vision import prepared_image_identity
from stark_convert.model_inventory import model_display
from stark_convert.model_support import preflight_model, preflight_source
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion
from stark_convert.vision_family import _PROCESSOR

REPO = "independent-publisher/flashnext-affine-images"


@pytest.fixture(autouse=True)
def reviewed_synthetic_templates(monkeypatch):
    # Routing controls substitute tiny reviewed programs instead of copying
    # checkpoint templates with independently governed redistribution terms.
    synthetic = {
        hashlib.sha256(source.encode()).hexdigest() for source in ("{{ messages[0].content }}", "{{ messages }}")
    }
    monkeypatch.setattr(
        qwen38_retained, "SUPPORTED_TEMPLATE_HASHES", qwen38_retained.SUPPORTED_TEMPLATE_HASHES | synthetic
    )
    # The tiny graphs exercise source routing and content preservation. Native
    # geometry admission has separate controls with production dimensions.
    monkeypatch.setattr(qwen38_retained, "validate_native_geometry", lambda _architecture: None)


def image_checkpoint(root, config):
    """Write a tiny complete external affine graph with a retained draft head."""
    root.mkdir()
    config["text_config"]["rope_parameters"].update(
        mrope_section=[11, 11, 10], mrope_interleaved=True, rope_type="default"
    )
    config["text_config"]["mtp"] = {
        "hybrid": True,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "rope_theta": 10_000,
    }
    config.update(image_token_id=2, video_token_id=3, vision_start_token_id=4, vision_end_token_id=5)
    config["vision_config"] = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "depth": 1,
        "num_position_embeddings": 16,
        "patch_size": 16,
        "in_channels": 3,
        "out_hidden_size": config["text_config"]["hidden_size"],
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
    }
    tensors, table = tensor_files(config)

    def control(name, shape, integers=False):
        if name.startswith("mtp."):
            tensors[name] = ("I64" if integers else "BF16", shape)

    def matrix(stem, shape, *, group=None, fixed_q8=False, embedding=False, ngram=False):
        if not stem.startswith("mtp."):
            return
        if stem.endswith((".block_inject_weight", ".mlp.shared_expert_gate")):
            control(stem + ".weight", shape)
            return
        bits = 8 if fixed_q8 or group in {"attn", "gdn", "shared", "head"} else 4
        leading, columns = shape[:-1], shape[-1]
        for name, dtype, width in (
            ("weight", "U32", columns * bits // 32),
            ("scales", "BF16", columns // 64),
            ("biases", "BF16", columns // 64),
        ):
            tensors[stem + "." + name] = dtype, (*leading, width)

    describe_text_weights(qwen38_external._architecture(config), {}, matrix, control, include_mtp=True)
    shapes = {
        "patch_embed.proj.weight": (16, 3, 2, 16, 16),
        "patch_embed.proj.bias": (16,),
        "pos_embed.weight": (16, 16),
        "merger.norm.weight": (16,),
        "merger.norm.bias": (16,),
        "merger.linear_fc1.weight": (64, 64),
        "merger.linear_fc1.bias": (64,),
        "merger.linear_fc2.weight": (128, 64),
        "merger.linear_fc2.bias": (128,),
        "blocks.0.norm1.weight": (16,),
        "blocks.0.norm1.bias": (16,),
        "blocks.0.attn.qkv.weight": (48, 16),
        "blocks.0.attn.qkv.bias": (48,),
        "blocks.0.attn.proj.weight": (16, 16),
        "blocks.0.attn.proj.bias": (16,),
        "blocks.0.norm2.weight": (16,),
        "blocks.0.norm2.bias": (16,),
        "blocks.0.mlp.linear_fc1.weight": (32, 16),
        "blocks.0.mlp.linear_fc1.bias": (32,),
        "blocks.0.mlp.linear_fc2.weight": (16, 32),
        "blocks.0.mlp.linear_fc2.bias": (16,),
    }
    tensors.update({"model.visual." + name: ("BF16", shape) for name, shape in shapes.items()})
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}"}))
    (root / "preprocessor_config.json").write_text(json.dumps(_PROCESSOR))
    (root / "LICENSE").write_text("Synthetic source notice")
    (root / "model.safetensors").write_bytes(tensor_bytes(tensors))
    (root / "ngram_table.bin").write_bytes(tensor_bytes(table))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    return tensors


@pytest.mark.parametrize("remote", [False, True])
def test_generic_image_source_replay_and_preparation_preserve_tower_and_draft_head(tmp_path, external_config, remote):
    root = tmp_path / "checkpoint"
    tensors = image_checkpoint(root, external_config)
    assert any(name.startswith("mtp.") for name in tensors)
    assert any(name.startswith("model.visual.") for name in tensors)
    hub = RangeHub()
    hub.repos[REPO] = root, SimpleNamespace(sha=PIN)
    report = preflight_source(
        REPO if remote else root,
        **({"hub": hub} if remote else {"repo": REPO, "revision": PIN, "offline": True}),
        mode="vision-language",
    )
    assert report["status"] == "conditional"
    assert report["source"]["revision"] == PIN
    assert report["qualification"]["status"] == "not_assessed"
    snapshot = resolve_source(
        REPO if remote else root,
        **({"hub": hub} if remote else {"repo": REPO, "offline": True}),
        revision=PIN,
        metadata_profile="hf-generic",
        artifact_mode="vision-language",
    )
    if remote:
        first_payload = hub.events.index(("download", REPO, "model.safetensors"))
        assert any(event[0] == "range" for event in hub.events[:first_payload])
    assert {record["role"] for record in snapshot.manifest["files"]} >= {
        "model_config",
        "tokenizer",
        "tokenizer_config",
        "processor_config",
        "weights",
        "weight_index",
        "ngram_table",
    }
    manifest = tmp_path / "source.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(root, source_manifest=manifest, offline=True, artifact_mode="vision-language")
    output = tmp_path / "prepared"
    receipt = convert(replay, output, repo=REPO, artifact_mode="vision-language")
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert prepared_identity(output)["repo"] == REPO
    image = prepared_image_identity(output)
    assert (image["model_source"], image["model_revision"]) == (REPO, PIN)
    for name in ("model.safetensors", "ngram_table.bin", "model.safetensors.index.json", "LICENSE"):
        assert (output / name).read_bytes() == (root / name).read_bytes()
    assert (
        image["weight_files"]["model.safetensors"]["sha256"]
        == hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
    )
    prepared_config = json.loads((output / "config.json").read_text())
    assert prepared_config["text_config"]["mtp"] == external_config["text_config"]["mtp"]
    assert model_valid(output)
    display = model_display({"id": "fixture", "name": "Synthetic Flash Next", "path": str(output)})
    assert display["artifact_mode"] == "vision-language"
    assert display["source_revision"] == PIN


def test_lily_image_source_uses_same_generic_preparation_and_synthesizes_bound_index(tmp_path, external_config):
    reference = tmp_path / "external-shape-reference"
    image_tensors = image_checkpoint(reference, external_config)
    root = tmp_path / "lily-checkpoint"
    config, tensors = lily_checkpoint(root)
    # This is a source-format fixture. Native model geometry/execution is not
    # exercised by this small graph.
    for name in list(tensors):
        if name.startswith(("model.visual.", "mtp.")):
            del tensors[name]
    config["vision_config"] = {**external_config["vision_config"], "out_hidden_size": 64}
    config["lily"]["vision"] = {"dtype": "bf16", "tensors": 21}
    config["text_config"]["rope_parameters"].update(
        partial_rotary_factor=0.25, mrope_section=[11, 11, 10], mrope_interleaved=True, rope_type="default"
    )
    for field in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        config[field] = external_config[field]
    for name, (dtype, shape) in image_tensors.items():
        if not name.startswith("model.visual."):
            continue
        if name == "model.visual.merger.linear_fc2.weight":
            shape = (64, 64)
        elif name == "model.visual.merger.linear_fc2.bias":
            shape = (64,)
        tensors[name] = dtype, np.zeros(shape, dtype="<u2")
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(_PROCESSOR))
    save_tensors(root / "model.safetensors", tensors)
    snapshot = resolve_source(
        root, repo=REPO, revision=PIN, metadata_profile="hf-generic", artifact_mode="vision-language", offline=True
    )
    output = tmp_path / "prepared-lily"
    receipt = convert(snapshot, output, repo=REPO, artifact_mode="vision-language")
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert (output / "model.safetensors").read_bytes() == (root / "model.safetensors").read_bytes()
    assert prepared_identity(output)["repo"] == REPO
    assert prepared_image_identity(output)["model_revision"] == PIN
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == dict.fromkeys(tensors, "model.safetensors")
    assert model_valid(output)


def test_offline_image_replay_cannot_substitute_processor_after_source_binding(tmp_path, external_config):
    root = tmp_path / "checkpoint"
    image_checkpoint(root, external_config)
    snapshot = resolve_source(
        root, repo=REPO, revision=PIN, metadata_profile="hf-generic", artifact_mode="vision-language", offline=True
    )
    manifest = tmp_path / "source.json"
    snapshot.write_manifest(manifest)
    processor = root / "preprocessor_config.json"
    processor.write_text(json.dumps({**_PROCESSOR, "patch_size": 32}))
    with pytest.raises(ConversionError):
        resolve_source(root, source_manifest=manifest, artifact_mode="vision-language", offline=True)


def test_retained_runtime_contract_admits_generic_immutable_source_without_runtime_qualification():
    value = contract(
        artifact_schema_version=6,
        family_profile=vision_runtime.LILY_FAMILY_PROFILE,
        family_version=1,
        model_source=REPO,
        model_revision=PIN,
        max_processed_pixels=16_777_216,
        max_patches=65_536,
        max_image_tokens=16_384,
    )
    assert vision_runtime.image_input_contract(value) == value
    for source, revision in ((REPO, "main"), ("/private/model", PIN), (REPO, None)):
        changed = {**value, "model_source": source, "model_revision": revision}
        assert vision_runtime.image_input_contract(changed) is None


def test_headless_external_reports_only_modes_implemented_for_this_checkpoint(external_config):
    expected = {"text": "conditional", "decision": "unsupported", "vision-language": "unsupported"}
    for mode in expected:
        report = preflight_model(external_config, mode=mode)
        assert report["modes"] == expected
        assert report["alternatives"] == ([] if mode == "text" else ["text"])


def test_external_image_report_does_not_offer_unimplemented_decision_mode(tmp_path, external_config):
    image_checkpoint(tmp_path / "checkpoint", external_config)
    report = preflight_model(external_config, mode="vision-language")
    assert report["status"] == "conditional"
    assert report["modes"] == {"text": "conditional", "decision": "unsupported", "vision-language": "conditional"}
    assert report["alternatives"] == ["text"]


@pytest.mark.parametrize("mode", ["text", "decision", "vision-language"])
def test_unknown_flashnext_layout_never_offers_impossible_alternatives(mode):
    report = preflight_model(
        {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}, "quantization": {"bits": 3}},
        mode=mode,
    )
    assert report["status"] == "unsupported"
    assert report["modes"] == dict.fromkeys(("text", "decision", "vision-language"), "unsupported")
    assert report["alternatives"] == []


@pytest.mark.parametrize("repo", [None, REPO])
def test_unbound_local_image_source_stops_before_weight_hashing(tmp_path, monkeypatch, external_config, repo):
    root = tmp_path / "checkpoint"
    image_checkpoint(root, external_config)
    monkeypatch.setattr(
        sources, "_verified_hashes", lambda *_args, **_kwargs: pytest.fail("unbound source hashed weights")
    )
    report = preflight_source(root, repo=repo, mode="vision-language", offline=True)
    assert report["status"] == "inconclusive"
    assert report["reasons"] == ["image_source_identity_required"]
    with pytest.raises(UnsupportedConversion, match="immutable.*source revision"):
        resolve_source(root, repo=repo, metadata_profile="hf-generic", artifact_mode="vision-language", offline=True)
