"""Gemma image-profile admission and complete tensor-inventory planning."""

import json
import math
from pathlib import Path

import pytest

from stark_convert.adapters import gemma4_vision as vision
from stark_convert.streaming import ConversionError, ConversionPlan, TensorInfo, UnsupportedConversion


def source(tmp_path, model_dim=1536):
    config = {
        "model_type": "gemma4",
        "text_config": {"hidden_size": model_dim},
        "vision_config": dict(vision.GEOMETRY),
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "video_token_id": 258884,
        "audio_token_id": 258881,
    }
    processor = {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_processor_type": "Gemma4ImageProcessor",
        "image_seq_length": 280,
        "max_soft_tokens": 280,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1 / 255,
        "image_mean": [0.0] * 3,
        "image_std": [1.0] * 3,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": processor}))
    return config, processor


def test_gemma_image_contract_is_separate_and_has_exact_processing_semantics(tmp_path):
    config, _ = source(tmp_path)
    contract = vision.gemma4_vision_contract(tmp_path, vision.REPO, vision.REVISION)
    assert contract["schema_version"] == 4 and contract["transport"] == "mediaref-v1"
    assert contract["input_modalities"] == ["text", "image"]
    assert contract["source"] == {"repo": vision.REPO, "revision": vision.REVISION}
    config["vision_config"]["unimplemented_attention"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="unimplemented"):
        vision.gemma4_vision_contract(tmp_path)


@pytest.mark.parametrize(
    "key,value", [("pooling_kernel_size", 2), ("num_hidden_layers", 17), ("use_clipped_linears", False)]
)
def test_gemma_image_architectural_drift_is_rejected(tmp_path, key, value):
    config, _ = source(tmp_path)
    config["vision_config"][key] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion):
        vision.gemma4_vision_contract(tmp_path)


def test_gemma_image_processor_drift_and_packed_sources_are_rejected(tmp_path):
    config, processor = source(tmp_path)
    processor["resample"] = 2
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": processor}))
    with pytest.raises(UnsupportedConversion, match="processing"):
        vision.gemma4_vision_contract(tmp_path)
    source(tmp_path)
    config["quantization"] = {"bits": 4, "group_size": 64}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="BF16"):
        vision.gemma4_vision_contract(tmp_path)


def inventory(model_dim=1536):
    names = {
        "model.vision_tower.patch_embedder.input_proj.weight": (768, 768),
        "model.vision_tower.patch_embedder.position_embedding_table": (2, 10240, 768),
        "model.embed_vision.embedding_projection.weight": (model_dim, 768),
    }
    for i in range(16):
        prefix = f"model.vision_tower.encoder.layers.{i}"
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            names[f"{prefix}.{norm}.weight"] = (768,)
        for norm in ("q_norm", "k_norm"):
            names[f"{prefix}.self_attn.{norm}.weight"] = (64,)
        for projection, shape in (
            ("self_attn.q_proj", (768, 768)),
            ("self_attn.k_proj", (768, 768)),
            ("self_attn.v_proj", (768, 768)),
            ("self_attn.o_proj", (768, 768)),
            ("mlp.gate_proj", (3072, 768)),
            ("mlp.up_proj", (3072, 768)),
            ("mlp.down_proj", (768, 3072)),
        ):
            names[f"{prefix}.{projection}.linear.weight"] = shape
            for bound in ("input_min", "input_max", "output_min", "output_max"):
                names[f"{prefix}.{projection}.{bound}"] = ()
    return {
        name: TensorInfo(Path("header-only.safetensors"), "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0))
        for name, shape in names.items()
    }


def test_gemma_image_plan_preserves_text_and_requires_all_vision_tensors(tmp_path, monkeypatch):
    source(tmp_path)
    tensors = inventory()
    base = ConversionPlan(
        {"vocab_size": 262144}, (), {"source": "header-only"}, 0, len(tensors), tuple(tensors.values())
    )
    monkeypatch.setattr(vision, "build_gemma4_plan", lambda *_a, **_k: base)
    monkeypatch.setattr(vision, "load_checkpoint", lambda _p: tensors)
    plan = vision.build_gemma4_vision_plan(tmp_path)
    assert plan.decoder_config["gemma4_image_embeddings"] is True
    assert len(plan.operations) == 659
    assert all(op.name.startswith("vision.") for op in plan.operations)
    assert plan.input_tensors == base.input_tensors
    assert {op.name for op in plan.operations} >= {
        "vision.patch_embedder.position_embedding_table",
        "vision.embed_vision.embedding_projection.weight",
    }
    missing = tensors.pop("model.vision_tower.encoder.layers.15.mlp.down_proj.output_min")
    with pytest.raises(UnsupportedConversion, match="inventory"):
        vision.build_gemma4_vision_plan(tmp_path)

    tensors["model.vision_tower.encoder.layers.15.mlp.down_proj.output_min"] = missing
    tensors["model.embed_vision.embedding_projection.weight"] = TensorInfo(
        Path("header-only"), "F32", (1536, 768), 0, 1536 * 768 * 4, (0, 0, 0, 0)
    )
    with pytest.raises(ConversionError, match="BF16"):
        vision.build_gemma4_vision_plan(tmp_path)


def test_unified_12b_inventory_has_no_transformer_tower_and_strict_source_names(tmp_path, monkeypatch):
    config, processor = source(tmp_path, 3840)
    config.update(model_type="gemma4_unified", vision_config=dict(vision.UNIFIED_GEOMETRY))
    processor.pop("image_seq_length")
    processor["image_processor_type"] = "Gemma4UnifiedImageProcessor"
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_processor": processor, "image_seq_length": 280})
    )
    repo = "google/gemma-4-12B-it"
    contract = vision.gemma4_vision_contract(tmp_path, repo, vision.PROFILES[repo]["revision"])
    assert contract["family"] == {"id": "gemma4-12b-bf16-v1", "version": 1}
    tensors = {}
    shapes = {"patch_dense.weight": (3840, 6912), "patch_dense.bias": (3840,), "pos_embedding": (1120, 2, 3840)}
    for norm, width in (("patch_ln1", 6912), ("patch_ln2", 3840), ("pos_norm", 3840)):
        for plane in ("weight", "bias"):
            shapes[f"{norm}.{plane}"] = (width,)
    for name, shape in shapes.items():
        tensors["model.vision_embedder." + name] = TensorInfo(
            Path("header-only"), "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0)
        )
    shape = (3840, 3840)
    tensors["model.embed_vision.embedding_projection.weight"] = TensorInfo(
        Path("header-only"), "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0)
    )
    base = ConversionPlan({"vocab_size": 262144}, (), {}, 0, len(tensors), tuple(tensors.values()))
    monkeypatch.setattr(vision, "build_gemma4_plan", lambda *_a, **_k: base)
    monkeypatch.setattr(vision, "load_checkpoint", lambda _p: tensors)
    plan = vision.build_gemma4_vision_plan(tmp_path)
    assert len(plan.operations) == 10
    assert plan.decoder_config["scalar_image_embeddings"] is True
    assert plan.decoder_config["bidirectional_image_attention"] is True
    assert {op.name for op in plan.operations} == {"vision." + name for name in shapes} | {
        "vision.embed_vision.embedding_projection.weight"
    }
    tensor = tensors.pop("model.vision_embedder.patch_ln1.weight")
    tensors["model.embed_vision.patch_ln1.weight"] = tensor
    with pytest.raises(UnsupportedConversion, match="inventory"):
        vision.build_gemma4_vision_plan(tmp_path)


def test_e4b_profile_binds_its_projector_decoder_and_immutable_source(tmp_path, monkeypatch):
    source(tmp_path, 2560)
    repo = "google/gemma-4-E4B-it"
    profile = vision.PROFILES[repo]
    contract = vision.gemma4_vision_contract(tmp_path, repo, profile["revision"])
    assert contract["family"] == {"id": "gemma4-e4b-bf16-v1", "version": 1}
    with pytest.raises(UnsupportedConversion, match="pinned official"):
        vision.gemma4_vision_contract(tmp_path, vision.REPO, vision.REVISION)
    with pytest.raises(UnsupportedConversion, match="pinned official"):
        vision.gemma4_vision_contract(tmp_path, repo, "0" * 40)
    tensors = inventory(2560)
    base = ConversionPlan({"vocab_size": 262144}, (), {}, 0, len(tensors), tuple(tensors.values()))
    monkeypatch.setattr(vision, "build_gemma4_plan", lambda *_a, **_k: base)
    monkeypatch.setattr(vision, "load_checkpoint", lambda _p: tensors)
    plan = vision.build_gemma4_vision_plan(tmp_path)
    assert plan.metadata["stark.vision.family"] == "gemma4-e4b-bf16-v1"
    projector = next(op for op in plan.operations if op.name == "vision.embed_vision.embedding_projection.weight")
    assert projector.shape == (2560, 768)
    tensors["model.embed_vision.embedding_projection.weight"] = inventory()[
        "model.embed_vision.embedding_projection.weight"
    ]
    with pytest.raises(ConversionError, match="shape"):
        vision.build_gemma4_vision_plan(tmp_path)


@pytest.mark.parametrize("repo", ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"])
def test_registered_image_profiles_keep_execution_unverified(repo):
    from stark_convert.cli import capability
    from stark_convert.contracts import ConversionRequest

    result = capability(ConversionRequest(repo, artifact_mode="vision-language"))
    assert result.adapter == "gemma4_text" and result.conversion == "implemented"
    assert result.native_loading == "supported_family_unverified" and result.execution == "unverified"
    assert (
        capability(ConversionRequest(repo, dtype="float16", artifact_mode="vision-language")).conversion
        == "unsupported"
    )


@pytest.mark.parametrize("model_dim", [2816, 3840])
def test_larger_architectures_do_not_inherit_efficient_family_image_conversion(tmp_path, model_dim):
    source(tmp_path, model_dim)
    with pytest.raises(UnsupportedConversion, match="geometry"):
        vision.gemma4_vision_contract(tmp_path)


@pytest.mark.parametrize("model_dim", [2816, 5376])
def test_larger_vision_inventory_requires_standardization_and_plain_linear_weights(tmp_path, monkeypatch, model_dim):
    config, _ = source(tmp_path, model_dim)
    config["vision_config"] = dict(vision.LARGE_GEOMETRY)
    (tmp_path / "config.json").write_text(json.dumps(config))
    contract = vision.gemma4_vision_contract(tmp_path)
    assert contract["family"]["id"] == ("gemma4-31b-bf16-v1" if model_dim == 5376 else "gemma4-26b-a4b-bf16-v1")
    # Header-only source binds the complete 27-layer tower without allocating
    # the real 31B checkpoint or inheriting its execution qualification.
    tensors = {}
    for name, info in inventory(model_dim).items():
        if ".encoder.layers." in name:
            continue
        shape = tuple(1152 if dim == 768 else dim for dim in info.shape)
        tensors[name] = TensorInfo(info.path, "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0))
    for layer in range(27):
        prefix = f"model.vision_tower.encoder.layers.{layer}"
        shapes = {
            f"{norm}.weight": (1152,)
            for norm in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            )
        }
        shapes.update({"self_attn.q_norm.weight": (72,), "self_attn.k_norm.weight": (72,)})
        for projection, shape in (
            ("self_attn.q_proj", (1152, 1152)),
            ("self_attn.k_proj", (1152, 1152)),
            ("self_attn.v_proj", (1152, 1152)),
            ("self_attn.o_proj", (1152, 1152)),
            ("mlp.gate_proj", (4304, 1152)),
            ("mlp.up_proj", (4304, 1152)),
            ("mlp.down_proj", (1152, 4304)),
        ):
            shapes[projection + ".weight"] = shape
        for suffix, shape in shapes.items():
            tensors[prefix + "." + suffix] = TensorInfo(
                Path("header-only"), "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0)
            )
    tensors["model.vision_tower.patch_embedder.input_proj.weight"] = TensorInfo(
        Path("header-only"), "BF16", (1152, 768), 0, 1152 * 768 * 2, (0, 0, 0, 0)
    )
    for name in ("std_bias", "std_scale"):
        tensors["model.vision_tower." + name] = TensorInfo(Path("header-only"), "BF16", (1152,), 0, 2304, (0, 0, 0, 0))
    base = ConversionPlan({"vocab_size": 262144}, (), {}, 0, len(tensors), tuple(tensors.values()))
    monkeypatch.setattr(vision, "build_gemma4_plan", lambda *_a, **_k: base)
    monkeypatch.setattr(vision, "load_checkpoint", lambda _p: tensors)
    plan = vision.build_gemma4_vision_plan(tmp_path)
    assert plan.decoder_config["scalar_image_embeddings"] is True
    assert plan.decoder_config["bidirectional_image_attention"] is True
    assert plan.decoder_config["gemma4_image_embeddings"] is False
    assert len(plan.operations) == 356
    tensors.pop("model.vision_tower.std_scale")
    with pytest.raises(UnsupportedConversion, match="inventory"):
        vision.build_gemma4_vision_plan(tmp_path)
