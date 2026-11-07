"""Gemma image-profile admission and complete tensor-inventory planning."""

import json
import math
from pathlib import Path

import pytest

from stark_convert.adapters import gemma4_vision as vision
from stark_convert.streaming import ConversionError, ConversionPlan, TensorInfo, UnsupportedConversion


def source(tmp_path):
    config = {
        "model_type": "gemma4",
        "text_config": {"hidden_size": 1536},
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


def inventory():
    names = {
        "model.vision_tower.patch_embedder.input_proj.weight": (768, 768),
        "model.vision_tower.patch_embedder.position_embedding_table": (2, 10240, 768),
        "model.embed_vision.embedding_projection.weight": (1536, 768),
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
