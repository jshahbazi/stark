"""Gemma 3 image inventory, size admission and streamed conversion checks."""

import hashlib
import json

import ml_dtypes
import numpy as np
import pytest
from test_gemma import gemma_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.gemma import build_gemma_text_plan
from stark_convert.adapters.gemma3_vision import (
    TEXT_GEOMETRIES,
    TOKENS,
    VISION,
    gemma3_config,
    image_decoder_config,
    plan_vision_tensors,
    vision_shapes,
)
from stark_convert.gemma3_contract import PROCESSOR, normalize_processor
from stark_convert.streaming import (
    ConversionError,
    UnsupportedConversion,
    load_checkpoint,
    read_safetensors_header,
    write_plan,
)


def config_for_size(hidden):
    intermediate, layers, heads, groups, dim, scalar = TEXT_GEOMETRIES[hidden]
    return {
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "image_token_index": TOKENS["image"],
        "boi_token_index": TOKENS["vision_start"],
        "eoi_token_index": TOKENS["vision_end"],
        "mm_tokens_per_image": 256,
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": groups,
            "head_dim": dim,
            "query_pre_attn_scalar": scalar,
            "sliding_window": 1024,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        },
        "vision_config": dict(VISION),
    }


@pytest.mark.parametrize("hidden", TEXT_GEOMETRIES)
def test_all_actual_multimodal_sizes_admitted_by_geometry(hidden):
    text, vision = gemma3_config(config_for_size(hidden))
    assert text["hidden_size"] == hidden
    assert text["architectures"] == ["Gemma3ForCausalLM"]
    assert vision["num_hidden_layers"] == 27
    shapes = vision_shapes(hidden)
    assert shapes["multi_modal_projector.mm_input_projection_weight"] == (1152, hidden)
    assert len(shapes) == 439


def test_image_decoder_preserves_source_global_and_local_mask_composition():
    decoder = {
        "transformer_config": {
            "layer_configs": [
                {"mixer_config": {"sliding_window_size": 1024 if (index + 1) % 6 else None}} for index in range(12)
            ]
        }
    }
    result = image_decoder_config(decoder)
    policies = [
        layer["mixer_config"]["image_attention_policy"] for layer in result["transformer_config"]["layer_configs"]
    ]
    assert policies == (["windowed_same_image"] * 5 + ["same_image"]) * 2
    assert result["scalar_image_embeddings"] and result["bidirectional_image_attention"]
    assert all(
        "image_attention_policy" not in layer["mixer_config"]
        for layer in decoder["transformer_config"]["layer_configs"]
    )


@pytest.mark.parametrize(
    "pointer,value",
    [
        (("text_config", "hidden_size"), 1152),
        (("vision_config", "image_size"), 224),
        (("vision_config", "num_hidden_layers"), 26),
        (("vision_config", "vision_use_head"), True),
        (("vision_config", "unknown_operation"), True),
        (("image_token_index",), 1),
        (("mm_tokens_per_image",), 128),
        (("text_config", "num_attention_heads"), True),
    ],
)
def test_unimplemented_geometry_rejected(pointer, value):
    config = config_for_size(2560)
    target = config
    for key in pointer[:-1]:
        target = target[key]
    target[pointer[-1]] = value
    with pytest.raises((ConversionError, UnsupportedConversion)):
        gemma3_config(config)


def test_official_processor_optional_nulls_resolve_without_pan_and_scan():
    official = {
        **PROCESSOR,
        "do_convert_rgb": None,
        "do_pan_and_scan": None,
        "pan_and_scan_max_num_crops": None,
        "pan_and_scan_min_crop_size": None,
        "pan_and_scan_min_ratio_to_activate": None,
        "processor_class": "Gemma3Processor",
    }
    assert normalize_processor(official) == PROCESSOR
    for key, value in (
        ("do_pan_and_scan", True),
        ("resample", 3),
        ("image_mean", [0, 0, 0]),
        ("size", {"height": 896, "width": 448}),
        ("rescale_factor", 1 / 256),
    ):
        with pytest.raises(UnsupportedConversion):
            normalize_processor({**official, key: value})


def test_vision_streaming_fuses_qkv_and_retains_source_bound_norm(tmp_path):
    source = tmp_path / "source"
    config, text_tensors = gemma_fixture(source, "gemma3_text", head_dim=4)
    shapes = vision_shapes(8, hidden=16, intermediate=24, layers=2, image_size=8, patch=2)
    vision_tensors = {}
    for name, shape in shapes.items():
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 32 + len(vision_tensors)
        vision_tensors[name] = ("BF16", values.astype(ml_dtypes.bfloat16))
    base = build_gemma_text_plan(config, load_checkpoint(source), family="gemma3_text")
    save_tensors(source / "vision.safetensors", vision_tensors)
    _, selected = read_safetensors_header(source / "vision.safetensors")
    ops = plan_vision_tensors(selected, 8, shapes=shapes)
    from dataclasses import replace

    plan = replace(base, operations=ops, metadata={})
    output = tmp_path / "vision.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    _, tensors = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    for layer in range(2):
        source_prefix = f"vision_tower.vision_model.encoder.layers.{layer}.self_attn"
        for suffix in ("weight", "bias"):
            expected = b"".join(
                vision_tensors[f"{source_prefix}.{name}_proj.{suffix}"][1].tobytes() for name in ("q", "k", "v")
            )
            assert tensor_bytes(tensors[f"vision.encoder.layers.{layer}.self_attn.qkv.{suffix}"]) == expected
    assert tensor_bytes(tensors["vision.mm_soft_emb_norm.scales"]) == (
        vision_tensors["multi_modal_projector.mm_soft_emb_norm.weight"][1].astype("<f4").tobytes()
    )
    assert "vision.mm_input_projection_weight" in tensors
    assert tensors["vision.mm_input_projection_weight"].shape == (16, 8)
    for changed in (
        {**selected, "surplus": next(iter(selected.values()))},
        {name: info for name, info in selected.items() if name != next(iter(selected))},
    ):
        with pytest.raises(ConversionError):
            plan_vision_tensors(changed, 8, shapes=shapes)


def test_gemma3_source_selects_own_processor_and_replays_complete_provenance(tmp_path, monkeypatch):
    from stark_convert import gemma3_contract
    from stark_convert.sources import resolve_source

    source = tmp_path / "source"
    source.mkdir()
    template = "{% for message in messages %}{{message.content}}{% endfor %}"
    # This substitutes only the review identity for a tiny source-selection
    # fixture. It does not qualify a template or any native model execution.
    monkeypatch.setattr(gemma3_contract, "TEMPLATE_SHA256", hashlib.sha256(template.encode()).hexdigest())
    (source / "config.json").write_text(json.dumps(config_for_size(2560)))
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    (source / "preprocessor_config.json").write_text(
        json.dumps({**PROCESSOR, "do_convert_rgb": None, "do_pan_and_scan": None})
    )
    (source / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106]}))
    (source / "model.safetensors").write_bytes(b"source-selection-only")
    snapshot = resolve_source(
        source, repo="google/gemma-3-4b-it", revision="1" * 40, artifact_mode="vision-language", architecture="gemma3"
    )
    assert snapshot.conversion_profile == {"architecture": "gemma3", "metadata_profile": "hf-generic"}
    assert not (source / "chat_template.jinja").exists()
    contract = gemma3_contract.validate_gemma3_source(snapshot, "bfloat16")
    assert (
        contract["source"]["hashes"]["generation_config.json"]
        == hashlib.sha256((source / "generation_config.json").read_bytes()).hexdigest()
    )
    assert contract["source"]["hashes"]["chat_template.jinja"] == hashlib.sha256(template.encode()).hexdigest()
    assert contract["processor_config"] == PROCESSOR
    manifest = tmp_path / "source.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(
        "google/gemma-3-4b-it", source_manifest=manifest, artifact_mode="vision-language", offline=True, hub=object()
    )
    assert replay.receipt() == snapshot.receipt()
    (source / "preprocessor_config.json").write_text("{}")
    with pytest.raises(ConversionError):
        replay.assert_unchanged()
