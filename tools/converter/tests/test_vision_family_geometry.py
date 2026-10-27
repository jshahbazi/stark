"""Config-derived geometry and native indexing bounds; no GPU qualification."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from stark_convert import vision_family
from stark_convert.streaming import TensorOperation, UnsupportedConversion
from stark_convert.vision_family import validate_family_geometry, validate_family_plan


def geometry(*, model=1024, hidden=3584, layers=24, q=8, kv=2, linear=16, tied=True, tower=(12, 768, 3072, 12)):
    depth, width, mlp, heads = tower
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": model,
            "intermediate_size": hidden,
            "num_hidden_layers": layers,
            "vocab_size": 248320,
            "num_attention_heads": q,
            "num_key_value_heads": kv,
            "head_dim": 256,
            "linear_num_value_heads": linear,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": tied,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            * (layers // 4),
            "full_attention_interval": 4,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": depth,
            "hidden_size": width,
            "intermediate_size": mlp,
            "num_heads": heads,
            "num_position_embeddings": 2304,
            "out_hidden_size": model,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
        },
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"model": 2048, "hidden": 6144, "tower": (24, 1024, 4096, 16)},
        {"model": 2560, "hidden": 9216, "layers": 32, "q": 16, "kv": 4, "linear": 32, "tower": (24, 1024, 4096, 16)},
        {
            "model": 4096,
            "hidden": 12288,
            "layers": 32,
            "q": 16,
            "kv": 4,
            "linear": 32,
            "tied": False,
            "tower": (27, 1152, 4304, 16),
        },
        {
            "model": 5120,
            "hidden": 17408,
            "layers": 64,
            "q": 24,
            "kv": 4,
            "linear": 48,
            "tied": False,
            "tower": (27, 1152, 4304, 16),
        },
    ],
)
def test_geometry_accepts_official_dense_config_dimensions(arguments):
    config = geometry(**arguments)
    text, vision = validate_family_geometry(config)
    assert text["hidden_size"] == config["text_config"]["hidden_size"]
    assert text["tie_word_embeddings"] == config["text_config"]["tie_word_embeddings"]
    assert vision["out_hidden_size"] == text["hidden_size"]
    assert vision["hidden_size"] == config["vision_config"]["hidden_size"]


def test_geometry_accepts_a_supported_shape_absent_from_official_size_names():
    config = geometry(model=3072, hidden=11008, layers=36, q=12, kv=3, linear=48, tower=(18, 896, 3584, 14))
    config["text_config"].update(vocab_size=300000, max_position_embeddings=131072, rms_norm_eps=1e-5)
    text, vision = validate_family_geometry(config)
    assert text["vocab_size"] == 300000
    assert vision["depth"] == 18


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("text_config", "hidden_size", 1025),
        ("text_config", "intermediate_size", 3585),
        ("text_config", "num_hidden_layers", 257),
        ("text_config", "num_attention_heads", 7),
        ("text_config", "num_key_value_heads", 0),
        ("text_config", "linear_num_value_heads", 17),
        ("text_config", "head_dim", 128),
        ("text_config", "linear_key_head_dim", 64),
        ("text_config", "linear_value_head_dim", 256),
        ("text_config", "linear_conv_kernel_dim", 3),
        ("text_config", "max_position_embeddings", 1048577),
        ("text_config", "vocab_size", 1048577),
        ("text_config", "tie_word_embeddings", 1),
        ("text_config", "rms_norm_eps", 0),
        ("text_config", "rms_norm_eps", 1e-50),
        ("text_config", "rms_norm_eps", 0.1),
        ("text_config", "rms_norm_eps", float("nan")),
        ("vision_config", "depth", 65),
        ("vision_config", "out_hidden_size", 2048),
        ("vision_config", "hidden_size", 4097),
        ("vision_config", "num_heads", 13),
        ("vision_config", "num_position_embeddings", 2305),
        ("vision_config", "num_position_embeddings", 16641),
        ("vision_config", "patch_size", 14),
    ],
)
def test_geometry_rejects_unsupported_ops_inconsistent_dimensions_and_budget_overflow(section, key, value):
    config = geometry()
    config[section][key] = value
    with pytest.raises(UnsupportedConversion):
        validate_family_geometry(config)


@pytest.mark.parametrize(
    "layers", [["linear_attention"] * 24, ["full_attention"] * 24, [{"type": "full_attention"}] * 24]
)
def test_geometry_rejects_unsupported_or_malformed_layer_schedule(layers):
    config = geometry()
    config["text_config"]["layer_types"] = layers
    with pytest.raises(UnsupportedConversion):
        validate_family_geometry(config)


def test_geometry_uses_the_explicit_supported_mixer_schedule():
    config = geometry()
    config["text_config"].pop("full_attention_interval")
    config["text_config"]["layer_types"][0] = "full_attention"
    validate_family_geometry(config)
    config["text_config"]["full_attention_interval"] = 4
    with pytest.raises(UnsupportedConversion, match="schedule"):
        validate_family_geometry(config)


def test_vision_head_vector_width_must_match_the_kernel_without_fixing_it_to_64():
    config = geometry(tower=(27, 1152, 4304, 16))
    validate_family_geometry(config)  # 72 elements per vision attention head.
    config = deepcopy(config)
    config["vision_config"]["hidden_size"] = 1056  # 66 is not a multiple of four.
    with pytest.raises(UnsupportedConversion, match="tower geometry"):
        validate_family_geometry(config)


def test_plan_rejects_tensor_element_indices_larger_than_uint32():
    operation = TensorOperation("synthetic", (), "BF16", (65536, 65536), "copy")
    with pytest.raises(UnsupportedConversion, match="indexing"):
        validate_family_plan(SimpleNamespace(operations=(operation,), metadata={}))


def test_plan_rejects_total_output_weights_larger_than_128_gib():
    operations = tuple(
        TensorOperation(f"synthetic.{index}", (), "BF16", (65535, 65535), "copy") for index in range(17)
    )
    with pytest.raises(UnsupportedConversion, match="weights exceed"):
        validate_family_plan(SimpleNamespace(operations=operations, metadata={}))


@pytest.mark.parametrize("nested", [False, True])
def test_processor_normalizes_the_recognized_class_serialization_alias(nested):
    recipe = deepcopy(vision_family._PROCESSOR)
    recipe["image_processor_type"] = "Qwen2VLImageProcessor"
    filename = "preprocessor_config.json"
    if nested:
        filename = "processor_config.json"
        recipe = {"processor_class": recipe.pop("processor_class"), "image_processor": recipe}
    assert vision_family.normalize_family_processor(recipe, filename) == vision_family._PROCESSOR
