"""Gemma image capabilities remain distinct from execution qualification."""

import pytest
from test_vision_runtime import contract

from stark_convert import vision_runtime


def gemma_contract(**overrides):
    value = contract(
        artifact_schema_version=4,
        model_source="google/gemma-4-E2B-it",
        model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        family_profile=vision_runtime.GEMMA4_FAMILY_PROFILE,
        family_version=1,
        max_patches=2520,
        max_image_tokens=280,
        max_processed_pixels=645120,
    )
    value.update(overrides)
    return value


def test_gemma_image_contract_is_recognized_without_inheriting_qwen_qualification(tmp_path):
    value = gemma_contract()
    assert vision_runtime.image_input_contract(value) == value
    assert vision_runtime.family_image_input_contract(value)
    binary = tmp_path / "worker"
    binary.write_bytes(b"unqualified synthetic worker")
    assert not vision_runtime.qualified_image_worker(value, binary)


@pytest.mark.parametrize(
    "overrides",
    [
        {"artifact_schema_version": 3},
        {"model_source": "other/model"},
        {"model_revision": "0" * 40},
        {"family_version": 2},
        {"max_patches": 4096},
        {"max_image_tokens": 1024},
        {"max_processed_pixels": 1048576},
    ],
)
def test_gemma_contract_rejects_mixed_family_and_geometry(overrides):
    assert vision_runtime.image_input_contract(gemma_contract(**overrides)) is None


def test_stored_family_descriptor_has_no_cross_family_fallback():
    config = {
        "type": "Gemma4VisionLanguageModelConfig",
        "artifact_contract": {
            "schema_version": 4,
            "family": {"id": vision_runtime.GEMMA4_FAMILY_PROFILE, "version": 1},
        },
    }
    assert vision_runtime.artifact_image_family(config) == {
        "profile": vision_runtime.GEMMA4_FAMILY_PROFILE,
        "version": 1,
    }
    config["type"] = "Qwen35VisionLanguageModelConfig"
    assert vision_runtime.artifact_image_family(config) is None
    config["type"] = "Gemma4VisionLanguageModelConfig"
    config["artifact_contract"]["schema_version"] = 3
    assert vision_runtime.artifact_image_family(config) is None
