import hashlib
import json
from copy import deepcopy

import pytest

from stark_convert import glm_vision


def config():
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "vocab_size": 154880,
            "num_nextn_predict_layers": 1,
        },
        "vision_config": dict(glm_vision._VISION),
        **glm_vision._TOKENS,
    }


def plan(value):
    return glm_vision.plan_image_source(
        json.dumps(value).encode(),
        metadata_repo=glm_vision.METADATA_REPO,
        metadata_revision=glm_vision.METADATA_REVISION,
    )


def test_glm_source_plan_retains_head_declaration_without_granting_runtime_capability():
    value = config()
    original = deepcopy(value)
    result = plan(value)
    assert result["declared_mtp_layers"] == 1
    assert result["image_execution"] == "unimplemented"
    assert result["weight_admission"] == "not_checked"
    assert result["required_vision_sidecar"]["filename"] == "mmproj-BF16.gguf"
    assert result["required_vision_sidecar"]["size"] == 1_164_010_176
    assert result["config_sha256"] == hashlib.sha256(json.dumps(value).encode()).hexdigest()
    assert value == original
    result["required_vision_sidecar"]["sha256"] = "0" * 64
    assert plan(value)["required_vision_sidecar"]["sha256"] == glm_vision.SIDECAR["sha256"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "model_type", "qwen4_exp"),
        (None, "image_token_id", 154855),
        (None, "video_end_token_id", None),
        ("vision_config", "patch_size", 16),
        ("vision_config", "depth", 27),
        ("vision_config", "attention_bias", 1),
        ("vision_config", "swiglu_limit", 0.0),
        ("text_config", "hidden_size", 1024),
        ("text_config", "num_nextn_predict_layers", True),
        ("text_config", "num_nextn_predict_layers", 0),
    ],
)
def test_glm_source_plan_rejects_changed_tower_tokens_and_head(section, key, value):
    candidate = config()
    (candidate if section is None else candidate[section])[key] = value
    with pytest.raises(ValueError, match="unsupported GLM"):
        plan(candidate)


@pytest.mark.parametrize("payload", [b"{}", b"[]", b'{"a":1,"a":2}', b"\xff", b"x" * (1024 * 1024 + 1)])
def test_glm_source_plan_rejects_ambiguous_or_excessive_metadata(payload):
    with pytest.raises(ValueError):
        glm_vision.plan_image_source(
            payload,
            metadata_repo=glm_vision.METADATA_REPO,
            metadata_revision=glm_vision.METADATA_REVISION,
        )


def test_glm_source_plan_does_not_accept_another_revision():
    with pytest.raises(ValueError, match="metadata source"):
        glm_vision.plan_image_source(
            json.dumps(config()).encode(), metadata_repo=glm_vision.METADATA_REPO, metadata_revision="f" * 40
        )
