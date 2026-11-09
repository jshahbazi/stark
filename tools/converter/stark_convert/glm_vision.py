"""Fail-closed source planning for the pinned GLM image implementation.

This validates a bounded configuration/profile only. It does not admit weights,
prepare an executable artifact or grant image/MTP runtime capability. Native
loading, tensor validation and model-bound qualification remain required.
"""

from __future__ import annotations

import hashlib
import json

PROFILE = "glm53-flash-bf16-vision-v1"
METADATA_REPO = "zai-org/GLM-5.3-Flash"
METADATA_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
SIDECAR = {
    "repo": "unsloth/GLM-5.3-Flash-GGUF",
    "revision": "a38483c8cd5df544f53d70fb281afe97369d5ab6",
    "filename": "mmproj-BF16.gguf",
    "size": 1_164_010_176,
    "sha256": "82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac",
}

_VISION = {
    "attention_bias": True,
    "attention_dropout": 0.0,
    "depth": 24,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "image_size": 448,
    "in_channels": 3,
    "intermediate_size": 4096,
    "model_type": "glm5_next_vision",
    "num_heads": 16,
    "out_hidden_size": 4096,
    "patch_size": 14,
    "projection_intermediate_size": 10240,
    "rms_norm_eps": 1e-5,
    "spatial_merge_size": 2,
    "swiglu_limit": 10.0,
    "temporal_patch_size": 2,
}
_TOKENS = {
    "image_token_id": 154854,
    "video_token_id": 154855,
    "image_start_token_id": 154830,
    "image_end_token_id": 154831,
    "video_start_token_id": 154832,
    "video_end_token_id": 154833,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate GLM image source key")
        result[key] = value
    return result


def _matches(value, expected):
    # bool compares equal to integer 0/1 in Python. Source shape/token fields
    # must retain their actual JSON scalar type before future native admission.
    return type(value) is type(expected) and value == expected


def plan_image_source(config_bytes: bytes, *, metadata_repo: str, metadata_revision: str) -> dict:
    """Bind the supported source geometry and required sidecar without I/O.

    The repository/revision arguments are caller claims until a preparation
    receipt and native content checks establish provenance. Returning this plan
    is deliberately insufficient for advertising or loading images.
    """
    if metadata_repo != METADATA_REPO or metadata_revision != METADATA_REVISION:
        raise ValueError("unsupported GLM image metadata source")
    if type(config_bytes) is not bytes or not 0 < len(config_bytes) <= 1024 * 1024:
        raise ValueError("invalid GLM image configuration size")
    try:
        config = json.loads(config_bytes, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("invalid GLM image configuration") from None
    if (
        not isinstance(config, dict)
        or config.get("model_type") != "glm5_next"
        or config.get("architectures") != ["Glm5NextForConditionalGeneration"]
        or not isinstance(config.get("text_config"), dict)
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise ValueError("unsupported GLM image architecture")
    text = config["text_config"]
    required_text = {
        "model_type": "glm5_next_text",
        "hidden_size": 4096,
        "vocab_size": 154880,
        "num_nextn_predict_layers": 1,
    }
    if not all(_matches(text.get(key), expected) for key, expected in required_text.items()):
        raise ValueError("unsupported GLM image decoder/head profile")
    if not all(_matches(config["vision_config"].get(key), expected) for key, expected in _VISION.items()):
        raise ValueError("unsupported GLM vision profile")
    if not all(_matches(config.get(key), expected) for key, expected in _TOKENS.items()):
        raise ValueError("unsupported GLM image token contract")
    return {
        "family_profile": PROFILE,
        "metadata_source": {"repo": metadata_repo, "revision": metadata_revision},
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "required_vision_sidecar": dict(SIDECAR),
        "declared_mtp_layers": text["num_nextn_predict_layers"],
        "weight_admission": "not_checked",
        "image_execution": "unimplemented",
    }
