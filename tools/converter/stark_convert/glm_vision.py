"""Pinned GLM image-source planning and content-bound artifact preparation.

Preparation verifies payload bytes; bounded host inspection checks descriptors.
Neither grants runtime image/MTP capability. Native mapped-file admission,
operator loading and model-bound execution checks remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ._build_metadata import file_hash, open_regular

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
MANIFEST = "stark-image-artifact.json"
SIDECAR_PATH = "vision/mmproj-BF16.gguf"
PRECISION = "f32-vision-activations-v1"
# Public X-Linked-Size / X-Linked-ETag at the immutable source revision,
# independently rechecked before the native-image integration attempt.
DECODER_FILES = {
    "GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf": {
        "size": 9_429_984,
        "sha256": "2f32eba33ebc75a715e9c85a010faac4c67542e38e8970bd961d544bfe815f9e",
    },
    "GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf": {
        "size": 49_294_975_936,
        "sha256": "f4a9e1ab13d5d9620f5590c9a4aba4c169e6707a1554f3be3fe3112f80a66825",
    },
    "GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf": {
        "size": 49_949_266_048,
        "sha256": "330a8ad76c787b3ab6df062dd30abea7aafe6a0e3d5860e8720dfa4abb2434a5",
    },
    "GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf": {
        "size": 9_466_399_584,
        "sha256": "5c294f42edc5d69cf00a79ab444e61f0742e9933105fc5c85f97da54dab8946f",
    },
}
PROCESSOR = {
    "image_processor_type": "Glm5NextImageProcessor",
    "do_resize": True,
    "resample": 3,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "do_convert_rgb": True,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "patch_expand_factor": 1,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
    "native_max_patches": 1024,
}
PROCESSOR_BYTES = (json.dumps(PROCESSOR, sort_keys=True, separators=(",", ":")) + "\n").encode()

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


def _read(path: Path, maximum: int) -> bytes:
    with open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > maximum:
            raise ValueError("GLM image metadata exceeds bounds")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if len(data) > maximum or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("GLM image metadata changed")
        return data


def _weight_records() -> dict:
    return {
        **{name: dict(record) for name, record in DECODER_FILES.items()},
        SIDECAR_PATH: {key: SIDECAR[key] for key in ("size", "sha256")},
    }


def _aggregate(records: dict) -> str:
    digest = hashlib.sha256()
    for name, record in sorted(records.items()):
        digest.update(name.encode() + b"\0" + record["sha256"].encode())
    return digest.hexdigest()


def _same_records(actual, expected) -> bool:
    return (
        isinstance(actual, dict)
        and set(actual) == set(expected)
        and all(
            isinstance(actual[name], dict)
            and set(actual[name]) == {"size", "sha256"}
            and all(_matches(actual[name].get(key), record[key]) for key in ("size", "sha256"))
            for name, record in expected.items()
        )
    )


def _source(directory: Path) -> dict:
    from .hybrid_prepare import prepared_identity

    base = prepared_identity(directory)
    if base is None or (base["repo"], base["revision"], base["architecture"]) != (
        SIDECAR["repo"],
        SIDECAR["revision"],
        "glm5_next",
    ):
        raise ValueError("complete pinned GLM decoder required")
    source = json.loads(_read(directory / "native-source.json", 65536), object_pairs_hook=_unique_object)
    if (
        source.get("metadata_source") != {"repo": METADATA_REPO, "revision": METADATA_REVISION}
        or not _matches(source.get("schema_version"), 1)
        or not _matches(source.get("weight_bytes"), base["weight_bytes"])
    ):
        raise ValueError("unsupported GLM image metadata source")
    plan_image_source(
        _read(directory / "config.json", 1024 * 1024), metadata_repo=METADATA_REPO, metadata_revision=METADATA_REVISION
    )
    return base


def prepared_image_identity(directory: Path) -> dict | None:
    """Bounded manifest inspection; native loading rehashes every payload."""
    try:
        _source(directory)
        value = json.loads(_read(directory / MANIFEST, 65536), object_pairs_hook=_unique_object)
        fixed = {
            "kind": "Glm53NativeVisionArtifact",
            "schema_version": 7,
            "family_profile": PROFILE,
            "family_version": 1,
            "precision_policy": PRECISION,
            "model_source": SIDECAR["repo"],
            "model_revision": SIDECAR["revision"],
            "metadata_source": METADATA_REPO,
            "metadata_revision": METADATA_REVISION,
        }
        components = {
            "config.json": ("config_sha256", 1024 * 1024),
            "tokenizer.json": ("tokenizer_sha256", 64 * 1024 * 1024),
            "preprocessor_config.json": ("processor_sha256", 65536),
            "chat_template.jinja": ("template_sha256", 256 * 1024),
            "native-source.json": ("source_receipt_sha256", 65536),
        }
        if not isinstance(value, dict) or set(value) != set(fixed) | {key for key, _ in components.values()} | {
            "artifact_sha256",
            "weight_files",
        }:
            return None
        if not all(_matches(value.get(key), expected) for key, expected in fixed.items()):
            return None
        records = _weight_records()
        if not _same_records(value["weight_files"], records) or value["artifact_sha256"] != _aggregate(records):
            return None
        metadata = {}
        for name, (key, maximum) in components.items():
            data = _read(directory / name, maximum)
            if hashlib.sha256(data).hexdigest() != value[key]:
                return None
            metadata[name] = data
        config = json.loads(metadata["config.json"], object_pairs_hook=_unique_object)
        if (
            metadata["preprocessor_config.json"] != PROCESSOR_BYTES
            or metadata["chat_template.jinja"].decode() != config["token_codec_config"]["prompt_template"]
        ):
            return None
        vision = directory / "vision"
        if vision.is_symlink() or not vision.is_dir():
            return None
        for name, record in records.items():
            with open_regular(directory / name) as stream:
                if os.fstat(stream.fileno()).st_size != record["size"]:
                    return None
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
        return None


def prepare_image_artifact(directory: Path, sidecar: Path) -> dict:
    """Add the pinned image descriptor without modifying decoder/config bytes.

    The sidecar must already exist locally. Native installation owns download
    policy; this operation hashes all exact pinned payloads before publishing.
    """
    _source(directory)
    config = json.loads(_read(directory / "config.json", 1024 * 1024), object_pairs_hook=_unique_object)
    template = config["token_codec_config"]["prompt_template"].encode()
    if not 0 < len(template) <= 256 * 1024:
        raise ValueError("invalid GLM image template")
    # Verify every source payload before changing metadata or linking a sidecar.
    for name, record in DECODER_FILES.items():
        with open_regular(directory / name) as stream:
            if os.fstat(stream.fileno()).st_size != record["size"]:
                raise ValueError("GLM decoder payload size disagrees")
        if file_hash(directory / name) != record["sha256"]:
            raise ValueError("GLM decoder payload identity disagrees")
    with open_regular(sidecar) as stream:
        if os.fstat(stream.fileno()).st_size != SIDECAR["size"]:
            raise ValueError("GLM vision payload size disagrees")
    if file_hash(sidecar) != SIDECAR["sha256"]:
        raise ValueError("GLM vision payload identity disagrees")
    for name, expected in (("preprocessor_config.json", PROCESSOR_BYTES), ("chat_template.jinja", template)):
        path = directory / name
        if path.is_symlink() or path.exists() and _read(path, max(len(expected), 65536)) != expected:
            raise ValueError("GLM image metadata disagrees")
    vision = directory / "vision"
    if vision.is_symlink() or vision.exists() and not vision.is_dir():
        raise ValueError("invalid GLM image sidecar directory")
    vision.mkdir(mode=0o700, exist_ok=True)
    target = directory / SIDECAR_PATH
    if target.exists() or target.is_symlink():
        with open_regular(target) as stream:
            if os.fstat(stream.fileno()).st_size != SIDECAR["size"] or file_hash(target) != SIDECAR["sha256"]:
                raise ValueError("existing GLM vision payload disagrees")
    else:
        # The owned staging directory and verified immutable local file share
        # bytes. A cross-volume caller must stage it explicitly; never start an
        # unbudgeted gigabyte copy or replace an existing payload here.
        os.link(sidecar, target, follow_symlinks=False)
        with open_regular(target) as stream:
            if os.fstat(stream.fileno()).st_size != SIDECAR["size"] or file_hash(target) != SIDECAR["sha256"]:
                raise ValueError("linked GLM vision payload disagrees")
    for name, expected in (("preprocessor_config.json", PROCESSOR_BYTES), ("chat_template.jinja", template)):
        path = directory / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(expected)
    records = _weight_records()
    result = {
        "kind": "Glm53NativeVisionArtifact",
        "schema_version": 7,
        "family_profile": PROFILE,
        "family_version": 1,
        "precision_policy": PRECISION,
        "model_source": SIDECAR["repo"],
        "model_revision": SIDECAR["revision"],
        "metadata_source": METADATA_REPO,
        "metadata_revision": METADATA_REVISION,
        "config_sha256": file_hash(directory / "config.json"),
        "artifact_sha256": _aggregate(records),
        "weight_files": records,
        "tokenizer_sha256": file_hash(directory / "tokenizer.json"),
        "processor_sha256": file_hash(directory / "preprocessor_config.json"),
        "template_sha256": file_hash(directory / "chat_template.jinja"),
        "source_receipt_sha256": file_hash(directory / "native-source.json"),
    }
    if (directory / MANIFEST).is_symlink():
        raise ValueError("invalid GLM image manifest file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".glm-image-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(directory / MANIFEST)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
