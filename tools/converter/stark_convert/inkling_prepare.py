"""Prepare the published Inkling Mini without changing its packed tensors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ._build_metadata import file_hash
from .streaming import open_regular

REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
CHOICE = "inkling2-mini"
FAMILY = "inkling2_loop"
VISION_PROFILE = "inkling2-mini-native-v1"
LAYOUT = "inkling2-mini-original"
TARGET = "model-00001-of-00001.safetensors"
DRAFT = "mtp.safetensors"
WEIGHTS = {
    TARGET: (19_526_019_168, "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"),
    DRAFT: (635_801_416, "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"),
}
# Stark's group-64 affine 8-bit dense artifact of the same checkpoint, built by
# inkling_affine8_artifact.py. The worker pins the same target hash.
AFFINE8_LAYOUT = "inkling2-mini-affine8-dense"
AFFINE8_TARGET_SHA = "c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd"
AFFINE8_TARGET_BYTES = 18_052_013_648
# Target hashes that carry the native Inkling image profile.
IMAGE_TARGET_SHAS = (WEIGHTS[TARGET][1], AFFINE8_TARGET_SHA)
CONFIG_SHA = "8216dd43157f5474cfff88f486a1e433fef703c674f694e485a86a761c412b1e"
CONFIG = "source-config.json"
TEMPLATE_SHA = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"
PROCESSOR_SHA = "d7350adce18c8be4b607f261c8f0f7eabbf720bb7aaf648a43d4f8372bbacc2d"
TOKENIZER_SHA = "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"
QUANT_SHA = "7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143"
INDEX = "model.safetensors.index.json"
INDEX_SHA = "c1631065b7b00787993522dddc66a68b3abee869192977fb3dcb796e63bc5838"
METADATA = [
    "config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "hf_quant_config.json",
    INDEX,
    "README.md",
]
SOURCE = {
    "architecture": FAMILY,
    "weights_repo": REPO,
    "weights_revision": REVISION,
    "weights_patterns": list(WEIGHTS),
    "metadata_repo": REPO,
    "metadata_revision": REVISION,
    "weight_bytes": sum(size for size, _ in WEIGHTS.values()),
    "weight_shards": len(WEIGHTS),
    "required_weight_files": list(WEIGHTS),
    "required_files": METADATA,
    "quantization": "Original NVFP4 / MXFP8",
}
_BINDING = {"source": REPO, "revision": REVISION}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(root: Path, name: str, limit: int, expected: str | None = None) -> bytes:
    with open_regular(root / name) as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError("Invalid Inkling metadata length")
    if expected is not None and _digest(data) != expected:
        raise ValueError("Inkling metadata identity changed")
    return data


def _installed_config(raw: bytes, template: bytes) -> bytes:
    config = json.loads(raw)
    config["stark_inkling"] = _BINDING
    config["token_codec_config"] = {
        "type": "ChatCodecConfig",
        "prompt_template": template.decode("utf-8"),
    }
    return (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()


def _receipt(encoded: bytes) -> dict:
    return {
        "schema_version": 7,
        "layout": LAYOUT,
        "architecture": FAMILY,
        "artifact_mode": "vision-language",
        "quantization": SOURCE["quantization"],
        "weights_source": {"repo": REPO, "revision": REVISION},
        "config_sha256": _digest(encoded),
        "weight_bytes": SOURCE["weight_bytes"],
        "artifact_sha256": WEIGHTS[TARGET][1],
        "draft_sha256": WEIGHTS[DRAFT][1],
        "processor_sha256": PROCESSOR_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "template_sha256": TEMPLATE_SHA,
        "native_loading": "unverified",
        "execution": "unverified",
    }


def stage(root: Path) -> dict:
    source_config = CONFIG if (root / CONFIG).exists() else "config.json"
    original = _read(root, source_config, 1024 * 1024, CONFIG_SHA)
    template = _read(root, "chat_template.jinja", 256 * 1024, TEMPLATE_SHA)
    _read(root, "processor_config.json", 128 * 1024, PROCESSOR_SHA)
    _read(root, "hf_quant_config.json", 1024 * 1024, QUANT_SHA)
    _read(root, INDEX, 1024 * 1024, INDEX_SHA)
    _read(root, "tokenizer.json", 32 * 1024 * 1024, TOKENIZER_SHA)
    for name, (size, digest) in WEIGHTS.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size or file_hash(path) != digest:
            raise ValueError("Inkling weight identity changed")
    encoded = _installed_config(original, template)
    (root / CONFIG).write_bytes(original)
    (root / "config.json").write_bytes(encoded)
    receipt = _receipt(encoded)
    (root / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def prepared_identity(root: Path, config: dict, receipt: dict) -> dict | None:
    if config.get("stark_inkling") != _BINDING:
        return None
    if receipt.get("layout") == AFFINE8_LAYOUT:
        # The derived 8-bit artifact has its own config binding, receipt and target size.
        from . import inkling_affine8_artifact

        return inkling_affine8_artifact.prepared_identity(root, config, receipt)
    if receipt.get("layout") != LAYOUT:
        return None
    raw = _read(root, CONFIG, 1024 * 1024, CONFIG_SHA)
    template = _read(root, "chat_template.jinja", 256 * 1024, TEMPLATE_SHA)
    encoded = _read(root, "config.json", 1024 * 1024)
    if encoded != _installed_config(raw, template) or receipt != _receipt(encoded):
        return None
    _read(root, "processor_config.json", 128 * 1024, PROCESSOR_SHA)
    _read(root, "hf_quant_config.json", 1024 * 1024, QUANT_SHA)
    _read(root, INDEX, 1024 * 1024, INDEX_SHA)
    for name, (size, _) in WEIGHTS.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            return None
    tokenizer = root / "tokenizer.json"
    if tokenizer.is_symlink() or not tokenizer.is_file() or not 0 < tokenizer.stat().st_size <= 32 * 1024 * 1024:
        return None
    return {
        "repo": REPO,
        "revision": REVISION,
        "architecture": FAMILY,
        "weight_bytes": SOURCE["weight_bytes"],
        "quantization": SOURCE["quantization"],
    }
