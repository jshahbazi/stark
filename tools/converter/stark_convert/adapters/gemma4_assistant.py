"""Bounded companion conversion for the official Gemma 4 E2B/E4B assistants.

Source architecture: Transformers 5.15.1 modeling_gemma4_assistant.py,
SHA-256 a77e67673ce7d2248294e361c3c074c29d8d6a092655d7b27b95fc9777a2d0d1,
Copyright 2026 HuggingFace Team, Apache-2.0. Source tensors are preserved;
this companion is not a standalone chat model and grants no runtime capability.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .._build_metadata import _signature, open_regular
from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    _open_tensor,
    _unique_object,
    authorize_source_path,
    load_checkpoint,
)

PROFILES = {
    "7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9": {
        "repo": "google/gemma-4-E2B-it-assistant",
        "revision": "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df",
        "target_repo": "google/gemma-4-E2B-it",
        "target_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "backbone_dim": 1536,
        "kv_heads": 1,
    },
    "13cdf5418e5a89cf951717a953cd11cef4e5c2743c4767e9e5dd72802954f822": {
        "repo": "google/gemma-4-E4B-it-assistant",
        "revision": "8d0031ea8c2109e2b1e86bb9368a4539b537f80a",
        "target_repo": "google/gemma-4-E4B-it",
        "target_revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
        "backbone_dim": 2560,
        "kv_heads": 2,
    },
}


def tensor_contract(backbone_dim: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Complete source inventory; no K/V projections or independent KV cache."""
    if backbone_dim not in (1536, 2560):
        raise ConversionError("Unsupported Gemma assistant backbone")
    tensors = {
        "pre_projection.weight": ("BF16", (256, 2 * backbone_dim)),
        "post_projection.weight": ("BF16", (backbone_dim, 256)),
        "model.norm.weight": ("BF16", (256,)),
        "model.embed_tokens.weight": ("BF16", (262144, 256)),
        "masked_embedding.centroids.weight": ("BF16", (2048, 256)),
        "masked_embedding.token_ordering": ("I64", (262144,)),
    }
    for index in range(4):
        stem = f"model.layers.{index}"
        width = 512 if index == 3 else 256
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[f"{stem}.{norm}.weight"] = ("BF16", (256,))
        tensors[f"{stem}.layer_scalar"] = ("BF16", (1,))
        tensors[f"{stem}.self_attn.q_norm.weight"] = ("BF16", (width,))
        tensors[f"{stem}.self_attn.q_proj.weight"] = ("BF16", (4 * width, 256))
        tensors[f"{stem}.self_attn.o_proj.weight"] = ("BF16", (256, 4 * width))
        for branch in ("up", "gate"):
            tensors[f"{stem}.mlp.{branch}_proj.weight"] = ("BF16", (2048, 256))
        tensors[f"{stem}.mlp.down_proj.weight"] = ("BF16", (256, 2048))
    return tensors


def _validate_ordering(tensor) -> None:
    # Two MiB input plus bounded sort scratch. Inspect permutation values only;
    # never log them or trust indices before the complete range/uniqueness check.
    with _open_tensor(tensor) as stream:
        raw = stream.read(tensor.nbytes)
        if _signature(os.fstat(stream.fileno())) != tensor.file_signature:
            raise ConversionError("Gemma assistant changed while checking its ordering")
    if len(raw) != 262144 * 8:
        raise ConversionError("Gemma assistant ordering was truncated")
    values = np.frombuffer(raw, dtype="<i8")
    if not np.array_equal(np.sort(values), np.arange(262144, dtype=np.int64)):
        raise ConversionError("Gemma assistant ordering must be a vocabulary permutation")


def _config_bytes(path: Path) -> bytes:
    selected = authorize_source_path(path.parent.resolve(strict=True), path)
    with open_regular(selected) as stream:
        before = _signature(os.fstat(stream.fileno()))
        if before[2] > 1024 * 1024:
            raise ConversionError("Gemma assistant configuration exceeds its bound")
        raw = stream.read(1024 * 1024 + 1)
        if len(raw) != before[2] or _signature(os.fstat(stream.fileno())) != before:
            raise ConversionError("Gemma assistant configuration changed while reading")
    return raw


def build_gemma4_assistant_plan(source: Path, *, target: Path) -> ConversionPlan:
    """Plan an exact, separately bound companion without importing a framework.

    The selected target's config binds its weight digest. The runtime must
    verify both selected target descriptors before attaching this companion.
    No assistant or target filename is an authority for compatible execution.
    """
    source_bytes = _config_bytes(source / "config.json")
    digest = hashlib.sha256(source_bytes).hexdigest()
    profile = PROFILES.get(digest)
    if profile is None:
        raise ConversionError("Gemma assistant requires a pinned complete source configuration")
    config = json.loads(source_bytes)
    target_bytes = _config_bytes(target / "config.json")
    try:
        target_config = json.loads(target_bytes, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ConversionError("Cannot read Gemma assistant target configuration") from error
    if not isinstance(target_config, dict):
        raise ConversionError("Gemma assistant target configuration must be an object")
    contract = target_config.get("artifact_contract", {})
    decoder = target_config.get("decoder_config", {})
    if not isinstance(contract, dict) or not isinstance(decoder, dict):
        raise ConversionError("Gemma assistant target configuration is malformed")
    target_source = contract.get("source", {})
    transformer = decoder.get("transformer_config", {})
    components = contract.get("component_hashes", {})
    if not all(isinstance(value, dict) for value in (target_source, transformer, components)):
        raise ConversionError("Gemma assistant target configuration is malformed")
    if (
        target_config.get("type") != "Gemma4VisionLanguageModelConfig"
        or target_source.get("repo") != profile["target_repo"]
        or target_source.get("revision") != profile["target_revision"]
        or transformer.get("model_dim") != profile["backbone_dim"]
    ):
        raise ConversionError("Gemma assistant target does not match its pinned backbone")
    weights_hash = components.get("model.safetensors")
    if (
        not isinstance(weights_hash, str)
        or len(weights_hash) != 64
        or any(c not in "0123456789abcdef" for c in weights_hash)
    ):
        raise ConversionError("Gemma assistant target requires a complete selected weight identity")
    tensors = load_checkpoint(source)
    expected = tensor_contract(profile["backbone_dim"])
    if set(tensors) != set(expected) or any(
        (tensors[name].dtype, tensors[name].shape) != spec for name, spec in expected.items()
    ):
        raise ConversionError("Gemma assistant tensor inventory, dtype or shape mismatch")
    _validate_ordering(tensors["masked_embedding.token_ordering"])
    operations = tuple(
        TensorOperation("assistant." + name, (tensors[name],), dtype, shape)
        for name, (dtype, shape) in sorted(expected.items())
    )
    companion = {
        "type": "Gemma4AssistantConfig",
        "schema_version": 1,
        "source": {"repo": profile["repo"], "revision": profile["revision"], "config_sha256": digest},
        "target": {
            "source": {"repo": target_source["repo"], "revision": target_source["revision"]},
            "config_sha256": hashlib.sha256(target_bytes).hexdigest(),
            "weights_sha256": weights_hash,
        },
        "backbone_dim": profile["backbone_dim"],
        "kv_heads": profile["kv_heads"],
        "text_config": config["text_config"],
        "num_centroids": 2048,
        "centroid_top_k": 32,
        "execution": "single-position-read-only-target-kv-v1",
    }
    return ConversionPlan(
        decoder_config=companion,
        operations=operations,
        metadata={"stark.gemma4_assistant": json.dumps(companion, sort_keys=True, separators=(",", ":"))},
        input_bytes=sum(t.nbytes for t in tensors.values()),
        source_tensors=len(tensors),
        input_tensors=tuple(tensors[name] for name in sorted(tensors)),
    )
