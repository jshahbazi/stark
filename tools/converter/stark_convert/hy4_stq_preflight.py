"""Metadata-only compatibility preflight for one pinned Sushi Hy4 STQ pack."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path

from ._build_metadata import ConversionError
from .streaming import read_json, read_safetensors_header

REPOSITORY = "beamster/Hy4-Flash-Sushi-STQ-4bpw"
REVISION = "61528ba635a623f127c5a7b1e1f4ac2c2b0d4885"
PACK_REVISION = "capture-2a0815359e97"
SMALL_FILE_SHA256 = {
    "config.json": "c1c5af8449a8e7db0dc5c5e8aa768c80673c94604997cfe13c6ee6aa98e9a3e3",
    "model.safetensors.index.json": "6a3d938ac63427349ff8e647edd894cdfa00d3075c45cda4d1e7b662a99359ca",
}
TOTAL_FILES = 99
TOTAL_BYTES = 208_397_806_219
ENGRAM_LAYERS = (2, 12, 22, 32)
ENGRAM_FILES = tuple(f"engram-model-{index:05}-of-00148.safetensors" for index in range(1, 5))
ENGRAM_SHAPES = {
    1: (165_008_942, 96),
    2: (165_004_086, 96),
    3: (164_999_008, 96),
    4: (164_994_380, 96),
}
SIDECAR_FILES = frozenset(
    {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def _expected_files() -> set[str]:
    return {
        *SIDECAR_FILES,
        *ENGRAM_FILES,
        "model-dfly.safetensors",
        "model-globals.safetensors",
        *(f"model-layer-{layer:02}.safetensors" for layer in range(1, 42)),
        *(f"model-trunk-{layer:02}.safetensors" for layer in range(42)),
    }


def _require(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise ConversionError(f"Incompatible Sushi Hy4 STQ {field}")


def _require_quant(config: dict, field: str, expected: dict) -> None:
    value = config.get(field)
    if not isinstance(value, dict):
        raise ConversionError(f"Missing Sushi Hy4 STQ {field}")
    for key, required in expected.items():
        _require(value.get(key), required, f"{field}.{key}")


def _check_config(config: dict) -> tuple[dict, dict]:
    _require(config.get("model_type"), "hy_v4_flash", "model_type")
    _require(config.get("architectures"), ["HYV4FlashForConditionalGeneration"], "architecture")
    _require(config.get("dtype"), "bfloat16", "dtype")
    pack_revision = config.get("pack_revision")
    if not isinstance(pack_revision, dict):
        raise ConversionError("Missing Sushi Hy4 STQ pack revision")
    _require(pack_revision.get("id"), PACK_REVISION, "pack revision")
    _require(
        set(pack_revision.get("unstamped", [])),
        set(ENGRAM_FILES),
        "unstamped Engram shard set",
    )
    _require_quant(
        config,
        "trunk_quant",
        {
            "format": "stq",
            "k": 6,
            "codebook": "spec1",
            "window": 15,
            "codebook_table": "spec1-w15",
            "codebook_digest": "37f72431",
        },
    )
    _require_quant(
        config,
        "expert_quant",
        {
            "format": "stq",
            "k": 4,
            "codebook": "pair1",
            "window": 16,
            "codebook_table": "pair1-w16-v1",
            "codebook_digest": "c089b4eb",
            "tile_order": "pair_major",
        },
    )
    _require_quant(
        config,
        "drafter_quant",
        {
            "format": "stq",
            "k": 4,
            "codebook": "spec1",
            "window": 15,
            "codebook_table": "spec1-w15",
            "codebook_digest": "37f72431",
        },
    )
    _require(config.get("engram_tables"), "bf16", "Engram table dtype")

    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ConversionError("Sushi Hy4 STQ requires text and image configurations")
    _require(text.get("model_type"), "hy_v4_flash", "text model type")
    _require(text.get("num_hidden_layers"), 42, "trunk layer count")
    _require(text.get("n_routed_experts"), 512, "routed expert count")
    _require(text.get("num_experts_per_tok"), 16, "routed expert top-k")
    _require(text.get("engram_layer_ids"), list(ENGRAM_LAYERS), "Engram layer IDs")
    layer_types = text.get("mlp_layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != 42 or layer_types.count("dense") != 1:
        raise ConversionError("Incompatible Sushi Hy4 STQ MLP layer layout")
    routed_layers = [index for index, kind in enumerate(layer_types) if kind == "sparse"]
    if routed_layers != list(range(1, 42)):
        raise ConversionError("Incompatible Sushi Hy4 STQ sparse layer IDs")
    _require(vision.get("model_type"), "hy_v4_flash", "vision model type")
    _require(vision.get("hidden_size"), 1152, "vision hidden size")
    _require(vision.get("num_hidden_layers"), 27, "vision layer count")

    speculative = config.get("speculative_config")
    if not isinstance(speculative, dict):
        raise ConversionError("Missing Sushi Hy4 STQ DFly2 configuration")
    _require(speculative.get("method"), "dfly", "drafter method")
    _require(speculative.get("model"), "dfly", "drafter model")
    _require(speculative.get("num_speculative_tokens"), 5, "drafter depth")
    _require(config.get("dfly_num_hidden_layers"), 5, "DFly2 layer count")
    _require(config.get("dfly_enable_hidden_correction"), True, "DFly2 hidden correction")
    return text, vision


def _scan_files(root: Path) -> tuple[set[str], int]:
    if root.is_symlink() or not root.is_dir():
        raise ConversionError("Sushi Hy4 STQ source must be a real directory")
    names: set[str] = set()
    total = 0
    for entry in root.iterdir():
        try:
            info = entry.lstat()
        except OSError as exc:
            raise ConversionError("Cannot stat Sushi Hy4 STQ source entry") from exc
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or entry.name in names:
            raise ConversionError("Sushi Hy4 STQ source contains a non-regular or duplicate entry")
        names.add(entry.name)
        total += info.st_size
    _require(len(names), TOTAL_FILES, "file count")
    _require(total, TOTAL_BYTES, "total file bytes")
    _require(names, _expected_files(), "top-level file inventory")
    return names, total


def _check_small_file_identity(root: Path) -> None:
    for name, expected in SMALL_FILE_SHA256.items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        _require(digest, expected, f"{name} SHA-256")


def _check_shard_metadata(name: str, metadata: dict[str, str]) -> None:
    if name.startswith("model-layer-"):
        expected = {
            "format": "stq",
            "k": "4",
            "codebook": "pair1",
            "window": "16",
            "quantizer": "ldlq",
            "tile_order": "pair_major",
            "pack_revision": PACK_REVISION,
        }
    elif name == "model-dfly.safetensors":
        expected = {"format": "stq", "k": "4", "codebook": "spec1", "window": "15", "pack_revision": PACK_REVISION}
    elif name.startswith("model-trunk-") or name == "model-globals.safetensors":
        expected = {"format": "stq", "k": "6", "codebook": "spec1", "window": "15", "pack_revision": PACK_REVISION}
    else:
        _require(metadata, {}, f"{name} metadata")
        return
    for key, value in expected.items():
        _require(metadata.get(key), value, f"{name} {key}")


def _tensor_summary(config: dict, index: dict, tensors: dict) -> dict:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConversionError("Invalid Sushi Hy4 STQ Safetensors index")
    if not isinstance(index.get("metadata"), dict):
        raise ConversionError("Missing Sushi Hy4 STQ index metadata")
    _require(index["metadata"].get("total_size"), 208_387_139_446, "indexed tensor bytes")

    actual_names = set(tensors)
    indexed_names = set(weight_map)
    _require(actual_names, indexed_names, "header and index tensor inventory")
    _require(len(actual_names), 3179, "tensor count")
    if any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items()):
        raise ConversionError("Invalid Sushi Hy4 STQ index entry")

    for name, tensor in tensors.items():
        if weight_map.get(name) != tensor.path.name:
            raise ConversionError("Sushi Hy4 STQ index points to the wrong tensor shard")

    suffixes = Counter(name.rsplit(".", 1)[-1] for name in actual_names)
    _require(suffixes["suh"], 587, "SUH tensor count")
    _require(suffixes["svh"], 587, "SVH tensor count")
    _require(suffixes["trellis"], 464, "SPEC1 trellis tensor count")
    _require(suffixes["trellis_pm"], 123, "pair-major trellis tensor count")

    text, _vision = _check_config(config)
    routed_layers = [index for index, kind in enumerate(text["mlp_layer_types"]) if kind == "sparse"]
    expert_prefixes = set()
    for layer in routed_layers:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}.g0"
            expert_prefixes.add(prefix)
            for suffix in ("suh", "svh", "trellis_pm"):
                tensor_name = f"{prefix}.{suffix}"
                tensor = tensors.get(tensor_name)
                if tensor is None:
                    raise ConversionError("Incomplete Sushi Hy4 STQ pair-major expert group")
                if weight_map[tensor_name] != f"model-layer-{layer:02}.safetensors":
                    raise ConversionError("Sushi Hy4 STQ pair-major expert shard mismatch")
                if suffix == "trellis_pm" and (
                    tensor.dtype != "U16" or not tensor.shape or tensor.shape[0] != text["n_routed_experts"]
                ):
                    raise ConversionError("Incompatible Sushi Hy4 STQ pair-major expert plane")
    _require(len(expert_prefixes), 123, "pair-major expert projection count")

    for tensor_name in (
        "model.embed_tokens.suh",
        "model.embed_tokens.svh",
        "model.embed_tokens.trellis",
        "lm_head.suh",
        "lm_head.svh",
        "lm_head.trellis",
        "model.layers.1.self_attn.q_a_proj.suh",
        "model.layers.1.self_attn.q_a_proj.svh",
        "model.layers.1.self_attn.q_a_proj.trellis",
    ):
        if tensor_name not in tensors:
            raise ConversionError("Incomplete Sushi Hy4 STQ SPEC1 trunk projection")

    dfly_ids = {
        int(name.split(".")[2])
        for name in actual_names
        if name.startswith("model.dfly_layers.") and name.split(".")[2].isdigit()
    }
    _require(dfly_ids, set(range(5)), "DFly2 layer inventory")
    for layer in range(5):
        for projection in ("q_a_proj", "q_b_proj", "kv_b_proj"):
            if f"model.dfly_layers.{layer}.self_attn.{projection}.trellis" not in tensors:
                raise ConversionError("Incomplete Sushi Hy4 STQ DFly2 attention inventory")

    vision_names = {name for name in actual_names if name.startswith("vit.")}
    if len(vision_names) < 400 or "vit.embeddings.patch_embedding.weight" not in tensors:
        raise ConversionError("Incomplete Sushi Hy4 STQ vision tensor inventory")
    if not any(name.startswith("vit.perceive.proj.") for name in vision_names):
        raise ConversionError("Missing Sushi Hy4 STQ vision projector tensors")

    expected_lookup_layers = {layer - 1 for layer in ENGRAM_LAYERS}
    lookup_names = {name for name in actual_names if name.endswith(".engram_layer.lookup_table")}
    _require(
        {int(name.split(".")[2]) for name in lookup_names},
        expected_lookup_layers,
        "Engram lookup tensor layers",
    )
    engram_weight_names = {
        name for name in actual_names if name.endswith(".engram_layer.multi_head_embedding.embedding.weight")
    }
    _require(len(engram_weight_names), 4, "Engram embedding table count")
    engram_shapes = {}
    for name in engram_weight_names:
        tensor = tensors[name]
        if tensor.dtype != "BF16" or len(tensor.shape) != 2:
            raise ConversionError("Incompatible Sushi Hy4 STQ Engram table tensor")
        layer_id = int(name.split(".")[2])
        config_layer = next((layer for layer in ENGRAM_LAYERS if layer - 1 == layer_id), None)
        if config_layer is None:
            raise ConversionError("Unexpected Sushi Hy4 STQ Engram layer")
        file_index = len(ENGRAM_LAYERS) - ENGRAM_LAYERS.index(config_layer)
        expected_file = f"engram-model-{file_index:05}-of-00148.safetensors"
        if tensor.path.name != expected_file:
            raise ConversionError("Sushi Hy4 STQ Engram table shard mismatch")
        _require(tensor.shape, ENGRAM_SHAPES[file_index], "Engram table shape")
        engram_shapes[file_index] = tensor.shape
    _require(set(engram_shapes), set(range(1, 5)), "Engram table shard inventory")

    return {
        "tensor_count": len(actual_names),
        "dtype_counts": dict(sorted(Counter(tensor.dtype for tensor in tensors.values()).items())),
        "stq_planes": {key: suffixes[key] for key in ("suh", "svh", "trellis", "trellis_pm")},
        "routed_layers": len(routed_layers),
        "pair_major_expert_projection_groups": len(expert_prefixes),
        "trunk_spec1_examples": {
            name: {"dtype": tensors[name].dtype, "shape": list(tensors[name].shape)}
            for name in ("model.layers.1.self_attn.q_a_proj.trellis", "lm_head.trellis")
        },
        "dfly2_layers": len(dfly_ids),
        "dfly2_example": {
            name: {"dtype": tensors[name].dtype, "shape": list(tensors[name].shape)}
            for name in (
                "model.dfly_layers.0.self_attn.q_a_proj.trellis",
                "model.dfly_layers.0.mlp.gate_proj.trellis",
            )
        },
        "vision_tensor_count": len(vision_names),
        "vision_examples": {
            name: {"dtype": tensors[name].dtype, "shape": list(tensors[name].shape)}
            for name in ("vit.embeddings.patch_embedding.weight", "vit.perceive.proj.0.weight")
        },
        "engram_table_count": len(engram_weight_names),
        "engram_table_shapes": {str(index): list(shape) for index, shape in sorted(engram_shapes.items())},
        "engram_lookup_layers": sorted(expected_lookup_layers),
    }


def inspect_pack(model_dir: Path) -> dict:
    """Check pinned layout/config/header metadata without reading weight payloads."""
    root = Path(os.path.abspath(model_dir))
    _scan_files(root)
    _check_small_file_identity(root)
    config = read_json(root / "config.json")
    _check_config(config)
    index = read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConversionError("Invalid Sushi Hy4 STQ Safetensors index")
    shard_names = set(weight_map.values())
    expected_shards = {name for name in _expected_files() if name.endswith(".safetensors")}
    _require(shard_names, expected_shards, "indexed shard set")

    tensors = {}
    for shard_name in sorted(expected_shards):
        shard = root / shard_name
        padding_allowance = 8 if shard_name in {"model-dfly.safetensors", "model-globals.safetensors"} else 0
        metadata, shard_tensors = read_safetensors_header(shard, max_zero_padding_bytes=padding_allowance)
        _check_shard_metadata(shard_name, metadata)
        if not shard_name.startswith("engram-model-") and not metadata:
            raise ConversionError("Missing Sushi Hy4 STQ shard quantization metadata")
        for name, tensor in shard_tensors.items():
            if name in tensors:
                raise ConversionError("Duplicate Sushi Hy4 STQ tensor across shards")
            tensors[name] = tensor
    summary = _tensor_summary(config, index, tensors)
    return {
        "status": "metadata_preflight_passed",
        "evidence_scope": "local files, config, Safetensors indexes and bounded headers only",
        "model_repo": REPOSITORY,
        "model_revision": REVISION,
        "pack_revision": PACK_REVISION,
        "small_file_sha256": SMALL_FILE_SHA256,
        "file_count": TOTAL_FILES,
        "total_file_bytes": TOTAL_BYTES,
        "tensor_shard_count": len(expected_shards),
        "weight_payload_bytes_read": 0,
        **summary,
        "limits": [
            "No tensor payload or per-shard payload digest was read by this preflight.",
            "No dequantization, model load, inference, image or DFly2 execution, or memory admission was tested.",
            "This is format inventory evidence, not artifact qualification or catalog admission.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local unpacked Sushi Hy4 STQ model directory")
    args = parser.parse_args(argv)
    try:
        result = inspect_pack(args.model)
    except (ConversionError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
