"""Conservative observed-memory preflight for retained image artifacts.

This is an allocation estimate, not artifact admission or a memory reservation.
The native loader verifies all payload hashes before allocating selected weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .streaming import ConversionError, UnsupportedConversion

_GIB = 1024**3
_MIB = 1024**2
_LILY_IDENTITY = {
    "schema_version": 6,
    "family_profile": "qwen38-lily-retained-bf16-vision-v1",
    "family_version": 1,
    "model_source": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
    "model_revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
    "artifact_sha256": "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4",
    "config_sha256": "2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899",
    "index_sha256": "f7e35f06cb23dba09dea6197545111c092459f001715beead877c5f84959ccd4",
}
_LILY_TOTAL = 106_905_155_192
# Header-only source inventory. Includes the BF16 tower. The adapter selects
# NgramStorage::Paged. Vision-loaded models also retain the text MTP head;
# image execution disables its use without unloading the resident weights.
_LILY_RESIDENT = 73_062_399_712
_LILY_PAGED = 32_000_153_600
_LILY_MTP = 1_842_067_616

# Frozen adapter/vendor allocation recipe for a 2048-token image check:
# - startup KV capacity is 16384, despite the smaller requested context;
# - target KV/indexer/GDN states total 394,616,832 bytes, plus 23,089,152
#   bytes of MTP state; both fit the 512 MiB initial-state reserve;
# - prefill is bounded to 2048 rows and vision to 4096 patches;
# - page-touch allowance covers two contexts' 16 hashed rows/token, three
#   independently faulted planes/row and 4096-byte pages (768 MiB).
# Loaded-head SpecScratch adds 346,788,072 bytes even with image MTP off;
# preserve the old decoder allowance and round their sum up to 768 MiB.
# Two extra BF16 prefill owners add 83,886,080 bytes at 2048 rows, covered
# by an additional 128 MiB. Decoder dequant staging peaks at 62,914,560 bytes.
# Rounded buckets cover
# those typed tensors plus CPU patches, staging and temporary copies. The
# additional 8 GiB covers compiler/driver/allocator and available-memory drift.
_LILY_RESERVES = {
    "initial_state": 512 * _MIB,
    "decoder_scratch": 768 * _MIB,
    "prefill_scratch": 2 * _GIB + 128 * _MIB,
    "vision_cpu_and_gpu": 512 * _MIB,
    "paged_working_set": _GIB,
    "driver_allocator_and_host_margin": 8 * _GIB,
}


def retained_check_memory(identity: dict, *, directory: Path | None = None, config: dict | None = None) -> int:
    """Estimate selected residency from validated headers or frozen evidence.

    ``identity`` comes from the bounded retained-artifact metadata recognizer.
    The optional generic path reads headers and bounded metadata only; it never
    maps weights, hashes their payloads, allocates model state or reserves RAM.
    Calls without a model keep the original conservative observation fallback.
    """
    files = identity.get("weight_files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 256:
        raise ValueError("invalid retained image memory inventory")
    sizes = [record.get("size") if isinstance(record, dict) else None for record in files.values()]
    if any(type(size) is not int or not 0 < size <= 128 * _GIB for size in sizes):
        raise ValueError("invalid retained image memory inventory")
    total = sum(sizes)
    if total > 256 * _GIB:
        raise ValueError("invalid retained image memory inventory")
    exact = all(
        type(identity.get(key)) is type(value) and identity[key] == value for key, value in _LILY_IDENTITY.items()
    )
    if exact and len(files) == 55 and total == _LILY_TOTAL:
        return _LILY_RESIDENT + _LILY_MTP + sum(_LILY_RESERVES.values())
    if directory is None and config is None:
        return total + 2 * _GIB
    if directory is None or not isinstance(config, dict):
        raise ValueError("invalid retained image memory inventory")
    try:
        return _generic_check_memory(identity, Path(directory), config, total)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        ConversionError,
        UnsupportedConversion,
    ) as error:
        raise ValueError("invalid retained image memory inventory") from error


def _generic_check_memory(identity, directory, config, total):
    from . import qwen38_external, qwen38_kquant, qwen38_retained
    from .hybrid_vision import PROFILE, _read
    from .streaming import UnsupportedConversion, load_checkpoint, open_regular, read_safetensors_header

    if (
        identity.get("schema_version") != 6
        or identity.get("family_profile") != PROFILE
        or identity.get("family_version") != 1
        or not qwen38_retained.is_retained_affine(config)
    ):
        raise ValueError("unsupported retained image memory family")
    config_bytes = _read(directory / "config.json", 1048576)
    if hashlib.sha256(config_bytes).hexdigest() != identity.get("config_sha256") or json.loads(config_bytes) != config:
        raise ValueError("retained image memory metadata changed")
    index_bytes = _read(directory / "model.safetensors.index.json", 16777216)
    if hashlib.sha256(index_bytes).hexdigest() != identity.get("index_sha256"):
        raise ValueError("retained image memory index changed")
    files = identity["weight_files"]
    for name, record in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "\\" in name
            or name.startswith(".")
            or not (name.endswith(".safetensors") or name == qwen38_external.NGRAM_TABLE)
        ):
            raise ValueError("invalid retained image memory filename")
        with open_regular(directory / name) as stream:
            if stream.seek(0, 2) != record["size"]:
                raise ValueError("retained image memory file changed")
    tensors = load_checkpoint(directory)
    external = qwen38_external.is_external_affine(config)
    table = read_safetensors_header(directory / qwen38_external.NGRAM_TABLE)[1] if external else None
    try:
        architecture = qwen38_retained.validate_vision_config(config)
        qwen38_retained.validate_layout(config, tensors, table)
        if qwen38_kquant.is_kquant_affine(config):
            qwen38_kquant.validate_manifest(
                config, json.loads(_read(directory / qwen38_kquant.PLE_MANIFEST, 1048576)), tensors
            )
    except UnsupportedConversion as error:
        raise ValueError("unsupported retained image memory layout") from error
    expected_files = {tensor.path.name for tensor in tensors.values()}
    if external:
        expected_files.add(qwen38_external.NGRAM_TABLE)
    if set(files) != expected_files:
        raise ValueError("retained image memory shard inventory disagrees")
    alias = qwen38_kquant.alias if qwen38_kquant.is_kquant_affine(config) else qwen38_external._alias
    layers = architecture["ple_layer_ids"]
    paged_names = {
        f"model.language_model.layers.{layer - 1}.ple.ple_embedding.ngram_embedding.shard_{part}.{plane}"
        for layer in layers
        for part in range(architecture["split_ngram_parts"])
        for plane in ("weight", "scales", "biases")
    }
    if external:
        # The complete external table's three planes were independently
        # checked against the text architecture above; none are GPU-resident.
        resident = sum(info.nbytes for info in tensors.values())
        paged = sum(info.nbytes for info in table.values())
    else:
        present = {alias(name) for name in tensors if alias(name) in paged_names}
        if present != paged_names:
            raise ValueError("retained image memory requires complete paged PLE planes")
        resident = sum(info.nbytes for name, info in tensors.items() if alias(name) not in paged_names)
        paged = sum(info.nbytes for name, info in tensors.items() if alias(name) in paged_names)
    if resident <= 0 or resident + paged > total or (layers and paged <= 0):
        raise ValueError("invalid retained image memory payload accounting")
    reserves = dict(_LILY_RESERVES)
    vision = config["vision_config"]
    # Mirror request_memory::vision_request_inventory for the 4096-patch
    # check ceiling. Keep the frozen 512 MiB bucket when it is larger.
    patches = 4096
    patch_dim = vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2
    vision_bytes = (
        10 * patches * patch_dim
        + 14 * patches * vision["hidden_size"]
        + 2 * patches * vision["intermediate_size"]
        + patches * vision["out_hidden_size"]
        + 5 * patches * 36 * 4
        + 64 * _MIB
    )
    reserves["vision_cpu_and_gpu"] = max(reserves["vision_cpu_and_gpu"], vision_bytes)
    page_size = os.sysconf("SC_PAGE_SIZE")
    if type(page_size) is not int or not 4096 <= page_size <= 65536 or page_size & (page_size - 1):
        raise ValueError("unsupported host page size for retained image memory")
    rows_per_token = (architecture["ngram_size"] - 1) * architecture["heads_per_ngram"]
    touched_pages = 2 * 2048 * rows_per_token * 3
    page_bytes = touched_pages * page_size
    reserves["paged_working_set"] = max(_GIB, ((page_bytes + _GIB - 1) // _GIB) * _GIB)
    return resident + sum(reserves.values())
