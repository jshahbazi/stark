"""Conservative observed-memory preflight for exact retained image artifacts.

This is an allocation estimate, not artifact admission or a memory reservation.
The native loader verifies all payload hashes before allocating selected weights.
"""

from __future__ import annotations

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


def retained_check_memory(identity: dict) -> int:
    """Use the selected-residency recipe only for its immutable exact artifact.

    ``identity`` comes from the bounded retained-artifact metadata recognizer.
    This helper performs no file reads or writes, weight mapping or allocation.
    Unknown identities keep the original full-payload plus 2 GiB observation.
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
    if not exact or len(files) != 55 or total != _LILY_TOTAL:
        return total + 2 * _GIB
    return _LILY_RESIDENT + _LILY_MTP + sum(_LILY_RESERVES.values())
