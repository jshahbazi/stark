"""Pinned paged-table residency cannot weaken unrelated memory preflight."""

import pytest

from stark_convert import hybrid_image_memory as memory


def identity():
    # No sparse files, mappings, source payloads or GPU allocations are needed.
    files = {f"shard-{index}": {"size": 1} for index in range(55)}
    files["shard-0"]["size"] = memory._LILY_TOTAL - 54
    return {**memory._LILY_IDENTITY, "weight_files": files}


def test_exact_paged_artifact_budget_counts_selected_weights_and_peak_reserves():
    value = identity()
    required = memory.retained_check_memory(value)
    assert memory._LILY_RESIDENT + memory._LILY_PAGED + memory._LILY_MTP < memory._LILY_TOTAL
    assert memory._LILY_TOTAL - memory._LILY_RESIDENT - memory._LILY_PAGED - memory._LILY_MTP < 1024**2
    assert memory._LILY_RESERVES["initial_state"] > 394_616_832 + 23_089_152
    assert memory._LILY_RESERVES["decoder_scratch"] >= 256 * 1024**2 + 346_788_072
    assert memory._LILY_RESERVES["prefill_scratch"] >= 2 * 1024**3 + 83_886_080
    assert memory._LILY_RESERVES["paged_working_set"] >= 2 * 2048 * 16 * 3 * 4096
    assert required == memory._LILY_RESIDENT + memory._LILY_MTP + sum(memory._LILY_RESERVES.values())
    # A 96 GiB observation admits this pinned paged recipe, while retaining a
    # substantial reserve above the resident weights. Counting the whole file
    # payload would incorrectly reject the same observation.
    assert memory._LILY_RESIDENT + 12 * 1024**3 < required < 96 * 1024**3 < memory._LILY_TOTAL


@pytest.mark.parametrize("key", list(memory._LILY_IDENTITY))
def test_identity_drift_uses_full_payload_fallback(key):
    value = identity()
    value[key] = True if type(value[key]) is int else "changed"
    assert memory.retained_check_memory(value) == memory._LILY_TOTAL + 2 * 1024**3


def test_inventory_drift_and_external_source_use_full_payload_fallback():
    value = identity()
    value["weight_files"]["shard-0"]["size"] += 1
    assert memory.retained_check_memory(value) == memory._LILY_TOTAL + 1 + 2 * 1024**3
    value = identity()
    value["model_source"] = "ddalcu/Qwen3.8-Flash-Next-Mixed-MLX-Serve"
    assert memory.retained_check_memory(value) == memory._LILY_TOTAL + 2 * 1024**3


@pytest.mark.parametrize("files", [None, {}, {"shard": {"size": True}}, {"shard": {"size": -1}}])
def test_malformed_inventory_is_rejected(files):
    with pytest.raises(ValueError, match="memory inventory"):
        memory.retained_check_memory({**memory._LILY_IDENTITY, "weight_files": files})
