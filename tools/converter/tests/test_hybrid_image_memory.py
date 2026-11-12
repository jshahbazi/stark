"""Paged-table estimates require exact evidence or validated source headers."""

import hashlib
import json
import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_flashnext_image_source import image_checkpoint
from test_qwen38_external import PIN
from test_qwen38_external import external_config as external_config

from stark_convert import hybrid_image_memory as memory
from stark_convert import image_checks, qwen38_retained, streaming
from stark_convert.hybrid_prepare import stage_retained_qwen
from stark_convert.hybrid_vision import prepared_image_identity
from stark_convert.sources import resolve_source


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


@pytest.fixture
def generic_image(tmp_path, monkeypatch, external_config):
    template = "{{ messages[0].content }}"
    monkeypatch.setattr(
        qwen38_retained, "SUPPORTED_TEMPLATE_HASHES", frozenset({hashlib.sha256(template.encode()).hexdigest()})
    )
    # This byte-sized graph exercises structural accounting, not native kernels.
    monkeypatch.setattr(qwen38_retained, "validate_native_geometry", lambda _architecture: None)
    source = tmp_path / "source"
    tensors = image_checkpoint(source, external_config)
    snapshot = resolve_source(
        source,
        repo="example/memory-fixture",
        revision=PIN,
        metadata_profile="hf-generic",
        artifact_mode="vision-language",
        offline=True,
    )
    model = tmp_path / "model"
    model.mkdir()
    stage_retained_qwen(snapshot, model)
    config = json.loads((model / "config.json").read_bytes())
    return model, config, prepared_image_identity(model), tensors


def test_generic_residency_counts_full_decoder_mtp_and_tower_without_external_paged_table(generic_image, monkeypatch):
    model, config, value, tensors = generic_image
    monkeypatch.setattr(memory.os, "sysconf", lambda _key: 4096)
    resident = sum(math.prod(shape) * {"U32": 4, "BF16": 2, "I64": 8}[dtype] for dtype, shape in tensors.values())
    assert any(name.startswith("mtp.") for name in tensors)
    assert any(name.startswith("model.visual.") for name in tensors)
    assert memory.retained_check_memory(value, directory=model, config=config) == resident + sum(
        memory._LILY_RESERVES.values()
    )
    first = memory.retained_check_memory(value, directory=model, config=config)
    monkeypatch.setattr(memory.os, "sysconf", lambda _key: 65536)
    # Four synthetic n-gram heads touch 3 GiB at 64 KiB pages; the frozen
    # 4 KiB recipe reserves 1 GiB. The real sixteen-head graph scales likewise.
    assert memory.retained_check_memory(value, directory=model, config=config) == first + 2 * 1024**3


def test_generic_memory_reads_only_tensor_headers_and_keeps_payload_hash_admission_separate(
    generic_image, monkeypatch
):
    model, config, value, _ = generic_image
    original = streaming.open_regular
    boundaries = {}
    for name in value["weight_files"]:
        with (model / name).open("rb") as stream:
            boundaries[name] = 8 + int.from_bytes(stream.read(8), "little")

    @contextmanager
    def header_only(path):
        with original(path) as stream:
            if path.name not in boundaries:
                yield stream
                return
            boundary = boundaries[path.name]

            class HeaderStream:
                def __getattr__(self, key):
                    return getattr(stream, key)

                def read(self, count=-1):
                    assert count >= 0 and stream.tell() + count <= boundary, "memory preflight read a tensor payload"
                    return stream.read(count)

            yield HeaderStream()

    monkeypatch.setattr(streaming, "open_regular", header_only)
    required = memory.retained_check_memory(value, directory=model, config=config)
    name = "model.safetensors"
    with (model / name).open("r+b") as stream:
        stream.seek(boundaries[name])
        stream.write(b"\x01")
    # A same-size weight edit leaves the structural memory estimate unchanged.
    # Its stored hash still disagrees; only native admission can accept payloads.
    assert memory.retained_check_memory(value, directory=model, config=config) == required
    assert hashlib.sha256((model / name).read_bytes()).hexdigest() != value["weight_files"][name]["sha256"]


@pytest.mark.parametrize("case", ["missing_table", "bad_header", "wrong_size", "wrong_index", "wrong_config"])
def test_generic_estimate_rejects_malformed_or_unbound_inventory(generic_image, case):
    model, config, value, _ = generic_image
    if case == "missing_table":
        (model / "ngram_table.bin").unlink()
    elif case == "bad_header":
        with (model / "ngram_table.bin").open("r+b") as stream:
            stream.write(bytes(8))
    elif case == "wrong_size":
        value["weight_files"]["ngram_table.bin"]["size"] = 128 * 1024**3
    elif case == "wrong_index":
        (model / "model.safetensors.index.json").write_text("{}")
    else:
        config["vision_config"]["depth"] += 1
    with pytest.raises(ValueError, match="memory inventory"):
        memory.retained_check_memory(value, directory=model, config=config)


def test_generic_preflight_uses_observed_budget_and_fixed_inspection_errors(generic_image, monkeypatch):
    model, config, value, _ = generic_image
    required = memory.retained_check_memory(value, directory=model, config=config)
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: SimpleNamespace(available=required - 1))
    with pytest.raises(image_checks.ImageCheckError, match="available memory"):
        image_checks.preflight_family_memory(model, config)
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: SimpleNamespace(available=required))
    image_checks.preflight_family_memory(model, config)
    with (model / "ngram_table.bin").open("r+b") as stream:
        stream.write(bytes(8))
    with pytest.raises(image_checks.ImageCheckError, match="Cannot inspect"):
        image_checks.preflight_family_memory(model, config)
