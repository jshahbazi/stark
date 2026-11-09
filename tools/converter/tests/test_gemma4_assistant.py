"""Companion inventory, immutable target binding and index safety controls."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from stark_convert.adapters import gemma4_assistant as assistant
from stark_convert.streaming import ConversionError, read_safetensors_header

# Public source header inventories, read independently via HTTP range on
# 2026-11-09. Fixture records retain checkpoint revision and original header hash.
HEADERS = json.loads((Path(__file__).parent / "fixtures/gemma4_assistant_headers.json").read_text())


def fixture(tmp_path, monkeypatch, *, backbone=1536):
    source, target = tmp_path / "assistant", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    config = json.dumps({"text_config": {"fixture": True}}).encode()
    digest = hashlib.sha256(config).hexdigest()
    profile = next(p for p in assistant.PROFILES.values() if p["backbone_dim"] == backbone)
    monkeypatch.setitem(assistant.PROFILES, digest, profile)
    (source / "config.json").write_bytes(config)
    target_config = {
        "type": "Gemma4VisionLanguageModelConfig",
        "decoder_config": {"transformer_config": {"model_dim": backbone}},
        "artifact_contract": {
            "source": {"repo": profile["target_repo"], "revision": profile["target_revision"], "hashes": {}},
            "component_hashes": {"model.safetensors": "a" * 64},
        },
    }
    (target / "config.json").write_text(json.dumps(target_config))
    header, offset = {}, 0
    inventory = next(row["tensors"] for row in HEADERS if row["repo"] == profile["repo"])
    for name, spec in inventory.items():
        dtype, shape = spec["dtype"], spec["shape"]
        size = int(np.prod(shape)) * (8 if dtype == "I64" else 2)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    # Sparse all-zero matrices avoid materializing a full assistant fixture.
    # The ordering plane is real data because its content is an admission gate.
    with (source / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(8 + len(encoded) + offset)
        stream.seek(8 + len(encoded) + header["masked_embedding.token_ordering"]["data_offsets"][0])
        stream.write(np.arange(262144, dtype="<i8").tobytes())
    return source, target, target_config


@pytest.mark.parametrize("backbone,payload", [(1536, 157559816), (2560, 159132680)])
def test_complete_companion_inventory_and_explicit_target_binding(tmp_path, monkeypatch, backbone, payload):
    source, target, _ = fixture(tmp_path, monkeypatch, backbone=backbone)
    plan = assistant.build_gemma4_assistant_plan(source, target=target)
    assert len(plan.operations) == plan.source_tensors == 50
    assert plan.input_bytes == sum(op.nbytes for op in plan.operations) == payload
    assert all(op.kind == "copy" and op.name.startswith("assistant.") for op in plan.operations)
    assert plan.decoder_config["type"] == "Gemma4AssistantConfig"
    assert plan.decoder_config["execution"] == "single-position-read-only-target-kv-v1"
    assert (
        plan.decoder_config["target"]["config_sha256"]
        == hashlib.sha256((target / "config.json").read_bytes()).hexdigest()
    )
    assert plan.decoder_config["target"]["weights_sha256"] == "a" * 64
    assert plan.mtp_config is None  # A companion alone cannot grant text MTP.


@pytest.mark.parametrize(
    "mutation", ["source", "target", "missing", "dtype", "shape", "extra", "duplicate", "negative", "overflow"]
)
def test_rejects_unsupported_source_target_and_unsafe_tensor_indices(tmp_path, monkeypatch, mutation):
    source, target, target_config = fixture(tmp_path, monkeypatch)
    if mutation == "source":
        (source / "config.json").write_text("{}")
    elif mutation == "target":
        target_config["artifact_contract"]["source"]["revision"] = "0" * 40
        (target / "config.json").write_text(json.dumps(target_config))
    else:
        _, tensors = read_safetensors_header(source / "model.safetensors")
        if mutation in {"duplicate", "negative", "overflow"}:
            plane = tensors["masked_embedding.token_ordering"]
            with plane.path.open("r+b") as stream:
                stream.seek(plane.offset)
                stream.write(struct.pack("<q", {"duplicate": 1, "negative": -1, "overflow": 262144}[mutation]))
        else:
            # Interpose malformed parsed headers to isolate the complete
            # planner inventory gate from the separately tested file parser.
            from dataclasses import replace

            changed = dict(tensors)
            name = "pre_projection.weight"
            if mutation == "missing":
                del changed[name]
            elif mutation == "extra":
                changed["model.layers.0.self_attn.k_proj.weight"] = changed[name]
            else:
                changed[name] = replace(
                    changed[name], **({"dtype": "F16"} if mutation == "dtype" else {"shape": (1,)})
                )
            monkeypatch.setattr(assistant, "load_checkpoint", lambda _: changed)
    with pytest.raises(ConversionError):
        assistant.build_gemma4_assistant_plan(source, target=target)
