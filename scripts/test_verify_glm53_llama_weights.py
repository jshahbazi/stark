from __future__ import annotations

import hashlib
import json
import os
import stat

import benchmark_hybrid
import pytest
import verify_glm53_llama_weights as verifier


@pytest.fixture
def synthetic_model(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    payloads = {
        name: f"synthetic-shard-{index}".encode()
        for index, name in enumerate(verifier.SHARDS, start=1)
    }
    monkeypatch.setattr(
        verifier,
        "SHARDS",
        {
            name: (len(data), hashlib.sha256(data).hexdigest())
            for name, data in payloads.items()
        },
    )
    total_bytes = sum(len(data) for data in payloads.values())
    monkeypatch.setattr(verifier, "TOTAL_WEIGHT_BYTES", total_bytes)
    for name, data in payloads.items():
        (model / name).write_bytes(data)

    config = {
        "model_type": verifier.ARCHITECTURE,
        "stark_native_hybrid": {
            "source": verifier.WEIGHTS_REPO,
            "revision": verifier.WEIGHTS_REVISION,
        },
    }
    config_bytes = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
    (model / "config.json").write_bytes(config_bytes)
    native_source = {
        "architecture": verifier.ARCHITECTURE,
        "quantization": verifier.QUANTIZATION,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weight_bytes": total_bytes,
        "weights_source": {
            "repo": verifier.WEIGHTS_REPO,
            "revision": verifier.WEIGHTS_REVISION,
        },
        "metadata_source": {
            "repo": verifier.METADATA_REPO,
            "revision": verifier.METADATA_REVISION,
        },
    }
    (model / "native-source.json").write_text(json.dumps(native_source))
    (model / "tokenizer.json").write_text("{\"synthetic\":true}\n")
    return model, tmp_path / "receipt.json", payloads


def test_success_receipt_is_compatible_with_benchmark_weight_identity(synthetic_model):
    model, output, payloads = synthetic_model
    receipt, shards_verified, mismatches = verifier.verify(model, output)

    assert receipt is not None
    assert shards_verified == len(payloads) == 4
    assert mismatches == 0
    verifier.write_private_new(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    identity = benchmark_hybrid.weight_identity(model, output)
    assert set(identity["files"]) == set(payloads)
    for name, payload in payloads.items():
        assert identity["files"][name]["prior_verified_sha256"] == hashlib.sha256(payload).hexdigest()


def test_corrupted_shard_is_rejected_without_receipt(synthetic_model):
    model, output, payloads = synthetic_model
    name = next(iter(payloads))
    shard = model / name
    shard.write_bytes(b"X" + shard.read_bytes()[1:])

    receipt, shards_verified, mismatches = verifier.verify(model, output)

    assert receipt is None
    assert shards_verified == 4
    assert mismatches > 0
    assert not output.exists()


def test_wrong_metadata_source_pin_is_rejected_without_receipt(synthetic_model):
    model, output, _ = synthetic_model
    native_path = model / "native-source.json"
    native_source = json.loads(native_path.read_text())
    native_source["metadata_source"]["revision"] = "0" * 40
    native_path.write_text(json.dumps(native_source))

    receipt, shards_verified, mismatches = verifier.verify(model, output)

    assert receipt is None
    assert shards_verified == 4
    assert mismatches > 0
    assert not output.exists()


def test_symlinked_shard_is_rejected_without_receipt(synthetic_model, tmp_path):
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform has no O_NOFOLLOW support")
    model, output, payloads = synthetic_model
    name = next(iter(payloads))
    shard = model / name
    target = tmp_path / "external-shard"
    target.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(target)

    receipt, shards_verified, mismatches = verifier.verify(model, output)

    assert receipt is None
    assert shards_verified == 3
    assert mismatches > 0
    assert not output.exists()
