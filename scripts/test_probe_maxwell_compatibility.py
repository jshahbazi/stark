"""Bounded transport and corrupt-header controls for the metadata prototype."""

import json
import struct

import probe_maxwell_compatibility as probe
import pytest


class Response:
    status = 206

    def __init__(self):
        self.headers = {"Content-Range": "bytes 0-7/1000000"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        assert size == 9
        return b"12345678"


def test_range_refusal_never_reads_tensor_payload(monkeypatch):
    class IgnoredRange(Response):
        status = 200

        def read(self, size):
            pytest.fail("must refuse before reading unbounded response")

    monkeypatch.setattr(probe, "urlopen", lambda *a, **kw: IgnoredRange())
    with pytest.raises(ValueError, match="exact header byte range"):
        probe.fetch("model-00001.safetensors", (0, 7))


def test_wrong_cached_range_is_refused(monkeypatch):
    monkeypatch.setattr(probe, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="exact header byte range"):
        probe.fetch("model-00001.safetensors", (8, 15))


def test_exact_range_is_bounded(monkeypatch):
    monkeypatch.setattr(probe, "urlopen", lambda *a, **kw: Response())
    assert probe.fetch("model-00001.safetensors", (0, 7)) == b"12345678"


@pytest.mark.parametrize("body", [b"short", b"123456789"])
def test_short_or_long_range_body(monkeypatch, body):
    response = Response()
    response.read = lambda size: body
    monkeypatch.setattr(probe, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="size mismatch"):
        probe.fetch("model-00001.safetensors", (0, 7))


def test_impossible_total_refused(monkeypatch):
    response = Response()
    response.headers = {"Content-Range": "bytes 0-7/7"}
    monkeypatch.setattr(probe, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="exact header byte range"):
        probe.fetch("model-00001.safetensors", (0, 7))


def test_memory_units_and_unbounded_geometry_refusal():
    config = {
        "num_hidden_layers": 45,
        "num_key_value_heads": 4,
        "attention_head_size": 128,
        "block_sparse_gqa_indexer_head_dim": 128,
    }
    # 45 layers * (K+V: 1024 elements + index K: 128 elements) * BF16 2 bytes.
    assert (
        probe.memory_payload(config, 0)["bf16_backbone_cache_bytes_per_token"] == 103680
    )
    with pytest.raises(ValueError, match="pinned Maxwell geometry"):
        probe.geometry_checks({"num_hidden_layers": 2**63}, {})


@pytest.mark.parametrize("length", [0, 1, probe.MAX_METADATA + 1, 2**64 - 1])
def test_untrusted_header_size_limit(length):
    with pytest.raises(ValueError):
        probe.header_length(struct.pack("<Q", length))


def test_inventory_rejects_missing_and_overlapping_tensors():
    tensor = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
    with pytest.raises(ValueError, match="incomplete"):
        probe.tensor_inventory({"s": {"a": tensor}}, {"a": "s", "b": "s"})
    with pytest.raises(ValueError, match="overlapping"):
        probe.tensor_inventory({"s": {"a": tensor, "b": tensor}}, {"a": "s", "b": "s"})


def test_index_owner_and_tensor_byte_count():
    tensor = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 8]}
    with pytest.raises(ValueError, match="ownership"):
        probe.tensor_inventory({"s": {"a": tensor}}, {"a": "wrong"})
    with pytest.raises(ValueError, match="byte count"):
        probe.tensor_inventory({"s": {"a": tensor}}, {"a": "s"})


def test_report_controls_are_aggregate_only():
    controls = probe.equation_controls()
    assert controls["hc_normalization_max_abs_differences"][0] > 1
    assert controls["sinkhorn_max_abs_error"] == 0
    assert controls["router_selection_matches"] == controls["cases"]
    assert controls["sparse_max_vs_mean_counterexample"]
    assert "probabilities" not in json.dumps(controls)
