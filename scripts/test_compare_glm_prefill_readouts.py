from __future__ import annotations

import hashlib
import os
import struct
import sys
from argparse import Namespace
from array import array

import compare_glm_prefill_readouts as comparison
import pytest


def arm(values):
    data = array("f", values)
    raw = b"".join(struct.pack("<f", value) for value in data)
    return ({"input_tokens": 2, "input_sha256": "synthetic", "corpus_tokens": 6,
             "cyclic_corpus": False, "readout_rows": 1, "vocabulary": len(data),
             "dispatch": {}, "prefill_seconds": 0.1}, data, hashlib.sha256(raw).hexdigest())


def test_full_readout_metrics_and_declared_tolerance():
    reference = arm([2.0, 0.0, -1.0])
    exact = comparison.compare(reference, reference)
    assert exact["exact"] and exact["passed"]
    assert exact["relative_l2_error"] == exact["max_absolute_error"] == 0
    close = comparison.compare(reference, arm([2.002, 0.0, -1.0]))
    assert close["all_close"] and close["passed"] and not close["exact"]
    different = comparison.compare(reference, arm([2.004, 0.0, -1.0]))
    assert not different["all_close"] and not different["passed"]
    assert different["greedy_agreement"] == 1
    assert different["relative_l2_error"] > 0


def test_qualification_requires_verified_weight_receipt_before_spawning():
    with pytest.raises(ValueError, match="verified_weight_receipt_required"):
        comparison.run(Namespace(verified_weights=None))


def test_top1_uses_lowest_index_for_ties_and_gates_acceptance():
    result = comparison.compare(arm([1.0, 1.0]), arm([1.0, 1.0001]))
    assert result["all_close"]
    assert result["greedy_agreement"] == 0
    assert not result["passed"]


def test_requested_dispatch_must_be_observed_and_reference_must_not_use_new_paths():
    metadata = {"input_tokens": 65, "dispatch": {
        "native.glm.prefill.wide_tokens": {"bytes": 64},
        "glm.control.prefill.batch.tokens": {"bytes": 64},
        "glm.kda.prefill.register.tokens": {"bytes": 64},
    }}
    requested = dict(zip(comparison.NEW_FLAGS, ["64", "1", "1"], strict=True))
    comparison.routes_observed(metadata, requested)
    metadata["dispatch"].pop("glm.kda.prefill.register.tokens")
    with pytest.raises(ValueError, match="requested_route_not_observed"):
        comparison.routes_observed(metadata, requested)
    with pytest.raises(ValueError, match="t8_route_not_observed"):
        comparison.routes_observed(metadata, {})


@pytest.fixture
def fake_child(tmp_path):
    executable = tmp_path / "fake-readouts"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, struct, sys, time
output = os.fdopen(int(sys.argv[4]), "wb", buffering=0)
mode = os.environ.get("STARK_READOUT_TEST_MODE")
if mode == "hang":
    time.sleep(10)
if mode == "short":
    output.write(b"SPR")
    sys.exit(0)
lengths = [int(value) for value in sys.argv[3].split(",")]
output.write(b"SPRF001\\n" + struct.pack("<I", len(lengths)))
for length in lengths:
    width = 4096
    metadata = json.dumps({
        "input_tokens": length, "input_sha256": "synthetic", "corpus_tokens": 6,
        "cyclic_corpus": False, "readout_rows": 5, "vocabulary": width,
        "dispatch": {"native.glm.prefill.chunk_tokens": {"bytes": length - 1}},
        "prefill_seconds": 0.1,
    }).encode()
    output.write(struct.pack("<I", len(metadata)) + metadata)
    values = [float(index % 7) for index in range(5 * width)]
    if mode == "nonfinite":
        values[-1] = float("nan")
    output.write(struct.pack("<" + "f" * len(values), *values))
output.close()
''')
    executable.chmod(0o700)
    return Namespace(example=executable, model=tmp_path / "model", corpus=tmp_path / "corpus",
                     lengths=[2, 7], timeout=5)


def test_anonymous_pipe_roundtrip_drains_payload_larger_than_pipe_capacity(fake_child):
    rows = comparison.run_arm(fake_child, dict(os.environ), "synthetic")
    assert len(rows) == 2
    assert len(rows[0][1]) == 5 * 4096
    assert comparison.compare(rows[0], rows[0])["passed"]


@pytest.mark.parametrize(("mode", "expected"), [
    ("short", "transport_truncated"),
    ("nonfinite", "transport_nonfinite_readout"),
    ("hang", "transport_timeout"),
    ("bound", "transport_payload_bound_exceeded"),
])
def test_transport_failure_stops_and_reaps_child(fake_child, monkeypatch, mode, expected):
    processes = []
    original = comparison.subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(comparison.subprocess, "Popen", spawn)
    if mode == "hang":
        fake_child.timeout = 0.05
    if mode == "bound":
        monkeypatch.setattr(comparison, "MAX_ARM_BYTES", 32)
    with pytest.raises(ValueError, match=expected):
        comparison.run_arm(fake_child, dict(os.environ, STARK_READOUT_TEST_MODE=mode), "synthetic")
    assert len(processes) == 1 and processes[0].poll() is not None
