"""Model-free counterexamples for Phase 4 private-pipe qualification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import validate_edge0_metal as qualification


def test_comparison_checks_both_independent_numerical_gates():
    reference = np.asarray([1.0, 2.0, 3.0])
    assert qualification.comparison(reference, reference)["passed"]
    assert not qualification.comparison(reference, reference + 0.001)["passed"]
    large = reference * 1000
    report = qualification.comparison(large, large + 0.003)
    assert report["relative_l2"] < qualification.REL_L2
    assert not report["passed"]


def test_margin_guard_allows_only_close_top1_changes_below_both_error_limits():
    reference = np.asarray([100.0, 100.001, 99.0])
    report = qualification.comparison(reference, np.asarray([100.0015, 100.0, 99.0]))
    assert report["passed"] and not report["top1_agrees"]
    assert not report["top1_margin_gate_applies"]
    report = qualification.comparison(
        np.asarray([100.0, 100.01]), np.asarray([100.0, 100.01])
    )
    assert report["top1_margin_gate_applies"] and report["top1_agrees"]


@pytest.mark.parametrize(
    "values", [[1.0, float("nan")], [1.0, float("inf")], [True, 2.0], ["1", 2.0], [1]]
)
def test_nonfinite_malformed_rows_are_rejected(values):
    with pytest.raises(qualification.Failure):
        qualification.row(values, 2)


def test_comparison_rejects_nonfinite_and_mismatched_rows():
    for candidate in (np.asarray([1.0]), np.asarray([1.0, float("nan")])):
        with pytest.raises(qualification.Failure):
            qualification.comparison(np.asarray([1.0, 2.0]), candidate)


def test_summary_never_contains_vectors_or_indices():
    report = qualification.comparison(np.asarray([1.2, 2.3]), np.asarray([1.2, 2.3]))
    assert set(report) == {
        "max_abs",
        "relative_l2",
        "top1_agrees",
        "top1_margin_gate_applies",
        "passed",
    }
    assert "1.2" not in json.dumps(report)
    assert qualification.row_hash(np.asarray([1.0])) == qualification.row_hash(
        np.asarray([1.0], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "values",
    [
        ["SECRET=value"],
        ["STARK_EDGE0_REFERENCE=0"],
        ["STARK_EDGE0_METAL=private"],
        ["STARK_EDGE0_METAL=1", "STARK_EDGE0_METAL=1"],
    ],
)
def test_selectors_cannot_capture_unrelated_values(values):
    with pytest.raises(qualification.Failure):
        qualification.selectors(values)


def test_candidate_selector_is_explicit():
    assert qualification.selectors(["STARK_EDGE0_METAL=1"]) == {
        "STARK_EDGE0_REFERENCE": "1",
        "STARK_EDGE0_METAL": "1",
    }


def test_trickled_bytes_do_not_renew_monotonic_deadline(monkeypatch):
    probe = qualification.Probe.__new__(qualification.Probe)
    probe.buffer = bytearray()
    probe.process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 10))
    deadlines = []

    def sample(deadline):
        deadlines.append(deadline)
        if len(deadlines) == 4:
            raise qualification.Failure("probe_timeout")

    probe.sample = sample
    monkeypatch.setattr(qualification.time, "monotonic", lambda: 12)
    monkeypatch.setattr(qualification.select, "select", lambda *args: ([10], [], []))
    monkeypatch.setattr(qualification.os, "read", lambda *args: b" ")
    with pytest.raises(qualification.Failure, match="^probe_timeout$"):
        probe.read()
    assert deadlines == [12 + qualification.TIMEOUT] * 4


def test_reply_size_is_bounded_before_json_parser(monkeypatch):
    probe = qualification.Probe.__new__(qualification.Probe)
    probe.buffer = bytearray(b" " * (16 * 1024 * 1024))
    probe.process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 10))
    probe.sample = lambda deadline: None
    monkeypatch.setattr(qualification.select, "select", lambda *args: ([10], [], []))
    monkeypatch.setattr(qualification.os, "read", lambda *args: b"x")
    with pytest.raises(qualification.Failure, match="^reply_limit$"):
        probe.read()


def test_fixed_work_counts_prefill_once_and_all_forced_steps(monkeypatch):
    calls = []

    def request(operation, **values):
        calls.append((operation, values))
        if operation == "prefill":
            return {
                "prefix_tokens": len(values["ids"]),
                "ingest_seconds": 2.0,
                "readout_seconds": 1.0,
                "logits": [1.0, 2.0],
            }
        return {"seconds": 0.5, "logits": [1.0, 2.0]}

    monkeypatch.setattr(qualification, "row", lambda values, vocab: np.asarray(values))
    monkeypatch.setattr(qualification, "DECODE_STEPS", 2)
    rows, timing = qualification.forward(
        SimpleNamespace(request=request), [11, 12, 13, 14, 15], 3, reset=False
    )
    assert len(rows) == 3
    assert calls == [
        ("prefill", {"ids": [11, 12, 13], "reset": False}),
        ("step", {"id": 14}),
        ("step", {"id": 15}),
    ]
    assert timing["ingest_tokens"] == 2
    assert timing["prefix_tokens"] == 3
    assert timing["decode_steps"] == 2
    assert timing["decode_seconds"] == 1
    assert timing["prefill_seconds"] == 3


def metadata(candidate=True, count=0):
    return {
        "backend": "edge0-metal-graph-v1" if candidate else "edge0-f32-reference",
        "activation_precision": "f32",
        "accumulation_precision": "f32",
        "recurrent_state_precision": "f32",
        "attention_cache_precision": "f32",
        "prefill_max_batch": 8 if candidate else 1,
        "prefix_reuse": False,
        "source_binding": "0721afc5",
        "weights": {
            "resident_enabled": candidate,
            "registered_bank_bytes": 100 if candidate else 0,
            "graph_scratch_high_water_bytes": 1024 if candidate else 0,
            "counters": {
                "batched_matrix_calls": count,
                "batched_q8_calls": count,
                "batched_expert_calls": count,
            },
        },
    }


def test_loaded_backend_and_actual_batched_dispatch_are_required():
    baseline = metadata(candidate=False)
    assert qualification.execution_metadata(baseline, "baseline") == baseline
    ready = metadata()
    final = metadata(count=3)
    assert qualification.execution_metadata(final, "candidate", ready) == final
    with pytest.raises(qualification.Failure, match="^execution_backend$"):
        qualification.execution_metadata(baseline, "candidate")
    with pytest.raises(
        qualification.Failure, match="^candidate_batched_dispatch_missing$"
    ):
        qualification.execution_metadata(ready, "candidate", ready)


@pytest.mark.parametrize(
    "counter", ["batched_matrix_calls", "batched_q8_calls", "batched_expert_calls"]
)
def test_each_required_batched_operator_must_run(counter):
    final = metadata(count=3)
    final["weights"]["counters"][counter] = 0
    with pytest.raises(
        qualification.Failure, match="^candidate_batched_dispatch_missing$"
    ):
        qualification.execution_metadata(final, "candidate", metadata())


@pytest.mark.parametrize(
    "key,value",
    [
        ("activation_precision", "f16"),
        ("accumulation_precision", "f16"),
        ("recurrent_state_precision", "f16"),
        ("attention_cache_precision", "f16"),
        ("prefill_max_batch", 1),
        ("prefill_max_batch", True),
        ("prefix_reuse", True),
    ],
)
def test_execution_metadata_rejects_wrong_precision_or_capability(key, value):
    final = metadata(count=3)
    final[key] = value
    with pytest.raises(qualification.Failure):
        qualification.execution_metadata(final, "candidate", metadata())


def test_counter_booleans_cannot_pass_as_dispatch_counts():
    final = metadata(count=3)
    final["weights"]["counters"]["batched_q8_calls"] = True
    with pytest.raises(qualification.Failure, match="^execution_counter_type$"):
        qualification.execution_metadata(final, "candidate", metadata())


@pytest.mark.parametrize(
    "key,value",
    [
        ("resident_enabled", 1),
        ("registered_bank_bytes", -1),
        ("registered_bank_bytes", True),
    ],
)
def test_weight_metadata_uses_typed_byte_counts_and_residency(key, value):
    final = metadata(count=3)
    final["weights"][key] = value
    with pytest.raises(qualification.Failure):
        qualification.execution_metadata(final, "candidate", metadata())


@pytest.mark.parametrize("scratch", [0, None, True, -1, 64 * 1024**2 + 1])
def test_actual_graph_scratch_must_be_positive_and_inside_segment_cap(scratch):
    final = metadata(count=3)
    if scratch is None:
        final["weights"].pop("graph_scratch_high_water_bytes")
    else:
        final["weights"]["graph_scratch_high_water_bytes"] = scratch
    with pytest.raises(qualification.Failure):
        qualification.execution_metadata(final, "candidate", metadata())


def test_baseline_immutable_probe_may_omit_graph_scratch_metadata():
    baseline = metadata(candidate=False)
    baseline["weights"].pop("graph_scratch_high_water_bytes")
    assert qualification.execution_metadata(baseline, "baseline") == baseline
