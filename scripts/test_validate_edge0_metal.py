"""Model-free counterexamples for Phase 4 private-pipe qualification."""

from __future__ import annotations

import json
import subprocess
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


def metadata(candidate=True, count=0, resident=False):
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
            "resident_enabled": resident,
            "registered_bank_bytes": qualification.COMPLETE_BANK_BYTES
            if candidate
            else 0,
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


@pytest.mark.parametrize(
    "timeouts,mode,expected",
    [
        (0, "eof", ["stdin_close", "wait", "stdout_close", "gpu_close"]),
        (
            1,
            "terminate",
            ["stdin_close", "wait", "terminate", "wait", "stdout_close", "gpu_close"],
        ),
        (
            2,
            "kill",
            [
                "stdin_close",
                "wait",
                "terminate",
                "wait",
                "kill",
                "wait",
                "stdout_close",
                "gpu_close",
            ],
        ),
    ],
)
def test_teardown_tries_eof_before_owned_signal_fallbacks(
    monkeypatch, timeouts, mode, expected
):
    calls = []
    waits = 0

    def wait(*, timeout):
        nonlocal waits
        assert timeout == 5
        calls.append("wait")
        waits += 1
        if waits <= timeouts:
            raise subprocess.TimeoutExpired("owned-probe", timeout)
        return 0 if timeouts == 0 else -15 if timeouts == 1 else -9

    probe = qualification.Probe.__new__(qualification.Probe)
    probe.process = SimpleNamespace(
        stdin=SimpleNamespace(close=lambda: calls.append("stdin_close")),
        stdout=SimpleNamespace(close=lambda: calls.append("stdout_close")),
        wait=wait,
        terminate=lambda: calls.append("terminate"),
        kill=lambda: calls.append("kill"),
    )
    probe.gpu_read = 101
    probe.metrics = {}

    def close_gpu(fd):
        assert fd == 101
        calls.append("gpu_close")

    monkeypatch.setattr(qualification.os, "close", close_gpu)
    monkeypatch.setattr(
        qualification.psutil, "virtual_memory", lambda: SimpleNamespace(available=1234)
    )
    probe.close()
    assert calls == expected
    assert probe.metrics["teardown_mode"] == mode
    assert probe.metrics["available_memory_after_close_bytes"] == 1234
    assert "worker_exit_code" in probe.metrics
    probe.close()
    assert calls == expected


def test_exited_worker_broken_stdin_still_reaps_without_signals(monkeypatch):
    calls = []

    def stdin_close():
        calls.append("stdin_close")
        raise BrokenPipeError

    probe = qualification.Probe.__new__(qualification.Probe)
    probe.process = SimpleNamespace(
        stdin=SimpleNamespace(close=stdin_close),
        stdout=SimpleNamespace(close=lambda: calls.append("stdout_close")),
        wait=lambda **kwargs: 0,
    )
    probe.gpu_read = 101
    probe.metrics = {}
    monkeypatch.setattr(qualification.os, "close", lambda fd: calls.append("gpu_close"))
    monkeypatch.setattr(
        qualification.psutil, "virtual_memory", lambda: SimpleNamespace(available=1234)
    )
    probe.close()
    assert calls == ["stdin_close", "stdout_close", "gpu_close"]
    assert probe.metrics["teardown_mode"] == "eof"


def test_resident_candidate_accepts_only_clean_eof_exit():
    registration = {"candidate_selectors": {"STARK_EDGE0_RESIDENT": "1"}}
    qualification.candidate_teardown_gate(
        {"memory": {"teardown_mode": "eof", "worker_exit_code": 0}}, registration
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"teardown_mode": "terminate", "worker_exit_code": 0},
        {"teardown_mode": "kill", "worker_exit_code": -9},
        {"teardown_mode": "eof", "worker_exit_code": 1},
        {"teardown_mode": "eof", "worker_exit_code": -15},
        {"teardown_mode": "eof", "worker_exit_code": False},
        {"teardown_mode": "eof", "worker_exit_code": "0"},
    ],
)
def test_resident_candidate_rejects_missing_abnormal_or_signalled_teardown(metrics):
    registration = {"candidate_selectors": {"STARK_EDGE0_RESIDENT": "1"}}
    with pytest.raises(
        qualification.Failure, match="^candidate_resident_teardown_gate$"
    ):
        qualification.candidate_teardown_gate({"memory": metrics}, registration)


@pytest.mark.parametrize("selectors", [{}, {"STARK_EDGE0_RESIDENT": "0"}])
def test_virtual_candidate_does_not_acquire_resident_teardown_gate(selectors):
    qualification.candidate_teardown_gate({}, {"candidate_selectors": selectors})


def test_registration_declares_resident_teardown_condition_and_fixed_gates(monkeypatch):
    monkeypatch.setattr(qualification, "bindings", lambda args: {})
    registration = qualification.recipe(
        SimpleNamespace(
            candidate_env=["STARK_EDGE0_METAL=1", "STARK_EDGE0_RESIDENT=1"],
            minimum_available_gib=64,
        )
    )
    assert registration["candidate_selectors"]["STARK_EDGE0_RESIDENT"] == "1"
    assert registration["candidate_teardown_gate"] == {
        "condition": "candidate_selectors.STARK_EDGE0_RESIDENT equals 1",
        "required_teardown_mode": "eof",
        "required_worker_exit_code": 0,
    }
    assert (
        registration["execution_gate"]["candidate_final_registered_bank_bytes"]
        == 43_819_991_040
    )
    assert registration["execution_gate"]["resident_enabled_policy"] == (
        "ready and final must equal requested arm selector: true exactly when STARK_EDGE0_RESIDENT=1, false otherwise"
    )


@pytest.mark.parametrize(
    "requested", [{}, {"STARK_EDGE0_RESIDENT": "0"}, {"STARK_EDGE0_RESIDENT": "1"}]
)
@pytest.mark.parametrize("final", [False, True])
def test_resident_policy_must_match_requested_arm_at_ready_and_final(requested, final):
    resident = requested.get("STARK_EDGE0_RESIDENT") == "1"
    value = metadata(count=3, resident=resident)
    initial = metadata(resident=resident) if final else None
    assert (
        qualification.execution_metadata(
            value, "candidate", initial, requested_selectors=requested
        )
        == value
    )


@pytest.mark.parametrize(
    "requested,resident", [({"STARK_EDGE0_RESIDENT": "1"}, False), ({}, True)]
)
@pytest.mark.parametrize("final", [False, True])
def test_wrong_residency_mode_cannot_pass_either_stage(requested, resident, final):
    value = metadata(count=3, resident=resident)
    with pytest.raises(qualification.Failure, match="^execution_residency_policy$"):
        qualification.execution_metadata(
            value,
            "candidate",
            metadata() if final else None,
            requested_selectors=requested,
        )


@pytest.mark.parametrize("bank_bytes", [0, 43_819_991_039, 43_819_991_041])
def test_final_inventory_must_equal_the_complete_pinned_bank_bytes(bank_bytes):
    value = metadata(count=3)
    value["weights"]["registered_bank_bytes"] = bank_bytes
    with pytest.raises(
        qualification.Failure, match="^execution_complete_bank_inventory$"
    ):
        qualification.execution_metadata(value, "candidate", metadata())


def test_ready_can_report_empty_lazy_bank_registry():
    value = metadata()
    value["weights"]["registered_bank_bytes"] = 0
    assert qualification.execution_metadata(value, "candidate") == value


def test_run_session_rejects_wrong_resident_policy_before_dispatch(monkeypatch):
    events = []

    class WrongPolicyProbe:
        def __init__(self, *args):
            self.metrics = {}

        def read(self):
            return {
                "ready": True,
                "protocol": 1,
                "vocab_size": 248320,
                "max_context": 512,
                "load_seconds": 0.0,
                "request_memory_bytes": 0,
                "execution": metadata(resident=False),
            }

        def request(self, *args, **kwargs):
            pytest.fail("mismatched resident policy must reject before model requests")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(qualification, "Probe", WrongPolicyProbe)
    registration = {
        "candidate_selectors": {"STARK_EDGE0_METAL": "1", "STARK_EDGE0_RESIDENT": "1"},
        "minimum_available_bytes": 0,
    }
    with pytest.raises(qualification.Failure, match="^execution_residency_policy$"):
        qualification.run_session(
            SimpleNamespace(candidate_probe="unused", model="unused"),
            "candidate",
            registration,
            {},
        )
    assert events == ["closed"]
