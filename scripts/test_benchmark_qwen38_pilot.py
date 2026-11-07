from argparse import Namespace
import json
import os

import benchmark_qwen38_mtp as mtp
import benchmark_qwen38_pilot as pilot
from test_benchmark_qwen38_mtp import FakeWorker, make_args
import pytest


def args_for(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    args.input_lengths = [5, 8]
    args.draft_depths = [1, 2]
    args.candidate_worker = None
    args.cells = tmp_path / "cells"
    args.screened = False
    monkeypatch.setattr(pilot, "hardware_identity", lambda: {"fixture": True})
    return args


def fake_run(args):
    return mtp.run(args, worker_factory=FakeWorker)


def test_grid_uses_real_pair_validation_without_exporting_content(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    report = pilot.run(args, cell_runner=fake_run)
    assert report["passed"] and not report["timing_eligible"]
    assert len(report["cells"]) == 4
    assert {(cell["input_tokens"], cell["drafts"]) for cell in report["cells"]} == {
        (count, depth) for count in [5, 8] for depth in [1, 2]
    }
    assert json.loads(args.output.read_text()) == report
    assert args.output.stat().st_mode & 0o777 == 0o600
    assert "safe corpus material" not in json.dumps(report)
    assert "generated_token_ids" not in json.dumps(report)


def test_failed_cell_retains_static_failure_and_stops(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)

    def failure(_args):
        raise ValueError("private model content")

    report = pilot.run(args, cell_runner=failure)
    assert not report["passed"] and not report["timing_eligible"]
    assert len(report["cells"]) == 1
    assert "private model content" not in args.output.read_text()


def test_candidate_sessions_compare_same_routes_and_use_abba(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    args.input_lengths, args.draft_depths = [5], [2]
    args.candidate_worker = tmp_path / "candidate"
    args.candidate_worker.write_bytes(b"candidate")
    report = pilot.run(args, cell_runner=fake_run)
    assert report["session_order"] == ["baseline", "candidate", "candidate", "baseline"]
    assert report["passed"]
    assert report["worker_comparisons"] == [{
        "input_tokens": 5, "drafts": 2, "workers": ["baseline", "candidate"],
        "same_route_hashes_equal": True,
    }]
    # Exact rejection consumes randomness differently with drafting enabled.
    assert all(not cell["report"]["cross_route_hashes_equal"] for cell in report["cells"])


def test_candidate_numerical_regression_fails_grid(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    args.input_lengths, args.draft_depths = [5], [2]
    args.candidate_worker = tmp_path / "candidate"
    args.candidate_worker.write_bytes(b"candidate")

    def changed(args):
        report = fake_run(args)
        if args.worker.name == "candidate":
            for request in report["requests"]:
                request["output_sha256"] = "different"
        return report

    report = pilot.run(args, cell_runner=changed)
    assert not report["passed"]
    assert not report["worker_comparisons"][0]["same_route_hashes_equal"]


def test_host_screen_failure_prevents_timing_qualification(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    args.input_lengths, args.draft_depths = [5], [2]
    args.screened = True

    class Monitor:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def set_phase(self, _phase): pass

    def screened(args):
        return mtp.run(args, worker_factory=FakeWorker, monitor_factory=Monitor,
                       paging_reader=lambda: {}, admission_fn=lambda _: {},
                       screen_fn=lambda *_: {"eligible": False})

    report = pilot.run(args, cell_runner=screened)
    assert report["passed"] and not report["timing_eligible"]
    assert len(report["cells"][0]["report"]["pairs"]) == 4


def test_environment_removed_and_restored_on_failure(monkeypatch):
    monkeypatch.setenv("STARK_QWEN38_PROFILE_OUTPUT", "private-path")
    with pytest.raises(RuntimeError):
        with pilot.clean_environment():
            assert "STARK_QWEN38_PROFILE_OUTPUT" not in os.environ
            raise RuntimeError()
    assert os.environ["STARK_QWEN38_PROFILE_OUTPUT"] == "private-path"


def test_entire_grid_validated_before_running(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    args.input_lengths = [5, 16384]
    with pytest.raises(ValueError, match="context_limit_exceeded"):
        pilot.run(args, cell_runner=lambda _: pytest.fail("must not run"))
    assert not args.cells.exists()


@pytest.mark.parametrize("message,expected", [
    ("quiet_admission_timeout", "quiet_admission_timeout"),
    ("private model content", "paired_benchmark_failed"),
])
def test_screen_failure_retains_sanitized_partial_receipt(tmp_path, message, expected):
    args = make_args(tmp_path)
    args.screened = True

    class Monitor:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def set_phase(self, _phase): pass

    def fail(_monitor):
        raise TimeoutError(message)

    with pytest.raises(TimeoutError):
        mtp.run(args, worker_factory=FakeWorker, monitor_factory=Monitor,
                paging_reader=lambda: {}, admission_fn=fail,
                screen_fn=lambda *_: {"eligible": True})
    partial = json.loads(args.output.read_text())
    assert not partial["passed"] and not partial["timing_eligible"]
    assert partial["failure_code"] == expected
    assert "private model content" not in args.output.read_text()


def test_weight_guard_rejects_changed_payload_and_missing_inventory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    payload = model / "fixture.safetensors"
    payload.write_bytes(b"initial")
    stat = payload.stat()
    receipt = tmp_path / "receipt.json"
    value = {"schema": "stark-flashnext-pilot-weights-v1", "passed": True,
             "files": {payload.name: {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                      "inode": stat.st_ino}}}
    receipt.write_text(json.dumps(value))
    assert pilot.weight_guard(model, receipt) == mtp.sha256_file(receipt)
    payload.write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="weight_identity_changed"):
        pilot.weight_guard(model, receipt)
    value["files"] = {}
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="weight_receipt_invalid"):
        pilot.weight_guard(model, receipt)
