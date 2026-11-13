import struct
import contextlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmark_idle_wake as benchmark
from benchmark_idle_wake import generation_item, scoring_summary, summarize_positions


def test_fast_worker_rejection_survives_process_group_cleanup():
    from benchmark_stock_engines import Pipe

    command = [sys.executable, "-c", 'print(\'{"ready":false,"error":"model_load_failed"}\'); raise SystemExit(1)']
    with pytest.raises(RuntimeError, match="^model_load_failed$"):
        with contextlib.ExitStack() as stack:
            Pipe(command, stack, ".", None)


def reply(value):
    return {"logits": [value]*4, "log_probabilities": [-1.0]*4,
            "candidate_probabilities": [.25]*4, "cache_hit": False,
            "cache": {"native_score_calls": 1}, "input_tokens": 512, "metrics": {}}


def test_score_fingerprint_keeps_float_bits_without_persisting_vectors():
    positive, negative = scoring_summary(reply(0.0)), scoring_summary(reply(-0.0))
    assert struct.pack("<d", 0.0) != struct.pack("<d", -0.0)
    assert positive["output_sha256"] != negative["output_sha256"]
    assert not {"logits", "log_probabilities", "candidate_probabilities"}.intersection(positive)


def test_score_summary_rejects_nonfinite_or_partial_scores():
    with pytest.raises(ValueError):
        scoring_summary(reply(float("nan")))
    raw = reply(1.0)
    raw["logits"].pop()
    with pytest.raises(ValueError):
        scoring_summary(raw)


def test_latency_summary_keeps_mtp_routes_and_positions_separate():
    rows = [{"route": route, "position": position, "wall_seconds": wall,
             "prefill_seconds": wall, "host_screen": {"eligible": True}}
            for route, position, wall in [("off", 0, 1), ("off", 1, .3), ("on", 0, .8), ("on", 1, .2)]]
    assert [r["median_wall_seconds"] for r in summarize_positions(rows)] == [1, .3, .8, .2]


@pytest.mark.parametrize("nested", [True, False])
def test_generation_counter_layouts_share_acknowledgement_controls(nested):
    counts = {"rounds": 3, "proposed": 6, "accepted": 4, "verified_rows": 9}
    mtp = {"enabled": True, "drafts": 2, "sampling": "exact-rejection-v1", "verification": "test-verifier"}
    mtp.update({"stats": counts} if nested else counts)
    raw = {"mtp": mtp, "generated_token_ids": list(range(8)), "sampled_tokens": 8,
           "decode_seconds": 1, "prefill_seconds": .3, "generation_seconds": 1.3,
           "output_tokens_per_second": 7, "finish_reason": "length"}
    item = generation_item(raw, {"mtp": {"verification": "test-verifier"}}, 2)
    assert item["mtp"]["proposed"] == 6
    assert "generated_token_ids" not in item


def test_short_cache_hit_is_a_correctness_control(monkeypatch):
    class Monitor:
        def set_phase(self, phase):
            self.phase = phase

    def fail(*args):
        raise AssertionError("cache hit must not require a sampled host interval")

    monkeypatch.setattr(benchmark, "screen_attempt", fail)
    monitor = Monitor()
    assert benchmark.request_screen(monitor, "hit", {}, {}, cache_hit=True) == {
        "eligible": False, "reason": "cache_correctness_only"}
    assert monitor.phase == "between_requests"


def test_unsampled_fresh_request_remains_unqualified(monkeypatch):
    def missing(*args):
        raise ValueError("host_observations_missing")

    monkeypatch.setattr(benchmark, "screen_attempt", missing)
    assert benchmark.request_screen(None, "fresh", {}, {}, cache_hit=False) == {
        "eligible": False, "reason": "host_observations_missing"}


class FakeMonitor:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def set_phase(self, phase):
        self.phase = phase


class FakeHost:
    def __init__(self, _pid):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def begin_request(self):
        return object()

    def end_request(self, _marker):
        pass

    def request_result(self, _marker):
        return {"available": True}


class FakeProcess:
    class Stdin:
        def close(self):
            pass

    stdin = Stdin()
    pid = 101

    def wait(self, timeout=None):
        assert timeout == 10
        return 0


class FakePipe:
    ready = {"mtp": {"verification": "test-verifier", "supported": True, "max_drafts": 2},
             "prefix_cache": {"enabled": True, "chunk_tokens": 2}}
    process = FakeProcess()

    def __init__(self, *_):
        self.cached = False
        self.native_calls = 0
        self.prefix_calls = {}

    def request(self, payload):
        operation = payload["operation"]
        if operation == "tokenize":
            return {"token_ids": list(range(32))}
        if operation == "reset_cache":
            self.cached = False
            return {"cache_reset": True}
        if operation == "score":
            hit = self.cached
            if not hit:
                self.native_calls += 1
                self.cached = True
            return {"logits": [1.0] * 4, "log_probabilities": [-1.0] * 4,
                    "candidate_probabilities": [.25] * 4, "cache_hit": hit,
                    "vocab_logits_sha256": "d" * 64,
                    "cache": {"native_score_calls": self.native_calls},
                    "input_tokens": 4, "vocab_size": 32000, "metrics": {}}
        if operation == "score-digest":
            hit = self.cached
            if not hit:
                self.native_calls += 1
                self.cached = True
            return {"vocab_logits_sha256": "a" * 64, "cache_hit": hit,
                    "cache": {"native_score_calls": self.native_calls},
                    "input_tokens": 4, "vocab_size": 32000}
        if operation == "generate":
            requested = payload["mtp_drafts"]
            is_prefix = len(payload["input_token_ids"]) > 4
            if is_prefix:
                index = self.prefix_calls.get(requested, 0)
                self.prefix_calls[requested] = index + 1
                reused = 2 if index == 1 else 0
            else:
                reused = 0
            return {"generated_token_ids": [3, 4, 5], "sampled_tokens": 3,
                    "decode_seconds": .2, "prefill_seconds": .1,
                    "generation_seconds": .3, "output_tokens_per_second": 10,
                    "finish_reason": "length", "reused_input_tokens": reused,
                    "mtp": {"enabled": requested > 0, "drafts": requested,
                            "verification": "test-verifier",
                            "stats": {"proposed": 2 if requested else 0, "rounds": 1}}}
        raise AssertionError(f"unexpected synthetic operation: {operation}")


@contextmanager
def fake_clean_environment():
    yield


def run_fake_campaign(monkeypatch, tmp_path, endpoint, *, prefix=False):
    monkeypatch.setattr(benchmark, "artifact_guard", lambda *_: {"receipt_sha256": "b" * 64})
    monkeypatch.setattr(benchmark, "sha256_file", lambda *_: "c" * 64)
    monkeypatch.setattr(benchmark, "clean_environment", fake_clean_environment)
    monkeypatch.setattr(benchmark, "Pipe", FakePipe)
    monkeypatch.setattr(benchmark, "Monitor", FakeMonitor)
    monkeypatch.setattr(benchmark, "HostMetricsSampler", FakeHost)
    monkeypatch.setattr(benchmark, "hardware_identity", lambda: {"model": "test"})
    monkeypatch.setattr(benchmark, "execution_profile", lambda **_: {"profile": "test"})
    monkeypatch.setattr(benchmark, "quiet_admission", lambda _monitor: {"eligible": True})
    monkeypatch.setattr(benchmark, "screen_attempt", lambda *_: {"eligible": True})
    monkeypatch.setattr(benchmark, "paging", lambda: {})
    monkeypatch.setattr(benchmark, "snapshot", lambda _pid: {"available": False})
    monkeypatch.setattr(benchmark, "delta", lambda *_: {"available": False})
    monkeypatch.setattr(benchmark, "exact_prefix", lambda ids, count: ids[:count])
    monkeypatch.setattr(benchmark, "ids_sha256", lambda ids: str(len(ids)).zfill(64))
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("worker", "receipt", "corpus"):
        (tmp_path / name).write_text("synthetic")
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        output=output, rest_seconds=1, input_tokens=4, tokens=2, endpoint=endpoint,
        model=model, artifact_receipt=tmp_path / "receipt", worker=tmp_path / "worker",
        corpus=tmp_path / "corpus", screened=True, drafts=2, clock_sampler=None,
        prefix_control=prefix,
    )
    return benchmark.run(args)


def test_run_generation_emits_two_routes_and_four_positioned_requests(monkeypatch, tmp_path):
    report = run_fake_campaign(monkeypatch, tmp_path, "generation")

    assert report["correctness_passed"] is True
    assert {(row["route"], row["position"]) for row in report["requests"]} == {
        (route, position) for route in ("off", "on") for position in (0, 1)}
    assert len(report["warmups"]) == 2
    assert report["cache_controls"] == []
    assert report["prefix_controls"] == []
    assert report["eof_exit_code"] == 0


@pytest.mark.parametrize("endpoint", ["score", "score-digest"])
def test_run_scoring_emits_idle_cache_control_without_qualifying_it(monkeypatch, tmp_path, endpoint):
    report = run_fake_campaign(monkeypatch, tmp_path, endpoint)

    assert report["correctness_passed"] is True
    assert len(report["requests"]) == 2
    assert len(report["idle_intervals"]) == 2
    assert len(report["cache_controls"]) == 1
    assert report["cache_controls"][0]["cache_hit"] is True
    assert report["cache_controls"][0]["host_screen"] == {
        "eligible": False, "reason": "cache_correctness_only"}
    assert report["timing_qualified"] is True
    assert report["eof_exit_code"] == 0


def test_run_generation_prefix_control_checks_capture_reuse_and_fresh(monkeypatch, tmp_path):
    report = run_fake_campaign(monkeypatch, tmp_path, "generation", prefix=True)

    assert len(report["prefix_controls"]) == 2
    assert all(control["passed"] for control in report["prefix_controls"])
    assert all([row["reused_input_tokens"] for row in control["requests"]] == [0, 2, 0]
               for control in report["prefix_controls"])
