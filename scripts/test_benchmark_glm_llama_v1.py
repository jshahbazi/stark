from __future__ import annotations

import json
from argparse import Namespace

import benchmark_glm_llama_v1 as benchmark
import pytest


def valid_reply(**overrides):
    reply = {
        "generated_token_ids": [101, 102],
        "sampled_tokens": 2,
        "finish_reason": "length",
        "stop_token_id": None,
        "input_tokens": 4,
        "reused_input_tokens": 0,
        "prefill_seconds": 0.1,
        "decode_seconds": 0.2,
        "generation_seconds": 0.3,
    }
    reply.update(overrides)
    return reply


def timed_request(*, warmup=False, output_sha="same", wall=1.0, useful=1.0):
    return {
        "input_tokens": 4,
        "requested_output_tokens": 8,
        "warmup": warmup,
        "input_sha256": "input",
        "sampled_tokens": 2,
        "output_sha256": output_sha,
        "wall_seconds": wall,
        "useful_decode_tokens_per_second": useful,
    }


def test_validate_reply_requires_reuse_accounting():
    reply = valid_reply()
    del reply["reused_input_tokens"]

    with pytest.raises(benchmark.ExperimentFailure, match="missing_reuse_accounting"):
        benchmark.validate_reply(reply, {"eos_token_ids": [99]}, 4, 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"finish_reason": "eos"},
        {"finish_reason": "length", "stop_token_id": 99, "sampled_tokens": 3},
        {"finish_reason": "other"},
    ],
)
def test_validate_reply_rejects_inconsistent_eos_and_finish(overrides):
    with pytest.raises(benchmark.ExperimentFailure, match="invalid_finish_reason"):
        benchmark.validate_reply(valid_reply(**overrides), {"eos_token_ids": [99]}, 4, 8)


@pytest.mark.parametrize(
    "reply, error",
    [
        (valid_reply(sampled_tokens=3), "invalid_output_count"),
        (valid_reply(input_tokens=5), "input_or_reuse_mismatch"),
        (valid_reply(reused_input_tokens=1), "input_or_reuse_mismatch"),
        (valid_reply(prefill_seconds=float("nan")), "invalid_timing"),
        (valid_reply(decode_seconds=-0.1), "invalid_timing"),
        (valid_reply(generation_seconds=True), "invalid_timing"),
        (valid_reply(decode_seconds=0.0), "empty_decode_interval"),
    ],
)
def test_validate_reply_rejects_bad_counts_and_timings(reply, error):
    with pytest.raises(benchmark.ExperimentFailure, match=error):
        benchmark.validate_reply(reply, {"eos_token_ids": [99]}, 4, 8)


def test_validate_reply_allows_zero_useful_decode_tokens():
    result = benchmark.validate_reply(
        valid_reply(
            generated_token_ids=[101],
            sampled_tokens=2,
            finish_reason="eos",
            stop_token_id=99,
        ),
        {"eos_token_ids": [99]},
        4,
        8,
    )

    assert result["useful_decode_tokens_per_second"] == 0


def test_compare_sessions_rejects_output_mismatch_in_abba_order():
    sessions = [
        {"requests": [timed_request(output_sha="before")]},
        {"requests": [timed_request(output_sha="after")]},
        {"requests": [timed_request(output_sha="after")]},
        {"requests": [timed_request(output_sha="before")]},
    ]

    with pytest.raises(benchmark.ExperimentFailure, match="output_parity_failed"):
        benchmark.compare_sessions(sessions, [(4, 8)])


def test_comparison_reports_negative_speed_without_promoting_it_to_pass():
    sessions = [
        {"requests": [timed_request(wall=1.0)]},
        {"requests": [timed_request(wall=1.2)]},
        {"requests": [timed_request(wall=1.2)]},
        {"requests": [timed_request(wall=1.0)]},
    ]

    summary = benchmark.compare_sessions(sessions, [(4, 8)])[0]

    assert summary["median_paired_wall_speedup"] < 0
    assert summary["wall_pair_wins"] == 0
    assert "passed" not in summary


def test_zero_useful_decode_has_no_decode_gain_claim():
    sessions = [
        {"requests": [timed_request(useful=0.0)]},
        {"requests": [timed_request(useful=0.0)]},
        {"requests": [timed_request(useful=0.0)]},
        {"requests": [timed_request(useful=0.0)]},
    ]

    summary = benchmark.compare_sessions(sessions, [(4, 8)])[0]

    assert summary["median_paired_decode_gain"] is None
    assert summary["decode_hurdle"] is None
    assert summary["decode_pair_wins"] == 0


def test_session_retains_started_request_when_worker_raises(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("synthetic corpus")
    output = tmp_path / "run"
    output.mkdir()

    class RaisingWorker:
        def __init__(self, *_args, gpu_memory_callback=None):
            self.process = object()

        def read(self):
            return {"ready": True, "mtp": {"supported": False}}

        def request(self, request):
            if request["operation"] == "tokenize":
                return {"token_ids": list(range(32))}
            raise RuntimeError("synthetic inference failure")

        def close(self):
            pass

    monkeypatch.setattr(benchmark, "Worker", RaisingWorker)
    args = Namespace(
        worker=tmp_path / "worker",
        model=tmp_path / "model",
        timeout=1,
        corpus=corpus,
        output=output,
    )

    result = benchmark.session(args, "defaults", {}, 0, [(4, 8)], repeats=0)

    assert result["failed_stage"] == "generation"
    assert result["passed"] is False
    assert len(result["requests"]) == 1
    request = result["requests"][0]
    assert request["status"] == "started"
    assert request["wall_seconds"] >= 0
    persisted = json.loads((output / "session-0.json").read_text())
    assert persisted["requests"][0]["status"] == "started"
