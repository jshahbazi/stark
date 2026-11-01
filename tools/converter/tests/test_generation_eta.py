"""Content-free timing estimates and native progress validation."""

import math

import pytest

from stark_convert.generation_eta import GenerationEstimates, progress_frame


def frame(sampled=0, elapsed=0.0, prefill=None, budget=100, inputs=20):
    return {
        "event": "progress",
        "input_tokens": inputs,
        "sampled_tokens": sampled,
        "output_token_budget": budget,
        "elapsed_seconds": elapsed,
        "prefill_seconds": prefill,
    }


def measurement(**changes):
    return {
        "input_tokens": 20,
        "sampled_tokens": 21,
        "output_tokens": 20,
        "prefill_seconds": 0.2,
        "decode_seconds": 0.4,
        "finish_reason": "eos",
        **changes,
    }


def test_cold_start_waits_for_measurements_and_uses_live_speed():
    estimator = GenerationEstimates()
    start = estimator.estimate("model", frame())
    assert not start["calibrated"]
    assert start["ttft_seconds"] is None and start["ttft_lower_seconds"] is None
    assert start["remaining_upper_seconds"] is None
    first = estimator.estimate("model", frame(1, 0.2, 0.2))
    assert first["ttft_seconds"] == 0.2 and first["remaining_upper_seconds"] is None
    live = estimator.estimate("model", frame(6, 0.3, 0.2))
    assert live["remaining_lower_seconds"] == 0
    assert live["remaining_upper_seconds"] == pytest.approx(94 * 0.02 * 1.35)
    assert not live["calibrated"]


def test_history_predicts_prefill_and_completion_then_adapts_to_current_request():
    estimator = GenerationEstimates()
    estimator.record("model", measurement())
    start = estimator.estimate("model", frame())
    assert start["calibrated"]
    assert start["ttft_lower_seconds"] <= 0.2 <= start["ttft_upper_seconds"]
    assert 0 < start["remaining_lower_seconds"] < start["remaining_upper_seconds"]
    quick = estimator.estimate("model", frame(10, 0.38, 0.2))
    slow = estimator.estimate("model", frame(10, 1.1, 0.2))
    assert slow["remaining_upper_seconds"] > quick["remaining_upper_seconds"]
    assert slow["ttft_seconds"] == 0.2
    overrun = estimator.estimate("model", frame(50, 1.2, 0.2))
    assert overrun["remaining_upper_seconds"] > overrun["remaining_lower_seconds"] > 0
    finished = estimator.estimate("model", frame(100, 2.2, 0.2))
    assert finished["remaining_lower_seconds"] == finished["remaining_upper_seconds"] == 0


def test_prefill_overrun_does_not_leave_the_estimated_upper_time_in_the_past():
    estimator = GenerationEstimates()
    estimator.record("model", measurement())
    waiting = estimator.estimate("model", frame(elapsed=2))
    assert waiting["ttft_upper_seconds"] > 2
    assert waiting["remaining_upper_seconds"] > 0


def test_capped_output_history_uses_the_current_budget_and_preserves_eos_sample_count():
    estimator = GenerationEstimates()
    estimator.record("model", measurement(sampled_tokens=5, output_tokens=4, decode_seconds=0.08))
    eos = estimator.history["model"][0]
    assert eos.sampled_tokens == 5 and eos.seconds_per_token == 0.02
    estimator.record("capped", measurement(finish_reason="length"))
    small = estimator.estimate("capped", frame(1, 0.2, 0.2, budget=20))
    large = estimator.estimate("capped", frame(1, 0.2, 0.2, budget=200))
    assert large["remaining_upper_seconds"] > small["remaining_upper_seconds"]


def test_profiles_are_isolated_bounded_and_require_similar_prompt_sizes():
    estimator = GenerationEstimates(profiles=2, samples=2)
    for _ in range(4):
        estimator.record(("model", "greedy", 1), measurement())
    assert len(estimator.history[("model", "greedy", 1)]) == 2
    assert not estimator.estimate(("model", "sampled", 1), frame())["calibrated"]
    assert not estimator.estimate(("model", "greedy", 2), frame())["calibrated"]
    assert not estimator.estimate(("model", "greedy", 1), frame(inputs=1000))["calibrated"]
    estimator.record("other", measurement())
    estimator.record("new load", measurement())
    assert len(estimator.history) == 2
    assert ("model", "greedy", 1) not in estimator.history


@pytest.mark.parametrize(
    "changes",
    [
        {"error": "native_generation_failed"},
        {"prefill_seconds": -1},
        {"prefill_seconds": math.nan},
        {"sampled_tokens": True},
        {"sampled_tokens": 0},
        {"input_tokens": -1},
    ],
)
def test_failed_or_invalid_measurements_never_train(changes):
    estimator = GenerationEstimates()
    estimator.record("model", measurement(**changes))
    assert not estimator.history


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "private marker"},
        {"sampled_tokens": True},
        {"sampled_tokens": 101},
        {"elapsed_seconds": math.nan},
        {"elapsed_seconds": -1},
        {"output_token_budget": 0},
        {"prefill_seconds": 0},
        {"input_tokens": 1.5},
    ],
)
def test_invalid_progress_is_rejected(changes):
    assert progress_frame({**frame(), **changes}) is None


def test_progress_enforces_monotonic_counts_time_and_stable_request_identity():
    start = progress_frame(frame())
    first = progress_frame(frame(1, 0.2, 0.2), start)
    assert first is not None
    assert progress_frame(frame(2, 0.3, 0.2), first) is not None
    for bad in (frame(), frame(2, 0.1, 0.2), frame(2, 0.3, 0.21), frame(2, 0.3, 0.2, inputs=21)):
        assert progress_frame(bad, first) is None
