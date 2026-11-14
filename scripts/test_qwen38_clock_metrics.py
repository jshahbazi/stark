import pytest

from qwen38_clock_metrics import summarize


def sample(start, end, mhz, active):
    return {"start": start, "end": end, "duration": end-start,
            "gpu_freq_mhz": mhz, "gpu_active_ratio": active,
            "gpu_temperature_c": 60, "boundary_uncertainty": .0001}


def test_request_join_excludes_boundary_intervals_and_weights_active_residency():
    rows = [sample(9, 10, 9000, 1), sample(10, 11, 1000, 1),
            sample(11, 12, 2000, .5), sample(12, 13, 9000, 1)]
    result = summarize(rows, 9.5, 12.5)
    assert result["fully_contained_intervals"] == 2
    assert result["covered_seconds"] == 2
    assert result["mean_active_gpu_mhz"] == pytest.approx(4000/3)
    assert result["gpu_active_ratio"] == .75


def test_off_state_is_retained_without_an_active_frequency():
    result = summarize([sample(1, 2, 0, 0)], 1, 2)
    assert result["fully_contained_intervals"] == 1
    assert result["mean_active_gpu_mhz"] is None
    assert result["gpu_active_ratio"] == 0


def test_no_covered_interval_does_not_claim_a_measurement():
    result = summarize([sample(1, 2, 1600, 1)], 1.1, 1.9)
    assert result["fully_contained_intervals"] == 0
    assert result["mean_active_gpu_mhz"] is None
