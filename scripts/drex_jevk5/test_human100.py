"""Model-free independent count/calibration checks for the Human100 instrument."""

import pytest

from human100_audit import analyze


def synthetic_records():
    gold = {str(i): i < 67 for i in range(100)}
    rows = [
        {
            "id": cid,
            "arm": arm,
            "repeat": repeat,
            "p": [0.9, 0.05, 0.05] if accepted else [0.05, 0.9, 0.05],
            "seconds": 0.1,
        }
        for cid, accepted in gold.items()
        for arm in ("J0", "E1")
        for repeat in (1, 2)
    ]
    return rows, gold


def test_perfect_binary_judgments_have_expected_calibration():
    rows, gold = synthetic_records()
    result = analyze(rows, gold)
    for arm in ("J0", "E1"):
        metrics = result["contexts"][arm]
        assert metrics["matches"] == 200
        assert metrics["false_accepts"] == metrics["false_rejects"] == 0
        assert metrics["repeat_binary_flips"] == 0
        assert metrics["agreement_case_cluster_95_interval"] == [1, 1]
        assert metrics["binary_brier"] == pytest.approx(0.007525)
        assert metrics["ece_10_bins"] == pytest.approx(0.0835)


def test_partial_rejects_and_false_accepts_are_separate():
    rows, gold = synthetic_records()
    for row in rows:
        if row["id"] == "0":
            row["p"] = [0.1, 0.1, 0.8]  # A partial answer is rejected, not accepted.
        if row["id"] == "99":
            row["p"] = [0.8, 0.1, 0.1]
    result = analyze(rows, gold)
    assert result["contexts"]["J0"]["matches"] == 196
    assert result["contexts"]["J0"]["false_accepts"] == 2
    assert result["contexts"]["J0"]["false_rejects"] == 2


def test_missing_or_nonfinite_judgments_are_refused():
    rows, gold = synthetic_records()
    with pytest.raises(ValueError, match="analysis_coverage_invalid"):
        analyze(rows[:-1], gold)
    rows[0]["p"][0] = float("nan")
    with pytest.raises(ValueError, match="analysis_readout_invalid"):
        analyze(rows, gold)
