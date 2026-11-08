"""Model-free checks for semantic answer mapping and paired evaluation."""

import pytest

import compare
from compare import align, drex_request, paired


def test_boolean_options_align_by_label_not_position():
    assert align(["true", "false"], ["false", "true"], [0.2, 0.8]) == [0.8, 0.2]


def test_sealed_case_mapping_handles_legacy_boolean_and_shuffled_choice(monkeypatch):
    sealed = [
        {"id": "bool", "labels": ["no", "yes"], "expected": "yes"},
        {"id": "choice", "labels": ["second", "first"], "expected": "first"},
    ]
    raw = [
        {"id": "bool", "state": "Synthetic", "question": {"type": "noul"}},
        {
            "id": "choice",
            "state": "Synthetic",
            "question": {"type": "choice", "criteria": {"first": None, "second": None}},
        },
    ]
    rows = [
        {
            "case_id": "bool",
            "condition": "fresh",
            "probabilities": [0.8, 0.2],
            "decision": {"noul": 0.8},
        },
        {
            "case_id": "choice",
            "condition": "fresh",
            "probabilities": [0.9, 0.1],
            "decision": {"probabilities": {"first": 0.9, "second": 0.1}},
        },
    ]
    monkeypatch.setattr(
        compare, "load_reference", lambda _: (sealed, {"reference_path": "private"})
    )
    monkeypatch.setattr(
        compare,
        "read",
        lambda p: {"cases": raw} if p.name == "dataset.json" else {"rows": rows},
    )
    loaded, _ = compare.load_cases("private")
    assert loaded[0]["labels"] == ["false", "true"]
    assert loaded[0]["expected"] == "true"
    assert loaded[0]["reference_probabilities"] == [0.2, 0.8]
    assert loaded[1]["reference_probabilities"] == [0.1, 0.9]


@pytest.mark.parametrize(
    ("keys", "values"),
    [
        (["true", "true"], [0.5, 0.5]),
        (["true", "unknown"], [0.5, 0.5]),
        (["true", "false"], [float("nan"), 0.5]),
        (["true", "false"], [0.8, 0.8]),
    ],
)
def test_invalid_readouts_refused(keys, values):
    with pytest.raises(ValueError):
        align(["true", "false"], keys, values)


def test_list_choice_preserves_order_and_question():
    question = {"type": "choice", "instructions": "Select", "criteria": ["z", "a"]}
    request = drex_request({"state": "Synthetic", "question": question})
    assert list(request["questions"]["decision"]["criteria"]) == ["z", "a"]
    assert question["criteria"] == ["z", "a"]


def test_paired_result_separates_public_and_synthetic():
    cases = [
        {"id": str(i), "tier": tier, "labels": ["yes", "no"], "expected": "yes"}
        for i, tier in enumerate(["public", "public", "synthetic"])
    ]
    left = {"0": [0.9, 0.1], "1": [0.1, 0.9], "2": [0.1, 0.9]}
    right = {"0": [0.1, 0.9], "1": [0.9, 0.1], "2": [0.9, 0.1]}
    result = paired(cases, left, right)
    assert result["public"]["jev_only_correct"] == 1
    assert result["public"]["drex_only_correct"] == 1
    assert result["public"]["drex_minus_jev_accuracy"] == 0
    assert result["public"]["mcnemar_exact_two_sided_p"] == 1
    assert result["synthetic"]["drex_minus_jev_accuracy"] == 1
