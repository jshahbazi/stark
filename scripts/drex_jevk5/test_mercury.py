"""Model-free response validation for all hosted JevBench question types."""

import pytest

from mercury import MODEL, build_request, is_correct, parse_response, supported_cases


def test_progress_normalizes_integer_score_gold_like_final_metrics():
    assert is_correct({"labels": ["0", "1", "2"], "expected": 1}, [0.1, 0.8, 0.1])
    assert not is_correct({"labels": ["0", "1", "2"], "expected": 2}, [0.1, 0.8, 0.1])


def test_provider_limit_excludes_only_unsupported_synthetic_control():
    cases = [
        {"tier": tier, "question": {"type": "score"}, "labels": list(range(size))}
        for tier, size in [("original", 5), ("synthetic", 16), ("synthetic", 4)]
    ]
    included, excluded = supported_cases(cases)
    assert included == [cases[0], cases[2]]
    assert excluded == {"synthetic_score_over_10_levels": 1}
    with pytest.raises(ValueError, match="public_case_unsupported"):
        supported_cases([{**cases[1], "tier": "hard"}])


def payload(answer):
    return {
        "model": MODEL,
        "answers": {"decision": answer},
        "usage": {"input_tokens": 20, "output_tokens": 1},
    }


def test_boolean_probability_aligns_with_canonical_gold():
    case = {"question": {"type": "noul"}, "labels": ["true", "false"]}
    p, usage = parse_response(payload({"type": "noul", "noul": 0.8}), case)
    assert p == pytest.approx([0.8, 0.2])
    assert usage == {"input_tokens": 20, "output_tokens": 1}


def test_choice_uses_named_mapping_and_verifies_native_argmax():
    case = {"question": {"type": "choice"}, "labels": ["second", "first"]}
    answer = {
        "type": "choice",
        "choice": "first",
        "probabilities": {"first": 0.9, "second": 0.1},
        "confidence": 0.8,
    }
    assert parse_response(payload(answer), case)[0] == [0.1, 0.9]
    with pytest.raises(ValueError, match="choice_argmax_invalid"):
        parse_response(payload({**answer, "choice": "second"}), case)


def test_score_validates_weighted_mean_and_ordinal_labels():
    case = {"question": {"type": "score"}, "labels": ["0", "1", "2"]}
    answer = {
        "type": "score",
        "score": 1.3,
        "probabilities": {"2": 0.5, "0": 0.2, "1": 0.3},
        "confidence": 0.7,
    }
    assert parse_response(payload(answer), case)[0] == [0.2, 0.3, 0.5]
    with pytest.raises(ValueError, match="score_readout_invalid"):
        parse_response(payload({**answer, "score": 0.1}), case)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_boolean_refused(invalid):
    with pytest.raises(ValueError, match="boolean_invalid"):
        parse_response(
            payload({"type": "noul", "noul": invalid}),
            {"question": {"type": "noul"}, "labels": ["false", "true"]},
        )


def test_wrong_served_model_refused():
    with pytest.raises(ValueError, match="served_model_invalid"):
        parse_response({"model": "other"}, {})


def test_request_preserves_original_case_question():
    question = {
        "type": "choice",
        "instructions": "Synthetic question",
        "criteria": {"z": None, "a": "Synthetic option"},
    }
    request = build_request({"state": "Synthetic state", "question": question})
    assert request == {
        "model": MODEL,
        "state": "Synthetic state",
        "questions": {"decision": question},
    }
    assert list(request["questions"]["decision"]["criteria"]) == ["z", "a"]
