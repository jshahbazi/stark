import pytest
import qualify_decider_flash as qualification


def choice(answer, probabilities):
    return {"type": "choice", "choice": answer, "probabilities": probabilities, "confidence": 0.5}


def test_probability_maps_compare_by_option_name_not_insertion_order():
    reference = {"answers": {"q": choice("A", {"A": 0.8, "B": 0.2})}}
    candidate = {"answers": {"q": choice("A", {"B": 0.2, "A": 0.8})}}

    errors, matches = qualification.compare(reference, candidate)

    assert errors == [0.0, 0.0]
    assert matches == [True]


def test_choice_result_must_match_the_reported_top_option():
    reference = {"answers": {"q": choice("A", {"A": 0.8, "B": 0.2})}}
    candidate = {"answers": {"q": choice("B", {"A": 0.8, "B": 0.2})}}

    with pytest.raises(qualification.QualificationError, match="answer_choice_differs"):
        qualification.compare(reference, candidate)


def test_probability_map_option_mismatch_is_rejected():
    reference = {"answers": {"q": choice("A", {"A": 0.8, "B": 0.2})}}
    candidate = {"answers": {"q": choice("A", {"A": 0.8, "C": 0.2})}}

    with pytest.raises(qualification.QualificationError, match="option_labels_differ"):
        qualification.compare(reference, candidate)
