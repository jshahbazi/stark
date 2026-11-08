"""Verify that recovery cannot replay already answered question identities."""

import pytest

from finish_mercury import digest, probe_cases, remaining_cases
from mercury import supported_cases


def population():
    output = []
    for i, (kind, size) in enumerate(
        [
            ("choice", 2),
            ("noul", 2),
            ("score", 4),
            ("choice", 3),
            ("choice", 16),
            ("score", 16),
            ("noul", 2),
        ]
    ):
        criteria = (
            {str(j): None for j in range(size)}
            if kind == "choice"
            else ["Synthetic level"] * size
        )
        question = {"type": kind}
        if kind != "noul":
            question["criteria"] = criteria
        output.append(
            {
                "id": str(i),
                "state": "Synthetic state",
                "question": question,
                "labels": list(map(str, range(size))),
                "tier": "synthetic",
            }
        )
    return output


def test_remaining_excludes_successful_prefix_and_probe_answers():
    cases = population()
    eligible, _ = supported_cases(cases)
    failed = {
        "status": "failed",
        "http_status": 429,
        "request_population_sha256": digest(eligible),
        "rounds": [{"round": 0, "completed": 2}],
        "physical_requests_succeeded": 2,
    }
    selected = probe_cases(cases)
    probe = {
        "probe": True,
        "status": "complete",
        "request_population_sha256": digest(selected),
        "physical_requests_succeeded": len(selected),
    }
    remaining, answered, total = remaining_cases(cases, failed, [probe, probe])
    assert [c["id"] for c in remaining] == ["3", "6"]
    assert answered == 4
    assert total == 6
    with pytest.raises(ValueError, match="prior_population_mismatch"):
        remaining_cases(
            cases, {**failed, "request_population_sha256": "wrong"}, [probe]
        )
