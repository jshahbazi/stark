"""MTP capability admission and model-specific toggle resolution."""

import pytest

from stark_convert.mtp import mtp_capabilities, resolve_mtp


def descriptor(**changes):
    value = {
        "supported": True,
        "max_drafts": 3,
        "recommended_drafts": 2,
        "default_drafts": 0,
        "sampling": "exact-rejection-v1",
        "verification": "batched-affine-decode-exact-v1",
        "operations": ["generate"],
    }
    value.update(changes)
    return value


def test_loaded_model_capabilities_select_its_recommended_depth_and_return_owned_data():
    raw = descriptor(max_drafts=7, recommended_drafts=4, verification="future-model-exact-v3")
    current = mtp_capabilities({"mtp": raw})
    assert current == raw and current is not raw
    assert resolve_mtp(True, current) == 4
    assert resolve_mtp(False, current) == 0
    current["operations"].clear()
    assert raw["operations"] == ["generate"]
    assert mtp_capabilities({"mtp": raw})["operations"] == ["generate"]


@pytest.mark.parametrize("maximum, expected", [(1, 1), (2, 2), (3, 2), (8, 2)])
def test_legacy_exact_mtp_readiness_uses_bounded_compatibility_depth(maximum, expected):
    raw = descriptor(max_drafts=maximum)
    del raw["recommended_drafts"]
    del raw["operations"]
    current = mtp_capabilities({"mtp": raw})
    assert current["supported"] is True
    assert current["recommended_drafts"] == expected
    assert resolve_mtp(True, current) == expected


@pytest.mark.parametrize("raw", [None, {}, [], {"mtp": None}, {"mtp": []}, {"mtp": {"supported": False}}])
def test_absent_or_unsupported_metadata_does_not_invent_capability(raw):
    current = mtp_capabilities(raw)
    assert current == {
        "supported": False,
        "max_drafts": 0,
        "recommended_drafts": 0,
        "default_drafts": 0,
        "operations": [],
    }
    assert resolve_mtp(False, current) == 0
    with pytest.raises(ValueError, match="^mtp_not_supported$"):
        resolve_mtp(True, current)


@pytest.mark.parametrize(
    "changes",
    [
        {"supported": 1},
        {"max_drafts": True},
        {"max_drafts": 0},
        {"max_drafts": 2**32},
        {"max_drafts": 3.0},
        {"recommended_drafts": True},
        {"recommended_drafts": 0},
        {"recommended_drafts": 4},
        {"recommended_drafts": None},
        {"default_drafts": False},
        {"default_drafts": 1},
        {"sampling": "approximate"},
        {"verification": None},
        {"verification": ""},
        {"verification": "x" * 129},
        {"verification": "private\nvalue"},
        {"verification": "é"},
        {"operations": ("generate",)},
        {"operations": []},
        {"operations": ["generate_image"]},
        {"operations": ["generate", "generate"]},
    ],
)
def test_malformed_or_nonexact_support_fails_closed(changes):
    raw = descriptor(**changes)
    assert mtp_capabilities({"mtp": raw})["supported"] is False
    with pytest.raises(ValueError, match="^mtp_not_supported$"):
        resolve_mtp(True, raw)


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", "false", [], {}])
def test_toggle_requires_an_actual_boolean(enabled):
    with pytest.raises(ValueError, match="^invalid_mtp_option$"):
        resolve_mtp(enabled, descriptor())


def test_disabling_does_not_require_capability_or_mutate_it():
    raw = descriptor()
    assert resolve_mtp(False, None) == 0
    assert resolve_mtp(True, raw) == 2
    assert resolve_mtp(False, raw) == 0
    assert resolve_mtp(True, raw) == 2
    assert raw == descriptor()


TOPK = {"mode": "stochastic", "temperature": 1, "top_k": 20, "top_p": 0.95}
POLICY = "lily-topk20-one-draft-v1"


@pytest.mark.parametrize("sampling", [
    TOPK, {**TOPK, "temperature": 1.0}, {**TOPK, "min_p": None},
    {**TOPK, "temperature": 1.0000000001, "top_p": 0.9500000001},
])
def test_loaded_sampling_policy_selects_one_and_off_still_wins(sampling):
    raw = descriptor(sampling_recommendation_policy=POLICY)
    current = mtp_capabilities({"mtp": raw})
    assert current["recommended_drafts"] == 2
    assert current["sampling_recommendation_policy"] == POLICY
    assert resolve_mtp(True, current, sampling) == 1
    assert resolve_mtp(False, current, sampling) == 0
    assert raw["recommended_drafts"] == 2


@pytest.mark.parametrize(
    "sampling",
    [
        None, {}, {"mode": "greedy"},
        {**TOPK, "temperature": 0.9}, {**TOPK, "temperature": True},
        {**TOPK, "temperature": 1.0000001192092896},
        {**TOPK, "top_k": 19}, {**TOPK, "top_k": 21}, {**TOPK, "top_k": 20.0},
        {k: v for k, v in TOPK.items() if k != "top_k"},
        {**TOPK, "top_p": 0.9}, {**TOPK, "top_p": 0.9500000476837158},
        {**TOPK, "temperature": float("inf")}, {**TOPK, "top_p": float("nan")},
        {**TOPK, "temperature": 10**1000},
        {k: v for k, v in TOPK.items() if k != "top_p"},
        {**TOPK, "min_p": 0}, {**TOPK, "min_p": 0.05},
        {**TOPK, "future_filter": 1},
    ],
)
def test_sampling_policy_does_not_expand_to_untested_filters(sampling):
    current = descriptor(sampling_recommendation_policy=POLICY)
    assert resolve_mtp(True, current, sampling) == 2


@pytest.mark.parametrize("policy", [None, "future-policy-v2", {}, [], True])
def test_missing_or_unrecognized_sampling_policy_uses_scalar_recommendation(policy):
    current = mtp_capabilities({"mtp": descriptor(sampling_recommendation_policy=policy)})
    assert "sampling_recommendation_policy" not in current
    assert resolve_mtp(True, current, TOPK) == 2


def test_model_name_or_sampling_alone_cannot_grant_a_policy():
    current = mtp_capabilities({"mtp": descriptor(), "model": "Qwen3.8-Flash-Next-lily-q4-xl"})
    assert resolve_mtp(True, current, TOPK) == 2


@pytest.mark.parametrize("extra", [{"ready": False}, {"ready": 1}, {"mode": "scoring-v1"}])
def test_failed_or_wrong_protocol_readiness_cannot_advertise_mtp(extra):
    assert mtp_capabilities({"mtp": descriptor(), **extra})["supported"] is False
