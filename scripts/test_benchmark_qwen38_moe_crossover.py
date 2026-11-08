import pytest

from benchmark_qwen38_moe_crossover import plan, summarize_groups


def test_plan_balances_each_route_and_candidate_in_both_directions():
    groups = plan(42, 2)
    assert len(groups) == 8
    for route in ("off", "on"):
        for candidate in (1, 2):
            selected = [g for g in groups if g["route"] == route and g["candidate"] == candidate]
            assert {tuple(g["order"]) for g in selected} == {(0, candidate, candidate, 0), (candidate, 0, 0, candidate)}


def test_each_group_uses_its_own_baseline_despite_time_drift():
    groups = []
    for candidate, seconds in ((1, 2), (2, 4)):
        groups.append({"candidate": candidate, "route": "on", "repetition": 0,
                       "requests": [{"variant": v, "decode_seconds": seconds, "wall_seconds": seconds+.3} for v in (0, candidate, candidate, 0)]})
    result = summarize_groups(groups)
    assert [r["candidate_over_reference_decode_ratio"] for r in result] == [1, 1]
    assert result[0]["reference_decode_tok_s"] == pytest.approx(127.5)
    assert result[1]["reference_decode_tok_s"] == pytest.approx(63.75)


def test_incomplete_group_is_not_ranked():
    with pytest.raises(ValueError, match="incomplete_crossover_group"):
        summarize_groups([{"candidate": 1, "requests": [{"variant": 0}]}])
