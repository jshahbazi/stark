"""Privacy and denominator checks for dense-Qwen benchmark reports."""
import json
from types import SimpleNamespace

import pytest
from benchmark_qwen38_dense import (
    collect_measurement_slots,
    finalize,
    require_mtp_depth,
    summarize,
)


def test_report_excludes_private_worker_payload_and_uses_decode_tokens():
    reply = {
        "sampled_tokens": 3, "generated_token_ids": [17, 23, 99],
        "text": "private generated text", "finish_reason": "length",
        "prefill_seconds": 1.0, "decode_seconds": 0.5,
        "generation_seconds": 1.5, "output_tokens_per_second": 6.0,
        "mtp": {"enabled": True, "drafts": 3, "proposed": 4,
                "rounds_by_accepted_drafts": [1, 2, 1, 0, 0, 0, 0, 0], "private_logits": [0.1]},
    }
    result = summarize(reply)
    assert result["useful_decode_tok_s"] == 4.0
    assert len(result["output_sha256"]) == 64
    assert "private" not in json.dumps(result)
    assert "generated_token_ids" not in result
    assert result["mtp"]["rounds_by_accepted_drafts"] == [1, 2, 1, 0, 0, 0, 0, 0]


def test_selected_long_depth_requires_readiness_advertisement():
    require_mtp_depth({"mtp": {"supported": True, "max_drafts": 7}}, 7)
    with pytest.raises(ValueError, match="mtp_depth_unavailable"):
        require_mtp_depth({"mtp": {"supported": True, "max_drafts": 3}}, 7)
    with pytest.raises(ValueError, match="mtp_depth_unavailable"):
        require_mtp_depth({"mtp": {"supported": False, "max_drafts": 0}}, 1)


def test_host_retries_fill_each_slot_and_exclude_without_speed_or_hash_selection():
    eligibility = [False, True, True, False, True, True, True, True]
    calls = []

    def measure(prompt_index, repeat_index, attempt):
        calls.append((prompt_index, repeat_index, attempt, 42))
        eligible = eligibility[attempt - 1]
        return {
            "useful_decode_tok_s": 900.0 if not eligible else 10.0,
            "output_sha256": "excluded" if not eligible else "accepted",
            "host_screen": {"eligible": eligible, "external_cpu_max": 999 if not eligible else 10},
        }

    accepted, excluded, attempts, stop = collect_measurement_slots(3, 2, 12, True, measure)
    assert attempts == 8 and stop is None
    assert len(accepted) == 6 and len(excluded) == 2
    assert [(row["prompt_index"], row["repeat_index"]) for row in accepted] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
    ]
    assert [row["measurement_attempt"] for row in accepted] == [2, 3, 5, 6, 7, 8]
    assert [row["reason"] for row in excluded] == ["host_screen_ineligible"] * 2
    assert all(row["metrics"]["output_sha256"] == "excluded" for row in excluded)
    assert all(row["useful_decode_tok_s"] == 10.0 for row in accepted)
    assert calls[:2] == [(0, 0, 1, 42), (0, 0, 2, 42)]


def test_host_retry_cap_stops_with_incomplete_quota_and_never_chooses_fast_ineligible_run():
    calls = []

    def measure(prompt_index, repeat_index, attempt):
        calls.append((prompt_index, repeat_index, attempt))
        return {"useful_decode_tok_s": 1000.0, "host_screen": {"eligible": False}}

    accepted, excluded, attempts, stop = collect_measurement_slots(3, 2, 4, True, measure)
    assert not accepted and len(excluded) == 4
    assert attempts == 4 and stop == "measurement_attempt_cap_reached"
    assert len(calls) == 4 and {call[:2] for call in calls} == {(0, 0)}


def test_without_retry_keeps_the_original_fixed_measurement_count():
    calls = []

    def measure(prompt_index, repeat_index, attempt):
        calls.append((prompt_index, repeat_index, attempt))
        return {"host_screen": {"eligible": False}}

    accepted, excluded, attempts, stop = collect_measurement_slots(3, 2, 24, False, measure)
    assert len(accepted) == attempts == 6 and not excluded and stop is None
    assert [row["measurement_attempt"] for row in accepted] == list(range(1, 7))


def test_retry_stops_on_quiet_admission_timeout():
    calls = 0

    def measure(prompt_index, repeat_index, attempt):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError
        return {"host_screen": {"eligible": True}}

    accepted, excluded, attempts, stop = collect_measurement_slots(3, 2, 12, True, measure)
    assert len(accepted) == 1 and not excluded
    assert attempts == 1 and stop == "quiet_admission_timeout"


def test_partial_retry_quota_cannot_qualify_target():
    args = SimpleNamespace(tokens=768, greedy=False, repeats=2, warmups=1, drafts=7)
    report = {"host_screen_passed": False, "runs": [{
        "warmup": False, "sampled_tokens": 768, "useful_decode_tok_s": 80.0,
        "mtp": {"enabled": True, "drafts": 7},
    }] * 5}
    finalize(report, args)
    assert not report["target_protocol_met"] and not report["target_met"]


@pytest.mark.parametrize("reply", [{"error": "private failure"}, {
    "sampled_tokens": 1, "generated_token_ids": [3], "decode_seconds": 0.5,
}, {
    "sampled_tokens": 3, "generated_token_ids": [3, 4, 5], "decode_seconds": float("nan"),
}])
def test_failed_or_prefill_only_output_never_counts_as_speed(reply):
    with pytest.raises(ValueError, match="invalid_benchmark_reply"):
        summarize(reply)


@pytest.mark.parametrize("tokens,greedy,repeats,warmups,count,expected", [
    (32, True, 1, 0, 3, False), (768, True, 2, 1, 6, False),
    (768, False, 1, 1, 3, False), (768, False, 2, 0, 6, False),
    (768, False, 2, 1, 6, True),
])
def test_debug_runs_cannot_qualify_full_target(tokens, greedy, repeats, warmups, count, expected):
    report = {"runs": [{"warmup": False, "sampled_tokens": tokens, "useful_decode_tok_s": 80.0,
                        "mtp": {"enabled": True, "drafts": 3}}] * count,
              "host_screen_passed": True}
    finalize(report, SimpleNamespace(tokens=tokens, greedy=greedy, repeats=repeats, warmups=warmups, drafts=3))
    assert report["throughput_threshold_met"]
    assert report["target_met"] is expected


def test_empty_or_unscreened_runs_cannot_qualify():
    args = SimpleNamespace(tokens=768, greedy=False, repeats=2, warmups=1, drafts=3)
    report = {"runs": []}
    finalize(report, args)
    assert report["median_tok_s"] is None and not report["target_met"]
    report = {"runs": [{"warmup": False, "sampled_tokens": 768, "useful_decode_tok_s": 80.0,
                        "mtp": {"enabled": True, "drafts": 3}}] * 6}
    finalize(report, args)
    assert report["target_protocol_met"] and not report["target_met"]


def test_screened_full_protocol_can_qualify_an_advertised_longer_depth():
    args = SimpleNamespace(tokens=768, greedy=False, repeats=2, warmups=1, drafts=7)
    report = {"host_screen_passed": True, "runs": [{
        "warmup": False, "sampled_tokens": 768, "useful_decode_tok_s": 71.0,
        "mtp": {"enabled": True, "drafts": 7},
    }] * 6}
    finalize(report, args)
    assert report["target_protocol_met"] and report["target_met"]


@pytest.mark.parametrize("requested,enabled,effective,qualifies", [
    (0, False, 0, False), (1, True, 1, True), (3, False, 0, False), (3, True, 2, False),
])
def test_only_matching_enabled_mtp_depth_can_qualify(requested, enabled, effective, qualifies):
    report = {"host_screen_passed": True, "runs": [{
        "warmup": False, "sampled_tokens": 768, "useful_decode_tok_s": 80.0,
        "mtp": {"enabled": enabled, "drafts": effective},
    }] * 6}
    finalize(report, SimpleNamespace(tokens=768, greedy=False, repeats=2, warmups=1, drafts=requested))
    assert report["throughput_threshold_met"]
    assert report["target_protocol_met"] is qualifies
    assert report["target_met"] is qualifies
