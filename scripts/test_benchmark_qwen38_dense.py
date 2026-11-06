"""Privacy and denominator checks for dense-Qwen benchmark reports."""
import json
from types import SimpleNamespace

import pytest
from benchmark_qwen38_dense import finalize, summarize


def test_report_excludes_private_worker_payload_and_uses_decode_tokens():
    reply = {
        "sampled_tokens": 3, "generated_token_ids": [17, 23, 99],
        "text": "private generated text", "finish_reason": "length",
        "prefill_seconds": 1.0, "decode_seconds": 0.5,
        "generation_seconds": 1.5, "output_tokens_per_second": 6.0,
        "mtp": {"enabled": True, "drafts": 3, "proposed": 4, "private_logits": [0.1]},
    }
    result = summarize(reply)
    assert result["useful_decode_tok_s"] == 4.0
    assert len(result["output_sha256"]) == 64
    assert "private" not in json.dumps(result)
    assert "generated_token_ids" not in result


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


@pytest.mark.parametrize("requested,enabled,effective", [(0, False, 0), (1, True, 1), (3, False, 0), (3, True, 2)])
def test_disabled_or_different_drafting_cannot_qualify(requested, enabled, effective):
    report = {"host_screen_passed": True, "runs": [{
        "warmup": False, "sampled_tokens": 768, "useful_decode_tok_s": 80.0,
        "mtp": {"enabled": enabled, "drafts": effective},
    }] * 6}
    finalize(report, SimpleNamespace(tokens=768, greedy=False, repeats=2, warmups=1, drafts=requested))
    assert report["throughput_threshold_met"] and not report["target_protocol_met"] and not report["target_met"]
