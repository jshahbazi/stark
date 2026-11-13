"""Counterexamples for the bounded reference qualification boundary."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
import validate_edge0_reference as qualification


def reply():
    return {
        "generated_token_ids": [31] * 32,
        "text": "synthetic",
        "sampled_tokens": 32,
        "finish_reason": "length",
        "stop_token_id": None,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"error": "private error text"},
        {"generated_token_ids": [248320]},
        {"sampled_tokens": 0},
        {"sampled_tokens": 33},
        {"sampled_tokens": 31},
        {"text": ""},
        {"finish_reason": "context"},
        {"reused_input_tokens": 1},
        {"finish_reason": "eos", "stop_token_id": 1},
    ],
)
def test_failed_or_out_of_scope_results_cannot_pass(change):
    with pytest.raises(qualification.Failure):
        qualification.result_summary({**reply(), **change})


def test_result_summary_retains_only_aggregates_and_hashes():
    summary = qualification.result_summary(reply())
    serialized = json.dumps(summary)
    assert "synthetic" not in serialized
    assert "generated_token_ids" not in summary
    assert len(summary["text_sha256"]) == len(summary["token_sha256"]) == 64
    eos = {
        **reply(),
        "generated_token_ids": [31] * 31,
        "finish_reason": "eos",
        "stop_token_id": 248046,
    }
    assert qualification.result_summary(eos)["finish_reason"] == "eos"


def test_rss_watchdog_rejects_before_reading_worker_output():
    worker = object.__new__(qualification.Worker)
    worker.owner = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=qualification.RSS_LIMIT + 1)
    )
    worker.buffer = bytearray()
    worker.peak_rss = 0
    with pytest.raises(qualification.Failure, match="^rss_limit$"):
        worker.read()


def test_reaper_only_terminates_owned_worker():
    actions = []
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: actions.append("terminate"),
        wait=lambda timeout: actions.append("wait"),
        stdin=None,
        stdout=None,
    )
    worker = object.__new__(qualification.Worker)
    worker.process = process
    worker.close()
    assert actions == ["terminate", "wait"]


def test_progress_frames_share_one_request_deadline():
    worker = object.__new__(qualification.Worker)
    worker.process = SimpleNamespace(stdin=io.BytesIO())
    deadlines = []
    replies = iter([{"event": "progress"}, {"event": "delta"}, {"finished": True}])

    def read(deadline):
        deadlines.append(deadline)
        return next(replies)

    worker.read = read
    assert worker.request({"operation": "generate"}) == {"finished": True}
    assert len(deadlines) == 3
    assert len(set(deadlines)) == 1
