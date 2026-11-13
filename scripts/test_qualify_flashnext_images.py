"""Model-free counterexamples for the opt-in Flash Next lifecycle harness."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import qualify_flashnext_images as qualification
from stark_convert.chat import ChatImageError, ChatRequestTooLarge


class WorkerFixture:
    def __init__(self, owner, defect=None):
        self.owner, self.defect = owner, defect
        self.previous = None
        self.closed = False
        self.images = True
        self.image_verification = "locally_tested"
        self.prefix_cache = True
        self.prefix_chunk_tokens = 4096
        self.mtp = {"supported": True, "max_drafts": 2}
        self.image_input = {"runtime_identity": {"backend": "synthetic"}}
        self.media = SimpleNamespace(root=owner.media)

    def read(self):
        return {"ready": True, "mode": "text-generation-v1", "max_context_tokens": 8192}

    @staticmethod
    def reply(
        drafts,
        reuse,
        input_tokens=4104,
        text="private synthetic completion",
        generated=None,
    ):
        generated = generated or [31, 32, 33]
        return {
            "generated_token_ids": generated,
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": len(generated),
            "finish_reason": "length",
            "reused_input_tokens": reuse,
            "mtp": {
                "enabled": drafts > 0,
                "drafts": drafts,
                "stats": {
                    "rounds": int(drafts > 0),
                    "proposed": drafts,
                    "accepted": 0,
                    "verified_rows": drafts,
                },
            },
        }

    def request(self, request):
        if request["operation"] == "tokenize":
            return {"token_ids": list(range(9000))}
        if request["operation"] == "reset_cache":
            self.previous = None
            return {"cache_reset": True}
        assert request["operation"] == "generate"
        self.owner.requests.append(request)
        assert request["context_tokens"] == 8192
        assert request["max_new_tokens"] <= 64
        assert len(request["cache_scope"]) == 32
        assert request["sampling"] == {"mode": "greedy"}
        if self.defect == "resource":
            raise ChatRequestTooLarge(
                "private resource explanation must not be recorded"
            )
        ids, drafts = request["input_token_ids"], request["mtp_drafts"]
        reuse = (
            4096
            if self.previous is not None and self.previous[:4096] == ids[:4096]
            else 0
        )
        self.previous = ids
        if self.defect == "no_reuse":
            reuse = 0
        generated = [41, 42, 43] if ids[-1] != 4103 else [31, 32, 33]
        if self.defect == "changed_output" and reuse:
            generated = [34, 35, 36]
        reply = self.reply(drafts, reuse, generated=generated)
        if self.defect == "no_proposals" and drafts:
            reply["mtp"]["stats"] = dict.fromkeys(
                ("rounds", "proposed", "accepted", "verified_rows"), 0
            )
        return reply

    def request_image(self, _prompt, image, _options):
        self.previous = None
        if image.data == b"\x89PNG\r\n\x1a\n\0":
            raise ChatImageError("invalid_image_request")
        self._last_image_tokens = 64
        reply = self.reply(0, 0, input_tokens=90, text="red")
        if self.defect == "image_mtp_leak":
            reply["mtp"]["stats"]["proposed"] = 1
        if self.defect == "negated_image_answer":
            reply["text"] = "not red; it is blue"
        if self.defect == "reasoning_only_image_answer":
            reply["text"] = "<think>It might be red.</think>blue"
        return reply

    def close(self):
        self.closed = True


class FixtureFactory:
    def __init__(self, root, defect=None):
        self.media = root / "media"
        self.media.mkdir()
        self.defect, self.workers, self.requests = defect, [], []

    def __call__(self, _worker, _model, _timeout):
        assert all(worker.closed for worker in self.workers), (
            "Harness loaded concurrent model instances"
        )
        worker = WorkerFixture(self, self.defect)
        self.workers.append(worker)
        return worker


def arguments(root):
    model = root / "model"
    model.mkdir()
    for name in ("config.json", "stark-image-artifact.json", "native-source.json"):
        (model / name).write_text("{}")
    worker = root / "worker"
    worker.write_bytes(b"synthetic worker identity")
    corpus = root / "corpus.rs"
    corpus.write_text("private source corpus must not be recorded")
    return Namespace(
        worker=worker,
        model=model,
        output=root / "report.json",
        corpus=corpus,
        tokens=64,
        timeout=30,
        progress=False,
    )


def test_complete_lifecycle_is_serialized_and_receipt_contains_only_aggregates(
    tmp_path,
):
    args = arguments(tmp_path)
    factory = FixtureFactory(tmp_path)
    report = qualification.qualify(
        args,
        worker_factory=factory,
        prompt_factory=lambda _model: "private image prompt",
    )
    assert report["status"] == "passed" and report["stage"] == "complete"
    assert report["workers_started"] == 2 and all(
        worker.closed for worker in factory.workers
    )
    assert report["checks"]["mtp0_exact_prefix_reuse"]
    assert report["checks"]["mtp2_exact_changed_tail_reuse"]
    assert report["checks"]["greedy_mtp0_mtp2_exact"]
    assert report["checks"]["restart_clears_prefix_and_preserves_output"]
    text = args.output.read_text()
    assert json.loads(text) == report
    for sensitive in (
        "input_token_ids",
        "generated_token_ids",
        "private synthetic completion",
        "private source corpus",
        "private image prompt",
        str(args.model),
    ):
        assert sensitive not in text
    assert args.output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "defect,stage,category",
    [
        ("no_reuse", "mtp0_reuse", "correctness"),
        ("changed_output", "mtp0_reuse", "correctness"),
        ("no_proposals", "mtp2_cold", "mtp"),
        ("image_mtp_leak", "valid_image", "mtp"),
        ("negated_image_answer", "valid_image", "image"),
        ("reasoning_only_image_answer", "valid_image", "image"),
        ("resource", "mtp0_cold", "resource"),
    ],
)
def test_failed_gates_preserve_sanitized_partial_receipt_and_close_owned_worker(
    tmp_path, defect, stage, category
):
    args = arguments(tmp_path)
    factory = FixtureFactory(tmp_path, defect)
    report = qualification.qualify(
        args,
        worker_factory=factory,
        prompt_factory=lambda _model: "private image prompt",
    )
    assert report["status"] == "failed"
    assert report["stage"] == stage and report["failure_category"] == category
    assert json.loads(args.output.read_text()) == report
    assert len(factory.workers) == 1 and factory.workers[0].closed
    assert "private" not in args.output.read_text()


@pytest.mark.parametrize(
    "answer", ["red", " Red. ", "RED!", "**red**", "<think>Color check.</think>red"]
)
def test_color_smoke_accepts_only_an_explicit_final_label(answer):
    assert qualification.red_answer(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "not red",
        "not red; it is blue",
        "red or blue",
        "red?",
        "I cannot identify whether it is red",
        "<think>red</think>blue",
        "<think>red",
    ],
)
def test_color_smoke_rejects_negated_ambiguous_and_reasoning_only_labels(answer):
    assert not qualification.red_answer(answer)


def test_summary_refuses_boolean_counters_and_unacknowledged_draft_depth():
    valid = WorkerFixture.reply(2, 4096)
    for change in (
        {"reused_input_tokens": True},
        {"mtp": {**valid["mtp"], "drafts": 1}},
        {"mtp": {**valid["mtp"], "stats": {**valid["mtp"]["stats"], "proposed": True}}},
    ):
        with pytest.raises(qualification.QualificationFailure):
            qualification.summarize(
                {**valid, **change}, input_tokens=4104, drafts=2, reuse=4096
            )
