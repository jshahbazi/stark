"""Private exactness check for Lily sampler-driven positive MTP depth changes."""

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import validate_generation_prefix_cache as prefix
from stark_convert.chat import load_template
from stark_convert.mtp import mtp_capabilities, resolve_mtp


TOP20 = {"mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": 0.95}
OTHER = {"mode": "stochastic", "temperature": 1.0, "top_p": 0.95}


def generate(worker, ids, *, context, max_new, sampling, depth, automatic, scope=None):
    request = {
        "operation": "generate",
        "input_token_ids": ids,
        "context_tokens": context,
        "max_new_tokens": max_new,
        "sampling": sampling,
        "seed": 17,
    }
    if scope is not None:
        request["cache_scope"] = scope
    request["mtp" if automatic else "mtp_drafts"] = True if automatic else depth
    return worker.request(request)


def identity_unchanged(model, identity):
    expected = identity.get("model_file_identities")
    if not isinstance(expected, dict) or len(expected) != 63:
        return False
    for name, stamp in expected.items():
        try:
            current = (model / name).stat()
        except OSError:
            return False
        if [current.st_size, current.st_mtime_ns, current.st_ino] != stamp:
            return False
    return True


def run(args):
    source_hash = prefix.file_sha256(Path(__file__))
    candidate_hash_before = prefix.file_sha256(args.candidate)
    baseline_hash_before = prefix.file_sha256(args.baseline) if args.baseline else None
    identity_hash = prefix.file_sha256(args.identity)
    identity = json.loads(args.identity.read_text())
    if not identity_unchanged(args.model, identity):
        raise SystemExit("model file identity differs from prior receipt")

    codec, template = load_template(args.model)
    test_args = Namespace(context=8192, max_new=8, progress=False)
    with prefix.Worker(args.candidate, args.model, prefix_cache=True, vision=True) as cached:
        cache = cached.ready.get("prefix_cache") or {}
        capability = mtp_capabilities(cached.ready)
        policy = capability.get("sampling_recommendation_policy")
        if not capability.get("supported") or not policy:
            raise SystemExit("candidate readiness lacks a supported sampling recommendation policy")
        if "mtp-enabled" not in cache.get("key", "") or "mtp-selection" in cache.get("key", ""):
            raise SystemExit("candidate prefix key is not the expected Lily positive-depth identity")
        if not isinstance(cached.ready.get("image_input"), dict):
            raise SystemExit("vision worker did not report image readiness")

        session, turns = prefix.build_turns(
            cached,
            template,
            codec,
            17,
            5000,
            tools=0,
            instruction_paragraphs=24,
            follow_up_sentences=(1, 1, 1),
            session_kind="user-assistant",
        )
        ids = turns[0]
        chunk = cache.get("chunk_tokens")
        if type(chunk) is not int or len(ids) <= chunk or len(ids) + test_args.max_new > test_args.context:
            raise SystemExit("synthetic input does not fit one-chunk qualification bounds")

        d1 = resolve_mtp(True, capability, TOP20)
        d2 = resolve_mtp(True, capability, OTHER)
        if (d1, d2) != (1, 2):
            raise SystemExit("loaded policy did not select the expected positive depths")
        steps = [
            ("top20_one_cold", TOP20, d1),
            ("top20_one_warm", TOP20, d1),
            ("other_sampler_two", OTHER, d2),
            ("other_sampler_two_warm", OTHER, d2),
            ("top20_one_restored", TOP20, d1),
        ]
        scope = secrets.token_hex(16)
        expectation = prefix.Expectation(chunk, boundary_only=True)
        rows = []
        cached_outcomes = []

        for label, sampling, depth in steps:
            cache_key = (scope, test_args.context, bool(depth))
            predicted = expectation.predict(ids, cache_key)
            reply = generate(
                cached,
                ids,
                context=test_args.context,
                max_new=test_args.max_new,
                sampling=sampling,
                depth=depth,
                automatic=True,
                scope=scope,
            )
            outcome = prefix.outcome(reply, session.tools, None)
            expectation.record(ids, cache_key, reply)
            ack = reply.get("mtp") if "error" not in reply else {}
            rows.append({
                "case": label,
                "depth": depth,
                "sampling": "top20" if sampling is TOP20 else "other",
                "input_tokens": len(ids),
                "reused_input_tokens": reply.get("reused_input_tokens", 0),
                "expected_reused_input_tokens": predicted,
                "acknowledged_depth": ack.get("drafts"),
                "output_sha256": outcome.get("generated_sha256"),
                "output_tokens": outcome.get("output_tokens"),
                "successful": "error" not in outcome,
            })
            cached_outcomes.append(outcome)

        image_reply = cached.image_transaction(test_args.context)
        image_ack = image_reply.get("mtp") if isinstance(image_reply, dict) else None
        image_cycle_ok = (
            isinstance(image_reply, dict)
            and "error" not in image_reply
            and image_reply.get("output_tokens") == 1
            and image_reply.get("reused_input_tokens", 0) == 0
            and isinstance(image_ack, dict)
            and image_ack.get("enabled") is False
            and image_ack.get("drafts") == 0
            and (image_ack.get("stats") or {}).get("proposed", 0) == 0
        )
        if not image_cycle_ok:
            raise SystemExit("synthetic image cycle failed")

        expectation.entry = None
        post_sampling, post_depth = TOP20, d1
        post_key = (scope, test_args.context, True)
        post_label = "top20_one_after_image"
        post_predicted = expectation.predict(ids, post_key)
        post_reply = generate(
            cached,
            ids,
            context=test_args.context,
            max_new=test_args.max_new,
            sampling=post_sampling,
            depth=post_depth,
            automatic=True,
            scope=scope,
        )
        post_outcome = prefix.outcome(post_reply, session.tools, None)
        expectation.record(ids, post_key, post_reply)
        post_ack = post_reply.get("mtp") if "error" not in post_reply else {}
        rows.append({
            "case": post_label,
            "depth": post_depth,
            "sampling": "top20",
            "input_tokens": len(ids),
            "reused_input_tokens": post_reply.get("reused_input_tokens", 0),
            "expected_reused_input_tokens": post_predicted,
            "acknowledged_depth": post_ack.get("drafts"),
            "output_sha256": post_outcome.get("generated_sha256"),
            "output_tokens": post_outcome.get("output_tokens"),
            "successful": "error" not in post_outcome,
        })
        cached_outcomes.append(post_outcome)

        warm_label = "top20_one_after_image_warm"
        warm_predicted = expectation.predict(ids, post_key)
        warm_reply = generate(
            cached,
            ids,
            context=test_args.context,
            max_new=test_args.max_new,
            sampling=post_sampling,
            depth=post_depth,
            automatic=True,
            scope=scope,
        )
        warm_outcome = prefix.outcome(warm_reply, session.tools, None)
        expectation.record(ids, post_key, warm_reply)
        warm_ack = warm_reply.get("mtp") if "error" not in warm_reply else {}
        rows.append({
            "case": warm_label,
            "depth": post_depth,
            "sampling": "top20",
            "input_tokens": len(ids),
            "reused_input_tokens": warm_reply.get("reused_input_tokens", 0),
            "expected_reused_input_tokens": warm_predicted,
            "acknowledged_depth": warm_ack.get("drafts"),
            "output_sha256": warm_outcome.get("generated_sha256"),
            "output_tokens": warm_outcome.get("output_tokens"),
            "successful": "error" not in warm_outcome,
        })
        cached_outcomes.append(warm_outcome)

    if not identity_unchanged(args.model, identity):
        raise SystemExit("model file identity changed before fresh reference")
    fresh_outcomes = []
    with prefix.Worker(args.candidate, args.model, vision=True) as worker:
        fresh_capability = mtp_capabilities(worker.ready)
        if fresh_capability != capability or not isinstance(worker.ready.get("image_input"), dict):
            raise SystemExit("fresh worker readiness differs from cached worker")
        for _, sampling, depth in [*steps, (post_label, post_sampling, post_depth),
                                   (warm_label, post_sampling, post_depth)]:
            reply = generate(
                worker,
                ids,
                context=test_args.context,
                max_new=test_args.max_new,
                sampling=sampling,
                depth=depth,
                automatic=False,
            )
            fresh_outcomes.append(prefix.outcome(reply, session.tools, None))
    if not identity_unchanged(args.model, identity):
        raise SystemExit("model file identity changed during fresh reference")
    for row, cached_outcome, fresh_outcome in zip(rows, cached_outcomes, fresh_outcomes, strict=True):
        row["identical_to_fresh"] = cached_outcome == fresh_outcome

    baseline_matches = None
    if args.baseline:
        if not identity_unchanged(args.model, identity):
            raise SystemExit("model file identity changed before baseline reference")
        baseline_outcomes = []
        with prefix.Worker(args.baseline, args.model, vision=True) as worker:
            for _, sampling, depth in [*steps, (post_label, post_sampling, post_depth),
                                       (warm_label, post_sampling, post_depth)]:
                reply = generate(
                    worker,
                    ids,
                    context=test_args.context,
                    max_new=test_args.max_new,
                    sampling=sampling,
                    depth=depth,
                    automatic=False,
                )
                baseline_outcomes.append(prefix.outcome(reply, session.tools, None))
        if not identity_unchanged(args.model, identity):
            raise SystemExit("model file identity changed during baseline reference")
        for row, baseline_outcome, cached_outcome in zip(rows, baseline_outcomes, cached_outcomes, strict=True):
            row["identical_to_baseline"] = cached_outcome == baseline_outcome
        baseline_matches = all(
            cached_outcome == baseline_outcome
            for cached_outcome, baseline_outcome in zip(cached_outcomes, baseline_outcomes, strict=True)
        )

    candidate_hash_after = prefix.file_sha256(args.candidate)
    baseline_hash_after = prefix.file_sha256(args.baseline) if args.baseline else None
    if (candidate_hash_after != candidate_hash_before or baseline_hash_after != baseline_hash_before
            or prefix.file_sha256(args.identity) != identity_hash
            or prefix.file_sha256(Path(__file__)) != source_hash
            or not identity_unchanged(args.model, identity)):
        raise SystemExit("qualification input identity changed during run")

    return {
        "schema": "stark-lily-sampler-prefix-transition-v1",
        "policy": policy,
        "candidate_sha256": candidate_hash_before,
        "baseline_sha256": baseline_hash_before,
        "source_sha256": source_hash,
        "prior_identity_sha256": identity_hash,
        "prior_model_content_manifest_sha256": identity.get("model_content_manifest_sha256"),
        "corpus_sha256": identity.get("corpus_sha256"),
        "model_file_identity_count": len(identity.get("model_file_identities", {})),
        "prefix_cache_key": cache.get("key"),
        "chunk_tokens": chunk,
        "vision_worker": True,
        "image_cycle_completed": image_cycle_ok,
        "image_mtp_ack_zero": image_cycle_ok,
        "input_tokens": len(ids),
        "transition_rows": rows,
        "cached_matches_baseline": baseline_matches,
        "passed": all(
            row["reused_input_tokens"] == row["expected_reused_input_tokens"]
            and row["reused_input_tokens"] > 0
            and row["acknowledged_depth"] == row["depth"]
            and row["successful"]
            and row["identical_to_fresh"]
            for row in rows[1:-2]
        ) and rows[0]["reused_input_tokens"] == 0
        and rows[0]["expected_reused_input_tokens"] == 0
        and rows[0]["acknowledged_depth"] == rows[0]["depth"]
        and rows[0]["successful"] and rows[0]["identical_to_fresh"]
        and rows[-2]["reused_input_tokens"] == 0
        and rows[-2]["expected_reused_input_tokens"] == 0
        and rows[-2]["acknowledged_depth"] == rows[-2]["depth"]
        and rows[-2]["successful"] and rows[-2]["identical_to_fresh"]
        and rows[-1]["reused_input_tokens"] > 0
        and rows[-1]["reused_input_tokens"] == rows[-1]["expected_reused_input_tokens"]
        and rows[-1]["acknowledged_depth"] == rows[-1]["depth"]
        and rows[-1]["successful"] and rows[-1]["identical_to_fresh"]
        and image_cycle_ok
        and (baseline_matches is None or baseline_matches),
    }


def main():
    os.umask(0o077)
    parser_arg = argparse.ArgumentParser(description=__doc__)
    parser_arg.add_argument("--model", required=True, type=Path)
    parser_arg.add_argument("--candidate", required=True, type=Path)
    parser_arg.add_argument("--baseline", type=Path)
    parser_arg.add_argument("--identity", required=True, type=Path)
    parser_arg.add_argument("--output", required=True, type=Path)
    parser_arg.add_argument("--vision", action="store_true", required=True)
    args = parser_arg.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    try:
        result = run(args)
    except BaseException as error:
        failure = {"schema": "stark-lily-sampler-prefix-transition-v1",
                   "failure_type": type(error).__name__}
        if args.candidate.is_file():
            failure["candidate_sha256"] = prefix.file_sha256(args.candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(failure, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({"passed": False, "failure_type": type(error).__name__}))
        raise SystemExit("sampler prefix qualification failed") from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "transitions": len(result["transition_rows"]),
                      "input_tokens": result["input_tokens"]}))


if __name__ == "__main__":
    main()
