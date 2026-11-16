"""Bounded adoption checks; request text, responses and IDs stay in RAM only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEALED = ROOT / "runtime/lily-verifier-study/candidate-source/target/release/stark-generate"
SEALED_SHA = "096cceb6b370fea71b49523c928c68c1ac172ca77b63a968acf7ceb0a26dbe2e"
SAMPLING = {"mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": 0.95}
SCOPE = "419a984a0e0d455fb2560eccf6b7ad47"
SIGNATURE = ("output_token_ids_sha256", "text_sha256", "output_tokens", "sampled_tokens", "finish_reason", "mtp_counters")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(value):
    if not value:
        raise RuntimeError("qualification_failed")


def signature(row):
    return {key: row[key] for key in SIGNATURE}


def summary(reply, ready, expected_input, limit, drafts, reuse=0):
    require(isinstance(reply, dict) and "error" not in reply)
    ids, text = reply.get("generated_token_ids"), reply.get("text")
    require(isinstance(ids, list) and bool(ids) and all(type(i) is int and 0 <= i < 2**32 for i in ids))
    require(isinstance(text, str) and bool(text.strip()))
    stop = reply.get("stop_token_id")
    eos = type(stop) is int and stop in ready.get("eos_token_ids", [])
    sampled, finish = reply.get("sampled_tokens"), reply.get("finish_reason")
    require(type(sampled) is int and sampled == len(ids) + int(eos) and sampled <= limit)
    require(finish in {"length", "eos"} and (finish == "eos") == eos)
    require(finish != "length" or (sampled == limit and stop is None))
    require(reply.get("input_tokens") == expected_input and reply.get("reused_input_tokens", 0) == reuse)
    mtp = reply.get("mtp") or {}
    require(mtp.get("enabled") is (drafts > 0) and mtp.get("drafts") == drafts)
    counters = {key: (mtp.get("stats") or {}).get(key) for key in ("rounds", "proposed", "accepted", "verified_rows")}
    require(all(type(i) is int and i >= 0 for i in counters.values()))
    require(counters["accepted"] <= counters["proposed"])
    if drafts:
        require(counters["rounds"] > 0 and counters["proposed"] > 0)
        require(mtp.get("verification") == "batched-decode-exact-v1")
    else:
        require(all(i == 0 for i in counters.values()))
    return {
        "input_tokens": expected_input, "output_tokens": len(ids), "sampled_tokens": sampled,
        "finish_reason": finish, "reused_input_tokens": reuse, "mtp_drafts": drafts,
        "mtp_counters": counters, "output_token_ids_sha256": hashlib.sha256(b"".join(i.to_bytes(4, "little") for i in ids)).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prefill_seconds": reply.get("prefill_seconds"), "decode_seconds": reply.get("decode_seconds"),
    }


def clear_env():
    for key in tuple(os.environ):
        if key.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(key)
    os.environ["STARK_REQUEST_BUDGET_MIB"] = "16"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--phase", choices=("lifecycle", "native-lifecycle", "arms", "prompts", "book"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    require(not args.output.exists() and sha(args.worker) == args.worker_sha256)
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools/converter")]
    prior = load("_adoption_prior", ROOT / "runtime/lily-slack-long/run.py")
    identities = load("_adoption_identities", prior.OLD_RUNNER)
    identity = json.loads(prior.IDENTITY.read_text())
    require(identities.current_model_identity(identity))
    require(sha(prior.BASE) == prior.BASE_SHA256 and sha(SEALED) == SEALED_SHA)
    clear_env()
    os.environ["PATH"] = str(prior.TEMPLATE_HELPER.parent) + os.pathsep + os.environ.get("PATH", "")
    from stark_convert import chat, reasoning, template_runtime
    from validate_generation_prefix_cache import Worker
    template_runtime._template_executable = lambda: prior.TEMPLATE_HELPER.resolve()
    report = {
        "schema": "stark-lily-projection-adoption-v1", "attempt": "20261116-LILY-REUSE-01",
        "phase": args.phase, "worker_sha256": args.worker_sha256, "harness_sha256": sha(Path(__file__)),
        "sealed_worker_sha256": SEALED_SHA, "baseline_worker_sha256": prior.BASE_SHA256,
        "model_identity_receipt_sha256": sha(prior.IDENTITY), "model_content_manifest_sha256": identity.get("model_content_manifest_sha256"),
        "config_sha256": sha(prior.MODEL / "config.json"), "template_helper_sha256": sha(prior.TEMPLATE_HELPER),
        "helper_sha256": {"generation": sha(Path(prior.__file__)), "identity": sha(prior.OLD_RUNNER)},
        "privacy": "all prompt/response text and token arrays stay in RAM; aggregate-only receipts",
        "performance": "not a new controlled timing comparison", "status": "running", "checks": {}, "rows": [],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    def save():
        prior.write(args.output, report)
    def check(name, value=True):
        require(value)
        report["checks"][name] = True
        save()
        print(json.dumps({"phase": args.phase, "check": name}), flush=True)
    def ready(worker):
        mtp, image = worker.ready.get("mtp") or {}, worker.ready.get("image_input") or {}
        require(mtp.get("supported") is True and mtp.get("recommended_drafts") == 2 and mtp.get("max_drafts", 0) >= 3)
        require(image.get("family_profile") == prior.IMAGE_PROFILE)
    def guard():
        check("model_guard_" + str(len(report["checks"])), identities.current_model_identity(identity))
    save()
    try:
        if args.phase == "lifecycle":
            qualifier = load("_adoption_image_qualifier", ROOT / "scripts/qualify_flashnext_images.py")
            receipt = args.output.with_name(args.output.stem + "-details.json")
            require(not receipt.exists())
            result = qualifier.qualify(argparse.Namespace(worker=args.worker, model=prior.MODEL,
                output=receipt, corpus=None, tokens=64, timeout=900, progress=True))
            report["details_sha256"] = sha(receipt)
            report["qualifier_sha256"] = sha(Path(qualifier.__file__))
            report["details_checks"] = result.get("checks", {})
            check("image_text_mtp_prefix_lifecycle", result.get("status") == "passed")
            guard()
        elif args.phase == "native-lifecycle":
            qualifier = load("_adoption_image_fixture", ROOT / "scripts/qualify_flashnext_images.py")
            prefix = load("_adoption_prefix_fixture", ROOT / "scripts/validate_generation_prefix_cache.py")
            corpus_path = ROOT / "vendor/lily-core/src/qwen4exp/model.rs"
            report["corpus_sha256"] = sha(corpus_path)
            report["image_fixture_sha256"] = hashlib.sha256(prefix.synthetic_png()).hexdigest()
            report["native_worker_helper_sha256"] = sha(Path(prefix.__file__))
            report["image_fixture_prompt_helper_sha256"] = sha(Path(qualifier.__file__))
            report["host_admission"] = "not exercised; prior ChatWorker RAM preflight refusal preserved separately"
            report["native_guards"] = "existing native loader, context and operation resource guards unchanged"
            report["sampling"] = {"mode": "greedy"}
            def text(worker, label, values, drafts, reused):
                reply = worker.request({"operation": "generate", "input_token_ids": values,
                    "context_tokens": 8192, "max_new_tokens": 64, "sampling": {"mode": "greedy"},
                    "seed": 42, "cache_scope": SCOPE, "mtp_drafts": drafts})
                row = summary(reply, worker.ready, len(values), 64, drafts, reused)
                row["case"] = label
                report["rows"].append(row)
                save()
                return row
            def exact(left, right):
                # Greedy MTP work can differ across depths; the visible target continuation must not.
                return all(left[key] == right[key] for key in ("output_token_ids_sha256", "text_sha256", "output_tokens", "sampled_tokens", "finish_reason"))
            references, changed_references = {}, {}
            with Worker(str(args.worker), str(prior.MODEL), prefix_cache=True, timeout=900, vision=True) as worker:
                ready(worker)
                all_ids = worker.tokenize(corpus_path.read_text())
                require(len(all_ids) >= 8208)
                base = all_ids[:4104]
                changed = base[:4096] + all_ids[4104:4112]
                require(base[4096:] != changed[4096:])
                report["input_token_ids_sha256"] = hashlib.sha256(b"".join(i.to_bytes(4,"little") for i in base)).hexdigest()
                report["changed_token_ids_sha256"] = hashlib.sha256(b"".join(i.to_bytes(4,"little") for i in changed)).hexdigest()
                for drafts in (0, 2):
                    require(worker.request({"operation": "reset_cache"}).get("cache_reset") is True)
                    cold = text(worker, f"mtp{drafts}_cold", base, drafts, 0)
                    reused = text(worker, f"mtp{drafts}_reuse", base, drafts, 4096)
                    check(f"mtp{drafts}_exact_prefix_reuse", exact(cold, reused))
                    tail_reused = text(worker, f"mtp{drafts}_changed_tail_reuse", changed, drafts, 4096)
                    require(worker.request({"operation": "reset_cache"}).get("cache_reset") is True)
                    tail_cold = text(worker, f"mtp{drafts}_changed_tail_cold", changed, drafts, 0)
                    check(f"mtp{drafts}_exact_changed_tail_reuse", exact(tail_reused, tail_cold))
                    references[drafts], changed_references[drafts] = cold, tail_cold
                check("greedy_mtp0_mtp2_exact", exact(references[0], references[2]) and exact(changed_references[0], changed_references[2]))
                before = text(worker, "before_malformed_image", base, 2, 4096)
                check("before_malformed_exact", exact(before, references[2]))
                # Use the native staging protocol, leaving normal host admission unchanged.
                import secrets
                bad = b"\x89PNG\r\n\x1a\n\0"
                name = secrets.token_hex(16)
                staged = worker.staging_root / name
                descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(bad)
                    malformed = worker.request({"operation": "prepare_image", "rendered_prompt": qualifier.image_prompt(prior.MODEL),
                        "image": {"id": name, "media_type": "image/png", "byte_length": len(bad)},
                        "context_tokens": 8192, "max_new_tokens": 1})
                    check("malformed_image_recoverable", malformed.get("error") == "invalid_image_request")
                finally:
                    staged.unlink(missing_ok=True)
                recovered = text(worker, "after_malformed_image", base, 2, 0)
                check("malformed_image_clears_prefix_and_exact_recovery", exact(recovered, references[2]))
                before = text(worker, "before_valid_image", base, 2, 4096)
                check("before_valid_exact", exact(before, references[2]))
                image = worker.image_transaction(8192)
                image_row = summary(image, worker.ready, image.get("input_tokens"), 1, 0, 0)
                image_row["case"] = "native_image"
                report["rows"].append(image_row)
                check("actual_image_zero_mtp_and_reuse", image_row["output_tokens"] == 1)
                check("image_staging_cleaned", not list(worker.staging_root.iterdir()))
                recovered = text(worker, "after_valid_image", base, 2, 0)
                check("image_to_text_mtp_exact_recovery", exact(recovered, references[2]))
                replay = text(worker, "post_image_text_reuse", base, 2, 4096)
                check("post_image_prefix_restored", exact(replay, references[2]))
                require(worker.request({"operation": "reset_cache"}).get("cache_reset") is True)
                reset = text(worker, "after_reset", base, 2, 0)
                check("explicit_reset_exact", exact(reset, references[2]))
            guard()
            with Worker(str(args.worker), str(prior.MODEL), prefix_cache=True, timeout=900, vision=True) as worker:
                ready(worker)
                restarted = text(worker, "after_restart", base, 2, 0)
                check("restart_clears_prefix_and_exact_output", exact(restarted, references[2]))
            guard()
        elif args.phase == "arms":
            corpus_path = ROOT / "vendor/lily-core/src/qwen4exp/model.rs"
            corpus = corpus_path.read_text()
            report["corpus_sha256"] = sha(corpus_path)
            reference = {}
            for arm, binary in (("final", args.worker), ("sealed", SEALED), ("baseline", prior.BASE)):
                clear_env()
                if arm == "sealed":
                    os.environ["STARK_LILY_VERIFY_PROJECTION_REUSE"] = "1"
                require(("STARK_LILY_VERIFY_PROJECTION_REUSE" in os.environ) is (arm == "sealed"))
                expected_binary_sha = args.worker_sha256 if arm == "final" else SEALED_SHA if arm == "sealed" else prior.BASE_SHA256
                require(sha(binary) == expected_binary_sha)
                with Worker(str(binary), str(prior.MODEL), prefix_cache=(arm == "final"), timeout=900, vision=True) as worker:
                    ready(worker)
                    ids = worker.tokenize(corpus)[:4104]
                    require(len(ids) == 4104)
                    input_hash = hashlib.sha256(b"".join(i.to_bytes(4, "little") for i in ids)).hexdigest()
                    report.setdefault("input_sha256", input_hash)
                    require(report["input_sha256"] == input_hash)
                    for drafts in (0, 1, 2, 3, "automatic"):
                        worker.request({"operation": "reset_cache"})
                        payload = {"operation": "generate", "input_token_ids": ids, "context_tokens": 8192,
                            "max_new_tokens": 64, "sampling": SAMPLING, "seed": 42}
                        if arm == "final":
                            payload["cache_scope"] = SCOPE
                        payload["mtp" if drafts == "automatic" else "mtp_drafts"] = True if drafts == "automatic" else drafts
                        effective = 2 if drafts == "automatic" else drafts
                        row = summary(worker.request(payload), worker.ready, len(ids), 64, effective)
                        row.update(arm=arm, case=str(drafts))
                        expected = reference.setdefault(str(drafts), signature(row))
                        row["matches_final"] = signature(row) == expected
                        report["rows"].append(row)
                        check(arm + "_depth_" + str(drafts), row["matches_final"])
                    if arm == "final":
                        # The preceding automatic request has populated one exact snapshot.
                        payload = {"operation": "generate", "input_token_ids": ids, "context_tokens": 8192,
                            "max_new_tokens": 64, "sampling": SAMPLING, "seed": 42, "cache_scope": SCOPE, "mtp_drafts": 2}
                        warm = summary(worker.request(payload), worker.ready, len(ids), 64, 2, 4096)
                        check("seeded_prefix_reuse", signature(warm) == reference["2"])
                        cancelled = worker.request(payload, cancel_after_first_progress=True)
                        check("native_cancellation", cancelled.get("error") == "generation_cancelled")
                        recovered = summary(worker.request(payload), worker.ready, len(ids), 64, 2, 0)
                        report["rows"].append(dict(recovered, arm=arm, case="after_cancel"))
                        check("cancel_clears_prefix_and_exact_recovery", signature(recovered) == reference["2"])
                check(arm + "_worker_unchanged", sha(binary) == expected_binary_sha)
                guard()
            clear_env()
        elif args.phase == "prompts":
            if sys.stdin.isatty():
                import termios
                flags = termios.tcgetattr(sys.stdin.fileno())
                flags[3] &= ~termios.ECHO
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, flags)
            print("ready_for_prompts", flush=True)
            supplied = json.loads(sys.stdin.readline())
            require(set(supplied) == {"coding", "planning"} and all(isinstance(v, str) and v.strip() for v in supplied.values()))
            previous_path = ROOT / "runtime/lily-verifier-study/compare-v4/session-1.json"
            previous = json.loads(previous_path.read_text())
            report["previous_receipt_sha256"] = sha(previous_path)
            codec, template = chat.load_template(prior.MODEL)
            with Worker(str(args.worker), str(prior.MODEL), prefix_cache=False, timeout=900, vision=True) as worker:
                ready(worker)
                for label, ceiling in (("coding", 2048), ("planning", 4096)):
                    rendered, ids, codec_hash = prior.render_and_tokenize(supplied[label], worker, codec, template, chat, reasoning)
                    expected_input = previous["inputs"][label]
                    actual_input = {"input_tokens": len(ids), "codec_sha256": codec_hash,
                        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                        "token_ids_sha256": hashlib.sha256(b"".join(i.to_bytes(4, "little") for i in ids)).hexdigest()}
                    check(label + "_same_input", actual_input == expected_input)
                    for warmup, limit in ((True, 64), (False, ceiling)):
                        started = time.monotonic()
                        reply = worker.request({"operation": "generate", "rendered_prompt": rendered, "context_tokens": 8192,
                            "max_new_tokens": limit, "sampling": SAMPLING, "seed": 42, "mtp_drafts": 2})
                        row = summary(reply, worker.ready, len(ids), limit, 2)
                        expected = next(row for row in previous["requests"] if row["case"] == label and row["warmup"] is warmup)
                        row.update(case=label, warmup=warmup, wall_seconds=time.monotonic() - started)
                        report["rows"].append(row)
                        check(label + ("_warmup" if warmup else "_complete") + "_same_output_work", signature(row) == signature(expected))
            guard()
        else:
            prepare = load("_adoption_book_prepare", ROOT / "runtime/lily-long-context/prepare.py")
            rendered, ids, source = prepare.prepare()
            expected_source_path = ROOT / "runtime/lily-long-context/preparation.json"
            require(source == json.loads(expected_source_path.read_text()))
            report["public_case"] = source
            report["prepare_sha256"] = sha(Path(prepare.__file__))
            previous_path = ROOT / "runtime/lily-long-context/run-01/session-1.json"
            previous = json.loads(previous_path.read_text())
            report["previous_receipt_sha256"] = sha(previous_path)
            with Worker(str(args.worker), str(prior.MODEL), prefix_cache=False, timeout=900, vision=True) as worker:
                ready(worker)
                check("book_native_input_exact", worker.tokenize(rendered) == ids)
                for warmup, limit in ((True, 64), (False, 4096)):
                    started = time.monotonic()
                    reply = worker.request({"operation": "generate", "rendered_prompt": rendered, "context_tokens": 98304,
                        "max_new_tokens": limit, "sampling": SAMPLING, "seed": 42, "mtp_drafts": 2, "stream": True})
                    row = summary(reply, worker.ready, len(ids), limit, 2)
                    expected = next(row for row in previous["requests"] if row["warmup"] is warmup)
                    row.update(case="book", warmup=warmup, wall_seconds=time.monotonic() - started)
                    report["rows"].append(row)
                    check("book_" + ("warmup" if warmup else "complete") + "_same_output_work", signature(row) == signature(expected))
            guard()
        check("worker_unchanged", sha(args.worker) == args.worker_sha256)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
    finally:
        clear_env()
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
    print(json.dumps({"phase": args.phase, "status": report["status"], "checks_passed": len(report["checks"])}), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        result = main()
    except BaseException as error:
        print(json.dumps({"status": "failed", "failure_type": type(error).__name__}), flush=True)
        result = 2
    raise SystemExit(result)
