"""Opt-in retained Flash Next image, text MTP and prefix lifecycle checks.

Run with the converter's locked uv environment. One native model is loaded at
a time. Prompts, completions, token vectors, image bytes and exception details
stay in memory/private worker pipes. The receipt contains hashes and aggregates.
This is local execution evidence, not full reference parity or a speed benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import tempfile
import time
import zlib
from pathlib import Path

CONTEXT = 8192
PREFIX = 4104
BOUNDARY = 4096
MAXIMUM = 64
SEED = 42
SCOPE = "6a69e81ec4834d8e898becad5fbdc7cb"
FINISH = {"eos", "stop_token", "stop_sequence", "length", "context"}


class QualificationFailure(Exception):
    """Fixed category attached to the last recorded stage."""

    def __init__(self, category="correctness"):
        self.category = category
        super().__init__(category)


def require(value, category="correctness"):
    if not value:
        raise QualificationFailure(category)


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def token_hash(values):
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "receipt")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".flashnext-qualification-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            os.fchmod(output.fileno(), 0o600)
            json.dump(report, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def red_png():
    """A deterministic public 256-by-256 solid red RGB fixture."""
    side = 256

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * side) * side))
        + chunk(b"IEND", b"")
    )


def image_prompt(model):
    from stark_convert.chat import load_template

    codec, template = load_template(model)
    return template.render(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Name the main color."},
                ],
            }
        ],
        tools=None,
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token=codec.get("bos_token") or "",
        eos_token=codec.get("eos_token") or "",
    )


def summarize(reply, *, input_tokens=None, drafts, reuse, require_proposals=True):
    require(isinstance(reply, dict) and "error" not in reply, "protocol")
    generated = reply.get("generated_token_ids")
    text = reply.get("text")
    require(
        isinstance(generated, list)
        and generated
        and all(type(token) is int and 0 <= token < 2**32 for token in generated)
        and isinstance(text, str)
        and bool(text.strip())
        and len(text.encode()) <= 65536
        and type(reply.get("input_tokens")) is int
        and 0 < reply["input_tokens"] < CONTEXT
        and (input_tokens is None or reply["input_tokens"] == input_tokens)
        and type(reply.get("output_tokens")) is int
        and reply["output_tokens"] == len(generated)
        and 0 < len(generated) <= MAXIMUM
        and reply["input_tokens"] + len(generated) <= CONTEXT
        and reply.get("finish_reason") in FINISH,
        "protocol",
    )
    require(
        type(reply.get("reused_input_tokens")) is int
        and reply["reused_input_tokens"] == reuse
    )
    metadata = reply.get("mtp")
    require(
        isinstance(metadata, dict)
        and metadata.get("enabled") is (drafts > 0)
        and type(metadata.get("drafts")) is int
        and metadata["drafts"] == drafts,
        "mtp",
    )
    stats = metadata.get("stats")
    require(isinstance(stats, dict), "mtp")
    counters = {}
    for name in ("rounds", "proposed", "accepted", "verified_rows"):
        value = stats.get(name)
        require(type(value) is int and value >= 0, "mtp")
        counters[name] = value
    require(counters["accepted"] <= counters["proposed"], "mtp")
    if drafts:
        if require_proposals:
            require(counters["proposed"] > 0 and counters["rounds"] > 0, "mtp")
    else:
        require(all(value == 0 for value in counters.values()), "mtp")
    return {
        "input_tokens": reply["input_tokens"],
        "output_tokens": len(generated),
        "output_tokens_sha256": token_hash(generated),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": reply["finish_reason"],
        "reused_input_tokens": reuse,
        "mtp_drafts": drafts,
        "mtp_stats": counters,
    }


def exact(left, right):
    require(
        all(
            left[key] == right[key]
            for key in (
                "output_tokens_sha256",
                "text_sha256",
                "output_tokens",
                "finish_reason",
            )
        )
    )


def qualify(args, *, worker_factory=None, prompt_factory=image_prompt):
    from stark_convert.chat import (
        ChatError,
        ChatImageError,
        ChatRequestTooLarge,
        Worker,
    )
    from stark_convert.media import ImageData

    worker_factory = worker_factory or Worker
    require(1 <= args.tokens <= MAXIMUM and 1 <= args.timeout <= 3600, "arguments")
    repository = Path(__file__).resolve().parents[1]
    corpus_path = args.corpus or repository / "vendor/lily-core/src/qwen4exp/model.rs"
    report = {
        "schema_version": 1,
        "scope": "retained_flashnext_local_image_text_mtp_prefix_lifecycle",
        "evidence_level": "locally_tested",
        "reference_parity": "not_assessed",
        "performance": "not_measured",
        "status": "running",
        "stage": "identity",
        "context_tokens": CONTEXT,
        "input_tokens": PREFIX,
        "prefix_boundary_tokens": BOUNDARY,
        "max_new_tokens": args.tokens,
        "sampling": {"mode": "greedy"},
        "seed": SEED,
        "platform": {
            "architecture": platform.machine(),
            "macos": platform.mac_ver()[0],
        },
        "checks": {},
        "generations": [],
        "workers_started": 0,
    }
    worker = None

    def stage(name):
        report["stage"] = name
        save_report(args.output, report)
        if args.progress:
            print(json.dumps({"stage": name}), flush=True)

    def check(name):
        report["checks"][name] = True
        save_report(args.output, report)

    def start_worker():
        nonlocal worker
        stage("worker_start")
        worker = worker_factory(
            args.worker.resolve(), args.model.resolve(), args.timeout
        )
        report["workers_started"] += 1
        worker.image_check_dir = args.output.parent / ".flashnext-image-checks"
        stage("worker_ready")
        ready = worker.read()
        require(
            ready.get("ready") is True
            and ready.get("mode") == "text-generation-v1"
            and worker.images is True
            and worker.image_verification in {"locally_tested", "reference_verified"}
            and worker.prefix_cache is True
            and worker.prefix_chunk_tokens == BOUNDARY
            and worker.mtp.get("supported") is True
            and worker.mtp.get("max_drafts", 0) >= 2,
            "capability",
        )
        require(
            type(ready.get("max_context_tokens")) is int
            and ready["max_context_tokens"] >= CONTEXT,
            "capability",
        )
        report["image_verification"] = worker.image_verification
        report["runtime_identity"] = (worker.image_input or {}).get("runtime_identity")
        return ready

    def reset(name):
        stage(name)
        require(
            worker.request({"operation": "reset_cache"}).get("cache_reset") is True,
            "protocol",
        )

    def generate(name, ids, drafts, reuse):
        stage(name)
        started = time.monotonic()
        reply = worker.request(
            {
                "operation": "generate",
                "input_token_ids": ids,
                "context_tokens": CONTEXT,
                "max_new_tokens": args.tokens,
                "sampling": {"mode": "greedy"},
                "seed": SEED,
                "cache_scope": SCOPE,
                "mtp_drafts": drafts,
            }
        )
        aggregate = summarize(reply, input_tokens=len(ids), drafts=drafts, reuse=reuse)
        aggregate.update(name=name, wall_seconds=time.monotonic() - started)
        report["generations"].append(aggregate)
        save_report(args.output, report)
        return aggregate

    try:
        report["worker_sha256"] = file_hash(args.worker)
        report["config_sha256"] = file_hash(args.model / "config.json")
        report["harness_sha256"] = file_hash(Path(__file__))
        report["image_manifest_sha256"] = file_hash(
            args.model / "stark-image-artifact.json"
        )
        report["source_receipt_sha256"] = file_hash(args.model / "native-source.json")
        corpus = corpus_path.read_bytes()
        require(0 < len(corpus) <= 4 * 1024 * 1024, "corpus")
        report["corpus_sha256"] = hashlib.sha256(corpus).hexdigest()
        start_worker()
        stage("tokenization")
        tokens = worker.request(
            {
                "operation": "tokenize",
                "text": corpus.decode(),
                "add_special_tokens": False,
            }
        )
        ids = tokens.get("token_ids")
        require(
            "error" not in tokens
            and isinstance(ids, list)
            and len(ids) >= 2 * PREFIX
            and all(type(token) is int and 0 <= token < 2**32 for token in ids),
            "corpus",
        )
        base = ids[:PREFIX]
        changed = [*base[:BOUNDARY], *ids[PREFIX : PREFIX + PREFIX - BOUNDARY]]
        require(base[BOUNDARY:] != changed[BOUNDARY:], "corpus")
        # Reject accidental short cyclic/repeated-token fixtures. The full
        # prefix comes directly from the bound source file, without cycling.
        require(
            all(
                base[:-period] != base[period:]
                for period in (1, 2, 4, 8, 16, 64, 256, 1024)
            ),
            "corpus",
        )
        report["input_sha256"] = token_hash(base)
        report["changed_input_sha256"] = token_hash(changed)
        report["corpus_token_count"] = len(ids)
        reference = {}
        changed_reference = {}
        for drafts in (0, 2):
            prefix = f"mtp{drafts}"
            reset(prefix + "_reset")
            cold = generate(prefix + "_cold", base, drafts, 0)
            replay = generate(prefix + "_reuse", base, drafts, BOUNDARY)
            exact(cold, replay)
            check(prefix + "_exact_prefix_reuse")
            changed_cached = generate(
                prefix + "_changed_tail_reuse", changed, drafts, BOUNDARY
            )
            reset(prefix + "_changed_tail_reset")
            changed_cold = generate(prefix + "_changed_tail_cold", changed, drafts, 0)
            exact(changed_cached, changed_cold)
            check(prefix + "_exact_changed_tail_reuse")
            reference[drafts], changed_reference[drafts] = cold, changed_cold
        exact(reference[0], reference[2])
        exact(changed_reference[0], changed_reference[2])
        check("greedy_mtp0_mtp2_exact")

        # Populate a text snapshot before each image operation, proving that
        # both rejected and successful image transactions invalidate it.
        before = generate("before_malformed_image", base, 2, BOUNDARY)
        exact(before, reference[2])
        prompt = prompt_factory(args.model)
        image_options = {
            "context_tokens": CONTEXT,
            "max_new_tokens": args.tokens,
            "sampling": {"mode": "greedy"},
        }
        stage("malformed_image")
        try:
            worker.request_image(
                prompt, ImageData(b"\x89PNG\r\n\x1a\n\0", "image/png"), image_options
            )
        except ChatImageError as error:
            require(error.code == "invalid_image_request", "image")
        else:
            raise QualificationFailure("image")
        check("malformed_image_recoverable")
        recovered = generate("after_malformed_image", base, 2, 0)
        exact(recovered, reference[2])
        check("malformed_image_clears_text_prefix")
        warm = generate("before_valid_image", base, 2, BOUNDARY)
        exact(warm, reference[2])
        stage("valid_image")
        image_bytes = red_png()
        reply = worker.request_image(
            prompt, ImageData(image_bytes, "image/png"), image_options
        )
        image_summary = summarize(reply, drafts=0, reuse=0)
        require(worker._last_image_tokens == 64, "image")
        require(re.search(r"\bred\b", reply["text"].casefold()) is not None, "image")
        image_summary.update(name="valid_image", image_tokens=worker._last_image_tokens)
        report["generations"].append(image_summary)
        report["image_fixture"] = {
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "width": 256,
            "height": 256,
        }
        check("synthetic_red_image_bounded_answer")
        check("image_zero_mtp_and_prefix_reuse")
        require(not list(worker.media.root.iterdir()), "image")
        check("image_staging_cleaned")
        recovered = generate("after_valid_image", base, 2, 0)
        exact(recovered, reference[2])
        check("image_to_text_mtp_recovery")
        replay = generate("after_image_text_reuse", base, 2, BOUNDARY)
        exact(replay, reference[2])
        check("post_image_text_prefix_restored")
        reset("explicit_reset")
        reset_cold = generate("after_explicit_reset", base, 2, 0)
        exact(reset_cold, reference[2])
        check("explicit_reset_clears_prefix")
        stage("worker_close_before_restart")
        worker.close()
        worker = None
        start_worker()
        restarted = generate("after_worker_restart", base, 2, 0)
        exact(restarted, reference[2])
        check("restart_clears_prefix_and_preserves_output")
        report["status"] = "passed"
        stage("complete")
    except BaseException as error:  # noqa: BLE001 -- Record interrupts and unexpected failures without content dumps.
        report["status"] = "failed"
        if isinstance(error, QualificationFailure):
            category = error.category
        elif isinstance(error, ChatRequestTooLarge):
            category = "resource"
        elif isinstance(error, ChatError):
            category = "worker"
        elif isinstance(error, KeyboardInterrupt):
            category = "interrupted"
        else:
            category = "harness"
        report["failure_category"] = category
        save_report(args.output, report)
    finally:
        if worker is not None:
            try:
                worker.close()
            except BaseException:  # noqa: BLE001 -- Cleanup failures must remain content-free too.
                report.update(
                    status="failed", failure_category="cleanup", cleanup_failed=True
                )
                save_report(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tokens", type=int, default=MAXIMUM)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = qualify(args)
    except BaseException:  # noqa: BLE001 -- The opt-in CLI must never print exception content.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "arguments_or_receipt",
                    "failure_category": "harness",
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage": report["stage"],
                "checks_passed": len(report["checks"]),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
