#!/usr/bin/env python3
"""Record source-matched native Flash Next measurements, one loaded worker at a time."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_generation_prefix_cache import Worker, digest_ids, generate

ARMS = {
    "baseline": {"STARK_LILY_PREFILL_CHUNK": "4096"},
    "chunk1024": {"STARK_LILY_PREFILL_CHUNK": "1024"},
    "chunk2048": {"STARK_LILY_PREFILL_CHUNK": "2048"},
    "hc": {"STARK_LILY_PREFILL_CHUNK": "4096", "STARK_LILY_VERIFY_HC_DOWN_REUSE": "1"},
    "moe": {"STARK_LILY_PREFILL_CHUNK": "4096", "STARK_FLASHNEXT_PREFILL_GROUPED_SWIGLU": "1"},
}
COUNTS = ("rounds", "proposed", "accepted", "verified_rows")
STAGE_TIMES = ("draft_seconds", "verify_seconds", "rollback_seconds")
GREEDY = {"mode": "greedy"}
SAMPLED = {"mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": 0.95}
SEED = 42
CACHE_PAIRS = {8700: 7700, 13288: 12288, 32768: 31768}


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path):
    stat = path.stat()
    return [stat.st_size, stat.st_ino, stat.st_mtime_ns]


def host_snapshot(pid=None):
    """Collect only numeric memory, paging and worker CPU/RSS observations."""
    values = {}
    try:
        line = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=3)
        for row in line.splitlines()[1:]:
            key, _, number = row.partition(":")
            if key in ("Pages free", "Pages wired down", "Pageouts", "Swapins", "Swapouts"):
                values[key] = int(number.strip().rstrip("."))
        if pid is not None:
            row = subprocess.check_output(
                ["/bin/ps", "-p", str(pid), "-o", "rss=,pcpu="], text=True, timeout=3
            ).split()
            values["worker_rss_kib"], values["worker_cpu_percent"] = int(row[0]), float(row[1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        values["incomplete"] = True
    return values


def record(stream, row):
    stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def timings(reply, keys):
    values = {key: reply.get(key) for key in keys}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in values.values()):
        raise ValueError("invalid_timings")
    return values


def validate(reply, prompt_count, limit, drafts, reused, verifier):
    if "error" in reply:
        raise ValueError("native_request_failed")
    ids = reply.get("generated_token_ids")
    sampled = reply.get("sampled_tokens")
    finish = reply.get("finish_reason")
    if not isinstance(ids, list) or not ids or not isinstance(reply.get("text"), str):
        raise ValueError("missing_output")
    if (reply.get("input_tokens") != prompt_count or reply.get("reused_input_tokens") != reused
            or type(sampled) is not int or sampled != len(ids) + (finish == "eos")
            or finish not in ("eos", "length") or sampled > limit
            or (finish == "length" and sampled != limit)):
        raise ValueError("invalid_generation_counts")
    mtp = reply.get("mtp") or {}
    stats = mtp.get("stats") or {}
    counts = {key: stats.get(key) for key in COUNTS}
    stage_times = timings(stats, STAGE_TIMES)
    if (any(type(value) is not int or value < 0 for value in counts.values())
            or counts["accepted"] > counts["proposed"]
            or counts["verified_rows"] < counts["accepted"]
            or mtp.get("enabled") is not bool(drafts) or mtp.get("drafts") != drafts
            or mtp.get("verification") != verifier
            or mtp.get("sampling") != "exact-rejection-v1"
            or (drafts and sampled > drafts + 1 and not counts["proposed"])
            or (not drafts and any((*counts.values(), *stage_times.values())))):
        raise ValueError("invalid_mtp_acknowledgement")
    native = timings(reply, ("prefill_seconds", "decode_seconds"))
    return {
        **native, "sampled_tokens": sampled, "output_tokens": len(ids), "finish": finish,
        "generated_sha256": digest_ids(ids),
        "text_sha256": hashlib.sha256(reply["text"].encode()).hexdigest(),
        "native_decode_tok_s": (len(ids) - 1) / native["decode_seconds"] if native["decode_seconds"] else None,
        "mtp_counts": counts, "mtp_seconds": stage_times,
    }


def make_vectors(worker, corpus, lengths, vocab_size):
    seeds = (
        "Review the following source. Explain how to implement a bounded queue with "
        "cancellation, synchronization, and tests.\n",
        "Audit a different subsystem. Compare rotation schedules and describe recovery tests.\n",
    )
    vectors = []
    for seed in seeds:
        result = worker.request({"operation": "tokenize", "text": "<|im_start|>user\n" + seed + corpus})
        ids = result.get("token_ids")
        if ("error" in result or not isinstance(ids, list) or len(ids) < max(lengths)
                or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)):
            raise ValueError("invalid_corpus_tokenization")
        vectors.append(ids[:max(lengths)])
    if vectors[0][:1024] == vectors[1][:1024]:
        raise ValueError("interleaved_corpus_prefix_matches")
    return vectors


def decode_suffix(worker):
    suffix = (
        "\nEnd of source excerpt. Write a detailed implementation guide for a Rust work queue. "
        "Cover ownership, cancellation, backpressure, shutdown, race conditions, and tests. "
        "Give at least six sections with practical examples."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    tokens = worker.tokenize(suffix)
    if not tokens:
        raise ValueError("invalid_decode_suffix")
    return tokens


def run(args, output, profile):
    opts = ARMS[args.arm].copy()
    if args.profile:
        name = "STARK_QWEN38_PROFILE_OUTPUT" if args.profile == "stage" else "STARK_NATIVE_PROFILE_OUTPUT"
        opts[name] = str(profile)
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    os.environ.update(opts)

    tracked = [path for path in args.model.iterdir() if path.is_file() and
               (path.suffix in (".json", ".safetensors") or path.name == "ngram_table.bin")]
    identities = {path: file_identity(path) for path in sorted(tracked)}
    weights = [path for path in tracked if path.suffix == ".safetensors"]
    if len(weights) != 55 or sum(path.stat().st_size for path in weights) != 106905155192:
        raise ValueError("unexpected_lily_weight_layout")
    corpus = args.corpus.read_bytes()
    prefill_lengths = args.prefill_length or (4096, 7700, 8700, 12288, 32768)
    cache_lengths = args.cache_length or tuple(CACHE_PAIRS)
    decode_lengths = (args.decode_length,) if args.decode_length else (4096, 32768)
    lengths = (*prefill_lengths, *cache_lengths, *decode_lengths)
    context = max(40960, max(lengths) + args.decode_tokens)
    record(output, {
        "type": "manifest", "schema": 1, "arm": args.arm, "suite": args.suite,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worker_sha256": hash_file(args.worker), "runner_sha256": hash_file(Path(__file__)),
        "cargo_lock_sha256": hash_file(ROOT / "Cargo.lock"),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(), "model_shards": len(weights),
        "model_bytes": sum(identities[path][0] for path in weights),
        "model_config_sha256": hash_file(args.model / "config.json"),
        "options": {k: "<private numeric JSONL>" if "OUTPUT" in k else v for k, v in opts.items()},
        "vision": not args.no_vision, "repetitions": args.reps, "context_tokens": context,
        "prefill_lengths": prefill_lengths, "cache_lengths": cache_lengths,
        "decode_lengths": decode_lengths, "interleave": not args.no_interleave,
        "decode_mtp_off": not args.no_decode_off,
        "decode_tokens": args.decode_tokens, "sampling": SAMPLED, "seed": SEED, "host": host_snapshot(),
    })
    with Worker(str(args.worker), str(args.model), prefix_cache=True, vision=not args.no_vision,
                timeout=args.timeout) as worker:
        ready = worker.ready
        cache = ready.get("prefix_cache") or {}
        mtp = ready.get("mtp") or {}
        image = ready.get("image_input") or {}
        chunk = int(opts["STARK_LILY_PREFILL_CHUNK"])
        verifier = mtp.get("verification")
        if (ready.get("mode") != "text-generation-v1" or cache.get("enabled") is not True
                or cache.get("chunk_tokens") != chunk or verifier != "batched-decode-exact-v1"
                or mtp.get("sampling") != "exact-rejection-v1"
                or (not args.no_vision and image.get("family_profile") != "qwen38-lily-retained-bf16-vision-v1")):
            raise ValueError("worker_readiness_mismatch")
        record(output, {"type": "ready", "chunk": chunk, "verifier": verifier,
                        "image_profile": image.get("family_profile"), "host": host_snapshot(worker.process.pid)})
        a, b = make_vectors(worker, corpus.decode("utf-8"), lengths, ready["vocab_size"])
        comparable = {}

        def observe(case, ids, repetition, *, limit=12, drafts=0, reused=0, scope=None,
                    sampling=GREEDY):
            before = host_snapshot(worker.process.pid)
            started = time.monotonic()
            reply = generate(worker, ids, context=context, max_new=limit, scope=scope,
                             sampling=sampling, seed=SEED, mtp_drafts=drafts)
            wall = time.monotonic() - started
            after = host_snapshot(worker.process.pid)
            try:
                row = validate(reply, len(ids), limit, drafts, reused, verifier)
            except ValueError as error:
                record(output, {"type": "request_failure", "case": case, "repetition": repetition,
                                "reason": str(error), "wall_seconds": wall, "host_before": before,
                                "host_after": after})
                raise
            input_hash = digest_ids(ids)
            key = (input_hash, limit, drafts)
            control = (row["generated_sha256"], row["text_sha256"], row["finish"],
                       row["sampled_tokens"], tuple(row["mtp_counts"].items()))
            matches = comparable.setdefault(key, control) == control
            record(output, {"type": "request", "case": case, "repetition": repetition,
                            "warmup": repetition == 0, "input_tokens": len(ids),
                            "input_sha256": input_hash, "requested_drafts": drafts,
                            "max_new_tokens": limit, "cached_tokens": reused, "wall_seconds": wall,
                            "host_before": before, "host_after": after,
                            "same_arm_output_and_work_match": matches, **row})
            if not matches:
                raise ValueError("same_arm_output_or_work_changed")

        if args.suite in ("all", "prefill"):
            for length in prefill_lengths:
                for repetition in range(args.reps + 1):
                    observe(f"cold-{length}", a[:length], repetition)
        if args.suite in ("all", "cache"):
            for long in cache_lengths:
                short = CACHE_PAIRS[long]
                for repetition in range(args.reps + 1):
                    scope = hashlib.sha256(f"{short}-{repetition}".encode()).hexdigest()[:32]
                    reused = (short - 1) // chunk * chunk
                    full_reuse = (long - 1) // chunk * chunk
                    observe(f"cache-{short}-start", a[:short], repetition, scope=scope)
                    observe(f"cache-{long}-append", a[:long], repetition, scope=scope, reused=reused)
                    if not args.no_interleave:
                        observe(f"cache-{long}-interleave", b[:long], repetition, scope=scope)
                        observe(f"cache-{long}-return", a[:long], repetition, scope=scope)
                    observe(f"cache-{long}-repeat", a[:long], repetition, scope=scope, reused=full_reuse)
                    if args.suite == "cache" or long not in prefill_lengths:
                        observe(f"cache-{long}-cold-control", a[:long], repetition)
        if args.suite in ("all", "decode"):
            suffix = decode_suffix(worker)
            for length in decode_lengths:
                if len(suffix) >= length:
                    raise ValueError("decode_prefix_too_short")
                ids = a[:length - len(suffix)] + suffix
                for repetition in range(args.reps + 1):
                    order = (2,) if args.no_decode_off else (0, 2)
                    if repetition % 2:
                        order = order[::-1]
                    for drafts in order:
                        observe(f"decode-{length}", ids, repetition, limit=args.decode_tokens,
                                drafts=drafts, sampling=SAMPLED)
    changed = [path.name for path, identity in identities.items() if file_identity(path) != identity]
    if changed:
        record(output, {"type": "model_identity_change", "files": changed})
        raise ValueError("model_files_changed")
    record(output, {"type": "complete", "host": host_snapshot(), "model_identity_unchanged": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "vendor/lily-core/src/qwen4exp/model.rs")
    parser.add_argument("--output", required=True, type=Path, help="New private JSONL measurement path")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--suite", choices=("all", "prefill", "cache", "decode"), default="all")
    parser.add_argument("--profile", choices=("stage", "native"), help="Separate diagnostic run")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--prefill-length", type=int, action="append", help="Repeat for multiple lengths")
    parser.add_argument("--cache-length", type=int, action="append", choices=CACHE_PAIRS,
                        help="Long side of a built-in grow/reuse pair. Repeat for multiple lengths.")
    parser.add_argument("--no-interleave", action="store_true", help="Skip the B and A-return cache requests")
    parser.add_argument("--decode-length", type=int)
    parser.add_argument("--decode-tokens", type=int, default=96)
    parser.add_argument("--no-decode-off", action="store_true", help="Measure only exact two-draft decode")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-vision", action="store_true", help="Explicitly use the text-only readiness path")
    args = parser.parse_args()
    if (args.reps < 1 or args.decode_tokens < 8 or any(length < 32 for length in args.prefill_length or ())
            or (args.decode_length is not None and args.decode_length < 32)):
        parser.error("use at least one repetition, eight generated tokens and 32 decode input tokens")
    args.worker, args.model = args.worker.resolve(strict=True), args.model.resolve(strict=True)
    output = args.output.absolute()
    profile = output.with_suffix(output.suffix + ".profile.jsonl")
    os.umask(0o077)
    with output.open("x", encoding="utf-8") as stream:
        try:
            run(args, stream, profile)
        except (Exception, SystemExit) as error:
            location = traceback.extract_tb(error.__traceback__)[-1]
            record(stream, {"type": "failure", "exception_type": type(error).__name__,
                            "source": Path(location.filename).name, "line": location.lineno})
            print("Native benchmark failed. See the private measurement file.", file=sys.stderr)
            return 1
    print("Native benchmark complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
