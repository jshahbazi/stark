"""Fixed-corpus Lily decode benchmark under Sushi's release-bench conditions.

Attempt 20261119-LILYBENCH-01. Drives one image-loaded `stark-generate` worker
with token-id requests on two fixed corpora: a seeded synthetic TypeScript
corpus (the code arm) and a caller-supplied public-domain novel (the prose
arm). Each arm walks Sushi's ladder: a cold 4,096-token request, then
8,192-token windows, with three greedy decode samples at every selected rung.
Every arm is repeated with MTP off and with each fixed draft depth.

The report keeps counts, timings, MTP work counters and SHA-256 hashes only.
Prompt text, token vectors and generated output are never written.

Provenance of the code-arm generator: `SplitMix64`, `code_chunks` and
`prose_chunks` are a Python port of `Rng`, `Corpus.nextChunk`,
`Corpus.writeModule` and `ProseChunks.next` in Sushi `src/bench.zig`
(commit 297d1e2cf360984340298cfcf86261abbc2eb223, file SHA-256
47de56fc083b72628861664b1412ac2cbb7fc8e729d5c04cf5b99f0448616ced).
Sushi is MIT licensed: Copyright (c) 2026 Theinruj Toranavikrai and
Copyright (c) 2026 David Dalcu. The notice is preserved in
`LICENSE.sushi-reference`. The ladder constants and plan follow the same file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import secrets
import statistics
import subprocess
import sys
import time
from pathlib import Path

ATTEMPT_ID = "20261119-LILYBENCH-01"
SCHEMA = "stark-lily-fixed-corpus-bench-v1"
SUSHI_COMMIT = "297d1e2cf360984340298cfcf86261abbc2eb223"
SUSHI_BENCH_SHA256 = "47de56fc083b72628861664b1412ac2cbb7fc8e729d5c04cf5b99f0448616ced"
# Sushi src/fixtures/prose_pride_and_prejudice.txt at SUSHI_COMMIT: Project
# Gutenberg #1342 with the Gutenberg header and footer removed.
SUSHI_PROSE_FIXTURE_SHA256 = "0a0ead8d7e5be2a1ffb4a929528ce245412af22d7848c723b43b2d91105e63f0"

CORPUS_SEED = 0x5A5AC0DE
WINDOW = 8192
FIRST_RUNG = 4096
MAX_RUNG = 131072
SAMPLES_PER_RUNG = 3
CORPUS_MARGIN = 928
CHUNK_BYTES = 24 * 1024
PROSE_DECODE_CAP = 128
DEFAULT_RUNGS = "4k,8k"
DEFAULT_DEPTHS = "off,1,2,3"
MAX_DRAFTS = 3
ARMS = ("code", "prose")
MASK64 = (1 << 64) - 1

# Request fields the benchmark owns. Pass-through options may not override them.
OWNED_REQUEST_KEYS = frozenset({
    "operation", "rendered_prompt", "input_token_ids", "context_tokens", "max_new_tokens",
    "seed", "sampling", "mtp", "mtp_drafts", "stop_token_ids", "stop", "stream",
    "request_id", "cache_scope",
})
REMOVED_ENV_PREFIXES = ("STARK_", "UZU_", "LILY_", "GGML_")
MTP_COUNTERS = ("rounds", "proposed", "accepted", "verified_rows")
MTP_TIMINGS = ("draft_seconds", "verify_seconds", "rollback_seconds")
EXPECTED_IMAGE_READY = {
    "version": "mediaref-v1",
    "family_profile": "qwen38-lily-retained-bf16-vision-v1",
    "model_source": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
    "model_revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
}


class BenchError(ValueError):
    """A failure whose message is a fixed code, safe to persist."""


# -- The corpus (port of Sushi src/bench.zig) ---------------------------------


class SplitMix64:
    def __init__(self, seed: int):
        self.s = seed & MASK64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def pick(self, items):
        return items[self.below(len(items))]


NOUNS = ("order", "invoice", "session", "token", "cart", "user", "ledger", "queue", "cache", "route",
         "tenant", "webhook", "report", "schedule", "policy", "metric", "stream", "batch", "index", "profile")
VERBS = ("load", "save", "parse", "merge", "drop", "fetch", "retry", "emit", "build", "verify", "resolve",
         "flush", "rotate", "sync", "encode")
TS_TYPES = ("string", "number", "boolean", "Date", "Uint8Array", "Record<string, unknown>", "string[]",
            "number | null")
KEY_SEPARATORS = (":", "/", "-", "|")


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:]


def _write_module(rng: SplitMix64, n: int, out: list[str]) -> None:
    # Every random draw happens in the same order as the Zig source, which
    # evaluates the argument tuple of each print left to right.
    noun = rng.pick(NOUNS)
    other = rng.pick(NOUNS)
    noun_c = _cap(noun)
    other_c = _cap(other)
    w = out.append
    w(f"// src/{noun}/{noun}-{n}.ts\n")
    first = rng.below(900)
    second = rng.below(900)
    w(f"import {{ {other_c}Record{first}, {other_c}Options }} from \"../{other}/{other}-{second}\";\n")
    w("import type { Logger } from \"../core/logger\";\n\n")
    w(f"export interface {noun_c}Record{n} {{\n  id: string;\n")
    for _ in range(3 + rng.below(4)):
        verb = rng.pick(VERBS)
        field_noun = rng.pick(NOUNS)
        ts_type = rng.pick(TS_TYPES)
        w(f"  {verb}{field_noun}: {ts_type};\n")
    w("}\n\n")
    w(f"export class {noun_c}Service{n} {{\n  private readonly cache = new Map<string, {noun_c}Record{n}>();\n")
    w(f"  constructor(private readonly log: Logger, private readonly limit = {16 + rng.below(4096)}) {{}}\n")
    for _ in range(3 + rng.below(3)):
        verb = rng.pick(VERBS)
        verb_c = _cap(verb)
        w(f"\n  async {verb}{noun_c}(id: string, attempt = {rng.below(5)}): "
          f"Promise<{noun_c}Record{n} | undefined> {{\n")
        w(f"    const hit = this.cache.get(id);\n    if (hit && attempt < {1 + rng.below(9)}) return hit;\n")
        w(f"    try {{\n      const next = await {verb}{other_c}(id, {{ timeoutMs: {50 * (1 + rng.below(200))} }});\n")
        w("      this.cache.set(id, next);\n"
          f"      this.log.info(`{verb_c}{noun_c} ${{id}} size=${{this.cache.size}}`);\n"
          "      return next;\n")
        w(f"    }} catch (err) {{\n      if (attempt >= {2 + rng.below(6)}) "
          f"throw new Error(`{verb} failed for ${{id}}: ${{String(err)}}`);\n")
        w(f"      return this.{verb}{noun_c}(id, attempt + 1);\n    }}\n  }}\n")
    w("}\n\n")
    w(f"export function {noun}Key{n}(parts: string[]): string {{\n"
      f"  return parts.map((p) => p.trim().toLowerCase()).filter(Boolean).join(\"{rng.pick(KEY_SEPARATORS)}\");\n}}")


def code_chunks(seed: int = CORPUS_SEED):
    """Yield Sushi's code-arm chunks: whole modules, each chunk at least 24 KiB."""
    rng = SplitMix64(seed)
    module = 0
    while True:
        parts: list[str] = []
        size = 0
        while size < CHUNK_BYTES:
            before = len(parts)
            _write_module(rng, module, parts)
            module += 1
            parts.append("\n\n")
            size += sum(len(p.encode()) for p in parts[before:])
        yield "".join(parts)


def prose_chunks(data: bytes):
    """Yield Sushi's prose pieces: at most 24 KiB, cut after the last blank line."""
    pos = 0
    while pos < len(data):
        end = min(pos + CHUNK_BYTES, len(data))
        if end < len(data):
            at = data.rfind(b"\n\n", pos, end)
            if at != -1:
                end = at + 2
        piece = data[pos:end]
        pos = end
        try:
            yield piece.decode("utf-8")
        except UnicodeDecodeError:
            raise BenchError("prose_chunk_not_utf8") from None


# -- The plan -------------------------------------------------------------------


def valid_rung(length: int) -> bool:
    return length == FIRST_RUNG or (WINDOW <= length <= MAX_RUNG and length % WINDOW == 0)


def parse_rungs(text: str) -> list[int]:
    values = set()
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            raise BenchError("invalid_rungs")
        kilo = item[-1] in "kK"
        digits = item[:-1] if kilo else item
        if not digits.isdigit():
            raise BenchError("invalid_rungs")
        length = int(digits) * (1024 if kilo else 1)
        if not valid_rung(length):
            raise BenchError("invalid_rungs")
        values.add(length)
    return sorted(values)


def parse_depths(text: str) -> list[str]:
    depths: list[str] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if item == "0":
            item = "off"
        if item not in {"off", "default"} and not (item.isdigit() and 1 <= int(item) <= MAX_DRAFTS):
            raise BenchError("invalid_depths")
        if item in depths:
            raise BenchError("duplicate_depth")
        depths.append(item)
    if not depths:
        raise BenchError("invalid_depths")
    return depths


def parse_arms(text: str) -> list[str]:
    arms = [item.strip() for item in text.split(",") if item.strip()]
    if not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise BenchError("invalid_arms")
    return [arm for arm in ARMS if arm in arms]


def parse_request_options(items: list[str] | None) -> dict:
    options: dict = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise BenchError("invalid_request_option")
        if key in OWNED_REQUEST_KEYS:
            raise BenchError("request_option_owned_by_benchmark")
        if key in options:
            raise BenchError("duplicate_request_option")
        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            raise BenchError("invalid_request_option") from None
    return options


def arm_decode_tokens(arm: str, tokens: int) -> int:
    return min(tokens, PROSE_DECODE_CAP) if arm == "prose" else tokens


def corpus_tokens_needed(rungs: list[int]) -> int:
    return rungs[-1] + CORPUS_MARGIN


def plan(decode_tokens: int, rungs: list[int]) -> list[dict]:
    """Sushi's plan: the cold 4k request, then every 8k window to the largest rung.

    A listed rung decodes `decode_tokens` and is followed at once by its two
    repeats; a window that is not a rung decodes one token.
    """
    requests = []
    previous = 0
    length = FIRST_RUNG
    step = 0
    while True:
        decode = length in rungs
        requests.append({"prompt_len": length, "max_tokens": decode_tokens if decode else 1,
                         "kind": "ladder", "base": previous, "sample": 1 if decode else 0})
        if decode:
            for index in range(2, SAMPLES_PER_RUNG + 1):
                requests.append({"prompt_len": length, "max_tokens": decode_tokens,
                                 "kind": "repeat", "base": length, "sample": index})
        previous = length
        if length >= rungs[-1]:
            break
        step += 1
        length = WINDOW * step
    return requests


# -- Hashes, validation and statistics ------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_sha256(ids: list[int]) -> str:
    return sha256_bytes(json.dumps(ids, separators=(",", ":")).encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _count(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def depth_request_fields(depth: str) -> dict:
    if depth == "default":
        return {}
    return {"mtp_drafts": 0 if depth == "off" else int(depth)}


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def sample_record(result, request: dict, depth: str, eos_ids: list[int], vocab_size: int) -> dict:
    """Reduce one worker reply to counts, timings and hashes."""
    item = {
        "prompt_len": request["prompt_len"], "kind": request["kind"], "base": request["base"],
        "sample": request["sample"], "max_tokens": request["max_tokens"], "depth": depth,
    }
    if not isinstance(result, dict) or "error" in result:
        error = result.get("error") if isinstance(result, dict) else None
        code = error.get("code") if isinstance(error, dict) else error
        item["error"] = code if isinstance(code, str) and code.isidentifier() else "unknown"
        item["valid"] = False
        item["invalid_reason"] = "worker_error"
        return item
    generated = result.get("generated_token_ids")
    ids_ok = isinstance(generated, list) and all(type(t) is int and 0 <= t < vocab_size for t in generated)
    sampled = _count(result.get("sampled_tokens"))
    finish = result.get("finish_reason")
    item.update({
        "input_tokens": _count(result.get("input_tokens")),
        "reused_input_tokens": _count(result.get("reused_input_tokens")) or 0,
        "sampled_tokens": sampled,
        "output_tokens": len(generated) if ids_ok else None,
        "finish_reason": finish if finish in {"eos", "stop_token", "stop_sequence", "length", "context"} else "other",
        "stopped_on_model_eos": result.get("stop_token_id") in eos_ids,
        "prefill_seconds": _number(result.get("prefill_seconds")),
        "decode_seconds": _number(result.get("decode_seconds")),
        "generation_seconds": _number(result.get("generation_seconds")),
        "output_sha256": ids_sha256(generated) if ids_ok else None,
    })
    reason = None
    if not ids_ok or sampled is None or item["input_tokens"] != request["prompt_len"]:
        reason = "malformed_reply"
    elif finish != "length" or sampled < request["max_tokens"]:
        reason = "early_stop"
    elif sampled != request["max_tokens"]:
        reason = "count_mismatch"
    new_tokens = (item["input_tokens"] or 0) - item["reused_input_tokens"]
    if item["prefill_seconds"]:
        item["prefill_tokens_per_second"] = new_tokens / item["prefill_seconds"]
    if item["decode_seconds"] and sampled and sampled > 1:
        # Stark's documented rate: sampling steps after the first token.
        item["decode_tokens_per_second"] = (sampled - 1) / item["decode_seconds"]
        # Sushi's formula: completion tokens over the decode time after the first token.
        item["sushi_formula_tokens_per_second"] = sampled / item["decode_seconds"]
    ack = result.get("mtp")
    if isinstance(ack, dict):
        stats = ack.get("stats") if isinstance(ack.get("stats"), dict) else {}
        mtp = {"enabled": ack.get("enabled") is True, "drafts": _count(ack.get("drafts")),
               "sampling": ack.get("sampling") if isinstance(ack.get("sampling"), str) else None,
               "verification": ack.get("verification") if isinstance(ack.get("verification"), str) else None}
        for key in MTP_COUNTERS:
            mtp[key] = _count(stats.get(key))
        for key in MTP_TIMINGS:
            mtp[key] = _number(stats.get(key))
        item["mtp"] = mtp
        if depth != "default" and reason is None:
            expected = 0 if depth == "off" else int(depth)
            if mtp["drafts"] != expected or mtp["enabled"] is not (expected > 0):
                reason = "mtp_acknowledgement_mismatch"
    elif depth not in {"off", "default"} and reason is None:
        reason = "mtp_acknowledgement_missing"
    if reason is None and request["max_tokens"] > 1 and "decode_tokens_per_second" not in item:
        reason = "decode_timing_unavailable"
    item["valid"] = reason is None
    if reason:
        item["invalid_reason"] = reason
    return item


def summarize(samples: list[dict]) -> list[dict]:
    rows = []
    keys = []
    for s in samples:
        if s.get("warmup") or s["max_tokens"] <= 1:
            continue
        key = (s["arm"], s["depth"], s["prompt_len"])
        if key not in keys:
            keys.append(key)
    for arm, depth, rung in keys:
        group = [s for s in samples if not s.get("warmup") and s["arm"] == arm and s["depth"] == depth
                 and s["prompt_len"] == rung and s["max_tokens"] > 1]
        valid = [s for s in group if s["valid"]]
        row = {
            "arm": arm, "depth": depth, "rung": rung, "decode_tokens": group[0]["max_tokens"],
            "samples": len(group), "valid_samples": len(valid),
            "early_stops": sum(s.get("invalid_reason") == "early_stop" for s in group),
            "invalid_reasons": sorted({s["invalid_reason"] for s in group if "invalid_reason" in s}),
            "decode_tokens_per_second": [round(s["decode_tokens_per_second"], 3) for s in valid],
            "decode_tokens_per_second_median": None,
            "sushi_formula_tokens_per_second_median": None,
            "reused_input_tokens": [s.get("reused_input_tokens") for s in group],
            "output_sha256": sorted({s["output_sha256"] for s in valid if s.get("output_sha256")}),
        }
        if valid:
            row["decode_tokens_per_second_median"] = statistics.median(s["decode_tokens_per_second"] for s in valid)
            row["sushi_formula_tokens_per_second_median"] = statistics.median(
                s["sushi_formula_tokens_per_second"] for s in valid)
            row["ms_per_token_median"] = 1000.0 / row["decode_tokens_per_second_median"]
            mtp = [s["mtp"] for s in valid if isinstance(s.get("mtp"), dict)]
            if mtp and all(all(m[k] is not None for k in MTP_COUNTERS + MTP_TIMINGS) for m in mtp):
                totals = {k: sum(m[k] for m in mtp) for k in MTP_COUNTERS + MTP_TIMINGS}
                rounds = totals["rounds"]
                totals["acceptance"] = totals["accepted"] / totals["proposed"] if totals["proposed"] else None
                totals["tokens_per_round"] = ((totals["accepted"] + rounds) / rounds) if rounds else None
                for key in MTP_TIMINGS:
                    totals[key.replace("_seconds", "_ms_per_round")] = (
                        1000.0 * totals[key] / rounds if rounds else None)
                row["mtp_totals"] = totals
        rows.append(row)
    return rows


# -- The run --------------------------------------------------------------------


def other_workers(own: set[int] | None = None) -> list[int]:
    out = subprocess.run(["pgrep", "-x", "stark-generate"], capture_output=True, text=True).stdout
    own = own or set()
    return [int(x) for x in out.split() if x.strip().isdigit() and int(x) not in own]


def default_worker_factory():
    from stark_convert.chat import Worker

    class TextWorker(Worker):
        """Image-loaded native launch without Python's extra image-request check.

        Same approach as scripts/benchmark_lily_release_baseline.py: the native
        worker still receives --vision-staging-root; hiding the host media
        handle only during initial readiness changes no native load option.
        """

        def read(self, *a, **k):
            media = self.media
            initial = not self._ready_seen
            if initial:
                self.media = None
            try:
                return super().read(*a, **k)
            finally:
                if initial:
                    self.media = media

    return TextWorker


@contextlib.contextmanager
def scrubbed_environment(enabled: bool):
    removed = {}
    if enabled:
        for name in tuple(os.environ):
            if name.startswith(REMOVED_ENV_PREFIXES):
                removed[name] = os.environ.pop(name)
    try:
        yield sorted(removed)
    finally:
        os.environ.update(removed)


def tokenize_corpus(worker, chunks, need: int) -> tuple[list[int], int, str]:
    """Tokenize chunk by chunk, as Sushi does, until `need` ids exist."""
    ids: list[int] = []
    digest = hashlib.sha256()
    count = 0
    for chunk in chunks:
        if len(ids) >= need:
            break
        reply = worker.request({"operation": "tokenize", "text": chunk, "add_special_tokens": False})
        piece = reply.get("token_ids") if isinstance(reply, dict) else None
        if not isinstance(piece, list) or not piece or not all(type(t) is int and t >= 0 for t in piece):
            raise BenchError("corpus_tokenization_failed")
        ids.extend(piece)
        digest.update(chunk.encode())
        count += 1
    if len(ids) < need:
        raise BenchError("corpus_too_short")
    return ids, count, digest.hexdigest()


def run(args: argparse.Namespace, worker_factory=None, host_sampler_factory=None, clock=time.monotonic) -> dict:
    rungs = parse_rungs(args.rungs)
    depths = parse_depths(args.depths)
    arms = parse_arms(args.arms)
    options = parse_request_options(args.request_option)
    if not 2 <= args.tokens <= 4096:
        raise BenchError("invalid_tokens")
    if not 0 <= args.warmup_tokens <= 256:
        raise BenchError("invalid_warmup_tokens")
    if args.prefix not in {"scoped", "fresh"}:
        raise BenchError("invalid_prefix_mode")
    if "prose" in arms and args.prose_corpus is None:
        raise BenchError("prose_corpus_required")
    if args.output.exists():
        raise BenchError("output_exists")
    prose_bytes = args.prose_corpus.read_bytes() if "prose" in arms else None
    report = {
        "schema": SCHEMA,
        "attempt": ATTEMPT_ID,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_label": args.label,
        "sushi_reference": {
            "commit": SUSHI_COMMIT, "file": "src/bench.zig", "file_sha256": SUSHI_BENCH_SHA256,
            "corpus_seed": f"0x{args.corpus_seed:x}", "license": "MIT, see LICENSE.sushi-reference",
        },
        "protocol": {
            "transport": "stark-generate JSON lines, input_token_ids, non-streaming",
            "sampling": "greedy", "ignore_eos": False,
            "eos_policy": "model EOS always honoured by the worker; an early stop marks the sample invalid",
            "rungs": rungs, "samples_per_rung": SAMPLES_PER_RUNG, "window": WINDOW,
            "decode_tokens": {arm: arm_decode_tokens(arm, args.tokens) for arm in arms},
            "depths": depths, "request_options": options, "prefix_mode": args.prefix,
            "warmup_tokens": args.warmup_tokens, "host_observation": bool(args.observe_host),
            "arms": arms,
        },
        "samples": [], "arms": {}, "failures": [],
    }
    factory = worker_factory or default_worker_factory()
    with scrubbed_environment(not args.keep_environment) as removed:
        report["environment_names_removed"] = removed
        if args.worker is not None:
            report["worker_sha256"] = sha256_file(args.worker)
        started = clock()
        worker = factory(args.worker, args.model, args.timeout)
    try:
        ready = worker.read()
        report["load_seconds"] = clock() - started
        if not isinstance(ready, dict) or ready.get("ready") is not True:
            raise BenchError("worker_not_ready")
        image = ready.get("image_input")
        report["ready"] = {k: ready.get(k) for k in ("mode", "version", "mtp", "max_context_tokens",
                                                      "prefix_cache", "state_policy")}
        report["image_loaded"] = isinstance(image, dict)
        report["image_family_profile"] = image.get("family_profile") if isinstance(image, dict) else None
        if args.require_lily_image and (not isinstance(image, dict) or any(
                image.get(k) != v for k, v in EXPECTED_IMAGE_READY.items())):
            raise BenchError("required_image_ready_missing")
        eos_ids = [t for t in ready.get("eos_token_ids", []) if type(t) is int]
        vocab_size = ready.get("vocab_size") if type(ready.get("vocab_size")) is int else 1 << 31
        prefix_enabled = isinstance(ready.get("prefix_cache"), dict) and ready["prefix_cache"].get("enabled") is True
        max_context = ready.get("max_context_tokens")
        sampler_cm = (host_sampler_factory(getattr(getattr(worker, "process", None), "pid", None))
                      if args.observe_host and host_sampler_factory else contextlib.nullcontext())
        with sampler_cm as host:
            for arm in arms:
                decode_tokens = arm_decode_tokens(arm, args.tokens)
                context_tokens = rungs[-1] + decode_tokens
                if type(max_context) is int and context_tokens > max_context:
                    raise BenchError("context_exceeds_worker_limit")
                chunks = code_chunks(args.corpus_seed) if arm == "code" else prose_chunks(prose_bytes)
                ids, chunk_count, text_sha = tokenize_corpus(worker, chunks, corpus_tokens_needed(rungs))
                arm_info = {
                    "chunks_tokenized": chunk_count, "chunk_text_sha256": text_sha, "corpus_tokens": len(ids),
                    "corpus_ids_sha256": ids_sha256(ids), "context_tokens": context_tokens,
                    "prompt_ids_sha256": {str(r): ids_sha256(ids[:r]) for r in rungs},
                }
                if arm == "prose":
                    arm_info["prose_file_sha256"] = sha256_bytes(prose_bytes)
                    arm_info["prose_file_bytes"] = len(prose_bytes)
                    arm_info["matches_sushi_fixture"] = arm_info["prose_file_sha256"] == SUSHI_PROSE_FIXTURE_SHA256
                report["arms"][arm] = arm_info
                requests = plan(decode_tokens, rungs)
                for depth in depths:
                    if prefix_enabled:
                        worker.request({"operation": "reset_cache"})
                    if args.warmup_tokens:
                        warm = {"operation": "generate", "input_token_ids": ids[:512],
                                "context_tokens": 512 + args.warmup_tokens,
                                "max_new_tokens": args.warmup_tokens, "sampling": {"mode": "greedy"},
                                "seed": args.seed, **depth_request_fields(depth), **options}
                        reply = worker.request(warm)
                        report["samples"].append({
                            "arm": arm, "depth": depth, "warmup": True, "prompt_len": 512, "kind": "warmup",
                            "max_tokens": args.warmup_tokens,
                            "valid": isinstance(reply, dict) and "error" not in reply})
                        if prefix_enabled:
                            worker.request({"operation": "reset_cache"})
                    scope = secrets.token_hex(16) if prefix_enabled and args.prefix == "scoped" else None
                    for request in requests:
                        payload = {"operation": "generate", "input_token_ids": ids[:request["prompt_len"]],
                                   "context_tokens": context_tokens, "max_new_tokens": request["max_tokens"],
                                   "sampling": {"mode": "greedy"}, "seed": args.seed,
                                   **depth_request_fields(depth), **options}
                        if scope is not None:
                            payload["cache_scope"] = scope
                        marker = host.begin_request() if host is not None else None
                        t0 = clock()
                        reply = worker.request(payload)
                        wall = clock() - t0
                        if marker is not None:
                            host.end_request(marker)
                        item = sample_record(reply, request, depth, eos_ids, vocab_size)
                        item["arm"] = arm
                        item["wall_seconds"] = wall
                        if marker is not None:
                            item["host_marker"] = marker
                        report["samples"].append(item)
                        if args.progress:
                            print(json.dumps({"stage": "sample", "arm": arm, "depth": depth,
                                              "prompt_len": item["prompt_len"], "kind": item["kind"],
                                              "valid": item["valid"],
                                              "decode_tokens_per_second": item.get("decode_tokens_per_second")}),
                                  flush=True)
                        if item.get("invalid_reason") in {"worker_error", "malformed_reply"}:
                            raise BenchError("generation_failed")
        if host is not None:
            for item in report["samples"]:
                marker = item.pop("host_marker", None)
                if marker is not None:
                    item["host"] = host.request_result(marker)
            report["host_summary"] = host.summary()
    finally:
        with contextlib.suppress(Exception):
            worker.close()
        for item in report["samples"]:
            item.pop("host_marker", None)
    measured = [s for s in report["samples"] if not s.get("warmup") and s["max_tokens"] > 1]
    report["summary"] = summarize(report["samples"])
    report["early_stop_count"] = sum(s.get("invalid_reason") == "early_stop" for s in measured)
    report["decode_sample_count"] = len(measured)
    report["finish_reason_counts"] = {}
    for s in measured:
        reason = s.get("finish_reason", "error")
        report["finish_reason_counts"][reason] = report["finish_reason_counts"].get(reason, 0) + 1
    report["passed"] = bool(measured) and all(s["valid"] for s in measured)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--arms", default="code,prose", help="code, prose or both (default)")
    p.add_argument("--prose-corpus", type=Path, help="UTF-8 public-domain text for the prose arm")
    p.add_argument("--rungs", default=DEFAULT_RUNGS, help="4k or multiples of 8k up to 128k (default 4k,8k)")
    p.add_argument("--tokens", type=int, default=256, help="decoded tokens per sample; prose decodes at most 128")
    p.add_argument("--depths", default=DEFAULT_DEPTHS, help="off, 1, 2, 3 or default (worker's own policy)")
    p.add_argument("--request-option", action="append", metavar="KEY=JSON",
                   help="extra generate field sent on every request, for example a draft selector")
    p.add_argument("--prefix", default="scoped", choices=("scoped", "fresh"),
                   help="scoped: one cache scope per ladder so windows can reuse the previous prompt")
    p.add_argument("--warmup-tokens", type=int, default=32,
                   help="one unmeasured 512-token request per depth before its ladder; 0 disables")
    p.add_argument("--corpus-seed", type=lambda v: int(v, 0), default=CORPUS_SEED)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--label", default="screening", help="evidence label stored in the report")
    p.add_argument("--observe-host", action="store_true", help="per-request host counters (hybrid_host_metrics)")
    p.add_argument("--keep-environment", action="store_true", help="do not remove STARK_/UZU_/LILY_/GGML_ variables")
    p.add_argument("--no-require-lily-image", dest="require_lily_image", action="store_false")
    p.add_argument("--plan-only", action="store_true", help="print the request plan and exit without a worker")
    p.add_argument("--progress", action="store_true", help="print one count/timing line per sample")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.plan_only:
            rungs = parse_rungs(args.rungs)
            arms = parse_arms(args.arms)
            print(json.dumps({arm: plan(arm_decode_tokens(arm, args.tokens), rungs) for arm in arms}, indent=1))
            return 0
        if args.model is None or args.worker is None or args.output is None:
            raise BenchError("model_worker_and_output_required")
        if args.output.exists():
            raise BenchError("output_exists")
        if other_workers():
            raise BenchError("other_worker_present")
        os.umask(0o077)
        host_factory = None
        if args.observe_host:
            from hybrid_host_metrics import HostMetricsSampler
            host_factory = HostMetricsSampler
        report = run(args, host_sampler_factory=host_factory)
        repo = Path(__file__).resolve().parents[1]
        with contextlib.suppress(Exception):
            git = ["git", "-C", str(repo)]
            report["source_commit"] = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
            report["source_dirty"] = bool(subprocess.check_output([*git, "status", "--porcelain"], text=True).strip())
        report["runner_sha256"] = sha256_file(Path(__file__))
        report["worker_unchanged"] = sha256_file(args.worker) == report.get("worker_sha256")
        write_report(args.output, report)
        print(json.dumps({"passed": report["passed"], "samples": report["decode_sample_count"],
                          "early_stops": report["early_stop_count"]}))
        return 0 if report["passed"] else 1
    except BenchError as error:
        print(json.dumps({"passed": False, "stage": str(error)}))
        return 2
    except Exception as error:  # noqa: BLE001 - persist the type only, never a message.
        print(json.dumps({"passed": False, "failure_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
