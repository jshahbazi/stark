#!/usr/bin/env python3
"""Compare two native GLM score workers on private in-memory public text inputs.

Only aggregate numerical errors, protocol counts, and provenance hashes are
written. Prompts, token IDs, logits, and log probabilities remain in memory.
This narrow comparison is not a broad answer-quality qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from benchmark_hybrid import checkout_identity, sha, weight_identity
from stark_convert.chat import load_template
from tokenizers import Tokenizer

CONTEXT = 512
MAX_CONTEXT = 4096
MMA_PROBE_ENV = "STARK_GLM53_MMA_SCORE_PROBE"
MMA_PROBE_LIMIT = 65_536
U64_MAX = (1 << 64) - 1
ARCHIVED_REFERENCE_SHA256 = "190930839f1525722a6f90e2b27c12171a9ddb6af6e5e2bb0595ed319b5d9c3d"
PROMPTS = [
    ("user", "Explain why a stable sort preserves the original order of equal keys, and give a short example."),
    ("user", "Solve 17 times 23 without a calculator. Show the arithmetic steps and verify the product."),
    ("user", "Describe how a database transaction provides atomicity and isolation in a small application."),
    ("user", "Write a Python function that merges two sorted integer lists and explain its running time."),
    ("user", "What causes seasons on Earth? Explain the role of axial tilt and distinguish it from distance."),
    ("user", "Compare breadth-first search and depth-first search, including when each is useful."),
    ("partial", "We can derive the result by first observing that"),
    ("partial", "A concise implementation would start with"),
]
CANDIDATE_WORDS = [
    " the", " a", " and", " is", " of", " to", " in", " for",
    " with", " that", " as", " by", " from", " can", " will", " when",
    " because", " therefore", " result", " example", " function", " return",
    " value", " true", " false", " however", " first", " second", " while",
    " data", " model", " code", " Python", " token", " answer", " use",
    " should", " must", " would", " may", " not", " at", " on", " this",
    " these", " each", " also", " then", " after", " before", " we", " they",
]


class CheckFailed(RuntimeError):
    pass


def require(ok: bool, code: str) -> None:
    if not ok:
        raise CheckFailed(code)


def validate_context_tokens(context_tokens: int) -> None:
    require(type(context_tokens) is int and 1 <= context_tokens <= MAX_CONTEXT,
            "context_tokens_invalid")


def token_hash(rows: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()


class PipeWorker:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    async def reply(self) -> dict:
        line = await asyncio.wait_for(self.process.stdout.readline(), 600)
        require(bool(line) and len(line) <= 1_048_576 and line.endswith(b"\n"), "worker_reply_boundary_invalid")
        value = json.loads(line)
        require(isinstance(value, dict), "worker_reply_invalid")
        return value

    async def request(self, value: dict) -> dict:
        data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        require(len(data) <= 1_048_576, "worker_request_boundary_invalid")
        self.process.stdin.write(data)
        await self.process.stdin.drain()
        return await self.reply()


async def run_worker(binary: Path, model: Path, inputs: list[list[int]], candidates: list[int],
                     inherited_flags: bool, probe_path: Path | None = None,
                     context_tokens: int = CONTEXT) -> list[list[float]]:
    validate_context_tokens(context_tokens)
    require(all(1 <= len(ids) <= context_tokens for ids in inputs), "input_length_out_of_range")
    env = dict(os.environ)
    if not inherited_flags:
        env = {k: v for k, v in env.items() if not k.startswith(("STARK_", "UZU_"))}
    if inherited_flags and probe_path is not None:
        env[MMA_PROBE_ENV] = str(probe_path)
    process = await asyncio.create_subprocess_exec(
        str(binary), "--score-model-path", str(model), "--consecutive-cache",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, env=env, limit=1_048_577,
    )
    clean = False
    try:
        worker = PipeWorker(process)
        ready = await worker.reply()
        require(ready.get("ready") is True and ready.get("cache_enabled") is True,
                "worker_readiness_invalid")
        rows = []
        for ids in inputs:
            result = await worker.request({
                "operation": "score", "input_token_ids": ids,
                "candidate_token_ids": candidates, "context_tokens": context_tokens,
                "temperature": 1.0,
            })
            require("error" not in result and result.get("input_tokens") == len(ids), "score_request_failed")
            require(result.get("candidate_token_ids") == candidates, "candidate_coverage_invalid")
            logps = result.get("log_probabilities")
            require(isinstance(logps, list) and len(logps) == len(candidates), "logprob_coverage_invalid")
            require(all(type(x) in (float, int) and math.isfinite(x) and x <= 1e-6 for x in logps),
                    "logprob_values_invalid")
            rows.append([float(x) for x in logps])
        clean = True
        return rows
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 15)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        extra = await process.stdout.read(1_048_577)
        if clean:
            require(process.returncode == 0 and not extra, "worker_shutdown_boundary_invalid")


def requested_probe_path() -> Path | None:
    value = os.environ.get(MMA_PROBE_ENV)
    if value is None:
        return None
    path = Path(value).expanduser().absolute()
    require(not path.exists() and not path.is_symlink(), "mma_probe_path_not_fresh")
    require(path.parent.is_dir(), "mma_probe_parent_invalid")
    return path


def read_mma_probe(path: Path, expected_rows: int = len(PROMPTS)) -> dict:
    require(type(expected_rows) is int and expected_rows > 0, "mma_probe_expected_row_count_invalid")
    try:
        info = path.lstat()
        require(path.is_file() and not path.is_symlink(), "mma_probe_file_invalid")
        require(info.st_mode & 0o777 == 0o600, "mma_probe_permissions_invalid")
        require(info.st_size <= MMA_PROBE_LIMIT, "mma_probe_too_large")
        raw = path.read_bytes()
    except CheckFailed:
        raise
    except OSError:
        raise CheckFailed("mma_probe_missing_or_unreadable") from None
    require(len(raw) <= MMA_PROBE_LIMIT, "mma_probe_too_large")
    require(raw.endswith(b"\n"), "mma_probe_jsonl_boundary_invalid")
    try:
        lines = raw.splitlines()
        require(len(lines) == expected_rows, "mma_probe_row_count_invalid")
        rows = [json.loads(line) for line in lines]
    except CheckFailed:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckFailed("mma_probe_json_invalid") from None
    calls = 0
    weight_bytes = 0
    q5_calls = 0
    q5_weight_bytes = 0
    legacy_keys = {"success", "encoded_dispatch_calls", "encoded_weight_bytes"}
    current_keys = legacy_keys | {"encoded_q5_dispatch_calls", "encoded_q5_weight_bytes"}
    schema = None
    mma_enabled = os.environ.get("STARK_GLM53_PREFILL_MMA") == "1"
    require_q5 = mma_enabled and os.environ.get("STARK_GLM53_PREFILL_CHUNK") == "8"
    for row in rows:
        require(isinstance(row, dict) and set(row) in (legacy_keys, current_keys), "mma_probe_schema_invalid")
        keys = set(row)
        if schema is None:
            schema = keys
        require(keys == schema, "mma_probe_schema_changed")
        require(not require_q5 or keys == current_keys, "mma_probe_q5_schema_required")
        require(row["success"] is True, "mma_probe_request_failed")
        for key in keys - {"success"}:
            value = row[key]
            require(type(value) is int and 0 <= value <= U64_MAX, "mma_probe_counter_invalid")
        # Archived V47 three-key records counted Q6 only. Current totals are
        # Q5 + Q6; the explicit Q5 subset lets us derive Q6 without ambiguity.
        row_q5_calls = row.get("encoded_q5_dispatch_calls", 0)
        row_q5_bytes = row.get("encoded_q5_weight_bytes", 0)
        require(row_q5_calls <= row["encoded_dispatch_calls"]
                and row_q5_bytes <= row["encoded_weight_bytes"], "mma_probe_subset_invalid")
        row_q6_calls = row["encoded_dispatch_calls"] - row_q5_calls
        row_q6_bytes = row["encoded_weight_bytes"] - row_q5_bytes
        if mma_enabled:
            require(row["encoded_dispatch_calls"] > 0, "mma_probe_dispatch_missing")
        if require_q5:
            require(row_q5_calls > 0 and row_q5_bytes > 0, "mma_probe_q5_dispatch_missing")
            require(row_q6_calls > 0 and row_q6_bytes > 0, "mma_probe_q6_dispatch_missing")
        calls += row["encoded_dispatch_calls"]
        weight_bytes += row["encoded_weight_bytes"]
        q5_calls += row_q5_calls
        q5_weight_bytes += row_q5_bytes
    require(calls <= U64_MAX and weight_bytes <= U64_MAX, "mma_probe_total_overflow")
    return {
        "request_count": len(rows),
        "encoded_dispatch_calls": calls,
        "encoded_weight_bytes": weight_bytes,
        "encoded_q5_dispatch_calls": q5_calls,
        "encoded_q5_weight_bytes": q5_weight_bytes,
        "encoded_q6_dispatch_calls": calls - q5_calls,
        "encoded_q6_weight_bytes": weight_bytes - q5_weight_bytes,
        "probe_sha256": hashlib.sha256(raw).hexdigest(),
        "qualification": "Software encoded-dispatch counters; successful records follow the request GPU completion fence and reset. Counts do not measure hardware utilization.",
    }


def collect_inputs(model: Path) -> tuple[list[list[int]], list[int], str]:
    codec, template = load_template(model)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    inputs = []
    for kind, text in PROMPTS:
        if kind == "user":
            rendered = template.render(
                messages=[{"role": "user", "content": text}], tools=None,
                add_generation_prompt=True, enable_thinking=True,
                bos_token=codec.get("bos_token") or "", eos_token=codec.get("eos_token") or "",
            )
        else:
            rendered = template.render(
                messages=[{"role": "user", "content": "Continue this explanation."}], tools=None,
                add_generation_prompt=True, enable_thinking=True,
                bos_token=codec.get("bos_token") or "", eos_token=codec.get("eos_token") or "",
            ) + text
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        require(14 <= len(ids) <= 128, "prompt_length_out_of_range")
        inputs.append(ids)
    candidate_ids = []
    for word in CANDIDATE_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False).ids
        if len(ids) == 1 and ids[0] not in candidate_ids:
            candidate_ids.append(ids[0])
        if len(candidate_ids) == 16:
            break
    require(len(candidate_ids) == 16 and len(set(candidate_ids)) == len(candidate_ids),
            "candidate_token_set_invalid")
    require(all(max(ids) < len(tokenizer.get_vocab()) for ids in [*inputs, candidate_ids]), "token_id_out_of_range")
    template_hash = hashlib.sha256(codec["prompt_template"].encode()).hexdigest()
    return inputs, candidate_ids, template_hash


def validate_prefix_lengths(lengths: list[int], context_tokens: int = CONTEXT) -> None:
    validate_context_tokens(context_tokens)
    require(isinstance(lengths, list) and bool(lengths)
            and all(type(length) is int and 1 <= length <= context_tokens for length in lengths)
            and len(set(lengths)) == len(lengths), "prefix_lengths_invalid")


def parse_prefix_lengths(value: str) -> list[int]:
    fields = value.split(",")
    if not all(field.strip().isascii() and field.strip().isdecimal() for field in fields):
        raise argparse.ArgumentTypeError("prefix-lengths must be distinct integers from 1 through 4096")
    try:
        lengths = [int(field) for field in fields]
        validate_prefix_lengths(lengths, MAX_CONTEXT)
    except (ValueError, CheckFailed):
        raise argparse.ArgumentTypeError("prefix-lengths must be distinct integers from 1 through 4096") from None
    return lengths


def collect_corpus_prefixes(tokenizer: Tokenizer, corpus: Path,
                           lengths: list[int], context_tokens: int = CONTEXT) -> tuple[list[list[int]], str]:
    validate_prefix_lengths(lengths, context_tokens)
    require(corpus.is_file(), "corpus_path_invalid")
    raw = corpus.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckFailed("corpus_encoding_invalid") from None
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    require(len(ids) >= max(lengths), "corpus_tokens_insufficient")
    vocab_size = len(tokenizer.get_vocab())
    prefixes = [ids[:length] for length in lengths]
    require(all(type(token) is int and 0 <= token < vocab_size
                for prefix in prefixes for token in prefix), "token_id_out_of_range")
    return prefixes, hashlib.sha256(raw).hexdigest()


async def run(args) -> dict:
    context_tokens = getattr(args, "context_tokens", CONTEXT)
    validate_context_tokens(context_tokens)
    prefix_lengths = getattr(args, "prefix_lengths", None)
    corpus = getattr(args, "corpus", None)
    require((corpus is None) == (prefix_lengths is None), "corpus_prefix_options_required_together")
    if prefix_lengths is not None:
        validate_prefix_lengths(prefix_lengths, context_tokens)
    model = args.model.resolve(strict=True)
    reference = args.reference_worker.resolve(strict=True)
    candidate = args.candidate_worker.resolve(strict=True)
    output = args.output.resolve()
    require(model.is_dir() and reference.is_file() and candidate.is_file(), "input_path_invalid")
    require(not output.exists(), "output_already_exists")
    weights_before = weight_identity(model, args.verified_weights)
    reference_sha = sha(reference)
    candidate_sha = sha(candidate)
    inputs, candidates, template_hash = collect_inputs(model)
    rendered_context_count = len(inputs)
    corpus_sha = None
    if corpus is not None:
        prefixes, corpus_sha = collect_corpus_prefixes(
            Tokenizer.from_file(str(model / "tokenizer.json")), corpus, prefix_lengths, context_tokens,
        )
        inputs.extend(prefixes)
    input_lengths = [len(ids) for ids in inputs]
    require(all(1 <= length <= context_tokens for length in input_lengths), "input_length_out_of_range")
    input_hash = token_hash(inputs)
    candidate_hash = token_hash([candidates])
    probe_path = requested_probe_path()
    reference_rows = await run_worker(reference, model, inputs, candidates, False, context_tokens=context_tokens)
    candidate_rows = await run_worker(candidate, model, inputs, candidates, True, probe_path,
                                      context_tokens=context_tokens)
    mma_probe = read_mma_probe(probe_path, expected_rows=len(inputs)) if probe_path is not None else None
    deltas = [b - a for ref, cand in zip(reference_rows, candidate_rows, strict=True)
              for a, b in zip(ref, cand, strict=True)]
    maximum = max(abs(x) for x in deltas)
    mean_delta = statistics.fmean(deltas)
    rms_delta = math.sqrt(statistics.fmean(x * x for x in deltas))
    weights_after = weight_identity(model, args.verified_weights)
    require(weights_before == weights_after, "verified_weights_changed")
    require(sha(reference) == reference_sha and sha(candidate) == candidate_sha, "worker_executable_changed")
    flags = {key: os.environ[key] for key in (
        "STARK_GLM53_BOUNDED_GRAPH", "STARK_GLM53_Q4K_BLOCK", "STARK_GLM53_KQUANT_FACTORED",
        "STARK_GLM53_CONTROL_SPLIT", "STARK_GLM53_CONTROL_SIMD", "STARK_HYBRID_PACKED_REFERENCE",
        "STARK_GLM53_KDA_TRANSPOSED", "STARK_GLM53_ENCODER_BANKS", "STARK_GLM53_PACKED_FAST_MATH",
        "STARK_GLM53_PARALLEL_PROJECTIONS", "STARK_GLM53_BANK_GROUP_LAYERS", "STARK_GLM53_Q8_SHAPES",
        "STARK_GLM53_CONCURRENT_ENCODER", "STARK_GLM53_KDA_FUSED_STATE", "STARK_GLM53_SERIAL_ROUTER",
        "STARK_GLM53_CONTROL_CHUNKS", "STARK_GLM53_PARALLEL_RMS", "STARK_GLM53_RMS_PREFETCH",
        "STARK_GLM53_IQ4_VECTOR", "STARK_GLM53_RESIDENT_BANKS", "STARK_GLM53_BATCH_FINITE",
        "STARK_GLM53_IQ_MULTIROW", "STARK_GLM53_Q6_GROUPED16",
        "STARK_GLM53_Q5_GROUPED32", "STARK_GLM53_CONTROL_RMS_PROJECT",
        "STARK_GLM53_IQ_GPU_BOUND", "STARK_GLM53_GREEDY_OVERLAP",
        "STARK_GLM53_FINITE_SELECTIVE",
        "STARK_GLM53_IQ_TYPED16", "STARK_GLM53_WIDE_ENCODER",
        "STARK_GLM53_IQ_SIGNED_FACTOR", "STARK_GLM53_POLL_WAIT",
        "STARK_GLM53_SERIAL_FINITE_ORDER", "STARK_GLM53_UNRETAINED_GRAPH",
        "STARK_GLM53_SCRATCH_HEAP", "STARK_GLM53_CONTROL_FUSED_PARALLEL",
        "STARK_GLM53_Q8_GROUPED8",
        "STARK_GLM53_Q4_GROUPED32", "STARK_GLM53_GPU_GREEDY",
        "STARK_GLM53_PREFILL_OVERLAP", "STARK_GLM53_KDA_PARALLEL_NORM",
        "STARK_GLM53_PREFILL_QUEUED",
        "STARK_GLM53_PREFILL_MMA",
        "STARK_GLM53_PREFILL_CHUNK",
        "STARK_GLM53_IQ_FUSED_GATE_UP",
        "STARK_GLM53_IQ_FUSED_GATE_UP_R2",
        "STARK_GLM53_DECODE_GRAPH", "STARK_HYBRID_ORDERED_RMS",
        "STARK_HYBRID_BYTE_LOADS", "STARK_HYBRID_VECTOR8",
        "STARK_HYBRID_SCALAR_UINT_LOADS", "STARK_HYBRID_FACTORED_AFFINE",
        "STARK_HYBRID_SEPARATE_FINITE", "STARK_QWEN38_SERIAL_LEVELS",
    ) if key in os.environ}
    report = {
        "schema_version": 1,
        "diagnostic_profile_output_requested": "STARK_QWEN38_PROFILE_OUTPUT" in os.environ,
        "native_profile_output_requested": "STARK_NATIVE_PROFILE_OUTPUT" in os.environ,
        "scope": (
            "8 public rendered contexts x up to 16 candidate tokens; next-token normalized logprob agreement only"
            if corpus is None else
            f"{rendered_context_count} public rendered contexts + {len(prefix_lengths)} exact corpus token prefixes x up to 16 candidate tokens; next-token normalized logprob agreement only"
        ),
        "passed": maximum <= args.max_logprob_error,
        "threshold_max_absolute_logprob_error": args.max_logprob_error,
        "context_count": len(inputs), "candidate_count": len(candidates), "score_point_count": len(deltas),
        "min_input_tokens": min(input_lengths), "max_input_tokens": max(input_lengths),
        "total_input_tokens": sum(input_lengths),
        **({
            "corpus_sha256": corpus_sha,
            "requested_prefix_lengths": prefix_lengths,
            "corpus_prefix_tokenization": "Direct tokenizer encoding with add_special_tokens=False; exact leading prefixes of one corpus tokenization.",
        } if corpus is not None else {}),
        "max_absolute_logprob_delta": maximum, "mean_signed_logprob_delta": mean_delta,
        "rms_logprob_delta": rms_delta,
        "reference_mean_candidate_nll": statistics.fmean(-x for row in reference_rows for x in row),
        "candidate_mean_candidate_nll": statistics.fmean(-x for row in candidate_rows for x in row),
        **({"mma_score_probe": mma_probe} if mma_probe is not None else {}),
        "reference_worker_sha256": reference_sha, "candidate_worker_sha256": candidate_sha,
        "reference_identity": (
            "Archived GLM candidate using original F32 scoring math (source 3eca plus patch 7056); not an MLX teacher."
            if reference_sha == ARCHIVED_REFERENCE_SHA256 else "Caller-supplied reference; source identified separately."
        ),
        "harness_sha256": sha(Path(__file__)), "configuration_sha256": sha(model / "config.json"),
        "tokenizer_sha256": sha(model / "tokenizer.json"), "prompt_template_sha256": template_hash,
        "input_token_ids_sha256": input_hash, "candidate_token_ids_sha256": candidate_hash,
        "harness_checkout": checkout_identity(), "weights": weights_before,
        "context_tokens": context_tokens, "temperature": 1.0, "candidate_flags": flags,
        "candidate_set_note": "Distinct single-token encodings of fixed public continuation words; candidate IDs and text are not stored.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-worker", type=Path, required=True)
    parser.add_argument("--candidate-worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-weights", type=Path)
    parser.add_argument("--corpus", type=Path, help="Public UTF-8 document for additional exact token prefixes")
    parser.add_argument("--context-tokens", type=int, default=CONTEXT,
                        help="Scoring context capacity from 1 through 4096 (default: 512)")
    parser.add_argument("--prefix-lengths", type=parse_prefix_lengths,
                        help="Distinct comma-separated token lengths, each at most --context-tokens")
    parser.add_argument("--max-logprob-error", type=float, default=0.001)
    args = parser.parse_args()
    if not 1 <= args.context_tokens <= MAX_CONTEXT:
        parser.error("context-tokens must be from 1 through 4096")
    if not math.isfinite(args.max_logprob_error) or args.max_logprob_error <= 0:
        parser.error("max-logprob-error must be finite and positive")
    if (args.corpus is None) != (args.prefix_lengths is None):
        parser.error("corpus and prefix-lengths must be supplied together")
    try:
        result = asyncio.run(run(args))
        print(json.dumps({"passed": result["passed"], "report_written": True}, sort_keys=True))
        return 0 if result["passed"] else 1
    except CheckFailed as failure:
        print(json.dumps({"passed": False, "check": str(failure)}, sort_keys=True))
        return 1
    except (Exception, KeyboardInterrupt):  # Never expose prompts, worker output, or exceptions.
        print(json.dumps({"passed": False, "check": "score_worker_comparison_failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
