"""Untimed, private acceptance for the standalone Stark scoring rename.

Required arguments identify a private, portable reference directory, two Stark
binaries, the matching BF16 model and a compiled validation reaper. References
can be exported from sealed scoring artifacts with export_scoring_reference.py.
Only the Python standard library and the supplied native tools are required.

Acceptance covers all 243 cases, 48 boundary requests, and 534 full-vocabulary
digests, each replayed after reset and as an identical consecutive cache hit.
It also checks tokenization, default-off operation, cache transitions, errors,
clean EOF, and restart. Exactness compares JSON-decoded f64 bits and complete
vocabulary f32-bit digests. Timing and memory fields are never interpreted as
performance evidence. This finite corpus does not establish arbitrary-model,
host-app/HTTP/MCP, concurrent scheduling, native fault, or failed-flush behavior;
the latter two require the separate worker/native tests. Raw protocol records
and paths stay in mode-0600 files under a mode-0700 output directory. Stdout
contains only aggregate counts, booleans, hashes, and authored failure codes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

import scoring_fixture as prompt
from scoring_process import ProcessOwner

sys.dont_write_bytecode = True
logging.disable(logging.CRITICAL)

REVISION = "8d01581474760bd1fc7f4a547ff1d02f7dada437"
CACHE_POLICY = "consecutive-full-vocabulary-logits-v1"
CACHE_KEY = "exact-token-ids+context-tokens+model-instance"
CONTEXT = 8192
TEMPERATURE = 1.22
LIMIT = 1_048_576
LENGTHS = (1, 2, 255, 256, 257, 1023, 1024, 1025, 2048, 2049, 4096, 8192)
OUTPUT_FIELDS = ("candidate_token_ids", "logits", "log_probabilities", "candidate_probabilities", "temperature")
METRICS = {"state_creation_seconds", "prefill_seconds", "readout_seconds", "score_seconds", "total_seconds"}
SCORE_FIELDS = {"input_tokens", "prompt_tokens_sha256", "candidate_token_ids", "logits", "log_probabilities",
                "candidate_probabilities", "temperature", "cache_hit", "truncated", "metrics",
                "peak_memory_bytes", "cache"}
DIGEST_FIELDS = {"vocab_logits_sha256", "vocab_size", "prompt_tokens_sha256", "input_tokens",
                 "cache_hit", "truncated", "cache"}


class ValidationFailure(Exception):
    """Only fixed, authored codes can escape to the scalar report."""


def require(condition, code):
    if not condition:
        raise ValidationFailure(code)


class Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValidationFailure("arguments_invalid")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ordinary_file(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "artifact_file_invalid")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with ordinary_file(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_private(path):
    path = ordinary_file(path)
    require(path.stat().st_mode & 0o077 == 0, "reference_privacy_invalid")
    return json.loads(path.read_bytes())


def private_directory(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "private_location_invalid")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        require(not current.is_symlink(), "private_symlink")
        current.mkdir(mode=0o700, exist_ok=True)
    require(not list(path.iterdir()), "output_directory_not_empty")
    path.chmod(0o700)
    return path


def private_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def exact(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("!d", left) == struct.pack("!d", right)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[key], right[key]) for key in left)
    return left == right


def token_hash(tokens):
    require(type(tokens) is list and tokens and all(type(token) is int and 0 <= token < 2**32 for token in tokens),
            "tokens_invalid")
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little"))
    return digest.hexdigest()


def score_request(tokens, candidates, *, context=CONTEXT, temperature=TEMPERATURE):
    return {"operation": "score", "input_token_ids": list(tokens), "candidate_token_ids": list(candidates),
            "context_tokens": context, "temperature": temperature}


def row_expected(row):
    return {"candidate_token_ids": row["candidate_token_ids"], "logits": row["logits"],
            "log_probabilities": row["log_probabilities"], "candidate_probabilities": row["probabilities"],
            "temperature": TEMPERATURE}


def validate_values(value, request):
    count = len(request["candidate_token_ids"])
    require(value.get("input_tokens") == len(request["input_token_ids"])
            and value.get("prompt_tokens_sha256") == token_hash(request["input_token_ids"])
            and exact(value.get("candidate_token_ids"), request["candidate_token_ids"])
            and exact(value.get("temperature"), request["temperature"])
            and value.get("truncated") is False, "score_identity_invalid")
    for key in ("logits", "log_probabilities", "candidate_probabilities"):
        values = value.get(key)
        require(type(values) is list and len(values) == count
                and all(type(item) is float and math.isfinite(item) for item in values), "score_values_invalid")
    require(all(0 <= item <= 1 for item in value["candidate_probabilities"])
            and abs(math.fsum(value["candidate_probabilities"]) - 1.0) <= 1e-12, "probabilities_invalid")


def validate_score(value, request, expected, *, hit):
    require(set(value) == SCORE_FIELDS, "score_schema_invalid")
    validate_values(value, request)
    require(all(exact(value[key], expected[key]) for key in OUTPUT_FIELDS), "native_bit_equality_failed")
    metrics = value.get("metrics")
    require(type(metrics) is dict and set(metrics) == METRICS
            and all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in metrics.values()),
            "native_metrics_invalid")
    if hit:
        require(metrics["state_creation_seconds"] == 0.0 and metrics["prefill_seconds"] == 0.0,
                "cache_hit_executed_native")
    require(value["peak_memory_bytes"] is None or type(value["peak_memory_bytes"]) is int
            and value["peak_memory_bytes"] >= 0, "memory_schema_invalid")


def validate_digest(value, request, expected, vocab_size):
    require(set(value) == DIGEST_FIELDS and value.get("truncated") is False
            and value.get("vocab_size") == vocab_size
            and value.get("input_tokens") == len(request["input_token_ids"])
            and value.get("prompt_tokens_sha256") == token_hash(request["input_token_ids"]),
            "digest_identity_invalid")
    for key in ("vocab_logits_sha256", "vocab_size", "prompt_tokens_sha256", "input_tokens"):
        require(exact(value[key], expected[key]), "vocabulary_bit_equality_failed")


def verify_bindings(bindings):
    for name, expected in bindings.items():
        require(file_hash(name) == expected, "artifact_changed")


def load_reference(args):
    directory = args.reference_directory
    require(directory.is_dir() and not directory.is_symlink()
            and directory.stat().st_mode & 0o077 == 0, "reference_privacy_invalid")
    manifest = read_private(directory / "MANIFEST.json")
    require(manifest.get("schema_version") == 1
            and manifest.get("reference_sha256") == file_hash(directory / "REFERENCE.json"), "reference_seal_invalid")
    config = read_private(directory / "REFERENCE.json")
    require(config.get("schema_version") == 1 and config.get("context_tokens") == CONTEXT
            and exact(config.get("temperature"), TEMPERATURE)
            and config.get("boundary_lengths") == list(LENGTHS), "full_reference_required")
    require(args.model.is_dir() and not args.model.is_symlink(), "model_reference_mismatch")
    model_files = config.get("model_files")
    require(type(model_files) is dict and model_files, "reference_bindings_invalid")
    bindings = {}
    for name, expected in model_files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and type(expected) is str and len(expected) == 64, "reference_bindings_invalid")
        path = args.model / relative
        require(file_hash(path) == expected, "model_reference_mismatch")
        bindings[str(path)] = expected
    config_hash = config.get("source_configuration_sha256")
    require(type(config_hash) is str and len(config_hash) == 64, "reference_identity_invalid")
    cases = config.get("cases")
    require(type(cases) is list and len(cases) == 243
            and [case["id"] for case in cases] == config.get("case_ids"), "reference_cases_invalid")
    records = config.get("records")
    names = {f"round-{number}-{engine}.json" for number in (1, 2) for engine in ("baseline", "candidate")}
    require(type(records) is dict and set(records) == names | {"digest-baseline.json", "digest-candidate.json"},
            "reference_records_invalid")
    for record in records.values():
        require(record.get("complete") is True and record.get("ownership_settled") is True
                and record.get("configuration_sha256") == config_hash, "reference_record_invalid")
    worker = records["round-1-baseline.json"]
    require(len(worker.get("rows", [])) == 486 and len(worker.get("boundaries", [])) == 48,
            "reference_counts_invalid")
    vocab_size = worker["readiness"]["vocab_size"]
    require(type(vocab_size) is int and vocab_size > 0, "reference_vocab_invalid")
    ordering = [(case["id"], condition) for case in cases for condition in ("fresh", "identical_repeat")]
    boundary_order = [(length, kind) for length in LENGTHS for kind in ("fresh", "repeat", "changed", "return")]
    for name in names:
        block = records[name]
        require(block["readiness"]["vocab_size"] == vocab_size
                and [(row["case_id"], row["condition"]) for row in block["rows"]] == ordering
                and [(row["length"], row["kind"]) for row in block["boundaries"]] == boundary_order,
                "reference_order_invalid")
        for row, expected in zip(block["rows"], worker["rows"], strict=True):
            require(all(exact(row[key], expected[key]) for key in
                        ("input_token_ids", "candidate_token_ids", "logits", "log_probabilities", "probabilities", "decision")),
                    "reference_equality_invalid")
            validate_values({**row_expected(row), "input_tokens": row["input_tokens"],
                             "prompt_tokens_sha256": token_hash(row["input_token_ids"]), "truncated": row["truncated"]},
                            score_request(row["input_token_ids"], row["candidate_token_ids"]))
        for row, expected in zip(block["boundaries"], worker["boundaries"], strict=True):
            require(len(row["tokens"]) == row["length"] and exact(row["tokens"], expected["tokens"])
                    and all(exact(row["response"][key], expected["response"][key]) for key in OUTPUT_FIELDS),
                    "reference_boundary_invalid")
    requests = [score_request(row["input_token_ids"], row["candidate_token_ids"]) for row in worker["rows"]]
    requests += [score_request(row["tokens"], row["response"]["candidate_token_ids"]) for row in worker["boundaries"]]
    digest = records["digest-baseline.json"]
    require(len(digest.get("outputs", [])) == len(requests) == 534, "reference_counts_invalid")
    for name in ("digest-baseline.json", "digest-candidate.json"):
        block = records[name]
        require(block["readiness"]["vocab_size"] == vocab_size, "reference_vocab_invalid")
        for value, request, expected in zip(block["outputs"], requests, digest["outputs"], strict=True):
            require(value.get("cache_hit") is False and value.get("truncated") is False
                    and value.get("vocab_size") == vocab_size
                    and value.get("prompt_tokens_sha256") == token_hash(request["input_token_ids"])
                    and value.get("input_tokens") == len(request["input_token_ids"])
                    and all(exact(value[key], expected[key]) for key in
                            ("vocab_logits_sha256", "vocab_size", "prompt_tokens_sha256", "input_tokens")),
                    "reference_digest_invalid")
    patch_hash = config.get("scoring_patch_sha256")
    require(type(patch_hash) is str and len(patch_hash) == 64, "reference_identity_invalid")
    for record in records.values():
        require(record["readiness"].get("upstream_revision") == REVISION
                and record["readiness"].get("scoring_patch_sha256") == patch_hash, "reference_identity_invalid")
    files = dict(bindings)
    files.update({str(path): file_hash(path) for path in directory.iterdir() if path.is_file()})
    helper_directory = Path(__file__).absolute().parent
    files.update({str(path): file_hash(path) for path in (Path(__file__), args.worker, args.probe, args.reaper,
                  helper_directory / "scoring_process.py", helper_directory / "scoring_fixture.py")})
    return {"cases": cases, "worker": worker, "digest": digest, "requests": requests, "vocab_size": vocab_size,
            "patch_sha256": patch_hash, "files": files, "ProcessOwner": ProcessOwner,
            "prompt": prompt, "reaper": ordinary_file(args.reaper), "configuration_sha256": config_hash}


class Client:
    def __init__(self, process, readiness, enabled):
        self.process, self.readiness, self.enabled = process, readiness, enabled
        self.key, self.hits, self.calls = None, 0, 0
        self.validate_counters(readiness)

    def validate_counters(self, value):
        actual = value.get("cache")
        expected = {"enabled": self.enabled, "hits": self.hits, "native_score_calls": self.calls}
        require(exact(actual, expected), "native_counter_invalid")

    async def exchange(self, payload=None, *, raw=None):
        process = self.process
        require(process.stdin is not None and process.stdout is not None and process.returncode is None
                and not process.ownership_lost, "native_io_unavailable")
        async with asyncio.timeout(240):
            process.stdin.write(raw if raw is not None else encoded(payload))
            await process.stdin.drain()
            reply = await process.stdout.readline()
        require(reply and len(reply) < LIMIT, "native_response_invalid")
        value = json.loads(reply)
        require(type(value) is dict, "native_response_invalid")
        if "error" in value:
            self.key = None
        elif payload.get("operation") == "reset_cache":
            self.key = None
            require(set(value) == {"cache_reset", "cache"} and value["cache_reset"] is True, "cache_reset_invalid")
        elif payload.get("operation") == "score":
            key = (tuple(payload["input_token_ids"]), payload["context_tokens"])
            hit = self.enabled and self.key == key
            require(type(value.get("cache_hit")) is bool and value["cache_hit"] is hit, "cache_expectation_failed")
            self.hits += int(hit)
            self.calls += int(not hit)
            self.key = key if self.enabled else None
        self.validate_counters(value)
        return value

    async def post(self, payload):
        value = await self.exchange(payload)
        require("error" not in value, "native_inference_failed")
        return value

    async def reset(self):
        return await self.post({"operation": "reset_cache"})

    async def tokenize(self, text):
        value = await self.post({"operation": "tokenize", "text": text, "add_special_tokens": False})
        tokens = value.get("token_ids")
        require(type(tokens) is list and value.get("token_count") == len(tokens)
                and value.get("token_ids_sha256") == token_hash(tokens)
                and value.get("vocab_size") == self.readiness["vocab_size"], "tokenization_invalid")
        return tokens


async def launch(owner, args, reference, scratch, *, digest=False, enabled=True):
    binary = args.probe if digest else args.worker
    env = {"HOME": str(scratch), "TMPDIR": str(scratch), "PATH": "/usr/bin:/bin", "LANG": "C", "RUST_LOG": "off"}
    arguments = ["--score-model-path", str(args.model)] + (["--consecutive-cache"] if enabled else [])
    process = await owner.launch(binary, arguments, Path(__file__).absolute().parents[1], env,
                                 io_mode="streams", owned_group=True, stop_grace=.5)
    try:
        require(process.stdout is not None, "native_io_unavailable")
        async with asyncio.timeout(240):
            raw = await process.stdout.readline()
        require(raw and len(raw) < LIMIT, "native_readiness_invalid")
        ready = json.loads(raw)
        require(type(ready) is dict and ready.get("ready") is True and ready.get("engine") == "stark"
                and ready.get("version") == "0.1.0" and ready.get("upstream_version") == "0.6.1"
                and ready.get("mode") == ("exact-token-digest-cache-v1" if digest else "exact-token-scoring-cache-v1")
                and ready.get("cache_policy") == (CACHE_POLICY if enabled else "fresh-state")
                and type(ready.get("cache_enabled")) is bool and ready["cache_enabled"] is enabled
                and ready.get("cache_key") == CACHE_KEY and ready.get("upstream_revision") == REVISION
                and ready.get("scoring_patch_sha256") == reference["patch_sha256"]
                and ready.get("vocab_size") == reference["vocab_size"], "native_readiness_invalid")
        return process, Client(process, ready, enabled)
    except BaseException:
        await process.stop()
        raise


async def clean_eof(process):
    require(process.stdin is not None, "native_io_unavailable")
    async with asyncio.timeout(30):
        process.stdin.close()
        await process.stdin.wait_closed()
        code = await process.wait()
    require(code == 0, "native_eof_failed")


def settled(owner, process):
    require(not process.ownership_lost and not owner.instances, "ownership_unsettled")


async def worker_run(owner, args, reference, scratch):
    process, client = await launch(owner, args, reference, scratch)
    records, boundaries = [], []
    prompt = reference["prompt"]
    try:
        candidates = []
        for letter in prompt.LETTERS:
            tokens = await client.tokenize(letter)
            require(len(tokens) == 1, "candidate_not_single_token")
            candidates.append(tokens[0])
        require(len(candidates) == len(set(candidates)) == 16, "candidate_duplicate")
        for index, case in enumerate(reference["cases"]):
            expected_rows = reference["worker"]["rows"][index * 2:index * 2 + 2]
            rendered = prompt.prompt_text(case["state"], case["question"]["instructions"],
                                          [text for _, text in prompt.decision_options(case["question"])])
            tokens = await client.tokenize(rendered)
            require(exact(tokens, expected_rows[0]["input_token_ids"]), "tokenization_equality_failed")
            selected = candidates[:len(prompt.decision_options(case["question"]))]
            require(exact(selected, expected_rows[0]["candidate_token_ids"]), "candidate_equality_failed")
            request = score_request(tokens, selected)
            await client.reset()
            for offset, expected in enumerate(expected_rows):
                value = await client.post(request)
                validate_score(value, request, row_expected(expected), hit=bool(offset))
                decision = prompt.answer(case["question"],
                                         dict(zip([key for key, _ in prompt.decision_options(case["question"])],
                                                  value["candidate_probabilities"], strict=True)), len(tokens))
                require(exact(decision, expected["decision"]), "decision_equality_failed")
                records.append({"case_id": case["id"], "condition": "hit" if offset else "cold",
                                "request": request, "response": value, "decision": decision})
            if (index + 1) % 20 == 0:
                print(json.dumps({"stage": "worker", "cases_completed": index + 1}), flush=True)
        for expected in reference["worker"]["boundaries"]:
            request = score_request(expected["tokens"], expected["response"]["candidate_token_ids"])
            await client.reset()
            for hit in (False, True):
                value = await client.post(request)
                validate_score(value, request, expected["response"], hit=hit)
                boundaries.append({"length": expected["length"], "kind": expected["kind"],
                                   "condition": "hit" if hit else "cold", "request": request, "response": value})
        require(client.calls == client.hits == 291, "worker_native_count_invalid")
        await clean_eof(process)
    finally:
        await process.stop()
    settled(owner, process)
    return {"complete": True, "ownership_settled": True, "readiness": client.readiness,
            "rows": records, "boundaries": boundaries}, candidates


async def digest_run(owner, args, reference, scratch):
    process, client = await launch(owner, args, reference, scratch, digest=True)
    records = []
    try:
        for index, (request, expected) in enumerate(zip(reference["requests"], reference["digest"]["outputs"], strict=True)):
            await client.reset()
            for hit in (False, True):
                value = await client.post(request)
                validate_digest(value, request, expected, reference["vocab_size"])
                records.append({"condition": "hit" if hit else "cold", "request": request, "response": value})
            if (index + 1) % 40 == 0:
                print(json.dumps({"stage": "digest", "requests_completed": index + 1}), flush=True)
        require(client.calls == client.hits == 534, "digest_native_count_invalid")
        await clean_eof(process)
    finally:
        await process.stop()
    settled(owner, process)
    return {"complete": True, "ownership_settled": True, "readiness": client.readiness, "outputs": records}


def sequence_plan(tokens, candidates):
    changed = list(tokens)
    changed[-1] = next(token for token in candidates if token != changed[-1])
    selected = candidates[:2]
    first = score_request(tokens, selected)
    context = max(len(tokens), CONTEXT - 1)
    require(context != CONTEXT, "sequence_context_invalid")
    return [("A_first", first), ("A_repeat", first), ("B_replace", score_request(changed, selected)),
            ("A_return", first), ("tokenize_preserves", {"operation": "tokenize", "text": "A", "add_special_tokens": False}),
            ("A_after_tokenize", first), ("context_changed", score_request(tokens, selected, context=context)),
            ("context_repeat", score_request(tokens, selected, context=context)), ("context_return", first),
            ("candidate_order", score_request(tokens, list(reversed(selected)))),
            ("candidate_set", score_request(tokens, candidates[2:4])),
            ("temperature_changed", score_request(tokens, selected, temperature=TEMPERATURE / 2)),
            ("malformed_request", None), ("A_after_malformed", first),
            ("invalid_score", score_request(tokens, [selected[0], selected[0]])), ("A_after_invalid_score", first),
            ("invalid_tokenize", {"operation": "tokenize", "text": "", "add_special_tokens": False}),
            ("A_after_invalid_tokenize", first),
            ("malformed_tokenize", {"operation": "tokenize", "text": "A", "unexpected": True}),
            ("A_after_malformed_tokenize", first)]


async def sequence_run(owner, args, reference, scratch, candidates):
    first_row = reference["worker"]["rows"][0]
    request = score_request(first_row["input_token_ids"], first_row["candidate_token_ids"])
    plan = sequence_plan(first_row["input_token_ids"], candidates)
    errors = {"malformed_request": "invalid_request", "invalid_score": "invalid_scoring_request",
              "invalid_tokenize": "invalid_tokenization_request", "malformed_tokenize": "invalid_request"}
    process, control = await launch(owner, args, reference, scratch, enabled=False)
    try:
        default_off, controls = [], {}
        for _ in range(2):
            value = await control.post(request)
            validate_score(value, request, row_expected(first_row), hit=False)
            default_off.append(value)
        for name, item in plan:
            if name not in errors and item["operation"] == "score":
                value = await control.post(item)
                validate_score(value, item, value, hit=False)
                controls[name] = value
        require(control.hits == 0, "default_off_hit_invalid")
        await clean_eof(process)
    finally:
        await process.stop()
    settled(owner, process)
    process, client = await launch(owner, args, reference, scratch)
    records = []
    try:
        await client.reset()
        for name, item in plan:
            value = await client.exchange(raw=b"{\n") if item is None else await client.exchange(item)
            if name in errors:
                require(set(value) == {"error", "cache"} and value["error"] == errors[name], "expected_error_invalid")
            elif item["operation"] == "score":
                validate_score(value, item, controls[name], hit=value["cache_hit"])
            else:
                require(value.get("token_count") == len(value["token_ids"])
                        and value.get("token_ids_sha256") == token_hash(value["token_ids"]), "tokenization_invalid")
            records.append({"name": name, "request": item, "response": value})
        await clean_eof(process)
    finally:
        await process.stop()
    settled(owner, process)
    process, restarted = await launch(owner, args, reference, scratch)
    try:
        value = await restarted.post(request)
        validate_score(value, request, row_expected(first_row), hit=False)
        require(restarted.calls == 1 and restarted.hits == 0, "restart_cache_invalid")
        await clean_eof(process)
    finally:
        await process.stop()
    settled(owner, process)
    return {"complete": True, "ownership_settled": True,
            "default_off": {"readiness": control.readiness, "responses": default_off},
            "readiness": client.readiness, "rows": records, "controls": controls,
            "restart": {"readiness": restarted.readiness, "response": value}, "eof_exit_code": 0}


async def run(args):
    reference = load_reference(args)
    directory = private_directory(args.output_root)
    configuration = {"schema_version": 1, "mode": "untimed-stark-scoring-rename-acceptance",
                     "reference_directory": str(args.reference_directory), "worker": str(args.worker), "probe": str(args.probe),
                     "model": str(args.model), "files": reference["files"],
                     "reference_configuration_sha256": reference["configuration_sha256"],
                     "timing_not_reported": True}
    private_json(directory / "configuration.json", configuration)
    configuration_hash = file_hash(directory / "configuration.json")
    private_json(directory / "pre-run-seal.json", {"configuration_sha256": configuration_hash})
    owner = reference["ProcessOwner"](reference["reaper"])
    try:
        with tempfile.TemporaryDirectory(prefix="stark-scoring-validation-") as temporary:
            scratch = Path(temporary)
            scratch.chmod(0o700)
            worker, candidates = await worker_run(owner, args, reference, scratch)
            private_json(directory / "worker.json", worker)
            digest = await digest_run(owner, args, reference, scratch)
            private_json(directory / "digest.json", digest)
            sequences = await sequence_run(owner, args, reference, scratch, candidates)
            private_json(directory / "sequences.json", sequences)
        verify_bindings(reference["files"])
        require(not owner.instances, "ownership_unsettled")
        report = {"schema_version": 1, "status": "passed", "mode": "untimed-stark-scoring-rename-acceptance",
                  "cases": 243, "case_responses": 486, "boundary_requests": 48, "boundary_responses": 96,
                  "digest_requests": 534, "digest_responses": 1068,
                  "worker_native_score_calls": 291, "worker_cache_hits": 291,
                  "probe_native_score_calls": 534, "probe_cache_hits": 534,
                  "tokenization_cases": 243, "candidate_tokens": 16, "sequence_steps": 20,
                  "default_off_responses": 2, "bit_exact_raw_outputs": True, "bit_exact_vocabulary_digests": True,
                  "readiness_identity_valid": True, "ownership_settled": True, "clean_eof_valid": True,
                  "restart_cache_cold": True, "timing_not_reported": True,
                  "reference_configuration_sha256": reference["configuration_sha256"],
                  "worker_sha256": reference["files"][str(args.worker)],
                  "probe_sha256": reference["files"][str(args.probe)]}
        private_json(directory / "REPORT.json", report)
        records = {name: file_hash(directory / name) for name in ("worker.json", "digest.json", "sequences.json", "REPORT.json")}
        private_json(directory / "post-run-seal.json", {"configuration_sha256": configuration_hash,
                     "ownership_settled": True, "records": records})
        print(json.dumps(report), flush=True)
    finally:
        await owner.close()


def main():
    try:
        parser = Parser(description=__doc__)
        for flag in ("reference-directory", "reaper", "worker", "probe", "model", "output-root"):
            parser.add_argument("--" + flag, type=Path, required=True)
        args = parser.parse_args()
        for key, value in vars(args).items():
            setattr(args, key, value.absolute())
        asyncio.run(run(args))
    except (Exception, KeyboardInterrupt) as failure:
        code = str(failure) if isinstance(failure, ValidationFailure) else "scoring_rename_validation_failed"
        print(json.dumps({"status": "failed", "error": code, "timing_not_reported": True}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
