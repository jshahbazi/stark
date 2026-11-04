"""Model-bound validation and timing for generation prefix reuse.

Runs deterministic synthetic agent sessions, rendered by the model's own chat
template with tool definitions and tool-call history, against a worker started
with ``--prefix-cache``, the same build without it and a baseline build. It
checks that generated token IDs, decoded text and parsed tool calls are
identical, that every reuse count matches the documented rule, that every
invalidation path reuses nothing, and measures the native first-token time of
first and follow-up turns.

Workers run one at a time so models that fill most of unified memory can be
validated. Only counts, timings, booleans and SHA-256 hashes are written.
Prompts, generated text and token vectors stay in memory. Run from the
repository root:

    uv run --project tools/converter --locked --python 3.13.13 python \\
        scripts/validate_generation_prefix_cache.py --model MODEL_DIR \\
        --baseline BASELINE_WORKER --candidate CANDIDATE_WORKER \\
        --output runtime/prefix-cache-validation.json

Add ``--reuse boundary`` for backends that resume only from a snapshot at the
previous prompt's last full prefill chunk (native hybrid models).
The pinned raw EXL3/FP8 pack also requires ``--download-receipt RECEIPT``;
its complete original pack stays unchanged and its chat codec is adapted in RAM.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import secrets
import select
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path

from stark_convert.api_tools import parse_tool_output, tool_parser_for
from stark_convert.chat import load_template
from stark_convert import hybrid_prepare, template_runtime
from stark_convert.local_api import APIError

WORDS = (
    "agent",
    "tool",
    "file",
    "read",
    "write",
    "search",
    "pattern",
    "directory",
    "result",
    "request",
    "option",
    "value",
    "error",
    "status",
    "context",
    "session",
    "project",
    "module",
    "function",
    "return",
    "string",
    "number",
    "object",
    "array",
    "schema",
    "parameter",
    "description",
    "required",
    "property",
    "include",
    "exclude",
    "limit",
    "offset",
    "path",
    "line",
    "column",
    "match",
    "replace",
    "edit",
    "command",
    "output",
    "input",
    "timeout",
    "permission",
    "approve",
    "deny",
    "workspace",
    "branch",
    "commit",
    "diff",
    "patch",
    "test",
    "build",
)
INVALIDATIONS = (
    "error_reply",
    "unscoped_request",
    "reset_cache",
    "image_operation",
    "cancellation",
)
EXL3_REPOSITORY = "OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw"
EXL3_REVISION = "50237d5c26ceb98bfd85f779990b0f5e6d7af144"
EXL3_FILES = frozenset({
    ".gitattributes", "LICENSE", "README.md", "chat_template.jinja", "config.json",
    "expert-manifest-v2.json", "expert-manifest.json", "expert-usage-v1.json", "experts.bin",
    "generation_config.json", "hf_quant_config.json", "manifest.sha256",
    "model.safetensors.index.json", "mtp-expert-manifest-nvfp4.json", "mtp-experts-nvfp4.bin",
    "processor_config.json", "residents-fp8-manifest.json", "tokenizer.json", "tokenizer_config.json",
    *(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
})


def digest_ids(ids):
    return hashlib.sha256(
        b"".join(int(i).to_bytes(4, "little") for i in ids)
    ).hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_receipt_identity(model, receipt_path):
    """Bind the complete pinned flat pack to prior verified content hashes.

    This is a trusted local download receipt, not a fresh payload hash audit.
    Every file is checked against its recorded size, mtime and inode, and the
    caller must repeat this guard after model-bound execution.
    """
    receipt = hybrid_prepare._json(receipt_path, 2 * 1024 * 1024)
    if (not isinstance(receipt, dict) or receipt.get("passed") is not True
            or receipt.get("repository") != EXL3_REPOSITORY or receipt.get("revision") != EXL3_REVISION
            or not isinstance(receipt.get("destination"), str)
            or Path(receipt["destination"]).resolve() != model.resolve()):
        raise ValueError("download receipt source or completion mismatch")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != EXL3_FILES:
        raise ValueError("download receipt must include the complete EXL3 pack")
    for path in model.iterdir():
        if path.is_symlink() or not ((path.is_file() and path.name in files)
                                     or (path.is_dir() and path.name == ".cache")):
            raise ValueError("unlisted or linked EXL3 pack entry")
    identities = {}
    total = 0
    for name, expected in sorted(files.items()):
        if not isinstance(expected, dict) or expected.get("matched") is not True:
            raise ValueError("download receipt contains an unverified file")
        digest = expected.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("download receipt content hash is invalid")
        if any(type(expected.get(key)) is not int for key in ("bytes", "mtime_ns", "inode")):
            raise ValueError("download receipt file identity is invalid")
        path = model / name
        try:
            current = path.lstat()
        except OSError:
            raise ValueError("EXL3 pack file is missing") from None
        if (not stat.S_ISREG(current.st_mode)
                or (current.st_size, current.st_mtime_ns, current.st_ino)
                != (expected["bytes"], expected["mtime_ns"], expected["inode"])):
            raise ValueError("EXL3 pack file changed since download verification")
        identities[name] = {"bytes": current.st_size, "mtime_ns": current.st_mtime_ns,
                            "inode": current.st_ino, "prior_verified_sha256": digest}
        total += current.st_size
    if type(receipt.get("total_bytes")) is not int or receipt["total_bytes"] != total:
        raise ValueError("download receipt byte total mismatch")
    manifest = "".join(f"{name}\t{row['prior_verified_sha256']}\n" for name, row in identities.items())
    return {"repository": EXL3_REPOSITORY, "revision": EXL3_REVISION,
            "receipt_sha256": file_sha256(receipt_path), "content_sha256": digest_text(manifest),
            "files": identities, "total_bytes": total,
            "verification": "Prior verified SHA-256 hashes with current size/mtime/inode guards; payloads are not rehashed."}


def streaming_contract(config):
    stream = config.get("expert_streaming")
    return (config.get("model_type") == "glm5_next" and isinstance(stream, dict)
            and all(stream.get(key) == value for key, value in {
                "family": "mlx-stream", "format": "mtplx-expert-manifest-v2",
                "manifest": "expert-manifest-v2.json", "runtime_manifest": "expert-manifest.json",
                "quant": "exl3-k3-mul1",
            }.items()) and config.get("residents_format") == "fp8_block128")


def load_validation_template(model, download_receipt=None):
    """Adapt verified raw HF metadata in RAM using native template evaluation.

    Normal prepared models retain chat.load_template behaviour. The raw pack
    uses the same EOS/BOS/template normalization rules as finish_metadata,
    without invoking its file-writing preparation step.
    """
    config = hybrid_prepare._json(model / "config.json", 2 * 1024 * 1024)
    if not isinstance(config, dict):
        raise ValueError("model configuration is invalid")
    if "expert_streaming" not in config:
        if download_receipt is not None:
            raise ValueError("download receipt requires the explicit EXL3 streaming pack")
        codec, template = load_template(model)
        return codec, template, None
    if not streaming_contract(config) or download_receipt is None:
        raise ValueError("EXL3 streaming validation requires its pinned download receipt")
    identity = download_receipt_identity(model, download_receipt)
    tokenizer = hybrid_prepare._json(model / "tokenizer_config.json", 2 * 1024 * 1024)
    generation = hybrid_prepare._json(model / "generation_config.json")
    if not isinstance(tokenizer, dict) or not isinstance(generation, dict):
        raise ValueError("raw tokenizer metadata is invalid")
    source_path = model / "chat_template.jinja"
    if source_path.stat().st_size > 256 * 1024:
        raise ValueError("missing or excessive chat template")
    source = source_path.read_text(encoding="utf-8")
    if not 0 < len(source.encode("utf-8")) <= 256 * 1024:
        raise ValueError("missing or excessive chat template")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("raw text metadata is invalid")
    stops = hybrid_prepare._ids(text.get("eos_token_id"))
    if generation.get("eos_token_id") is not None:
        stops.extend(hybrid_prepare._ids(generation["eos_token_id"]))
    vocab = text.get("vocab_size")
    if type(vocab) is not int or not 0 < vocab <= 1_000_000 or any(not 0 <= item < vocab for item in stops):
        raise ValueError("invalid model vocabulary or EOS IDs")
    codec = {"type": "ChatCodecConfig", "prompt_template": source}
    for field in ("bos_token", "eos_token"):
        token = tokenizer.get(field)
        if isinstance(token, dict):
            token = token.get("content")
        if token is not None:
            if not isinstance(token, str) or not 0 < len(token.encode()) <= 8192:
                raise ValueError("invalid special-token metadata")
            codec[field] = token
    metadata_hashes = {name: file_sha256(model / name) for name in
                       ("config.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")}
    if any(digest != identity["files"][name]["prior_verified_sha256"] for name, digest in metadata_hashes.items()):
        raise ValueError("raw chat metadata content changed")
    template = template_runtime.IsolatedTemplate(source)
    provenance = {"adapter": "raw_hf_exl3_in_memory_v1", "source_template_sha256": digest_text(source),
                  "codec_sha256": digest_text(json.dumps(codec, sort_keys=True, ensure_ascii=False)),
                  "metadata_sha256": metadata_hashes,
                  "metadata_normalization_sha256": file_sha256(Path(hybrid_prepare.__file__)),
                  "template_runtime_sha256": file_sha256(Path(template_runtime.__file__)),
                  "native_template_executable_sha256": file_sha256(template._executable)}
    if download_receipt_identity(model, download_receipt) != identity:
        raise ValueError("EXL3 pack changed during template preparation")
    return codec, template, provenance


def weights_sha256(model, download_receipt=None):
    """Bind actual native or converted weights, including external n-gram data."""
    if download_receipt is not None:
        return download_receipt_identity(model, download_receipt)["content_sha256"]
    if (model / "experts.bin").exists() or (model / "mtp-experts-nvfp4.bin").exists():
        raise ValueError("EXL3 banks require a complete verified download receipt")
    shards = sorted([*model.glob("*.safetensors"), *model.glob("*.gguf")])
    if (model / "ngram_table.bin").is_file():
        shards.append(model / "ngram_table.bin")
    if not shards:
        raise SystemExit("no supported weight files found")
    if [shard.name for shard in shards] == ["model.safetensors"]:
        return file_sha256(shards[0])
    manifest = "".join(f"{shard.name}\t{file_sha256(shard)}\n" for shard in shards)
    return digest_text(manifest)


def report_stage(enabled, **counts):
    if enabled:
        print(json.dumps(counts), file=sys.stderr, flush=True)


class Worker:
    """One private worker process; replies are parsed in memory only."""

    def __init__(
        self, binary, model, *, prefix_cache=False, timeout=900, progress=False
    ):
        self.timeout = timeout
        started = time.monotonic()
        self.cancel_read, self.cancel_write = os.pipe()
        command = [binary, "--model-path", model] + (
            ["--prefix-cache"] if prefix_cache else []
        )
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                pass_fds=(self.cancel_read,),
                env=dict(os.environ, STARK_CANCEL_FD=str(self.cancel_read)),
            )
        except BaseException:
            os.close(self.cancel_read)
            os.close(self.cancel_write)
            raise
        os.close(self.cancel_read)
        report_stage(progress, phase="worker_spawned", prefix_cache=prefix_cache)
        self.request_id = 0
        self.buffer = bytearray()
        try:
            self.ready = self.read()
            if self.ready.get("ready") is not True:
                raise SystemExit(
                    "worker failed to start: " + str(self.ready.get("error"))
                )
            report_stage(
                progress,
                phase="worker_ready",
                startup_seconds=round(time.monotonic() - started, 3),
                prefix_cache_enabled=(self.ready.get("prefix_cache") or {}).get(
                    "enabled", False
                ),
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self):
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = max(0, deadline - time.monotonic())
            if not select.select([self.process.stdout], [], [], remaining)[0]:
                raise SystemExit("worker timed out")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise SystemExit("worker exited")
            self.buffer.extend(block)
        line, _, self.buffer = self.buffer.partition(b"\n")
        return json.loads(line)

    def send(self, payload):
        self.process.stdin.write(json.dumps(payload).encode() + b"\n")
        self.process.stdin.flush()

    def request(self, payload, *, cancel_after_first_progress=False):
        if payload.get("operation") == "generate":
            self.request_id += 1
            payload = {**payload, "request_id": self.request_id}
        self.send(payload)
        progress = []
        while True:
            reply = self.read()
            if reply.get("event") == "progress":
                progress.append(reply)
                if cancel_after_first_progress and len(progress) == 1:
                    os.write(self.cancel_write, self.request_id.to_bytes(8, "little"))
                continue
            if reply.get("event") == "delta":
                continue
            reply["_progress"] = progress
            return reply

    def tokenize(self, text):
        return self.request({"operation": "tokenize", "text": text})["token_ids"]

    def close(self):
        # EOF requests native cancellation before waiting for the worker.
        if self.cancel_write is not None:
            os.close(self.cancel_write)
            self.cancel_write = None
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def tool_definition(rng, index):
    properties = {
        f"{rng.choice(WORDS)}_{p}": {
            "type": rng.choice(["string", "integer", "boolean"]),
            "description": " ".join(
                rng.choice(WORDS) for _ in range(rng.randint(6, 16))
            ),
        }
        for p in range(rng.randint(2, 5))
    }
    return {
        "type": "function",
        "function": {
            "name": f"tool_{index}_{rng.choice(WORDS)}",
            "description": " ".join(
                rng.choice(WORDS) for _ in range(rng.randint(30, 90))
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties)[:1],
            },
        },
    }


def paragraph(rng, sentences, words=(8, 20)):
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(*words))).capitalize()
        + "."
        for _ in range(sentences)
    )


class Session:
    """A deterministic synthetic agent conversation."""

    def __init__(self, seed, tools=24, instruction_paragraphs=40):
        rng = random.Random(seed)
        self.seed = seed
        self.tools = [tool_definition(rng, index) for index in range(tools)]
        system = "\n\n".join(paragraph(rng, 4) for _ in range(instruction_paragraphs))
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": paragraph(rng, 3)},
        ]
        self.step = 0

    def add_tool_step(self, result_sentences):
        rng = random.Random(self.seed * 7919 + self.step)
        call_id = f"call_{self.step}"
        tool = self.tools[self.step % len(self.tools)]["function"]["name"]
        arguments = {"path": "/".join(rng.choice(WORDS) for _ in range(3))}
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": arguments},
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": paragraph(rng, result_sentences),
            }
        )
        self.step += 1

    def add_user_turn(self, sentences):
        rng = random.Random(self.seed * 104729 + self.step)
        self.messages.append({"role": "assistant", "content": paragraph(rng, 2)})
        self.messages.append({"role": "user", "content": paragraph(rng, sentences)})
        self.step += 1

    def render(self, template, codec, messages=None):
        return template.render(
            messages=messages if messages is not None else self.messages,
            tools=self.tools,
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token=codec.get("bos_token") or "",
            eos_token=codec.get("eos_token") or "",
        )


def build_turns(
    worker,
    template,
    codec,
    seed,
    first_tokens,
    *,
    tools=24,
    instruction_paragraphs=40,
    follow_up_sentences=(40, 25, 80),
):
    """Return token IDs for a first turn of about `first_tokens` and three follow-ups."""
    session = Session(seed, tools=tools, instruction_paragraphs=instruction_paragraphs)
    while len(worker.tokenize(session.render(template, codec))) < first_tokens - 400:
        session.add_tool_step(result_sentences=12)
    turns = [worker.tokenize(session.render(template, codec))]
    for grow in (
        lambda: session.add_tool_step(follow_up_sentences[0]),
        lambda: session.add_tool_step(follow_up_sentences[1]),
        lambda: session.add_user_turn(follow_up_sentences[2]),
    ):
        grow()
        turns.append(worker.tokenize(session.render(template, codec)))
    return session, turns


class Expectation:
    """The documented reuse rule, simulated over the cached worker's requests."""

    def __init__(self, chunk, boundary_only):
        self.chunk = chunk
        self.boundary_only = boundary_only
        self.entry = None

    def predict(self, ids, key):
        if self.entry is None or key[0] is None or key != self.entry[0]:
            return 0
        prefix = self.entry[1]
        if self.boundary_only:
            return (
                len(prefix)
                if len(ids) >= len(prefix) + 2 and ids[: len(prefix)] == prefix
                else 0
            )
        common = next(
            (i for i, (a, b) in enumerate(zip(prefix, ids, strict=False)) if a != b),
            min(len(prefix), len(ids)),
        )
        return (
            min(common, (len(ids) - 1) // self.chunk * self.chunk)
            // self.chunk
            * self.chunk
        )

    def record(self, ids, key, reply):
        boundary = (len(ids) - 1) // self.chunk * self.chunk
        if "error" in reply or key[0] is None or boundary == 0:
            self.entry = None
        else:
            self.entry = (key, list(ids[:boundary]))


def generate(
    worker,
    ids,
    *,
    context,
    max_new,
    scope=None,
    sampling=None,
    seed=0,
    mtp_drafts=None,
    **extra,
):
    payload = {
        "operation": "generate",
        "input_token_ids": ids,
        "context_tokens": context,
        "max_new_tokens": max_new,
        "sampling": sampling or {"mode": "greedy"},
        "seed": seed,
        **extra,
    }
    if mtp_drafts is not None:
        payload["mtp_drafts"] = mtp_drafts
    if scope is not None:
        payload["cache_scope"] = scope
    return worker.request(payload)


def outcome(reply, tools, parser):
    """Content-free comparison key: generated IDs, text and parsed tool calls."""
    if "error" in reply:
        return {"error": reply["error"]}
    result = {
        "generated_sha256": digest_ids(reply["generated_token_ids"]),
        "text_sha256": digest_text(reply["text"]),
        "finish_reason": reply.get("finish_reason"),
        "output_tokens": len(reply["generated_token_ids"]),
    }
    if parser is not None:
        try:
            message, finish = parse_tool_output(
                reply["text"],
                tools,
                parser=parser,
                finish_reason=reply.get("finish_reason"),
                call_id_prefix="v",
            )
            parsed = {"message": message, "finish_reason": finish}
        except APIError as error:
            parsed = {"error": error.code}
        result["tool_parse_sha256"] = digest_text(json.dumps(parsed, sort_keys=True))
        result["tool_calls"] = len(parsed.get("message", {}).get("tool_calls") or [])
    return result


def plan(turns, chunk, context, mtp_drafts):
    """The cached worker's request sequence: comparison cases and invalidating actions."""
    steps = []

    def case(label, ids, **options):
        steps.append(
            ("case", label, ids, {"context": context, "scope": "main", **options})
        )

    case("first_turn_cold", turns[0])
    for index in range(1, len(turns)):
        case(f"follow_up_{index}", turns[index])
    stochastic = {"mode": "stochastic", "temperature": 0.7, "top_k": 20, "top_p": 0.95}
    case("follow_up_repeat_stochastic", turns[-1], sampling=stochastic, seed=17)
    case("shrink", turns[1])
    case("regrow", turns[-1])
    early = diverge(turns[-1], min(chunk // 2, len(turns[-1]) - 2))
    case("diverge_early", early)
    case("restore_after_divergence", turns[-1])
    late = diverge(turns[-1], len(turns[-1]) - 2)
    case("diverge_in_final_chunk", late)
    case("restore_final_chunk", turns[-1])
    boundary = (len(turns[-1]) - 1) // chunk * chunk
    case("snapshot_plus_one", turns[-1][:boundary] + turns[-1][-1:])
    case("snapshot_plus_two", turns[-1][:boundary] + turns[-1][-2:])
    case("restore_after_boundary_edges", turns[-1])
    case("context_change", turns[-1], context=context - 1)
    case("context_restored", turns[-1])
    case("scope_change", turns[-1], scope="other")
    case("scope_restored", turns[-1])
    case("warm_again", turns[-1])
    if mtp_drafts:
        case("mtp_after_plain", turns[-1], mtp_drafts=mtp_drafts)
        case("mtp_warm", turns[-1], mtp_drafts=mtp_drafts)
        case("plain_after_mtp", turns[-1])
        case("plain_warm", turns[-1])
    for label in (*INVALIDATIONS, "tokenize_decode"):
        case(f"before_{label}", turns[-1])
        steps.append(("act", label, turns[-1], None))
        case(f"after_{label}", turns[-1])
    return steps


def diverge(ids, position):
    changed = list(ids)
    replacement = next((token for token in ids if token != ids[position]), None)
    if replacement is None:
        raise SystemExit("synthetic prompt needs distinct tokens for divergence checks")
    changed[position] = replacement
    return changed


def act(worker, label, ids, context, scope):
    """Run one action on the cached worker; return its reply and whether it keeps the entry."""
    if label == "error_reply":
        payload = {
            "operation": "generate",
            "input_token_ids": ids,
            "context_tokens": 16,
            "max_new_tokens": 1,
        }
        return worker.request({**payload, "cache_scope": scope}), False
    if label == "unscoped_request":
        return generate(worker, ids[:64], context=context, max_new=1), False
    if label == "reset_cache":
        return worker.request({"operation": "reset_cache"}), False
    if label == "image_operation":
        return worker.request({"operation": "discard_image"}), False
    if label == "cancellation":
        payload = {
            "operation": "generate",
            "input_token_ids": ids,
            "context_tokens": context,
            "max_new_tokens": 64,
            "cache_scope": scope,
            "stream": True,
        }
        return worker.request(payload, cancel_after_first_progress=True), False
    worker.tokenize("synthetic")
    worker.request({"operation": "decode", "token_ids": [1, 2, 3]})
    return {}, True


def comparison_key(ids, options):
    return digest_ids(ids) + json.dumps(
        {k: v for k, v in options.items() if k != "scope"}, sort_keys=True
    )


def report(args, **counts):
    report_stage(args.progress, **counts)


def run_cases(worker, steps, args, tools, parser, *, scopes=None, expectation=None):
    """Run every case once per distinct request (all steps on a cached worker)."""
    outcomes, observed = {}, {}
    for kind, label, ids, options in steps:
        if kind == "act":
            if scopes is None:
                continue
            reply, keeps = act(worker, label, ids, args.context, scopes["main"])
            observed[label] = reply.get("error", "ok")
            if label == "cancellation" and "error" not in reply:
                # Finished before the signal arrived: an ordinary scoped request.
                expectation.record(ids, (scopes["main"], args.context, False), reply)
            elif not keeps:
                expectation.entry = None
            continue
        key = comparison_key(ids, options)
        request = {
            "context": options["context"],
            "max_new": args.max_new,
            "sampling": options.get("sampling"),
            "seed": options.get("seed", 0),
            "mtp_drafts": options.get("mtp_drafts"),
        }
        if scopes is None:
            if key not in outcomes:
                report(args, case_start=label, input_tokens=len(ids))
                outcomes[key] = outcome(generate(worker, ids, **request), tools, parser)
                report(args, case=label, successful="error" not in outcomes[key])
            continue
        cache_key = (
            scopes[options["scope"]],
            options["context"],
            bool(options.get("mtp_drafts")),
        )
        predicted = expectation.predict(ids, cache_key)
        report(args, case_start=label, input_tokens=len(ids))
        reply = generate(worker, ids, scope=scopes[options["scope"]], **request)
        expectation.record(ids, cache_key, reply)
        reused = reply.get("reused_input_tokens")
        observed[label] = {
            "key": key,
            "input_tokens": len(ids),
            "reused_input_tokens": reused,
            "predicted_reuse": predicted,
            "progress_consistent": {
                f.get("reused_input_tokens") for f in reply["_progress"]
            }
            <= {reused},
            "outcome": outcome(reply, tools, parser),
        }
        report(
            args,
            case=label,
            reused_input_tokens=reused,
            successful="error" not in reply,
        )
    return outcomes, observed


def session_options(args):
    return {
        "tools": args.tools,
        "instruction_paragraphs": args.instruction_paragraphs,
        "follow_up_sentences": args.follow_up_sentences,
    }


def preflight_sessions(args, template, codec):
    """Reject unsupported synthetic histories before hashing/loading weights."""
    seeds = [args.seed] if args.mode == "exactness" else [args.seed + 1]
    if args.mode == "all":
        seeds.insert(0, args.seed)
    renders = 0
    for seed in seeds:
        session = Session(
            seed, tools=args.tools, instruction_paragraphs=args.instruction_paragraphs
        )
        session.render(template, codec)
        renders += 1
        for sentences in args.follow_up_sentences[:2]:
            session.add_tool_step(sentences)
            session.render(template, codec)
            renders += 1
        session.add_user_turn(args.follow_up_sentences[2])
        session.render(template, codec)
        renders += 1
    report(args, phase="template_preflight", renders=renders)


def run_exactness(args, template, codec, parser):
    started = time.monotonic()
    report(args, phase="cached")
    with Worker(
        args.candidate, args.model, prefix_cache=True, progress=args.progress
    ) as cached:
        settings = cached.ready.get("prefix_cache") or {}
        if settings.get("enabled") is not True:
            raise SystemExit("candidate did not enable prefix reuse for this model")
        chunk = settings["chunk_tokens"]
        mtp = cached.ready.get("mtp") or {}
        mtp_drafts = mtp.get("recommended_drafts") if mtp.get("supported") else None
        session, turns = build_turns(
            cached,
            template,
            codec,
            args.seed,
            args.first_tokens,
            **session_options(args),
        )
        if len(turns[-1]) + args.max_new >= args.context:
            raise SystemExit("synthetic session exceeds the requested context")
        if (len(turns[0]) - 1) // chunk == 0:
            raise SystemExit("first prompt must exceed one backend reuse chunk")
        steps = plan(turns, chunk, args.context, mtp_drafts)
        expectation = Expectation(chunk, args.reuse == "boundary")
        scopes = {"main": secrets.token_hex(16), "other": secrets.token_hex(16)}
        _, observed = run_cases(
            cached,
            steps,
            args,
            session.tools,
            parser,
            scopes=scopes,
            expectation=expectation,
        )
    phase_seconds = {"cached": round(time.monotonic() - started, 1)}
    references = {}
    for name, binary in (("fresh", args.candidate), ("baseline", args.baseline)):
        report(args, phase=name)
        started = time.monotonic()
        with Worker(binary, args.model, progress=args.progress) as worker:
            references[name], _ = run_cases(worker, steps, args, session.tools, parser)
        phase_seconds[name] = round(time.monotonic() - started, 1)
    checks, invalidations = [], []
    for kind, label, _, _ in steps:
        if kind == "act":
            continue
        row = observed[label]
        checks.append(
            {
                "case": label,
                "input_tokens": row["input_tokens"],
                "reused_input_tokens": row["reused_input_tokens"],
                "reuse_matches_rule": row["reused_input_tokens"]
                == row["predicted_reuse"]
                and row["progress_consistent"],
                "identical_to_fresh": row["outcome"] == references["fresh"][row["key"]],
                "identical_to_baseline_worker": row["outcome"]
                == references["baseline"][row["key"]],
                "output_sha256": row["outcome"].get("generated_sha256"),
                "tool_calls": row["outcome"].get("tool_calls"),
                "finish_reason": row["outcome"].get("finish_reason"),
                "output_tokens": row["outcome"].get("output_tokens"),
                "successful": "error" not in row["outcome"],
            }
        )
    # Reuse must not survive a worker restart, even with the same host scope.
    report(args, phase="restart")
    with Worker(
        args.candidate, args.model, prefix_cache=True, progress=args.progress
    ) as restarted:
        reply = generate(
            restarted,
            turns[-1],
            context=args.context,
            max_new=args.max_new,
            scope=scopes["main"],
        )
        result = outcome(reply, session.tools, parser)
        key = comparison_key(turns[-1], {"context": args.context, "scope": "main"})
        checks.append(
            {
                "case": "worker_restart",
                "input_tokens": len(turns[-1]),
                "reused_input_tokens": reply.get("reused_input_tokens"),
                "reuse_matches_rule": reply.get("reused_input_tokens") == 0,
                "identical_to_fresh": result == references["fresh"][key],
                "identical_to_baseline_worker": result == references["baseline"][key],
                "output_sha256": result.get("generated_sha256"),
                "tool_calls": result.get("tool_calls"),
                "finish_reason": result.get("finish_reason"),
                "output_tokens": result.get("output_tokens"),
                "successful": "error" not in result,
            }
        )
    for label in (*INVALIDATIONS, "tokenize_decode"):
        before, after = observed[f"before_{label}"], observed[f"after_{label}"]
        invalidations.append(
            {
                "path": label,
                "action_reply": observed[label],
                "reused_before": before["reused_input_tokens"],
                "reused_after": after["reused_input_tokens"],
                "as_documented": before["reused_input_tokens"] > 0
                and (after["reused_input_tokens"] > 0) == (label == "tokenize_decode")
                and (
                    label != "cancellation" or observed[label] == "generation_cancelled"
                ),
            }
        )
    follow_ups = [
        check
        for check in checks
        if check["case"].startswith("follow_up_") and "stochastic" not in check["case"]
    ]
    return {
        "chunk_tokens": chunk,
        "reuse_rule": args.reuse,
        "mtp_drafts": mtp_drafts,
        "session_turn_tokens": [len(ids) for ids in turns],
        "phase_seconds": phase_seconds,
        "checks": checks,
        "invalidations": invalidations,
        "follow_ups_reuse": all(
            check["reused_input_tokens"] > 0 for check in follow_ups
        ),
    }


def run_timing(args, template, codec):
    """Alternate blocks of rounds between the workers, one loaded at a time."""
    samples = {"baseline": [], "prefix_cache": []}
    turns = None
    for block in range(args.timing_blocks):
        order = [
            ("baseline", args.baseline, False),
            ("prefix_cache", args.candidate, True),
        ]
        for name, binary, cache in order if block % 2 == 0 else order[::-1]:
            report(args, phase="timing", worker=name, block=block + 1)
            with Worker(
                binary, args.model, prefix_cache=cache, progress=args.progress
            ) as worker:
                if turns is None:
                    _, turns = build_turns(
                        worker,
                        template,
                        codec,
                        args.seed + 1,
                        args.first_tokens,
                        **session_options(args),
                    )
                generate(worker, turns[0][:2048], context=args.context, max_new=1)
                for _ in range(args.rounds_per_block):
                    scope = secrets.token_hex(16) if cache else None
                    row = []
                    for ids in turns:
                        reply = generate(
                            worker,
                            ids,
                            context=args.context,
                            max_new=args.timing_new,
                            scope=scope,
                        )
                        if "error" in reply:
                            raise SystemExit("timing request failed: " + reply["error"])
                        row.append(
                            {
                                "prefill_seconds": reply["prefill_seconds"],
                                "reused_input_tokens": reply.get(
                                    "reused_input_tokens", 0
                                ),
                                "output_sha256": digest_ids(
                                    reply["generated_token_ids"]
                                ),
                            }
                        )
                    samples[name].append(row)
    summary = []
    for index, ids in enumerate(turns):
        entry = {"turn": index + 1, "input_tokens": len(ids)}
        for name, rows in samples.items():
            times = [row[index]["prefill_seconds"] for row in rows]
            entry[name] = {
                "median_prefill_seconds": statistics.median(times),
                "min_prefill_seconds": min(times),
                "max_prefill_seconds": max(times),
                "reused_input_tokens": sorted(
                    {row[index]["reused_input_tokens"] for row in rows}
                ),
            }
        hashes = {
            row[index]["output_sha256"] for rows in samples.values() for row in rows
        }
        entry["outputs_identical"] = len(hashes) == 1
        summary.append(entry)
    return {
        "blocks": args.timing_blocks,
        "rounds_per_block": args.rounds_per_block,
        "max_new_tokens": args.timing_new,
        "turns": summary,
    }


def exactness_passed(exactness):
    return (
        bool(exactness["checks"])
        and all(
            check["successful"]
            and check["identical_to_fresh"]
            and check["identical_to_baseline_worker"]
            and check["reuse_matches_rule"]
            for check in exactness["checks"]
        )
        and all(item["as_documented"] for item in exactness["invalidations"])
        and exactness["follow_ups_reuse"]
    )


def timing_passed(timing):
    turns = timing["turns"]
    return bool(turns) and all(
        turn["outputs_identical"]
        and turn["baseline"]["reused_input_tokens"] == [0]
        and (
            turn["prefix_cache"]["reused_input_tokens"] == [0]
            if index == 0
            else min(turn["prefix_cache"]["reused_input_tokens"], default=0) > 0
        )
        for index, turn in enumerate(turns)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--download-receipt", type=Path,
                        help="Verified pinned EXL3 flat-pack receipt; checks all file identities before and after execution")
    parser.add_argument(
        "--baseline", required=True, help="Worker built without prefix reuse"
    )
    parser.add_argument(
        "--candidate", required=True, help="Worker built with prefix reuse"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse", choices=("rewind", "boundary"), default="rewind")
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--first-tokens", type=int, default=18560)
    parser.add_argument("--tools", type=int, default=24)
    parser.add_argument("--instruction-paragraphs", type=int, default=40)
    parser.add_argument(
        "--follow-up-sentences", type=int, nargs=3, default=(40, 25, 80)
    )
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--timing-new", type=int, default=8)
    parser.add_argument("--timing-blocks", type=int, default=2)
    parser.add_argument("--rounds-per-block", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mode", choices=("all", "exactness", "timing"), default="all")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print only phase names and aggregate case counters",
    )
    args = parser.parse_args()
    if (
        min(
            args.context,
            args.first_tokens,
            args.max_new,
            args.timing_new,
            args.timing_blocks,
            args.rounds_per_block,
            args.tools,
            args.instruction_paragraphs,
            *args.follow_up_sentences,
        )
        <= 0
    ):
        parser.error("session sizes and run counts must be positive")

    codec, template, template_provenance = load_validation_template(args.model, args.download_receipt)
    download_identity = (download_receipt_identity(args.model, args.download_receipt)
                         if args.download_receipt is not None else None)
    preflight_sessions(args, template, codec)
    tool_parser = tool_parser_for(codec)
    started = time.time()
    report(args, phase="artifact_hashing")
    record = {
        "schema": "stark-prefix-cache-validation-v4",
        "harness_sha256": file_sha256(Path(__file__)),
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
        "model": {
            "weights_sha256": weights_sha256(args.model, args.download_receipt),
            "config_sha256": file_sha256(args.model / "config.json"),
            "tokenizer_sha256": file_sha256(args.model / "tokenizer.json"),
            "metadata_sha256": {
                name: file_sha256(args.model / name)
                for name in (
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "native-source.json",
                    "conversion.json",
                )
                if (args.model / name).is_file()
            },
        },
        "workers": {
            "baseline_sha256": file_sha256(args.baseline),
            "candidate_sha256": file_sha256(args.candidate),
        },
        "context_tokens": args.context,
        "session": {
            "seed": args.seed,
            "first_tokens_target": args.first_tokens,
            **session_options(args),
        },
        "tool_parser": tool_parser,
    }
    if download_identity is not None:
        record["model"]["download_receipt"] = download_identity
        record["model"]["template_adapter"] = template_provenance
    if args.mode in ("all", "exactness"):
        record["exactness"] = run_exactness(args, template, codec, tool_parser)
    if args.mode in ("all", "timing"):
        record["timing"] = run_timing(args, template, codec)
    record["elapsed_seconds"] = round(time.time() - started, 1)
    exactness = record.get("exactness")
    if exactness is not None:
        record["passed"] = exactness_passed(exactness)
    if "timing" in record:
        record["passed"] = record.get("passed", True) and timing_passed(
            record["timing"]
        )
    if download_identity is not None:
        if download_receipt_identity(args.model, args.download_receipt) != download_identity:
            raise ValueError("EXL3 pack changed during validation")
        if (file_sha256(template._executable) != template_provenance["native_template_executable_sha256"]
                or file_sha256(Path(hybrid_prepare.__file__)) != template_provenance["metadata_normalization_sha256"]
                or file_sha256(Path(template_runtime.__file__)) != template_provenance["template_runtime_sha256"]):
            raise ValueError("chat template tooling changed during validation")
        record["model"]["download_guard_after_matched"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(json.dumps({"passed": record.get("passed"), "output": str(args.output)}))
    if record.get("passed") is False:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(json.dumps({"passed": False, "error": "validation_interrupted"}))
        sys.exit(130)
    # Validation failures must never dump request content from an exception.
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": "validation_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)
