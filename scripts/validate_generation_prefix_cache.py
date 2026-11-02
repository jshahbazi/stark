"""Model-bound validation and timing for generation prefix reuse.

Runs deterministic synthetic agent sessions, rendered by the model's own chat
template with tool definitions and tool-call history, against a worker that
starts every request from fresh state and against a worker started with
``--prefix-cache``. It checks that generated token IDs, decoded text and parsed
tool calls are identical, that every invalidation path reuses nothing, and
measures the native first-token time of first and follow-up turns.

Only counts, timings, booleans and SHA-256 hashes are written. Prompts,
generated text and token vectors stay in memory. Run from the repository root:

    uv run --project tools/converter --locked --python 3.13.13 python \\
        scripts/validate_generation_prefix_cache.py --model MODEL_DIR \\
        --baseline BASELINE_WORKER --candidate CANDIDATE_WORKER \\
        --output runtime/prefix-cache-validation.json
"""

import argparse
import hashlib
import json
import os
import platform
import random
import secrets
import select
import statistics
import subprocess
import sys
import time
from pathlib import Path

from stark_convert.api_tools import parse_tool_output, tool_parser_for
from stark_convert.chat import load_template
from stark_convert.local_api import APIError

CHUNK = 1024
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


class Worker:
    """One private worker process; replies are parsed in memory only."""

    def __init__(self, binary, model, *, prefix_cache=False, timeout=600):
        self.timeout = timeout
        self.cancel_read, self.cancel_write = os.pipe()
        command = [binary, "--model-path", model] + (
            ["--prefix-cache"] if prefix_cache else []
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=(self.cancel_read,),
            env=dict(os.environ, STARK_CANCEL_FD=str(self.cancel_read)),
        )
        os.close(self.cancel_read)
        self.request_id = 0
        self.ready = self.read()
        if self.ready.get("ready") is not True:
            raise SystemExit("worker failed to start: " + str(self.ready.get("error")))

    def read(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise SystemExit("worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise SystemExit("worker exited")
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
        self.process.stdin.close()
        self.process.wait(timeout=60)
        os.close(self.cancel_write)


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


def build_turns(worker, template, codec, seed, first_tokens):
    """Return token IDs for a first turn of about `first_tokens` and three follow-ups."""
    session = Session(seed)
    while len(worker.tokenize(session.render(template, codec))) < first_tokens - 400:
        session.add_tool_step(result_sentences=12)
    turns = [worker.tokenize(session.render(template, codec))]
    for grow in (
        lambda: session.add_tool_step(40),
        lambda: session.add_tool_step(25),
        lambda: session.add_user_turn(80),
    ):
        grow()
        turns.append(worker.tokenize(session.render(template, codec)))
    return session, turns


def generate(
    worker, ids, *, context, max_new, scope=None, sampling=None, seed=0, **extra
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
    if scope is not None:
        payload["cache_scope"] = scope
    return worker.request(payload)


def outcome(reply, tools, parser):
    """Content-free comparison key: generated IDs, text and parsed tool calls."""
    if "error" in reply:
        return {"error": reply["error"]}
    try:
        message, finish = parse_tool_output(
            reply["text"],
            tools,
            parser=parser,
            finish_reason=reply.get("finish_reason"),
            call_id_prefix="validation",
        )
        parsed = {"message": message, "finish_reason": finish}
    except APIError as error:
        parsed = {"error": error.code}
    return {
        "generated_sha256": digest_ids(reply["generated_token_ids"]),
        "text_sha256": digest_text(reply["text"]),
        "finish_reason": reply.get("finish_reason"),
        "tool_parse_sha256": digest_text(json.dumps(parsed, sort_keys=True)),
        "tool_calls": len(parsed.get("message", {}).get("tool_calls") or []),
    }


def run_exactness(args, template, codec, parser):
    reference = Worker(args.candidate, args.model)
    cached = Worker(args.candidate, args.model, prefix_cache=True)
    baseline = Worker(args.baseline, args.model)
    if cached.ready.get("prefix_cache", {}).get("enabled") is not True:
        raise SystemExit("candidate did not enable prefix reuse for this model")
    session, turns = build_turns(cached, template, codec, args.seed, args.first_tokens)
    scope = secrets.token_hex(16)
    context, other_context = args.context, args.context - 8192
    checks = []

    def compare(
        label, ids, *, expect_reuse, scope_value=scope, context_value=context, **options
    ):
        expected = outcome(
            generate(
                reference, ids, context=context_value, max_new=args.max_new, **options
            ),
            session.tools,
            parser,
        )
        legacy = outcome(
            generate(
                baseline, ids, context=context_value, max_new=args.max_new, **options
            ),
            session.tools,
            parser,
        )
        reply = generate(
            cached,
            ids,
            context=context_value,
            max_new=args.max_new,
            scope=scope_value,
            **options,
        )
        actual = outcome(reply, session.tools, parser)
        reused = reply.get("reused_input_tokens")
        progress_reuse = {
            frame.get("reused_input_tokens") for frame in reply["_progress"]
        }
        ok_reuse = expect_reuse(reused, len(ids)) and progress_reuse <= {reused}
        checks.append(
            {
                "case": label,
                "input_tokens": len(ids),
                "prompt_sha256": digest_ids(ids),
                "reused_input_tokens": reused,
                "identical_to_fresh": actual == expected,
                "identical_to_baseline_worker": actual == legacy,
                "reuse_expected": ok_reuse,
                "output_sha256": actual.get("generated_sha256"),
                "tool_calls": actual.get("tool_calls"),
                "finish_reason": actual.get("finish_reason"),
            }
        )

    none = lambda reused, length: reused == 0
    some = lambda reused, length: reused > 0 and reused % CHUNK == 0 and reused < length
    full = lambda previous: (
        lambda reused, length: reused == min(previous - 1, length - 1) // CHUNK * CHUNK
    )

    compare("first_turn_cold", turns[0], expect_reuse=none)
    for index in range(1, len(turns)):
        compare(
            f"follow_up_{index}", turns[index], expect_reuse=full(len(turns[index - 1]))
        )
    # Seeded stochastic sampling reuses the prefix and keeps its reference sequence.
    compare(
        "follow_up_repeat_stochastic",
        turns[-1],
        expect_reuse=full(len(turns[-1])),
        sampling={"mode": "stochastic", "temperature": 0.7, "top_k": 20, "top_p": 0.95},
        seed=17,
    )
    # A shorter, truncated conversation reuses its own aligned prefix only.
    compare("shrink", turns[1], expect_reuse=full(len(turns[1])))
    # An early divergence keeps only whole chunks before it.
    diverged = list(turns[-1])
    diverged[3 * CHUNK + 100] = diverged[3 * CHUNK + 101]
    compare(
        "diverge_early",
        diverged,
        expect_reuse=lambda reused, length: reused == 3 * CHUNK,
    )
    compare(
        "restore_after_divergence",
        turns[-1],
        expect_reuse=lambda reused, length: reused == 3 * CHUNK,
    )
    compare("context_change", turns[-1], expect_reuse=none, context_value=other_context)
    compare("context_restored", turns[-1], expect_reuse=none)
    compare(
        "scope_change", turns[-1], expect_reuse=none, scope_value=secrets.token_hex(16)
    )
    compare("scope_restored", turns[-1], expect_reuse=none)
    compare("warm_again", turns[-1], expect_reuse=some)

    invalidations = []
    for label, action in (
        (
            "error_reply",
            lambda: cached.request(
                {
                    "operation": "generate",
                    "input_token_ids": turns[0],
                    "context_tokens": 16,
                    "max_new_tokens": 1,
                    "cache_scope": scope,
                }
            ),
        ),
        (
            "unscoped_request",
            lambda: generate(cached, turns[0][:64], context=context, max_new=1),
        ),
        ("reset_cache", lambda: cached.request({"operation": "reset_cache"})),
        ("image_operation", lambda: cached.request({"operation": "discard_image"})),
        (
            "cancellation",
            lambda: cached.request(
                {
                    "operation": "generate",
                    "input_token_ids": turns[-1],
                    "context_tokens": context,
                    "max_new_tokens": 64,
                    "cache_scope": scope,
                    "stream": True,
                },
                cancel_after_first_progress=True,
            ),
        ),
    ):
        compare(
            f"before_{label}",
            turns[-1],
            expect_reuse=some if label != "cancellation" else lambda r, n: True,
        )
        acted = action()
        reply = generate(cached, turns[-1], context=context, max_new=1, scope=scope)
        invalidations.append(
            {
                "path": label,
                "action_reply": acted.get("error", "ok"),
                "reused_after": reply.get("reused_input_tokens"),
                "cleared": reply.get("reused_input_tokens") == 0,
            }
        )
    # Tokenize and decode between turns keep the entry.
    generate(cached, turns[-1], context=context, max_new=1, scope=scope)
    cached.tokenize("synthetic")
    cached.request({"operation": "decode", "token_ids": [1, 2, 3]})
    kept = (
        generate(cached, turns[-1], context=context, max_new=1, scope=scope).get(
            "reused_input_tokens", 0
        )
        > 0
    )
    for worker in (reference, cached, baseline):
        worker.close()
    return {
        "session_turn_tokens": [len(ids) for ids in turns],
        "checks": checks,
        "invalidations": invalidations,
        "tokenize_decode_keep_entry": kept,
    }


def run_timing(args, template, codec):
    baseline = Worker(args.baseline, args.model)
    cached = Worker(args.candidate, args.model, prefix_cache=True)
    _, turns = build_turns(cached, template, codec, args.seed + 1, args.first_tokens)
    warm = turns[0][: 2 * CHUNK]
    for worker, scope in ((baseline, None), (cached, secrets.token_hex(16))):
        generate(worker, warm, context=args.context, max_new=1, scope=scope)
    samples = {"baseline": [], "prefix_cache": []}
    for round_index in range(args.rounds):
        order = (("baseline", baseline), ("prefix_cache", cached))
        for name, worker in order if round_index % 2 == 0 else order[::-1]:
            scope = secrets.token_hex(16) if name == "prefix_cache" else None
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
                        "reused_input_tokens": reply.get("reused_input_tokens", 0),
                        "output_sha256": digest_ids(reply["generated_token_ids"]),
                    }
                )
            samples[name].append(row)
    baseline.close()
    cached.close()
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
        entry["outputs_identical"] = (
            len(
                {
                    row[index]["output_sha256"]
                    for rows in samples.values()
                    for row in rows
                }
            )
            == 1
        )
        summary.append(entry)
    return {"rounds": args.rounds, "max_new_tokens": args.timing_new, "turns": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--baseline", required=True, help="Worker built without prefix reuse"
    )
    parser.add_argument(
        "--candidate", required=True, help="Worker built with prefix reuse"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--first-tokens", type=int, default=18560)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--timing-new", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mode", choices=("all", "exactness", "timing"), default="all")
    args = parser.parse_args()

    codec, template = load_template(args.model)
    tool_parser = tool_parser_for(codec)
    if tool_parser is None:
        raise SystemExit("model template has no qualified tool parser")
    started = time.time()
    record = {
        "schema": "stark-prefix-cache-validation-v1",
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
        "model": {
            "weights_sha256": file_sha256(args.model / "model.safetensors"),
            "config_sha256": file_sha256(args.model / "config.json"),
            "tokenizer_sha256": file_sha256(args.model / "tokenizer.json"),
        },
        "workers": {
            "baseline_sha256": file_sha256(args.baseline),
            "candidate_sha256": file_sha256(args.candidate),
        },
        "context_tokens": args.context,
        "tool_parser": tool_parser,
    }
    if args.mode in ("all", "exactness"):
        record["exactness"] = run_exactness(args, template, codec, tool_parser)
    if args.mode in ("all", "timing"):
        record["timing"] = run_timing(args, template, codec)
    record["elapsed_seconds"] = round(time.time() - started, 1)
    exactness = record.get("exactness")
    if exactness is not None:
        record["passed"] = (
            all(
                c["identical_to_fresh"]
                and c["identical_to_baseline_worker"]
                and c["reuse_expected"]
                for c in exactness["checks"]
            )
            and all(item["cleared"] for item in exactness["invalidations"])
            and exactness["tokenize_decode_keep_entry"]
        )
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
    main()
