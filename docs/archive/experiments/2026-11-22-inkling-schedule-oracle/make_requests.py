"""Write private, deterministic capture requests for chosen prompt lengths.

For each case this writes, under an ignored private directory:

- `request.json` for the pinned capture runner: `Prompt`, `Start = N - 1`,
  the worker's bulk and tail chunks, and `FullTarget`.
- `tokens.safetensors`, the release tokenizer's N + 3 token IDs, so the Stark
  self-comparison checks can run before any capture. The capture runner
  later replaces this file with its own tokenization; `capture.py` reports
  whether the two agree.
- `expected.json` with the case geometry and SHA-256 values only.

The text generator is a parameterised copy of `request_text` in
`../2026-11-20-inkling-mini/workloads.py`. It keeps that file's message
headers and exact-length padding and uses its odd-window synthetic records,
which need no other repository module, with as many records as the length
requires. Prompts of fewer than 512 tokens use one short user turn because
the full headers alone exceed the shortest lengths. Nothing printed contains
prompt text or token IDs.
"""

import argparse
import hashlib
import json
import os
import random
import struct
from pathlib import Path

from tokenizers import Tokenizer

from schedule import CASES, TEACHER_ROWS, check_case, request_fields

FULL_PREFIX = (
    "<|message_system|><|content_text|>{seed:x} is the case number. "
    "Answer the engineering request in detail.<|end_message|>"
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
    "<|message_user|><|content_text|>"
)
FULL_SUFFIX = (
    "\nUsing these records, write a detailed design review. Provide twelve numbered "
    "findings, with a concrete implementation example and an acceptance test for each. "
    "Explain how each example behaves during retry and partial failure.<|end_message|>"
)
SHORT_PREFIX = "<|message_user|><|content_text|>"
SHORT_SUFFIX = "\nSummarize these records.<|end_message|>"
NOUNS = ("invoice", "queue", "shipment", "ledger", "snapshot", "manifest", "lease")
ACTIONS = ("retries", "checks", "schedules", "commits", "reads", "renews")


def count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def records(seed, needed, tokenizer):
    rng = random.Random(seed)
    lines = []
    while True:
        for _ in range(64):
            i = len(lines)
            lines.append(
                f"In trial {i}, the {rng.choice(NOUNS)} worker {rng.choice(ACTIONS)} "
                f"the {rng.choice(NOUNS)} after {rng.randrange(3, 250)} milliseconds. "
                f"The next recovery window has {rng.randrange(1, 19)} entries."
            )
        payload = "\n".join(lines)
        if count(tokenizer, payload) >= needed + 32:
            return payload


def request_text(tokenizer, seed, length):
    if length >= 512:
        prefix, suffix = FULL_PREFIX.format(seed=seed), FULL_SUFFIX
    else:
        prefix, suffix = SHORT_PREFIX, SHORT_SUFFIX
    overhead = count(tokenizer, prefix + suffix)
    if length - overhead < 16:
        raise ValueError("length is too short for the request frame")
    body_ids = tokenizer.encode(records(seed, length - overhead, tokenizer), add_special_tokens=False).ids
    for take in range(length - overhead - 2, length - overhead - 12, -1):
        body = tokenizer.decode(body_ids[:take], skip_special_tokens=False)
        text = prefix + body + suffix
        gap = length - count(tokenizer, text)
        if 0 <= gap <= 12:
            text = prefix + body + " ." * gap + suffix
            if count(tokenizer, text) == length:
                return text
    raise ValueError("Could not construct an exact-length fixture")


def write_private(path, data):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "wb") as stream:
        stream.write(data)


def tokens_safetensors(ids):
    header = json.dumps(
        {"tokens": {"dtype": "U32", "shape": [len(ids)], "data_offsets": [0, 4 * len(ids)]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack(f"<{len(ids)}I", *ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path, help="Release tokenizer.json")
    parser.add_argument("output", type=Path, help="New or existing ignored private directory")
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args()
    os.umask(0o077)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for case in filter(None, args.cases.split(",")):
        prompt, context, capture = CASES[case]
        check_case(prompt, context)
        seed = 20261122 + prompt
        text = request_text(tokenizer, seed, prompt + TEACHER_ROWS)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) != prompt + TEACHER_ROWS:
            raise ValueError("tokenizer length changed")
        directory = args.output / case
        directory.mkdir(parents=True, exist_ok=True)
        fields = request_fields(prompt)
        write_private(directory / "request.json", json.dumps({"Prompt": text, **fields}).encode())
        write_private(directory / "tokens.safetensors", tokens_safetensors(ids))
        token_sha = hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
        expected = {
            "case": case,
            "prompt_tokens": prompt,
            "context": context,
            "capture": capture,
            **{key: value for key, value in fields.items() if key != "FullTarget"},
            "seed": seed,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            "token_sha256": token_sha,
        }
        write_private(directory / "expected.json", json.dumps(expected, indent=1).encode())
        print(f"{case}: N={prompt} context={context} chunks={len(fields['Chunks'])} tokens_sha256={token_sha}")


if __name__ == "__main__":
    main()
