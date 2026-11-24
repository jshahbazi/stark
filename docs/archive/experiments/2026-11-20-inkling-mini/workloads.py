"""Create private, deterministic input fixtures for the two reference servers.

Use the release tokenizer and the repository's fixed-corpus code generator.
Output contains synthetic inputs. Supply an ignored private output directory.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from benchmark_lily_fixed_corpus import code_chunks


def request_text(tokenizer, seed, length, window):
    # The pinned Jinja template emits these headers for a text system turn,
    # effort 0.9, and one text user turn. Raw requests bypass server templates.
    prefix = (
        f"<|message_system|><|content_text|>{seed:x} is the case number. "
        "Answer the engineering request in detail.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
        "<|message_user|><|content_text|>"
    )
    suffix = (
        "\nUsing these records, write a detailed design review. Provide twelve numbered "
        "findings, with a concrete implementation example and an acceptance test for each. "
        "Explain how each example behaves during retry and partial failure.<|end_message|>"
    )
    if window % 2 == 0:
        payload = next(code_chunks(seed))
    else:
        rng = random.Random(seed)
        nouns = ("invoice", "queue", "shipment", "ledger", "snapshot", "manifest", "lease")
        actions = ("retries", "checks", "schedules", "commits", "reads", "renews")
        payload = "\n".join(
            f"In trial {i}, the {rng.choice(nouns)} worker {rng.choice(actions)} "
            f"the {rng.choice(nouns)} after {rng.randrange(3, 250)} milliseconds. "
            f"The next recovery window has {rng.randrange(1, 19)} entries."
            for i in range(350)
        )
    body_ids = tokenizer.encode(payload, add_special_tokens=False).ids
    overhead = len(tokenizer.encode(prefix + suffix, add_special_tokens=False).ids)
    for take in range(length - overhead - 2, length - overhead - 12, -1):
        body = tokenizer.decode(body_ids[:take], skip_special_tokens=False)
        text = prefix + body + suffix
        gap = length - len(tokenizer.encode(text, add_special_tokens=False).ids)
        if 0 <= gap <= 12:
            text = prefix + body + " ." * gap + suffix
            if len(tokenizer.encode(text, add_special_tokens=False).ids) == length:
                return text
    raise ValueError("Could not construct an exact-length fixture")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = []
    for window in range(12):
        cells = [(512, "greedy"), (4096, "sampled"), (4096, "greedy"), (512, "sampled")]
        if window % 2:
            cells.reverse()
        for length, mode in cells:
            seed = 20261120 + 997 * window + length + (71 if mode == "sampled" else 0)
            prompt = request_text(tokenizer, seed, length, window)
            cases.append({
                "id": f"{mode[0]}{length}-{window:02d}",
                "mode": mode,
                "input_tokens": length,
                "window": window,
                "seed": seed,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
    with os.fdopen(os.open(args.output, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w") as stream:
        json.dump({"version": 1, "cases": cases}, stream, ensure_ascii=False)
    print(f"Prepared {len(cases)} exact-length cases")


if __name__ == "__main__":
    main()
