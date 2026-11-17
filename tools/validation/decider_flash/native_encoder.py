"""Private JSONL encoder bridge for Stark's protocol-v4 Decider formatter."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def quiet_output():
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(sys.stdout.fileno())
    stderr_fd = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            os.dup2(sink.fileno(), sys.stdout.fileno())
            os.dup2(sink.fileno(), sys.stderr.fileno())
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            os.dup2(stdout_fd, sys.stdout.fileno())
            os.dup2(stderr_fd, sys.stderr.fileno())
            os.close(stdout_fd)
            os.close(stderr_fd)


def build_encoder(snapshot: Path):
    from stark_convert.decider_protocol import Encoder
    from stark_convert.template_runtime import IsolatedTemplate
    from tokenizers import Tokenizer

    decision = json.loads((snapshot / "decision_config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    template = IsolatedTemplate((snapshot / "chat_template.jinja").read_text(encoding="utf-8"))
    return Encoder(tokenizer, template, decision["codes"])


def encode(encoder, request: dict, max_length: int) -> dict:
    plan = encoder.plan(request, max_length)
    return {
        "encodings": [
            {
                "keys": [row.key],
                "input_token_ids": list(row.input_ids),
                "readout_positions": [len(row.input_ids) - 1],
                "option_counts": [row.option_count],
                "segments": [row.context_length, len(row.input_ids) - row.context_length],
                "image_grid_thw": None,
            }
            for row in plan
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if args.max_length < 1:
        print(json.dumps({"error": "invalid_max_length"}), flush=True)
        return 2
    try:
        with quiet_output():
            encoder = build_encoder(args.snapshot)
    except Exception:
        print(json.dumps({"error": "native_encoder_initialization_failed"}), flush=True)
        return 3
    print(json.dumps({
        "ready": True,
        "encoding_only": True,
        "model_loaded": False,
        "max_length": args.max_length,
    }), flush=True)
    for line in sys.stdin:
        try:
            envelope = json.loads(line)
            request = envelope.get("request") if isinstance(envelope, dict) else None
            if not isinstance(request, dict):
                raise ValueError
            with quiet_output():
                result = encode(encoder, request, args.max_length)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception:
            print(json.dumps({"error": "native_encoding_failed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
