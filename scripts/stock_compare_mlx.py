"""Aggregate-only JSON-lines worker for an unmodified stock MLX-LM runtime.

The worker deliberately imports MLX-LM lazily.  It is intended to run in a
task-owned environment and must never be given private prompts or model data
that should be persisted.  Responses contain hashes and scalar measurements;
score arrays are returned only on the private pipe and are never logged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def digest_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def error(category: str) -> None:
    print(json.dumps({"error": category}), flush=True)


def load_runtime(model_path: Path):
    """Load only through stock MLX-LM's public loader."""
    from mlx_lm.utils import load  # type: ignore

    started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    return model, tokenizer, time.perf_counter() - started


def tokenizer_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in encoded]


def generate(model, tokenizer, request: dict):
    # generate() is the stock MLX-LM generation API; no model adapter or
    # alternate sampling implementation is introduced here.
    from mlx_lm import stream_generate  # type: ignore
    from mlx_lm.sample_utils import make_sampler  # type: ignore

    prompt = [int(value) for value in request["input_token_ids"]]
    limit = int(request["max_new_tokens"])
    started = time.perf_counter()
    generated_ids: list[int] = []
    text_parts: list[str] = []
    finish = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=limit,
        sampler=make_sampler(temp=0.0),
    ):
        # MLX-LM emits a final response carrying the last token again.  The
        # final response supplies the finish reason and any detokenizer tail;
        # only non-final responses contribute token IDs.
        if response.finish_reason is None:
            generated_ids.append(int(response.token))
            text_parts.append(response.text)
        else:
            finish = response.finish_reason
            if finish == "length":
                generated_ids.append(int(response.token))
            if response.text:
                text_parts.append(response.text)
    elapsed = time.perf_counter() - started
    text = "".join(text_parts)
    return {
        "input_tokens": len(prompt),
        "output_tokens": len(generated_ids),
        "sampled_tokens": len(generated_ids),
        "finish_reason": finish
        or ("length" if len(generated_ids) >= limit else "stop"),
        "generated_token_ids": generated_ids,
        "token_sha256": digest_ids(generated_ids),
        "text_sha256": digest_text(text),
        "text": text,
        "generation_seconds": elapsed,
        "wall_seconds": elapsed,
    }


def score(model, tokenizer, request: dict):
    """Score candidates through the stock model call, if its API admits it."""
    import mlx.core as mx  # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore

    ids = [int(value) for value in request["input_token_ids"]]
    candidates = [int(value) for value in request["candidate_ids"]]
    if not candidates:
        raise ValueError("empty_candidate_set")
    started = time.perf_counter()
    # The ordinary MLX-LM model callable is the only model path used.  A
    # missing/unsupported callable is reported as an admission failure.
    cache = make_prompt_cache(model)
    # Match the stock generation prefill pattern: populate the cache, then
    # project only the final input row into the vocabulary.
    for offset in range(0, len(ids) - 1, 2048):
        prefix = ids[offset : min(len(ids) - 1, offset + 2048)]
        model(mx.array([prefix]), cache=cache)
        mx.eval([entry.state for entry in cache])
    outputs = model(mx.array([[ids[-1]]]), cache=cache)
    last = outputs[0, -1].astype(mx.float32)
    normalizer = mx.logsumexp(last)
    mx.eval(last, normalizer)
    values = [float(last[candidate].item()) for candidate in candidates]
    log_probs = [
        float((last[candidate] - normalizer).item()) for candidate in candidates
    ]
    elapsed = time.perf_counter() - started
    return {
        "input_tokens": len(ids),
        "candidate_count": len(candidates),
        "logit_sha256": hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest(),
        "candidate_log_probabilities": log_probs,
        "finite_logits": all(
            value == value and abs(value) != float("inf") for value in values
        ),
        "finite_log_probabilities": all(
            value == value and abs(value) != float("inf") for value in log_probs
        ),
        "score_seconds": elapsed,
        "candidate_logits": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            model, tokenizer, load_seconds = load_runtime(args.model_path)
    except Exception:
        error("model_load_failed")
        return 2
    print(
        json.dumps(
            {
                "ready": True,
                "load_seconds": load_seconds,
                "model_path": str(args.model_path),
                "mlx_lm_version": importlib.metadata.version("mlx-lm"),
                "mlx_version": importlib.metadata.version("mlx"),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("operation")
            if operation == "tokenize":
                ids = tokenizer_ids(tokenizer, str(request["text"]))
                print(
                    json.dumps(
                        {
                            "token_ids": ids,
                            "input_tokens": len(ids),
                            "token_sha256": digest_ids(ids),
                        }
                    ),
                    flush=True,
                )
            elif operation == "generate":
                result = generate(model, tokenizer, request)
                # Keep text/token IDs private to the pipe; the parent may
                # reduce them to hashes before writing an aggregate report.
                print(json.dumps(result), flush=True)
            elif operation == "score":
                print(json.dumps(score(model, tokenizer, request)), flush=True)
            else:
                error("unsupported_operation")
        except ValueError as exc:
            error(
                str(exc) if str(exc) in {"empty_candidate_set"} else "invalid_request"
            )
        except Exception:
            error("runtime_operation_failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
