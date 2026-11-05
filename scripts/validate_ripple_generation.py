"""Check bounded full-checkpoint Ripple generation without recording content.

Run through the locked converter environment. The two elementary cases are
functional smoke checks, not a model-quality evaluation. Requests, responses,
token IDs and model scores remain in memory; receipts contain hashes and
aggregate checks only. The native worker runs alone during this check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from stark_convert.chat import Worker, load_template, render_template


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("ripple_generation_smoke_failed")


def identity(result: dict) -> str:
    fields = {key: result.get(key) for key in ("text", "generated_token_ids", "finish_reason", "stop_token_id")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def run(args) -> dict:
    config = json.loads((args.model / "config.json").read_bytes())
    require(config.get("model_type") == "ricursive")
    codec, template = load_template(args.model)
    peaks = []

    def memory(_worker, amount):
        if type(amount) is int:
            peaks.append(amount)

    worker = Worker(args.worker.resolve(), args.model.resolve(), 3600, gpu_memory_callback=memory)
    try:
        ready = worker.read()
        require(ready.get("ready") is True and ready.get("state_policy") == "fresh-per-request")

        def generate(content):
            prompt = render_template(
                template,
                messages=[{"role": "user", "content": content}],
                tools=None,
                add_generation_prompt=True,
                bos_token=codec.get("bos_token") or "",
                eos_token=codec.get("eos_token") or "",
            )
            result = worker.request({
                "operation": "generate",
                "rendered_prompt": prompt,
                "context_tokens": 4096,
                "max_new_tokens": 128,
                "sampling": {"mode": "greedy"},
            })
            require("error" not in result and result.get("sampled_tokens", 0) > 0)
            return result

        arithmetic = generate("What is 19 + 23? Reply with just the answer.")
        other = generate("Reply with exactly the word STARK.")
        replay = generate("What is 19 + 23? Reply with just the answer.")
        failed = worker.request({
            "operation": "generate",
            "input_token_ids": [config["vocab_size"]],
            "context_tokens": 4096,
            "max_new_tokens": 1,
        })
        require(failed.get("error") == "invalid_generation_request")
        recovered = generate("What is 19 + 23? Reply with just the answer.")

        def answer(result):
            return re.sub(r"<think>.*?</think>", "", result["text"], flags=re.DOTALL).strip()

        checks = {
            "arithmetic": bool(re.fullmatch(r"42\.?", answer(arithmetic))),
            "literal_instruction": answer(other) == "STARK",
            "intervening_request_isolation": identity(arithmetic) == identity(replay),
            "invalid_request_rejected": True,
            "failure_recovery_exact": identity(arithmetic) == identity(recovered),
        }
        receipts = [{
            "output_sha256": identity(result),
            **{key: result.get(key) for key in (
                "input_tokens", "output_tokens", "sampled_tokens", "finish_reason",
                "prefill_seconds", "decode_seconds", "output_tokens_per_second",
            )},
        } for result in (arithmetic, other, replay, recovered)]
        return {
            "scope": "bounded_full_checkpoint_functional_smoke",
            "passed": all(checks.values()),
            "checks": checks,
            "sampling": "greedy",
            "context_tokens": 4096,
            "max_new_tokens": 128,
            "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "peak_reported_gpu_memory_bytes": max(peaks, default=None),
            "requests": receipts,
        }
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        require(not args.output.exists())
        require(subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(args.output.resolve())], cwd=root, check=False
        ).returncode == 0)
        result = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except Exception:  # noqa: BLE001 - worker exceptions must not expose private content.
        print(json.dumps({"passed": False, "reason": "ripple_generation_smoke_failed"}))
        return 1
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
