"""Serialize full-checkpoint worker timings; emit only aggregates and hashes."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from stark_convert.chat import ChatError, Worker, load_template


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    codec, template = load_template(args.model)
    prompt = template.render(
        messages=[{"role": "user", "content": "Hello."}],
        tools=None,
        add_generation_prompt=True,
        enable_thinking=True,
        bos_token=codec.get("bos_token") or "",
        eos_token=codec.get("eos_token") or "",
    )
    report = {
        "schema_version": 1,
        "scope": "serialized_native_text_greedy",
        "worker_sha256": sha(args.worker),
        "configuration_sha256": sha(args.model / "config.json"),
        "source_receipt_sha256": sha(args.model / "native-source.json"),
        "requested_output_tokens": args.tokens,
        "replays": [],
    }
    worker = Worker(args.worker.resolve(), args.model.resolve(), 3600)
    try:
        if worker.read().get("ready") is not True:
            raise ValueError("worker_not_ready")
        tokenized = worker.request(
            {"operation": "tokenize", "text": prompt, "add_special_tokens": False}
        )
        ids = tokenized["token_ids"]
        report["input_tokens"] = len(ids)
        report["input_sha256"] = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        for replay in range(args.replays):
            started = time.monotonic()
            result = worker.request(
                {
                    "operation": "generate",
                    "input_token_ids": ids,
                    "context_tokens": 512,
                    "max_new_tokens": args.tokens,
                    "sampling": {"mode": "greedy"},
                    "request_id": replay + 1,
                }
            )
            if "error" in result or result.get("sampled_tokens", 0) < 2:
                raise ValueError("generation_failed_or_too_short")
            item = {
                "replay": replay,
                "wall_seconds": time.monotonic() - started,
                "sampled_tokens": result["sampled_tokens"],
                "finish_reason": result["finish_reason"],
                "output_sha256": hashlib.sha256(
                    json.dumps(
                        result["generated_token_ids"], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            for key in (
                "prefill_seconds",
                "decode_seconds",
                "input_tokens_per_second",
                "output_tokens_per_second",
                "generation_seconds",
            ):
                item[key] = result[key]
            report["replays"].append(item)
            print(json.dumps({"completed_replay": replay}), flush=True)
        stable = report["replays"][1:] or report["replays"]
        report["median_warm_decode_tokens_per_second"] = statistics.median(
            item["output_tokens_per_second"] for item in stable
        )
        report["median_warm_prefill_tokens_per_second"] = statistics.median(
            item["input_tokens_per_second"] for item in stable
        )
        report["repeat_exact"] = (
            len({item["output_sha256"] for item in report["replays"]}) == 1
        )
        if not report["repeat_exact"]:
            raise ValueError("nondeterministic_replay")
        report["passed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True))
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32, choices=range(8, 129))
    parser.add_argument("--replays", type=int, default=3, choices=range(2, 9))
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, ChatError):
        print(json.dumps({"passed": False, "stage": "native_benchmark_failed"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
