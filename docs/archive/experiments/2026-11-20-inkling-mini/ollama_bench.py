"""Measure raw, serial MLX requests against an isolated local Ollama server."""

import argparse
import hashlib
import http.client
import json
import time
from pathlib import Path


def request(port, model, case, output_tokens):
    options = {
        "num_ctx": 8192,
        "num_predict": output_tokens,
        "draft_num_predict": 0,
        "temperature": 0 if case["mode"] == "greedy" else 1,
        "seed": case["seed"],
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }
    body = json.dumps({
        "model": model, "prompt": case["prompt"], "raw": True, "stream": True,
        "keep_alive": "20m", "truncate": False, "options": options,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    digest = hashlib.sha256()
    text_bytes = 0
    replacement_character = False
    first = None
    final = None
    start_unix_ns = time.time_ns()
    started = time.perf_counter()
    try:
        conn.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"generate returned HTTP {response.status}")
        for line in response:
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError("generate stream reported an error")
            fragment = chunk.get("response", "")
            if fragment:
                if first is None:
                    first = time.perf_counter()
                encoded = fragment.encode()
                digest.update(encoded)
                text_bytes += len(encoded)
                replacement_character |= "\ufffd" in fragment
            if chunk.get("done"):
                final = chunk
        ended = time.perf_counter()
        end_unix_ns = time.time_ns()
    finally:
        conn.close()
    if final is None:
        raise RuntimeError("missing final frame")
    seconds = float(final.get("eval_duration", 0)) / 1e9
    prefill = float(final.get("prompt_eval_duration", 0)) / 1e9
    count = final.get("eval_count", 0)
    input_count = final["prompt_eval_count"]
    cached = final.get("prompt_eval_cached_count")
    return {
        "id": case["id"], "window": case["window"], "mode": case["mode"],
        "start_unix_ns": start_unix_ns, "end_unix_ns": end_unix_ns,
        "seed": case["seed"], "requested_input_tokens": case["input_tokens"],
        "prompt_sha256": case["prompt_sha256"], "text_sha256": digest.hexdigest(),
        "text_bytes": text_bytes, "replacement_character": replacement_character,
        "prompt_eval_count": input_count, "prompt_eval_cached_count": cached,
        "requested_outputs": output_tokens, "eval_count": count,
        "done_reason": final.get("done_reason"),
        "prefill_seconds": prefill, "decode_seconds": seconds,
        "prefill_tps": (input_count - cached) / prefill if cached is not None and prefill > 0 else None,
        "useful_decode_tps": (count - 1) / seconds if count > 1 and seconds > 0 else None,
        "first_visible_seconds": first - started if first is not None else None,
        "wall_seconds": ended - started, "load_seconds": final.get("load_duration", 0) / 1e9,
    }


def await_host(path):
    """Read the external sampler's last observation before launching a request."""
    started = time.monotonic()
    while True:
        with path.open("rb") as samples:
            samples.seek(max(0, path.stat().st_size - 8192))
            lines = samples.read().splitlines()
        if lines:
            try:
                sample = json.loads(lines[-1])
            except json.JSONDecodeError:
                sample = {}
            if time.time_ns() - sample.get("unix_ns", 0) < 10_000_000_000:
                if sample.get("thermal") in (0, 1) and sample.get("pressure") == 1:
                    return time.monotonic() - started
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--host-samples", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.workloads.read_text())["cases"]
    with args.receipts.open("x", encoding="utf-8", buffering=1) as out:
        for case in cases:
            if case["window"] not in args.windows or (args.ids and case["id"] not in args.ids):
                continue
            if hashlib.sha256(case["prompt"].encode()).hexdigest() != case["prompt_sha256"]:
                raise ValueError(f"fixture hash mismatch for {case['id']}")
            waited = await_host(args.host_samples) if args.host_samples else 0
            result = request(args.port, args.model, case, args.output_tokens)
            result["host_wait_seconds"] = waited
            print(json.dumps(result), file=out)
            print(case["id"], result["prompt_eval_count"], result["eval_count"],
                  result["prompt_eval_cached_count"], round(result["wall_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
