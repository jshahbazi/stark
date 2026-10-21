"""Aggregate-only generation comparison using existing, unmodified engine interfaces."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import selectors
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def token_sha(ids):
    # TensorFold's documented runtime fingerprint uses comma-separated decimal IDs.
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class Pipe:
    def __init__(self, command, stack):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=0)
        stack.callback(stop, self.process)
        stack.callback(self.process.stdout.close)
        stack.callback(self.process.stdin.close)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        stack.callback(self.selector.close)
        self.pending = bytearray()
        self.ready = self.read()
        if not self.ready.get("ready"):
            raise RuntimeError("worker_not_ready")

    def read(self):
        deadline = time.monotonic() + 180
        while b"\n" not in self.pending:
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise RuntimeError("worker_timeout")
            data = self.process.stdout.read(65536)
            if not data:
                raise RuntimeError("worker_closed")
            self.pending.extend(data)
            if len(self.pending) > 8 << 20:
                raise RuntimeError("worker_response_too_large")
        line, _, self.pending = self.pending.partition(b"\n")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError("worker_request_failed")
        return result

    def request(self, value):
        data = memoryview((json.dumps(value) + "\n").encode())
        while data:
            data = data[self.process.stdin.write(data):]
        return self.read()


def server(args, stack):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ)
    env.pop("TENSORFOLD_REQUEST_LOG", None)
    command = [str(args.tensorfold.resolve()), "serve", str(args.source.resolve()),
               "--name", "stock-qwen35", "--host", "127.0.0.1", "--port", str(port),
               "--parallel", "1", "--context", "2048", "--prompt-cache-gib", "0", "--keep-warm", "0"]
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stack.callback(stop, process)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("stock_server_stopped")
        try:
            with urlopen(base + "/health", timeout=1) as response:
                if response.status == 200:
                    return base
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("stock_server_not_ready")


def tensorfold(base, ids, count, drafts):
    value = {"model": "stock-qwen35", "prompt": ids, "max_tokens": count,
             "temperature": 0, "seed": 0, "stream": False, "draft": drafts}
    req = Request(base + "/v1/completions", data=json.dumps(value).encode(),
                  headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(req, timeout=180) as response:
        result = json.load(response)
    wall = time.perf_counter() - start
    runtime, usage = result["tensorfold"], result["usage"]
    if usage["prompt_tokens"] != len(ids) or usage["prompt_tokens_details"]["cached_tokens"] != 0:
        raise RuntimeError("unexpected_prompt_or_cache")
    return {"wall_seconds": wall, "output_tokens": usage["completion_tokens"],
            "finish_reason": result["choices"][0]["finish_reason"],
            "token_sha256": runtime["token_sha256"],
            "text_sha256": hashlib.sha256(result["choices"][0]["text"].encode()).hexdigest(),
            "server_decode_tps": runtime.get("tokens_per_second"),
            "server_prefill_seconds": runtime.get("prefill_seconds"),
            "speculative": runtime.get("speculative"), "drafts": runtime.get("drafts")}


def stark(worker, ids, count):
    start = time.perf_counter()
    result = worker.request({"operation": "generate", "input_token_ids": ids, "context_tokens": 2048,
                             "max_new_tokens": count, "seed": 0, "sampling": {"mode": "greedy"}})
    wall = time.perf_counter() - start
    if result["input_tokens"] != len(ids):
        raise RuntimeError("unexpected_prompt_length")
    return {"wall_seconds": wall, "output_tokens": result["output_tokens"],
            "sampled_tokens": result["sampled_tokens"], "finish_reason": result["finish_reason"],
            "token_sha256": token_sha(result["generated_token_ids"]),
            "text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
            "generation_seconds": result["generation_seconds"]}


def run(args):
    tok = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    code = "\n".join(f"def section_{i}(items):\n    subtotal = 0\n    for value in items:\n        subtotal += value\n    return subtotal\n" for i in range(12))
    fixtures = [
        ("fibonacci", "Write a short Python function that computes the Fibonacci sequence and explain it."),
        ("gpu-explanation", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."),
        ("copy-edit", "Rewrite the following Python file, changing subtotal to running_total everywhere. Return the complete updated file and no explanation.\n```python\n" + code + "```"),
    ]
    report = {"schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": "mlx-community/Qwen3.5-2B-MLX-4bit",
              "revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3",
              "stark_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "tensorfold_version": subprocess.check_output([str(args.tensorfold), "--version"], text=True).strip(),
              "tensorfold_binary_sha256": sha(args.tensorfold), "stark_binary_sha256": sha(args.stark),
              "harness_sha256": sha(__file__), "tools_lock_sha256": sha("tools/uv.lock"),
              "source_config_sha256": sha(args.source / "config.json"),
              "stark_conversion": json.loads((args.stark_model / "conversion.json").read_text()),
              "host": {"macos": platform.mac_ver()[0], "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                       "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))},
              "protocol": {"context_tokens": 2048, "output_limit": args.tokens, "rounds": args.rounds,
                           "sampling": "greedy", "cache": "fresh requests; retained prompt cache disabled",
                           "timing": "client end-to-end wall time including prefill, generation, cleanup and transport; model load excluded"},
              "fixtures": []}
    with contextlib.ExitStack() as stack:
        base = server(args, stack)
        worker = Pipe([str(args.stark.resolve()), "--model-path", str(args.stark_model.resolve())], stack)
        for name, prompt in fixtures:
            rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(rendered, add_special_tokens=False)
            check = worker.request({"operation": "tokenize", "text": rendered, "add_special_tokens": False})
            if check["token_ids"] != ids:
                raise RuntimeError("stark_tokenization_differs")
            modes = {"tensorfold_default": lambda: tensorfold(base, ids, args.tokens, True),
                     "tensorfold_plain": lambda: tensorfold(base, ids, args.tokens, False),
                     "stark_default": lambda: stark(worker, ids, args.tokens)}
            for call in modes.values():
                call()
            samples = {key: [] for key in modes}
            names = list(modes)
            for rep in range(args.rounds):
                order = names[rep % len(names):] + names[:rep % len(names)]
                for key in order:
                    samples[key].append(modes[key]())
            summary = {}
            for key, values in samples.items():
                summary[key] = {"median_wall_seconds": statistics.median(v["wall_seconds"] for v in values),
                                "median_end_to_end_tokens_per_second": statistics.median(v["output_tokens"] / v["wall_seconds"] for v in values),
                                "output_token_counts": sorted({v["output_tokens"] for v in values}),
                                "unique_token_hashes": len({v["token_sha256"] for v in values})}
            default_hashes = {v["token_sha256"] for v in samples["tensorfold_default"]}
            plain_hashes = {v["token_sha256"] for v in samples["tensorfold_plain"]}
            stark_hashes = {v["token_sha256"] for v in samples["stark_default"]}
            row = {"name": name, "input_tokens": len(ids), "prompt_token_sha256": token_sha(ids),
                   "summary": summary, "samples": samples,
                   "tensorfold_draft_plain_tokens_match": len(default_hashes | plain_hashes) == 1,
                   "cross_engine_tokens_match": len(default_hashes | stark_hashes) == 1}
            report["fixtures"].append(row)
            print(json.dumps({"completed": name, "summary": summary}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "stark-model", "tensorfold", "stark", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--rounds", type=int, default=6)
    args = p.parse_args()
    if args.tokens < 1 or args.tokens > 512 or args.rounds < 3:
        p.error("tokens must be 1..512 and rounds at least 3")
    try:
        run(args)
    except Exception:
        raise SystemExit("stock_comparison_failed") from None


if __name__ == "__main__":
    main()
