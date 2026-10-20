"""Serial stock-engine benchmark; persist aggregate measurements and hashes only.

Configuration contains local executable/model paths and remains private. Engine
stdout replies stay in private pipes or memory. No request/reply payload is logged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import selectors
import signal
import socket
import statistics
import subprocess
import threading
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def token_sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop(process):
    owned_group = getattr(process, "_owned_group", None)
    if owned_group:
        try:
            os.killpg(owned_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        if not owned_group:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if owned_group:
                try:
                    os.killpg(owned_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=10)
    if owned_group:
        try:
            os.killpg(owned_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Pipe:
    def __init__(self, command, stack, cwd, env):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        self.process._owned_group = self.process.pid
        # ExitStack is LIFO: stop the process before closing private pipes.
        stack.callback(self.process.stdout.close)
        stack.callback(self.process.stdin.close)
        stack.callback(stop, self.process)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        stack.callback(self.selector.close)
        self.pending = bytearray()
        self.ready = self.read()
        if not self.ready.get("ready"):
            raise RuntimeError("worker_not_ready")

    def read(self):
        deadline = time.time() + 300
        while b"\n" not in self.pending:
            if not self.selector.select(max(0, deadline - time.time())):
                raise RuntimeError("worker_timeout")
            data = self.process.stdout.read(65536)
            if not data:
                raise RuntimeError("worker_closed")
            self.pending.extend(data)
            if len(self.pending) > 16 << 20:
                raise RuntimeError("worker_response_too_large")
        line, _, self.pending = self.pending.partition(b"\n")
        result = json.loads(line)
        if "error" in result:
            category = result["error"]
            raise RuntimeError(
                category
                if category
                in {
                    "model_load_failed",
                    "runtime_operation_failed",
                    "context_init_failed",
                    "decode_failed",
                    "nonfinite_output",
                }
                else "worker_request_failed"
            )
        return result

    def request(self, value):
        line = value if isinstance(value, str) else json.dumps(value)
        data = memoryview((line + "\n").encode())
        while data:
            written = self.process.stdin.write(data)
            if not written:
                raise RuntimeError("worker_write_failed")
            data = data[written:]
        return self.read()


def http(base, route, value=None):
    request = Request(
        base + route,
        data=None if value is None else json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def process_times():
    result = {}
    text = subprocess.check_output(
        ["/bin/ps", "-A", "-o", "pid=,ppid=,time=,comm="], text=True
    )
    for line in text.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        pid, parent, elapsed, command = fields
        parts = elapsed.split(":")
        try:
            seconds = sum(float(x) * 60**i for i, x in enumerate(reversed(parts)))
            result[int(pid)] = (int(parent), seconds, Path(command).name)
        except ValueError:
            continue
    return result


class HostMonitor:
    def __init__(self):
        self.done = threading.Event()
        self.max_background_cpu_percent = 0.0
        self.max_background_model_cpu_percent = 0.0
        self.samples = 0
        self.failures = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            previous = process_times()
            prior_time = time.time()
            while not self.done.wait(0.5):
                current = process_times()
                now = time.time()
                owned = {os.getpid()}
                for _ in range(12):
                    owned |= {
                        pid
                        for pid, (parent, _, _) in current.items()
                        if parent in owned
                    }
                for pid, (_, cpu, name) in current.items():
                    if pid not in previous or pid in owned:
                        continue
                    load = (
                        max(0, cpu - previous[pid][1])
                        / max(0.001, now - prior_time)
                        * 100
                    )
                    self.max_background_cpu_percent = max(
                        self.max_background_cpu_percent, load
                    )
                    if any(
                        x in name.lower()
                        for x in ("llama", "stark", "tensorfold", "vllm", "mlx")
                    ):
                        self.max_background_model_cpu_percent = max(
                            self.max_background_model_cpu_percent, load
                        )
                self.samples += 1
                previous, prior_time = current, now
        except Exception:
            self.failures += 1

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=5)

    def result(self):
        return {
            "samples": self.samples,
            "sampling_failures": self.failures,
            "max_background_process_cpu_percent": self.max_background_cpu_percent,
            "max_background_model_process_cpu_percent": self.max_background_model_cpu_percent,
            "cpu_guard_exceeded": self.max_background_cpu_percent > 200,
            "background_model_active": self.max_background_model_cpu_percent > 10,
        }


def sleep_events(start, end):
    try:
        value = subprocess.check_output(
            ["/usr/bin/pmset", "-g", "log"], timeout=30, text=True
        )
        count = 0
        for line in value.splitlines():
            if "Entering Sleep state" not in line:
                continue
            try:
                timestamp = datetime.strptime(
                    line[:25], "%Y-%m-%d %H:%M:%S %z"
                ).timestamp()
                if start <= timestamp <= end:
                    count += 1
            except ValueError:
                pass
        return {"checked": True, "sleep_events": count}
    except Exception:
        return {"checked": False, "sleep_events": None}


class Engine:
    def __init__(self, spec, stack, task_root, context):
        self.spec = spec
        self.kind = spec["kind"]
        cwd = Path(task_root) / "sessions" / spec["id"]
        cwd.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "XDG_CACHE_HOME": str(Path(task_root) / "cache"),
                "HF_HOME": str(Path(task_root) / "hf"),
            }
        )
        env.update(spec.get("env", {}))
        env.pop("TENSORFOLD_REQUEST_LOG", None)
        if self.kind == "vllm":
            short = stack.enter_context(tempfile.TemporaryDirectory(prefix="svm-"))
            env.update(
                {
                    "TMPDIR": short,
                    "VLLM_RPC_BASE_PATH": short,
                    "VLLM_CACHE_ROOT": str(Path(task_root) / "vllm/runtime-cache"),
                    "TORCHINDUCTOR_CACHE_DIR": str(
                        Path(task_root) / "vllm/torch-cache"
                    ),
                }
            )
        self.metadata = {}
        started = time.perf_counter()
        if self.kind in {"stark", "mlx", "stark_score", "llama_score"}:
            self.pipe = Pipe(spec["command"], stack, cwd, env)
            self.metadata = {
                k: self.pipe.ready[k]
                for k in ("mlx_version", "mlx_lm_version", "vocab_size", "cache_policy")
                if k in self.pipe.ready
            }
        else:
            port = free_port()
            command = [
                str(s).replace("{port}", str(port)).replace("{context}", str(context))
                for s in spec["command"]
            ]
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            process._owned_group = process.pid
            stack.callback(stop, process)
            self.process = process
            self.base = f"http://127.0.0.1:{port}"
            deadline = time.time() + 300
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("server_exited_before_ready")
                try:
                    with urlopen(self.base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError("server_readiness_timeout")
        self.load_seconds = time.perf_counter() - started

    def tokenize(self, text):
        if self.kind in {"stark", "mlx", "stark_score"}:
            return self.pipe.request(
                {"operation": "tokenize", "text": text, "add_special_tokens": False}
            )["token_ids"]
        if self.kind == "llama":
            return http(
                self.base, "/tokenize", {"content": text, "add_special": False}
            )["tokens"]
        # These servers are supplied the exact integer input IDs directly.
        return None

    def generate(self, fixture, context):
        ids, limit = fixture["ids"], fixture["limit"]
        start = time.perf_counter()
        if self.kind in {"stark", "mlx"}:
            result = self.pipe.request(
                {
                    "operation": "generate",
                    "input_token_ids": ids,
                    "context_tokens": context,
                    "max_new_tokens": limit,
                    "seed": 0,
                    "sampling": {"mode": "greedy"},
                }
            )
            tokens = result["generated_token_ids"]
            text = result["text"]
            row = {
                "input_tokens": result["input_tokens"],
                "output_tokens": len(tokens),
                "token_sha256": token_sha(tokens),
                "finish_reason": result["finish_reason"],
            }
        elif self.kind == "tensorfold":
            result = http(
                self.base,
                "/v1/completions",
                {
                    "model": "stock-qwen35",
                    "prompt": ids,
                    "max_tokens": limit,
                    "temperature": 0,
                    "seed": 0,
                    "stream": False,
                    "draft": self.spec.get("draft", True),
                },
            )
            runtime, usage = result["tensorfold"], result["usage"]
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached:
                raise RuntimeError("unexpected_cache_hit")
            row = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "token_sha256": runtime["token_sha256"],
                "finish_reason": result["choices"][0]["finish_reason"],
                "cached_tokens": cached,
            }
            text = result["choices"][0]["text"]
        elif self.kind == "llama":
            result = http(
                self.base,
                "/completion",
                {
                    "prompt": ids,
                    "n_predict": limit,
                    "temperature": 0,
                    "seed": 0,
                    "cache_prompt": False,
                    "stream": False,
                    "return_tokens": True,
                    "n_probs": 0,
                },
            )
            tokens = result["tokens"]
            text = result["content"]
            row = {
                "input_tokens": result["tokens_evaluated"],
                "output_tokens": len(tokens),
                "token_sha256": token_sha(tokens),
                "finish_reason": "length" if result.get("stopped_limit") else "stop",
                "cached_tokens": result.get("tokens_cached", 0),
            }
            # tokens_cached is the resident slot's state size, not necessarily reused work.
            row["prompt_evaluated_tokens"] = result.get("timings", {}).get("prompt_n")
        elif self.kind == "vllm":
            result = http(
                self.base,
                "/v1/completions",
                {
                    "model": "stock-qwen35",
                    "prompt": ids,
                    "max_tokens": limit,
                    "temperature": 0,
                    "seed": 0,
                    "stream": False,
                    "return_token_ids": True,
                },
            )
            choice, usage = result["choices"][0], result["usage"]
            tokens = choice.get("token_ids", choice.get("output_token_ids"))
            text = choice["text"]
            row = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "token_sha256": token_sha(tokens) if tokens is not None else None,
                "finish_reason": choice["finish_reason"],
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                ),
            }
            row["returned_token_id_count"] = len(tokens) if tokens is not None else None
            if tokens is not None and len(tokens) != usage["completion_tokens"]:
                raise RuntimeError("output_token_count_mismatch")
            if row["cached_tokens"]:
                raise RuntimeError("unexpected_cache_hit")
        else:
            raise RuntimeError("generation_unsupported")
        row["wall_seconds"] = time.perf_counter() - start
        if row["input_tokens"] != len(ids):
            raise RuntimeError("input_token_count_mismatch")
        row["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        row["fixed_work_eligible"] = (
            row["finish_reason"] == "length" and row["output_tokens"] == limit
        )
        if "answer" in fixture:
            normalized = "".join(text.strip().split()).lower()
            row["strict_sanity_correct"] = normalized == fixture["answer"]
        return row


def generation_fixtures(tokenizer):
    code = "\n".join(
        f"def section_{i}(items):\n    subtotal = 0\n    for value in items:\n        subtotal += value\n    return subtotal\n"
        for i in range(12)
    )
    items = [
        (
            "fibonacci-module",
            "Write a complete Python module for Fibonacci sequences, with detailed docstrings, three algorithms, complexity analysis, and ten unit tests. Include the full code.",
            128,
        ),
        (
            "gpu-explanation",
            "Explain how matrix multiplication uses a GPU. Give twelve worked numerical examples, then explain tiling, memory bandwidth, and synchronization in detail.",
            128,
        ),
        (
            "copy-edit",
            "Return the complete Python file with subtotal renamed to running_total everywhere, without explanation.\n```python\n"
            + code
            + "```",
            128,
        ),
        (
            "context-medium",
            "Summarize these project records into a detailed implementation plan with tests and risks.\n"
            + "\n".join(
                f"Record {i}: preserve deterministic ordering, validate input bounds, and report failures before retrying."
                for i in range(24)
            ),
            128,
        ),
        (
            "context-long",
            "Summarize these project records into a detailed implementation plan with tests and risks.\n"
            + "\n".join(
                f"Record {i}: preserve deterministic ordering, validate input bounds, and report failures before retrying."
                for i in range(100)
            ),
            128,
        ),
        (
            "long-output",
            "Write a complete Python module implementing a persistent immutable binary search tree. Include insertion, lookup, deletion, traversal, full docstrings, complexity analysis, and twenty tests. Return the full source.",
            512,
        ),
    ]
    quality = [
        (
            "arithmetic",
            "What is 17 multiplied by 23? Reply with the integer only.",
            "391",
        ),
        (
            "prime",
            "Which number is prime? A: 21. B: 29. C: 39. D: 49. Reply with only the letter.",
            "b",
        ),
        (
            "sorting",
            "Sort [3, 1, 2] in increasing order. Reply with the JSON array only.",
            "[1,2,3]",
        ),
        (
            "python",
            "What integer does sum(i*i for i in range(4)) return in Python? Reply with the integer only.",
            "14",
        ),
    ]
    fixtures = []
    for name, content, limit in items:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        fixtures.append(
            {
                "name": name,
                "ids": tokenizer.encode(rendered, add_special_tokens=False),
                "rendered": rendered,
                "limit": limit,
                "population": "performance",
            }
        )
    for name, content, answer in quality:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        fixtures.append(
            {
                "name": name,
                "ids": tokenizer.encode(rendered, add_special_tokens=False),
                "rendered": rendered,
                "limit": 64,
                "population": "sanity",
                "answer": answer,
            }
        )
    return fixtures


def summary(samples):
    walls = [s["wall_seconds"] for s in samples]
    return {
        "samples": len(samples),
        "median_wall_seconds": statistics.median(walls),
        "minimum_wall_seconds": min(walls),
        "maximum_wall_seconds": max(walls),
        "median_end_to_end_tokens_per_second": statistics.median(
            s["output_tokens"] / s["wall_seconds"] for s in samples
        ),
        "output_token_counts": sorted({s["output_tokens"] for s in samples}),
        "unique_token_hashes": len(
            {s["token_sha256"] for s in samples if s["token_sha256"]}
        ),
        "fixed_work_eligible_samples": sum(s["fixed_work_eligible"] for s in samples),
        "finish_reasons": sorted({s["finish_reason"] for s in samples}),
    }


def public_spec(spec):
    result = {
        k: spec[k]
        for k in ("id", "kind", "version", "source_commit", "representation", "draft")
        if k in spec
    }
    result["executable_sha256"] = sha(spec["command"][0])
    if spec.get("model_file"):
        result["model_sha256"] = sha(spec["model_file"])
    if spec.get("lock"):
        result["lock_sha256"] = sha(spec["lock"])
    return result


def run(config, output, probe):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["source"], local_files_only=True)
    fixtures = generation_fixtures(tokenizer)
    specs = config["engines"]
    report = {
        "schema": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "phase": "admission" if probe else "generation",
        "harness_sha256": sha(__file__),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "model_revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3",
        "host": {
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "chip": subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "memory_bytes": int(
                subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True
                )
            ),
        },
        "protocol": {
            "context_tokens": config["context"],
            "blocks": config["blocks"],
            "repeats_per_fixture_per_block": config["repeats"],
            "sampling": "greedy, seed 0",
            "cache_policy": "fresh state requested; retained/prefix cache disabled; missing server counters remain unknown",
            "model_load_in_primary_timing": False,
            "timing": "client complete request including engine work, transport and teardown",
            "host_screen_scope": "sampled per-process CPU and whole-window sleep checks; no continuous GPU/power/thermal instrumentation; preexisting idle model server left untouched",
            "early_stops": "retained; unequal output counts invalidate fixed-work speed ranking",
        },
        "fixtures": [
            {
                "name": f["name"],
                "input_tokens": len(f["ids"]),
                "output_limit": f["limit"],
                "input_token_sha256": token_sha(f["ids"]),
                "population": f["population"],
            }
            for f in fixtures
        ],
        "engines": [public_spec(s) for s in specs],
        "blocks": [],
        "failures": [],
    }
    save(output, report)
    start_wall, start_mono = time.time(), time.monotonic()
    # Prevent idle sleep only for this owned run; process is reaped on every exit.
    with contextlib.ExitStack() as outer:
        assertion = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        outer.callback(stop, assertion)
        for block in range(1 if probe else config["blocks"]):
            rotation = (block // 2) % len(specs)
            order = specs[rotation:] + specs[:rotation]
            if block % 2:
                order = list(reversed(order))
            for spec in order:
                row = {
                    "block": block,
                    "engine": spec["id"],
                    "status": "running",
                    "samples": [],
                }
                report["blocks"].append(row)
                try:
                    with contextlib.ExitStack() as stack, HostMonitor() as monitor:
                        engine = Engine(
                            spec, stack, config["task_root"], config["context"]
                        )
                        row["load_seconds"] = engine.load_seconds
                        row["runtime"] = engine.metadata
                        for fixture in fixtures:
                            checked = engine.tokenize(fixture["rendered"])
                            if checked is not None and checked != fixture["ids"]:
                                raise RuntimeError("tokenizer_mismatch")
                            if fixture["population"] == "performance":
                                engine.generate(fixture, config["context"])
                            count = (
                                1
                                if probe or fixture["population"] == "sanity"
                                else config["repeats"]
                            )
                            for repeat in range(count):
                                result = engine.generate(fixture, config["context"])
                                row["samples"].append(
                                    {
                                        "fixture": fixture["name"],
                                        "repeat": repeat,
                                        **result,
                                    }
                                )
                        row["status"] = "complete"
                    row["host_screen"] = monitor.result()
                    if not probe and (
                        row["host_screen"]["cpu_guard_exceeded"]
                        or row["host_screen"]["background_model_active"]
                        or row["host_screen"]["sampling_failures"]
                    ):
                        row["status"] = "excluded_interference"
                except Exception as exc:
                    safe = (
                        str(exc)
                        if isinstance(exc, RuntimeError)
                        and str(exc).replace("_", "").isalnum()
                        else "engine_operation_failed"
                    )
                    row.update(status="failed", error=safe)
                    report["failures"].append(
                        {"engine": spec["id"], "block": block, "error": safe}
                    )
                save(output, report)
                print(
                    json.dumps(
                        {
                            "completed_engine": spec["id"],
                            "block": block,
                            "status": row["status"],
                            "samples": len(row["samples"]),
                            "error": row.get("error"),
                        }
                    ),
                    flush=True,
                )
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    report["wall_elapsed_seconds"] = time.time() - start_wall
    report["monotonic_elapsed_seconds"] = time.monotonic() - start_mono
    report["sleep_clock_discrepancy_seconds"] = (
        report["wall_elapsed_seconds"] - report["monotonic_elapsed_seconds"]
    )
    report["power_log_screen"] = sleep_events(start_wall, time.time())
    report["timing_accepted"] = (
        report["power_log_screen"]["checked"]
        and report["power_log_screen"]["sleep_events"] == 0
        and abs(report["sleep_clock_discrepancy_seconds"]) < 2
        and all(b["status"] == "complete" for b in report["blocks"])
    )
    report["summary"] = {}
    for spec in specs:
        values = {}
        for fixture in fixtures:
            samples = [
                s
                for b in report["blocks"]
                if b["engine"] == spec["id"] and b["status"] == "complete"
                for s in b["samples"]
                if s["fixture"] == fixture["name"]
            ]
            if samples:
                values[fixture["name"]] = summary(samples)
                if fixture["population"] == "sanity":
                    values[fixture["name"]]["strict_sanity_correct_count"] = sum(
                        s["strict_sanity_correct"] for s in samples
                    )
        report["summary"][spec["id"]] = values
    report["status"] = "complete_with_failures" if report["failures"] else "complete"
    save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text()), args.output, args.probe)
    except Exception:
        raise SystemExit("stock_benchmark_failed") from None


if __name__ == "__main__":
    main()
