"""Aggregate-only greedy GLM baseline against a local MLX-Serve process.

The GGUF model and rendered prompt stay in memory/private pipes. The JSON
receipt contains identities, scalar timings/counts and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(base: str, path: str, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urlopen(request, timeout=timeout) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("response_too_large")
    return json.loads(data)


def safe_settings(value):
    """Keep only documented scalar operational settings from /props."""
    if not isinstance(value, dict):
        return None
    source = value.get("settings", value)
    if not isinstance(source, dict):
        return None
    keep = ("version", "engine", "kv_quant", "kv_attn_mode", "decode_attn_quant",
            "prefill_chunk", "mtp", "drafter", "max_concurrent")
    out = {key: source[key] for key in keep if key in source and isinstance(source[key], (str, int, float, bool, type(None)))}
    cache = source.get("prefix_cache")
    if isinstance(cache, dict):
        out["prefix_cache"] = {
            key: cache[key] for key in ("ram_enabled", "mem_bytes", "disk_bytes")
            if key in cache and isinstance(cache[key], (str, int, float, bool, type(None)))
        }
    return out


def model_identity(base, requested_engine):
    value = http_json(base, "/v1/models", timeout=10)
    rows = value.get("data") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("model_list_invalid")
    row = rows[0]
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        raise ValueError("model_identity_invalid")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    actual_engine = meta.get("engine")
    if requested_engine != "auto" and actual_engine != requested_engine:
        raise ValueError("engine_selection_mismatch")
    allowed_meta = ("engine", "architecture", "vocab_size", "hidden_size", "num_layers", "quantization")
    return row["id"], {
        "engine": actual_engine,
        "model_id_basename": Path(row["id"]).name,
        "loaded": row.get("loaded") is True,
        "state": row.get("state") if isinstance(row.get("state"), str) else None,
        "context_length": row.get("context_length") if isinstance(row.get("context_length"), int) else None,
        "meta": {key: meta[key] for key in allowed_meta if key in meta and isinstance(meta[key], (str, int, float, bool, type(None)))},
    }


def model_files(path: Path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return [
        {"name": p.relative_to(path).as_posix() if path.is_dir() else p.name,
         "bytes": p.stat().st_size}
        for p in files
    ]


def weight_identity(model: Path, manifest: Path | None):
    if manifest is None:
        return None
    receipt = json.loads(manifest.read_text())
    directory = model if model.is_dir() else model.parent
    rows = {}
    for name, expected in receipt.items():
        if Path(name).name != name or not name.endswith(".gguf"):
            raise ValueError("weight_manifest_invalid")
        stat = (directory / name).stat()
        if stat.st_size != expected["bytes"]:
            raise ValueError("weight_size_changed")
        rows[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "inode": stat.st_ino, "prior_verified_sha256": expected["sha256"]}
    return {"prior_manifest_sha256": sha(manifest), "files": rows,
            "verification": "Prior content hashes with current size/mtime/inode guards; payload hashes were not recomputed."}


def quiet_admission(monitor):
    monitor.set_phase("quiet")
    deadline = time.monotonic() + 120
    streak = None
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if monitor.failures:
            raise ValueError("host_monitor_failed")
        if not monitor.rows:
            continue
        row = monitor.rows[-1]
        quiet = (
            row["external_cpu_percent"] <= 150
            and row["max_process_cpu_percent"] <= 80
            and row["max_model_cpu_percent"] <= 10
            and row["gpu_device_utilization_percent"] <= 5
            and row["thermal_state"] <= 1
        )
        if not quiet:
            streak = None
        elif streak is None:
            streak = time.monotonic()
        if streak is not None and time.monotonic() - streak >= 10:
            admitted = [item for item in monitor.rows if item["end_seconds"] >= monitor.rows[-1]["end_seconds"] - 10]
            return {
                "streak_seconds": time.monotonic() - streak,
                "samples": len(admitted),
                "external_cpu_max": max(item["external_cpu_percent"] for item in admitted),
                "max_process_cpu": max(item["max_process_cpu_percent"] for item in admitted),
                "external_model_cpu_max": max(item["max_model_cpu_percent"] for item in admitted),
                "gpu_utilization_max": max(item["gpu_device_utilization_percent"] for item in admitted),
                "thermal_max": max(item["thermal_state"] for item in admitted),
            }
    raise TimeoutError("quiet_admission_timeout")


def screen_attempt(monitor, phase, before, after):
    monitor.set_phase("between_requests")
    time.sleep(0.6)
    rows = [row for row in monitor.rows if phase in row["phases"]]
    if not rows:
        raise ValueError("host_observations_missing")
    compiler = any(
        item["name"] in {"cargo", "rustc", "(rustc)", "clang", "clang++", "ld"}
        for row in rows for item in row["active_processes"]
    )
    result = {
        "samples": len(rows),
        "external_cpu_max": max(row["external_cpu_percent"] for row in rows),
        "external_model_cpu_max": max(row["max_model_cpu_percent"] for row in rows),
        "thermal_max": max(row["thermal_state"] for row in rows),
        "compiler_active": compiler,
        "sampling_gap": any(row["interval_seconds"] > 2 for row in rows),
        "sampling_failures": monitor.failures,
        "paging_delta": {key: after[key] - before[key] for key in before},
    }
    result["eligible"] = (
        result["external_cpu_max"] <= 150
        and result["external_model_cpu_max"] <= 10
        and result["thermal_max"] <= 1
        and not compiler and not result["sampling_gap"]
        and not result["sampling_failures"]
        and not any(result["paging_delta"].values())
    )
    return result


def run(args):
    from stark_convert.chat import load_template

    model = args.model.resolve()
    tokenizer = args.tokenizer.resolve()
    binary = args.binary.resolve()
    if not model.exists() or not tokenizer.is_dir() or not binary.is_file():
        raise ValueError("input_path_invalid")
    weights_before = weight_identity(model, args.weights_manifest)

    codec, template = load_template(tokenizer)
    def render_prompt(repetitions: int) -> str:
        return template.render(
            messages=[{"role": "user", "content": " ".join(["Hello."] * repetitions)}],
            tools=None,
            add_generation_prompt=True,
            enable_thinking=True,
            bos_token=codec.get("bos_token") or "",
            eos_token=codec.get("eos_token") or "",
        )

    prompt_text = render_prompt(1)
    config_path = tokenizer / "config.json"
    provenance_path = tokenizer / "native-source.json"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        str(binary), "--model", str(model), "--engine", args.engine,
        "--host", "127.0.0.1", "--port", str(port),
        "--no-mtp", "--no-drafter", "--no-pld", "--no-ds4-mtp",
        "--serve", "--ctx-size", "512", "--max-tokens", str(args.tokens),
        "--metrics", "--log-file", "off", "--no-vision",
        "--llama-mtp-drafts", "0",
        "--prefix-cache-entries", "0", "--prefix-cache-disk", "off",
        "--tokenize-cache-entries", "0", "--log-level", "ERROR",
    ]
    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    report = {
        "schema_version": 1,
        "scope": "mlx_serve_glm_greedy_baseline",
        "engine_requested": args.engine,
        "model_files": model_files(model),
        "weights": weights_before,
        "tokenizer_config_sha256": sha(config_path),
        "native_source_sha256": sha(provenance_path) if provenance_path.is_file() else None,
        "mlx_serve_executable_sha256": sha(binary),
        "harness_sha256": sha(Path(__file__)),
        "context_tokens": 512,
        "sampling": "greedy_no_speculation",
        "requested_output_tokens": args.tokens,
        "requested_replays": args.replays,
        "replays": [],
        "excluded_attempts": [],
        "screened": args.screened,
        "screen_policy": "10s quiet admission before each measured request: external CPU <=150%, one process <=80%, external model CPU <=10%, idle GPU <=5%, thermal <=fair. Each request requires external CPU <=150%, model CPU <=10%, thermal <=fair, no active compiler, paging, monitor failures or gaps. Fair thermal state is allowed; an idle/nominal benchmark is not claimed.",
        "settings_note": "Existing user MLX-Serve settings were left unchanged. Embedded-engine sequence reuse is independent of MLX prefix-cache flags; each response must report zero cached tokens or this uncached benchmark fails.",
    }
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=env, start_new_session=True,
    )
    monitor = None
    stage = "readiness"
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("server_exited_before_ready")
            try:
                http_json(base, "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise TimeoutError("server_readiness_timeout")

        try:
            report["effective_settings"] = safe_settings(http_json(base, "/props", timeout=5))
        except Exception:
            report["effective_settings"] = None
        try:
            from tokenizers import Tokenizer
            source_tokenizer = Tokenizer.from_file(str(tokenizer / "tokenizer.json"))
            ids = source_tokenizer.encode(prompt_text, add_special_tokens=False).ids
        except Exception as exc:
            raise ValueError("source_tokenizer_unavailable") from exc
        if not isinstance(ids, list) or not ids:
            raise ValueError("tokenization_invalid")
        report["input_tokens"] = len(ids)
        report["input_token_sha256"] = compact_sha(ids)
        report["prompt_text_sha256"] = hashlib.sha256(prompt_text.encode()).hexdigest()
        stage = "model_identity"
        model_id, model_info = model_identity(base, args.engine)
        if model_info["loaded"] is not True or model_info["state"] != "ready":
            raise ValueError("model_not_ready")
        report["loaded_model"] = model_info

        if args.screened:
            from benchmark_clean_qwen import Monitor, paging
            monitor = Monitor()
            monitor.__enter__()
        attempts = args.max_attempts if args.screened else args.replays
        for attempt in range(attempts):
            index = len(report["replays"])
            admission = None
            if monitor is not None and index > 0:
                stage = "quiet_admission"
                admission = quiet_admission(monitor)
            phase = f"attempt_{attempt}"
            if monitor is not None:
                monitor.set_phase(phase)
                before_paging = paging()
            stage = "before_metrics"
            before = http_json(base, "/metrics.json")
            started = time.monotonic()
            stage = "completion"
            response = http_json(
                base, "/v1/completions",
                {
                    "model": model_id,
                    "prompt": ids,
                    "max_tokens": args.tokens, "temperature": 0,
                    "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                    "repeat_penalty": 1.0, "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": False,
                    "enable_mtp": False,
                    "enable_drafter": False, "enable_pld": False,
                },
                timeout=600,
            )
            wall = time.monotonic() - started
            stage = "after_metrics"
            after = http_json(base, "/metrics.json")
            stage = "response_validation"
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0].get("text"), str):
                raise ValueError("response_invalid")
            content = choices[0]["text"]
            usage = response.get("usage") or {}
            output_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            if type(output_tokens) is not int or type(prompt_tokens) is not int or output_tokens < 1:
                raise ValueError("usage_invalid")
            if prompt_tokens != len(ids):
                raise ValueError("source_tokenizer_disagreement")
            if after["counters"]["requests_success_total"] - before["counters"]["requests_success_total"] != 1:
                raise ValueError("request_count_mismatch")
            durations = {}
            for name in ("prefill_time_seconds", "decode_time_seconds"):
                old, new = before["histograms"][name], after["histograms"][name]
                if new["count"] - old["count"] != 1:
                    raise ValueError("histogram_count_mismatch")
                durations[name] = float(new["sum"]) - float(old["sum"])
            prefill_s = durations["prefill_time_seconds"]
            decode_s = durations["decode_time_seconds"]
            cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            if not all(math.isfinite(x) and x > 0 for x in (prefill_s, decode_s)) or cached_tokens != 0:
                raise ValueError("timing_or_cache_invalid")
            row = {
                "replay": index + 1,
                "attempt": attempt,
                "warmup": index == 0,
                "admission": admission,
                "wall_seconds": wall,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "finish_reason": choices[0].get("finish_reason"),
                "response_text_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "prefill_seconds": prefill_s,
                "decode_seconds": decode_s,
                "cached_tokens": cached_tokens,
                "metric_count_delta": 1,
                "prefill_tokens_per_second": (prompt_tokens - cached_tokens) / prefill_s,
                "output_tokens_per_second": output_tokens / decode_s,
                "useful_decode_tokens_per_second": max(0, output_tokens - 1) / decode_s,
            }
            if monitor is not None:
                stage = "host_screen"
                row["host_screen"] = screen_attempt(monitor, phase, before_paging, paging())
            if monitor is not None and index > 0 and not row["host_screen"]["eligible"]:
                report["excluded_attempts"].append(row)
                print(json.dumps({"excluded_attempt": attempt}), flush=True)
                continue
            report["replays"].append(row)
            print(json.dumps({"completed_replay": index + 1}), flush=True)
            if len(report["replays"]) == args.replays:
                break
        if len(report["replays"]) != args.replays:
            raise ValueError("insufficient_screened_requests")
        stage = "timing_summary"
        report["repeat_exact"] = len({x["response_text_sha256"] for x in report["replays"]}) == 1
        if not report["repeat_exact"]:
            raise ValueError("non_deterministic_greedy_replay")
        report["median_warm_decode_tokens_per_second"] = statistics.median(
            x["output_tokens_per_second"] for x in report["replays"][1:]
        ) if len(report["replays"]) > 1 else report["replays"][0]["output_tokens_per_second"]
        report["median_warm_useful_decode_tokens_per_second"] = statistics.median(
            x["useful_decode_tokens_per_second"] for x in report["replays"][1:]
        )
        report["timer_note"] = "Phase durations are per-request server histogram deltas, each bracketed count advances once; wall time includes local HTTP transport and client overhead. Both server completion-count and first-token-excluded useful decode numerators are retained."
        stage = "weight_identity_guard"
        if weight_identity(model, args.weights_manifest) != weights_before:
            raise ValueError("weights_changed_during_run")
        report["passed"] = True
    except KeyboardInterrupt:
        report.update(passed=False, error="interrupted", failure_stage=stage)
    except Exception as error:
        report.update(
            passed=False, error="benchmark_validation_failed", failure_stage=stage,
            error_type=type(error).__name__,
            http_status=error.code if isinstance(error, HTTPError) else None,
            server_exit_code=process.poll(),
        )
    finally:
        if monitor is not None:
            monitor.__exit__(None, None, None)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="GGUF file or directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="native GLM directory used to render the Stark chat template")
    parser.add_argument("--binary", type=Path, required=True, help="MLX-Serve executable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, help="prior verified per-shard hash receipt; payloads are not rehashed")
    parser.add_argument("--engine", choices=("auto", "llama", "ds4"), default="auto")
    parser.add_argument("--screened", action="store_true", help="admit and screen each measured request for host interference")
    parser.add_argument("--max-attempts", type=int, default=20, choices=range(5, 41))
    parser.add_argument("--tokens", type=int, default=32, choices=range(8, 129))
    parser.add_argument("--replays", type=int, default=5, choices=range(2, 9))
    args = parser.parse_args()
    if args.output.exists():
        print(json.dumps({"passed": False, "error": "output_exists"}))
        return 2
    try:
        report = run(args)
    except KeyboardInterrupt:
        report = {"schema_version": 1, "scope": "mlx_serve_glm_greedy_baseline", "passed": False, "error": "interrupted"}
    except TimeoutError:
        report = {"schema_version": 1, "scope": "mlx_serve_glm_greedy_baseline", "passed": False, "error": "server_timeout"}
    except (OSError, URLError):
        report = {"schema_version": 1, "scope": "mlx_serve_glm_greedy_baseline", "passed": False, "error": "local_runtime_unavailable"}
    except Exception:
        report = {"schema_version": 1, "scope": "mlx_serve_glm_greedy_baseline", "passed": False, "error": "benchmark_validation_failed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(json.dumps({"passed": report["passed"], "replays": len(report.get("replays", []))}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
