"""Serialize full-checkpoint worker timings; emit only aggregates and hashes."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from stark_convert.chat import ChatError, Worker, load_template


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_identity(model: Path, receipt_path: Path | None):
    if receipt_path is None:
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("passed") is not True:
        raise ValueError("unverified_weight_receipt")
    rows = {}
    for name, expected in receipt["files"].items():
        if Path(name).name != name or not name.endswith(".gguf") or expected.get("matched") is not True:
            raise ValueError("invalid_weight_receipt")
        stat = (model / name).stat()
        actual = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("verified_weight_changed")
        rows[name] = dict(actual, prior_verified_sha256=expected["sha256"])
    if not rows:
        raise ValueError("empty_weight_receipt")
    if {path.name for path in model.glob("*.gguf")} != set(rows):
        raise ValueError("incomplete_weight_receipt")
    return {"receipt_sha256": sha(receipt_path), "files": rows,
            "verification": "Prior content hashes; size, modification time and inode checked before and after execution."}


def checkout_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    sources = sorted((root / "crates/stark-engine/src/engine/hybrid").glob("*"))
    for path in sources:
        if path.suffix in {".rs", ".metal"}:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "--no-color", "HEAD", "--"], cwd=root,
        )).hexdigest(),
        "tracked_diff_format": "git diff --no-ext-diff --no-textconv --binary --no-color HEAD --",
        "hybrid_source_bundle_sha256": digest.hexdigest(),
        "binding": "Harness checkout identity; executable source must be bound separately by its build receipt.",
    }


def hardware_identity():
    def sysctl(key):
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    return {"model": sysctl("hw.model"), "physical_memory_bytes": int(sysctl("hw.memsize")),
            "os_version": platform.mac_ver()[0], "architecture": platform.machine()}


def run(args):
    from tokenizers import Tokenizer

    codec, template = load_template(args.model)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    weights_before = weight_identity(args.model, args.verified_weights)
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
        "harness_sha256": sha(Path(__file__)),
        "worker_sha256": sha(args.worker),
        "configuration_sha256": sha(args.model / "config.json"),
        "source_receipt_sha256": sha(args.model / "native-source.json"),
        "tokenizer_sha256": sha(args.model / "tokenizer.json"),
        "prompt_template_sha256": hashlib.sha256(codec["prompt_template"].encode()).hexdigest(),
        "hardware": hardware_identity(),
        "harness_checkout": checkout_identity(),
        "weights": weights_before,
        "context_tokens": 512,
        "sampling": "greedy_no_speculation",
        "native_decode_rate_policy": "(sampled_tokens - 1) / decode_seconds; the first sampled token comes from prefill.",
        "requested_output_tokens": args.tokens,
        "warmup_replays": args.warmups,
        "screened": args.screened,
        "diagnostic_profile_output_requested": "STARK_QWEN38_PROFILE_OUTPUT" in os.environ,
        "native_profile_output_requested": "STARK_NATIVE_PROFILE_OUTPUT" in os.environ,
        "configuration_flags": {
            key: os.environ.get(key)
            for key in (
                "STARK_GLM53_BOUNDED_GRAPH", "STARK_GLM53_Q4K_BLOCK",
                "STARK_GLM53_KQUANT_FACTORED", "STARK_GLM53_CONTROL_SPLIT",
                "STARK_GLM53_CONTROL_SIMD", "STARK_HYBRID_PACKED_REFERENCE",
                "STARK_GLM53_KDA_TRANSPOSED", "STARK_GLM53_ENCODER_BANKS",
                "STARK_GLM53_PACKED_FAST_MATH", "STARK_GLM53_PARALLEL_PROJECTIONS",
                "STARK_GLM53_BANK_GROUP_LAYERS", "STARK_GLM53_Q8_SHAPES",
                "STARK_GLM53_CONCURRENT_ENCODER",
                "STARK_GLM53_KDA_FUSED_STATE", "STARK_GLM53_SERIAL_ROUTER",
                "STARK_GLM53_CONTROL_CHUNKS",
                "STARK_GLM53_PARALLEL_RMS", "STARK_GLM53_RMS_PREFETCH",
                "STARK_GLM53_IQ4_VECTOR", "STARK_GLM53_RESIDENT_BANKS",
                "STARK_GLM53_BATCH_FINITE",
                "STARK_GLM53_IQ_MULTIROW",
                "STARK_GLM53_Q6_GROUPED16",
                "STARK_GLM53_Q5_GROUPED32", "STARK_GLM53_CONTROL_RMS_PROJECT",
                "STARK_GLM53_IQ_GPU_BOUND", "STARK_GLM53_GREEDY_OVERLAP",
                "STARK_GLM53_FINITE_SELECTIVE",
                "STARK_GLM53_IQ_TYPED16", "STARK_GLM53_WIDE_ENCODER",
                "STARK_GLM53_IQ_SIGNED_FACTOR", "STARK_GLM53_POLL_WAIT",
                "STARK_GLM53_SERIAL_FINITE_ORDER", "STARK_GLM53_UNRETAINED_GRAPH",
                "STARK_GLM53_SCRATCH_HEAP", "STARK_GLM53_CONTROL_FUSED_PARALLEL",
                "STARK_GLM53_Q8_GROUPED8",
                "STARK_GLM53_Q4_GROUPED32", "STARK_GLM53_GPU_GREEDY",
                "STARK_GLM53_PREFILL_OVERLAP", "STARK_GLM53_KDA_PARALLEL_NORM",
                "STARK_GLM53_PREFILL_QUEUED",
                "STARK_GLM53_IQ_FUSED_GATE_UP",
                "STARK_GLM53_IQ_FUSED_GATE_UP_R2",
                "STARK_GLM53_DECODE_GRAPH", "STARK_HYBRID_ORDERED_RMS",
                "STARK_HYBRID_BYTE_LOADS", "STARK_HYBRID_VECTOR8",
                "STARK_HYBRID_SCALAR_UINT_LOADS", "STARK_HYBRID_FACTORED_AFFINE",
                "STARK_HYBRID_SEPARATE_FINITE", "STARK_QWEN38_SERIAL_LEVELS",
            )
        },
        "excluded_attempts": [],
        "replays": [],
    }
    worker = Worker(args.worker.resolve(), args.model.resolve(), 3600)
    monitor = None
    stage = "readiness"
    try:
        if worker.read().get("ready") is not True:
            raise ValueError("worker_not_ready")
        stage = "tokenization"
        tokenized = worker.request(
            {"operation": "tokenize", "text": prompt, "add_special_tokens": False}
        )
        ids = tokenized["token_ids"]
        report["input_tokens"] = len(ids)
        report["input_sha256"] = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        if args.screened:
            from benchmark_clean_qwen import Monitor, paging
            from benchmark_glm_mlx_serve import quiet_admission, screen_attempt
            monitor = Monitor()
            monitor.__enter__()
        for attempt in range(args.max_attempts if args.screened else args.replays):
            replay = len(report["replays"])
            admission = None
            if monitor is not None and replay >= args.warmups:
                stage = "quiet_admission"
                admission = quiet_admission(monitor)
            phase = f"attempt_{attempt}"
            if monitor is not None:
                monitor.set_phase(phase)
                before_paging = paging()
            started = time.monotonic()
            stage = "generation"
            result = worker.request(
                {
                    "operation": "generate",
                    "input_token_ids": ids,
                    "context_tokens": 512,
                    "max_new_tokens": args.tokens,
                    "sampling": {"mode": "greedy"},
                    "request_id": attempt + 1,
                }
            )
            if "error" in result or result.get("sampled_tokens", 0) < 2:
                raise ValueError("generation_failed_or_too_short")
            item = {
                "replay": replay,
                "attempt": attempt,
                "warmup": replay < args.warmups,
                "admission": admission,
                "wall_seconds": time.monotonic() - started,
                "sampled_tokens": result["sampled_tokens"],
                "finish_reason": result["finish_reason"],
                "output_sha256": hashlib.sha256(
                    json.dumps(
                        result["generated_token_ids"], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "output_text_sha256": hashlib.sha256(
                    tokenizer.decode(result["generated_token_ids"], skip_special_tokens=False).encode()
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
            item["useful_decode_tokens_per_second"] = (result["sampled_tokens"] - 1) / result["decode_seconds"]
            item["all_sampled_tokens_per_decode_second"] = result["sampled_tokens"] / result["decode_seconds"]
            if monitor is not None:
                stage = "host_screen"
                item["host_screen"] = screen_attempt(monitor, phase, before_paging, paging())
                if not item["warmup"] and not item["host_screen"]["eligible"]:
                    report["excluded_attempts"].append(item)
                    print(json.dumps({"excluded_attempt": attempt}), flush=True)
                    continue
            report["replays"].append(item)
            print(json.dumps({"completed_replay": replay,
                              "useful_decode_tokens_per_second": item["useful_decode_tokens_per_second"]}), flush=True)
            if len(report["replays"]) == args.replays:
                break
        stage = "aggregation"
        if len(report["replays"]) != args.replays:
            raise ValueError("insufficient_screened_requests")
        stable = report["replays"][args.warmups:] or report["replays"]
        report["median_warm_decode_tokens_per_second"] = statistics.median(
            item["output_tokens_per_second"] for item in stable
        )
        report["median_warm_prefill_tokens_per_second"] = statistics.median(
            item["input_tokens_per_second"] for item in stable
        )
        report["median_warm_all_sampled_tokens_per_decode_second"] = statistics.median(
            item["all_sampled_tokens_per_decode_second"] for item in stable
        )
        if weight_identity(args.model, args.verified_weights) != weights_before:
            raise ValueError("weight_identity_changed_during_execution")
        report["repeat_exact"] = (
            len({item["output_sha256"] for item in report["replays"]}) == 1
        )
        if not report["repeat_exact"]:
            raise ValueError("nondeterministic_replay")
        report["passed"] = True
    except (OSError, ValueError, KeyError, TypeError, ChatError, subprocess.SubprocessError) as error:
        report.update(passed=False, error="native_benchmark_failed", failed_stage=stage,
                      error_type=type(error).__name__)
    finally:
        worker.close()
        if monitor is not None:
            monitor.__exit__(None, None, None)
            tail = monitor.rows[-50:]
            if tail:
                report["last_host_observations"] = {
                    "samples": len(tail), "sampling_failures": monitor.failures,
                    "external_cpu_max": max(row["external_cpu_percent"] for row in tail),
                    "external_model_cpu_max": max(row["max_model_cpu_percent"] for row in tail),
                    "gpu_utilization_min": min(row["gpu_device_utilization_percent"] for row in tail),
                    "gpu_utilization_median": statistics.median(row["gpu_device_utilization_percent"] for row in tail),
                    "gpu_utilization_max": max(row["gpu_device_utilization_percent"] for row in tail),
                    "thermal_max": max(row["thermal_state"] for row in tail),
                }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key not in {"replays", "excluded_attempts"}}, sort_keys=True))
    except OSError:
        raise ValueError("receipt_write_failed") from None
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32, choices=range(8, 129))
    parser.add_argument("--replays", type=int, default=3, choices=range(2, 9))
    parser.add_argument("--warmups", type=int, default=1, choices=range(0, 8))
    parser.add_argument("--screened", action="store_true")
    parser.add_argument("--verified-weights", type=Path)
    parser.add_argument("--max-attempts", type=int, default=32, choices=range(2, 65))
    args = parser.parse_args()
    if args.warmups >= args.replays:
        parser.error("warmups must be smaller than replays")
    if args.screened and args.max_attempts < args.replays:
        parser.error("max-attempts must be at least replays")
    if args.output.exists():
        parser.error("output already exists")
    try:
        passed = run(args)
    except (OSError, ValueError, KeyError, TypeError, ChatError, subprocess.SubprocessError):
        print(json.dumps({"passed": False, "stage": "native_benchmark_failed"}))
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
