"""Private serialized dense-Qwen benchmark: publish aggregates and hashes only.

The 70.8 tok/s reference used six 768-token code responses, temperature 0.6,
thinking disabled and Lightning MTP. A different prompt corpus or artifact is
an independent target test, not a reproduction of that reference measurement.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from stark_convert.chat import Worker, load_template


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(reply):
    """Allowlist aggregate fields; never persist worker text, vectors or errors."""
    count = reply.get("sampled_tokens", 0)
    seconds = reply.get("decode_seconds", 0)
    ids = reply.get("generated_token_ids")
    if "error" in reply or not isinstance(ids, list) or count < 2 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid_benchmark_reply")
    item = {name: reply[name] for name in (
        "sampled_tokens", "finish_reason", "prefill_seconds", "decode_seconds",
        "generation_seconds", "output_tokens_per_second",
    )}
    item["useful_decode_tok_s"] = (count - 1) / seconds
    if "input_tokens" in reply:
        item["input_tokens"] = reply["input_tokens"]
    item["output_sha256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    mtp = reply.get("mtp", {})
    item["mtp"] = {key: mtp[key] for key in (
        "enabled", "drafts", "sampling", "verification", "rounds", "proposed", "accepted", "verified_rows",
        "draft_seconds", "verify_seconds", "rollback_seconds",
        "target_encode_seconds", "target_gpu_seconds", "target_readout_seconds",
    ) if key in mtp}
    return item


def finalize(report, args):
    measured = [r for r in report["runs"] if not r["warmup"]]
    if not measured:
        report.update(median_tok_s=None, all_outputs_reached_budget=False,
                      throughput_threshold_met=False, target_protocol_met=False, target_met=False)
        return
    report["median_tok_s"] = statistics.median(r["useful_decode_tok_s"] for r in measured)
    report["all_outputs_reached_budget"] = all(r["sampled_tokens"] == args.tokens for r in measured)
    report["throughput_threshold_met"] = report["median_tok_s"] >= 70.8
    report["target_protocol_met"] = (args.tokens == 768 and not args.greedy and args.repeats == 2
                                     and args.warmups >= 1 and len(measured) == 6 and args.drafts == 3
                                     and all(r.get("mtp", {}).get("enabled") is True
                                             and r["mtp"].get("drafts") == 3 for r in measured))
    report["target_met"] = (report["target_protocol_met"] and report["all_outputs_reached_budget"]
                            and report["throughput_threshold_met"] and report.get("host_screen_passed", False))


def run(args):
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or len(prompts) != 3 or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("three_code_prompts_required")
    codec, template = load_template(args.model)
    sampling = {"mode": "greedy"} if args.greedy else {
        "mode": "stochastic", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    }
    report = {
        "schema": "stark-dense-qwen-code-benchmark-v1", "target_tok_s": 70.8,
        "reference_reproduction": False,
        "reference_limit": "Original public run did not pin the artifact revision; corpus identity must be compared separately.",
        "worker_sha256": digest(args.worker), "configuration_sha256": digest(args.model / "config.json"),
        "conversion_receipt_sha256": digest(args.model / "conversion.json"),
        "harness_sha256": digest(Path(__file__)), "corpus_sha256": digest(args.prompts),
        "hardware": {key: subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                     for key in ("machdep.cpu.brand_string", "hw.memsize")},
        "sampling": sampling, "thinking": False, "mtp_drafts": args.drafts,
        "context_tokens": args.context, "requested_output_tokens": args.tokens,
        "warmups": args.warmups, "repeats_per_prompt": args.repeats, "runs": [],
        "host_screened": args.screened,
    }
    monitor_context = contextlib.nullcontext(None)
    if args.screened:
        from benchmark_clean_qwen import Monitor, paging
        from benchmark_glm_mlx_serve import quiet_admission, screen_attempt

        monitor_context = Monitor()
        report["host_screen_policy"] = {
            "monitor_sha256": digest(Path(__file__).with_name("benchmark_clean_qwen.py")),
            "screen_sha256": digest(Path(__file__).with_name("benchmark_glm_mlx_serve.py")),
            "quiet_seconds_before_measurement": 10,
        }
    worker = Worker(args.worker.resolve(), args.model.resolve(), 3600)
    with monitor_context as monitor, contextlib.closing(worker):
        ready = worker.read()
        if ready.get("ready") is not True:
            raise ValueError("worker_not_ready")
        if args.drafts and not ready.get("mtp", {}).get("supported"):
            raise ValueError("mtp_not_loaded")
        for i in range(args.warmups + args.repeats * len(prompts)):
            warm = i < args.warmups
            try:
                admission = quiet_admission(monitor) if monitor is not None and not warm else None
            except TimeoutError:
                report["admission_failed"] = "quiet_admission_timeout"
                break
            index = 0 if warm else (i - args.warmups) // args.repeats
            rendered = template.render(
                messages=[{"role": "user", "content": prompts[index]}], tools=None,
                add_generation_prompt=True, enable_thinking=False,
                bos_token=codec.get("bos_token") or "", eos_token=codec.get("eos_token") or "",
            )
            started = time.monotonic()
            if monitor is not None:
                before_paging = paging()
                monitor.set_phase(f"request_{i}")
            reply = worker.request({
                "operation": "generate", "rendered_prompt": rendered,
                "context_tokens": args.context, "max_new_tokens": args.tokens,
                "sampling": sampling, "seed": 42, "mtp_drafts": args.drafts,
            })
            item = summarize(reply)
            item.update(warmup=warm, prompt_index=index, wall_seconds=time.monotonic() - started)
            if monitor is not None:
                item["host_screen"] = screen_attempt(monitor, f"request_{i}", before_paging, paging())
                item["quiet_admission"] = admission
            report["runs"].append(item)
            print(json.dumps({"completed": i + 1, "warmup": warm, "tok_s": item["useful_decode_tok_s"]}), flush=True)
    report["host_screen_passed"] = (args.screened and not report.get("admission_failed") and all(
        item["host_screen"]["eligible"] for item in report["runs"] if not item["warmup"]
    ))
    finalize(report, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"median_tok_s": report["median_tok_s"], "target_met": report["target_met"]}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True, help="Private JSON array of exactly three code prompts")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=768)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--drafts", type=int, choices=(0, 1, 2, 3), default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--screened", action="store_true", help="Require observed quiet admission and screen CPU/GPU, thermal and paging state")
    args = p.parse_args()
    if not 2 <= args.tokens <= 4096 or args.context <= args.tokens or args.warmups < 0 or args.repeats < 1:
        p.error("invalid benchmark limits")
    try:
        run(args)
    except Exception:  # noqa: BLE001 - private worker failures must not become exception dumps.
        print(json.dumps({"status": "failed", "error": "dense_benchmark_failed"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
