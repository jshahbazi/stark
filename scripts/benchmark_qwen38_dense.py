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


def require_mtp_depth(ready, drafts):
    """Require readiness to advertise the selected dense-MTP depth."""
    if drafts == 0:
        return
    mtp = ready.get("mtp", {})
    if not isinstance(mtp, dict):
        raise TypeError("mtp_depth_unavailable")
    maximum = mtp.get("max_drafts")
    if mtp.get("supported") is not True or type(maximum) is not int or maximum < drafts:
        raise ValueError("mtp_depth_unavailable")


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
        "rounds_by_accepted_drafts",
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
                                     and args.warmups >= 1 and len(measured) == 6 and args.drafts > 0
                                     and all(r.get("mtp", {}).get("enabled") is True
                                             and r["mtp"].get("drafts") == args.drafts for r in measured))
    report["target_met"] = (report["target_protocol_met"] and report["all_outputs_reached_budget"]
                            and report["throughput_threshold_met"] and report.get("host_screen_passed", False))


def collect_measurement_slots(prompt_count, repeats, max_attempts, retry_ineligible, measure):
    """Fill per-prompt quotas, retrying only host-ineligible attempts."""
    accepted, excluded = [], []
    attempts = 0
    expected = prompt_count * repeats
    cap = max_attempts if retry_ineligible else expected
    stop_reason = None
    for prompt_index in range(prompt_count):
        for repeat_index in range(repeats):
            filled = False
            while attempts < cap:
                attempt = attempts + 1
                try:
                    item = measure(prompt_index, repeat_index, attempt)
                except TimeoutError:
                    stop_reason = "quiet_admission_timeout"
                    break
                attempts += 1
                item.update(
                    warmup=False,
                    prompt_index=prompt_index,
                    repeat_index=repeat_index,
                    measurement_attempt=attempt,
                )
                eligible = item.get("host_screen", {}).get("eligible") is True
                if retry_ineligible and not eligible:
                    excluded.append({
                        "prompt_index": prompt_index,
                        "repeat_index": repeat_index,
                        "measurement_attempt": attempt,
                        "reason": "host_screen_ineligible",
                        "metrics": item,
                    })
                    continue
                accepted.append(item)
                filled = True
                break
            if stop_reason is not None:
                break
            if not filled:
                stop_reason = "measurement_attempt_cap_reached"
                break
        if stop_reason is not None:
            break
    return accepted, excluded, attempts, stop_reason


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
        "reference_limit": ("Original public run did not pin the artifact revision; "
                            "corpus identity must be compared separately."),
        "worker_sha256": digest(args.worker), "configuration_sha256": digest(args.model / "config.json"),
        "conversion_receipt_sha256": digest(args.model / "conversion.json"),
        "harness_sha256": digest(Path(__file__)), "corpus_sha256": digest(args.prompts),
        "hardware": {key: subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                     for key in ("machdep.cpu.brand_string", "hw.memsize")},
        "sampling": sampling, "thinking": False, "mtp_drafts": args.drafts,
        "context_tokens": args.context, "requested_output_tokens": args.tokens,
        "warmups": args.warmups, "repeats_per_prompt": args.repeats, "runs": [],
        "host_screened": args.screened, "retry_ineligible": args.retry_ineligible,
        "max_measurement_attempts": args.max_measurement_attempts,
        "excluded_attempts": [],
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
        require_mtp_depth(ready, args.drafts)
        def measure(index, repeat_index, attempt, *, warmup):
            phase_index = attempt - 1 + args.warmups if not warmup else attempt - 1
            phase = f"request_{phase_index}"
            rendered = template.render(
                messages=[{"role": "user", "content": prompts[index]}], tools=None,
                add_generation_prompt=True, enable_thinking=False,
                bos_token=codec.get("bos_token") or "", eos_token=codec.get("eos_token") or "",
            )
            admission = quiet_admission(monitor) if monitor is not None and not warmup else None
            started = time.monotonic()
            if monitor is not None:
                before_paging = paging()
                monitor.set_phase(phase)
            reply = worker.request({
                "operation": "generate", "rendered_prompt": rendered,
                "context_tokens": args.context, "max_new_tokens": args.tokens,
                "sampling": sampling, "seed": 42, "mtp_drafts": args.drafts,
            })
            item = summarize(reply)
            item.update(warmup=warmup, prompt_index=index, wall_seconds=time.monotonic() - started)
            if monitor is not None:
                item["host_screen"] = screen_attempt(monitor, phase, before_paging, paging())
                item["quiet_admission"] = admission
            return item

        for index in range(args.warmups):
            try:
                item = measure(0, 0, index + 1, warmup=True)
            except TimeoutError:
                report["admission_failed"] = "quiet_admission_timeout"
                break
            report["runs"].append(item)
            print(json.dumps({"completed_warmup": index + 1,
                              "tok_s": item["useful_decode_tok_s"]}), flush=True)

        if not report.get("admission_failed"):
            accepted, excluded, attempts, stop_reason = collect_measurement_slots(
                len(prompts), args.repeats, args.max_measurement_attempts,
                args.retry_ineligible,
                lambda index, repeat_index, attempt: measure(index, repeat_index, attempt, warmup=False),
            )
            report["runs"].extend(accepted)
            report["excluded_attempts"].extend(excluded)
            report["measurement_attempts"] = attempts
            if stop_reason == "quiet_admission_timeout":
                report["admission_failed"] = stop_reason
            elif stop_reason == "measurement_attempt_cap_reached":
                report["measurement_attempt_cap_reached"] = True
            for measurement_number, item in enumerate(accepted, start=1):
                print(json.dumps({"completed_measurement": measurement_number,
                                  "prompt_index": item["prompt_index"],
                                  "tok_s": item["useful_decode_tok_s"]}), flush=True)
            for item in excluded:
                print(json.dumps({"excluded_attempt": item["measurement_attempt"],
                                  "reason": item["reason"]}), flush=True)
    measured = [item for item in report["runs"] if not item["warmup"]]
    expected = args.repeats * len(prompts)
    report["host_screen_passed"] = (
        args.screened
        and not report.get("admission_failed")
        and len(measured) == expected
        and all(item.get("host_screen", {}).get("eligible") is True for item in measured)
    )
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
    p.add_argument("--drafts", type=int, choices=tuple(range(8)), default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--screened", action="store_true",
                   help="Require observed quiet admission and screen CPU/GPU, thermal and paging state")
    p.add_argument("--retry-ineligible", action="store_true",
                   help="Retry only measured requests rejected by the host screen")
    p.add_argument("--max-measurement-attempts", type=int, default=None,
                   help="Total measured request attempts, 1..24; defaults to 12 with --retry-ineligible")
    args = p.parse_args()
    if not 2 <= args.tokens <= 4096 or args.context <= args.tokens or args.warmups < 0 or args.repeats < 1:
        p.error("invalid benchmark limits")
    if args.retry_ineligible and not args.screened:
        p.error("--retry-ineligible requires --screened")
    if args.max_measurement_attempts is not None and not args.retry_ineligible:
        p.error("--max-measurement-attempts requires --retry-ineligible")
    if args.retry_ineligible:
        if args.max_measurement_attempts is None:
            args.max_measurement_attempts = 12
        if not 1 <= args.max_measurement_attempts <= 24:
            p.error("--max-measurement-attempts must be between 1 and 24")
    try:
        run(args)
    except Exception:  # noqa: BLE001 - private worker failures must not become exception dumps.
        print(json.dumps({"status": "failed", "error": "dense_benchmark_failed"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
