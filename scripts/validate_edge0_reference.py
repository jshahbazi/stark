"""Bounded local Edge0 reference-worker smoke; publish aggregates/hashes only.

Run with the locked converter environment. Requests and replies remain inside
private pipes/in memory. A sampled 48 GiB RSS watchdog reaps only this helper's
worker. This is functional evidence, not an external full-model parity oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import select
import struct
import subprocess
import time
from pathlib import Path

import psutil
from stark_convert.template_runtime import IsolatedTemplate

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e"
RSS_LIMIT = 48 * 1024**3
TIMEOUT = 1200


class Failure(Exception):
    """Only a fixed, content-free code may cross this boundary."""


def require(value, code):
    if not value:
        raise Failure(code)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class Worker:
    def __init__(self, binary, model, *, metal=False, resident=False):
        require(not resident or metal, "resident_requires_metal")
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("STARK_", "UZU_"))
        }
        self.process = subprocess.Popen(
            [str(binary), "--model-path", str(model), "--prefix-cache"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                **environment,
                "STARK_EDGE0_REFERENCE": "1",
                "STARK_REQUEST_BUDGET_MIB": "16",
                **({"STARK_EDGE0_METAL": "1"} if metal else {}),
                **({"STARK_EDGE0_RESIDENT": "1"} if resident else {}),
            },
        )
        self.owner = psutil.Process(self.process.pid)
        self.buffer = bytearray()
        self.peak_rss = 0

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()

    def read(self, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + TIMEOUT
        self.check_resources(deadline)
        while b"\n" not in self.buffer:
            self.check_resources(deadline)
            if not select.select([self.process.stdout], [], [], 0.1)[0]:
                continue
            block = os.read(self.process.stdout.fileno(), 65536)
            require(bool(block), "worker_exited")
            self.buffer.extend(block)
            require(len(self.buffer) <= 4 * 1024 * 1024, "reply_limit")
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        return json.loads(line)

    def check_resources(self, deadline):
        try:
            self.peak_rss = max(self.peak_rss, self.owner.memory_info().rss)
        except psutil.NoSuchProcess:
            raise Failure("worker_exited") from None
        require(self.peak_rss <= RSS_LIMIT, "rss_limit")
        require(time.monotonic() < deadline, "worker_timeout")

    def request(self, payload):
        deadline = time.monotonic() + TIMEOUT
        self.process.stdin.write(json.dumps(payload, allow_nan=False).encode() + b"\n")
        self.process.stdin.flush()
        while True:
            reply = self.read(deadline)
            if reply.get("event") in {"progress", "delta"}:
                continue
            return reply


def result_summary(reply):
    require("error" not in reply, "generation_failed")
    ids = reply.get("generated_token_ids")
    text = reply.get("text")
    count = reply.get("sampled_tokens")
    require(
        isinstance(ids, list) and all(type(v) is int and 0 <= v < 248320 for v in ids),
        "invalid_ids",
    )
    require(
        isinstance(text, str)
        and bool(text.strip())
        and bool(ids)
        and type(count) is int
        and 1 <= count <= 32,
        "invalid_result",
    )
    reason = reply.get("finish_reason")
    require(reason in {"eos", "length"}, "unexpected_stop")
    require(reason != "length" or count == 32, "incorrect_length")
    require(count == len(ids) + int(reason == "eos"), "incorrect_sample_count")
    require(
        reason != "eos"
        or reply.get("stop_token_id") in {248044, 248046, 248063, 248064, 248065},
        "incorrect_eos",
    )
    require(reply.get("reused_input_tokens", 0) == 0, "unexpected_reuse")
    return {
        "token_sha256": hashlib.sha256(
            b"".join(struct.pack("<I", value) for value in ids)
        ).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_tokens": len(ids),
        "sampled_tokens": count,
        "finish_reason": reason,
    }


def run(args):
    metal = getattr(args, "metal", False)
    resident = getattr(args, "resident", False)
    require(not resident or metal, "resident_requires_metal")
    model, binary, output = (
        args.model.resolve(),
        args.worker.resolve(),
        args.output.resolve(),
    )
    require(not output.exists(), "output_exists")
    require(
        subprocess.run(
            ["git", "check-ignore", "--quiet", str(output)], cwd=ROOT, check=False
        ).returncode
        == 0,
        "output_not_ignored",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    require(digest(model / "native-source.json") == RECEIPT, "source_receipt")
    template_path = model / "tokenizer/chat_template.jinja"
    tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
    template = IsolatedTemplate(template_path.read_text())
    prompt = template.render(
        messages=[{"role": "user", "content": "Reply with one short greeting."}],
        tools=None,
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token=tokenizer_config.get("bos_token") or "",
        eos_token=tokenizer_config.get("eos_token") or "",
    )
    require(
        isinstance(prompt, str) and 0 < len(prompt.encode()) < 8192, "template_result"
    )
    base = {
        "operation": "generate",
        "rendered_prompt": prompt,
        "context_tokens": 512,
        "max_new_tokens": 32,
        "mtp": False,
        "seed": 42,
    }
    requests = {
        "greedy": {**base, "sampling": {"mode": "greedy"}},
        "stochastic": {
            **base,
            "sampling": {
                "mode": "stochastic",
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
            },
        },
    }
    registration = {
        "schema": 1,
        "attempt": "20261112-EDGE0-06" if metal else "20261112-EDGE0-05",
        "scope": "local_noncommercial_evaluation",
        "worker_sha256": digest(binary),
        "qualifier_sha256": digest(Path(__file__)),
        "source_receipt_sha256": RECEIPT,
        "artifact_rehashed": False,
        "template_sha256": digest(template_path),
        "tokenizer_sha256": digest(model / "tokenizer.json"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "request_sha256": {
            name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            for name, value in requests.items()
        },
        "context": 512,
        "maximum": 32,
        "seed": 42,
        "rss_limit_bytes": RSS_LIMIT,
        "watchdog": "sampled-owned-worker-rss-100ms",
        "timeout_seconds_per_reply": TIMEOUT,
        "worker_selectors": {
            "STARK_EDGE0_REFERENCE": "1",
            "STARK_REQUEST_BUDGET_MIB": "16",
            **({"STARK_EDGE0_METAL": "1"} if metal else {}),
            **({"STARK_EDGE0_RESIDENT": "1"} if resident else {}),
        },
        "inherited_stark_uzu_selectors_removed": True,
        "platform": platform.platform(),
        "available_memory_before_bytes": psutil.virtual_memory().available,
        "cases": [
            "greedy",
            "stochastic",
            "reset_greedy",
            "invalid_token_rejection",
            "failure_recovery",
            "new_process_greedy",
            "new_process_stochastic",
        ],
    }
    write_new(output.with_suffix(".registration.json"), registration)
    report = {
        "registration_sha256": digest(output.with_suffix(".registration.json")),
        "status": "failed",
        "stages": [],
        "peak_rss_bytes": 0,
    }
    worker = None
    try:
        require(
            registration["available_memory_before_bytes"] >= 16 * 1024**3,
            "available_memory",
        )
        expected = {}
        for process_index in range(2):
            worker = Worker(binary, model, metal=metal, resident=resident)
            ready = worker.read()
            require(ready.get("ready") is True, "load_failed")
            require(
                ready.get("vocab_size") == 248320
                and ready.get("max_context_tokens") == 512,
                "readiness_geometry",
            )
            require(
                ready.get("prefix_cache", {}).get("supported") is False,
                "prefix_capability",
            )
            require(ready.get("mtp", {}).get("supported") is False, "mtp_capability")
            report["stages"].append({"stage": "ready", "process": process_index})
            for name, request in requests.items():
                summary = result_summary(worker.request(request))
                if process_index == 0:
                    expected[name] = summary
                else:
                    require(summary == expected[name], "new_process_replay")
                report["stages"].append(
                    {"stage": name, "process": process_index, **summary}
                )
            if process_index == 0:
                # Fresh-per-request execution itself resets the whole backend.
                require(
                    result_summary(worker.request(requests["greedy"]))
                    == expected["greedy"],
                    "reset_replay",
                )
                invalid = {
                    **requests["greedy"],
                    "rendered_prompt": None,
                    "input_token_ids": [0xFFFFFFFF],
                }
                require(
                    worker.request(invalid).get("error")
                    == "invalid_generation_request",
                    "invalid_token_rejection",
                )
                require(
                    result_summary(worker.request(requests["greedy"]))
                    == expected["greedy"],
                    "failure_recovery",
                )
                report["stages"].append(
                    {
                        "stage": "fresh_reset_and_request_failure_recovery",
                        "passed": True,
                    }
                )
            report["peak_rss_bytes"] = max(report["peak_rss_bytes"], worker.peak_rss)
            worker.close()
            worker = None
        require(digest(binary) == registration["worker_sha256"], "worker_changed")
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- Suppress all text-bearing exception details.
        report["failure_code"] = (
            str(error) if isinstance(error, Failure) else "qualifier_failure"
        )
    finally:
        if worker is not None:
            report["peak_rss_bytes"] = max(report["peak_rss_bytes"], worker.peak_rss)
            worker.close()
        write_new(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failure_code": report.get("failure_code"),
                "receipt_sha256": digest(output),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metal", action="store_true", help="Select the explicit Phase 4 candidate"
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="Request its separately gated residency policy",
    )
    try:
        return run(parser.parse_args())
    except Exception as error:  # noqa: BLE001 -- Do not expose requests or exception dumps.
        print(
            json.dumps(
                {
                    "status": "setup_failed",
                    "failure_code": str(error)
                    if isinstance(error, Failure)
                    else "setup_failure",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
