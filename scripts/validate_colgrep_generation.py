"""Source-bound ColGrep generation validation; persist aggregates and hashes only.

Run with the locked tools environment. Checkpoints and IPC stay outside Git.
The tool observations below are controlled fixtures, not a live ColGrep index.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import json
import re
import selectors
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REVISION = "f798a61129c6decb5166864e0a3c98373e62c764"
UPSTREAM = "c6c091d7ee14c81206bb28af90d8ff506fd0235e"
ROOT = Path(__file__).resolve().parents[1]
EOS = {1, 130073}
ASSET_HASHES = {
    "system_prompt.txt": "ea2097d3674ae70d0e0de9447f6d2118af8e091c65b7cbe5ba1c2787223e2897",
    "tools.json": "902a4bba8e19c34b00df8626c02c0ff685a578db8de7e54efc934f4cd72b65df",
    "chat_template.jinja": "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c",
}
FINAL_REMINDER = (
    "This is your final turn. Call finish now with your best "
    "file:START-END locations based on the evidence already gathered. Do not search further."
)
FIXTURE = (
    "import time\n\nMAX_RETRIES = 3\nBASE_DELAY_SECONDS = 0.5\n\n"
    "def retry_delay(attempt):\n    return BASE_DELAY_SECONDS * (2 ** attempt)\n\n"
    "def request_with_retry(send):\n    for attempt in range(MAX_RETRIES):\n"
    "        try:\n            return send()\n        except TimeoutError:\n"
    "            if attempt + 1 == MAX_RETRIES:\n                raise\n"
    "            time.sleep(retry_delay(attempt))\n"
)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def digest_ids(ids):
    return digest_bytes(b"".join(value.to_bytes(4, "little") for value in ids))


class Worker:
    def __init__(self, executable, model):
        self.process = subprocess.Popen(
            [str(executable), "--model-path", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.read()
            if self.ready.get("ready") is not True:
                raise RuntimeError("worker_not_ready")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self):
        if not self.selector.select(300):
            raise RuntimeError("worker_timeout")
        data = self.process.stdout.readline(1_048_577)
        if not data or len(data) > 1_048_576 or not data.endswith(b"\n"):
            raise RuntimeError("worker_protocol_invalid")
        return json.loads(data)

    def request(self, value):
        self.process.stdin.write(json.dumps(value, ensure_ascii=False).encode() + b"\n")
        self.process.stdin.flush()
        return self.read()

    def generate(self, prompt, max_tokens=256, **options):
        return self.request(
            {
                "operation": "generate",
                "rendered_prompt": prompt,
                "context_tokens": 4096,
                "max_new_tokens": max_tokens,
                **options,
            }
        )

    def close(self):
        try:
            if self.process.poll() is None:
                with contextlib.suppress(BrokenPipeError):
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            if self.process.stderr.read():
                raise RuntimeError("worker_diagnostics_leaked")
            if self.process.returncode != 0:
                raise RuntimeError("worker_exit_failed")
        finally:
            self.selector.close()
            self.process.stdout.close()
            self.process.stderr.close()
            with contextlib.suppress(BrokenPipeError):
                self.process.stdin.close()


def calls(text):
    """Validate the controlled XML subset; this is not the full upstream parser."""
    result = []
    text = text.rsplit("</think>", 1)[-1]
    for match in re.finditer(r'<function name=["\']([^"\']+)["\']>(.*?)</function>', text, re.S):
        try:
            node = ET.fromstring(match[0])
        except ET.ParseError:
            return []
        if node.attrib != {"name": match[1]} or node.text and node.text.strip():
            return []
        params = {}
        for param in node:
            if param.tag != "param" or set(param.attrib) != {"name"} or len(param):
                return []
            if param.attrib["name"] in params or param.tail and param.tail.strip():
                return []
            params[param.attrib["name"]] = param.text or ""
        name = match[1]
        if name not in {"colgrep", "terminal", "finish"}:
            return []
        allowed = {
            "colgrep": {"query", "k", "path", "include", "pattern", "command"},
            "terminal": {"command"},
            "finish": {"locations"},
        }
        if not set(params).issubset(allowed[name]):
            return []
        if name == "terminal" and not params.get("command", "").strip():
            return []
        if name == "colgrep" and not any(params.get(key, "").strip() for key in ("query", "pattern", "command")):
            return []
        if name == "finish" and not params.get("locations", "").strip():
            return []
        if name == "finish":
            try:
                locations = ast.literal_eval(params["locations"])
            except (ValueError, SyntaxError):
                return []
            if (
                not isinstance(locations, list)
                or not locations
                or not all(isinstance(item, str) for item in locations)
            ):
                return []
        result.append((name, params))
    return result


def assistant(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def observation(command, output, turn):
    return {
        "role": "tool",
        "content": f"$ {command}\n{output}\n\n[This is the output of turn {turn} (10 turn limit).]",
    }


def fixtures(tokenizer, upstream):
    assets = upstream / "agent" / "assets"
    tools = json.loads((assets / "tools.json").read_text())
    messages = [
        {"role": "system", "content": (assets / "system_prompt.txt").read_text()},
        {
            "role": "user",
            "content": "Problem statement:\nThe client retries too quickly after a timeout. "
            "Find where the retry delay is calculated and applied.\n\nYou are at the repository root. "
            "Top-level contents:\nsrc/\nREADME.md\n\nFind the code locations relevant to this problem.",
        },
    ]
    search = observation(
        "colgrep 'retry delay'",
        "[1] src/retry.py:3-16 (retry_delay, request_with_retry)\n"
        "BASE_DELAY_SECONDS = 0.5\nreturn BASE_DELAY_SECONDS * (2 ** attempt)\n"
        "time.sleep(retry_delay(attempt))",
        1,
    )
    lines = "\n".join(f"{index:6}\t{line}" for index, line in enumerate(FIXTURE.splitlines(), 1))
    histories = [("search", messages, tools, "", "colgrep")]
    messages = [*messages, assistant("colgrep", {"query": "retry delay"}), search]
    histories.append(("read", messages, tools, "", "terminal"))
    messages = [
        *messages,
        assistant("terminal", {"command": "cat -n src/retry.py"}),
        observation("cat -n src/retry.py", lines, 2),
    ]
    histories.append(("finish", messages, tools, "", "finish"))
    histories.append(
        (
            "forced_finish",
            [*messages, {"role": "user", "content": FINAL_REMINDER}],
            [tools[-1]],
            '<function name="finish">',
            "finish",
        )
    )
    cases = []
    for name, history, active_tools, prefill, expected in histories:
        prompt = (
            tokenizer.apply_chat_template(
                history, tools=active_tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            + prefill
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        cases.append({"name": name, "prompt": prompt, "ids": ids, "prefill": prefill, "expected": expected})
    return cases, tools


def finish_valid(parsed):
    values = [params.get("locations", "") for name, params in parsed if name == "finish"]
    if len(values) != 1:
        return False
    try:
        locations = ast.literal_eval(values[0])
    except (SyntaxError, ValueError):
        return False
    if not isinstance(locations, list) or not locations:
        return False
    covered = set()
    for location in locations:
        match = (
            re.fullmatch(r"src/retry\.py:([1-9][0-9]*)-([1-9][0-9]*)", location) if isinstance(location, str) else None
        )
        if not match or not 1 <= int(match[1]) <= int(match[2]) <= len(FIXTURE.splitlines()):
            return False
        covered.update(range(int(match[1]), int(match[2]) + 1))
    return {7, 16}.issubset(covered) and len(covered) <= 12


def seal_source(source):
    result = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix not in {".json", ".jinja", ".safetensors"}:
            continue
        metadata = source / ".cache/huggingface/download" / (path.name + ".metadata")
        identity = metadata.read_text().splitlines()
        if not identity or identity[0] != REVISION:
            raise RuntimeError("source_revision_mismatch")
        sha256 = digest_file(path)
        if len(identity[1]) == 64 and identity[1] != sha256:
            raise RuntimeError("source_content_mismatch")
        if len(identity[1]) == 40:
            content = path.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if identity[1] != git_blob:
                raise RuntimeError("source_content_mismatch")
        result[path.name] = sha256
    if not any(name.endswith(".safetensors") for name in result):
        raise RuntimeError("source_incomplete")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    upstream_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream).decode().strip()
    if upstream_head != UPSTREAM:
        raise RuntimeError("upstream_revision_mismatch")
    for name, expected in ASSET_HASHES.items():
        if digest_file(args.upstream / "agent/assets" / name) != expected:
            raise RuntimeError("upstream_content_mismatch")
    source_seal = seal_source(args.source)
    model_seal = {
        path.name: digest_file(path)
        for path in sorted(args.model.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    }
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    template = args.upstream / "agent/assets/chat_template.jinja"
    if digest_file(template) != digest_file(args.source / "chat_template.jinja"):
        raise RuntimeError("upstream_template_mismatch")
    cases, _ = fixtures(tokenizer, args.upstream)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.source, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.to(args.device).eval()
    for case in cases:
        inputs = torch.tensor([case["ids"]], device=args.device)
        with torch.inference_mode():
            output = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                eos_token_id=sorted(EOS),
                pad_token_id=1,
            )
        ids = output[0, len(case["ids"]) :].tolist()
        case["reference_eos"] = ids[-1] if ids and ids[-1] in EOS else None
        case["reference"] = ids[:-1] if case["reference_eos"] is not None else ids
    del output, inputs, model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    reference_seconds = time.monotonic() - started
    print(json.dumps({"phase": "reference_completed", "cases": len(cases)}), flush=True)
    worker = Worker(args.worker, args.model)
    results = []
    controls = {}
    try:
        controls["eos_metadata"] = set(worker.ready["eos_token_ids"]) == EOS
        for case in cases:
            tokenized = worker.request({"operation": "tokenize", "text": case["prompt"]})
            response = worker.generate(case["prompt"])
            if "error" in response:
                raise RuntimeError("native_generation_failed")
            ids = response["generated_token_ids"]
            parsed = calls(case["prefill"] + response["text"])
            prefix = 0
            for a, b in zip(ids, case["reference"], strict=False):
                if a != b:
                    break
                prefix += 1
            results.append(
                {
                    "case": case["name"],
                    "input_tokens": len(case["ids"]),
                    "tokenization_exact": tokenized.get("token_ids") == case["ids"],
                    "decode_exact": response["text"] == tokenizer.decode(ids, skip_special_tokens=False),
                    "generated_tokens": len(ids),
                    "reference_tokens": len(case["reference"]),
                    "greedy_tokens_exact": ids == case["reference"],
                    "common_prefix_tokens": prefix,
                    "generated_ids_sha256": digest_ids(ids),
                    "reference_ids_sha256": digest_ids(case["reference"]),
                    "finish_reason": response["finish_reason"],
                    "stop_token_id": response["stop_token_id"],
                    "stop_exact": response["stop_token_id"] == case["reference_eos"],
                    "tool_contract": any(name == case["expected"] for name, _ in parsed),
                    "verified_location": finish_valid(parsed) if case["expected"] == "finish" else None,
                    "generation_seconds": response["generation_seconds"],
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "native_case_completed",
                        "case": case["name"],
                        "tokenization_exact": results[-1]["tokenization_exact"],
                        "greedy_tokens_exact": results[-1]["greedy_tokens_exact"],
                        "tool_contract": results[-1]["tool_contract"],
                    }
                ),
                flush=True,
            )
        request = {
            "operation": "generate",
            "input_token_ids": cases[0]["ids"],
            "context_tokens": 4096,
            "max_new_tokens": 32,
            "sampling": {"mode": "stochastic", "temperature": 0.6, "top_k": 20, "top_p": 0.95},
            "seed": 1,
        }
        a, b = worker.request(request), worker.request(request)
        controls["seeded_repeat"] = a.get("generated_token_ids") == b.get("generated_token_ids") and "error" not in a
        greedy = worker.generate(cases[0]["prompt"], max_tokens=8)
        controls["length_stop"] = greedy.get("output_tokens") == 8 and greedy.get("finish_reason") == "length"
        stop = worker.generate(cases[0]["prompt"], stop_token_ids=[greedy["generated_token_ids"][0]])
        controls["custom_stop"] = stop.get("finish_reason") == "stop_token" and stop.get("output_tokens") == 0
        bad = worker.request(
            {"operation": "generate", "input_token_ids": [130560], "context_tokens": 32, "max_new_tokens": 1}
        )
        controls["invalid_id_rejected"] = bad.get("error") == "invalid_generation_request"
        controls["error_recovery"] = (
            worker.generate(cases[0]["prompt"], max_tokens=8).get("generated_token_ids")
            == greedy["generated_token_ids"]
        )
        bound = worker.request(
            {
                "operation": "generate",
                "input_token_ids": cases[0]["ids"],
                "context_tokens": len(cases[0]["ids"]) + 2,
                "max_new_tokens": 8,
            }
        )
        controls["context_stop"] = bound.get("finish_reason") == "context" and bound.get("sampled_tokens") == 2
    finally:
        worker.close()
    with Worker(args.worker, args.model) as restart:
        controls["restart_repeat"] = (
            restart.generate(cases[0]["prompt"], max_tokens=8).get("generated_token_ids")
            == greedy["generated_token_ids"]
        )
    controls["source_unchanged"] = source_seal == seal_source(args.source)
    controls["model_unchanged"] = model_seal == {
        path.name: digest_file(path)
        for path in sorted(args.model.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    }
    report = {
        "schema": 1,
        "model": "lightonai/colgrep-agent-2B",
        "revision": REVISION,
        "upstream_harness_revision": UPSTREAM,
        "scope": "controlled_generation_and_tool_protocol",
        "live_search_or_app_integration": False,
        "reference": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "device": args.device,
            "attention": "eager",
            "seconds": reference_seconds,
        },
        "controls": controls,
        "cases": results,
        "worker_sha256": digest_file(args.worker),
        "cargo_lock_sha256": digest_file(ROOT / "Cargo.lock"),
        "tools_lock_sha256": digest_file(ROOT / "tools/uv.lock"),
        "source_files": source_seal,
        "model_files": model_seal,
        "harness_files": {
            path.name: digest_file(path)
            for path in sorted((args.upstream / "agent/assets").iterdir())
            if path.is_file()
        },
        "validator_sha256": digest_file(Path(__file__)),
        "named_build": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "profile": "release",
            "package": "stark-generate",
            "features": ["metal"],
            "rust_toolchain": "1.98.1",
            "source_files": {
                str(path.relative_to(ROOT)): digest_file(path)
                for directory in (
                    ROOT / "crates/stark-generate",
                    ROOT / "crates/stark-engine",
                    ROOT / "crates/stark-engine-macros",
                )
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix in {".rs", ".metal", ".h", ".cpp", ".toml"}
            },
        },
    }
    report["passed"] = all(controls.values()) and all(
        case["tokenization_exact"]
        and case["decode_exact"]
        and case["greedy_tokens_exact"]
        and case["stop_exact"]
        and case["tool_contract"]
        and case["verified_location"] is not False
        for case in results
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "validation_completed",
                "passed": report["passed"],
                "exact_greedy_cases": sum(case["greedy_tokens_exact"] for case in results),
                "cases": len(results),
                "controls_passed": sum(controls.values()),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        code = (
            str(error)
            if isinstance(error, RuntimeError) and re.fullmatch(r"[a-z_]+", str(error))
            else "generation_validation_failed"
        )
        print(code, flush=True)
        raise SystemExit(1) from None
