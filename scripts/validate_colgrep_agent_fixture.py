"""Drive generated search/read/finish turns with controlled, read-only observations.

This checks inference integration. It does not run a live ColGrep search index or
execute model-proposed shell commands. Persist aggregates and hashes only.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

import validate_colgrep_generation as validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    reference = json.loads(args.reference_report.read_text())
    worker_hash = validation.digest_file(args.worker)
    source_seal = validation.seal_source(args.source)
    model_seal = {
        p.name: validation.digest_file(p)
        for p in args.model.iterdir()
        if p.is_file() and p.suffix in {".json", ".safetensors"}
    }
    if (
        source_seal != reference["source_files"]
        or model_seal != reference["model_files"]
        or worker_hash != reference["worker_sha256"]
    ):
        raise RuntimeError("reference_identity_mismatch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream).decode().strip()
    if head != validation.UPSTREAM:
        raise RuntimeError("upstream_revision_mismatch")
    for name, expected in validation.ASSET_HASHES.items():
        if validation.digest_file(args.upstream / "agent/assets" / name) != expected:
            raise RuntimeError("upstream_content_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    cases, tools = validation.fixtures(tokenizer, args.upstream)
    messages = list(cases[0]["messages"])
    lines = "\n".join(f"{i:6}\t{line}" for i, line in enumerate(validation.FIXTURE.splitlines(), 1))
    counts = {"colgrep": 0, "terminal": 0, "finish": 0}
    turns = []
    locations_valid = False
    with validation.Worker(args.worker, args.model) as worker:
        for turn in range(1, 11):
            prompt = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            response = worker.generate(prompt, max_tokens=512, seed=turn)
            if "error" in response:
                raise RuntimeError("native_generation_failed")
            text = response["text"]
            parsed = validation.calls(text)
            turns.append(
                {
                    "turn": turn,
                    "generated_tokens": response["output_tokens"],
                    "generated_ids_sha256": validation.digest_ids(response["generated_token_ids"]),
                    "finish_reason": response["finish_reason"],
                    "stop_token_id": response["stop_token_id"],
                    "valid_xml": bool(parsed),
                    "calls": len(parsed),
                    "call_sequence": [name for name, _ in parsed],
                }
            )
            if not parsed or len(parsed) > 5 or response["finish_reason"] != "eos":
                break
            typed = []
            for name, params in parsed:
                arguments = {
                    key: ast.literal_eval(value) if key in {"k", "locations"} else value
                    for key, value in params.items()
                }
                typed.append({"type": "function", "function": {"name": name, "arguments": arguments}})
            messages.append({"role": "assistant", "content": validation.assistant_content(text), "tool_calls": typed})
            for name, params in parsed:
                if name == "finish":
                    counts[name] += 1
                    locations_valid = (
                        counts["colgrep"] > 0 and counts["terminal"] > 0 and validation.finish_valid(parsed)
                    )
                    break
                if name == "colgrep":
                    if not params.get("query", "").strip() and not params.get("command", "").strip():
                        raise RuntimeError("semantic_search_missing")
                    command = params.get("command") or "colgrep " + repr(params["query"])
                    output = "[1] src/retry.py:3-16 (retry_delay, request_with_retry)\n" + validation.FIXTURE
                    counts[name] += 1
                else:
                    command = params["command"]
                    if not re.match(r"^(?:cat|head|tail|sed)\b", command.strip()) or "src/retry.py" not in command:
                        raise RuntimeError("unsupported_fixture_read")
                    output = lines
                    counts[name] += 1
                messages.append(validation.observation(command, output, turn))
            if counts["finish"]:
                break
    passed = locations_valid and counts["finish"] == 1
    result = {
        "schema": 1,
        "passed": passed,
        "scope": "controlled_agent_loop",
        "live_search_or_app_integration": False,
        "arbitrary_shell_execution": False,
        "forced_finish": False,
        "context_tokens": 4096,
        "max_new_tokens": 512,
        "model_revision": validation.REVISION,
        "upstream_revision": validation.UPSTREAM,
        "worker_sha256": worker_hash,
        "source_files": source_seal,
        "model_files": model_seal,
        "reference_report_sha256": validation.digest_file(args.reference_report),
        "script_sha256": validation.digest_file(Path(__file__)),
        "validator_sha256": validation.digest_file(Path(validation.__file__)),
        "counts": counts,
        "verified_calculation_and_application": locations_valid,
        "turns": turns,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phase": "agent_fixture_completed", "passed": passed, "turns": len(turns), "counts": counts}))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        code = (
            str(error)
            if isinstance(error, RuntimeError) and re.fullmatch(r"[a-z_]+", str(error))
            else "agent_fixture_failed"
        )
        print(code)
        raise SystemExit(1) from None
