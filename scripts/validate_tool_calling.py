"""Qualify a bounded synthetic tool workflow; write aggregate results and hashes.

No tool is executed. Requests, generated text, arguments and token vectors stay
in memory/private pipes. Run separately from other native model workloads.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from stark_convert.api_tools import (
    parse_tool_output,
    template_messages,
    template_tools,
    tool_parser_for,
    validate_parser_tools,
)
from stark_convert.chat import Worker, load_template, read_model_config, render_template
from stark_convert.template_runtime import _template_executable
from stark_convert.tool_protocols import protocol_stop_ids


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_files(model):
    """Bind optional runtime metadata and shard selection as well as payloads."""
    optional = (
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "ngram_table.bin",
    )
    return sorted(
        {
            model / "config.json",
            model / "tokenizer.json",
            *(model / name for name in optional if (model / name).exists()),
            *model.rglob("*.safetensors"),
            *model.rglob("*.gguf"),
        }
    )


def final_text(message, *, initial_thinking):
    content = message.get("content") or ""
    if initial_thinking and "</think>" not in content:
        return None
    return content.split("</think>")[-1].strip()


def run_checks(worker, template, codec, *, context, maximum):
    parser = tool_parser_for(codec)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up the supplied key and return its result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The lookup key."}
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    validate_parser_tools(tools, parser)

    def generate(messages, definitions, choice):
        prompt = render_template(
            template,
            messages=template_messages(
                messages, parser=parser, tools=definitions if choice != "none" else None
            ),
            tools=template_tools(
                definitions if choice != "none" else None, parser=parser
            ),
            add_generation_prompt=True,
            enable_thinking=False,
            **{k: codec[k] for k in ("bos_token", "eos_token") if k in codec},
        )
        request = {
            "operation": "generate",
            "rendered_prompt": prompt,
            "context_tokens": context,
            "max_new_tokens": maximum,
            "sampling": {"mode": "greedy"},
        }
        stop_ids = protocol_stop_ids(
            worker,
            parser,
            active=bool(definitions or any(m.get("tool_calls") for m in messages)),
        )
        if stop_ids:
            request["stop_token_ids"] = stop_ids
        if getattr(worker, "mtp", {}).get("supported"):
            request["mtp_drafts"] = 0
        reply = worker.request(request)
        if (
            "error" in reply
            or not isinstance(reply.get("text"), str)
            or reply.get("finish_reason") not in {"eos", "stop_token", "stop"}
        ):
            raise ValueError("generation_failed")
        message, reason = parse_tool_output(
            reply["text"],
            definitions,
            parser=parser,
            finish_reason=reply.get("finish_reason"),
            call_id_prefix="qualification",
            parallel_tool_calls=False,
            tool_choice=choice,
            initial_thinking=prompt.endswith("<think>"),
        )
        return message, reason, prompt.endswith("<think>")

    messages = [
        {
            "role": "user",
            "content": "Use lookup exactly once with key alpha. Wait for its result before answering.",
        }
    ]
    checks = {
        "single_call_and_arguments": False,
        "tool_result_use": False,
        "ordinary_chat": False,
    }
    try:
        assistant, reason, _ = generate(messages, tools, "required")
        calls = assistant.get("tool_calls", [])
        checks["single_call_and_arguments"] = (
            reason == "tool_calls"
            and len(calls) == 1
            and calls[0]["function"]["name"] == "lookup"
            and json.loads(calls[0]["function"]["arguments"]) == {"key": "alpha"}
        )
        if checks["single_call_and_arguments"]:
            messages += [
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "The lookup result is K7V3M9.",
                },
                {
                    "role": "user",
                    "content": "Reply with only the exact lookup result code. Do not call tools.",
                },
            ]
            answer, reason, initial_thinking = generate(messages, tools, "none")
            visible = final_text(answer, initial_thinking=initial_thinking)
            checks["tool_result_use"] = reason == "stop" and visible == "K7V3M9"
    except Exception:  # noqa: BLE001 - discard native/model diagnostics, never log them.
        # Model output and native errors are never included in the report.
        checks["tool_result_use"] = False
    try:
        answer, reason, initial_thinking = generate(
            [{"role": "user", "content": "Reply with only HELLO."}], [], "none"
        )
        checks["ordinary_chat"] = (
            reason == "stop"
            and final_text(answer, initial_thinking=initial_thinking) == "HELLO"
        )
    except Exception:  # noqa: BLE001 - discard native/model diagnostics, never log them.
        checks["ordinary_chat"] = False
    return checks


def main():
    args_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args_parser.add_argument("--model", type=Path, required=True)
    args_parser.add_argument("--worker", type=Path, required=True)
    args_parser.add_argument("--output", type=Path, required=True)
    args_parser.add_argument("--context-tokens", type=int, default=8192)
    args_parser.add_argument("--max-new-tokens", type=int, default=2048)
    args_parser.add_argument("--timeout", type=float, default=300)
    args = args_parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError("Choose a new validation output path.")
    if (
        not 0 < args.max_new_tokens < args.context_tokens
        or not 0 < args.timeout <= 3600
    ):
        args_parser.error(
            "Choose a positive output budget below context and a timeout up to 3600 seconds."
        )
    report = {
        "schema_version": 1,
        "scope": "bounded_synthetic_tool_workflow",
        "date_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "context_tokens": args.context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "sampling": "greedy",
        "mtp_drafts": 0,
        "passed": False,
    }
    try:
        # Seal actual bytes, not saved model names or an unverified receipt.
        files = artifact_files(args.model)
        if not any(p.suffix in {".safetensors", ".gguf"} for p in files):
            raise ValueError("weights_unavailable")
        if (
            (args.model / "experts.bin").exists()
            or (args.model / "mtp-experts-nvfp4.bin").exists()
            or read_model_config(args.model).get("expert_streaming") is not None
        ):
            raise ValueError("external_expert_banks_need_separate_validation")

        def identity(path):
            current = path.stat()
            return current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns

        identities = {p: identity(p) for p in files}
        digests = {str(p.relative_to(args.model)): sha256(p) for p in files}
        report["artifact_sha256"] = hashlib.sha256(
            json.dumps(digests, sort_keys=True).encode()
        ).hexdigest()
        helper = _template_executable()
        helper_stat = identity(helper)
        report["template_executable_sha256"] = sha256(helper)
        codec, template = load_template(args.model)
        if template._executable != helper:
            raise ValueError("template_executable_changed")
        parser = tool_parser_for(codec)
        if parser is None:
            raise ValueError("tools_unsupported")
        worker_stat = identity(args.worker)
        report.update(
            parser=parser,
            template_sha256=hashlib.sha256(
                codec["prompt_template"].encode()
            ).hexdigest(),
            rendered_template_sha256=hashlib.sha256(
                template._source.encode()
            ).hexdigest(),
            worker_sha256=sha256(args.worker),
            validator_sha256=sha256(Path(__file__)),
        )
        worker = Worker(args.worker.resolve(), args.model.resolve(), args.timeout)
        try:
            ready = worker.read()
            if (
                ready.get("ready") is not True
                or ready.get("mode") != "text-generation-v1"
            ):
                raise ValueError("worker_unavailable")
            report["checks"] = run_checks(
                worker,
                template,
                codec,
                context=args.context_tokens,
                maximum=args.max_new_tokens,
            )
        finally:
            worker.close()
        unchanged = (
            all(identity(p) == previous for p, previous in identities.items())
            and identity(args.worker) == worker_stat
            and identity(helper) == helper_stat
            and artifact_files(args.model) == files
        )
        unchanged = (
            unchanged
            and {str(p.relative_to(args.model)): sha256(p) for p in files} == digests
            and sha256(args.worker) == report["worker_sha256"]
            and sha256(helper) == report["template_executable_sha256"]
        )
        report["artifact_unchanged"] = unchanged
        report["passed"] = unchanged and all(report["checks"].values())
    except Exception:  # noqa: BLE001 - report a fixed category, never exception dumps.
        report["failure"] = "validation_unavailable_or_failed"
    # Refuse to replace a prior run or an existing model/config file.
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "checks": report.get("checks", {})}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError:
        print(
            "Cannot write the validation record; choose a new writable output path.",
            file=sys.stderr,
        )
        sys.exit(1)
