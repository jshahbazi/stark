#!/usr/bin/env python3
"""Run isolated sessions of the installed Pi CLI against the source Stark API."""

import argparse
import contextlib
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import secrets
import signal
import struct
import subprocess
import threading
import time
from urllib.request import Request, urlopen

from fixtures import TASKS, grade, prepare
from stark_convert.desktop import Desktop
from stark_convert.local_api import APIError


ROOT = Path(__file__).resolve().parents[4]
MODEL = "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
CONTEXT = 40960
MAX_OUTPUT = 2048
SAMPLING = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 42}
TOOLS = ("read", "edit", "write")
TOOL_LIMIT = 8
# Every task runs twice per grid, with the two opposite worker-launch orders.
FIRST_TASKS, SECOND_TASKS = tuple(TASKS)[:2], tuple(TASKS)[2:]
SCHEDULE = [(grid, batch) for order in ((4096, 2048, 2048, 4096), (2048, 4096, 4096, 2048))
            for grid, batch in zip(order, (FIRST_TASKS, FIRST_TASKS, SECOND_TASKS, SECOND_TASKS))]


class TrialError(Exception):
    pass


def require(condition, code):
    if not condition:
        raise TrialError(code)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def token_digest(ids):
    return sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def host_snapshot(pid=None):
    values = {}
    try:
        for row in subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=3).splitlines():
            key, _, value = row.partition(":")
            if key in {"Pages wired down", "Pageins", "Pageouts", "Swapins", "Swapouts"}:
                values[key] = int(value.strip().rstrip("."))
        if pid is not None:
            values["worker_rss_kib"] = int(subprocess.check_output(
                ["/bin/ps", "-p", str(pid), "-o", "rss="], text=True, timeout=3))
    except (OSError, ValueError, subprocess.SubprocessError):
        values["incomplete"] = True
    return values


class Recorder:
    def __init__(self, file):
        self.file = file
        self.session = None
        self.phase = None

    def emit(self, kind, **details):
        row = {"type": kind, "session": self.session, "phase": self.phase, **details}
        self.file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())


class AgentDesktop(Desktop):
    def api_complete(self, payload, on_delta=None, cancelled=None):
        # Engineering controls shared across arms. Actual tools, prompts and
        # histories come from Pi and pass through the normal Stark renderer.
        try:
            return super().api_complete({**payload, **SAMPLING, "mtp": True}, on_delta, cancelled)
        except APIError as error:
            self.report.emit("api_error", status=error.status, code=error.code)
            raise


class NativeTrace:
    def __init__(self, worker, grid, report):
        self.original = worker.request
        self.grid = grid
        self.report = report
        self.previous = None
        self.count = 0

    def request(self, payload, *args, **kwargs):
        if payload.get("operation") != "generate":
            return self.original(payload, *args, **kwargs)
        self.count += 1
        tokens = tuple(payload["input_token_ids"])
        context = payload["context_tokens"]
        scope = payload.get("cache_scope")
        require(payload.get("mtp_drafts") == 2 and context == CONTEXT and scope, "request_settings")
        expected = 0
        if self.previous is not None:
            old, old_context, old_scope = self.previous
            boundary = (len(old) - 1) // self.grid * self.grid
            if context == old_context and scope == old_scope and len(tokens) >= boundary + 2:
                if tokens[:boundary] == old[:boundary]:
                    expected = boundary

        started = time.monotonic()
        reply = self.original(payload, *args, **kwargs)
        elapsed = time.monotonic() - started
        if "error" in reply:
            self.previous = None
            self.report.emit("native_error", request=self.count, input_tokens=len(tokens))
            return reply
        self.previous = tokens, context, scope
        mtp = reply.get("mtp", {})
        stats = mtp.get("stats", {})
        counts = {key: stats.get(key) for key in ("rounds", "proposed", "accepted", "verified_rows")}
        reused = reply.get("reused_input_tokens")
        sampled = reply["sampled_tokens"]
        valid_mtp = (
            mtp.get("enabled") is True and mtp.get("drafts") == 2
            and mtp.get("verification") == "batched-decode-exact-v1" and mtp.get("sampling") == "exact-rejection-v1"
            and all(type(value) is int and value >= 0 for value in counts.values())
            and 0 <= counts["accepted"] <= counts["proposed"] <= 2 * counts["rounds"]
            and counts["verified_rows"] == counts["rounds"] + counts["proposed"]
            and (sampled <= 3 or counts["proposed"] > 0)
        )
        self.report.emit("native_request", request=self.count, input_sha256=token_digest(tokens),
                         output_sha256=token_digest(reply["generated_token_ids"]), input_tokens=len(tokens),
                         cached_tokens=reused, expected_cached=expected, mtp_valid=valid_mtp, mtp=counts,
                         sampled_tokens=sampled, output_tokens=reply["output_tokens"], finish=reply["finish_reason"],
                         prefill_seconds=reply["prefill_seconds"], decode_seconds=reply["decode_seconds"],
                         worker_wall_seconds=elapsed, output_budget=payload["max_new_tokens"])
        require(valid_mtp and reused == expected and reply["input_tokens"] == len(tokens), "native_acknowledgement")
        return reply


def agent_command(args, work, profile, port):
    # Allow the installed runtime and reviewed guard to be read. Pi's tools
    # can write only to this session's private workspace/profile. The only
    # permitted network connection is to this run's Stark listener.
    literal = lambda path: json.dumps(str(path))
    roots = {work.resolve(), profile.resolve(), args.guard.resolve().parents[1],
             args.pi.resolve().parents[1] / "install", args.pi.resolve().parent,
             args.node.resolve().parent.parent}
    reads = " ".join(f"(subpath {literal(path)})" for path in sorted(roots))
    writes = " ".join(f"(subpath {literal(path.resolve())})" for path in (work, profile))
    sandbox = (f"(version 1)(allow default)(deny file-read* (subpath {literal(Path.home())}))"
               f"(allow file-read-metadata (subpath {literal(Path.home())}))"
               f"(allow file-read* {reads})(deny file-write*)(allow file-write* {writes} (literal \"/dev/null\"))"
               f"(deny network*)(allow network-outbound (remote tcp \"localhost:{port}\"))")
    return ["/usr/bin/sandbox-exec", "-p", sandbox, str(args.pi.resolve()), "--mode", "rpc", "--no-session",
            "--provider", "stark-local", "--model", MODEL, "--thinking", "off", "--offline", "--no-extensions",
            "--extension", str(args.guard), "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-themes",
            "--no-context-files", "--no-approve", "--tools", ",".join(TOOLS)]


class Pi:
    def __init__(self, args, work, profile, port, key):
        self.queue = Queue(maxsize=512)
        self.sequence = 0
        self.process = subprocess.Popen(agent_command(args, work, profile, port), cwd=work,
            env={"PATH": f"{args.node.parent}:/usr/bin:/bin", "HOME": str(profile), "TMPDIR": str(profile),
                 "LANG": "en_US.UTF-8", "PI_CODING_AGENT_DIR": str(profile), "PI_OFFLINE": "1",
                 "PI_TELEMETRY": "0", "STARK_API_KEY": key},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, start_new_session=True)
        self.reader = threading.Thread(target=self.read_events, daemon=True)
        self.reader.start()

    def read_events(self):
        for line in self.process.stdout:
            if len(line) > 4 * 1024 * 1024:
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            self.queue.put(event)
        self.queue.put(None)

    def send(self, command, **payload):
        self.sequence += 1
        ident = str(self.sequence)
        self.process.stdin.write(json.dumps({"type": command, "id": ident, **payload}) + "\n")
        self.process.stdin.flush()
        return ident

    def wait(self, predicate, deadline, observe=lambda event: None):
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                event = self.queue.get(timeout=remaining)
            except Empty:
                break
            require(event is not None, "pi_exited")
            observe(event)
            if predicate(event):
                return event
        raise TrialError("turn_timeout")

    def rpc(self, command, timeout=20, **payload):
        ident = self.send(command, **payload)
        event = self.wait(lambda row: row.get("type") == "response" and row.get("id") == ident,
                          time.monotonic() + timeout)
        require(event.get("success") is True, "pi_rpc_failed")
        return event.get("data") or {}

    def prompt(self, text, timeout):
        started = time.monotonic()
        ident = self.send("prompt", message=text)
        tools = {"calls": 0, "errors": 0}
        tool_totals = {}
        first_text = None
        assistant = []
        aborted_at_limit = False

        def observe(event):
            nonlocal first_text, aborted_at_limit
            kind = event.get("type")
            if kind == "response" and event.get("id") == ident:
                require(event.get("success") is True, "pi_prompt_rejected")
            if kind == "message_update" and event.get("assistantMessageEvent", {}).get("type") == "text_delta":
                first_text = first_text if first_text is not None else time.monotonic() - started
            if kind == "tool_execution_end":
                tools["calls"] += 1
                tools["errors"] += int(event.get("isError") is True)
                name = event.get("toolName")
                totals = tool_totals.setdefault(name if name in TOOLS else "other", {"calls": 0, "errors": 0})
                totals["calls"] += 1
                totals["errors"] += int(event.get("isError") is True)
                if tools["calls"] >= TOOL_LIMIT and not aborted_at_limit:
                    aborted_at_limit = True
                    self.send("abort")
            message = event.get("message", {})
            if kind == "message_end" and message.get("role") == "assistant":
                reason = message.get("stopReason")
                reason = reason if reason in {"stop", "length", "toolUse", "aborted", "error"} else "unknown"
                match = re.search(r"stark_http_\d{3}|missing_usage|missing_finish_reason|missing_done|invalid_usage|"
                                  r"upstream_error|transport_failed|invalid_request_route",
                                  message.get("errorMessage") or "")
                assistant.append((reason, match.group(0) if match else None))

        self.wait(lambda row: row.get("type") == "agent_settled", started + timeout, observe)
        reason, category = assistant[-1] if assistant else ("missing", None)
        if aborted_at_limit:
            reason = "tool_limit"
        return {**tools, "wall_seconds": time.monotonic() - started, "first_text_seconds": first_text,
                "assistant_messages": len(assistant), "client_finish": reason, "client_error": category,
                "tool_totals": tool_totals}

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()


def configure_pi(profile, models, base_url):
    model = models[0]
    require(len(models) == 1 and model["id"] == MODEL and model["tool_calling"], "api_model")
    require(model["defaults"]["mtp"] and model["capabilities"]["mtp"]["recommended_drafts"] == 2, "api_mtp")
    provider = {"baseUrl": base_url, "api": "stark-guarded-chat-completions", "apiKey": "${STARK_API_KEY}",
        "authHeader": True, "compat": {"supportsStore": False, "supportsDeveloperRole": False,
          "supportsReasoningEffort": False, "supportsStrictMode": False, "supportsUsageInStreaming": True,
          "supportsFinishReason": True, "supportsLongCacheRetention": False, "promptCacheKeyMode": "disabled",
          "maxTokensField": "max_tokens"},
        "models": [{"id": MODEL, "name": "Stark running model", "reasoning": False, "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": CONTEXT, "maxTokens": MAX_OUTPUT}]}
    (profile / "models.json").write_text(json.dumps({"providers": {"stark-local": provider}}))
    (profile / "settings.json").write_text(json.dumps({"retry": {"enabled": False, "provider": {"maxRetries": 0}},
                                                      "compaction": {"enabled": False}}))


def call_api(url, key, payload=None):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with urlopen(Request(url, data=json.dumps(payload).encode() if payload else None, headers=headers), timeout=180) as response:
        return json.load(response)


def session(args, desktop, tasks, grid, block, report, base, key, models):
    worker = desktop.worker
    for task in tasks:
        report.session = f"{block}-{task}"
        # Pi embeds the actual cwd in its system prompt. Every session needs
        # fresh files at the same path for any exact native inputs to match.
        workspace = args.output / "active-workspace"
        profile = args.output / f"{report.session}-pi"
        workspace.mkdir()
        profile.mkdir()
        turns = prepare(task, workspace)
        configure_pi(profile, models, base)
        desktop.reset_prefix_cache()
        trace = NativeTrace(worker, grid, report)
        worker.request = trace.request
        report.emit("session_start", grid=grid, block=block, task=task, host=host_snapshot(worker.process.pid))
        started = time.monotonic()
        try:
            with contextlib.closing(Pi(args, workspace, profile, desktop.api_port, key)) as pi:
                state = pi.rpc("get_state")
                require(state.get("model", {}).get("provider") == "stark-local" and state["model"]["id"] == MODEL,
                        "pi_model_selection")
                passed = True
                for phase, prompt in enumerate(turns, 1):
                    trace.count = 0
                    report.phase = phase
                    result = pi.prompt(prompt, args.turn_timeout)
                    mark = grade(task, workspace, phase)
                    turn_passed = mark["status"] == "ok" and mark["passed"] == mark["total"] and mark["total"] > 0
                    passed = passed and turn_passed and result["client_finish"] == "stop"
                    report.emit("turn", **result, grade=mark, native_requests=trace.count)
                    print(f"block {block} grid {grid} task {task} turn {phase}: {mark['passed']}/{mark['total']} "
                          f"({result['client_finish']})", flush=True)
            report.phase = None
            report.emit("session_end", passed=passed, wall_seconds=time.monotonic() - started,
                        host=host_snapshot(worker.process.pid))
        except (TrialError, OSError, ValueError) as error:
            report.emit("session_failure", code=str(error) if isinstance(error, TrialError) else type(error).__name__)
        finally:
            worker.request = trace.original
            workspace.rename(args.output / f"{report.session}-work")
            report.phase = report.session = None


def arm(args, report, block, grid, tasks):
    os.environ["STARK_LILY_PREFILL_CHUNK"] = str(grid)
    library = args.output / "library"
    with open(os.devnull, "w") as sink:
        desktop = AgentDesktop(args.worker, library, output=sink)
        desktop.report = report
        try:
            desktop.models = [{"id": "lily", "name": "Qwen3.8 Flash Next", "path": str(args.model),
                               "source": f"huggingface:{MODEL}", "mtp_enabled": True}]
            started = time.monotonic()
            desktop.run(desktop.epoch, "lily")
            require(desktop.worker.prefix_chunk_tokens == grid and desktop.worker.images, "native_ready")
            desktop.api_context = CONTEXT
            key = secrets.token_hex(32)
            desktop.start_api(key, 0)
            base = f"http://127.0.0.1:{desktop.api_port}/v1"
            models = call_api(base + "/models", key)["data"]
            # Warm the production host/template/native pipeline with a fixed,
            # unused request. Session start clears that request's cache.
            call_api(base + "/chat/completions", key, {"model": MODEL, "max_tokens": 16,
                "messages": [{"role": "user", "content": "Summarize the invariant in this Rust source excerpt:\n" +
                              (ROOT / "vendor/lily-core/src/qwen4exp/model.rs").read_text()[:12000]}]})
            report.emit("arm_ready", block=block, grid=grid, loaded_vision=True,
                        startup_and_warmup_seconds=time.monotonic() - started)
            session(args, desktop, tasks, grid, block, report, base, key, models)
        finally:
            desktop.stop(quiet=True)
            if desktop.api is not None:
                desktop.api.close()


def main():
    def interrupted(_number, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("worker", "model", "pi", "node", "guard", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--block", type=int, action="append", choices=range(1, 9))
    parser.add_argument("--turn-timeout", type=int, default=180)
    args = parser.parse_args()
    os.umask(0o077)
    for arg in ("worker", "model", "pi", "node", "guard", "output"):
        setattr(args, arg, getattr(args, arg).resolve())
    args.output.mkdir()
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    with (args.output / "aggregate.jsonl").open("x") as file:
        report = Recorder(file)
        report.emit("manifest", git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    worker_sha256=digest(args.worker), runner_sha256=digest(Path(__file__)),
                    fixtures_sha256=digest(Path(__file__).with_name("fixtures.py")),
                    model_manifest_sha256=digest(args.model / "stark-image-artifact.json"), guard_sha256=digest(args.guard),
                    context_tokens=CONTEXT, max_output=MAX_OUTPUT, tools=TOOLS, tool_limit=TOOL_LIMIT,
                    sampling=SAMPLING, schedule=SCHEDULE)
        for block, (grid, tasks) in enumerate(SCHEDULE, 1):
            if args.block and block not in args.block:
                continue
            try:
                arm(args, report, block, grid, tasks)
            except Exception as error:
                report.emit("arm_failure", block=block, grid=grid,
                            code=str(error) if isinstance(error, TrialError) else type(error).__name__)
                print(f"block {block} failed: {type(error).__name__}", flush=True)
        report.emit("complete")


if __name__ == "__main__":
    main()
