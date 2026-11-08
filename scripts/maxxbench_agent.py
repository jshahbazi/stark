"""Privacy-preserving host-side mini-swe-agent adapter for Pier.

The mini-swe-agent prompts are loaded from its installed 2.4.6 package at
runtime. Conversations exist only in process memory and are never serialized.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import http.client
import ipaddress
import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

# mini-swe-agent normally imports user credentials from its global .env file.
# This adapter uses an in-process credential and must not consult that file.
os.environ["MSWEA_SILENT_STARTUP"] = "1"
_MINI_CONFIG_DIR = tempfile.mkdtemp(prefix="stark-maxxbench-mini-config-")
os.chmod(_MINI_CONFIG_DIR, 0o700)
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = _MINI_CONFIG_DIR

with (
    open(os.devnull, "w") as _quiet_sink,
    contextlib.redirect_stdout(_quiet_sink),
    contextlib.redirect_stderr(_quiet_sink),
):
    from minisweagent import package_dir
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.exceptions import Submitted
    from minisweagent.models.utils.actions_toolcall import (
        BASH_TOOL,
        format_toolcall_observation_messages,
        parse_toolcall_actions,
    )
    from pier.agents.base import BaseAgent

MODEL_TEMPERATURE = 0.0
MODEL_MAX_TOKENS = 4096
MODEL_SEED = 0
DEFAULT_STEP_LIMIT = 250
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_ENDPOINT = ""
_AUTHORIZATION = ""
_MODEL_NAME = ""
_CONFIG_LOCK = threading.Lock()


@dataclass
class AggregateMetrics:
    """Only non-content metrics are retained across the adapter."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_overflow_errors: int = 0
    http_error_categories: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_error(self, category: str, *, context_overflow: bool = False) -> None:
        with self._lock:
            self.http_error_categories[category] = (
                self.http_error_categories.get(category, 0) + 1
            )
            if context_overflow:
                self.context_overflow_errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "context_overflow_errors": self.context_overflow_errors,
                "http_error_categories": dict(self.http_error_categories),
            }


METRICS = AggregateMetrics()


def configure_connection(base_url: str, key: str, model: str) -> None:
    """Set host-process connection data without using environment/config files."""
    global _ENDPOINT, _AUTHORIZATION, _MODEL_NAME
    parsed = urlsplit(base_url)
    host = parsed.hostname
    try:
        is_loopback_ip = bool(host and ipaddress.ip_address(host).is_loopback)
    except ValueError:
        is_loopback_ip = False
    if (
        parsed.scheme != "http"
        or not host
        or not (is_loopback_ip or host.lower() == "localhost")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be a plain HTTP loopback URL")
    try:
        _port = parsed.port
    except ValueError:
        raise ValueError("endpoint port is invalid") from None
    endpoint = base_url.rstrip("/") + "/chat/completions"
    with _CONFIG_LOCK:
        _ENDPOINT = endpoint
        _AUTHORIZATION = "Bearer " + key
        _MODEL_NAME = model


def clear_connection() -> None:
    """Drop the in-memory endpoint credential after a profile finishes."""
    global _ENDPOINT, _AUTHORIZATION, _MODEL_NAME
    with _CONFIG_LOCK:
        _ENDPOINT = ""
        _AUTHORIZATION = ""
        _MODEL_NAME = ""


def _connection_snapshot() -> tuple[str, str]:
    with _CONFIG_LOCK:
        return _ENDPOINT, _AUTHORIZATION


def _safe_http_category(status: int | None) -> str:
    if status is None:
        return "transport"
    if status == 400 or status == 413:
        return "request_rejected"
    if status in (401, 403):
        return "authentication"
    if status == 404:
        return "endpoint"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server"
    return "http_other"


class _CancellableHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, port, *, timeout, stop_event, connection_lock):
        super().__init__(host, port, timeout=timeout)
        self._stop_event = stop_event
        self._connection_lock = connection_lock

    def connect(self):
        with self._connection_lock:
            if self._stop_event.is_set():
                raise OSError("request cancelled")
            super().connect()


class LocalOpenAIModel:
    """Small OpenAI chat-completions client with redacted error handling."""

    def __init__(
        self,
        model_name: str,
        *,
        timeout: float = 600.0,
        max_tokens: int = MODEL_MAX_TOKENS,
        temperature: float = MODEL_TEMPERATURE,
        observation_template: str,
        format_error_template: str,
        stop_event: threading.Event | None = None,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.observation_template = observation_template
        self.format_error_template = format_error_template
        self.stop_event = stop_event or threading.Event()
        self._connection_lock = threading.Lock()
        self._active_connection: http.client.HTTPConnection | None = None
        self.config = type("ModelConfig", (), {"model_name": model_name})()

    def abort(self) -> None:
        """Close an active loopback connection when Pier cancels a trial."""
        with self._connection_lock:
            connection = self._active_connection
            if connection is not None:
                if connection.sock is not None:
                    try:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                connection.close()

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self.stop_event.is_set():
            METRICS.record_error("cancelled")
            raise RuntimeError("agent stopped")
        endpoint, authorization = _connection_snapshot()
        if not endpoint:
            METRICS.record_error("configuration")
            raise RuntimeError("model endpoint unavailable")

        body = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {k: v for k, v in m.items() if k != "extra"} for m in messages
                ],
                "tools": [BASH_TOOL],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": MODEL_SEED,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        parsed_endpoint = urlsplit(endpoint)
        connection = _CancellableHTTPConnection(
            parsed_endpoint.hostname,
            parsed_endpoint.port or 80,
            timeout=self.timeout,
            stop_event=self.stop_event,
            connection_lock=self._connection_lock,
        )

        with METRICS._lock:
            METRICS.calls += 1
        try:
            with self._connection_lock:
                if self.stop_event.is_set():
                    METRICS.record_error("cancelled")
                    raise RuntimeError("agent stopped")
                self._active_connection = connection
            target = parsed_endpoint.path or "/"
            if parsed_endpoint.query:
                target += "?" + parsed_endpoint.query
            connection.request("POST", target, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                METRICS.record_error("response_too_large")
                raise RuntimeError("model response rejected")
            if response.status < 200 or response.status >= 300:
                category = _safe_http_category(response.status)
                overflow = any(
                    term in raw[:4096].lower()
                    for term in (
                        b"context length",
                        b"context window",
                        b"too many tokens",
                    )
                )
                METRICS.record_error(category, context_overflow=overflow)
                raise RuntimeError(
                    "model request failed: "
                    + ("context_overflow" if overflow else category)
                )
            payload = json.loads(raw)
            message = payload["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            usage = payload.get("usage") or {}
            with METRICS._lock:
                METRICS.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                METRICS.completion_tokens += int(usage.get("completion_tokens") or 0)
            parsed_calls = [
                SimpleNamespace(
                    id=call.get("id", ""),
                    function=SimpleNamespace(
                        name=call.get("function", {}).get("name", ""),
                        arguments=call.get("function", {}).get("arguments", ""),
                    ),
                )
                for call in tool_calls
            ]
            actions = parse_toolcall_actions(
                parsed_calls,
                format_error_template=self.format_error_template,
                template_kwargs={
                    "finish_reason": payload["choices"][0].get("finish_reason")
                },
            )
            # Raw model output is held only in the current conversation memory.
            return {**message, "extra": {"actions": actions}}
        except OSError:
            if self.stop_event.is_set():
                METRICS.record_error("cancelled")
                raise RuntimeError("agent stopped") from None
            METRICS.record_error("transport")
            raise RuntimeError("model request failed: transport") from None
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            METRICS.record_error("invalid_response")
            raise RuntimeError("model response invalid") from None
        finally:
            with self._connection_lock:
                if self._active_connection is connection:
                    self._active_connection = None
            connection.close()

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs):
        return {"model_name": self.model_name}

    def serialize(self):
        return {"info": {"model": {"name": self.model_name}}}


class PierEnvironmentBridge:
    """Adapt mini-swe-agent's synchronous command calls to Pier's async exec."""

    def __init__(
        self, environment, loop: asyncio.AbstractEventLoop, *, command_timeout: int = 30
    ):
        self._environment = environment
        self._loop = loop
        self._command_timeout = command_timeout
        self._stop_event = threading.Event()
        self._pending: set[concurrent.futures.Future] = set()
        self._pending_lock = threading.Lock()
        self._abort_callback = None
        self.config = type("EnvironmentConfig", (), {})()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def stop(self) -> None:
        self._stop_event.set()
        if self._abort_callback is not None:
            self._abort_callback()
        with self._pending_lock:
            pending = tuple(self._pending)
        for future in pending:
            future.cancel()

    def set_abort_callback(self, callback) -> None:
        self._abort_callback = callback

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        if self._stop_event.is_set():
            raise RuntimeError("agent stopped")
        command = action.get("command")
        if not isinstance(command, str):
            return {"returncode": 2, "output": "command rejected"}
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                command, cwd=cwd or None, timeout_sec=self._command_timeout
            ),
            self._loop,
        )
        with self._pending_lock:
            self._pending.add(future)
        try:
            result = future.result(timeout=self._command_timeout + 5)
        except concurrent.futures.TimeoutError:
            future.cancel()
            METRICS.record_error("environment_timeout")
            raise RuntimeError("environment command timed out") from None
        except Exception:  # noqa: BLE001 - suppress command exception details at this boundary
            METRICS.record_error("environment")
            raise RuntimeError("environment command failed") from None
        finally:
            with self._pending_lock:
                self._pending.discard(future)
        stdout = result.stdout or ""
        lines = stdout.lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and result.return_code == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        output = "\n".join(part for part in (stdout, result.stderr) if part)
        return {"returncode": result.return_code, "output": output}

    def get_template_vars(self, **kwargs):
        return {"system": "Linux", "release": "", "version": "", "machine": ""}

    def serialize(self):
        return {}


class MemoryOnlyAgent(DefaultAgent):
    """Keep the upstream decision loop while eliminating all trajectory writes."""

    def save(self, path, *extra_dicts):
        # DefaultAgent calls save in every step's finally block. Do not serialize.
        return {}

    def handle_uncaught_exception(self, error):
        # DefaultAgent's implementation stores exception text and traceback.
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content="run_error",
                extra={"exit_status": "run_error", "submission": ""},
            )
        )


def _agent_templates() -> dict[str, Any]:
    import yaml

    path = Path(package_dir) / "config" / "mini.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class StarkMiniSweAgent(BaseAgent):
    """Pier entry point; conversation data and credentials remain host-side."""

    def __init__(
        self,
        *args,
        max_tokens=MODEL_MAX_TOKENS,
        temperature=MODEL_TEMPERATURE,
        step_limit=DEFAULT_STEP_LIMIT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if int(max_tokens) <= 0 or int(step_limit) <= 0:
            raise ValueError("invalid agent limits")
        if float(temperature) != MODEL_TEMPERATURE:
            raise ValueError("unsupported temperature")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.step_limit = int(step_limit)

    @staticmethod
    def name() -> str:
        return "maxxbench-mini-swe-agent-host"

    def version(self) -> str:
        return "2.4.6"

    async def setup(self, environment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        metrics_before = METRICS.snapshot()
        loop = asyncio.get_running_loop()
        bridge = PierEnvironmentBridge(environment, loop)
        settings = _agent_templates()
        prompt_config = settings["agent"]
        model_config = settings["model"]
        agent_config = AgentConfig(
            system_template=prompt_config["system_template"],
            instance_template=prompt_config["instance_template"],
            step_limit=self.step_limit,
            cost_limit=0.0,
            wall_time_limit_seconds=0,
            max_consecutive_format_errors=3,
            output_path=None,
        )
        model = LocalOpenAIModel(
            _MODEL_NAME or self.model_name or "stark-local",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
            stop_event=bridge.stop_event,
        )
        bridge.set_abort_callback(model.abort)
        agent = MemoryOnlyAgent(model, bridge, config_class=lambda **kw: agent_config)
        # mini-swe-agent's default logger logs complete messages at DEBUG.
        agent.logger.disabled = True
        started = time.monotonic()

        def execute_loop():
            return agent.run(task=instruction)

        worker = asyncio.create_task(asyncio.to_thread(execute_loop))
        try:
            result = await worker
            status = str(result.get("exit_status", "unknown"))
            submission = result.get("submission", "")
            if not isinstance(submission, str):
                submission = ""
        except asyncio.CancelledError:
            bridge.stop()
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=5)
            except TimeoutError:
                pass
            except Exception:  # noqa: BLE001, S110 - bounded cleanup intentionally drops details
                pass
            raise
        except Exception:  # noqa: BLE001 - suppress all model/task exception details
            # No exception value, traceback, prompts, or output enters Pier logs.
            METRICS.record_error("agent")
            status = "run_error"
            submission = ""

        metrics_after = METRICS.snapshot()
        context.n_agent_steps = agent.n_calls
        context.n_input_tokens = (
            metrics_after["prompt_tokens"] - metrics_before["prompt_tokens"]
        )
        context.n_output_tokens = (
            metrics_after["completion_tokens"] - metrics_before["completion_tokens"]
        )
        context.cost_usd = None
        context.metadata = {
            "agent_status": status,
            "model_calls": metrics_after["calls"] - metrics_before["calls"],
            "duration_seconds": round(time.monotonic() - started, 3),
            "http_errors": {
                category: count
                - metrics_before["http_error_categories"].get(category, 0)
                for category, count in metrics_after["http_error_categories"].items()
                if count > metrics_before["http_error_categories"].get(category, 0)
            },
            "context_overflow_errors": (
                metrics_after["context_overflow_errors"]
                - metrics_before["context_overflow_errors"]
            ),
        }
        # Pier persists this field as a result artifact; keep it empty and let
        # task.toml's existing collection hook gather the patch from the sandbox.
        context.metadata["submission_present"] = bool(submission)
