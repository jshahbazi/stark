"""Offline privacy and adapter tests; no model or Docker is started."""

import asyncio
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from maxxbench_agent import (
    METRICS,
    AggregateMetrics,
    LocalOpenAIModel,
    MemoryOnlyAgent,
    PierEnvironmentBridge,
    StarkMiniSweAgent,
    _agent_templates,
    clear_connection,
    configure_connection,
)
from minisweagent.agents.default import AgentConfig
from minisweagent.exceptions import FormatError
from pier.models.agent.context import AgentContext


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = json.dumps(payload).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, _limit=-1):
        return self.payload


class AdapterTests(unittest.TestCase):
    def tearDown(self):
        clear_connection()

    def test_model_request_uses_in_memory_auth_and_explicit_sampling(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"printf ok"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }

        class FakeConnection:
            def __init__(self, host, port, timeout, stop_event, connection_lock):
                self.host = host
                self.port = port
                self.timeout = timeout

            def request(self, method, target, body, headers):
                self.request_data = (method, target, body, headers)

            def getresponse(self):
                return FakeResponse(payload)

            def close(self):
                pass

        fake_connection = FakeConnection("127.0.0.1", 9000, 600, None, None)
        with patch(
            "maxxbench_agent._CancellableHTTPConnection",
            return_value=fake_connection,
        ) as connection_factory:
            model_config = _agent_templates()["model"]
            response = LocalOpenAIModel(
                "stark-test",
                observation_template=model_config["observation_template"],
                format_error_template=model_config["format_error_template"],
            ).query([{"role": "user", "content": "private prompt"}])
        connection = fake_connection
        self.assertEqual(
            connection_factory.call_args.kwargs["stop_event"].is_set(), False
        )
        method, target, request_body, headers = connection.request_data
        body = json.loads(request_body)
        self.assertEqual((connection.host, connection.port), ("127.0.0.1", 9000))
        self.assertEqual(method, "POST")
        self.assertEqual(target, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer ephemeral-key")
        self.assertEqual(connection_factory.call_args.kwargs["timeout"], 1810.0)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["seed"], 0)
        self.assertEqual(body["tools"][0]["function"]["name"], "bash")
        self.assertEqual(
            response["extra"]["actions"],
            [{"command": "printf ok", "tool_call_id": "call-1"}],
        )

    def test_socket_timeout_is_a_safe_client_timeout_category(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")

        class TimeoutConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                raise TimeoutError("private socket diagnostic")

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
        )
        before = METRICS.snapshot()
        with (
            patch("maxxbench_agent._CancellableHTTPConnection", TimeoutConnection),
            self.assertRaisesRegex(RuntimeError, "client_timeout") as raised,
        ):
            model.query([{"role": "user", "content": "private test input"}])
        after = METRICS.snapshot()
        self.assertEqual(str(raised.exception), "model request failed: client_timeout")
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertEqual(
            after["http_error_categories"].get("client_timeout", 0)
            - before["http_error_categories"].get("client_timeout", 0),
            1,
        )
        self.assertEqual(
            after["http_error_categories"].get("transport", 0)
            - before["http_error_categories"].get("transport", 0),
            0,
        )

    def test_generic_socket_error_remains_transport(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")

        class TransportConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                raise OSError("private socket diagnostic")

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
        )
        before = METRICS.snapshot()
        with (
            patch("maxxbench_agent._CancellableHTTPConnection", TransportConnection),
            self.assertRaisesRegex(RuntimeError, "transport") as raised,
        ):
            model.query([{"role": "user", "content": "private test input"}])
        after = METRICS.snapshot()
        self.assertEqual(str(raised.exception), "model request failed: transport")
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertEqual(
            after["http_error_categories"].get("transport", 0)
            - before["http_error_categories"].get("transport", 0),
            1,
        )
        self.assertEqual(
            after["http_error_categories"].get("client_timeout", 0)
            - before["http_error_categories"].get("client_timeout", 0),
            0,
        )

    def test_timeout_after_abort_is_classified_as_cancelled(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        stop_event = threading.Event()

        class CancelledTimeoutConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                stop_event.set()
                raise TimeoutError("private cancellation diagnostic")

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
            stop_event=stop_event,
        )
        before = METRICS.snapshot()
        with (
            patch(
                "maxxbench_agent._CancellableHTTPConnection", CancelledTimeoutConnection
            ),
            self.assertRaisesRegex(RuntimeError, "agent stopped") as raised,
        ):
            model.query([{"role": "user", "content": "private test input"}])
        after = METRICS.snapshot()
        self.assertEqual(str(raised.exception), "agent stopped")
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertEqual(
            after["http_error_categories"].get("cancelled", 0)
            - before["http_error_categories"].get("cancelled", 0),
            1,
        )
        self.assertEqual(
            after["http_error_categories"].get("client_timeout", 0)
            - before["http_error_categories"].get("client_timeout", 0),
            0,
        )

    def test_connection_rejects_non_loopback_and_ambiguous_urls(self):
        for endpoint in (
            "https://127.0.0.1/v1",
            "http://example.com/v1",
            "http://user:pass@127.0.0.1/v1",
            "http://127.0.0.1/v1#fragment",
            "http://127.0.0.1/v1?forward=elsewhere",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                configure_connection(endpoint, "ephemeral-key", "stark-test")

    def test_stark_context_error_code_increments_overflow_metric(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        error_payload = {
            "error": {
                "message": "Conversation exceeds the context budget. Shorten messages or the reply limit.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }

        class ErrorConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(error_payload, status=400)

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
        )
        before = METRICS.snapshot()
        with (
            patch("maxxbench_agent._CancellableHTTPConnection", ErrorConnection),
            self.assertRaisesRegex(RuntimeError, "context_overflow"),
        ):
            model.query([{"role": "user", "content": "private"}])
        after = METRICS.snapshot()
        self.assertEqual(
            after["context_overflow_errors"] - before["context_overflow_errors"], 1
        )
        self.assertEqual(
            after["http_error_categories"].get("request_rejected", 0)
            - before["http_error_categories"].get("request_rejected", 0),
            1,
        )

    def test_successful_length_finish_reason_is_not_http_overflow(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        completion = {
            "choices": [
                {
                    "message": {"role": "assistant", "tool_calls": []},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }

        class LengthConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(completion, status=200)

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
        )
        before = METRICS.snapshot()
        with (
            patch("maxxbench_agent._CancellableHTTPConnection", LengthConnection),
            self.assertRaises(FormatError),
        ):
            model.query([{"role": "user", "content": "private"}])
        after = METRICS.snapshot()
        self.assertEqual(
            after["context_overflow_errors"], before["context_overflow_errors"]
        )
        self.assertEqual(
            after["http_error_categories"], before["http_error_categories"]
        )

    def test_model_abort_closes_inflight_request(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        started = threading.Event()
        stopped = threading.Event()
        stop_event = threading.Event()

        class BlockingConnection:
            sock = None

            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                started.set()
                stopped.wait(timeout=2)
                raise OSError("closed")

            def close(self):
                stopped.set()

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
            stop_event=stop_event,
        )
        model_errors = []

        def run_query():
            try:
                model.query([{"role": "user", "content": "private"}])
            except RuntimeError as error:
                model_errors.append(str(error))

        with patch("maxxbench_agent._CancellableHTTPConnection", BlockingConnection):
            worker = threading.Thread(target=run_query)
            worker.start()
            self.assertTrue(started.wait(timeout=1))
            stop_event.set()
            model.abort()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(model_errors, ["agent stopped"])

    def test_bridge_executes_from_worker_thread_and_returns_output(self):
        class Environment:
            async def exec(self, command, cwd=None, timeout_sec=None):
                self.seen = (command, cwd, timeout_sec)
                return type(
                    "Result", (), {"stdout": "ok", "stderr": "", "return_code": 0}
                )()

        async def exercise():
            environment = Environment()
            bridge = PierEnvironmentBridge(environment, asyncio.get_running_loop())
            result = await asyncio.to_thread(bridge.execute, {"command": "pwd"}, "/app")
            self.assertEqual(
                result,
                {"returncode": 0, "output": "ok", "exception_info": ""},
            )
            self.assertEqual(environment.seen, ("pwd", "/app", 30))

        asyncio.run(exercise())

    def test_bridge_stop_cancels_inflight_exec(self):
        class Environment:
            def __init__(self):
                self.started = asyncio.Event()

            async def exec(self, command, cwd=None, timeout_sec=None):
                self.started.set()
                await asyncio.sleep(30)

        async def exercise():
            environment = Environment()
            bridge = PierEnvironmentBridge(environment, asyncio.get_running_loop())
            worker = asyncio.create_task(
                asyncio.to_thread(bridge.execute, {"command": "sleep 30"})
            )
            await environment.started.wait()
            bridge.stop()
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(worker, timeout=2)

        asyncio.run(exercise())

    def test_default_agent_loop_stops_on_submission_marker(self):
        class Model:
            config = type("Config", (), {})()

            def __init__(self):
                self.calls = 0

            def query(self, messages):
                self.calls += 1
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "finish",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}',
                            },
                        }
                    ],
                    "extra": {
                        "actions": [
                            {
                                "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                                "tool_call_id": "finish",
                            }
                        ]
                    },
                }

            def format_message(self, **kwargs):
                return kwargs

            def get_template_vars(self, **kwargs):
                return {}

            def serialize(self):
                return {}

        class Environment:
            async def exec(self, command, cwd=None, timeout_sec=None):
                return type(
                    "Result",
                    (),
                    {
                        "stdout": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
                        "stderr": "",
                        "return_code": 0,
                    },
                )()

        async def exercise():
            loop = asyncio.get_running_loop()
            bridge = PierEnvironmentBridge(Environment(), loop)
            model = Model()
            config = AgentConfig(system_template="system", instance_template="{{task}}")
            agent = MemoryOnlyAgent(model, bridge, config_class=lambda **_: config)
            result = await asyncio.to_thread(agent.run, task="private task")
            self.assertEqual(result["exit_status"], "Submitted")
            self.assertEqual(model.calls, 1)
            self.assertEqual(agent.messages[-1]["extra"]["submission"], "")

        asyncio.run(exercise())

    def test_full_adapter_runs_model_observation_and_submission(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        payloads = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"sleep 30"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 40},
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"echo check"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1260, "completion_tokens": 30},
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-3",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1320, "completion_tokens": 20},
            },
        ]
        executed = []
        exception_types = []

        class FakeConnection:
            def __init__(self, *args, **kwargs):
                self.payload = payloads.pop(0)

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(self.payload)

            def close(self):
                pass

        class FakeEnvironment:
            async def exec(self, command, cwd=None, timeout_sec=None):
                executed.append(command)
                if command == "sleep 30":
                    raise RuntimeError("Command timed out after 30 seconds")
                stdout = (
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
                    else "check passed\n"
                )
                return type(
                    "Result",
                    (),
                    {
                        "stdout": stdout,
                        "stderr": "",
                        "return_code": 0,
                    },
                )()

        original_handler = MemoryOnlyAgent.handle_uncaught_exception

        def record_safe_exception_type(agent, error):
            exception_types.append(type(error).__name__)
            return original_handler(agent, error)

        async def exercise():
            agent = StarkMiniSweAgent(
                logs_dir=Path("/tmp"),
                model_name="stark-test",
                max_tokens=4096,
                temperature=0.0,
                step_limit=250,
            )
            context = AgentContext()
            with (
                patch("maxxbench_agent._CancellableHTTPConnection", FakeConnection),
                patch.object(
                    MemoryOnlyAgent,
                    "handle_uncaught_exception",
                    record_safe_exception_type,
                ),
            ):
                await agent.run("synthetic task metadata", FakeEnvironment(), context)
            return context

        context = asyncio.run(exercise())
        self.assertEqual(exception_types, [])
        self.assertEqual(context.metadata["agent_status"], "Submitted")
        self.assertEqual(context.metadata["model_calls"], 3)
        self.assertEqual(context.metadata["http_errors"], {"environment_timeout": 1})
        self.assertEqual(context.n_input_tokens, 3780)
        self.assertEqual(context.n_output_tokens, 90)
        self.assertEqual(
            executed,
            ["sleep 30", "echo check", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        )
        self.assertEqual(payloads, [])

    def test_context_overflow_is_task_outcome_not_harness_exception(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        payload = {
            "error": {
                "message": "Conversation exceeds the context budget.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }

        class ErrorConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(payload, status=400)

            def close(self):
                pass

        class FakeEnvironment:
            async def exec(self, *args, **kwargs):
                raise AssertionError("budget rejection must occur before commands")

        async def exercise():
            agent = StarkMiniSweAgent(logs_dir=Path("/tmp"), model_name="stark-test")
            context = AgentContext()
            with patch("maxxbench_agent._CancellableHTTPConnection", ErrorConnection):
                await agent.run("synthetic task metadata", FakeEnvironment(), context)
            return context

        context = asyncio.run(exercise())
        self.assertEqual(context.metadata["agent_status"], "context_overflow")
        self.assertEqual(context.metadata["context_overflow_errors"], 1)
        self.assertEqual(context.metadata["http_errors"], {"request_rejected": 1})
        self.assertEqual(
            context.metadata["api_error_codes"], {"context_length_exceeded": 1}
        )
        self.assertIsNone(context.metadata["adapter_error_type"])

    def test_request_deadline_is_a_task_outcome_but_generation_failure_is_infra(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")

        class ResponseConnection:
            status = 0
            payload = None

            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(self.payload, status=self.status)

            def close(self):
                pass

        class NoCommandEnvironment:
            async def exec(self, *args, **kwargs):
                raise AssertionError("HTTP error should stop before tool execution")

        async def exercise(status, code):
            ResponseConnection.status = status
            ResponseConnection.payload = {
                "error": {
                    "message": "A private server diagnostic.",
                    "type": "server_error",
                    "code": code,
                }
            }
            agent = StarkMiniSweAgent(logs_dir=Path("/tmp"), model_name="stark-test")
            context = AgentContext()
            with patch(
                "maxxbench_agent._CancellableHTTPConnection", ResponseConnection
            ):
                if status == 504:
                    try:
                        await agent.run(
                            "synthetic task metadata", NoCommandEnvironment(), context
                        )
                    except RuntimeError:
                        pass
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "^maxxbench_adapter_failed$"
                    ):
                        await agent.run(
                            "synthetic task metadata", NoCommandEnvironment(), context
                        )
            return context

        request_deadline = asyncio.run(exercise(504, "request_timeout"))
        self.assertEqual(
            request_deadline.metadata["agent_status"],
            "request_timeout",
            request_deadline.metadata,
        )
        self.assertEqual(request_deadline.metadata["http_errors"], {"server": 1})
        self.assertEqual(
            request_deadline.metadata["api_error_codes"], {"request_timeout": 1}
        )
        self.assertIsNone(request_deadline.metadata["adapter_error_type"])

        generation_failure = asyncio.run(exercise(503, "generation_failed"))
        self.assertEqual(generation_failure.metadata["agent_status"], "adapter_error")
        self.assertEqual(
            generation_failure.metadata["http_errors"], {"server": 1, "agent": 1}
        )
        self.assertEqual(
            generation_failure.metadata["api_error_codes"], {"generation_failed": 1}
        )
        self.assertEqual(
            generation_failure.metadata["adapter_error_type"], "RuntimeError"
        )

    def test_model_tool_format_error_reenters_stock_format_loop(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        responses = [
            (
                502,
                {
                    "error": {
                        "message": "The model produced an incomplete, invalid or unavailable tool call.",
                        "type": "server_error",
                        "code": "invalid_tool_call",
                    }
                },
            ),
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1300, "completion_tokens": 25},
                },
            ),
        ]

        class FakeConnection:
            def __init__(self, *args, **kwargs):
                self.status, self.payload = responses.pop(0)

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(self.payload, status=self.status)

            def close(self):
                pass

        class FakeEnvironment:
            async def exec(self, command, cwd=None, timeout_sec=None):
                return type(
                    "Result",
                    (),
                    {
                        "stdout": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
                        "stderr": "",
                        "return_code": 0,
                    },
                )()

        async def exercise():
            agent = StarkMiniSweAgent(logs_dir=Path("/tmp"), model_name="stark-test")
            context = AgentContext()
            with patch("maxxbench_agent._CancellableHTTPConnection", FakeConnection):
                await agent.run("synthetic task metadata", FakeEnvironment(), context)
            return context

        context = asyncio.run(exercise())
        self.assertEqual(context.metadata["agent_status"], "Submitted")
        self.assertEqual(context.metadata["model_calls"], 2)
        self.assertEqual(context.metadata["api_error_codes"], {"invalid_tool_call": 1})
        self.assertEqual(responses, [])

    def test_authentication_api_code_remains_fatal(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        payload = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error",
                "code": "invalid_api_key",
            }
        }

        class AuthConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(payload, status=401)

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
        )
        before = METRICS.snapshot()
        with (
            patch("maxxbench_agent._CancellableHTTPConnection", AuthConnection),
            self.assertRaisesRegex(RuntimeError, "authentication"),
        ):
            model.query([{"role": "user", "content": "private"}])
        after = METRICS.snapshot()
        self.assertEqual(
            after["api_error_codes"].get("invalid_api_key", 0)
            - before["api_error_codes"].get("invalid_api_key", 0),
            1,
        )

    def test_tool_choice_failure_is_a_format_error(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        payload = {
            "error": {
                "message": "The model did not satisfy the requested tool choice.",
                "type": "server_error",
                "code": "tool_choice_not_satisfied",
            }
        }

        class ToolChoiceConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(payload, status=502)

            def close(self):
                pass

        model_config = _agent_templates()["model"]
        model = LocalOpenAIModel(
            "stark-test",
            observation_template=model_config["observation_template"],
            format_error_template=model_config["format_error_template"],
        )
        before = METRICS.snapshot()
        with (
            patch(
                "maxxbench_agent._CancellableHTTPConnection",
                ToolChoiceConnection,
            ),
            self.assertRaises(FormatError),
        ):
            model.query([{"role": "user", "content": "private"}])
        self.assertEqual(
            METRICS.snapshot()["api_error_codes"].get("tool_choice_not_satisfied", 0)
            - before["api_error_codes"].get("tool_choice_not_satisfied", 0),
            1,
        )

    def test_unexpected_adapter_error_fails_trial_after_safe_metrics(self):
        configure_connection("http://127.0.0.1:9000/v1", "ephemeral-key", "stark-test")
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo check"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }

        class FakeConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return FakeResponse(payload)

            def close(self):
                pass

        class BrokenEnvironment:
            async def exec(self, *args, **kwargs):
                raise RuntimeError("private command-specific exception text")

        async def exercise():
            agent = StarkMiniSweAgent(logs_dir=Path("/tmp"), model_name="stark-test")
            context = AgentContext()
            with (
                patch("maxxbench_agent._CancellableHTTPConnection", FakeConnection),
                self.assertRaisesRegex(RuntimeError, "^maxxbench_adapter_failed$"),
            ):
                await agent.run("synthetic task metadata", BrokenEnvironment(), context)
            return context

        context = asyncio.run(exercise())
        self.assertEqual(context.metadata["agent_status"], "adapter_error")
        self.assertEqual(context.metadata["adapter_error_type"], "RuntimeError")
        self.assertEqual(
            context.metadata["http_errors"], {"environment": 1, "agent": 1}
        )
        self.assertEqual(context.n_agent_steps, 1)

    def test_agent_save_never_serializes_conversation(self):
        class Model:
            pass

        agent = MemoryOnlyAgent(
            Model(),
            object(),
            config_class=lambda **_: type("Config", (), {})(),
        )
        agent.messages = [{"role": "user", "content": "private"}]
        self.assertEqual(agent.save("should-not-write.json"), {})

    def test_error_metric_retains_only_categories(self):
        metrics = AggregateMetrics()
        metrics.record_error("server", context_overflow=True)
        self.assertEqual(metrics.http_error_categories, {"server": 1})
        self.assertEqual(metrics.context_overflow_errors, 1)


if __name__ == "__main__":
    unittest.main()
