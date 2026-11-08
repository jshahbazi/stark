"""Offline privacy and adapter tests; no model or Docker is started."""

import asyncio
import json
import threading
import unittest
from unittest.mock import patch

from maxxbench_agent import (
    METRICS,
    AggregateMetrics,
    LocalOpenAIModel,
    MemoryOnlyAgent,
    PierEnvironmentBridge,
    _agent_templates,
    clear_connection,
    configure_connection,
)
from minisweagent.agents.default import AgentConfig
from minisweagent.exceptions import FormatError


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
        self.assertEqual(connection.timeout, 600.0)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["seed"], 0)
        self.assertEqual(body["tools"][0]["function"]["name"], "bash")
        self.assertEqual(
            response["extra"]["actions"],
            [{"command": "printf ok", "tool_call_id": "call-1"}],
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
            self.assertEqual(result, {"returncode": 0, "output": "ok"})
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
