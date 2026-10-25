import http.client
import io
import json
import socket
import struct
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from stark_convert.local_api import APIError, LocalAPI


class LocalAPITest(unittest.TestCase):
    def test_restart_keeps_old_requests_cancelled(self):
        monitor_entered = threading.Event()
        release_monitor = threading.Event()
        completion_entered = threading.Event()
        completion_cancelled = threading.Event()

        class DelayedMonitorAPI(LocalAPI):
            def _handler_class(self, session_key, session_closed):
                handler = super()._handler_class(session_key, session_closed)
                original = handler._monitor_peer

                def delayed_monitor(request, cancelled):
                    monitor_entered.set()
                    release_monitor.wait(5)
                    original(request, cancelled)

                handler._monitor_peer = delayed_monitor
                return handler

        def complete(payload, on_delta, cancelled):
            completion_entered.set()
            if cancelled.wait(5):
                completion_cancelled.set()
            raise APIError(499, "Cancelled", "cancelled")

        api = DelayedMonitorAPI(lambda: [], complete, port=0, cancellable=True)
        info = api.start()
        port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("POST", "/v1/chat/completions", body=b"{}", headers={
                "Authorization": "Bearer " + info["api_key"],
                "Content-Type": "application/json",
            })
            self.assertTrue(monitor_entered.wait(2))
            self.assertTrue(completion_entered.wait(2))
            api.close()
            api.start()
            release_monitor.set()
            self.assertTrue(completion_cancelled.wait(2))
            self.assertEqual(connection.getresponse().status, 499)
        finally:
            release_monitor.set()
            connection.close()
            api.close()

    def test_signed_content_length_is_rejected(self):
        status, _, _ = self.request(
            "POST", "/chat/completions", {}, key=self.info["api_key"], headers={"Content-Length": "+2"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(self.calls, [])

    def test_truncated_body_is_rejected_before_completion(self):
        called = []
        api = LocalAPI(lambda: [], lambda payload: called.append(True) or {"choices": []}, port=0)
        info = api.start()
        sock = None
        try:
            port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
            body = b'{"messages":[{"role":"user","content":"synthetic fixture"}]}'
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Authorization: Bearer " + info["api_key"].encode() + b"\r\n"
                b"Content-Type: application/json\r\nContent-Length: " + str(len(body) + 10).encode()
                + b"\r\n\r\n" + body
            )
            sock.shutdown(socket.SHUT_WR)
            response = b""
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                response += data
            self.assertIn(b"HTTP/1.1 400", response)
            self.assertEqual(called, [])
        finally:
            if sock is not None:
                sock.close()
            api.close()

    def setUp(self):
        self.calls = []
        self.api = LocalAPI(lambda: [{"id": "stark", "object": "model"}], self.complete)
        self.info = self.api.start()

    def tearDown(self):
        self.api.close()

    def complete(self, payload, on_delta=None):
        self.calls.append(payload)
        if on_delta is not None:
            on_delta("hello")
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "stark",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        }

    def request(self, method, path, body=None, key=None, headers=None):
        headers = dict(headers or {})
        if key is not None:
            headers["Authorization"] = "Bearer " + key
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.info["base_url"] + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def test_auth_and_models(self):
        status, _, _ = self.request("GET", "/models")
        self.assertEqual(status, 401)
        status, _, body = self.request("GET", "/models", key=self.info["api_key"])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "stark")

    def test_health_capabilities_and_model_metadata(self):
        api = LocalAPI(
            lambda: [{"id": "stark", "object": "model", "owned_by": "test"}],
            self.complete,
            status=lambda: {"status": "ready", "streaming": True},
            port=0,
        )
        info = api.start()
        try:
            for path in ("/health", "/v1/health"):
                status, _, body = self.request_to_path(info, path, "GET")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["status"], "ready")
            status, _, body = self.request_to_path(info, "/v1/capabilities", "GET")
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(body)["streaming"])
            status, _, body = self.request_to_path(info, "/v1/models/stark", "GET")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["owned_by"], "test")
        finally:
            api.close()

    def test_readable_model_detail_ids(self):
        for model_id, paths in (
            ("Qwen/Qwen3.8-27B", ["Qwen/Qwen3.8-27B", "Qwen%2FQwen3.8-27B"]),
            ("Local model's name", ["Local%20model%27s%20name"]),
            ("literal%2Fname", ["literal%252Fname"]),
        ):
            api = LocalAPI(lambda model_id=model_id: [{"id": model_id, "object": "model"}], self.complete, port=0)
            info = api.start()
            try:
                for path in paths:
                    status, _, body = self.request_to_path(info, "/v1/models/" + path, "GET")
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["id"], model_id)
            finally:
                api.close()

    def request_to_path(self, info, path, method, raw=None, headers=None):
        req = urllib.request.Request(
            info["base_url"].replace("/v1", "") + path,
            data=raw,
            headers={
                "Authorization": "Bearer " + info["api_key"],
                "Content-Type": "application/json",
                **(headers or {}),
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def test_origin_and_host_are_rejected(self):
        self.assertEqual(
            self.request("GET", "/models", key=self.info["api_key"], headers={"Origin": "http://evil"})[0], 403
        )
        self.assertEqual(
            self.request("GET", "/models", key=self.info["api_key"], headers={"Host": "evil.example"})[0], 403
        )

    def test_completion_and_buffered_stream(self):
        status, _, body = self.request("POST", "/chat/completions", key=self.info["api_key"], body={"messages": []})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "hello")
        status, headers, body = self.request(
            "POST", "/chat/completions", key=self.info["api_key"], body={"messages": [], "stream": True}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertIn(b'"role":"assistant"', body)
        self.assertIn(b"data: [DONE]", body)

    def test_stream_content_is_sent_before_completion_returns(self):
        content_written = threading.Event()
        release_completion = threading.Event()

        def complete(payload, on_delta=None):
            self.assertIsNotNone(on_delta)
            on_delta("first")
            content_written.set()
            self.assertTrue(release_completion.wait(2))
            return {
                "id": "ignored-after-stream-start",
                "model": "stark",
                "choices": [{"message": {"content": "first"}, "finish_reason": "stop"}],
            }

        api = LocalAPI(lambda: [], complete, port=0)
        info = api.start()
        try:
            port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True}),
                headers={"Authorization": "Bearer " + info["api_key"], "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(content_written.wait(2))
            line = response.fp.readline()
            self.assertIn(b'"role":"assistant"', line)
            self.assertEqual(response.fp.readline(), b"\n")
            line = response.fp.readline()
            self.assertIn(b'"content":"first"', line)
            release_completion.set()
            body = response.read()
            self.assertIn(b"[DONE]", body)
            conn.close()
        finally:
            release_completion.set()
            api.close()

    def test_stream_uses_public_id_when_request_uses_legacy_alias(self):
        public_id = "Qwen/Qwen3.8-27B"
        calls = []

        def complete(payload, on_delta=None):
            calls.append(payload)
            on_delta("fixture")
            return {"model": public_id, "choices": [{"message": {"content": "fixture"}, "finish_reason": "stop"}]}

        api = LocalAPI(lambda: [{"id": public_id}], complete, port=0)
        info = api.start()
        try:
            status, _, body = self.request_to(info, {"model": "legacy-private-id", "stream": True})
            self.assertEqual(status, 200)
            events = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: {")]
            self.assertTrue(events)
            self.assertTrue(all(event["model"] == public_id for event in events))
            self.assertEqual(calls[0]["model"], "legacy-private-id")
        finally:
            api.close()

    def test_empty_stream_and_usage_keep_one_stream_metadata(self):
        def complete(payload, on_delta=None):
            return {
                "id": "returned-id",
                "created": 1,
                "model": "returned-model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            }

        api = LocalAPI(lambda: [], complete, port=0)
        info = api.start()
        try:
            status, headers, body = self.request_to(info, {"stream": True, "stream_options": {"include_usage": True}})
            self.assertEqual(status, 200)
            self.assertEqual(headers.get_content_type(), "text/event-stream")
            events = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: {")]
            self.assertGreaterEqual(len(events), 3)
            self.assertEqual(
                {(event["id"], event["created"], event["model"]) for event in events},
                {(events[0]["id"], events[0]["created"], events[0]["model"])},
            )
            self.assertEqual(events[-1]["choices"], [])
            self.assertIn(b"data: [DONE]", body)
        finally:
            api.close()

    def request_to(self, info, body):
        req = urllib.request.Request(
            info["base_url"] + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + info["api_key"], "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status, response.headers, response.read()

    def test_stream_errors_after_content_have_no_success_terminator(self):
        for failure in (APIError(422, "bad request", "bad"), RuntimeError("secret")):

            def complete(payload, on_delta=None, failure=failure):
                on_delta("before-error")
                raise failure

            api = LocalAPI(lambda: [], complete, port=0)
            info = api.start()
            try:
                status, _, body = self.request_to(info, {"stream": True})
                self.assertEqual(status, 200)
                self.assertIn(b"before-error", body)
                self.assertIn(b'"error"', body)
                self.assertNotIn(b"[DONE]", body)
                self.assertNotIn(b'"finish_reason":"stop"', body)
            finally:
                api.close()

    def test_stream_client_reset_propagates_to_callback(self):
        content_written = threading.Event()
        client_gone = threading.Event()
        write_failed = threading.Event()

        def complete(payload, on_delta=None):
            on_delta("first")
            content_written.set()
            self.assertTrue(client_gone.wait(2))
            try:
                for index in range(16):
                    on_delta("after-reset-" + ("x" * (1024 * 1024)) + str(index))
            except Exception:
                write_failed.set()
                raise
            return {"choices": [{"message": {"content": "ignored"}, "finish_reason": "stop"}]}

        api = LocalAPI(lambda: [], complete, port=0)
        info = api.start()
        sock = None
        try:
            port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
            body = b'{"stream":true}'
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer " + info["api_key"].encode() + b"\r\n"
                b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            self.assertTrue(content_written.wait(2))
            received = b""
            while b'"content":"first"' not in received:
                chunk = sock.recv(4096)
                self.assertTrue(chunk)
                received += chunk
                self.assertLess(len(received), 64 * 1024)
            self.assertIn(b"HTTP/1.1 200", received)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            sock.close()
            sock = None
            client_gone.set()
            self.assertTrue(write_failed.wait(2))
        finally:
            client_gone.set()
            if sock is not None:
                sock.close()
            api.close()

    def test_disconnect_before_first_delta_sets_cancellation(self):
        cancelled = threading.Event()
        entered = threading.Event()

        def complete(payload, on_delta, request_cancelled):
            entered.set()
            self.assertIsNotNone(on_delta)
            self.assertTrue(request_cancelled.wait(2))
            cancelled.set()
            return {"choices": [{"finish_reason": "stop"}]}

        api = LocalAPI(lambda: [], complete, port=0, cancellable=True)
        info = api.start()
        try:
            self.disconnect_request(info, {"stream": True})
            self.assertTrue(entered.wait(2))
            self.assertTrue(cancelled.wait(2))
        finally:
            api.close()

    def test_disconnect_before_nonstream_completion_sets_cancellation(self):
        cancelled = threading.Event()
        entered = threading.Event()

        def complete(payload, on_delta, request_cancelled):
            entered.set()
            self.assertIsNone(on_delta)
            self.assertTrue(request_cancelled.wait(2))
            cancelled.set()
            return {"choices": [{"finish_reason": "stop"}]}

        api = LocalAPI(lambda: [], complete, port=0, cancellable=True)
        info = api.start()
        try:
            self.disconnect_request(info, {})
            self.assertTrue(entered.wait(2))
            self.assertTrue(cancelled.wait(2))
        finally:
            api.close()

    def disconnect_request(self, info, body):
        port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
        payload = json.dumps(body).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer "
            + info["api_key"].encode()
            + b"\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode()
            + b"\r\n\r\n"
            + payload
        )
        sock.close()

    def test_deadline_sets_cancellation(self):
        cancelled = threading.Event()

        def complete(payload, on_delta, request_cancelled):
            self.assertTrue(request_cancelled.wait(2))
            cancelled.set()
            return {"choices": [{"finish_reason": "stop"}]}

        api = LocalAPI(lambda: [], complete, port=0, cancellable=True, request_timeout=0.1)
        info = api.start()
        try:
            status, _, _ = self.request_to(info, {})
            self.assertEqual(status, 200)
            self.assertTrue(cancelled.wait(2))
        finally:
            api.close()

    def test_strict_json_and_duplicate_headers(self):
        for raw in (b'{"x":NaN}', b'{"x":1,"x":2}'):
            status, _, _ = self.request_to_path(
                self.info,
                "/v1/chat/completions",
                "POST",
                raw=raw,
                headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
            )
            self.assertEqual(status, 400)
        status, _, _ = self.request_to_path(self.info, "/models", "GET", headers={"Host": "127.0.0.1,127.0.0.1"})
        self.assertIn(status, (400, 403))

    def test_capacity_returns_retryable_429(self):
        entered = threading.Event()
        release = threading.Event()

        def complete(payload, on_delta=None):
            entered.set()
            release.wait(2)
            return {"choices": [{"finish_reason": "stop"}]}

        api = LocalAPI(lambda: [], complete, port=0, max_connections=1)
        info = api.start()
        try:
            first = threading.Thread(target=lambda: self.request_to(info, {}), daemon=True)
            first.start()
            self.assertTrue(entered.wait(2))
            try:
                self.request_to(info, {})
                self.fail("capacity request unexpectedly succeeded")
            except urllib.error.HTTPError as error:
                status, headers = error.code, error.headers
            self.assertEqual(status, 429)
            self.assertEqual(headers.get("Retry-After"), "1")
        finally:
            release.set()
            api.close()

    def test_bad_json_and_callback_error(self):
        req = urllib.request.Request(
            self.info["base_url"] + "/chat/completions",
            data=b"{",
            headers={
                "Authorization": "Bearer " + self.info["api_key"],
                "Content-Type": "application/json",
                "Content-Length": "1",
            },
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(req)
        self.assertEqual(context.exception.code, 400)

    def test_stream_type_and_api_error(self):
        self.assertEqual(
            self.request("POST", "/chat/completions", key=self.info["api_key"], body={"stream": "yes"})[0], 400
        )

        def fail(_):
            raise APIError(422, "bad request", "bad")

        api = LocalAPI(lambda: [], fail, port=0)
        info = api.start()
        try:
            req = urllib.request.Request(
                info["base_url"] + "/chat/completions",
                data=b"{}",
                headers={
                    "Authorization": "Bearer " + info["api_key"],
                    "Content-Type": "application/json",
                    "Content-Length": "2",
                },
                method="POST",
            )
            with urllib.request.urlopen(req):
                self.fail("callback error unexpectedly succeeded")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 422)
        finally:
            api.close()

    def test_deterministic_port_conflict_falls_back(self):
        import stark_convert.local_api as module

        real = module._BoundedHTTPServer
        calls = []

        class ConflictOnce(real):
            def __init__(self, address, handler, limit=16):
                calls.append(address[1])
                if len(calls) == 1:
                    raise OSError(98, "address in use")
                super().__init__(address, handler, limit)

        with mock.patch.object(module, "_BoundedHTTPServer", ConflictOnce):
            api = LocalAPI(lambda: [], lambda _: {}, port=8000)
            info = api.start()
            try:
                self.assertEqual(calls[0], 8000)
                self.assertEqual(calls[1], 0)
                self.assertNotIn(":8000/", info["base_url"])
            finally:
                api.close()

    def test_invalid_headers_and_method_are_fixed_json(self):
        base = self.info["base_url"] + "/chat/completions"
        for header, value in (("Transfer-Encoding", "identity"), ("Content-Length", "nope")):
            req = urllib.request.Request(
                base,
                data=b"{}",
                headers={"Authorization": "Bearer " + self.info["api_key"], header: value},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(req)
            self.assertIn(b'"error"', context.exception.read())
        req = urllib.request.Request(
            self.info["base_url"] + "/models",
            headers={"Authorization": "Bearer " + self.info["api_key"]},
            method="PATCH",
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(req)
        self.assertEqual(context.exception.headers.get_content_type(), "application/json")

    def test_callback_failure_is_generic_and_key_changes_after_restart(self):
        api = LocalAPI(lambda: [], lambda _: (_ for _ in ()).throw(RuntimeError("secret callback details")), port=0)
        first = api.start()
        output = io.StringIO()
        try:
            req = urllib.request.Request(
                first["base_url"] + "/chat/completions",
                data=b"{}",
                headers={"Authorization": "Bearer " + first["api_key"], "Content-Length": "2"},
                method="POST",
            )
            with (
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(urllib.error.HTTPError) as context,
            ):
                urllib.request.urlopen(req)
            self.assertNotIn(b"secret callback details", context.exception.read())
            self.assertEqual(output.getvalue(), "")
        finally:
            api.close()
        second = api.start()
        try:
            self.assertNotEqual(first["api_key"], second["api_key"])
        finally:
            api.close()

    def test_oversized_body_duplicate_length_and_empty_origin(self):
        self.assertEqual(self.request("GET", "/models", key=self.info["api_key"], headers={"Origin": ""})[0], 403)
        conn = http.client.HTTPConnection(
            "127.0.0.1", int(self.info["base_url"].rsplit(":", 1)[1].split("/")[0]), timeout=2
        )
        try:
            conn.putrequest("POST", "/v1/chat/completions")
            conn.putheader("Authorization", "Bearer " + self.info["api_key"])
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", "2")
            conn.putheader("Content-Length", "2")
            conn.endheaders(b"{}")
            self.assertEqual(conn.getresponse().status, 400)
        finally:
            conn.close()
        conn = http.client.HTTPConnection(
            "127.0.0.1", int(self.info["base_url"].rsplit(":", 1)[1].split("/")[0]), timeout=2
        )
        try:
            conn.putrequest("POST", "/v1/chat/completions")
            conn.putheader("Authorization", "Bearer " + self.info["api_key"])
            conn.putheader("Content-Length", str(1024 * 1024 + 1))
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 413)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
