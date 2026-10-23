import http.client
import io
import json
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from stark_convert.local_api import APIError, LocalAPI


class LocalAPITest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.api = LocalAPI(lambda: [{"id": "stark", "object": "model"}], self.complete)
        self.info = self.api.start()

    def tearDown(self):
        self.api.close()

    def complete(self, payload):
        self.calls.append(payload)
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

    def test_bad_json_and_callback_error(self):
        req = urllib.request.Request(
            self.info["base_url"] + "/chat/completions",
            data=b"{",
            headers={"Authorization": "Bearer " + self.info["api_key"], "Content-Length": "1"},
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
                headers={"Authorization": "Bearer " + info["api_key"], "Content-Length": "2"},
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
            conn.putheader("Host", "127.0.0.1")
            conn.putheader("Authorization", "Bearer " + self.info["api_key"])
            conn.putheader("Content-Length", str(1024 * 1024 + 1))
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 413)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
