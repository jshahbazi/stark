"""Small, private OpenAI-compatible HTTP API for the menu-bar app.

Generation is deliberately buffered: a streaming request is translated into
server-generated SSE events after the completion callback returns.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


class APIError(Exception):
    """An expected API error with a safe client-visible message."""

    def __init__(self, status: int, message: str, code: str):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class _BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, *_args):
        return

    def __init__(self, address, handler, limit: int = 16):
        super().__init__(address, handler)
        self.slots = threading.BoundedSemaphore(limit)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class LocalAPI:
    """Loopback-only API exposing model discovery and chat completion."""

    def __init__(
        self,
        models: Callable[[], list[dict[str, Any]]],
        complete: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        port: int = 8000,
        max_connections: int = 16,
    ):
        self._models = models
        self._complete = complete
        self._port = port
        self._max_connections = max_connections
        self._server: _BoundedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._api_key: str | None = None
        self._lock = threading.Lock()

    def start(self) -> dict[str, str]:
        with self._lock:
            if self._server is not None:
                raise RuntimeError("LocalAPI is already running")
            handler = self._handler_class()
            try:
                server = _BoundedHTTPServer(("127.0.0.1", self._port), handler, self._max_connections)
            except OSError as exc:
                if self._port != 8000 or exc.errno not in {48, 98, 10048}:
                    raise
                server = _BoundedHTTPServer(("127.0.0.1", 0), handler, self._max_connections)
            self._server = server
            self._api_key = secrets.token_urlsafe(32)
            self._thread = threading.Thread(target=server.serve_forever, name="stark-local-api", daemon=True)
            self._thread.start()
            return {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key": self._api_key}

    def close(self) -> None:
        with self._lock:
            server, thread = self._server, self._thread
            self._server = self._thread = None
            self._api_key = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def _handler_class(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def send_error(self, code, message=None, explain=None):
                self._send_json(
                    code,
                    {"error": {"message": "Request failed", "type": "invalid_request_error", "code": "http_error"}},
                )

            def handle(self):
                self.request.settimeout(3.0)
                try:
                    super().handle()
                except (OSError, ValueError):
                    return

            def _error(self, error: APIError):
                self._send_json(
                    error.status,
                    {"error": {"message": error.message, "type": "invalid_request_error", "code": error.code}},
                )

            def _send_json(self, status, value):
                payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def _send_sse(self, value):
                choices = value.get("choices") or [{}]
                choice = choices[0] if isinstance(choices[0], dict) else {}
                message = choice.get("message") or {}
                ident = value.get("id", "chatcmpl-buffered")
                model = value.get("model", "stark")
                created = value.get("created")

                def event(delta, finish=None):
                    item = {
                        "id": ident,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    if created is not None:
                        item["created"] = created
                    return "data: " + json.dumps(item, separators=(",", ":")) + "\n\n"

                text = event({"role": "assistant"})
                content = message.get("content")
                if content:
                    text += event({"content": content})
                text += event({}, choice.get("finish_reason", "stop")) + "data: [DONE]\n\n"
                payload = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def _guard(self):
                if "Origin" in self.headers:
                    raise APIError(403, "Origin requests are not allowed", "origin_not_allowed")
                host = self.headers.get("Host", "")
                hostname = urlsplit("//" + host).hostname
                if hostname not in {"127.0.0.1", "localhost", "::1"}:
                    raise APIError(403, "Host is not allowed", "host_not_allowed")
                expected = api._api_key or ""
                actual = self.headers.get("Authorization", "")
                if not expected or not hmac.compare_digest(
                    actual.encode("utf-8", "replace"), ("Bearer " + expected).encode("utf-8")
                ):
                    raise APIError(401, "Invalid API key", "invalid_api_key")

            def do_GET(self):
                try:
                    self._guard()
                    if self.path != "/v1/models":
                        raise APIError(404, "Not found", "not_found")
                    self._send_json(200, {"object": "list", "data": api._models()})
                except APIError as exc:
                    self._error(exc)
                except Exception:
                    self._error(APIError(500, "Internal server error", "internal_error"))

            def do_POST(self):
                try:
                    self._guard()
                    if self.path != "/v1/chat/completions":
                        raise APIError(404, "Not found", "not_found")
                    if "Transfer-Encoding" in self.headers:
                        raise APIError(411, "Chunked requests are not supported", "length_required")
                    content_lengths = self.headers.get_all("Content-Length", [])
                    if len(content_lengths) != 1:
                        raise APIError(400, "Content-Length is required", "length_required")
                    try:
                        length = int(content_lengths[0])
                    except (TypeError, ValueError):
                        raise APIError(400, "Invalid Content-Length", "invalid_request") from None
                    if length < 0:
                        raise APIError(400, "Invalid Content-Length", "invalid_request")
                    if length > 1024 * 1024:
                        raise APIError(413, "Request body is too large", "body_too_large")
                    try:
                        payload = json.loads(self.rfile.read(length))
                    except (ValueError, UnicodeDecodeError):
                        raise APIError(400, "Malformed JSON", "invalid_json") from None
                    if not isinstance(payload, dict):
                        raise APIError(400, "Request body must be an object", "invalid_request")
                    if "stream" in payload and not isinstance(payload["stream"], bool):
                        raise APIError(400, "stream must be a boolean", "invalid_request")
                    result = api._complete(payload)
                    if not isinstance(result, dict):
                        raise APIError(500, "Internal server error", "internal_error")
                    if payload.get("stream", False):
                        self._send_sse(result)
                    else:
                        self._send_json(200, result)
                except APIError as exc:
                    self._error(exc)
                except Exception:
                    self._error(APIError(500, "Internal server error", "internal_error"))

        return Handler
