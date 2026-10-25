"""Small, private OpenAI-compatible HTTP API for the menu-bar app.

Streaming requests are written as inference produces text deltas.
"""

from __future__ import annotations

import hmac
import json
import math
import secrets
import select
import socket
import threading
import time
import uuid
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

    def payload(self):
        error_type = (
            "authentication_error"
            if self.status == 401
            else "rate_limit_error"
            if self.status == 429
            else "server_error"
            if self.status >= 500
            else "invalid_request_error"
        )
        return {"error": {"message": self.message, "type": error_type, "code": self.code}}


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
            try:
                request.settimeout(0.01)
                body = b'{"error":{"message":"Server is busy","type":"rate_limit_error","code":"server_busy"}}'
                request.sendall(
                    b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n"
                    b"Retry-After: 1\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\nConnection: close\r\n\r\n"
                    + body
                )
            except OSError:
                pass
            finally:
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
    """Loopback-only API exposing model discovery and chat completion.

    ``models`` returns the ``/v1/models`` list and ``status`` the health document.
    ``complete`` runs one completion and its calling convention depends on
    ``cancellable``: with ``cancellable=False`` it is called as ``complete(payload)``
    for buffered requests and ``complete(payload, on_delta)`` for streaming ones;
    with ``cancellable=True`` (the menu bar app) it is always called as
    ``complete(payload, on_delta, cancelled)`` where ``on_delta`` is ``None`` for
    buffered requests and ``cancelled`` is a ``threading.Event`` that the handler
    sets when the client disconnects or ``request_timeout`` seconds pass; in the
    deadline case the handler also sets ``cancelled.reason = "deadline"`` so the
    callee can distinguish the two. ``complete`` reports failures by raising
    ``APIError``. The HTTP contract is documented in docs/LOCAL_API.md.
    """

    def __init__(
        self,
        models: Callable[[], list[dict[str, Any]]],
        complete: Callable[..., dict[str, Any]],
        *,
        port: int = 8000,
        max_connections: int = 16,
        status: Callable[[], dict[str, Any]] | None = None,
        cancellable: bool = False,
        request_timeout: float = 300,
        write_timeout: float = 10,
    ):
        self._models = models
        self._complete = complete
        self._port = port
        self._max_connections = max_connections
        self._server: _BoundedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._api_key: str | None = None
        self._lock = threading.Lock()
        self._status = status
        self._cancellable = cancellable
        self._request_timeout = request_timeout
        self._write_timeout = write_timeout
        self._closing = threading.Event()

    def start(self) -> dict[str, str]:
        with self._lock:
            if self._server is not None:
                raise RuntimeError("LocalAPI is already running")
            session_closed = threading.Event()
            session_key = secrets.token_urlsafe(32)
            handler = self._handler_class(session_key, session_closed)
            try:
                server = _BoundedHTTPServer(("127.0.0.1", self._port), handler, self._max_connections)
            except OSError as exc:
                if self._port != 8000 or exc.errno not in {48, 98, 10048}:
                    raise
                server = _BoundedHTTPServer(("127.0.0.1", 0), handler, self._max_connections)
            self._server = server
            self._api_key = session_key
            self._closing = session_closed
            self._thread = threading.Thread(target=server.serve_forever, name="stark-local-api", daemon=True)
            self._thread.start()
            return {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key": self._api_key}

    def close(self) -> None:
        with self._lock:
            self._closing.set()
            server, thread = self._server, self._thread
            self._server = self._thread = None
            self._api_key = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def _handler_class(self, session_key, session_closed):
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
                self._send_json(error.status, error.payload())

            def _send_json(self, status, value):
                payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                if status == 429:
                    self.send_header("Retry-After", "1")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Request-Id", "req-" + uuid.uuid4().hex)
                self.end_headers()
                self.wfile.write(payload)

            def _sse_item(self, ident, model, created, delta, finish=None):
                return {
                    "id": ident,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            def _start_sse(self, ident, model, created):
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("X-Request-Id", "req-" + uuid.uuid4().hex)
                self.end_headers()

                self._sse_write(self._sse_item(ident, model, created, {"role": "assistant"}))

            def _sse_write(self, value):
                payload = ("data: " + json.dumps(value, separators=(",", ":")) + "\n\n").encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()

            def _sse_done(self):
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def _sse_error(self, error):
                self._sse_write(error.payload())

            def _guard(self):
                if session_closed.is_set():
                    raise APIError(503, "Server is shutting down", "server_shutdown")
                if len(self.headers.get_all("Host", [])) > 1 or len(self.headers.get_all("Authorization", [])) > 1:
                    raise APIError(400, "Duplicate request headers are not allowed", "invalid_request")
                if "Origin" in self.headers:
                    raise APIError(403, "Origin requests are not allowed", "origin_not_allowed")
                host = self.headers.get("Host", "")
                hostname = urlsplit("//" + host).hostname
                if hostname not in {"127.0.0.1", "localhost", "::1"}:
                    raise APIError(403, "Host is not allowed", "host_not_allowed")
                expected = session_key
                actual = self.headers.get("Authorization", "")
                if not expected or not hmac.compare_digest(
                    actual.encode("utf-8", "replace"), ("Bearer " + expected).encode("utf-8")
                ):
                    raise APIError(401, "Invalid API key", "invalid_api_key")

            def do_GET(self):
                try:
                    self._guard()
                    if self.path in {"/health", "/v1/health"}:
                        self._send_json(200, api._status() if api._status else {"status": "ok"})
                        return
                    if self.path == "/v1/capabilities":
                        self._send_json(200, api._status() if api._status else {})
                        return
                    if self.path.startswith("/v1/models/"):
                        model_id = self.path[len("/v1/models/") :]
                        models = api._models()
                        model = next((item for item in models if item.get("id") == model_id), None)
                        if model is None:
                            raise APIError(404, "Not found", "not_found")
                        self._send_json(200, model)
                        return
                    if self.path != "/v1/models":
                        raise APIError(404, "Not found", "not_found")
                    self._send_json(200, {"object": "list", "data": api._models()})
                except APIError as exc:
                    self._error(exc)
                except Exception:
                    self._error(APIError(500, "Internal server error", "internal_error"))

            def do_POST(self):
                cancelled = threading.Event()
                monitor = None
                try:
                    self._guard()
                    if self.path != "/v1/chat/completions":
                        raise APIError(404, "Not found", "not_found")
                    if "Transfer-Encoding" in self.headers:
                        raise APIError(411, "Chunked requests are not supported", "length_required")
                    content_lengths = self.headers.get_all("Content-Length", [])
                    if len(content_lengths) != 1:
                        raise APIError(400, "Content-Length is required", "length_required")
                    raw_length = content_lengths[0].strip()
                    if not raw_length or not raw_length.isascii() or not raw_length.isdecimal():
                        raise APIError(400, "Invalid Content-Length", "invalid_request")
                    if len(raw_length.lstrip("0")) > 7:
                        raise APIError(413, "Request body is too large", "body_too_large")
                    length = int(raw_length.lstrip("0") or "0")
                    if length > 1024 * 1024:
                        raise APIError(413, "Request body is too large", "body_too_large")
                    content_type = self.headers.get("Content-Type")
                    if content_type and content_type.split(";", 1)[0].strip().lower() != "application/json":
                        raise APIError(415, "Content-Type must be application/json", "unsupported_media_type")
                    try:
                        body = self.rfile.read(length)
                        if len(body) != length:
                            raise APIError(400, "Incomplete request body", "invalid_request")
                        payload = json.loads(
                            body,
                            object_pairs_hook=self._pairs,
                            parse_constant=self._constant,
                            parse_float=self._float,
                        )
                    except (ValueError, UnicodeDecodeError, RecursionError):
                        raise APIError(400, "Malformed JSON", "invalid_json") from None
                    if not isinstance(payload, dict):
                        raise APIError(400, "Request body must be an object", "invalid_request")
                    if "stream" in payload and not isinstance(payload["stream"], bool):
                        raise APIError(400, "stream must be a boolean", "invalid_request")
                    streaming = payload.get("stream", False)
                    self.request.settimeout(api._write_timeout)
                    if api._cancellable:
                        monitor = threading.Thread(target=self._monitor_peer, args=(cancelled,), daemon=True)
                        monitor.start()
                    if not streaming:
                        result = (
                            api._complete(payload, None, cancelled) if api._cancellable else api._complete(payload)
                        )
                    else:
                        ident = "chatcmpl-" + secrets.token_urlsafe(16)
                        created = int(time.time())
                        model = payload.get("model")
                        models = api._models()
                        if len(models) == 1 and model in (None, models[0].get("id"), models[0].get("name")):
                            model = models[0]["id"]
                        model = model or "stark"
                        started = False

                        def on_delta(text):
                            nonlocal started
                            if not isinstance(text, str):
                                raise APIError(500, "Internal server error", "internal_error")
                            if not started:
                                started = True
                                self._start_sse(ident, model, created)
                            if text:
                                self._sse_write(self._sse_item(ident, model, created, {"content": text}))

                        try:
                            result = (
                                api._complete(payload, on_delta, cancelled)
                                if api._cancellable
                                else api._complete(payload, on_delta)
                            )
                        except APIError as exc:
                            if started:
                                self._sse_error(exc)
                                return
                            raise
                        if not isinstance(result, dict):
                            raise APIError(500, "Internal server error", "internal_error")
                        if not started:
                            started = True
                            self._start_sse(ident, model, created)
                    if not isinstance(result, dict):
                        raise APIError(500, "Internal server error", "internal_error")
                    if streaming:
                        choices = result.get("choices") or [{}]
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        self._sse_write(
                            self._sse_item(
                                ident,
                                model,
                                created,
                                {},
                                choice.get("finish_reason", "stop"),
                            )
                        )
                        stream_options = payload.get("stream_options")
                        if isinstance(stream_options, dict) and stream_options.get("include_usage"):
                            usage = result.get("usage")
                            if isinstance(usage, dict):
                                item = self._sse_item(ident, model, created, {}, None)
                                item["choices"] = []
                                item["usage"] = usage
                                self._sse_write(item)
                        self._sse_done()
                    else:
                        self._send_json(200, result)
                except APIError as exc:
                    if "streaming" in locals() and streaming and "started" in locals() and started:
                        self._sse_error(exc)
                    else:
                        self._error(exc)
                except OSError:
                    return
                except Exception:
                    error = APIError(500, "Internal server error", "internal_error")
                    if "streaming" in locals() and streaming and "started" in locals() and started:
                        self._sse_error(error)
                    else:
                        self._error(error)
                finally:
                    cancelled.set()
                    if monitor is not None:
                        monitor.join(timeout=0.2)

            @staticmethod
            def _pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate JSON key")
                    result[key] = value
                return result

            @staticmethod
            def _constant(value):
                raise ValueError(value)

            @staticmethod
            def _float(value):
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("nonfinite number")
                return number

            def _monitor_peer(self, cancelled):
                deadline = time.monotonic() + api._request_timeout
                while not cancelled.wait(0.05):
                    if session_closed.is_set():
                        cancelled.set()
                        return
                    if time.monotonic() >= deadline:
                        cancelled.reason = "deadline"
                        cancelled.set()
                        return
                    try:
                        ready, _, _ = select.select([self.request], [], [], 0)
                        if ready and not self.request.recv(1, socket.MSG_PEEK):
                            cancelled.set()
                            return
                    except (OSError, ValueError):
                        cancelled.set()
                        return

        return Handler
