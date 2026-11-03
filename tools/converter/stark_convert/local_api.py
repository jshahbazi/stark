"""Small, private OpenAI-compatible HTTP API for the menu-bar app.

Streaming requests are written as inference produces text deltas.
"""

from __future__ import annotations

import hmac
import json
import math
import re
import secrets
import select
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

MAX_TEXT_BODY_BYTES = 1024 * 1024
MAX_IMAGE_BODY_BYTES = 6 * 1024 * 1024
MAX_PENDING_BODY_BYTES = 24 * 1024 * 1024
# OpenWork (16) plus OpenCode Desktop (8) managed capabilities share this admission table.
MAX_API_CLIENTS = 24
MAX_API_CLIENT_TTL_SECONDS = 120
_CLIENT_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_API_SESSION = re.compile(r"[a-f0-9]{32}\Z")


@dataclass(frozen=True, repr=False)
class _ClientGrant:
    digest: bytes
    expires: float


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
        api_key: str,
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
        self._validate_key(api_key)
        self._api_key = api_key
        self._lock = threading.Lock()
        self._status = status
        self._cancellable = cancellable
        self._request_timeout = request_timeout
        self._write_timeout = write_timeout
        self._closing = threading.Event()
        self._body_lock = threading.Lock()
        self._pending_body_bytes = 0
        self._api_session = ""
        self._clients: dict[str, _ClientGrant] = {}
        # Kept separately from current admission grants: refreshing a token or
        # letting it expire must not interrupt an already admitted stream.
        self._client_requests: dict[str, set[threading.Event]] = {}
        self._clock = time.monotonic

    def _body_limit(self) -> int:
        status = self._status() if self._status is not None else {}
        capabilities = status.get("capabilities", {}) if isinstance(status, dict) else {}
        return (
            MAX_IMAGE_BODY_BYTES
            if isinstance(capabilities, dict) and capabilities.get("images") is True
            else MAX_TEXT_BODY_BYTES
        )

    def start(self) -> dict[str, str]:
        with self._lock:
            if self._server is not None:
                raise RuntimeError("LocalAPI is already running")
            session_closed = threading.Event()
            session_key = self._api_key
            handler = self._handler_class(session_key, session_closed)
            server = _BoundedHTTPServer(("127.0.0.1", self._port), handler, self._max_connections)
            self._server = server
            self._api_session = uuid.uuid4().hex
            self._api_key = session_key
            self._closing = session_closed
            self._thread = threading.Thread(target=server.serve_forever, name="stark-local-api", daemon=True)
            self._thread.start()
            return {
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": self._api_key,
                "api_session": self._api_session,
            }

    @property
    def api_session(self) -> str:
        """Private control-channel generation; never included in HTTP status."""
        with self._lock:
            return self._api_session

    @staticmethod
    def _validate_key(key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("A persisted 256-bit API credential is required.")

    def update_key(self, key):
        """Activate a credential already committed by the host to Keychain."""
        self._validate_key(key)
        with self._lock:
            if not hmac.compare_digest(self._api_key, key):
                self._revoke_clients_locked()
                if self._server is not None:
                    self._api_session = uuid.uuid4().hex
            self._api_key = key

    def _require_session_locked(self, api_session):
        if self._server is None or self._closing.is_set() or not self._cancellable:
            raise APIError(503, "The local API is unavailable.", "api_not_running")
        if (
            not isinstance(api_session, str)
            or _API_SESSION.fullmatch(api_session) is None
            or not hmac.compare_digest(api_session, self._api_session)
        ):
            raise APIError(409, "The local API connection changed.", "stale_api_session")

    def _prune_clients_locked(self, now):
        for client_id, grant in list(self._clients.items()):
            if grant.expires <= now and not self._client_requests.get(client_id):
                del self._clients[client_id]

    def grant_client(self, client_id: str, api_key: str, ttl_seconds: int, api_session: str) -> None:
        """Install a private, in-memory admission capability already vetted by Swift."""
        if (
            not isinstance(client_id, str)
            or _CLIENT_ID.fullmatch(client_id) is None
            or type(ttl_seconds) is not int
            or not 1 <= ttl_seconds <= MAX_API_CLIENT_TTL_SECONDS
        ):
            raise APIError(400, "Invalid API client grant.", "invalid_api_client_command")
        try:
            self._validate_key(api_key)
        except ValueError:
            raise APIError(400, "Invalid API client grant.", "invalid_api_client_command") from None
        with self._lock:
            self._require_session_locked(api_session)
            digest = hmac.digest(
                self._api_session.encode("ascii"),
                ("Bearer " + api_key).encode("ascii"),
                "sha256",
            )
            if hmac.compare_digest(api_key, self._api_key) or any(
                other != client_id and hmac.compare_digest(digest, grant.digest)
                for other, grant in self._clients.items()
            ):
                raise APIError(400, "Invalid API client grant.", "invalid_api_client_command")
            now = self._clock()
            self._prune_clients_locked(now)
            clients = self._clients.keys() | self._client_requests.keys()
            if client_id not in clients and len(clients) >= MAX_API_CLIENTS:
                raise APIError(429, "API client capacity reached.", "api_client_limit")
            # Refreshes retire prior admission credentials, preserving admitted
            # work. Explicit revoke still reaches requests from retired tokens.
            self._clients[client_id] = _ClientGrant(digest, now + ttl_seconds)

    def _revoke_clients_locked(self, client_id=None, client_prefix=None):
        clients = set(self._clients) | set(self._client_requests) if client_id is None else {client_id}
        if client_prefix is not None:
            clients = {identifier for identifier in clients if identifier.startswith(client_prefix)}
        for identifier in clients:
            self._clients.pop(identifier, None)
            for cancelled in self._client_requests.pop(identifier, ()):
                cancelled.set()

    def revoke_clients(self, api_session: str, client_id: str | None = None, client_prefix: str | None = None) -> None:
        """Revoke one client, every client sharing a managed-client prefix, or all clients."""
        if (client_id is not None and client_prefix is not None) or any(
            value is not None and (not isinstance(value, str) or _CLIENT_ID.fullmatch(value) is None)
            for value in (client_id, client_prefix)
        ):
            raise APIError(400, "Invalid API client revocation.", "invalid_api_client_command")
        with self._lock:
            self._require_session_locked(api_session)
            self._revoke_clients_locked(client_id, client_prefix)

    def _release_client(self, client_id, cancelled):
        if client_id is None:
            return
        with self._lock:
            active = self._client_requests.get(client_id)
            if active is not None:
                active.discard(cancelled)
                if not active:
                    del self._client_requests[client_id]

    @staticmethod
    def _client_route(method, path):
        if method == "POST":
            return path == "/v1/chat/completions"
        return method == "GET" and (
            path in {"/health", "/v1/health", "/v1/capabilities", "/v1/models"}
            or (
                path.startswith("/v1/models/")
                and len(path) > len("/v1/models/")
                and "?" not in path
                and "#" not in path
            )
        )

    def close(self) -> None:
        with self._lock:
            self._closing.set()
            self._revoke_clients_locked()
            self._api_session = ""
            server, thread = self._server, self._thread
            self._server = self._thread = None
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

            def _guard(self, cancelled):
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
                actual = self.headers.get("Authorization", "")
                encoded = actual.encode("utf-8", "replace")
                with api._lock:
                    expected = api._api_key
                    if expected and hmac.compare_digest(encoded, ("Bearer " + expected).encode("ascii")):
                        # In-memory principal for prefix-reuse scoping; never sent or logged.
                        cancelled.cache_principal = (api._api_session, None)
                        return None
                    digest = hmac.digest(api._api_session.encode("ascii"), encoded, "sha256")
                    now = api._clock()
                    for client_id, grant in api._clients.items():
                        if hmac.compare_digest(digest, grant.digest) and now < grant.expires:
                            if not api._client_route(self.command, self.path):
                                raise APIError(403, "API client access is limited to chat routes.", "api_client_scope")
                            # Register while holding the same lock as revoke.
                            # Body readers and queued requests cannot escape it.
                            api._client_requests.setdefault(client_id, set()).add(cancelled)
                            cancelled.cache_principal = (api._api_session, client_id)
                            return client_id
                raise APIError(401, "Invalid API key", "invalid_api_key")

            @staticmethod
            def _check_client_cancelled(client_id, cancelled):
                if client_id is not None and cancelled.is_set():
                    raise APIError(499, "Request cancelled.", "request_cancelled")

            def do_GET(self):
                cancelled = threading.Event()
                client_id = None
                try:
                    client_id = self._guard(cancelled)
                    if self.path in {"/health", "/v1/health"}:
                        status = api._status() if api._status else {"status": "ok"}
                        self._check_client_cancelled(client_id, cancelled)
                        self._send_json(200, status)
                        return
                    if self.path == "/v1/capabilities":
                        status = api._status() if api._status else {}
                        self._check_client_cancelled(client_id, cancelled)
                        self._send_json(200, status)
                        return
                    if self.path.startswith("/v1/models/"):
                        model_id = unquote(self.path[len("/v1/models/") :])
                        models = api._models()
                        model = next((item for item in models if item.get("id") == model_id), None)
                        if model is None:
                            raise APIError(404, "Not found", "not_found")
                        self._check_client_cancelled(client_id, cancelled)
                        self._send_json(200, model)
                        return
                    if self.path != "/v1/models":
                        raise APIError(404, "Not found", "not_found")
                    models = api._models()
                    self._check_client_cancelled(client_id, cancelled)
                    self._send_json(200, {"object": "list", "data": models})
                except APIError as exc:
                    self._error(exc)
                except Exception:
                    self._error(APIError(500, "Internal server error", "internal_error"))
                finally:
                    api._release_client(client_id, cancelled)

            def do_POST(self):
                cancelled = threading.Event()
                client_id = None
                monitor = None
                reserved_body = 0
                try:
                    client_id = self._guard(cancelled)
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
                    if length > api._body_limit():
                        raise APIError(413, "Request body is too large", "body_too_large")
                    with api._body_lock:
                        if api._pending_body_bytes + length > MAX_PENDING_BODY_BYTES:
                            raise APIError(429, "Pending request bytes exceed the server budget", "body_budget_full")
                        api._pending_body_bytes += length
                        reserved_body = length
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
                    if length > MAX_TEXT_BODY_BYTES and not self._contains_image(payload):
                        raise APIError(413, "Text request body is too large", "body_too_large")
                    if "stream" in payload and not isinstance(payload["stream"], bool):
                        raise APIError(400, "stream must be a boolean", "invalid_request")
                    streaming = payload.get("stream", False)
                    self._check_client_cancelled(client_id, cancelled)
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
                        if len(models) == 1:
                            model = models[0]["id"]
                        model = model or "stark"
                        started = False

                        def on_delta(text):
                            nonlocal started
                            if client_id is not None and cancelled.is_set():
                                # Worker catches OSError, requests native
                                # cancellation and drains the terminal frame.
                                raise OSError("Request cancelled.")
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
                    self._check_client_cancelled(client_id, cancelled)
                    if streaming:
                        choices = result.get("choices") or [{}]
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        # Text prefixes stream during generation. Calls stay
                        # private until terminal validation, then each is sent
                        # exactly once in model order with its buffered ID.
                        message = choice.get("message", {})
                        calls = message.get("tool_calls", []) if isinstance(message, dict) else []
                        for index, call in enumerate(calls):
                            self._sse_write(
                                self._sse_item(ident, model, created, {"tool_calls": [{"index": index, **call}]})
                            )
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
                    if reserved_body:
                        with api._body_lock:
                            api._pending_body_bytes -= reserved_body
                    cancelled.set()
                    if monitor is not None:
                        monitor.join(timeout=0.2)
                    api._release_client(client_id, cancelled)

            @staticmethod
            def _contains_image(payload):
                messages = payload.get("messages")
                return isinstance(messages, list) and any(
                    isinstance(message, dict)
                    and isinstance(message.get("content"), list)
                    and any(isinstance(part, dict) and part.get("type") == "image_url" for part in message["content"])
                    for message in messages
                )

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
