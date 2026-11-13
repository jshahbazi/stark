import base64
import http.client
import json
import socket
import threading
import time

import pytest

import stark_convert.local_api as local_api_module
from stark_convert.local_api import APIError, LocalAPI
from stark_convert.media import MAX_IMAGE_BYTES, parse_image_url


def start_api(complete, *, images=True, cancellable=False):
    api = LocalAPI(
        lambda: [{"id": "test-model"}],
        complete,
        api_key="a" * 64,
        port=0,
        cancellable=cancellable,
        status=lambda: {"capabilities": {"images": images}},
        request_timeout=10,
    )
    info = api.start()
    port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
    return api, info, port


def post_bytes(info, port, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={
            "Authorization": "Bearer " + info["api_key"],
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    result = response.read()
    status = response.status
    connection.close()
    return status, result


def body_for_image_bytes(data):
    url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    return json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]},
        separators=(",", ":"),
    ).encode("ascii")


def wait_for_pending(api, expected, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with api._body_lock:
            if api._pending_body_bytes == expected:
                return True
        time.sleep(0.01)
    return False


def send_headers(port, key, length):
    connection = socket.create_connection(("127.0.0.1", port), timeout=3)
    connection.settimeout(3)
    connection.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Authorization: Bearer "
        + key.encode("ascii")
        + b"\r\nContent-Type: application/json\r\nContent-Length: "
        + str(length).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
    )
    return connection


def test_maximum_supported_image_body_passes_real_http_server():
    observed = []

    def complete(payload, _on_delta=None):
        image = parse_image_url(payload["messages"][0]["content"][0]["image_url"])
        observed.append((len(image.data), image.media_type))
        return {"choices": []}

    api, info, port = start_api(complete)
    try:
        image = b"\x89PNG\r\n\x1a\n" + b"x" * (MAX_IMAGE_BYTES - 8)
        body = body_for_image_bytes(image)
        assert 1024 * 1024 < len(body) <= local_api_module.MAX_IMAGE_BODY_BYTES
        status, _ = post_bytes(info, port, body)
        assert status == 200
        assert observed == [(MAX_IMAGE_BYTES, "image/png")]
        assert wait_for_pending(api, 0)
    finally:
        api.close()


@pytest.mark.parametrize("images", [False, True])
def test_text_body_above_input_cap_is_rejected_even_when_images_are_enabled(images, monkeypatch):
    monkeypatch.setattr(local_api_module, "MAX_TEXT_BODY_BYTES", 256)
    monkeypatch.setattr(local_api_module, "MAX_IMAGE_BODY_BYTES", 512)
    called = []
    api, info, port = start_api(lambda *_args: called.append(True) or {"choices": []}, images=images)
    try:
        body = json.dumps(
            {"messages": [{"role": "user", "content": "x" * 300}]},
            separators=(",", ":"),
        ).encode("ascii")
        assert len(body) > local_api_module.MAX_TEXT_BODY_BYTES
        assert len(body) < local_api_module.MAX_IMAGE_BODY_BYTES
        if images:
            status, _ = post_bytes(info, port, body)
            assert status == 413
        else:
            connection = send_headers(port, info["api_key"], len(body))
            try:
                response = connection.recv(1024)
                assert b"HTTP/1.1 413" in response
            finally:
                connection.close()
        assert called == []
        assert wait_for_pending(api, 0)
    finally:
        api.close()


@pytest.mark.parametrize("images", [False, True])
def test_over_cap_content_length_is_rejected_before_body_read(images):
    called = []
    api, info, port = start_api(lambda *_args: called.append(True) or {"choices": []}, images=images)
    expected_limit = local_api_module.MAX_IMAGE_BODY_BYTES if images else local_api_module.MAX_TEXT_BODY_BYTES
    connection = None
    try:
        connection = send_headers(port, info["api_key"], expected_limit + 1)
        response = connection.recv(1024)
        assert b"HTTP/1.1 413" in response
        assert called == []
        assert wait_for_pending(api, 0)
    finally:
        if connection is not None:
            connection.close()
        api.close()


def test_pending_body_budget_rejects_before_read_and_releases_on_disconnect(monkeypatch):
    monkeypatch.setattr(local_api_module, "MAX_TEXT_BODY_BYTES", 256)
    monkeypatch.setattr(local_api_module, "MAX_IMAGE_BODY_BYTES", 512)
    monkeypatch.setattr(local_api_module, "MAX_PENDING_BODY_BYTES", 400)
    called = []
    api, info, port = start_api(lambda *_args: called.append(True) or {"choices": []})
    first = second = None
    try:
        first = send_headers(port, info["api_key"], 256)
        assert wait_for_pending(api, 256)
        second = send_headers(port, info["api_key"], 200)
        response = b""
        while True:
            chunk = second.recv(1024)
            if not chunk:
                break
            response += chunk
        assert b"HTTP/1.1 429" in response
        assert b"body_budget_full" in response
        assert called == []
        with api._body_lock:
            assert api._pending_body_bytes == 256
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()
        assert wait_for_pending(api, 0)
        api.close()


def test_memory_pressure_rejects_large_body_before_read_and_allows_small_retry(monkeypatch):
    monkeypatch.setattr(local_api_module, "available_memory_bytes", lambda: 1024)
    called = []
    api, info, port = start_api(lambda *_args: called.append(True) or {"choices": []})
    connection = None
    try:
        connection = send_headers(port, info["api_key"], 1024 * 1024 + 1)
        response = b""
        while chunk := connection.recv(4096):
            response += chunk
        assert b"HTTP/1.1 429" in response
        assert b"memory_pressure" in response
        assert not called
        assert wait_for_pending(api, 0)
        status, _ = post_bytes(info, port, b"{}")
        assert status == 200
        assert called == [True]
    finally:
        if connection is not None:
            connection.close()
        api.close()


def test_memory_admission_accounts_for_mixed_unicode_json_peak(monkeypatch):
    # One supplementary-plane character widens the ASCII-heavy decoded JSON
    # and parsed content to four bytes per character in CPython.
    body = json.dumps(
        {"messages": [{"role": "user", "content": "x" * (1024 * 1024) + "\U0001f600"}]},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    monkeypatch.setattr(local_api_module, "available_memory_bytes", lambda: 6 * len(body))
    called = []
    api, info, port = start_api(lambda *_args: called.append(True) or {"choices": []})
    connection = None
    try:
        connection = send_headers(port, info["api_key"], len(body))
        response = b""
        while chunk := connection.recv(4096):
            response += chunk
        assert b"HTTP/1.1 429" in response
        assert b"memory_pressure" in response
        assert not called
        with api._body_lock:
            assert api._pending_body_bytes == api._pending_parse_bytes == 0
    finally:
        if connection is not None:
            connection.close()
        api.close()


def test_concurrent_large_bodies_reserve_parse_headroom_before_read(monkeypatch):
    length = 1024 * 1024 + 1
    monkeypatch.setattr(local_api_module, "available_memory_bytes", lambda: 18 * length)
    api, info, port = start_api(lambda *_args: {"choices": []})
    first = second = None
    try:
        first = send_headers(port, info["api_key"], length)
        assert wait_for_pending(api, length)
        second = send_headers(port, info["api_key"], length)
        response = b""
        while chunk := second.recv(4096):
            response += chunk
        assert b"HTTP/1.1 429" in response
        assert b"memory_pressure" in response
        with api._body_lock:
            assert api._pending_body_bytes == length
            assert api._pending_parse_bytes == 12 * length
        first.close()
        first = None
        assert wait_for_pending(api, 0)
        with api._body_lock:
            assert api._pending_parse_bytes == 0
        assert post_bytes(info, port, b"{}")[0] == 200
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()
        api.close()


def test_parse_reservation_released_while_parsed_completion_waits(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    length = 1024 * 1024 + 1
    monkeypatch.setattr(local_api_module, "available_memory_bytes", lambda: 18 * length)

    def complete(*_args):
        entered.set()
        assert release.wait(3)
        return {"choices": []}

    api, info, port = start_api(complete)
    first = second = None
    try:
        first = send_headers(port, info["api_key"], length)
        first.sendall(b"{}" + b" " * (length - 2))
        assert entered.wait(2)
        with api._body_lock:
            assert api._pending_body_bytes == length
            assert api._pending_parse_bytes == 0
        # The parsed request's actual resident memory is reflected in current
        # telemetry; its transient allowance must not remain reserved in queue.
        second = send_headers(port, info["api_key"], length)
        assert wait_for_pending(api, 2 * length)
        with api._body_lock:
            assert api._pending_parse_bytes == 12 * length
    finally:
        if second is not None:
            second.close()
        release.set()
        if first is not None:
            first.close()
        assert wait_for_pending(api, 0)
        with api._body_lock:
            assert api._pending_parse_bytes == 0
        api.close()


@pytest.mark.parametrize("failure", [False, True])
def test_pending_budget_released_after_completion_success_or_failure(monkeypatch, failure):
    monkeypatch.setattr(local_api_module, "MAX_PENDING_BODY_BYTES", 1024)
    called = []

    def complete(_payload, _on_delta=None):
        called.append(True)
        if failure:
            raise APIError(400, "Invalid request.", "invalid_request")
        return {"choices": []}

    api, info, port = start_api(complete)
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "test"}]}).encode("ascii")
        status, _ = post_bytes(info, port, body)
        assert status == (400 if failure else 200)
        assert called == [True]
        assert wait_for_pending(api, 0)
    finally:
        api.close()


def test_pending_budget_released_after_client_disconnect():
    entered = threading.Event()
    cancellation_seen = threading.Event()

    def complete(_payload, _on_delta, cancelled):
        entered.set()
        if cancelled.wait(3):
            cancellation_seen.set()
        raise APIError(499, "Request cancelled.", "request_cancelled")

    api, info, port = start_api(complete, cancellable=True)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    body = json.dumps({"messages": [{"role": "user", "content": "test"}]}).encode("ascii")
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Authorization": "Bearer " + info["api_key"],
                "Content-Type": "application/json",
            },
        )
        assert entered.wait(2)
        connection.close()
        assert cancellation_seen.wait(2)
        assert wait_for_pending(api, 0)
    finally:
        connection.close()
        api.close()
