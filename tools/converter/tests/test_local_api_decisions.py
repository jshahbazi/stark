"""Decision routes share authenticated HTTP admission and cancellation controls."""

import http.client
import json
import socket
import threading

import pytest

from stark_convert.local_api import APIError, LocalAPI

KEY = "a" * 64


def send(api, path, payload, key=KEY, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", api._server.server_port, timeout=3)
    try:
        connection.request(
            "POST",
            path,
            json.dumps(payload),
            {"Authorization": "Bearer " + key, "Content-Type": "application/json", **(headers or {})},
        )
        reply = connection.getresponse()
        return reply.status, json.loads(reply.read())
    finally:
        connection.close()


@pytest.fixture
def api():
    calls = []

    def decide(payload, cancelled):
        calls.append((payload, cancelled))
        return {"model": "decider", "answers": {}, "usage": {"sequences": 1}}

    value = LocalAPI(lambda: [], lambda *_: {}, api_key=KEY, port=0, cancellable=True, decide=decide)
    value.start()
    yield value, calls
    value.close()


@pytest.mark.parametrize("path", ["/v1/decisions", "/v1/systemone"])
def test_decision_aliases_require_auth_and_preserve_payload(api, path):
    value, calls = api
    payload = {"state": {"synthetic": True}, "questions": {"check": {"type": "noul"}}}
    assert send(value, path, payload, key="b" * 64)[0] == 401
    assert not calls
    status, response = send(value, path, payload)
    assert status == 200 and response["model"] == "decider"
    assert calls[0][0] == payload and isinstance(calls[0][1], threading.Event)


def test_managed_chat_grants_do_not_gain_decision_scope(api):
    value, calls = api
    value.grant_client("openwork-peer", "b" * 64, 120, value.api_session)
    for path in ("/v1/decisions", "/v1/systemone"):
        status, body = send(value, path, {}, key="b" * 64)
        assert status == 403 and body["error"]["code"] == "api_client_scope"
    assert not calls


@pytest.mark.parametrize("payload", [[], {"stream": True}, {"stream": False}])
def test_decisions_reject_nonobjects_and_stream_before_callback(api, payload):
    value, calls = api
    assert send(value, "/v1/decisions", payload)[0] == 400
    assert not calls


def test_decision_callback_is_optional():
    value = LocalAPI(lambda: [], lambda *_: {}, api_key=KEY, port=0)
    value.start()
    try:
        assert send(value, "/v1/decisions", {})[0] == 404
    finally:
        value.close()


def test_decision_route_keeps_browser_origin_protection(api):
    value, calls = api
    assert send(value, "/v1/decisions", {}, headers={"Origin": "https://example.invalid"})[0] == 403
    assert not calls


def test_decision_disconnect_sets_cancellation():
    entered, stopped = threading.Event(), threading.Event()

    def decide(payload, cancelled):
        entered.set()
        if cancelled.wait(2):
            stopped.set()
        raise APIError(499, "Request cancelled.", "request_cancelled")

    value = LocalAPI(lambda: [], lambda *_: {}, api_key=KEY, port=0, cancellable=True, decide=decide)
    value.start()
    peer = socket.create_connection(("127.0.0.1", value._server.server_port))
    try:
        peer.sendall(
            b"POST /v1/decisions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer "
            + KEY.encode()
            + b"\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n{}"
        )
        assert entered.wait(2)
        peer.close()
        assert stopped.wait(2)
    finally:
        peer.close()
        value.close()
