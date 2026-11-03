"""Managed client admission uses synthetic capabilities and real loopback HTTP."""

import http.client
import json
import socket
import threading
import time

import pytest

from stark_convert.local_api import MAX_API_CLIENTS, APIError, LocalAPI

MASTER = "a" * 64
CAP = "b" * 64
REPLACEMENT = "c" * 64


def completion(payload, on_delta, cancelled):
    if on_delta is not None:
        on_delta("synthetic")
    return {"choices": [{"message": {"role": "assistant", "content": "synthetic"}, "finish_reason": "stop"}]}


def request(api, key, method="GET", path="/v1/models", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", api._server.server_port, timeout=4)
    values = {"Authorization": "Bearer " + key, **(headers or {})}
    data = None if body is None else json.dumps(body).encode()
    if body is not None:
        values["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=data, headers=values)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Synthetic request did not reach its expected lifecycle state")
        threading.Event().wait(0.01)


@pytest.fixture
def api():
    value = LocalAPI(
        lambda: [{"id": "owner/model", "object": "model"}],
        completion,
        api_key=MASTER,
        port=0,
        cancellable=True,
        status=lambda: {"status": "ok"},
    )
    clock = [1000.0]
    value._clock = lambda: clock[0]
    value.start()
    yield value, clock
    value.close()


def grant(api, key=CAP, client="openwork-peer", ttl=120):
    api.grant_client(client, key, ttl, api.api_session)


def test_managed_capability_authenticates_only_supported_routes(api):
    value, _clock = api
    grant(value)
    for path in ["/health", "/v1/health", "/v1/capabilities", "/v1/models", "/v1/models/owner%2Fmodel"]:
        assert request(value, CAP, path=path)[0] == 200
    assert request(value, CAP, "POST", "/v1/chat/completions", {"stream": False})[0] == 200
    for method, path in [("GET", "/v1/admin"), ("POST", "/v1/models"), ("GET", "/v1/models?query=1")]:
        status, body = request(value, CAP, method, path, {} if method == "POST" else None)
        assert status == 403 and json.loads(body)["error"]["code"] == "api_client_scope"
    assert request(value, MASTER, path="/v1/admin")[0] == 404
    assert not value._client_requests


def test_managed_auth_keeps_host_origin_and_duplicate_header_protections(api):
    value, _clock = api
    grant(value)
    assert request(value, CAP, headers={"Origin": "https://example.invalid"})[0] == 403
    assert request(value, CAP, headers={"Host": "example.invalid"})[0] == 403
    connection = http.client.HTTPConnection("127.0.0.1", value._server.server_port, timeout=3)
    try:
        connection.putrequest("GET", "/v1/models")
        connection.putheader("Authorization", "Bearer " + CAP)
        connection.putheader("Authorization", "Bearer " + CAP)
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        response.read()
    finally:
        connection.close()


def test_expiry_is_server_monotonic_admission_and_master_key_stays_valid(api):
    value, clock = api
    grant(value, ttl=10)
    clock[0] += 9.9
    assert request(value, CAP)[0] == 200
    clock[0] += 0.1
    assert request(value, CAP)[0] == 401
    assert request(value, MASTER)[0] == 200
    grant(value)
    assert request(value, CAP)[0] == 200
    value.revoke_clients(value.api_session, "openwork-peer")
    assert request(value, CAP)[0] == 401
    assert request(value, MASTER)[0] == 200


@pytest.mark.parametrize(
    "client,key,ttl",
    [
        ("", CAP, 120),
        ("x" * 65, CAP, 120),
        ("peer/other", CAP, 120),
        ("peer", "bad", 120),
        ("peer", MASTER, 120),
        ("peer", CAP, 121),
        ("peer", CAP, 0),
        ("peer", CAP, True),
        ("peer", CAP, 1.5),
    ],
)
def test_grants_reject_invalid_or_escalating_values(api, client, key, ttl):
    value, _clock = api
    with pytest.raises(APIError) as error:
        grant(value, key, client, ttl)
    assert error.value.code == "invalid_api_client_command"
    assert not value._clients


def test_grants_are_bounded_refreshable_and_expired_idle_entries_are_reclaimed(api):
    value, clock = api
    for index in range(MAX_API_CLIENTS):
        grant(value, f"{index + 1:064x}", f"peer-{index}")
    with pytest.raises(APIError) as error:
        grant(value, CAP, "overflow")
    assert error.value.code == "api_client_limit"
    grant(value, REPLACEMENT, "peer-0")
    assert request(value, REPLACEMENT)[0] == 200
    with pytest.raises(APIError) as error:
        grant(value, REPLACEMENT, "peer-1")
    assert error.value.code == "invalid_api_client_command"
    clock[0] += 121
    grant(value, CAP, "new-peer")
    assert len(value._clients) == 1


@pytest.mark.parametrize("action", ["expiry", "same-token-refresh", "new-token-refresh"])
def test_expiry_and_refresh_never_interrupt_an_admitted_stream(api, action):
    value, clock = api
    entered, release = threading.Event(), threading.Event()
    controls = []

    def complete(payload, on_delta, cancelled):
        controls.append(cancelled)
        on_delta("first")
        entered.set()
        assert release.wait(3)
        on_delta("second")
        return {"choices": [{"message": {"content": "firstsecond"}, "finish_reason": "stop"}]}

    value._complete = complete
    grant(value)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(request(value, CAP, "POST", "/v1/chat/completions", {"stream": True}))
    )
    thread.start()
    try:
        assert entered.wait(3)
        clock[0] += 121
        if action == "expiry":
            assert request(value, CAP)[0] == 401
        else:
            token = CAP if action == "same-token-refresh" else REPLACEMENT
            grant(value, token)
            assert request(value, token)[0] == 200
            if token != CAP:
                assert request(value, CAP)[0] == 401
        assert not controls[0].is_set()
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert results[0][0] == 200 and results[0][1].count(b"data: [DONE]") == 1
        assert results[0][1].count(b'"content":"first"') == 1
        assert results[0][1].count(b'"content":"second"') == 1
        assert not value._client_requests
    finally:
        release.set()
        thread.join(3)


def test_explicit_revoke_cancels_requests_from_current_and_retired_tokens(api):
    value, _clock = api
    entered = []
    condition = threading.Condition()

    def complete(payload, on_delta, cancelled):
        on_delta("partial")
        with condition:
            entered.append(cancelled)
            condition.notify_all()
        assert cancelled.wait(3)
        raise APIError(499, "Request cancelled.", "request_cancelled")

    value._complete = complete
    grant(value)
    results = []
    threads = []
    try:
        for index, key in enumerate([CAP, REPLACEMENT]):
            if index:
                grant(value, REPLACEMENT)
            thread = threading.Thread(
                target=lambda token=key: results.append(
                    request(value, token, "POST", "/v1/chat/completions", {"stream": True})
                )
            )
            threads.append(thread)
            thread.start()
            with condition:
                assert condition.wait_for(lambda expected=index + 1: len(entered) == expected, 3)
        value.revoke_clients(value.api_session, "openwork-peer")
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
        assert all(control.is_set() for control in entered)
        assert len(results) == 2 and all(b"request_cancelled" in body and b"[DONE]" not in body for _, body in results)
        assert request(value, CAP)[0] == 401 and request(value, REPLACEMENT)[0] == 401
        assert request(value, MASTER)[0] == 200
        assert not value._client_requests and value._pending_body_bytes == 0
    finally:
        value.revoke_clients(value.api_session)
        for thread in threads:
            thread.join(3)


def test_prefix_revoke_cancels_only_that_managed_clients_grants_and_requests(api):
    value, _clock = api
    entered = {}
    condition = threading.Condition()
    release = threading.Event()

    def complete(payload, on_delta, cancelled):
        on_delta("partial")
        with condition:
            entered[payload["messages"][0]["content"]] = cancelled
            condition.notify_all()
        # Each synthetic client waits on its intended terminal event. Waiting
        # three seconds for OpenWork cancellation races the caller's join.
        if payload["messages"][0]["content"] == "opencode":
            assert cancelled.wait(3)
            raise APIError(499, "Request cancelled.", "request_cancelled")
        assert release.wait(3)
        return {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}

    value._complete = complete
    grant(value, CAP, "openwork.10.1.2")
    grant(value, REPLACEMENT, "opencode.20.3.4")
    results = {}
    threads = []
    try:
        for name, key in [("openwork", CAP), ("opencode", REPLACEMENT)]:
            body = {"stream": True, "messages": [{"role": "user", "content": name}]}
            thread = threading.Thread(
                target=lambda token=key, label=name, payload=body: results.__setitem__(
                    label, request(value, token, "POST", "/v1/chat/completions", payload)
                )
            )
            threads.append(thread)
            thread.start()
            with condition:
                assert condition.wait_for(lambda expected=name: expected in entered, 3)
        value.revoke_clients(value.api_session, client_prefix="opencode.")
        threads[1].join(3)
        assert not threads[1].is_alive()
        assert entered["opencode"].is_set() and not entered["openwork"].is_set()
        assert b"request_cancelled" in results["opencode"][1]
        assert request(value, REPLACEMENT)[0] == 401
        assert request(value, CAP)[0] == 200
        release.set()
        threads[0].join(3)
        assert results["openwork"][0] == 200 and b"[DONE]" in results["openwork"][1]
        assert set(value._clients) == {"openwork.10.1.2"} and not value._client_requests
    finally:
        release.set()
        value.revoke_clients(value.api_session)
        for thread in threads:
            thread.join(3)


@pytest.mark.parametrize(
    "client,prefix",
    [
        (None, ""),
        (None, "opencode/"),
        (None, "x" * 65),
        (None, 1),
        ("opencode.1.2.3", "opencode."),
    ],
)
def test_revocations_reject_invalid_prefixes_and_ambiguous_scopes(api, client, prefix):
    value, _clock = api
    grant(value)
    with pytest.raises(APIError) as error:
        value.revoke_clients(value.api_session, client, prefix)
    assert error.value.code == "invalid_api_client_command"
    assert request(value, CAP)[0] == 200


def test_prefix_revoke_without_matches_keeps_other_clients(api):
    value, _clock = api
    grant(value)
    value.revoke_clients(value.api_session, client_prefix="opencode.")
    assert request(value, CAP)[0] == 200
    value.revoke_clients(value.api_session, client_prefix="openwork-")
    assert request(value, CAP)[0] == 401


@pytest.mark.parametrize("action", ["key-change", "restart"])
def test_master_rotation_and_listener_restart_fence_grants_and_cancel_work(api, action):
    value, _clock = api
    entered = threading.Event()

    def complete(payload, on_delta, cancelled):
        entered.set()
        assert cancelled.wait(3)
        raise APIError(499, "Request cancelled.", "request_cancelled")

    value._complete = complete
    grant(value)
    session = value.api_session
    results = []
    thread = threading.Thread(target=lambda: results.append(request(value, CAP, "POST", "/v1/chat/completions", {})))
    thread.start()
    try:
        assert entered.wait(3)
        if action == "key-change":
            value.update_key(REPLACEMENT)
        else:
            value.close()
            value.start()
        thread.join(3)
        assert not thread.is_alive() and results[0][0] == 499
        assert value.api_session != session
        assert request(value, CAP)[0] == 401
        with pytest.raises(APIError) as error:
            value.grant_client("stale", CAP, 120, session)
        assert error.value.code == "stale_api_session"
        assert request(value, REPLACEMENT if action == "key-change" else MASTER)[0] == 200
    finally:
        if thread.is_alive():
            value.revoke_clients(value.api_session)
            thread.join(3)


def test_revoke_during_body_read_never_dispatches_completion(api):
    value, _clock = api
    called = []
    value._complete = lambda *args: called.append(True)
    grant(value)
    peer = socket.create_connection(("127.0.0.1", value._server.server_port), timeout=3)
    try:
        peer.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            + b"Authorization: Bearer "
            + CAP.encode()
            + b"\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n"
        )
        wait_for(lambda: bool(value._client_requests))
        value.revoke_clients(value.api_session)
        peer.sendall(b"{}")
        response = http.client.HTTPResponse(peer)
        response.begin()
        assert response.status == 499
        response.read()
        assert not called
        wait_for(lambda: not value._client_requests and value._pending_body_bytes == 0)
    finally:
        peer.close()


def test_same_master_retry_preserves_grants_and_session(api):
    value, _clock = api
    grant(value)
    session = value.api_session
    value.update_key(MASTER)
    assert value.api_session == session and request(value, CAP)[0] == 200
