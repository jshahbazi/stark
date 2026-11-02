"""Private client-grant commands and Desktop lifecycle use synthetic workers."""

import json
import socket
import threading
import uuid
from unittest.mock import patch

import pytest
from test_desktop import desktop as desktop
from test_desktop import events, running
from test_desktop import executable as executable
from test_desktop import model as model
from test_local_api_clients import CAP, MASTER, REPLACEMENT, request, wait_for

from stark_convert.chat import ChatCancelled, ChatError


@pytest.fixture
def server(desktop):
    desktop.start_api(MASTER, port=0)
    yield desktop
    desktop.api.close()


def command(controller, **overrides):
    payload = {
        "command": "grant_api_client",
        "request_id": str(uuid.uuid4()),
        "client_id": "openwork-peer",
        "api_key": CAP,
        "ttl_seconds": 120,
        "api_session": controller.api.api_session if controller.api is not None else "d" * 32,
    }
    payload.update(overrides)
    controller.dispatch(payload)
    return events(controller)[-1]


def revoke(controller, *, all_clients=False):
    payload = {
        "command": "revoke_api_clients",
        "request_id": str(uuid.uuid4()),
        "api_session": controller.api.api_session,
    }
    if not all_clients:
        payload["client_id"] = "openwork-peer"
    controller.dispatch(payload)
    return events(controller)[-1]


def test_connection_and_client_acknowledgements_never_echo_credentials(server):
    connection = events(server)[-1]
    assert len(connection["api_session"]) == 32
    assert "api_key" not in connection
    ack = command(server, request_id=str(uuid.uuid4()).upper())
    assert set(ack) == {"event", "request_id", "ok", "api_session"}
    assert ack["event"] == "api_client" and ack["ok"] is True
    assert ack["api_session"] == connection["api_session"]
    assert CAP not in server.output.getvalue() and MASTER not in server.output.getvalue()
    assert request(server.api, CAP)[0] == 200
    assert revoke(server)["ok"] is True
    assert request(server.api, CAP)[0] == 401


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"request_id": "not-a-uuid"}, "invalid_api_client_command"),
        ({"request_id": CAP}, "invalid_api_client_command"),
        ({"client_id": None}, "invalid_api_client_command"),
        ({"client_id": "peer/invalid"}, "invalid_api_client_command"),
        ({"api_key": "bad"}, "invalid_api_client_command"),
        ({"api_key": MASTER}, "invalid_api_client_command"),
        ({"ttl_seconds": True}, "invalid_api_client_command"),
        ({"ttl_seconds": 121}, "invalid_api_client_command"),
        ({"expires": 123456789}, "invalid_api_client_command"),
        ({"api_session": CAP}, "invalid_api_client_command"),
        ({"api_session": "d" * 32}, "stale_api_session"),
    ],
)
def test_private_grant_validation_is_strict_and_content_free(server, overrides, code):
    ack = command(server, **overrides)
    assert ack["ok"] is False and ack["error_code"] == code
    assert CAP not in server.output.getvalue() and MASTER not in server.output.getvalue()
    assert not server.api._clients


def test_grants_require_a_live_configured_listener_and_revoke_all_is_explicit(desktop):
    ack = command(desktop)
    assert ack["ok"] is False and ack["error_code"] == "api_not_running"
    assert ack["api_session"] == ""
    desktop.start_api(MASTER, port=0)
    try:
        assert command(desktop)["ok"] is True
        assert command(desktop, client_id="other-peer", api_key=REPLACEMENT)["ok"] is True
        assert revoke(desktop, all_clients=True)["ok"] is True
        assert request(desktop.api, CAP)[0] == 401
        assert request(desktop.api, REPLACEMENT)[0] == 401
        desktop.api.close()
        ack = command(desktop, api_session="d" * 32)
        assert ack["error_code"] == "api_not_running"
    finally:
        desktop.api.close()


def test_port_failure_preserves_client_grant_and_successful_rebind_fences_it(server):
    assert command(server)["ok"] is True
    old, session = server.api, server.api.api_session
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    try:
        server.configure_api({"command": "configure_api", "api_key": MASTER, "port": occupied.getsockname()[1]})
        assert server.api is old and server.api.api_session == session
        assert events(server)[-1]["api_session"] == session
        assert request(server.api, CAP)[0] == 200
        server.start_api(MASTER, port=0)
        assert server.api is not old and server.api.api_session != session
        assert request(server.api, CAP)[0] == 401
        ack = command(server, api_session=session)
        assert ack["ok"] is False and ack["error_code"] == "stale_api_session"
    finally:
        occupied.close()


def test_key_regeneration_fences_broker_commands_and_same_key_retry_preserves_them(server):
    assert command(server)["ok"] is True
    session = server.api.api_session
    server.configure_api({"command": "configure_api", "api_key": MASTER, "port": server.api_port})
    assert server.api.api_session == session and request(server.api, CAP)[0] == 200
    server.configure_api({"command": "configure_api", "api_key": REPLACEMENT, "port": server.api_port})
    assert server.api.api_session != session and events(server)[-1]["api_session"] == server.api.api_session
    assert request(server.api, CAP)[0] == 401 and request(server.api, MASTER)[0] == 401
    assert request(server.api, REPLACEMENT)[0] == 200
    assert command(server, api_session=session)["error_code"] == "stale_api_session"


def test_managed_discovery_distinguishes_no_model_wrong_model_and_wrong_key(server, model):
    assert command(server)["ok"] is True
    status, body = request(server.api, CAP)
    assert status == 200 and json.loads(body)["data"] == []
    payload = {"messages": [{"role": "user", "content": "Synthetic"}]}
    status, body = request(server.api, CAP, "POST", "/v1/chat/completions", payload)
    assert status == 503 and json.loads(body)["error"]["code"] == "model_not_loaded"
    running(server, model)
    status, body = request(server.api, CAP, "POST", "/v1/chat/completions", {**payload, "model": "missing"})
    assert status == 404 and json.loads(body)["error"]["code"] == "model_not_found"
    assert request(server.api, "d" * 64)[0] == 401
    assert request(server.api, CAP, "POST", "/v1/chat/completions", payload)[0] == 200


def test_revoke_releases_running_and_queued_desktop_requests_and_preserves_model(server, model):
    running(server, model)
    assert command(server)["ok"] is True
    worker = server.worker
    entered = threading.Event()
    original = worker.request

    def blocked(payload, on_delta=None, cancelled=None, on_progress=None):
        if payload["operation"] != "generate":
            return original(payload, on_delta=on_delta, cancelled=cancelled, on_progress=on_progress)
        if on_delta is not None:
            on_delta("synthetic partial")
        entered.set()
        assert cancelled.wait(3)
        raise ChatCancelled("Request cancelled.")

    results, threads = [], []
    payload = {"messages": [{"role": "user", "content": "Synthetic queued task"}]}
    try:
        with patch.object(worker, "request", side_effect=blocked):
            for stream in [True, False]:
                thread = threading.Thread(
                    target=lambda streamed=stream: results.append(
                        request(server.api, CAP, "POST", "/v1/chat/completions", {**payload, "stream": streamed})
                    )
                )
                threads.append(thread)
                thread.start()
                if stream:
                    assert entered.wait(3)
            wait_for(lambda: len(server.api_queue) == 1)
            assert revoke(server)["ok"] is True
            for thread in threads:
                thread.join(3)
                assert not thread.is_alive()
        assert sorted(status for status, _body in results) == [200, 499]
        assert all(b"request_cancelled" in body and b"[DONE]" not in body for _status, body in results)
        assert server.worker is worker and worker.process.poll() is None
        assert server.job is None and not server.api_queue
        wait_for(lambda: not server.api._client_requests and server.api._pending_body_bytes == 0)
        assert command(server, api_key=REPLACEMENT)["ok"] is True
        status, body = request(server.api, REPLACEMENT, "POST", "/v1/chat/completions", payload)
        assert status == 200 and b"synthetic partial" not in body
    finally:
        revoke(server, all_clients=True)
        for thread in threads:
            thread.join(3)


def test_model_stop_and_restart_preserve_client_admission_with_old_request_fenced(server, model):
    identifier = running(server, model)
    assert command(server)["ok"] is True
    worker = server.worker
    entered = threading.Event()
    original = worker.request

    def blocked(payload, on_delta=None, cancelled=None, on_progress=None):
        if payload["operation"] != "generate":
            return original(payload, on_delta=on_delta, cancelled=cancelled, on_progress=on_progress)
        entered.set()
        wait_for(lambda: worker.process.poll() is not None)
        raise ChatError("Synthetic stopped worker")

    results = []
    payload = {"messages": [{"role": "user", "content": "Synthetic model replacement"}]}
    with patch.object(worker, "request", side_effect=blocked):
        thread = threading.Thread(
            target=lambda: results.append(request(server.api, CAP, "POST", "/v1/chat/completions", payload))
        )
        thread.start()
        assert entered.wait(3)
        server.stop()
        thread.join(3)
        assert not thread.is_alive() and results[0][0] == 503
    assert server.job is None and not server.api_queue
    status, body = request(server.api, CAP, "POST", "/v1/chat/completions", payload)
    assert status == 503 and json.loads(body)["error"]["code"] == "model_not_loaded"
    server.run(server.epoch, identifier)
    assert request(server.api, CAP, "POST", "/v1/chat/completions", payload)[0] == 200
