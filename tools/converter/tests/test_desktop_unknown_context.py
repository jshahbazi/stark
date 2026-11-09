"""Unknown model capacity must not become a multi-billion-token allocation."""

from unittest.mock import patch

import pytest
from test_desktop import (
    advanced_options,
    events,
    finish,
    running,
)
from test_desktop import desktop as desktop
from test_desktop import executable as executable
from test_desktop import model as model

from stark_convert.chat import Worker
from stark_convert.desktop import DesktopError
from stark_convert.local_api import APIError

PROTOCOL_CEILING = 0xFFFFFFFF


@pytest.fixture
def unknown_context(desktop, model):
    read = Worker.read

    def ready(worker):
        result = read(worker)
        result["max_context_tokens"] = PROTOCOL_CEILING
        result["generation_budget"] = "memory-context-v1"
        return result

    with patch("stark_convert.desktop.Worker.read", autospec=True, side_effect=ready):
        running(desktop, model)
    return desktop


def test_unknown_capacity_is_not_advertised_as_a_model_maximum(unknown_context):
    entry = unknown_context.api_models()[0]
    assert entry["context_length"] is None
    assert entry["context_capacity_known"] is False
    ready = [event for event in events(unknown_context) if event["event"] == "ready"][-1]
    assert ready["context_capacity_known"] is False
    assert ready["max_context_tokens"] == PROTOCOL_CEILING


@pytest.mark.parametrize("stream", [False, True])
def test_unknown_capacity_auto_http_needs_no_options_and_keeps_capacity_null(unknown_context, stream):
    import json
    from urllib.request import Request, urlopen

    from stark_convert.local_api import LocalAPI

    worker = unknown_context.worker
    api = LocalAPI(
        unknown_context.api_models,
        unknown_context.api_complete,
        api_key="a" * 64,
        port=0,
        cancellable=True,
    )
    info = api.start()
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "Synthetic automatic request"}], "stream": stream})
        with patch.object(worker, "request", wraps=worker.request) as request:
            with urlopen(
                Request(
                    info["base_url"] + "/chat/completions",
                    data=body.encode(),
                    headers={"Authorization": "Bearer " + "a" * 64, "Content-Type": "application/json"},
                ),
                timeout=10,
            ) as response:
                assert response.status == 200
                received = response.read()
            generated = [call.args[0] for call in request.call_args_list if call.args[0]["operation"] == "generate"][
                -1
            ]
            assert generated["context_tokens"] == 8192
            assert generated["max_new_tokens"] == 8172
        if stream:
            assert b"[DONE]" in received
        else:
            assert json.loads(received)["usage"]["prompt_tokens"] == 20
        assert unknown_context.api_models()[0]["context_length"] is None
        assert unknown_context.worker is worker and worker.process.poll() is None
    finally:
        api.close()


def test_unknown_auto_memory_rejection_preserves_worker_and_allows_retry(unknown_context):
    from stark_convert.chat import ChatRequestTooLarge

    worker = unknown_context.worker
    original = worker.request

    def memory_pressure(payload, *args, **kwargs):
        if payload["operation"] == "plan_generation":
            raise ChatRequestTooLarge("Automatic execution cannot fit safely in available memory.")
        return original(payload, *args, **kwargs)

    with patch.object(worker, "request", side_effect=memory_pressure):
        with pytest.raises(APIError) as rejected:
            unknown_context.api_complete({"messages": [{"role": "user", "content": "Synthetic request"}]})
    assert rejected.value.code == "request_too_large"
    assert unknown_context.worker is worker and worker.process.poll() is None
    assert unknown_context.job is None
    assert unknown_context.api_complete({"messages": [{"role": "user", "content": "Synthetic retry"}]})["choices"]


def test_legacy_unknown_worker_reports_update_without_breaking_explicit_requests(unknown_context):
    worker = unknown_context.worker
    worker.generation_budget = False
    with pytest.raises(APIError) as unavailable:
        unknown_context.api_complete({"messages": [{"role": "user", "content": "Synthetic request"}]})
    assert unavailable.value.code == "automatic_output_unavailable"
    assert unknown_context.worker is worker and worker.process.poll() is None
    assert unknown_context.job is None
    result = unknown_context.api_complete(
        {"messages": [{"role": "user", "content": "Synthetic explicit request"}], "max_tokens": 32}
    )
    assert result["choices"]
    assert unknown_context.worker is worker


@pytest.mark.parametrize("stream", [False, True])
def test_unknown_capacity_sizes_native_text_state_per_request(unknown_context, stream):
    worker = unknown_context.worker
    deltas = []
    with patch.object(worker, "request", wraps=worker.request) as request:
        response = unknown_context.api_complete(
            {"messages": [{"role": "user", "content": "Synthetic request"}], "stream": stream, "max_tokens": 32},
            on_delta=deltas.append if stream else None,
        )
        assert request.call_args_list[-1].args[0]["context_tokens"] == 20 + 32
        unknown_context.api_complete(
            {"messages": [{"role": "user", "content": "Synthetic follow-up"}], "max_tokens": 64}
        )
        assert request.call_args_list[-1].args[0]["context_tokens"] == 20 + 64
    assert response["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    if stream:
        assert "".join(deltas) == response["choices"][0]["message"]["content"]
    assert unknown_context.worker is worker
    assert unknown_context.job is None


def test_unknown_capacity_playground_also_reserves_only_its_request(unknown_context):
    # Explicit output reserves only the actual input plus that allowance.
    unknown_context.maximum = 32
    worker = unknown_context.worker
    with patch.object(worker, "request", wraps=worker.request) as request:
        unknown_context.dispatch({"command": "generate", "text": "Synthetic playground request"})
        finish(unknown_context)
        assert request.call_args_list[-1].args[0]["context_tokens"] == 20 + unknown_context.maximum
    assert unknown_context.worker is worker
    assert unknown_context.job is None


def test_unknown_capacity_respects_custom_budget_and_preserves_worker(unknown_context):
    unknown_context.dispatch(advanced_options(context_tokens=21, max_new_tokens=1))
    worker = unknown_context.worker
    with pytest.raises(DesktopError, match="Conversation is full"):
        unknown_context.infer([{"role": "user", "content": "Synthetic overflow"}], 2)
    with patch.object(worker, "request", wraps=worker.request) as request:
        unknown_context.infer([{"role": "user", "content": "Synthetic recovery"}], 1)
        assert request.call_args_list[-1].args[0]["context_tokens"] == 21
    assert unknown_context.worker is worker


def test_model_replacement_restores_known_capacity_reporting_and_allocation(unknown_context):
    read = Worker.read

    def ready(worker):
        result = read(worker)
        result["max_context_tokens"] = 40960
        return result

    with patch("stark_convert.desktop.Worker.read", autospec=True, side_effect=ready):
        unknown_context.dispatch({"command": "run", "model_id": unknown_context.active_model})
        finish(unknown_context)
    assert unknown_context.api_models()[0]["context_length"] == 40960
    assert unknown_context.api_models()[0]["context_capacity_known"] is True
    with patch.object(unknown_context.worker, "request", wraps=unknown_context.worker.request) as request:
        unknown_context.api_complete({"messages": [{"role": "user", "content": "Synthetic finite capacity"}]})
        assert request.call_args_list[-1].args[0]["context_tokens"] == 40960


def test_unknown_image_context_is_rejected_before_native_dispatch(unknown_context):
    worker = unknown_context.worker
    with patch.object(worker, "request", wraps=worker.request) as request:
        with pytest.raises(APIError) as error:
            unknown_context.infer([{"role": "user", "content": "Synthetic image request"}], 32, image=object())
        assert error.value.code == "context_capacity_unknown"
        request.assert_not_called()
    assert unknown_context.worker is worker


def test_unknown_context_does_not_advertise_image_inference(unknown_context):
    with (
        patch.object(unknown_context.worker, "images", True),
        patch.object(unknown_context.worker, "image_verification", "locally_tested"),
        patch.object(unknown_context, "attachments", object()),
    ):
        assert unknown_context.settings_summary()["images"] is False
        assert unknown_context.settings_summary()["image_verification"] == "unverified"
        status = unknown_context.api_status()
        assert status["capabilities"]["images"] is False
        assert status["image_verification"] == "unverified"


def test_unknown_playground_image_is_rejected_without_claiming_or_stopping_worker(unknown_context):
    worker = unknown_context.worker
    with pytest.raises(DesktopError, match="declared finite context capacity"):
        unknown_context.generate(unknown_context.epoch, "Synthetic image", reference={})
    assert unknown_context.worker is worker
    assert unknown_context.image_identity is None
