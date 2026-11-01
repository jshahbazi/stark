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
