"""Authenticated decision requests using the desktop's single worker queue."""

import json
import threading

from .chat import ChatCancelled
from .decision import DecisionInputError
from .local_api import APIError
from .media import MediaError, parse_image_url


def parse_decision_request(payload, encoder):
    """Validate System One fields and accept images only as inline data URLs."""
    if not isinstance(payload, dict) or set(payload) - {"model", "state", "questions", "images"}:
        raise APIError(400, "Unsupported decision request fields.", "unsupported_parameter")
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise APIError(400, "model must be a non-empty string.", "invalid_request")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise APIError(400, "questions must contain between 1 and 32 questions.", "invalid_request")
    for question in questions.values():
        if not isinstance(question, dict) or set(question) - {"type", "instructions", "criteria"}:
            raise APIError(400, "Unsupported decision question fields.", "unsupported_parameter")
        if "instructions" in question and not isinstance(question["instructions"], str):
            raise APIError(400, "Question instructions must be text.", "invalid_request")
        if question.get("type") == "noul" and isinstance(question.get("criteria"), dict):
            if set(question["criteria"]) - {"false", "true"}:
                raise APIError(400, "Binary criteria may contain only false and true.", "invalid_request")
    images = payload.get("images", [])
    if not isinstance(images, list) or len(images) > 1:
        raise APIError(400, "images must contain at most one inline PNG/JPEG image.", "invalid_request")
    try:
        image = parse_image_url(images[0]) if images else None
    except MediaError as error:
        raise APIError(error.status, str(error), error.code) from None
    request = {key: payload[key] for key in ("state", "questions") if key in payload}
    try:
        json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        encoder.validate(request)
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise APIError(400, "Invalid decision state, questions or criteria.", "invalid_request") from None
    return request, image


def run_decision(desktop, payload, cancelled):
    """Admit one complete decision without switching models or Playground mode."""
    # Imported here to keep Desktop's transport and worker modules independent.
    from .desktop import api_model_id, cancellation_error

    cancelled = cancelled or threading.Event()
    with desktop.lock:
        if desktop.api_closing:
            raise APIError(503, "Stark is shutting down.", "server_stopped")
        worker = desktop.worker
        if desktop.active_model is None or worker is None:
            raise APIError(503, "Run Decider Flash in Stark first.", "model_not_loaded")
        if desktop.current_state == "loading":
            raise APIError(429, "Stark is preparing the model. Retry after loading.", "model_busy")
        if getattr(worker, "backend", None) != "decision-native":
            raise APIError(400, "The running model does not support native decisions.", "mode_not_supported")
        request, image = parse_decision_request(payload, worker.encoder)
        active = next(model for model in desktop.models if model["id"] == desktop.active_model)
        public_id = api_model_id(active)
        if payload.get("model") not in (None, public_id, desktop.active_model, active["name"]):
            raise APIError(404, "Select the running model ID from /v1/models.", "model_not_found")
        epoch, identifier = desktop.epoch, desktop.active_model
        desktop.reserve_api_request(epoch, identifier, cancelled)
        desktop.status("deciding", "Deciding for an app…", epoch)
    try:
        result = worker.decide_request(request, image=image, cancelled=cancelled)
        with desktop.lock:
            if desktop.epoch != epoch or desktop.worker is not worker or desktop.active_model != identifier:
                raise APIError(503, "Model changed or stopped during this request.", "model_changed")
            if cancelled.is_set():
                raise cancellation_error(cancelled)
        return {"model": public_id, **result}
    except DecisionInputError:
        raise APIError(400, "Decision exceeds its input, image or memory limits.", "invalid_request") from None
    except MediaError as error:
        raise APIError(error.status, str(error), error.code) from None
    except ChatCancelled:
        with desktop.lock:
            if desktop.epoch != epoch or desktop.worker is not worker:
                raise APIError(503, "Model changed or stopped during this request.", "model_changed") from None
            if getattr(worker, "_closed", False):
                desktop.close_worker()
        raise cancellation_error(cancelled) from None
    except APIError:
        raise
    except BaseException:
        with desktop.lock:
            if desktop.epoch != epoch or desktop.worker is not worker:
                raise APIError(503, "Model changed or stopped during this request.", "model_changed") from None
            desktop.close_worker()
        if cancelled.is_set():
            raise cancellation_error(cancelled) from None
        raise APIError(503, "Decision failed. Run the model again in Stark.", "decision_failed") from None
    finally:
        with desktop.lock:
            if desktop.job is threading.current_thread():
                desktop.job = None
                desktop.api_condition.notify_all()
            if desktop.epoch == epoch:
                desktop.status(
                    "ready" if desktop.worker else "error", "Ready" if desktop.worker else "Model stopped", epoch
                )
