import base64
import io
import os
import signal
import threading

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from stark_convert.chat import ChatCancelled, ChatImageError, Worker
from stark_convert.desktop import Desktop
from stark_convert.local_api import APIError
from stark_convert.media import PrivateMediaStore
from stark_convert.vision_contract import REPO, REVISION
from stark_convert.vision_runtime import _ARTIFACT

PNG = b"\x89PNG\r\n\x1a\nsynthetic"
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
TEMPLATE = (
    "{% for m in messages %}{{ m.role }}:{% if m.content is string %}{{ m.content }}"
    "{% else %}{% for p in m.content %}{% if p.type == 'text' %}{{ p.text }}"
    "{% else %}[image]{% endif %}{% endfor %}{% endif %};{% endfor %}assistant:"
)
IMAGE_CONTRACT = {
    "version": "mediaref-v1",
    "transport": "owner-private-file-v1",
    "model_source": REPO,
    "model_revision": REVISION,
    "artifact_sha256": _ARTIFACT,
    "runtime_identity": {
        "backend": "metal",
        "device_name": "Apple Synthetic GPU",
        "apple_gpu_family": 1008,
        "gpu_core_count": 32,
        "supports_mxu": False,
    },
    "config_sha256": "c" * 64,
    "tokenizer_sha256": "d" * 64,
    "processor_sha256": "e" * 64,
    "template_sha256": "f" * 64,
    "artifact_schema_version": 2,
    "max_images": 1,
    "max_compressed_bytes": 4 * 1024 * 1024,
    "max_source_pixels": 32_000_000,
    "max_processed_pixels": 1_048_576,
    "max_patches": 4096,
    "max_image_tokens": 1024,
    "operations": ["prepare_image", "generate_image", "discard_image"],
    "formats": ["image/jpeg", "image/png"],
}


class FakeWorker:
    def __init__(self, *, images=True, outcomes=None):
        self.images = images
        self.image_input = dict(IMAGE_CONTRACT)
        self.streaming = True
        self.stop_sequences = True
        self.cancellation = True
        self.image_calls = []
        self.text_calls = []
        self.outcomes = list(outcomes or [])
        self.closed = False
        self.process = self

    def poll(self):
        return None

    def request(self, payload, **_options):
        self.text_calls.append(payload)
        return {"token_count": 7, "token_ids": [1] * 7}

    def request_image(self, prompt, image, options, on_delta=None, cancelled=None):
        self.image_calls.append(
            {
                "prompt": prompt,
                "media_type": image.media_type,
                "byte_length": len(image.data),
                "options": dict(options),
                "stream_callback": on_delta,
                "cancelled": cancelled,
            }
        )
        if self.outcomes:
            result = self.outcomes.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if on_delta is not None:
            on_delta("synthetic delta")
        return {
            "text": "synthetic answer",
            "input_tokens": 17,
            "output_tokens": 2,
            "finish_reason": "eos",
        }

    def close(self):
        self.closed = True


def make_desktop(worker):
    desktop = object.__new__(Desktop)
    desktop.lock = threading.RLock()
    desktop.write_lock = threading.Lock()
    desktop.output = io.StringIO()
    desktop.output_failed = False
    desktop.epoch = 1
    desktop.job = None
    desktop.worker = worker
    desktop.active_model = "private-model-id"
    desktop.active_mode = "chat"
    desktop.codec = {"bos_token": "", "eos_token": "", "default_system_prompt": None}
    desktop.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(TEMPLATE)
    desktop.models = [
        {
            "id": "private-model-id",
            "name": "vision-test-model",
            "path": "/private/synthetic/model",
            "source": "local:/private/synthetic/model",
        }
    ]
    desktop.api_closing = False
    desktop.api_condition = threading.Condition(desktop.lock)
    desktop.api_queue = []
    desktop.api_queue_limit = 8
    desktop.api_queue_timeout = 3
    desktop.api_context = 64
    desktop.api_maximum = 16
    desktop.api_vocab = 256
    desktop.thinking = False
    return desktop


def payload(*, stream=False):
    return {
        "model": "vision-test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "auto"}},
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        "max_tokens": 3,
        "temperature": 0.7,
        "top_k": 4,
        "seed": 11,
        "stop": ["END"],
        "stream": stream,
    }


def test_image_api_uses_prepared_image_path_and_full_token_usage_without_leaking_bytes():
    worker = FakeWorker(images=True)
    desktop = make_desktop(worker)
    deltas = []
    response = desktop.api_complete(payload(stream=True), on_delta=deltas.append)

    assert response["choices"][0]["message"]["content"] == "synthetic answer"
    assert response["usage"] == {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19}
    assert deltas == ["synthetic delta"]
    assert worker.text_calls == []
    assert len(worker.image_calls) == 1
    call = worker.image_calls[0]
    assert call["prompt"] == "user:before[image]after;assistant:"
    assert DATA_URL not in call["prompt"]
    assert call["media_type"] == "image/png"
    assert call["byte_length"] == len(PNG)
    assert call["options"] == {
        "context_tokens": 64,
        "max_new_tokens": 3,
        "sampling": {"mode": "stochastic", "temperature": 0.7, "top_k": 4},
        "seed": 11,
        "stop": ["END"],
        "stream": True,
    }
    output = desktop.output.getvalue()
    assert DATA_URL not in output
    assert "synthetic delta" not in output
    assert "synthetic answer" not in output
    assert desktop.job is None
    assert desktop.api_queue == []


def test_image_capability_false_rejects_before_worker_dispatch():
    worker = FakeWorker(images=False)
    desktop = make_desktop(worker)
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload())
    assert error.value.status == 400
    assert error.value.code == "images_unsupported"
    assert worker.image_calls == []
    assert worker.text_calls == []
    assert not worker.closed
    assert desktop.job is None


def test_recoverable_image_error_preserves_worker_and_retry_succeeds():
    worker = FakeWorker(outcomes=[ChatImageError("invalid_image_request")])
    desktop = make_desktop(worker)
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload())
    assert error.value.status == 400
    assert error.value.code == "invalid_image_request"
    assert desktop.worker is worker
    assert not worker.closed
    response = desktop.api_complete(payload())
    assert response["usage"]["prompt_tokens"] == 17
    assert desktop.worker is worker
    assert not worker.closed
    assert len(worker.image_calls) == 2


def test_image_context_overflow_maps_to_context_length_error():
    worker = FakeWorker(outcomes=[ChatImageError("context_overflow")])
    desktop = make_desktop(worker)
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload())
    assert error.value.status == 400
    assert error.value.code == "context_length_exceeded"
    assert desktop.worker is worker
    assert not worker.closed


def test_image_cancellation_maps_to_499_and_releases_job():
    worker = FakeWorker(outcomes=[ChatCancelled("Request cancelled.")])
    desktop = make_desktop(worker)
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload(), cancelled=threading.Event())
    assert error.value.status == 499
    assert error.value.code == "request_cancelled"
    assert desktop.job is None
    assert desktop.api_queue == []
    assert desktop.worker is worker
    assert not worker.closed


class BufferedStreamWorker(Worker):
    """Replay buffered protocol frames through the actual cancel/drain path."""

    def __init__(self, *, pid, fatal=False):
        self.timeout = 2
        self._transaction_lock = threading.RLock()
        self.request_sequence = 0
        self.cancellation = True
        self.cancelled_id = None
        self.media = PrivateMediaStore()
        self.image_input = dict(IMAGE_CONTRACT)
        self.images = self.streaming = self.stop_sequences = True
        self.process = self
        self.pid = pid
        self.group_signals = []
        self.stdin = self.Input(self)
        self.stdout = io.BytesIO()
        self.closed = False
        self.before_delta = None
        self.fatal = fatal
        self.operations = []
        self.replies = []
        self.cancelled_requests = []
        self.generation_terminals = 0

    class Input(io.BytesIO):
        def __init__(self, worker):
            super().__init__()
            self.worker = worker

        def write(self, data):
            self.worker.accept(data)
            return len(data)

    def accept(self, data):
        import json

        request = json.loads(bytes(data))
        operation = request["operation"]
        self.operations.append(operation)
        if operation == "tokenize":
            replies = [{"token_count": 7, "token_ids": [1] * 7}]
        elif operation == "prepare_image":
            replies = [{"prepared_id": "a" * 32, "input_tokens": 17, "image_tokens": 4}]
        elif operation in {"generate_image", "generate"}:
            terminal = (
                {"error": "native_generation_failed"}
                if self.fatal
                else {"text": "synthetic delta", "input_tokens": 17, "output_tokens": 2, "finish_reason": "eos"}
            )
            replies = (
                [{"event": "delta", "text": "synthetic delta"}, terminal] if request.get("stream") else [terminal]
            )
        else:
            assert operation == "discard_image"
            replies = [{"discarded": True}]
        self.replies.extend((operation, reply) for reply in replies)

    def _write_image_frame(self, data):
        self.accept(data)

    def read(self, _cancelled=None, _request_id=None):
        operation, reply = self.replies.pop(0)
        if "event" in reply and self.before_delta is not None:
            callback, self.before_delta = self.before_delta, None
            callback()
        if operation in {"generate_image", "generate"} and "event" not in reply:
            self.generation_terminals += 1
        return reply

    def cancel(self, request_id):
        self.cancelled_requests.append(request_id)

    def poll(self):
        return 0 if self.closed else None

    def terminate(self):
        self.closed = True

    kill = terminate

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def buffered_worker(monkeypatch):
    workers = {}

    def killpg(process_group, signum):
        # No signal in this file may reach the host, including an unexpected
        # fake process identity introduced by a later test change.
        assert process_group in workers, "unregistered fake process group"
        assert signum in {signal.SIGTERM, signal.SIGKILL}
        worker = workers[process_group]
        worker.group_signals.append(signum)
        worker.terminate() if signum == signal.SIGTERM else worker.kill()

    monkeypatch.setattr("stark_convert.chat.os.killpg", killpg)

    def create(*, fatal=False):
        pid = 2_000_000_000 + len(workers)
        worker = BufferedStreamWorker(pid=pid, fatal=fatal)
        workers[pid] = worker
        return worker

    yield create
    for worker in workers.values():
        worker.close()


def test_buffered_worker_shutdown_routes_only_to_its_fake_process_group(buffered_worker):
    first, second = buffered_worker(), buffered_worker()
    first.close()
    assert first.group_signals == [signal.SIGTERM] and first.closed
    assert second.group_signals == [] and not second.closed
    with pytest.raises(AssertionError, match="unregistered fake process group"):
        os.killpg(2_100_000_000, signal.SIGTERM)


def stream_payload(kind):
    request = payload(stream=True)
    if kind == "text":
        request["messages"] = [{"role": "user", "content": "synthetic text"}]
    return request


@pytest.mark.parametrize("kind", ["image", "text"])
@pytest.mark.parametrize("change", ["stop", "replace", "model"])
def test_buffered_stream_delta_is_fenced_after_dispatch(kind, change, buffered_worker):
    worker = buffered_worker()
    desktop = make_desktop(worker)
    replacement = FakeWorker()
    deltas = []

    def change_model():
        if change == "stop":
            desktop.install_progress = {}
            desktop.install_epoch = desktop.converter = None
            desktop.stop(quiet=True)
        elif change == "replace":
            # Object identity must fence delivery even if a caller has not
            # advanced the epoch or changed the public model ID yet.
            with desktop.lock:
                desktop.worker = replacement
        else:
            with desktop.lock:
                desktop.active_model = "replacement-model-id"
                desktop.models.append({**desktop.models[0], "id": desktop.active_model})

    worker.before_delta = change_model
    try:
        with pytest.raises(APIError) as error:
            desktop.api_complete(stream_payload(kind), on_delta=deltas.append)
        assert error.value.code == "request_cancelled"
        assert deltas == []
        assert worker.cancelled_requests
        assert worker.generation_terminals == 1
        assert worker.replies == []
        assert desktop.job is None and desktop.api_queue == []
        if change != "stop":
            current = replacement if change == "replace" else worker
            assert desktop.worker is current and not current.closed
            if kind == "image":
                assert worker.operations[-1] == "discard_image"
                assert list(worker.media.root.iterdir()) == []
            retry = desktop.api_complete(payload())
            assert retry["usage"]["prompt_tokens"] == 17
            assert not current.closed
        else:
            assert desktop.worker is None and worker.closed
    finally:
        worker.close()


def test_old_stream_fatal_cleanup_preserves_replacement_worker(buffered_worker):
    worker = buffered_worker(fatal=True)
    desktop = make_desktop(worker)
    replacement = FakeWorker()

    def replace():
        with desktop.lock:
            desktop.worker = replacement

    worker.before_delta = replace
    try:
        with pytest.raises(APIError) as error:
            desktop.api_complete(payload(stream=True), on_delta=lambda _text: pytest.fail("stale delta"))
        assert error.value.code == "generation_failed"
        assert worker.generation_terminals == 1 and worker.closed
        assert desktop.worker is replacement and not replacement.closed
        assert desktop.api_complete(payload())["usage"]["prompt_tokens"] == 17
        assert not replacement.closed
    finally:
        worker.close()


def test_stream_delivery_holds_epoch_lock_until_callback_returns(buffered_worker):
    worker = buffered_worker()
    desktop = make_desktop(worker)
    desktop.install_progress = {}
    desktop.install_epoch = desktop.converter = None
    entered, release, stop_started, stopped = (threading.Event() for _ in range(4))
    errors = []

    def deliver(_text):
        entered.set()
        assert release.wait(2)

    def complete():
        try:
            desktop.api_complete(payload(stream=True), on_delta=deliver)
        except APIError as error:
            errors.append(error.code)

    def stop():
        stop_started.set()
        desktop.stop(quiet=True)
        stopped.set()

    request = threading.Thread(target=complete)
    stopper = threading.Thread(target=stop)
    try:
        request.start()
        assert entered.wait(2)
        stopper.start()
        assert stop_started.wait(2)
        acquired = desktop.lock.acquire(blocking=False)
        if acquired:
            desktop.lock.release()
        assert not acquired
        assert desktop.epoch == 1 and not stopped.is_set()
        release.set()
        request.join(2)
        stopper.join(2)
        assert not request.is_alive() and not stopper.is_alive()
        assert stopped.is_set() and desktop.epoch == 2
        assert errors in ([], ["request_stopped"])
    finally:
        release.set()
        request.join(2)
        if stopper.ident is not None:
            stopper.join(2)
        worker.close()
