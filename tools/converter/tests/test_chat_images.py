import hashlib
import io
import json
import stat
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from stark_convert import chat
from stark_convert.chat import ChatCancelled, ChatError, ChatImageError, Worker
from stark_convert.media import ImageData, MediaError, PrivateMediaStore
from stark_convert.vision_contract import REPO, REVISION

PNG = b"\x89PNG\r\n\x1a\nsynthetic"
OPTIONS = {"context_tokens": 32, "max_new_tokens": 4}


@pytest.mark.parametrize(
    "code", ["native_generation_failed", "cancel_channel_failed", "invalid_image_preparation_output"]
)
def test_native_failure_is_not_masked_by_concurrent_cancellation(code):
    worker = object.__new__(Worker)
    worker._transaction_lock = threading.RLock()
    worker.process = type("Process", (), {"stdin": io.BytesIO()})()
    worker.cancellation = False
    cancelled = threading.Event()

    def reply(_control=None, _request_id=None):
        cancelled.set()
        return {"error": code}

    worker.read = reply
    with pytest.raises(ChatError) as failure:
        worker.request({"operation": "generate_image"}, cancelled=cancelled)
    assert not isinstance(failure.value, ChatCancelled)


def image_contract():
    from stark_convert.vision_runtime import _ARTIFACT

    return {
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


class Process:
    closed = False

    def poll(self):
        return 0 if self.closed else None


def make_worker(handler=None):
    worker = object.__new__(Worker)
    worker.timeout = 5
    worker._transaction_lock = threading.RLock()
    worker.media = PrivateMediaStore()
    worker.image_input = {**image_contract(), "max_image_tokens": 16}
    worker.images = True
    worker.process = Process()
    worker.sent = []
    worker.staged = []

    def close():
        worker.process.closed = True

    worker.close = close

    def default(payload, _on_delta=None, cancelled=None):
        operation = payload["operation"]
        if operation in {"prepare_image", "generate_image"} and cancelled is not None and cancelled.is_set():
            raise ChatCancelled("Request cancelled.")
        worker.sent.append(operation)
        if operation == "prepare_image":
            reference = payload["image"]
            path = worker.media.root / reference["id"]
            info = path.stat()
            worker.staged.append((path, stat.S_IMODE(info.st_mode), info.st_size))
            return {"prepared_id": "a" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "generate_image":
            return {
                "text": "synthetic response",
                "input_tokens": 12,
                "output_tokens": 1,
                "finish_reason": "eos",
            }
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic response"}

    def request(payload, on_delta=None, cancelled=None):
        with worker._transaction_lock:
            if handler is None:
                return default(payload, on_delta, cancelled)
            return handler(payload, on_delta, cancelled)

    worker.request = request
    return worker


def perform(worker, cancelled=None):
    return worker.request_image(
        "synthetic rendered prompt",
        ImageData(PNG, "image/png"),
        OPTIONS,
        cancelled=cancelled,
    )


def test_image_request_orders_prepare_generate_discard_and_removes_staged_file():
    worker = make_worker()
    root = worker.media.root
    try:
        reply = perform(worker)
        assert reply["finish_reason"] == "eos"
        assert worker.sent == ["prepare_image", "generate_image", "discard_image"]
        assert len(worker.staged) == 1
        path, mode, size = worker.staged[0]
        assert mode == 0o600
        assert size == len(PNG)
        assert path.parent == root
        assert not path.exists()
        assert list(root.iterdir()) == []
    finally:
        worker.media.close()


@pytest.mark.parametrize(
    ("failed_operation", "worker_error", "expected_type", "expected_code"),
    [
        ("prepare_image", "invalid_image_request", ChatImageError, "invalid_image_request"),
        ("generate_image", "image_read_failed", ChatImageError, "image_read_failed"),
        ("generate_image", "native_operation_failed", ChatError, None),
    ],
)
def test_recoverable_and_fatal_image_failures_discard_and_clean(
    failed_operation, worker_error, expected_type, expected_code
):
    def handler(payload, _on_delta, _cancelled):
        operation = payload["operation"]
        worker.sent.append(operation)
        if operation == failed_operation:
            return {"error": worker_error}
        if operation == "prepare_image":
            return {"prepared_id": "b" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic response"}

    worker = make_worker(handler)
    root = worker.media.root
    try:
        with pytest.raises(expected_type) as error:
            perform(worker)
        if expected_code is not None:
            assert error.value.code == expected_code
        if expected_code is not None:
            assert worker.sent[-1] == "discard_image"
            assert worker.process.poll() is None
        else:
            assert "discard_image" not in worker.sent
            assert worker.process.poll() is not None
        assert not list(root.iterdir())
    finally:
        worker.media.close()


@pytest.mark.parametrize("cancel_after_prepare", [False, True])
def test_cancellation_discards_prepared_state_and_removes_staged_image(cancel_after_prepare):
    cancelled = threading.Event()

    def handler(payload, _on_delta, control):
        operation = payload["operation"]
        if operation in {"prepare_image", "generate_image"} and control is not None and control.is_set():
            raise ChatCancelled("Request cancelled.")
        worker.sent.append(operation)
        if operation == "prepare_image":
            if cancel_after_prepare:
                cancelled.set()
            return {"prepared_id": "c" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic response"}

    worker = make_worker(handler)
    root = worker.media.root
    if not cancel_after_prepare:
        cancelled.set()
    try:
        with pytest.raises(ChatCancelled):
            perform(worker, cancelled)
        expected = ["discard_image"] if not cancel_after_prepare else ["prepare_image", "discard_image"]
        assert worker.sent == expected
        assert not list(root.iterdir())
    finally:
        worker.media.close()


def test_unexpected_prepare_reply_is_fatal_and_closes_without_discard():
    def handler(payload, _on_delta, _cancelled):
        operation = payload["operation"]
        worker.sent.append(operation)
        if operation == "prepare_image":
            return {"prepared_id": "bad-id", "input_tokens": 12, "image_tokens": 4}
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic response"}

    worker = make_worker(handler)
    root = worker.media.root
    try:
        with pytest.raises(ChatError, match="restart the model"):
            perform(worker)
        assert worker.sent == ["prepare_image"]
        assert worker.process.poll() is not None
        assert not list(root.iterdir())
    finally:
        worker.media.close()


def test_worker_lock_keeps_other_requests_out_of_prepare_generate_discard():
    entered_prepare = threading.Event()
    release_prepare = threading.Event()
    other_started = threading.Event()
    other_finished = threading.Event()

    def handler(payload, _on_delta, _cancelled):
        operation = payload["operation"]
        worker.sent.append(operation)
        if operation == "prepare_image":
            entered_prepare.set()
            if not release_prepare.wait(3):
                raise AssertionError("test did not release image preparation")
            return {"prepared_id": "d" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "generate_image":
            return {
                "text": "synthetic response",
                "input_tokens": 12,
                "output_tokens": 1,
                "finish_reason": "eos",
            }
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic text result"}

    worker = make_worker(handler)
    image_errors = []
    text_errors = []

    def image_request():
        try:
            perform(worker)
        except Exception as error:
            image_errors.append(type(error).__name__)

    def competing_request():
        other_started.set()
        try:
            worker.request({"operation": "text_request"})
        except Exception as error:
            text_errors.append(type(error).__name__)
        finally:
            other_finished.set()

    image_thread = threading.Thread(target=image_request)
    text_thread = threading.Thread(target=competing_request)
    root = worker.media.root
    try:
        image_thread.start()
        assert entered_prepare.wait(2)
        text_thread.start()
        assert other_started.wait(2)
        time.sleep(0.05)
        assert not other_finished.is_set()
        release_prepare.set()
        image_thread.join(3)
        text_thread.join(3)
        assert not image_thread.is_alive() and not text_thread.is_alive()
        assert image_errors == [] and text_errors == []
        assert worker.sent == ["prepare_image", "generate_image", "discard_image", "text_request"]
        assert not list(root.iterdir())
    finally:
        release_prepare.set()
        image_thread.join(3)
        text_thread.join(3)
        worker.media.close()


@pytest.mark.parametrize(
    "malformed_fields",
    [
        {"text": None},
        {"input_tokens": 11},
        {"output_tokens": True},
        {"output_tokens": 5},
        {"finish_reason": "unknown"},
    ],
)
def test_malformed_image_generation_reply_is_fatal_and_cleans(malformed_fields):
    def handler(payload, _on_delta, _cancelled):
        operation = payload["operation"]
        worker.sent.append(operation)
        if operation == "prepare_image":
            return {"prepared_id": "f" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "generate_image":
            return {
                "text": "synthetic response",
                "input_tokens": 12,
                "output_tokens": 1,
                "finish_reason": "eos",
                **malformed_fields,
            }
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic response"}

    worker = make_worker(handler)
    root = worker.media.root
    try:
        with pytest.raises(ChatError, match="restart the model"):
            perform(worker)
        assert worker.sent == ["prepare_image", "generate_image"]
        assert worker.process.poll() is not None
        assert not list(root.iterdir())
    finally:
        worker.media.close()


@pytest.mark.parametrize("discarded", ["yes", None])
def test_malformed_discard_reply_is_fatal_and_staged_file_is_removed(discarded):
    def handler(payload, _on_delta, _cancelled):
        operation = payload["operation"]
        worker.sent.append(operation)
        if operation == "prepare_image":
            return {"prepared_id": "e" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "discard_image":
            return {"discarded": discarded}
        return {
            "text": "synthetic response",
            "input_tokens": 12,
            "output_tokens": 1,
            "finish_reason": "eos",
        }

    worker = make_worker(handler)
    root = worker.media.root
    try:
        with pytest.raises(ChatError, match="cleanup failed"):
            perform(worker)
        assert worker.sent == ["prepare_image", "generate_image", "discard_image"]
        assert worker.process.poll() is not None
        assert not list(root.iterdir())
    finally:
        worker.media.close()


@pytest.mark.parametrize("first_failure", ["cancel", "recoverable", "oversized"])
def test_aligned_image_failure_discards_and_allows_valid_request_b(first_failure):
    from stark_convert.chat import ChatRequestTooLarge

    failures = [first_failure]

    def handler(payload, _on_delta, _cancelled):
        operation = payload["operation"]
        worker.sent.append(operation)
        if operation == "prepare_image":
            if failures:
                reason = failures.pop()
                if reason == "cancel":
                    raise ChatCancelled("Request cancelled.")
                if reason == "oversized":
                    raise ChatRequestTooLarge("Synthetic frame rejected before write.")
                return {"error": "invalid_image_request"}
            return {"prepared_id": "a" * 32, "input_tokens": 12, "image_tokens": 4}
        if operation == "discard_image":
            return {"discarded": True}
        return {"text": "synthetic", "input_tokens": 12, "output_tokens": 1, "finish_reason": "eos"}

    worker = make_worker(handler)
    try:
        with pytest.raises((ChatCancelled, ChatImageError, ChatRequestTooLarge)):
            perform(worker)
        assert worker.sent == ["prepare_image", "discard_image"]
        assert worker.process.poll() is None
        assert perform(worker)["finish_reason"] == "eos"
    finally:
        worker.media.close()


@pytest.mark.parametrize(
    "scenario", ["silent_prepare", "invalid_protocol", "stream", "blocked_write", "silent_discard"]
)
def test_image_transaction_real_pipes_have_one_deadline_and_fatal_errors_skip_discard(tmp_path, scenario):
    executable = tmp_path / "synthetic worker"
    operations = tmp_path / "operations.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,sys,time\n"
        f"scenario={scenario!r}\n"
        f"operations={str(operations)!r}\n"
        "print(json.dumps({'ready':True}),flush=True)\n"
        "if scenario=='blocked_write': time.sleep(30);sys.exit(0)\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line);op=request['operation']\n"
        " with open(operations,'a') as out:out.write(json.dumps(op)+'\\n')\n"
        " if op=='prepare_image':\n"
        "  if scenario=='silent_prepare':time.sleep(30)\n"
        "  if scenario=='silent_discard':time.sleep(.12)\n"
        "  if scenario=='invalid_protocol':print('not-json',flush=True);time.sleep(30)\n"
        "  print(json.dumps({'prepared_id':'a'*32,'input_tokens':12,'image_tokens':4}),flush=True)\n"
        " elif op=='generate_image':\n"
        "  if scenario=='silent_discard':\n"
        "   time.sleep(.03)\n"
        "   print(json.dumps({'text':'','input_tokens':12,'output_tokens':0,'finish_reason':'eos'}),flush=True)\n"
        "   continue\n"
        "  for _ in range(60):\n"
        "   print(json.dumps({'event':'delta','text':'x'}),flush=True);time.sleep(.02)\n"
        "  print(json.dumps({'text':'x'*60,'input_tokens':12,'output_tokens':4,'finish_reason':'eos'}),flush=True)\n"
        " else:\n"
        "  if scenario=='silent_discard':time.sleep(30)\n"
        "  print(json.dumps({'discarded':True}),flush=True)\n"
    )
    executable.chmod(0o755)
    worker = Worker(executable, tmp_path, 5)
    try:
        worker.read()
        worker.media = PrivateMediaStore()
        staging = worker.media.root
        worker.image_input = {"max_image_tokens": 16}
        worker.timeout = 0.2
        started = time.monotonic()
        with pytest.raises(ChatError, match="Invalid worker reply" if scenario == "invalid_protocol" else "timed out"):
            worker.request_image(
                "x" * (512 * 1024 if scenario == "blocked_write" else 1),
                ImageData(PNG, "image/png"),
                OPTIONS,
                on_delta=lambda _text: None,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 0.35
        if scenario != "invalid_protocol":
            assert elapsed >= worker.timeout * 0.8
        assert worker.process.poll() is not None
        assert not staging.exists()
        sent = [json.loads(line) for line in operations.read_text().splitlines()] if operations.exists() else []
        assert ("discard_image" in sent) == (scenario == "silent_discard")
        assert len(sent) == (
            0
            if scenario == "blocked_write"
            else 2
            if scenario == "stream"
            else 3
            if scenario == "silent_discard"
            else 1
        )
    finally:
        worker.close()


def test_fatal_image_error_survives_staging_and_process_cleanup_failures():
    def handler(payload, _on_delta, _cancelled):
        worker.sent.append(payload["operation"])
        raise ChatError("Synthetic original fatal failure.")

    worker = make_worker(handler)
    stage = worker.media.stage

    @contextmanager
    def failed_stage(image):
        with stage(image) as reference:
            try:
                yield reference
            finally:
                raise MediaError("synthetic_cleanup_failure")

    def failed_close():
        worker.process.closed = True
        raise OSError

    worker.media.stage = failed_stage
    worker.close = failed_close
    try:
        with pytest.raises(ChatError, match="Synthetic original fatal failure"):
            perform(worker)
        assert worker.sent == ["prepare_image"]
        assert worker.process.poll() is not None
        assert not list(worker.media.root.iterdir())
    finally:
        worker.media.close()


def test_worker_replacement_between_launch_and_ready_cannot_qualify_images(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"type": "Qwen35VisionLanguageModelConfig"}))
    executable = tmp_path / "worker"
    loaded = tmp_path / "worker-loaded"
    executable.write_text(
        f"#!{sys.executable}\n"
        f"import json,time,pathlib; pathlib.Path({str(loaded)!r}).touch()\n"
        "time.sleep(0.25)\n"
        "print(json.dumps({'ready':True,'image_input':{}}),flush=True)\n"
        "for _line in __import__('sys').stdin: pass\n"
    )
    executable.chmod(0o755)
    launched_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    replacement = b"replacement executable bytes"
    replacement_hash = hashlib.sha256(replacement).hexdigest()
    observed = {}

    monkeypatch.setattr(chat, "image_input_contract", lambda _value: image_contract())

    def qualified(_contract, path, expected_worker_sha256=None):
        observed["expected_worker_sha256"] = expected_worker_sha256
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        # Model the catalog containing the replacement file hash.
        return current_hash == replacement_hash and (
            expected_worker_sha256 is None or expected_worker_sha256 == current_hash
        )

    monkeypatch.setattr(chat, "qualified_image_worker", qualified)
    worker = Worker(executable, model, 5)
    staging = worker.media.root
    try:
        # The child has loaded the original script and waits before readiness;
        # the path now names a different, catalog-matching binary.
        deadline = time.monotonic() + 2
        while not loaded.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert loaded.exists()
        executable.write_bytes(replacement)
        assert worker.read().get("ready") is True
        assert observed["expected_worker_sha256"] == launched_hash
        assert worker.images is False
    finally:
        worker.close()
    assert not staging.exists()


def test_worker_missing_launch_hash_fails_closed_before_catalog_lookup(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"type": "Qwen35VisionLanguageModelConfig"}))
    executable = tmp_path / "worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        "print(json.dumps({'ready':True,'image_input':{}}),flush=True)\n"
        "for _line in sys.stdin: pass\n"
    )
    executable.chmod(0o755)
    lookup = {"called": False}

    def fail_hash(_path):
        raise OSError

    def qualified(*_args, **_kwargs):
        lookup["called"] = True
        return True

    monkeypatch.setattr(chat, "file_hash", fail_hash)
    monkeypatch.setattr(chat, "image_input_contract", lambda _value: image_contract())
    monkeypatch.setattr(chat, "qualified_image_worker", qualified)
    worker = Worker(executable, model, 5)
    staging = worker.media.root
    try:
        assert worker.read().get("ready") is True
        assert worker._launch_worker_sha256 is None
        assert worker.images is False
        assert lookup["called"] is False
    finally:
        worker.close()
    assert not staging.exists()


def test_worker_close_reaps_and_cleans_after_exit_between_poll_and_terminate():
    import os

    class ExitedDuringTerminate:
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        reaped = False

        def poll(self):
            return None

        def terminate(self):
            raise ProcessLookupError

        def wait(self, timeout=None):
            self.reaped = True
            return 0

    worker = object.__new__(Worker)
    worker.process = ExitedDuringTerminate()
    worker.media = PrivateMediaStore()
    root = worker.media.root
    read_fd, worker.cancel_fd = os.pipe()
    cancel_fd = worker.cancel_fd
    try:
        worker.close()
        assert worker.process.reaped
        assert worker.process.stdin.closed and worker.process.stdout.closed
        assert worker.cancel_fd is None
        assert worker.media is None and not root.exists()
        with pytest.raises(OSError):
            os.fstat(cancel_fd)
    finally:
        os.close(read_fd)
