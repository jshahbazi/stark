"""Opaque Swift attachment claims and conversation ownership, using synthetic bytes."""

import io
import json
import os
import threading
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from stark_convert.chat import ChatImageError
from stark_convert.desktop import Desktop, DesktopError
from stark_convert.desktop_media import DesktopAttachments
from stark_convert.media import MAX_IMAGE_BYTES, MediaError

PNG = b"\x89PNG\r\n\x1a\nsynthetic"
JPEG = b"\xff\xd8\xffsynthetic\xff\xd9"
IDENTIFIER = "a" * 32


def stage(root, data=PNG, *, name=IDENTIFIER, media_type="image/png", mode=0o600):
    path = root / name
    path.write_bytes(data)
    path.chmod(mode)
    return {"id": name, "media_type": media_type, "byte_length": len(data)}


@pytest.fixture
def attachments(tmp_path):
    root = (tmp_path / "attachments").resolve()
    root.mkdir(mode=0o700)
    store = DesktopAttachments(root)
    yield store, root
    store.close()


@pytest.mark.parametrize("data,media_type", [(PNG, "image/png"), (JPEG, "image/jpeg")])
def test_claim_unlinks_before_return_and_replay_is_rejected(attachments, data, media_type):
    store, root = attachments
    reference = stage(root, data, media_type=media_type)
    image = store.claim(reference)
    assert image.data == data and image.media_type == media_type
    assert not list(root.iterdir())
    with pytest.raises(MediaError, match="Image data is invalid"):
        store.claim(reference)
    assert not list(root.iterdir())


def test_exact_byte_limit_is_admitted_before_pixel_decode(attachments):
    store, root = attachments
    data = PNG + b"x" * (MAX_IMAGE_BYTES - len(PNG))
    assert len(store.claim(stage(root, data)).data) == MAX_IMAGE_BYTES
    assert not list(root.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        {"id": "../" + IDENTIFIER},
        {"id": "/" + IDENTIFIER},
        {"id": "A" * 32},
        {"id": IDENTIFIER + "\n"},
        {"id": 1},
        {"media_type": "image/gif"},
        {"media_type": []},
        {"byte_length": True},
        {"byte_length": 0},
        {"byte_length": -1},
        {"byte_length": MAX_IMAGE_BYTES + 1},
        {"path": "untrusted"},
    ],
)
def test_untrusted_reference_rejected_before_open(attachments, change, monkeypatch):
    store, root = attachments
    reference = stage(root)
    reference.update(change)
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("unexpected file open"))
    with pytest.raises(MediaError):
        store.claim(reference)


@pytest.mark.parametrize("reference", [None, [], {}, "opaque-id"])
def test_non_reference_is_rejected(attachments, reference):
    with pytest.raises(MediaError):
        attachments[0].claim(reference)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "public-file"])
def test_claim_rejects_non_private_regular_single_link(attachments, tmp_path, kind):
    store, root = attachments
    outside = tmp_path / "outside"
    outside.write_bytes(PNG)
    outside.chmod(0o600)
    path = root / IDENTIFIER
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    elif kind == "directory":
        path.mkdir(mode=0o700)
    else:
        stage(root, mode=0o644)
    with pytest.raises(MediaError):
        store.claim({"id": IDENTIFIER, "media_type": "image/png", "byte_length": len(PNG)})
    assert outside.read_bytes() == PNG


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_partial_or_mismatched_write_is_rejected_then_cleaned(attachments, size_delta):
    store, root = attachments
    reference = stage(root)
    reference["byte_length"] += size_delta
    with pytest.raises(MediaError):
        store.claim(reference)
    store.clear()
    assert not list(root.iterdir())


@pytest.mark.parametrize("data,media_type", [(PNG, "image/jpeg"), (JPEG, "image/png"), (b"invalid", "image/png")])
def test_mime_mismatch_is_consumed_without_retaining_file(attachments, data, media_type):
    store, root = attachments
    with pytest.raises(MediaError):
        store.claim(stage(root, data, media_type=media_type))
    assert not list(root.iterdir())


@pytest.mark.parametrize("mutation", ["append", "replace", "chmod"])
def test_read_mutation_is_rejected(attachments, monkeypatch, mutation):
    store, root = attachments
    reference = stage(root)
    original_read = os.read
    mutated = False

    def changing_read(descriptor, count):
        nonlocal mutated
        result = original_read(descriptor, count)
        if not mutated:
            mutated = True
            path = root / IDENTIFIER
            if mutation == "append":
                with path.open("ab") as output:
                    output.write(b"x")
            elif mutation == "chmod":
                path.chmod(0o644)
            else:
                path.unlink()
                stage(root)
        return result

    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(MediaError):
        store.claim(reference)


def test_pinned_root_does_not_follow_replacement_directory(attachments):
    store, root = attachments
    pinned = root.with_name("pinned")
    root.rename(pinned)
    root.mkdir(mode=0o700)
    stage(root, PNG + b"replacement")
    reference = stage(pinned)
    assert store.claim(reference).data == PNG
    assert not list(pinned.iterdir())
    assert (root / IDENTIFIER).exists()
    store.close()
    assert (root / IDENTIFIER).exists()


def test_clear_preserves_unknown_and_non_private_entries(attachments, tmp_path):
    store, root = attachments
    stage(root)
    unknown = root / "unknown"
    unknown.write_bytes(b"synthetic")
    unknown.chmod(0o600)
    directory = root / ("b" * 32)
    directory.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.write_bytes(PNG)
    outside.chmod(0o600)
    (root / ("c" * 32)).symlink_to(outside)
    os.link(outside, root / ("d" * 32))
    stage(root, name="e" * 32, mode=0o644)
    store.clear()
    assert not (root / IDENTIFIER).exists()
    assert len(list(root.iterdir())) == 5
    assert outside.read_bytes() == PNG


@pytest.mark.parametrize("kind", ["nonempty", "public", "symlink", "relative", "file"])
def test_root_requires_empty_canonical_private_directory(tmp_path, kind):
    root = (tmp_path / "root").resolve()
    root.mkdir(mode=0o700)
    selected = root
    if kind == "nonempty":
        stage(root)
    elif kind == "public":
        root.chmod(0o755)
    elif kind == "symlink":
        selected = tmp_path / "link"
        selected.symlink_to(root, target_is_directory=True)
    elif kind == "relative":
        selected = Path("relative")
    else:
        root.rmdir()
        root.write_bytes(b"synthetic")
    with pytest.raises(MediaError, match="Image staging is unavailable"):
        DesktopAttachments(selected)


def test_close_is_idempotent_and_prevents_new_claim(attachments):
    store, root = attachments
    reference = stage(root)
    store.close()
    store.close()
    assert not list(root.iterdir())
    with pytest.raises(MediaError):
        store.claim(reference)


def test_cleanup_failure_is_content_free_and_close_still_reaps_descriptor(attachments, monkeypatch):
    store, root = attachments
    stage(root)
    descriptor = store._fd

    def unavailable(_descriptor):
        raise PermissionError("private staging path")

    monkeypatch.setattr(os, "listdir", unavailable)
    with pytest.raises(MediaError, match="^Image staging is unavailable\\.$"):
        store.clear()
    with pytest.raises(MediaError, match="^Image staging is unavailable\\.$"):
        store.close()
    assert store._fd is None
    with pytest.raises(OSError):
        os.fstat(descriptor)


class ImageWorker:
    images = True
    streaming = True
    stop_sequences = True

    def __init__(self):
        self.calls = []
        self.next_error = None
        self.callback = None

    def request_image(self, prompt, image, options, **_kwargs):
        self.calls.append((prompt, image, options))
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        if self.callback is not None:
            self.callback()
        return {"text": "synthetic answer", "input_tokens": 17, "output_tokens": 2, "finish_reason": "eos"}


@pytest.fixture
def desktop(tmp_path, attachments):
    store, root = attachments
    value = Desktop(Path("/synthetic/worker"), tmp_path / "library", output=io.StringIO())
    value.attachments = store
    value.worker = ImageWorker()
    value.active_model = "synthetic-model"
    value.active_mode = "chat"
    value.codec = {"default_system_prompt": None}
    value.template = ImmutableSandboxedEnvironment().from_string(
        "{% for m in messages %}{{m.role}}:"
        "{% if m.content is string %}{{m.content}}{% else %}"
        "{% for p in m.content %}{% if p.type == 'text' %}{{p.text}}{% else %}[image]{% endif %}{% endfor %}"
        "{% endif %};{% endfor %}assistant:"
    )
    value.reset()
    return value, root


def event_names(desktop):
    return [json.loads(line)["event"] for line in desktop.output.getvalue().splitlines()]


def test_initial_and_followup_reuse_exactly_one_placeholder(desktop):
    controller, root = desktop
    controller.generate(controller.epoch, "first", stage(root))
    controller.generate(controller.epoch, "followup", stage(root))
    assert len(controller.worker.calls) == 2
    assert all(call[0].count("[image]") == 1 for call in controller.worker.calls)
    assert len(controller.messages) == 4
    assert not list(root.iterdir())
    assert event_names(controller).count("reply") == 2
    assert event_names(controller).count("image_progress") == 4


def test_different_followup_image_is_rejected_without_history_change(desktop):
    controller, root = desktop
    controller.generate(controller.epoch, "first", stage(root))
    history = list(controller.messages)
    with pytest.raises(DesktopError, match="Start a new chat"):
        controller.generate(controller.epoch, "followup", stage(root, PNG + b"changed"))
    assert controller.messages == history
    assert len(controller.worker.calls) == 1
    assert not list(root.iterdir())


def test_missing_followup_image_is_rejected_before_inference(desktop):
    controller, root = desktop
    controller.generate(controller.epoch, "first", stage(root))
    with pytest.raises(DesktopError, match="Resupply"):
        controller.generate(controller.epoch, "followup")
    assert len(controller.worker.calls) == 1


def test_recoverable_failure_does_not_commit_image_or_history_and_retry_passes(desktop):
    controller, root = desktop
    controller.worker.next_error = ChatImageError("invalid_image_request")
    with pytest.raises(DesktopError, match="could not be prepared"):
        controller.generate(controller.epoch, "first", stage(root))
    assert not controller.messages and controller.image_identity is None
    assert not list(root.iterdir())
    controller.generate(controller.epoch, "retry", stage(root))
    assert len(controller.messages) == 2 and controller.image_identity is not None


def test_reset_allows_new_image_and_clears_previous_placeholder(desktop):
    controller, root = desktop
    controller.generate(controller.epoch, "first", stage(root))
    controller.reset()
    assert not controller.messages and controller.image_identity is None
    controller.generate(controller.epoch, "new", stage(root, PNG + b"new"))
    assert controller.worker.calls[-1][0].count("[image]") == 1
    assert len(controller.messages) == 2


def test_unqualified_worker_is_rejected_before_claim(desktop):
    controller, root = desktop
    controller.worker.images = False
    reference = stage(root)
    with pytest.raises(DesktopError, match="not qualified"):
        controller.generate(controller.epoch, "first", reference)
    assert not controller.worker.calls and (root / IDENTIFIER).exists()
    assert controller.settings_summary()["images"] is False


def test_capability_requires_paired_attachment_store(desktop):
    controller, _root = desktop
    assert controller.settings_summary()["images"] is True
    controller.attachments = None
    assert controller.settings_summary()["images"] is False
    assert controller.settings_summary()["max_image_bytes"] == MAX_IMAGE_BYTES


def test_stale_epoch_does_not_claim_or_publish(desktop):
    controller, root = desktop
    reference = stage(root)
    controller.generate(controller.epoch - 1, "first", reference)
    assert not controller.worker.calls and not event_names(controller)
    assert (root / IDENTIFIER).exists()


def test_model_epoch_change_during_inference_does_not_commit_or_publish(desktop):
    controller, root = desktop
    controller.worker.callback = lambda: setattr(controller, "epoch", controller.epoch + 1)
    controller.generate(controller.epoch, "first", stage(root))
    assert not controller.messages and controller.image_identity is None
    assert "reply" not in event_names(controller)
    assert not list(root.iterdir())


def test_worker_close_releases_conversation_identity(desktop):
    controller, root = desktop
    controller.generate(controller.epoch, "first", stage(root))
    controller.worker = None
    controller.close_worker()
    assert not controller.messages and controller.image_identity is None
    assert controller.active_model is None


def test_stop_cleans_unclaimed_partial_file(desktop):
    controller, root = desktop
    stage(root)
    controller.worker = None
    controller.stop(quiet=True)
    assert not list(root.iterdir())
    assert controller.image_identity is None


def test_stop_converts_cleanup_failure_to_fixed_desktop_error(desktop, monkeypatch):
    controller, _root = desktop
    controller.worker = None

    def unavailable():
        raise MediaError("media_store_unavailable")

    monkeypatch.setattr(controller.attachments, "clear", unavailable)
    with pytest.raises(DesktopError, match="^Cannot clear private image storage\\. Quit and reopen Stark\\.$"):
        controller.stop(quiet=True)


def test_image_only_dispatch_is_admitted_and_empty_text_only_is_rejected(desktop):
    controller, root = desktop
    reference = stage(root)
    controller.dispatch({"command": "generate", "text": "", "image": reference})
    controller.job.join(timeout=2)
    assert not controller.job or not controller.job.is_alive()
    assert len(controller.worker.calls) == 1
    with pytest.raises(DesktopError, match="Invalid app command"):
        controller.dispatch({"command": "generate", "text": ""})


@pytest.mark.parametrize("event", ["ready", "reply", "decision", "error", "installed", "status"])
def test_terminal_event_waits_for_operation_cleanup_and_free_admission(desktop, event):
    controller, _root = desktop
    at_tail, release = threading.Event(), threading.Event()
    observations = []

    class Observer(io.StringIO):
        def write(self, value):
            result = super().write(value)
            parsed = json.loads(value)
            if parsed["event"] == event and (event != "status" or parsed.get("state") == "ready"):
                observations.append((controller.job is None, controller.install_epoch is None))
            return result

    controller.output = Observer()

    def operation(epoch):
        controller.emit(event, epoch, **({"state": "ready"} if event == "status" else {}))
        at_tail.set()
        assert release.wait(2)

    try:
        controller.start(operation, "generating", "Generating")
        assert at_tail.wait(2)
        current = controller.job
        assert current is not None and current.is_alive()
        assert observations == []
    finally:
        release.set()
        current = controller.job
        if current is not None:
            current.join(timeout=2)
    assert observations == [(True, True)]


def test_ready_observer_can_immediately_start_followup_without_busy_error(desktop):
    controller, _root = desktop
    ready, second_finished = threading.Event(), threading.Event()

    class Observer(io.StringIO):
        def write(self, value):
            result = super().write(value)
            parsed = json.loads(value)
            if parsed["event"] == "status" and parsed.get("state") == "ready":
                ready.set()
            return result

    controller.output = Observer()

    def first(epoch):
        controller.status("ready", "Ready", epoch)

    def second(_epoch):
        second_finished.set()

    controller.start(first, "generating", "Generating")
    assert ready.wait(2)
    controller.start(second, "generating", "Generating")
    assert second_finished.wait(2)
    current = controller.job
    if current is not None:
        current.join(timeout=2)
    assert not any(
        event.get("message") == "Wait for the current operation or choose Stop."
        for event in (json.loads(line) for line in controller.output.getvalue().splitlines())
    )


def test_install_completion_follows_finally_cleanup(desktop):
    controller, _root = desktop
    at_tail, release = threading.Event(), threading.Event()
    observations = []

    class Observer(io.StringIO):
        def write(self, value):
            result = super().write(value)
            parsed = json.loads(value)
            if parsed["event"] == "status" and parsed.get("state") == "idle":
                observations.append((controller.job is None, controller.install_epoch is None))
            return result

    controller.output = Observer()

    def installation(epoch, _kind):
        try:
            controller.status("idle", "Model installed", epoch)
        finally:
            at_tail.set()
            assert release.wait(2)

    try:
        controller.start(installation, "installing", "Installing", "local")
        assert at_tail.wait(2)
        assert controller.install_epoch == controller.epoch
        assert observations == []
    finally:
        release.set()
        current = controller.job
        if current is not None:
            current.join(timeout=2)
    assert observations == [(True, True)]


def test_stopped_operation_cannot_publish_buffered_terminal_events(desktop):
    controller, _root = desktop
    at_tail, release = threading.Event(), threading.Event()

    def operation(epoch):
        controller.emit("reply", epoch, text="synthetic answer")
        controller.status("ready", "Ready", epoch)
        at_tail.set()
        assert release.wait(2)

    try:
        controller.start(operation, "generating", "Generating")
        assert at_tail.wait(2)
        with controller.lock:
            controller.epoch += 1
        assert "reply" not in event_names(controller)
    finally:
        release.set()
        current = controller.job
        if current is not None:
            current.join(timeout=2)
    assert "reply" not in event_names(controller)
    assert not any(
        parsed.get("event") == "status" and parsed.get("state") == "ready"
        for parsed in (json.loads(line) for line in controller.output.getvalue().splitlines())
    )


def test_completion_without_explicit_epoch_is_fenced_after_stop(desktop):
    controller, _root = desktop
    at_tail, release = threading.Event(), threading.Event()

    def operation(_epoch):
        controller.emit("reply", text="synthetic answer")
        at_tail.set()
        assert release.wait(2)

    try:
        controller.start(operation, "generating", "Generating")
        assert at_tail.wait(2)
        with controller.lock:
            controller.epoch += 1
    finally:
        release.set()
        current = controller.job
        if current is not None:
            current.join(timeout=2)
    assert "reply" not in event_names(controller)


def test_progress_and_other_thread_events_are_not_deferred(desktop):
    controller, _root = desktop
    at_tail, release = threading.Event(), threading.Event()

    def operation(epoch):
        controller.emit("progress", epoch, phase="conversion", state="running")
        controller.emit("models", epoch, models=[])
        controller.emit("reply", epoch, text="synthetic answer")
        at_tail.set()
        assert release.wait(2)

    try:
        controller.start(operation, "generating", "Generating")
        assert at_tail.wait(2)
        assert event_names(controller) == ["status", "progress", "models"]
        controller.emit("error", message="Synthetic unrelated caller validation.")
        assert event_names(controller) == ["status", "progress", "models", "error"]
    finally:
        release.set()
        current = controller.job
        if current is not None:
            current.join(timeout=2)
    assert event_names(controller) == ["status", "progress", "models", "error", "reply"]


def test_completion_notifies_waiters_and_preserves_event_order(desktop):
    controller, _root = desktop
    at_tail, release, waiter_ready, notified = (threading.Event() for _ in range(4))

    def operation(epoch):
        controller.emit("ready", epoch)
        controller.status("ready", "Ready", epoch)
        at_tail.set()
        assert release.wait(2)

    def wait_for_admission():
        with controller.api_condition:
            waiter_ready.set()
            if controller.api_condition.wait(timeout=2) and controller.job is None:
                notified.set()

    waiter = threading.Thread(target=wait_for_admission)
    try:
        controller.start(operation, "loading", "Loading")
        assert at_tail.wait(2)
        waiter.start()
        assert waiter_ready.wait(2)
    finally:
        release.set()
        current = controller.job
        if current is not None:
            current.join(timeout=2)
        if waiter.ident is not None:
            waiter.join(timeout=2)
    assert notified.is_set()
    assert event_names(controller) == ["status", "ready", "status"]
