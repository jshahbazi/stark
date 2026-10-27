"""Local execution receipts and automatic checks using model-free workers."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark_convert import image_checks
from stark_convert.chat import ChatError, ChatImageError, Worker
from stark_convert.vision_runtime import FAMILY_PROFILE


def family_contract(**changes):
    value = {
        "version": "mediaref-v1",
        "transport": "owner-private-file-v1",
        "qualified": False,
        "family_profile": FAMILY_PROFILE,
        "family_version": 1,
        "model_source": "synthetic-owner/merged-model",
        "model_revision": None,
        "artifact_schema_version": 3,
        "artifact_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "processor_sha256": "d" * 64,
        "template_sha256": "e" * 64,
        "runtime_identity": {
            "backend": "metal",
            "device_name": "Apple Synthetic GPU",
            "apple_gpu_family": 1008,
            "gpu_core_count": 32,
            "supports_mxu": False,
        },
        "max_images": 1,
        "max_compressed_bytes": 4 * 1024 * 1024,
        "max_source_pixels": 32_000_000,
        "max_processed_pixels": 1_048_576,
        "max_patches": 4096,
        "max_image_tokens": 1024,
        "operations": ["prepare_image", "generate_image", "discard_image"],
        "formats": ["image/png", "image/jpeg"],
    }
    value.update(changes)
    return value


def reply(**changes):
    value = {"text": "synthetic answer", "input_tokens": 90, "output_tokens": 2, "finish_reason": "length"}
    value.update(changes)
    return value


class FakeWorker:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.executable = root / "worker"
        self.executable.write_bytes(b"synthetic worker")
        self._launch_worker_sha256 = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        self.model_path = root / "model"
        self.model_path.mkdir()
        self.image_input = family_contract()
        self.timeout = 3
        self.calls = []
        self.closed = False
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
        media_root = root / "media"
        media_root.mkdir()
        self.media = SimpleNamespace(root=media_root)
        self.image_check_dir = None
        self.text_reply = reply(input_tokens=7)
        self.token_reply = {"token_count": 27, "token_ids": [1] * 27}
        self.image_replies = [reply(), reply()]
        self.image_counts = [64, 64]
        self.reject_invalid = True
        self.after_image = None

    def request(self, payload):
        if payload["operation"] == "tokenize":
            assert payload["add_special_tokens"] is False
            self.calls.append("tokenize")
            return self.token_reply
        assert payload["operation"] == "generate"
        assert payload["context_tokens"] == 2048 and payload["max_new_tokens"] == 8
        assert self._image_deadline == self._verification_deadline
        self.calls.append("text")
        return self.text_reply

    def request_image(self, _prompt, image, _options):
        if len(image.data) == 9:
            self.calls.append("invalid")
            if self.reject_invalid:
                raise ChatImageError("invalid_image_request")
            return reply()
        self.calls.append("image")
        self._last_image_tokens = self.image_counts.pop(0)
        result = self.image_replies.pop(0)
        if self.after_image is not None:
            self.after_image(self)
        return result


@pytest.fixture
def worker(tmp_path, monkeypatch):
    instance = FakeWorker(tmp_path / "worker-fixture")
    original_hash = image_checks.file_hash

    def stable_checker_hash(path):
        path = Path(path)
        return original_hash(path) if path == instance.executable else "f" * 64

    monkeypatch.setattr(image_checks, "file_hash", stable_checker_hash)
    monkeypatch.setattr(image_checks, "_current_os_version", lambda: "26.7.1")

    class Template:
        def render(self, **_arguments):
            return "synthetic rendered fixture"

    monkeypatch.setattr("stark_convert.chat.load_template", lambda _model, **_kwargs: ({}, Template()))
    return instance


READY = {"max_context_tokens": 8192, "max_new_tokens": 4096}


@pytest.fixture(autouse=True)
def isolated_failure_keys(monkeypatch):
    monkeypatch.setattr(image_checks, "_INVALIDATED", set())


def identity(worker):
    return image_checks.check_identity(worker)


def receipt_path(store, expected, root):
    return root / (store.key(expected) + ".json")


def test_execution_check_rejects_bad_image_then_replays_fresh_requests(worker):
    assert image_checks.verify_images(worker, READY) is False
    assert worker.calls == ["text", "tokenize", "invalid", "image", "image"]
    assert worker._verification_deadline is None and worker._image_deadline is None
    assert worker._image_check_identity == identity(worker)
    assert not list(worker.media.root.iterdir())


def test_receipt_contains_only_bound_metadata_and_reuses_without_inference(worker, tmp_path):
    directory = tmp_path / "checks"
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    files = list(directory.iterdir())
    assert len(files) == 1 and directory.stat().st_mode & 0o777 == 0o700
    assert files[0].stat().st_mode & 0o777 == 0o600
    value = json.loads(files[0].read_bytes())
    assert set(value) == {"schema_version", "evidence_level", "identity", "checks"}
    assert value["evidence_level"] == "locally_tested"
    assert value["identity"] == identity(worker)
    assert value["checks"] == image_checks._CHECKS
    payload = files[0].read_bytes()
    assert b"synthetic answer" not in payload and b"rendered fixture" not in payload and b"PNG" not in payload
    worker.calls.clear()
    assert image_checks.verify_images(worker, READY, directory=directory) is True
    assert worker.calls == []


@pytest.mark.parametrize("helper", ["media.py", "_build_metadata.py"])
def test_transaction_helper_change_requires_fresh_execution(worker, tmp_path, monkeypatch, helper):
    directory = tmp_path / "checks"
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    old_identity = identity(worker)
    original = image_checks.file_hash
    monkeypatch.setattr(
        image_checks,
        "file_hash",
        lambda path: "1" * 64 if Path(path).name == helper else original(path),
    )
    changed_identity = identity(worker)
    assert changed_identity["checker_sha256"] != old_identity["checker_sha256"]
    worker.calls.clear()
    worker.image_replies = [reply(), reply()]
    worker.image_counts = [64, 64]
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    assert worker.calls == ["text", "tokenize", "invalid", "image", "image"]


@pytest.mark.parametrize("key", sorted(image_checks._ARTIFACT_HASHES))
def test_any_admitted_artifact_hash_change_invalidates_receipt(worker, tmp_path, key):
    directory = tmp_path / "checks"
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    worker.image_input[key] = "0" * 64
    worker.image_replies = [reply(), reply()]
    worker.image_counts = [64, 64]
    worker.calls.clear()
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    assert worker.calls == ["text", "tokenize", "invalid", "image", "image"]
    assert len(list(directory.iterdir())) == 2


@pytest.mark.parametrize("change", ["worker", "os", "runtime", "checker", "version", "python", "dependency", "limits"])
def test_runtime_or_checker_change_invalidates_receipt(worker, tmp_path, monkeypatch, change):
    directory = tmp_path / "checks"
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    if change == "worker":
        worker.executable.write_bytes(b"different synthetic worker")
        worker._launch_worker_sha256 = hashlib.sha256(worker.executable.read_bytes()).hexdigest()
    elif change == "os":
        monkeypatch.setattr(image_checks, "_current_os_version", lambda: "26.7.2")
    elif change == "runtime":
        worker.image_input["runtime_identity"]["gpu_core_count"] = 40
    elif change == "checker":
        previous = image_checks.file_hash
        monkeypatch.setattr(
            image_checks, "file_hash", lambda path: previous(path) if Path(path) == worker.executable else "e" * 64
        )
    elif change == "version":
        monkeypatch.setattr(image_checks, "CHECK_VERSION", 2)
    elif change == "python":
        monkeypatch.setattr(image_checks, "sys", SimpleNamespace(version_info=(3, 13, 99)))
    elif change == "dependency":
        monkeypatch.setattr(image_checks, "_dependency_versions", lambda: {"jinja2": "changed synthetic version"})
    else:
        worker.image_input["max_image_tokens"] = 512
    worker.image_replies = [reply(), reply()]
    worker.image_counts = [64, 64]
    worker.calls.clear()
    assert image_checks.verify_images(worker, READY, directory=directory) is False
    assert worker.calls == ["text", "tokenize", "invalid", "image", "image"]


def test_worker_replacement_after_launch_rejected_before_execution(worker):
    worker.executable.write_bytes(b"replacement")
    with pytest.raises(image_checks.ImageCheckError, match="changed while loading"):
        image_checks.verify_images(worker, READY)
    assert worker.calls == []


def test_cached_receipt_does_not_hide_replacement_during_cache_read(worker, tmp_path, monkeypatch):
    directory = tmp_path / "checks"
    image_checks.verify_images(worker, READY, directory=directory)
    original = image_checks.CheckStore.load

    def replace_after_load(store, expected):
        found = original(store, expected)
        worker.executable.write_bytes(b"replacement after cache read")
        return found

    monkeypatch.setattr(image_checks.CheckStore, "load", replace_after_load)
    with pytest.raises(image_checks.ImageCheckError, match="changed|identity"):
        image_checks.verify_images(worker, READY, directory=directory)


@pytest.mark.parametrize("limits", [{"max_context_tokens": True}, {"max_context_tokens": 511}, {"max_new_tokens": 0}])
def test_cached_receipt_still_validates_current_ready_limits(worker, tmp_path, limits):
    directory = tmp_path / "checks"
    image_checks.verify_images(worker, READY, directory=directory)
    with pytest.raises(image_checks.ImageCheckError, match="token limits"):
        image_checks.verify_images(worker, {**READY, **limits}, directory=directory)


@pytest.mark.parametrize(
    "changes",
    [{"text": ""}, {"output_tokens": True}, {"output_tokens": 9}, {"finish_reason": "wrong"}, {"input_tokens": 2048}],
)
def test_bad_text_execution_does_not_create_receipt(worker, tmp_path, changes):
    worker.text_reply = reply(**changes)
    directory = tmp_path / "checks"
    with pytest.raises(image_checks.ImageCheckError):
        image_checks.verify_images(worker, READY, directory=directory)
    assert list(directory.iterdir()) == []
    assert worker._verification_deadline is None and worker._image_deadline is None


@pytest.mark.parametrize(
    "tokens",
    [
        {"token_count": True, "token_ids": [1]},
        {"token_count": 2, "token_ids": [1, 2]},
        {"token_count": 27},
        {"token_count": 27, "token_ids": [1] * 26},
        {"token_count": 3, "token_ids": [1, 2, True]},
        {"token_count": 3, "token_ids": [1, 2, -1]},
        {"token_count": 3, "token_ids": [1, 2, 248320]},
        {"token_count": 27, "token_ids": [1] * 27, "error": "tokenization_failed"},
    ],
)
def test_invalid_prompt_tokenization_rejects_before_image_decode(worker, tmp_path, tokens):
    worker.token_reply = tokens
    directory = tmp_path / "checks"
    with pytest.raises(image_checks.ImageCheckError, match="token accounting"):
        image_checks.verify_images(worker, READY, directory=directory)
    assert worker.calls == ["text", "tokenize"]
    assert not list(directory.iterdir())


@pytest.mark.parametrize(
    "change", ["accepted-invalid", "bad-reply", "bad-count", "changing-count", "staged-file", "closed"]
)
def test_image_check_failure_never_publishes_receipt(worker, tmp_path, change):
    if change == "accepted-invalid":
        worker.reject_invalid = False
    elif change == "bad-reply":
        worker.image_replies[0] = reply(text="")
    elif change == "bad-count":
        worker.image_counts[0] = True
    elif change == "changing-count":
        worker.image_counts[1] = 63
    elif change == "staged-file":
        worker.after_image = lambda value: (value.media.root / "residual").write_bytes(b"synthetic staged fixture")
    else:
        worker.after_image = lambda value: setattr(value, "closed", True)
    directory = tmp_path / "checks"
    with pytest.raises(image_checks.ImageCheckError):
        image_checks.verify_images(worker, READY, directory=directory)
    assert list(directory.iterdir()) == []
    assert worker._verification_deadline is None and worker._image_deadline is None


def test_interrupted_worker_cannot_reuse_receipt(worker, tmp_path):
    directory = tmp_path / "checks"
    image_checks.verify_images(worker, READY, directory=directory)
    worker.closed = True
    with pytest.raises(image_checks.ImageCheckError, match="interrupted"):
        image_checks.verify_images(worker, READY, directory=directory)
    assert not list(directory.iterdir())


def test_whole_check_deadline_is_not_extended_by_image_transaction(monkeypatch):
    instance = object.__new__(Worker)
    instance._transaction_lock = __import__("threading").RLock()
    instance.timeout = 30
    instance._verification_deadline = 101
    observed = []
    monkeypatch.setattr("stark_convert.chat.time.monotonic", lambda: 100)
    instance._request_image = lambda *_args, **_kwargs: observed.append(instance._image_deadline)
    instance.request_image("synthetic", None, {})
    assert observed == [101] and instance._image_deadline is None


@pytest.mark.parametrize("kind", ["public", "symlink", "file"])
def test_receipt_root_must_be_private_real_directory(tmp_path, kind):
    root = tmp_path / "checks"
    if kind == "public":
        root.mkdir(mode=0o755)
    elif kind == "file":
        root.write_bytes(b"synthetic")
    else:
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    with pytest.raises(image_checks.ImageCheckError, match="private image-check"):
        image_checks.CheckStore(root)


def test_receipt_constructor_closes_fd_when_metadata_read_fails(tmp_path, monkeypatch):
    real_close = os.close
    closed = []

    def close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(image_checks.os, "fstat", lambda _fd: (_ for _ in ()).throw(OSError("synthetic")))
    monkeypatch.setattr(image_checks.os, "close", close)
    with pytest.raises(image_checks.ImageCheckError):
        image_checks.CheckStore(tmp_path / "checks")
    assert len(closed) == 1


@pytest.fixture
def store(worker, tmp_path):
    root = tmp_path / "checks"
    instance = image_checks.CheckStore(root)
    yield instance, root, identity(worker)
    instance.close()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "public", "oversize"])
def test_untrusted_receipt_entry_is_never_used(store, tmp_path, kind):
    instance, root, expected = store
    instance.save(expected)
    path = receipt_path(instance, expected, root)
    original = path.read_bytes()
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(original)
    outside.chmod(0o600)
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    elif kind == "directory":
        path.mkdir(mode=0o700)
    else:
        path.write_bytes(original if kind == "public" else b"x" * 65537)
        path.chmod(0o644 if kind == "public" else 0o600)
    assert instance.load(expected) is False
    assert outside.read_bytes() == original


@pytest.mark.parametrize(
    "change", ["reference-claim", "integer-check", "float-schema", "extra", "malformed", "duplicate"]
)
def test_malformed_or_inflated_receipt_claim_is_ignored(store, change):
    instance, root, expected = store
    instance.save(expected)
    path = receipt_path(instance, expected, root)
    value = json.loads(path.read_bytes())
    if change == "reference-claim":
        value["evidence_level"] = "reference_verified"
    elif change == "integer-check":
        value["checks"]["text_generation"] = 1
    elif change == "float-schema":
        value["schema_version"] = 1.0
    elif change == "extra":
        value["prompt"] = "synthetic private marker"
    if change == "malformed":
        path.write_bytes(b"\xff")
    elif change == "duplicate":
        path.write_text('{"schema_version":1,"schema_version":1}')
    else:
        path.write_text(json.dumps(value))
    assert instance.load(expected) is False


def test_directory_replacement_does_not_redirect_receipt_operations(store, tmp_path):
    instance, root, expected = store
    pinned = tmp_path / "pinned-original"
    root.rename(pinned)
    root.mkdir(mode=0o700)
    replacement = receipt_path(instance, expected, root)
    replacement.write_bytes(b"replacement sentinel")
    instance.save(expected)
    assert replacement.read_bytes() == b"replacement sentinel"
    assert instance.load(expected) is True
    instance.invalidate(expected)
    assert list(pinned.iterdir()) == []
    assert replacement.read_bytes() == b"replacement sentinel"


def test_closed_store_does_not_fall_back_to_current_directory(store, monkeypatch):
    instance, _root, expected = store
    instance.close()
    monkeypatch.setattr(image_checks.os, "open", lambda *_args, **_kwargs: pytest.fail("unexpected filesystem access"))
    try:
        assert instance.load(expected) is False
    except image_checks.ImageCheckError:
        pass
    with pytest.raises(image_checks.ImageCheckError):
        instance.save(expected)


def test_atomic_temp_collision_preserves_existing_file(store, monkeypatch):
    instance, root, expected = store
    collision = root / (".check-" + "0" * 32)
    collision.write_bytes(b"preexisting private metadata")
    collision.chmod(0o600)
    monkeypatch.setattr(image_checks.secrets, "token_hex", lambda _count: "0" * 32)
    with pytest.raises(image_checks.ImageCheckError):
        instance.save(expected)
    assert collision.read_bytes() == b"preexisting private metadata"


def test_cache_mutation_during_read_is_not_reused(store, monkeypatch):
    instance, root, expected = store
    instance.save(expected)
    path = receipt_path(instance, expected, root)
    original_fdopen = os.fdopen

    class MutatingRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return self.stream.__exit__(*arguments)

        def read(self, count):
            data = self.stream.read(count)
            metadata = path.stat()
            os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
            return data

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(image_checks.os, "fdopen", lambda fd, mode: MutatingRead(original_fdopen(fd, mode)))
    assert instance.load(expected) is False


def test_failed_invalidation_blacklists_key_even_if_receipt_root_cannot_open(worker, tmp_path, monkeypatch):
    worker.image_check_dir = tmp_path / "checks"
    image_checks.verify_images(worker, READY, directory=worker.image_check_dir)
    expected = worker._image_check_identity
    key = image_checks.CheckStore.key(expected)
    original = image_checks.CheckStore.__init__
    monkeypatch.setattr(
        image_checks.CheckStore,
        "__init__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(image_checks.ImageCheckError("fixed storage failure")),
    )
    with pytest.raises(image_checks.ImageCheckError):
        image_checks.invalidate_checks(worker)
    assert key in image_checks._INVALIDATED
    monkeypatch.setattr(image_checks.CheckStore, "__init__", original)
    instance = image_checks.CheckStore(worker.image_check_dir)
    try:
        assert instance.load(expected) is False
    finally:
        instance.close()


def test_receipt_removal_and_worker_invalidation_are_content_free(worker, tmp_path):
    worker.image_check_dir = tmp_path / "checks"
    image_checks.verify_images(worker, READY, directory=worker.image_check_dir)
    assert len(list(worker.image_check_dir.iterdir())) == 1
    image_checks.invalidate_checks(worker)
    assert worker._image_check_identity is None and not list(worker.image_check_dir.iterdir())


def test_native_failure_invalidation_error_does_not_skip_worker_cleanup(monkeypatch):
    instance = object.__new__(Worker)
    instance._transaction_lock = __import__("threading").RLock()
    instance.timeout = 1
    killed, reaped, closed = [], [], []
    instance.process = SimpleNamespace(
        poll=lambda: None, kill=lambda: killed.append(True), wait=lambda: reaped.append(True)
    )
    instance._request_image = lambda *_args, **_kwargs: (_ for _ in ()).throw(ChatError("synthetic native failure"))
    instance.invalidate_image_checks = lambda: (_ for _ in ()).throw(
        image_checks.ImageCheckError("fixed cleanup error")
    )
    instance.close = lambda: closed.append(True)
    with pytest.raises((ChatError, image_checks.ImageCheckError)):
        instance.request_image("synthetic", None, {})
    assert killed == [True] and reaped == [True] and closed == [True]


def test_memory_preflight_rejects_insufficient_observed_budget(worker, monkeypatch):
    (worker.model_path / "model.safetensors").write_bytes(b"synthetic weights")
    config = {"artifact_contract": {"schema_version": 3, "family": {"profile": FAMILY_PROFILE, "version": 1}}}
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    with pytest.raises(image_checks.ImageCheckError, match="available memory"):
        image_checks.preflight_family_memory(worker.model_path, config)


def test_memory_preflight_is_observation_only_and_does_not_run_for_text(worker, monkeypatch):
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: pytest.fail("unexpected memory probe"))
    image_checks.preflight_family_memory(worker.model_path, {})


def test_memory_preflight_rejects_boolean_family_version(worker, monkeypatch):
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: pytest.fail("unsupported profile memory probe"))
    config = {"artifact_contract": {"schema_version": 3, "family": {"profile": FAMILY_PROFILE, "version": True}}}
    with pytest.raises(image_checks.ImageCheckError, match="not supported"):
        image_checks.preflight_family_memory(worker.model_path, config)
