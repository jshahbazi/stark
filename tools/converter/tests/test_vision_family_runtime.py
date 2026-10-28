"""Family compatibility metadata never grants reference parity or image access alone."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from test_image_checks import READY, family_contract

from stark_convert import chat, image_checks, vision_runtime
from stark_convert.chat import ChatError, Worker


@pytest.mark.parametrize("revision", [None, "a" * 40, "b" * 64])
def test_family_contract_accepts_immutable_or_local_source_identity(revision):
    value = family_contract(model_revision=revision)
    assert vision_runtime.image_input_contract(value) == value
    assert vision_runtime.family_image_input_contract(value) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact_schema_version": True},
        {"artifact_schema_version": 4},
        {"family_profile": "another-family"},
        {"family_version": True},
        {"family_version": 2},
        {"model_source": "/private/model"},
        {"model_source": ".hidden/model"},
        {"model_source": "owner/model\nprivate"},
        {"model_source": "owner/modèl"},
        {"model_source": "https://example.test/model"},
        {"model_revision": "main"},
        {"model_revision": "A" * 40},
        {"model_revision": 12},
        {"max_images": 2},
        {"max_compressed_bytes": 20 * 1024 * 1024 + 1},
        {"max_source_pixels": 32_000_001},
        {"max_processed_pixels": 1_048_577},
        {"max_patches": 4097},
        {"max_image_tokens": 1025},
        {"max_image_tokens": True},
        {"operations": ["prepare_image", "generate_image", "generate_image"]},
        {"formats": ["image/png", "image/png"]},
        {"artifact_sha256": "not-a-hash"},
        {"transport": "client-path"},
    ],
)
def test_unknown_or_malformed_family_contracts_fail_closed(changes):
    assert vision_runtime.image_input_contract(family_contract(**changes)) is None


@pytest.mark.parametrize("change", [{"backend": "cpu"}, {"gpu_core_count": True}, {"supports_mxu": "yes"}])
def test_family_contract_requires_typed_native_runtime_identity(change):
    value = family_contract()
    value["runtime_identity"].update(change)
    assert vision_runtime.image_input_contract(value) is None


def test_schema2_checkpoint_identity_remains_pinned():
    value = family_contract(artifact_schema_version=2)
    assert vision_runtime.image_input_contract(value) is None
    value.update(
        model_source=vision_runtime.REPO,
        model_revision=vision_runtime.REVISION,
        artifact_sha256=vision_runtime._ARTIFACT,
    )
    assert vision_runtime.image_input_contract(value) == value
    assert vision_runtime.image_input_contract({**value, "artifact_sha256": "0" * 64}) is None


def test_family_self_claim_or_matching_catalog_fields_cannot_inherit_reference_parity(tmp_path, monkeypatch):
    executable = tmp_path / "worker"
    executable.write_bytes(b"synthetic executable")
    value = family_contract(qualified=True)
    # Even fabricated matching catalog data cannot turn schema3 family evidence
    # into an exact schema2 reference qualification.
    monkeypatch.setattr(
        vision_runtime,
        "_records",
        lambda: [{**value, "worker_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}],
    )
    assert vision_runtime.qualified_image_worker(value, executable) is False


def pending_worker(tmp_path):
    executable = tmp_path / "worker"
    executable.write_bytes(b"synthetic executable")
    instance = object.__new__(Worker)
    value = {
        "ready": True,
        "mode": "text-generation-v1",
        "cancellation": "pipe-v1",
        "image_input": family_contract(),
        **READY,
    }
    instance.pending = json.dumps(value).encode() + b"\n"
    instance.timeout = 1
    instance.media = SimpleNamespace(root=tmp_path / "media")
    instance.executable = executable
    instance._launch_worker_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    instance.image_check_dir = tmp_path / "checks"
    instance.image_verification = "unverified"
    instance.image_check_cached = False
    instance.images = False
    instance._closed = False
    instance._image_check_identity = None
    instance.closed = False
    instance.close = lambda: setattr(instance, "closed", True)
    return instance, value


@pytest.mark.parametrize("cached", [False, True])
def test_worker_only_enables_family_images_after_completed_local_checks(tmp_path, monkeypatch, cached):
    instance, value = pending_worker(tmp_path)
    called = []
    monkeypatch.setattr(chat, "qualified_image_worker", lambda *_args, **_kwargs: False)

    def verify(worker, ready, *, directory):
        assert worker.images is False and worker.image_verification == "unverified"
        assert worker is instance and ready == value and directory == instance.image_check_dir
        called.append(True)
        return cached

    monkeypatch.setattr(image_checks, "verify_images", verify)
    assert instance.read() == value
    assert called == [True]
    assert instance.images is True and instance.image_verification == "locally_tested"
    assert instance.image_check_cached is cached
    assert not instance.closed


def test_reference_worker_does_not_become_locally_tested_or_run_redundant_checks(tmp_path, monkeypatch):
    instance, value = pending_worker(tmp_path)
    value["image_input"].update(
        artifact_schema_version=2,
        model_source=vision_runtime.REPO,
        model_revision=vision_runtime.REVISION,
        artifact_sha256=vision_runtime._ARTIFACT,
    )
    instance.pending = json.dumps(value).encode() + b"\n"
    monkeypatch.setattr(chat, "qualified_image_worker", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(image_checks, "verify_images", lambda *_args, **_kwargs: pytest.fail("unexpected local check"))
    instance.read()
    assert instance.images and instance.image_verification == "reference_verified"
    assert instance.image_check_cached is False


def test_worker_check_failure_is_content_free_and_never_enables_images(tmp_path, monkeypatch):
    instance, _value = pending_worker(tmp_path)
    monkeypatch.setattr(chat, "qualified_image_worker", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        image_checks,
        "verify_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("private exception payload")),
    )
    with pytest.raises(ChatError) as error:
        instance.read()
    assert "private" not in str(error.value)
    assert instance.images is False and instance.image_verification == "unverified" and instance.closed


def test_worker_startup_cleanup_failure_cannot_escape_primary_content_boundary(tmp_path, monkeypatch):
    instance, _value = pending_worker(tmp_path)
    monkeypatch.setattr(chat, "qualified_image_worker", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        image_checks,
        "verify_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("private primary payload")),
    )
    instance.invalidate_image_checks = lambda: (_ for _ in ()).throw(OSError("private cleanup payload"))
    with pytest.raises(ChatError) as error:
        instance.read()
    assert "private" not in str(error.value) and instance.closed
    assert instance.images is False


def test_unrecognized_family_never_runs_automatic_checks(tmp_path, monkeypatch):
    instance, value = pending_worker(tmp_path)
    value["image_input"]["family_profile"] = "unrecognized"
    instance.pending = json.dumps(value).encode() + b"\n"
    monkeypatch.setattr(chat, "qualified_image_worker", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(image_checks, "verify_images", lambda *_args, **_kwargs: pytest.fail("unexpected local check"))
    instance.read()
    assert instance.images is False and instance.image_verification == "unverified"


def test_second_ready_frame_is_protocol_failure(tmp_path, monkeypatch):
    instance, value = pending_worker(tmp_path)
    monkeypatch.setattr(chat, "qualified_image_worker", lambda *_args, **_kwargs: True)
    instance.read()
    instance.pending = json.dumps(value).encode() + b"\n"
    with pytest.raises(ChatError, match="Unexpected worker readiness"):
        instance.read()
