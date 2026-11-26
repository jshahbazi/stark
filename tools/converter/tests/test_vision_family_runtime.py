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


def test_inkling_profile_is_bound_to_its_pinned_target_artifacts():
    from stark_convert import inkling_prepare

    def contract(artifact):
        return family_contract(
            family_profile=inkling_prepare.VISION_PROFILE,
            artifact_schema_version=7,
            model_source=inkling_prepare.REPO,
            model_revision=inkling_prepare.REVISION,
            artifact_sha256=artifact,
            processor_sha256=inkling_prepare.PROCESSOR_SHA,
            tokenizer_sha256=inkling_prepare.TOKENIZER_SHA,
            template_sha256=inkling_prepare.TEMPLATE_SHA,
            max_source_pixels=1_638_400,
            max_processed_pixels=1_638_400,
        )

    original = inkling_prepare.WEIGHTS[inkling_prepare.TARGET][1]
    assert inkling_prepare.IMAGE_TARGET_SHAS == (original, inkling_prepare.AFFINE8_TARGET_SHA)
    for artifact in inkling_prepare.IMAGE_TARGET_SHAS:
        assert vision_runtime.family_image_input_contract(contract(artifact)) is True
    # Recognition is an implementation profile only; any other artifact fails closed.
    assert vision_runtime.family_image_input_contract(contract("0" * 64)) is False
    draft = inkling_prepare.WEIGHTS[inkling_prepare.DRAFT][1]
    assert vision_runtime.family_image_input_contract(contract(draft)) is False


def test_hy4_profile_is_bound_to_the_pinned_gguf_projector_and_metadata():
    profile = vision_runtime.HY4_VISION_PROFILE
    value = family_contract(
        family_profile=profile,
        artifact_schema_version=7,
        model_source="AngelSlim/Hy4-flash-GGUF",
        model_revision="c50d4dd30be42f2b91138e5731de0e2f32ad56b2",
        artifact_sha256="f9a244c3a1f85e03847959b575528c3927ac05ab0a3f532b1c1831f9eb1ddd3c",
        config_sha256="1f56ac1a3063f926e91ec43842bfdc0fd6d4401ef91bcdecd1fc27dd506ea968",
        tokenizer_sha256="2e3f3d22065c2be5765ce99ab2410411ae338b32ce809ea20dcdb61d00198e83",
        processor_sha256="cc5cc63da33e0ebf53b8ac035879982ba18c47965f82a5f341bc82d98759dc1d",
        template_sha256="11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35",
        max_source_pixels=32_000_000,
        max_source_dimension=16_384,
        max_processed_pixels=4_194_304,
        max_patches=16_384,
        max_image_tokens=8_192,
    )
    value["runtime_identity"].update(
        hy4_load_profile="default",
        expert_storage="streamed",
        mapped_expert_residency=False,
        grouped_prefill_requested=True,
        grouped_prefill_operands="f32",
        grouped_prefill_fragments_requested=False,
        gpu_expert_plan=False,
        prefill_chunk_tokens=1024,
        prefill_stage_profile_requested=False,
        prefill_dense_packed_requested=False,
        prefill_dense_device_views_requested=False,
        prefill_dense_ffn_device_views_requested=False,
        prefill_tensor_mla_requested=False,
        token_embeddings_resident=False,
        draft_experts_resident=False,
        draft_head_format="q8_0",
        streamed_expert_demand_retention=False,
        streamed_expert_arena_bytes=27_996_717_056,
        streamed_expert_arena_requested_bytes=0,
    )
    assert vision_runtime.image_input_contract(value) == value
    assert vision_runtime.family_image_input_contract(value) is True
    for key, replacement in [
        ("model_source", "tencent/Hy4-flash"),
        ("model_revision", "a" * 40),
        ("artifact_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("tokenizer_sha256", "0" * 64),
        ("processor_sha256", "0" * 64),
        ("template_sha256", "0" * 64),
        ("max_source_dimension", 32_768),
        ("max_processed_pixels", 16_777_216),
        ("max_patches", 65_536),
        ("max_image_tokens", 16_384),
    ]:
        assert vision_runtime.family_image_input_contract({**value, key: replacement}) is False
    for key, replacement in [
        ("hy4_load_profile", "unknown"),
        ("expert_storage", "resident"),
        ("grouped_prefill_operands", "bf16"),
        ("prefill_chunk_tokens", True),
        ("draft_head_format", "q4_0"),
        ("streamed_expert_arena_bytes", -1),
        ("unreviewed_runtime_option", True),
    ]:
        runtime = {**value["runtime_identity"], key: replacement}
        candidate = {**value, "runtime_identity": runtime}
        assert vision_runtime.image_input_contract(candidate) is None


def test_hy4_prepared_image_identity_is_source_bound(tmp_path, monkeypatch):
    from stark_convert import hybrid_prepare, hybrid_vision

    identity = {
        "repo": "AngelSlim/Hy4-flash-GGUF",
        "revision": "c50d4dd30be42f2b91138e5731de0e2f32ad56b2",
        "architecture": "hy4_flash",
        "weight_bytes": 156_415_403_072,
        "quantization": "Q3_K",
    }
    monkeypatch.setattr(hybrid_prepare, "prepared_identity", lambda _directory: identity)
    result = hybrid_vision.prepared_image_identity(tmp_path)
    assert result == {
        "kind": "Hy4NativeVisionArtifact",
        "schema_version": 7,
        "family_profile": vision_runtime.HY4_VISION_PROFILE,
        "family_version": 1,
        "model_source": identity["repo"],
        "model_revision": identity["revision"],
    }


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
