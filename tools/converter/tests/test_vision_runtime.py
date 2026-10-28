import hashlib
import json

import pytest

from stark_convert import vision_runtime
from stark_convert._build_metadata import ConversionError, file_hash
from stark_convert.vision_contract import REPO, REVISION

RUNTIME_IDENTITY = {
    "backend": "metal",
    "device_name": "Apple Synthetic GPU",
    "apple_gpu_family": 1008,
    "gpu_core_count": 32,
    "supports_mxu": False,
}


def contract(**overrides):
    value = {
        "version": "mediaref-v1",
        "transport": "owner-private-file-v1",
        "model_source": REPO,
        "model_revision": REVISION,
        "artifact_sha256": vision_runtime._ARTIFACT,
        "runtime_identity": dict(RUNTIME_IDENTITY),
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
    value.update(overrides)
    return value


class Resource:
    def __init__(self, text):
        self.text = text

    def joinpath(self, _name):
        return self

    def read_text(self):
        return self.text


def record(worker_hash, **overrides):
    value = {
        "repo": REPO,
        "revision": REVISION,
        "artifact_sha256": vision_runtime._ARTIFACT,
        "config_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "processor_sha256": "e" * 64,
        "template_sha256": "f" * 64,
        "worker_sha256": worker_hash,
        "transport": "owner-private-file-v1",
        "runtime_identity": dict(RUNTIME_IDENTITY),
        "os_version": "15.6.1",
        "passed": True,
        "evidence_sha256": "e" * 64,
    }
    value.update(overrides)
    return value


def install_document(monkeypatch, value):
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value)
    monkeypatch.setattr(vision_runtime, "files", lambda _package: Resource(text))


def executable(tmp_path, name="worker"):
    path = tmp_path / name
    path.write_bytes(b"synthetic executable identity " + name.encode("ascii"))
    return path


def test_image_input_contract_rejects_malformed_raw_contracts():
    assert vision_runtime.image_input_contract(contract()) == contract()
    malformed = [
        {"version": "wrong"},
        contract(transport="path-v0"),
        contract(model_source="other/model"),
        contract(model_revision="0" * 40),
        contract(artifact_sha256="f" * 64),
        contract(runtime_identity={**RUNTIME_IDENTITY, "gpu_core_count": True}),
        contract(runtime_identity={**RUNTIME_IDENTITY, "backend": "cpu"}),
        contract(runtime_identity={**RUNTIME_IDENTITY, "device_name": "Apple GPU\nprivate"}),
        contract(config_sha256="invalid"),
        contract(artifact_schema_version=True),
        contract(max_images=True),
        contract(max_compressed_bytes=0),
        contract(max_image_tokens="1024"),
        contract(operations=["prepare_image", "prepare_image", "discard_image"]),
        contract(operations=["prepare_image", "generate_image"]),
        contract(formats=["image/png", "image/png"]),
        contract(formats=["image/png", 1]),
        None,
        [],
    ]
    assert all(vision_runtime.image_input_contract(value) is None for value in malformed)


def test_empty_default_qualification_catalog_fails_closed(monkeypatch, tmp_path):
    executable_path = executable(tmp_path)
    monkeypatch.setattr(vision_runtime, "_current_os_version", lambda: "15.6.1")
    assert vision_runtime.qualified_image_worker(contract(), executable_path) is False


@pytest.mark.parametrize("value", ["", "26", "26.1.2.3", "26.1 trailing", "01.2"])
def test_os_product_version_must_be_available_and_strict(monkeypatch, value):
    monkeypatch.setattr(vision_runtime.platform, "mac_ver", lambda: (value, ("", "", ""), "arm64"))
    assert vision_runtime._current_os_version() is None


def test_os_product_version_reads_only_the_numeric_product_version(monkeypatch):
    monkeypatch.setattr(vision_runtime.platform, "mac_ver", lambda: ("26.1.2", ("", "", ""), "arm64"))
    assert vision_runtime._current_os_version() == "26.1.2"


def test_only_exact_worker_and_runtime_tuple_qualifies(monkeypatch, tmp_path):
    candidate = executable(tmp_path, "candidate")
    other = executable(tmp_path, "other")
    candidate_hash = file_hash(candidate)
    monkeypatch.setattr(vision_runtime, "_current_os_version", lambda: "15.6.1")
    install_document(monkeypatch, {"schema_version": 2, "records": [record(candidate_hash)]})
    assert vision_runtime.qualified_image_worker(contract(), candidate) is True
    assert vision_runtime.qualified_image_worker(contract(), other) is False


def test_worker_hash_must_still_match_launch_snapshot(monkeypatch, tmp_path):
    candidate = executable(tmp_path, "candidate")
    original_hash = file_hash(candidate)
    replacement_hash = hashlib.sha256(b"replacement executable").hexdigest()
    monkeypatch.setattr(vision_runtime, "_current_os_version", lambda: "15.6.1")
    install_document(monkeypatch, {"schema_version": 2, "records": [record(replacement_hash)]})
    candidate.write_bytes(b"replacement executable")

    # The two-argument compatibility form continues to match the current file;
    # a worker that launched the earlier bytes must fail closed.
    assert vision_runtime.qualified_image_worker(contract(), candidate) is True
    assert vision_runtime.qualified_image_worker(contract(), candidate, expected_worker_sha256=original_hash) is False


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "cpu"},
        {"device_name": "Apple Different GPU"},
        {"apple_gpu_family": 1009},
        {"gpu_core_count": 64},
        {"supports_mxu": True},
    ],
)
def test_device_class_mismatch_disables_image_qualification(monkeypatch, tmp_path, changes):
    candidate = executable(tmp_path)
    monkeypatch.setattr(vision_runtime, "_current_os_version", lambda: "15.6.1")
    install_document(monkeypatch, {"schema_version": 2, "records": [record(file_hash(candidate))]})
    identity = {**RUNTIME_IDENTITY, **changes}
    assert vision_runtime.qualified_image_worker(contract(runtime_identity=identity), candidate) is False


def test_os_runtime_mismatch_or_unavailable_disables_image_qualification(monkeypatch, tmp_path):
    candidate = executable(tmp_path)
    install_document(monkeypatch, {"schema_version": 2, "records": [record(file_hash(candidate))]})
    monkeypatch.setattr(vision_runtime, "_current_os_version", lambda: "26.0")
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False
    monkeypatch.setattr(vision_runtime, "_current_os_version", lambda: None)
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False


@pytest.mark.parametrize("hash_key", sorted(vision_runtime._ARTIFACT_HASHES))
def test_each_artifact_file_hash_must_match_qualification(monkeypatch, tmp_path, hash_key):
    candidate = executable(tmp_path)
    install_document(
        monkeypatch,
        {"schema_version": 2, "records": [record(file_hash(candidate))]},
    )
    changed = contract(**{hash_key: "0" * 64})
    assert vision_runtime.qualified_image_worker(changed, candidate) is False


def test_invalid_runtime_contract_never_qualifies_even_with_matching_record(monkeypatch, tmp_path):
    candidate = executable(tmp_path)
    install_document(
        monkeypatch,
        {"schema_version": 2, "records": [record(file_hash(candidate))]},
    )
    assert vision_runtime.qualified_image_worker(contract(transport="different"), candidate) is False


def test_worker_hash_failure_fails_closed(monkeypatch, tmp_path):
    candidate = executable(tmp_path)
    install_document(monkeypatch, {"schema_version": 2, "records": [record("a" * 64)]})

    def fail_hash(_path):
        raise ConversionError("private path and digest")

    monkeypatch.setattr(vision_runtime, "file_hash", fail_hash)
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False


@pytest.mark.parametrize(
    "changes",
    [
        {"repo": "other/model"},
        {"revision": "0" * 40},
        {"artifact_sha256": "f" * 64},
        {"transport": "other-transport"},
        {"passed": False},
        {"worker_sha256": "invalid"},
        {"evidence_sha256": "invalid"},
        {"runtime_identity": {**RUNTIME_IDENTITY, "supports_mxu": "false"}},
        {"os_version": "15.6.1 malicious"},
    ],
)
def test_mismatched_or_failed_qualification_record_disables_images(monkeypatch, tmp_path, changes):
    candidate = executable(tmp_path)
    install_document(
        monkeypatch,
        {"schema_version": 2, "records": [record(file_hash(candidate), **changes)]},
    )
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 1, "records": []},
        {"schema_version": True, "records": []},
        {"schema_version": 2, "records": {}},
        {"schema_version": 2, "records": [], "extra": False},
        '{"schema_version":2,"records":[],"records":[]}',
    ],
)
def test_malformed_or_duplicate_qualification_documents_fail_closed(monkeypatch, tmp_path, document):
    candidate = executable(tmp_path)
    install_document(monkeypatch, document)
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False


def test_duplicate_worker_qualification_identity_disables_images(monkeypatch, tmp_path):
    candidate = executable(tmp_path)
    worker_hash = file_hash(candidate)
    install_document(
        monkeypatch,
        {"schema_version": 2, "records": [record(worker_hash), record(worker_hash)]},
    )
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False


def test_any_invalid_record_fails_closed_even_with_a_valid_match(monkeypatch, tmp_path):
    candidate = executable(tmp_path)
    worker_hash = file_hash(candidate)
    install_document(
        monkeypatch,
        {
            "schema_version": 2,
            "records": [record(worker_hash), record("f" * 64, transport="unrecognized")],
        },
    )
    assert vision_runtime.qualified_image_worker(contract(), candidate) is False


def test_executable_digest_is_content_based(tmp_path):
    first = executable(tmp_path, "first")
    second = executable(tmp_path, "second")
    second.write_bytes(first.read_bytes())
    assert file_hash(first) == file_hash(second) == hashlib.sha256(first.read_bytes()).hexdigest()


def test_image_qualification_lookup_is_exact_and_does_not_claim_current_worker(monkeypatch, tmp_path):
    named_worker_hash = "a" * 64
    install_document(monkeypatch, {"schema_version": 2, "records": [record(named_worker_hash)]})
    candidate = executable(tmp_path)
    hashes = {key: contract()[key] for key in vision_runtime._ARTIFACT_HASHES}

    found = vision_runtime.image_qualification_for(REPO, "bfloat16", hashes)
    assert found == record(named_worker_hash)
    assert file_hash(candidate) != found["worker_sha256"]
    assert vision_runtime.image_qualification_for("other/repo", "bfloat16", hashes) is None
    assert vision_runtime.image_qualification_for(REPO, "float16", hashes) is None
    assert vision_runtime.image_qualification_for(REPO, "bfloat16", {**hashes, "template_sha256": "0" * 64}) is None
