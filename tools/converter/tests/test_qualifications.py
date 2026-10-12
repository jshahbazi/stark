"""Native loader evidence applies only to its exact artifacts and named build."""

import json
from copy import deepcopy

import pytest

from stark_convert import qualifications
from stark_convert.streaming import ConversionError

REPO = "Qwen/Qwen3-0.6B"
HASHES = {
    "source_manifest_sha256": "1" * 64,
    "output_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "tokenizer_sha256": "4" * 64,
}


def evidence_record():
    return {
        "repo": REPO,
        "dtype": "bfloat16",
        **HASHES,
        "native_build": {
            "passed": True,
            "native_score_calls": 0,
            "worker_sha256": "5" * 64,
            "native_source_manifest_sha256": "6" * 64,
            "cargo_lock_sha256": "7" * 64,
            "rust_toolchain_sha256": "8" * 64,
            "rust": "1.98.1",
            "profile": "release",
            "features": ["default", "optimized"],
        },
    }


@pytest.fixture
def document(tmp_path, monkeypatch):
    path = tmp_path / "native_qualifications.json"
    path.write_text(json.dumps({"schema_version": 1, "records": [evidence_record()]}))
    monkeypatch.setattr(qualifications, "files", lambda package: tmp_path)
    return path


def test_exact_artifacts_return_named_build_without_execution_claim(document):
    result = qualifications.qualification_for(REPO, "bfloat16", HASHES)
    assert result == evidence_record()
    assert result["native_build"]["native_score_calls"] == 0
    assert "execution" not in result and "execution" not in result["native_build"]
    assert "current_engine" not in result and "current_engine" not in result["native_build"]


def test_returned_evidence_cannot_mutate_future_qualification(document):
    result = qualifications.qualification_for(REPO, "bfloat16", HASHES)
    result["native_build"]["passed"] = False
    result["native_build"]["features"].clear()
    assert qualifications.qualification_for(REPO, "bfloat16", HASHES) == evidence_record()


@pytest.mark.parametrize("key", HASHES)
def test_every_source_and_output_hash_is_required_to_match(document, key):
    changed = {**HASHES, key: "9" * 64}
    assert qualifications.qualification_for(REPO, "bfloat16", changed) is None
    missing = {name: value for name, value in HASHES.items() if name != key}
    assert qualifications.qualification_for(REPO, "bfloat16", missing) is None


@pytest.mark.parametrize(
    "repo,dtype",
    [("Qwen/Qwen3-1.7B", "bfloat16"), ("unknown/model", "bfloat16"), (REPO, "float16"), (REPO, "float32")],
)
def test_other_model_and_dtype_do_not_inherit_qualification(document, repo, dtype):
    assert qualifications.qualification_for(repo, dtype, HASHES) is None


@pytest.mark.parametrize("value", [None, "", "bad", "a" * 63, "A" * 64, 1])
def test_invalid_query_hashes_never_qualify(document, value):
    assert qualifications.qualification_for(REPO, "bfloat16", {**HASHES, "output_sha256": value}) is None


def test_empty_record_list_does_not_qualify(document):
    document.write_text(json.dumps({"schema_version": 1, "records": []}))
    assert qualifications.qualification_for(REPO, "bfloat16", HASHES) is None


def test_missing_resource_fails_closed(document):
    document.unlink()
    with pytest.raises(ConversionError, match="^Invalid native qualification catalog$"):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)


@pytest.mark.parametrize(
    "changed",
    [
        {"schema_version": 2, "records": []},
        {"schema_version": True, "records": []},
        {"schema_version": 1, "records": {}},
        {"schema_version": 1, "records": [], "prompt": "synthetic"},
    ],
)
def test_invalid_schema_does_not_downgrade_to_empty_evidence(document, changed):
    document.write_text(json.dumps(changed))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repo", "unknown/model"),
        ("dtype", "float16"),
        ("source_manifest_sha256", "bad"),
        ("output_sha256", "bad"),
        ("config_sha256", "bad"),
        ("tokenizer_sha256", "bad"),
        ("prompt", "synthetic forbidden field"),
        ("execution", "qualified"),
    ],
)
def test_forged_record_fields_are_rejected(document, field, value):
    record = evidence_record()
    record[field] = value
    document.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("passed", 1),
        ("native_score_calls", 1),
        ("native_score_calls", False),
        ("worker_sha256", "bad"),
        ("native_source_manifest_sha256", "bad"),
        ("cargo_lock_sha256", "bad"),
        ("rust_toolchain_sha256", "bad"),
        ("rust", "1.98.0"),
        ("profile", "debug"),
        ("features", ["default"]),
        ("features", ["default", "default"]),
        ("exception", "synthetic forbidden field"),
    ],
)
def test_forged_or_incomplete_native_build_evidence_is_rejected(document, field, value):
    record = evidence_record()
    record["native_build"][field] = value
    document.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)


def test_duplicate_matching_records_are_rejected_even_if_the_build_differs(document):
    first, second = evidence_record(), deepcopy(evidence_record())
    second["native_build"]["worker_sha256"] = "9" * 64
    document.write_text(json.dumps({"schema_version": 1, "records": [first, second]}))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)


def test_duplicate_json_keys_are_rejected(document):
    document.write_text('{"schema_version":1,"schema_version":1,"records":[]}')
    with pytest.raises(ConversionError):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)


def test_invalid_json_error_does_not_expose_contents(document):
    document.write_text("synthetic private exception content")
    with pytest.raises(ConversionError) as caught:
        qualifications.qualification_for(REPO, "bfloat16", HASHES)
    assert str(caught.value) == "Invalid native qualification catalog"


def classifier_evidence_record():
    record = evidence_record()
    record.update(repo="trymirai/chat-moderation-router", model_kind="classifier")
    record["native_build"].update(
        loader="classifier_load_harness",
        native_classification_calls=0,
        native_generation_calls=0,
        harness_source_sha256="a" * 64,
        harness_manifest_sha256="b" * 64,
        harness_lock_sha256="c" * 64,
    )
    return record


def test_classifier_qualification_requires_explicit_loader_and_zero_execution(document):
    record = classifier_evidence_record()
    document.write_text(json.dumps({"schema_version": 1, "records": [evidence_record(), record]}))
    assert qualifications.qualification_for(record["repo"], "bfloat16", HASHES) == record
    assert qualifications.qualification_for(REPO, "bfloat16", HASHES) == evidence_record()


@pytest.mark.parametrize("field", ["model_kind"])
@pytest.mark.parametrize("value", [None, "language", "tts", True])
def test_classifier_record_cannot_inherit_language_proof(document, field, value):
    record = classifier_evidence_record()
    if value is None:
        record.pop(field)
    else:
        record[field] = value
    document.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    with pytest.raises(ConversionError, match="^Invalid native qualification catalog$"):
        qualifications.qualification_for(record["repo"], "bfloat16", HASHES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("loader", "scoring_worker"),
        ("native_classification_calls", 1),
        ("native_classification_calls", False),
        ("native_generation_calls", 1),
        ("native_generation_calls", False),
        ("harness_source_sha256", "bad"),
        ("harness_manifest_sha256", "bad"),
        ("harness_lock_sha256", "bad"),
    ],
)
def test_classifier_execution_or_incomplete_harness_evidence_fails(document, field, value):
    record = classifier_evidence_record()
    record["native_build"][field] = value
    document.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(record["repo"], "bfloat16", HASHES)


@pytest.mark.parametrize(
    "field",
    sorted(qualifications._CLASSIFIER_HASHES | {"loader", "native_classification_calls", "native_generation_calls"}),
)
def test_classifier_harness_fields_cannot_be_missing(document, field):
    record = classifier_evidence_record()
    record["native_build"].pop(field)
    document.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(record["repo"], "bfloat16", HASHES)


def test_language_record_cannot_use_classifier_harness_schema(document):
    record = classifier_evidence_record()
    record["repo"] = REPO
    document.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    with pytest.raises(ConversionError):
        qualifications.qualification_for(REPO, "bfloat16", HASHES)
