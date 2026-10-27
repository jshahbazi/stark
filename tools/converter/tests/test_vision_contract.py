import hashlib
import json
from types import SimpleNamespace

import pytest

from stark_convert import vision_contract
from stark_convert.streaming import ConversionError
from stark_convert.vision_contract import REPO, REVISION, source_rules, validate_vision_source


def test_source_rules_text_preserves_catalog_and_vision_is_pinned():
    entry = {"repo": REPO, "files": {"model_config": {"repo": "x", "filename": "config.json"}}}
    assert source_rules(entry) == ({"role": "model_config", "repo": "x", "filename": "config.json"},)
    rules = source_rules(entry, "vision-language")
    assert {r["role"] for r in rules} == {
        "model_config",
        "tokenizer",
        "tokenizer_config",
        "processor_config",
        "chat_template",
    }
    assert all(r["repo"] == REPO for r in rules)
    assert dict((r["role"], r["filename"]) for r in rules)["processor_config"] == "preprocessor_config.json"


def test_validate_contract_with_synthetic_pinned_metadata(tmp_path, monkeypatch):
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vision_config": {"depth": 12, "patch_size": 16},
        "text_config": {"rope_parameters": {"mrope_section": [11, 11, 10], "mrope_interleaved": True}},
    }
    processor = {"processor_class": "Qwen3VLProcessor", "image_mean": [0.5] * 3}
    special = [
        {
            "content": f"<|{name}|>",
            "id": value,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for name, value in {
            "vision_start": 248053,
            "vision_end": 248054,
            "image_pad": 248056,
            "video_pad": 248057,
        }.items()
    ]
    vocab = {"<unk>": 0, **{item["content"]: item["id"] for item in special}}
    tokenizer = {
        "version": "1.0",
        "added_tokens": special,
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    payload = {
        "config.json": json.dumps(config).encode(),
        "preprocessor_config.json": json.dumps(processor).encode(),
        "chat_template.jinja": b"template",
    }
    tokenizer_config = {"added_tokens_decoder": {str(item["id"]): item for item in special}}
    for name, data in {
        **payload,
        "tokenizer.json": json.dumps(tokenizer).encode(),
        "generation_config.json": b"{}",
        "tokenizer_config.json": json.dumps(tokenizer_config).encode(),
    }.items():
        (tmp_path / name).write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    monkeypatch.setattr(vision_contract, "_HASHES", hashes)
    monkeypatch.setattr(vision_contract, "_WEIGHT_FILES", {})
    snapshot = SimpleNamespace(
        repo=REPO,
        revision=REVISION,
        manifest={"files": [{"repo": REPO, "revision": REVISION}]},
        resolve_file=lambda name, repo=None: tmp_path / name,
    )
    result = validate_vision_source(snapshot, "bfloat16")
    assert result["type"] == "Qwen35VisionLanguageArtifact"
    assert result["schema_version"] == 2
    assert result["runtime_supported"] is True
    assert result["qualification"] == "unverified"
    assert result["transport"] == "mediaref-v1"
    assert result["position_contract"]["interleaved"] is True


def test_validate_rejects_wrong_dtype_or_identity(tmp_path):
    snapshot = SimpleNamespace(
        repo=REPO, revision=REVISION, manifest={"files": []}, resolve_file=lambda *args, **kwargs: tmp_path / "x"
    )
    with pytest.raises(ConversionError, match="pinned artifact"):
        validate_vision_source(snapshot, "float16")


@pytest.mark.parametrize("mode", ["vision", "multimodal", "", None, 1])
def test_source_rules_rejects_unknown_mode(mode):
    with pytest.raises(ConversionError, match="Unsupported artifact mode"):
        source_rules({"repo": REPO, "files": {}}, mode)


def test_source_rules_rejects_foreign_repo():
    with pytest.raises(ConversionError, match="outside the pinned repository"):
        source_rules({"repo": "other/model", "files": {}}, "vision-language")


@pytest.mark.parametrize("repo,revision", [("other/model", REVISION), (REPO, "0" * 40)])
def test_validate_rejects_wrong_snapshot_identity(tmp_path, repo, revision):
    snapshot = SimpleNamespace(
        repo=repo, revision=revision, manifest={"files": []}, resolve_file=lambda *args, **kwargs: tmp_path / "x"
    )
    with pytest.raises(ConversionError, match="pinned artifact|Vision family (?:processor )?metadata"):
        validate_vision_source(snapshot, "bfloat16")


def test_validate_rejects_foreign_manifest_file(tmp_path):
    snapshot = SimpleNamespace(
        repo=REPO,
        revision=REVISION,
        manifest={"files": [{"repo": REPO, "revision": REVISION}, {"repo": "other/model", "revision": REVISION}]},
        resolve_file=lambda *args, **kwargs: tmp_path / "x",
    )
    with pytest.raises(ConversionError, match="pinned artifact"):
        validate_vision_source(snapshot, "bfloat16")


@pytest.mark.parametrize("dtype", ["float16", "float32", "BF16", None, 1])
def test_validate_rejects_non_bfloat16(dtype, tmp_path):
    snapshot = SimpleNamespace(
        repo=REPO, revision=REVISION, manifest={"files": []}, resolve_file=lambda *args, **kwargs: tmp_path / "x"
    )
    with pytest.raises(ConversionError, match="pinned artifact"):
        validate_vision_source(snapshot, dtype)
