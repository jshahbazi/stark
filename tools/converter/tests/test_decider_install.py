"""Pinned Decider source and catalog admission without network or model execution."""

import hashlib
import json
from types import SimpleNamespace

from stark_convert import decider_install, model_catalog, model_support
from stark_convert.decider_protocol import PROTOCOL
from stark_convert.decider_specs import REGULAR_DECIDER


def raw_source(path, *, generic=False, regular=False):
    path.mkdir()
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5Model"],
        "tie_word_embeddings": not regular,
        "text_config": {"rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 2, 2]}},
        "vision_config": {"rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"}},
    }
    decision = {
        "format_version": 2,
        "base_model": REGULAR_DECIDER.base_model if regular else "Qwen/Qwen3.5-4B",
        "model_type": "qwen3_5",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "codes": [f"code-{index}" for index in range(255)],
        "token_ids": list(range(255)),
        "temperature": 1.25,
        "max_pixels": 1638400,
    }
    processor = {
        "processor_class": "Qwen3VLProcessor",
        "image_processor": decider_install._EXPECTED_IMAGE_PROCESSOR,
    }
    values = {
        "config.json": config,
        "decision_config.json": decision,
        "joint_config.json": PROTOCOL,
        "processor_config.json": processor,
        "model.safetensors.index.json": {
            "weight_map": {
                f"{prefix}.weight": f"model-{index:05d}-of-{11 if regular else 2:05d}.safetensors"
                for index, prefix in enumerate(
                    ["language_model", *(f"visual.{index}" for index in range(10))]
                    if regular
                    else ["language_model", "visual"],
                    start=1,
                )
            }
        },
    }
    if generic:
        values.pop("decision_config.json")
    for name, value in values.items():
        (path / name).write_text(json.dumps(value))
    return path


def test_source_recognition_is_exact_task_and_base_model_only(tmp_path):
    source = raw_source(tmp_path / "source")
    assert decider_install.is_decider_source(source, kind="local")
    assert not decider_install.is_decider_source(source, kind="local", revision="a" * 40)
    generic = raw_source(tmp_path / "generic", generic=True)
    assert not decider_install.is_decider_source(generic, kind="local")
    assert not decider_install.is_decider_source("Qwen/Qwen3.5-4B", kind="huggingface")


def test_local_preflight_reports_decision_only_pinned_capabilities(tmp_path, monkeypatch):
    source = raw_source(tmp_path / "source")
    monkeypatch.setattr(decider_install, "_disk_check", lambda *_args: None)
    result = decider_install.preflight(str(source), kind="local")
    assert result["status"] == "conditional"
    assert result["mode"] == "decision" and result["model_kind"] == "decision"
    assert result["source"] == {
        "kind": "local",
        "repo": decider_install.SOURCE_REPO,
        "revision": decider_install.SOURCE_REVISION,
    }
    assert result["modes"] == {"text": "unsupported", "decision": "conditional", "vision-language": "unsupported"}
    assert result["capabilities"] == {"decision": True, "images": True, "max_images": 1}


def test_huggingface_preflight_rejects_other_qwen_before_hub_access():
    class NoNetwork:
        def hf_hub_download(self, **_kwargs):
            raise AssertionError("must not access arbitrary sources")

    result = decider_install.preflight("Qwen/Qwen3.5-4B", kind="huggingface", hub=NoNetwork())
    assert result["status"] == "unsupported"
    assert result["source"]["repo"] == decider_install.SOURCE_REPO


def test_hf_snapshot_metadata_accepts_only_snapshot_to_blob_symlinks(tmp_path, monkeypatch):
    source = raw_source(tmp_path / "source")
    snapshot = (
        tmp_path / "models--perplexity-ai--pplx-decider-flash-v1.3" / "snapshots" / decider_install.SOURCE_REVISION
    )
    blobs = snapshot.parent.parent / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    pins = {}
    for name in decider_install._FILE_PINS:
        original = source / name
        if not original.exists():
            original.write_bytes((name + "\n").encode())
        data = original.read_bytes()
        algorithm = "sha256" if name.endswith(".safetensors") or name == "tokenizer.json" else "git_blob_sha1"
        digest = (
            hashlib.sha256(data).hexdigest()
            if algorithm == "sha256"
            else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        )
        pins[name] = (algorithm, digest, len(data))
        target = blobs / digest
        target.write_bytes(data)
        (snapshot / name).symlink_to(target)
    monkeypatch.setattr(decider_install, "_FILE_PINS", pins)

    class CachedHub:
        def hf_hub_download(self, *, filename, dry_run=False, **_kwargs):
            if dry_run:
                return SimpleNamespace(file_size=pins[filename][2], is_cached=True)
            return snapshot / filename

    validated = decider_install._validate_hf_metadata(
        decider_install.SOURCE_REPO, decider_install.SOURCE_REVISION, hub=CachedHub()
    )
    assert validated == snapshot.resolve()


def test_regular_source_is_exactly_recognized_and_preflighted(tmp_path, monkeypatch):
    source = raw_source(tmp_path / "regular", regular=True)
    monkeypatch.setattr(decider_install, "_disk_check", lambda *_args: None)

    assert decider_install.source_profile(source, kind="local") == REGULAR_DECIDER
    result = decider_install.preflight(str(source), kind="local")
    assert result["status"] == "conditional"
    assert result["family_id"] == REGULAR_DECIDER.family_id
    assert result["source"] == {
        "kind": "local",
        "repo": REGULAR_DECIDER.repo,
        "revision": REGULAR_DECIDER.revision,
    }


def test_support_and_catalog_keep_decider_separate_from_qwen_architecture(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    families = {row["family_id"]: row for row in model_support.support_matrix()["families"]}
    decider = families[decider_install.FAMILY_ID]
    assert decider["model_kind"] == "decision"
    assert decider["architecture_aliases"] == []
    assert decider["modes"]["decision"]["status"] == "conditional"
    assert decider["modes"]["decision"]["images"] is True
    assert decider["modes"]["text"]["status"] == "unsupported"
    assert families["qwen3_5"]["model_kind"] == "language"
    catalog = model_catalog.family_catalog(decider_install.FAMILY_ID)
    assert catalog["status"] == "ready" and len(catalog["items"]) == 1
    item = catalog["items"][0]
    assert item["repo"] == decider_install.SOURCE_REPO
    assert item["revision"] == decider_install.SOURCE_REVISION
    assert item["quantization"] == "BF16"
    assert item["artifact_modes"] == ["decision"] and item["default_artifact_mode"] == "decision"
    assert item["images"] is True and item["max_images"] == 1

    regular = families[REGULAR_DECIDER.family_id]
    assert regular["model_kind"] == "decision"
    regular_catalog = model_catalog.family_catalog(REGULAR_DECIDER.family_id)
    assert regular_catalog["status"] == "ready" and len(regular_catalog["items"]) == 1
    regular_item = regular_catalog["items"][0]
    assert regular_item["repo"] == REGULAR_DECIDER.repo
    assert regular_item["revision"] == REGULAR_DECIDER.revision
    assert regular_item["weight_bytes"] == 52_170_802_736


def test_install_verifies_pinned_file_set_before_conversion(tmp_path, monkeypatch):
    source = raw_source(tmp_path / "source")
    # Use tiny deterministic files here; production keys retain the verified
    # upstream identities and their exact full sizes.
    pins = {}
    for name in decider_install._FILE_PINS:
        path = source / name
        if not path.exists():
            path.write_bytes((name + "\n").encode())
        data = path.read_bytes()
        algorithm = "sha256" if name.endswith(".safetensors") or name == "tokenizer.json" else "git_blob_sha1"
        digest = (
            hashlib.sha256(data).hexdigest()
            if algorithm == "sha256"
            else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        )
        pins[name] = (algorithm, digest, len(data))
    monkeypatch.setattr(decider_install, "_FILE_PINS", pins)
    monkeypatch.setattr(decider_install, "_disk_check", lambda *_args: None)
    monkeypatch.setattr(decider_install, "prepare_decider", lambda *_args, **_kwargs: {"type": "DeciderModelConfig"})
    result = decider_install.install(str(source), kind="local", output=tmp_path / "out")
    assert result == {
        "prepared": True,
        "type": "DeciderModelConfig",
        "images": True,
        "revision": decider_install.SOURCE_REVISION,
    }
    (source / "NOTICE").write_text("changed")
    try:
        decider_install.install(str(source), kind="local", output=tmp_path / "second")
    except ValueError as error:
        assert str(error) == "source_identity"
    else:
        raise AssertionError("modified source identity was accepted")


def test_regular_install_verifies_its_own_file_set_and_profile(tmp_path, monkeypatch):
    source = raw_source(tmp_path / "regular", regular=True)
    pins = {}
    for name in REGULAR_DECIDER.file_pins:
        path = source / name
        if not path.exists():
            path.write_bytes((name + "\n").encode())
        data = path.read_bytes()
        pins[name] = ("sha256", hashlib.sha256(data).hexdigest(), len(data))
    monkeypatch.setattr(decider_install, "_pins", lambda profile: pins if profile == REGULAR_DECIDER else {})
    monkeypatch.setattr(decider_install, "_disk_check", lambda *_args: None)
    captured = {}

    def prepare(path, output, *, revision, profile):
        captured.update(path=path, revision=revision, profile=profile)
        return {"type": "DeciderModelConfig"}

    monkeypatch.setattr(decider_install, "prepare_decider", prepare)
    result = decider_install.install(str(source), kind="local", output=tmp_path / "out")

    assert result["revision"] == REGULAR_DECIDER.revision
    assert captured == {
        "path": source,
        "revision": REGULAR_DECIDER.revision,
        "profile": REGULAR_DECIDER,
    }
