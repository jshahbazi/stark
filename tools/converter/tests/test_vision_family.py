"""Family admission and conversion orchestration; synthetic tensors prove no parity."""

import hashlib
import json
import math
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_cli import source_tooling  # noqa: F401
from test_streaming import save_tensors, tensor_bytes
from test_vision_conversion import vision_source as pinned_vision_source  # noqa: F401

from stark_convert import cli, vision_family
from stark_convert.adapters.qwen35 import normalize_qwen35_config
from stark_convert.contracts import ConversionRequest
from stark_convert.profiles import GENERIC_PROFILE
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header
from stark_convert.vision_contract import REPO
from stark_convert.vision_family import FAMILY_PROFILE, FAMILY_VERSION, family_source_files, validate_family_source

FINE_TUNE = "example/merged-qwen35"
REVISION = "a" * 40


@pytest.fixture
def family_source(request, monkeypatch, tmp_path):
    # Tiny dimensions make full atomic conversion model-free. The production
    # kernel constraints are explicitly reduced only in this fixture, so
    # successful conversion does not establish native support or parity.
    source = request.getfixturevalue("pinned_vision_source")
    config = json.loads((source / "config.json").read_text())
    config["vision_config"]["num_position_embeddings"] = 4
    (source / "config.json").write_text(json.dumps(config))
    weights = source / "model.safetensors"
    _, infos = read_safetensors_header(weights)
    values = {}
    for key, info in infos.items():
        array = (
            np.frombuffer(tensor_bytes(info), dtype={"BF16": "<u2", "F32": "<f4"}[info.dtype])
            .copy()
            .reshape(info.shape)
        )
        values[key] = (info.dtype, array[:4] if key == "model.visual.pos_embed.weight" else array)
    save_tensors(weights, values)
    text = normalize_qwen35_config(config)
    monkeypatch.setattr(vision_family, "_TEXT_ALIGNMENT", 4)
    monkeypatch.setattr(vision_family, "_ATTENTION_HEAD_DIM", 2)
    monkeypatch.setattr(vision_family, "_DELTA_HEAD_DIM", 2)
    monkeypatch.setattr(vision_family, "_VISION_HEAD_ALIGNMENT", 2)
    monkeypatch.setattr(vision_family, "_ROPE", text["rope_parameters"])
    monkeypatch.setattr(vision_family, "_PROCESSOR", json.loads((source / "preprocessor_config.json").read_text()))
    monkeypatch.setattr(
        vision_family, "_SOURCE_TOKENIZER_SHA256", hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        vision_family,
        "_TOKENIZER_PROGRAM_FINGERPRINTS",
        {vision_family.tokenizer_program_fingerprint((source / "tokenizer.json").read_bytes())},
    )
    monkeypatch.setattr(
        vision_family, "_TEMPLATE_SHA256", hashlib.sha256((source / "chat_template.jinja").read_bytes()).hexdigest()
    )
    from tokenizers import Tokenizer

    from stark_convert.metadata import _add_tokens

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    _add_tokens(tokenizer, json.loads((source / "tokenizer_config.json").read_text()))
    expected = tmp_path / "expected-tokenizer.json"
    tokenizer.save(str(expected))
    monkeypatch.setattr(cli, "OUTPUT_TOKENIZER_SHA256", hashlib.sha256(expected.read_bytes()).hexdigest())
    return source


def snapshot(source, *, repo=FINE_TUNE):
    return resolve_source(source, repo=repo, artifact_mode="vision-language", offline=True)


def test_family_source_requires_its_own_processor_and_template():
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    )
    rules = family_source_files(FINE_TUNE, names)
    assert {rule["repo"] for rule in rules} == {FINE_TUNE}
    assert {rule["filename"] for rule in rules} == set(names)
    for absent in ("preprocessor_config.json", "chat_template.jinja"):
        with pytest.raises(ConversionError, match="incomplete"):
            family_source_files(FINE_TUNE, tuple(name for name in names if name != absent))


@pytest.mark.parametrize("repo", [None, FINE_TUNE])
def test_local_family_auto_resolution_and_manifest_replay(family_source, tmp_path, repo):
    source = snapshot(family_source, repo=repo)
    assert source.conversion_profile == {"architecture": "qwen3_5", "metadata_profile": GENERIC_PROFILE}
    assert source.manifest["schema_version"] == 3
    assert {record["role"] for record in source.manifest["files"]} >= {"processor_config", "chat_template", "weights"}
    manifest = tmp_path / "family-manifest.json"
    source.write_manifest(manifest)
    replay = resolve_source(family_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)
    assert replay.receipt() == source.receipt()
    contract = validate_family_source(replay, "bfloat16")
    assert contract["schema_version"] == 3
    assert contract["family"] == {"profile": FAMILY_PROFILE, "version": FAMILY_VERSION}
    assert contract["source"]["repo"] == (repo or "local")
    assert contract["source"]["revision"] is None
    assert contract["qualification"] == "unverified"
    assert str(tmp_path) not in json.dumps(contract)


@pytest.mark.parametrize("repo", [None, FINE_TUNE])
def test_family_conversion_binds_all_output_components_without_inheriting_qualification(family_source, tmp_path, repo):
    output = tmp_path / "output"
    receipt = cli.convert(family_source, output, repo=repo, artifact_mode="vision-language", chunk_bytes=4096)
    config = json.loads((output / "config.json").read_text())
    contract = config["artifact_contract"]
    assert receipt["artifact_contract"] == contract
    assert contract["schema_version"] == 3
    assert contract["component_hashes"] == {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in vision_family.COMPONENT_FILES
    }
    assert config["token_codec_config"]["prompt_template"] == (output / "chat_template.jinja").read_text()
    assert receipt["native_qualification"] is None and receipt["capability"]["execution"] == "unverified"
    assert receipt["catalogued"] is False
    assert receipt["metadata_profile"] == GENERIC_PROFILE
    assert str(tmp_path) not in json.dumps(receipt)


@pytest.mark.parametrize("nested", [False, True])
def test_family_normalizes_recognized_processor_serializations(family_source, tmp_path, nested):
    source_path = family_source / "preprocessor_config.json"
    canonical = json.loads(source_path.read_text())
    processor = {**canonical, **vision_family._PROCESSOR_DEFAULTS}
    if nested:
        source_path.unlink()
        source_path = family_source / "processor_config.json"
        processor = {
            "processor_class": processor.pop("processor_class"),
            "image_processor": processor,
            "video_processor": {"video_processor_type": "Qwen3VLVideoProcessor"},
        }
    source_path.write_text(json.dumps(processor))
    output = tmp_path / "output"
    receipt = cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    contract = receipt["artifact_contract"]
    assert contract["processor_config"] == canonical
    assert json.loads((output / "preprocessor_config.json").read_text()) == canonical
    assert contract["source"]["hashes"][source_path.name] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert len(set(contract["source"]["hashes"]) & {"preprocessor_config.json", "processor_config.json"}) == 1
    if nested:
        assert "preprocessor_config.json" not in contract["source"]["hashes"]
        assert not (output / "processor_config.json").exists()
    manifest = tmp_path / "variant-manifest.json"
    source = snapshot(family_source)
    source.write_manifest(manifest)
    replay = resolve_source(family_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)
    assert replay.receipt() == source.receipt()


@pytest.mark.parametrize("mutation", ["disabled", "unknown_image", "unknown_top", "disagrees", "nested_class"])
def test_family_rejects_changed_or_ambiguous_processor_features(family_source, mutation):
    source_path = family_source / "preprocessor_config.json"
    processor = json.loads(source_path.read_text())
    if mutation == "disabled":
        processor["do_normalize"] = False
    elif mutation == "unknown_image":
        processor["crop_size"] = {"height": 32, "width": 32}
    else:
        nested = {"processor_class": processor.pop("processor_class"), "image_processor": processor}
        if mutation == "unknown_top":
            nested["custom_image_code"] = "unsupported"
        elif mutation == "nested_class":
            nested["image_processor"]["processor_class"] = "UnsupportedProcessor"
        elif mutation == "disagrees":
            nested["image_processor"]["image_mean"] = [0.0, 0.5, 0.5]
        (family_source / "processor_config.json").write_text(json.dumps(nested))
        if mutation != "disagrees":
            source_path.unlink()
    if mutation in {"disabled", "unknown_image"}:
        source_path.write_text(json.dumps(processor))
    with pytest.raises((UnsupportedConversion, ConversionError), match="processor"):
        validate_family_source(snapshot(family_source), "bfloat16")


def test_family_binds_both_equivalent_own_processor_serializations(family_source, tmp_path):
    original = json.loads((family_source / "preprocessor_config.json").read_text())
    nested = {
        "processor_class": original["processor_class"],
        "image_processor": {**original, **vision_family._PROCESSOR_DEFAULTS},
    }
    path = family_source / "processor_config.json"
    path.write_text(json.dumps(nested))
    receipt = cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    hashes = receipt["artifact_contract"]["source"]["hashes"]
    assert {"processor_config.json", "preprocessor_config.json"} <= set(hashes)
    assert hashes["processor_config.json"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["artifact_contract"]["processor_config"] == original


def test_cataloged_dense_vision_source_uses_own_metadata_and_replays(family_source, tmp_path):
    repo = "Qwen/Qwen3.5-4B"
    source = snapshot(family_source, repo=repo)
    assert {record["repo"] for record in source.manifest["files"]} == {repo}
    assert source.config_path == family_source / "config.json"
    manifest = tmp_path / "cataloged-vision.json"
    source.write_manifest(manifest)
    replay = resolve_source(family_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)
    assert replay.receipt() == source.receipt()
    receipt = cli.convert(replay, tmp_path / "output", repo=repo, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["family"] == {"profile": FAMILY_PROFILE, "version": 2}
    assert receipt["native_qualification"] is None


def test_family_emits_untied_embeddings_and_normalizes_the_vision_type_alias(family_source, tmp_path):
    path = family_source / "config.json"
    config = json.loads(path.read_text())
    config["tie_word_embeddings"] = False
    config["text_config"]["tie_word_embeddings"] = False
    config["vision_config"]["model_type"] = "qwen3_5_vision"
    path.write_text(json.dumps(config))
    output = tmp_path / "output"
    receipt = cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    published = json.loads((output / "config.json").read_text())
    assert published["decoder_config"]["embedding_config"]["type"] == "UntiedEmbeddingConfig"
    assert receipt["artifact_contract"]["vision_config"]["model_type"] == "qwen3_5"
    _, tensors = read_safetensors_header(output / "model.safetensors")
    assert "decoder.embedding.input_embedding.weights" in tensors
    assert "decoder.embedding.output_embedding.weights" in tensors


def test_family_accepts_recognized_normalized_source_tokenizer(family_source, tmp_path, monkeypatch):
    from tokenizers import Tokenizer

    from stark_convert.metadata import _add_tokens

    path = family_source / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    _add_tokens(tokenizer, json.loads((family_source / "tokenizer_config.json").read_text()))
    tokenizer.save(str(path))
    monkeypatch.setattr(
        vision_family, "_NORMALIZED_SOURCE_TOKENIZER_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    # The tiny fixture is already normalized; disable its first accepted hash
    # so this exercises the independent normalized-source admission branch.
    monkeypatch.setattr(vision_family, "_SOURCE_TOKENIZER_SHA256", "0" * 64)
    receipt = cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["source"]["hashes"]["tokenizer.json"] == cli.OUTPUT_TOKENIZER_SHA256


@pytest.mark.parametrize("mutation", [None, "symbolic", "top_token", "split", "decoder"])
def test_family_recognizes_embedded_added_tokens_only_with_consistent_metadata(
    family_source, tmp_path, monkeypatch, mutation
):
    path = family_source / "tokenizer_config.json"
    token_config = json.loads(path.read_text())
    token_config.pop("added_tokens_decoder")
    token_config["model_specific_special_tokens"] = {"image_token": "<|image_pad|>"}
    if mutation == "symbolic":
        token_config["model_specific_special_tokens"]["image_token"] = "<|video_pad|>"
    elif mutation == "top_token":
        token_config["vision_bos_token"] = "<|vision_end|>"
    elif mutation == "split":
        token_config["split_special_tokens"] = True
    elif mutation == "decoder":
        token_config["added_tokens_decoder"] = {}
    path.write_text(json.dumps(token_config))
    monkeypatch.setattr(
        vision_family,
        "_NORMALIZED_SOURCE_TOKENIZER_SHA256",
        hashlib.sha256((family_source / "tokenizer.json").read_bytes()).hexdigest(),
    )
    if mutation is None:
        receipt = cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
        assert receipt["artifact_contract"]["schema_version"] == 3
    else:
        with pytest.raises(ConversionError, match="tokenizer or template metadata"):
            validate_family_source(snapshot(family_source), "bfloat16")


def test_family_requires_the_recognized_normalized_tokenizer_for_embedded_metadata(family_source):
    path = family_source / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.pop("added_tokens_decoder")
    path.write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="tokenizer or template metadata"):
        validate_family_source(snapshot(family_source), "bfloat16")


def test_family_accepts_equivalent_tokenizer_serialization_and_preserves_raw_provenance(family_source, tmp_path):
    path = family_source / "tokenizer.json"
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(json.loads(path.read_text()), sort_keys=True, separators=(",", ":")))
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_hash != original_hash
    output = tmp_path / "output"
    receipt = cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["source"]["hashes"]["tokenizer.json"] == actual_hash
    assert receipt["artifact_contract"]["component_hashes"]["tokenizer.json"] == cli.OUTPUT_TOKENIZER_SHA256


def test_family_does_not_construct_an_unknown_tokenizer_program(family_source, monkeypatch):
    path = family_source / "tokenizer.json"
    value = json.loads(path.read_text())
    value["normalizer"] = {"type": "Lowercase"}
    path.write_text(json.dumps(value))
    monkeypatch.setattr(
        vision_family, "Tokenizer", SimpleNamespace(from_str=lambda *_args: pytest.fail("unknown program"))
    )
    with pytest.raises(UnsupportedConversion, match="outside the implemented profile"):
        validate_family_source(snapshot(family_source), "bfloat16")


def test_family_accepts_only_the_reviewed_alternate_template_hash(family_source, tmp_path, monkeypatch):
    path = family_source / "chat_template.jinja"
    template = path.read_text() + "\n{# synthetic reviewed variant #}"
    path.write_text(template)
    token_path = family_source / "tokenizer_config.json"
    token_config = json.loads(token_path.read_text())
    token_config["chat_template"] = template
    token_path.write_text(json.dumps(token_config))
    monkeypatch.setattr(vision_family, "_MAPPING_TEMPLATE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    output = tmp_path / "output"
    cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    assert (output / "chat_template.jinja").read_text() == template
    assert json.loads((output / "config.json").read_text())["token_codec_config"]["prompt_template"] == template
    path.write_text(template + " ")
    with pytest.raises(UnsupportedConversion, match="outside the implemented profile"):
        validate_family_source(snapshot(family_source), "bfloat16")


@pytest.mark.parametrize("thinking", [False, True])
def test_family_declares_the_known_template_reasoning_control(family_source, tmp_path, monkeypatch, thinking):
    if thinking:
        path = family_source / "chat_template.jinja"
        path.write_text(path.read_text() + "\n{# synthetic default-on variant #}")
        monkeypatch.setattr(vision_family, "_THINKING_TEMPLATE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    output = tmp_path / "output"
    cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    codec = json.loads((output / "config.json").read_text())["token_codec_config"]
    assert codec["reasoning_config"] == {
        "default_reasoning_effort": "medium" if thinking else "no_reasoning",
        "field_name": "enable_thinking",
        "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
    }


def test_family_request_reports_implemented_but_unverified_native_support():
    request = ConversionRequest(FINE_TUNE, artifact_mode="vision-language")
    capability = cli.capability(request)
    assert capability.adapter == "qwen3_5" and capability.conversion == "implemented"
    assert capability.native_loading == "supported_family_unverified"
    assert capability.source_validation == "not_performed"
    assert cli.capability(replace(request, metadata_profile=GENERIC_PROFILE)).conversion == "implemented"
    assert cli.capability(replace(request, dtype="float16")).conversion == "unsupported"


@pytest.mark.parametrize("initializer", [0, 0.0, 0.02, 0.1])
def test_family_ignores_known_training_initializer_without_changing_execution(family_source, initializer):
    path = family_source / "config.json"
    value = json.loads(path.read_text())
    value["vision_config"]["initializer_range"] = initializer
    path.write_text(json.dumps(value))
    contract = validate_family_source(snapshot(family_source), "bfloat16")
    assert contract["vision_config"]["initializer_range"] == initializer


@pytest.mark.parametrize("initializer", [True, -1, "0.02", float("inf"), float("nan")])
def test_family_rejects_invalid_training_metadata(family_source, initializer):
    path = family_source / "config.json"
    value = json.loads(path.read_text())
    value["vision_config"]["initializer_range"] = initializer
    path.write_text(json.dumps(value))
    nonfinite = isinstance(initializer, float) and not math.isfinite(initializer)
    with pytest.raises(
        ConversionError if nonfinite else UnsupportedConversion,
        match="metadata is invalid" if nonfinite else "tower operations",
    ):
        validate_family_source(snapshot(family_source), "bfloat16")


def test_family_normalizes_known_vision_defaults(family_source):
    path = family_source / "config.json"
    value = json.loads(path.read_text())
    for key in ("initializer_range", "hidden_act", "deepstack_visual_indexes"):
        value["vision_config"].pop(key)
    path.write_text(json.dumps(value))
    vision = validate_family_source(snapshot(family_source), "bfloat16")["vision_config"]
    assert vision["initializer_range"] == 0.02
    assert vision["hidden_act"] == "gelu_pytorch_tanh"
    assert vision["deepstack_visual_indexes"] == []


@pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_family_vision_dtype_serialization_is_only_bf16_and_not_an_execution_field(
    family_source, tmp_path, key, dtype
):
    path = family_source / "config.json"
    config = json.loads(path.read_text())
    config["vision_config"][key] = dtype
    path.write_text(json.dumps(config))
    if dtype == "bfloat16":
        receipt = cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
        contract = receipt["artifact_contract"]
        assert key not in contract["vision_config"]
    else:
        with pytest.raises(UnsupportedConversion, match="tower operations"):
            validate_family_source(snapshot(family_source), "bfloat16")


@pytest.mark.parametrize(
    "annotations",
    [
        {"model_name": "unsloth/Qwen3.5-0.8B", "unsloth_fixed": True, "unsloth_version": "2026.3.4"},
        {"model_name": {}},
        {"model_name": "x" * 257},
        {"unsloth_fixed": "true"},
        {"unsloth_version": "x" * 65},
    ],
)
def test_family_recognizes_only_bounded_training_annotations(family_source, tmp_path, annotations):
    path = family_source / "config.json"
    config = json.loads(path.read_text())
    config.update(annotations)
    path.write_text(json.dumps(config))
    if annotations.get("unsloth_fixed") is True:
        output = tmp_path / "output"
        cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
        published = json.loads((output / "config.json").read_text())
        assert all(key not in published["decoder_config"] for key in annotations)
    else:
        with pytest.raises(ConversionError, match="training annotation"):
            cli.plan_conversion(snapshot(family_source), ConversionRequest(FINE_TUNE, artifact_mode="vision-language"))


def replace_dtype(source, name, dtype):
    path = source / "model.safetensors"
    _, infos = read_safetensors_header(path)
    values = {
        key: (
            info.dtype,
            np.frombuffer(tensor_bytes(info), dtype={"BF16": "<u2", "F32": "<f4"}[info.dtype])
            .copy()
            .reshape(info.shape),
        )
        for key, info in infos.items()
    }
    values[name] = (dtype, np.zeros(infos[name].shape, dtype={"F16": "<f2", "F32": "<f4"}[dtype]))
    save_tensors(path, values)


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("model.language_model.embed_tokens.weight", "F16"),
        ("model.language_model.layers.0.linear_attn.conv1d.weight", "F32"),
    ],
)
def test_family_rejects_non_bf16_matrices_and_convolutions(family_source, tmp_path, name, dtype):
    replace_dtype(family_source, name, dtype)
    with pytest.raises(UnsupportedConversion, match="merged BF16 matrices"):
        cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()


def test_family_permits_explicit_f32_normalization_weights(family_source, tmp_path):
    replace_dtype(family_source, "model.language_model.norm.weight", "F32")
    receipt = cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["schema_version"] == 3
    assert receipt["native_qualification"] is None


def test_family_permits_unused_padded_embedding_and_output_rows(family_source, tmp_path, monkeypatch):
    path = family_source / "config.json"
    value = json.loads(path.read_text())
    value["text_config"]["vocab_size"] = 20
    path.write_text(json.dumps(value))
    weights = family_source / "model.safetensors"
    _, infos = read_safetensors_header(weights)
    tensors = {
        name: (
            info.dtype,
            np.frombuffer(tensor_bytes(info), dtype={"BF16": "<u2", "F32": "<f4"}[info.dtype])
            .copy()
            .reshape(info.shape),
        )
        for name, info in infos.items()
    }
    for name in tensors:
        if name.endswith(("embed_tokens.weight", "lm_head.weight")):
            tensors[name] = ("BF16", np.zeros((20, 8), dtype="<u2"))
    save_tensors(weights, tensors)
    receipt = cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert receipt["artifact_contract"]["schema_version"] == 3


def test_remote_family_pins_all_sidecars_to_the_actual_source_revision(family_source):
    calls = []

    class Hub:
        def HfApi(self):
            return self

        def model_info(self, *, repo_id, revision):
            assert repo_id == FINE_TUNE and revision is None
            return SimpleNamespace(sha=REVISION)

        def list_repo_files(self, *, repo_id, revision):
            assert repo_id == FINE_TUNE and revision == REVISION
            return [path.name for path in family_source.iterdir() if path.is_file()]

        def hf_hub_download(self, *, repo_id, filename, revision, local_files_only):
            assert repo_id == FINE_TUNE and revision == REVISION and not local_files_only
            calls.append(filename)
            return str(family_source / filename)

        def hf_hub_url(self, *, repo_id, filename, revision):
            return filename

        def get_hf_file_metadata(self, url):
            return SimpleNamespace(size=(family_source / url).stat().st_size)

    source = resolve_source(FINE_TUNE, artifact_mode="vision-language", hub=Hub())
    assert source.revision == REVISION and source.conversion_profile["metadata_profile"] == GENERIC_PROFILE
    assert {record["repo"] for record in source.manifest["files"]} == {FINE_TUNE}
    assert {record["revision"] for record in source.manifest["files"]} == {REVISION}
    assert {"preprocessor_config.json", "chat_template.jinja"} <= set(calls)
    assert validate_family_source(source, "bfloat16")["source"]["revision"] == REVISION


@pytest.mark.parametrize("shared_cache", [False, True])
def test_remote_family_resolves_huggingface_snapshot_symlinks_automatically(family_source, tmp_path, shared_cache):
    cache = tmp_path / "hub" / "models--example--merged-qwen35"
    cached_source = cache / "snapshots" / REVISION
    cached_source.mkdir(parents=True)
    blobs = cache / "blobs"
    blobs.mkdir()
    for path in family_source.iterdir():
        if path.is_file():
            blob = blobs / hashlib.sha256(path.read_bytes()).hexdigest()
            if shared_cache:
                shared_blob = cache.parent / "blobs" / blob.name[:2] / blob.name
                shared_blob.parent.mkdir(parents=True, exist_ok=True)
                shared_blob.write_bytes(path.read_bytes())
                blob.symlink_to(shared_blob)
            else:
                blob.write_bytes(path.read_bytes())
            (cached_source / path.name).symlink_to(blob)

    class Hub:
        def HfApi(self):
            return self

        def model_info(self, *, repo_id, revision):
            assert repo_id == FINE_TUNE and revision == REVISION
            return SimpleNamespace(sha=REVISION)

        def list_repo_files(self, *, repo_id, revision):
            return [path.name for path in cached_source.iterdir()]

        def hf_hub_download(self, *, repo_id, filename, revision, local_files_only):
            return str(cached_source / filename)

        def hf_hub_url(self, *, repo_id, filename, revision):
            return filename

        def get_hf_file_metadata(self, url):
            return SimpleNamespace(size=(cached_source / url).stat().st_size)

    source = resolve_source(FINE_TUNE, revision=REVISION, artifact_mode="vision-language", hub=Hub())
    assert source.conversion_profile == {"architecture": "qwen3_5", "metadata_profile": GENERIC_PROFILE}
    assert source.config_path.is_relative_to(cache.parent / "blobs" if shared_cache else blobs)
    assert validate_family_source(source, "bfloat16")["source"]["revision"] == REVISION
    manifest = tmp_path / "hub-source-manifest.json"
    source.write_manifest(manifest)
    replay = resolve_source(cached_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)
    assert replay.receipt() == source.receipt()
    request = ConversionRequest(FINE_TUNE, artifact_mode="vision-language")
    plan = cli.plan_conversion(replay, request)
    assert plan.request.metadata_profile == GENERIC_PROFILE
    assert plan.request.architecture == "qwen3_5"


def test_family_snapshot_symlink_does_not_authorize_an_unrelated_config(family_source, tmp_path):
    cache = tmp_path / "hub" / "models--example--merged-qwen35"
    cached_source = cache / "snapshots" / REVISION
    cached_source.mkdir(parents=True)
    for path in family_source.iterdir():
        if path.is_file():
            (cached_source / path.name).write_bytes(path.read_bytes())
    config = cached_source / "config.json"
    config.unlink()
    config.symlink_to(family_source / "config.json")
    with pytest.raises(ConversionError, match="authorized source roots"):
        resolve_source(cached_source, artifact_mode="vision-language", offline=True)


def test_cli_local_family_requires_no_manual_repo_or_metadata_profile(family_source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "automatic"
    monkeypatch.setattr(
        "sys.argv",
        ["stark-convert", str(family_source), "--artifact-mode", "vision-language", "--output", str(output)],
    )
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert receipt["artifact_contract"]["schema_version"] == 3
    assert receipt["artifact_contract"]["source"]["repo"] == "local"
    assert receipt["native_qualification"] is None
    assert str(tmp_path) not in captured.out


def test_family_binds_its_optional_generation_metadata_and_replays_it(family_source, tmp_path):
    generation = family_source / "generation_config.json"
    generation.write_text(json.dumps({"eos_token_id": 1, "temperature": 0.5}))
    source = snapshot(family_source)
    contract = validate_family_source(source, "bfloat16")
    assert (
        contract["source"]["hashes"]["generation_config.json"] == hashlib.sha256(generation.read_bytes()).hexdigest()
    )
    manifest = tmp_path / "with-generation.json"
    source.write_manifest(manifest)
    replay = resolve_source(family_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)
    assert replay.receipt() == source.receipt()
    generation.write_text(json.dumps({"eos_token_id": 2}))
    with pytest.raises(ConversionError):
        resolve_source(family_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)


def test_family_rejects_changed_normalization_even_when_raw_tokenizer_is_recognized(family_source, tmp_path):
    path = family_source / "tokenizer_config.json"
    value = json.loads(path.read_text())
    value["added_tokens_decoder"]["1"] = {
        "content": "[SYNTHETIC_1]",
        "special": True,
        "normalized": False,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
    }
    path.write_text(json.dumps(value))
    with pytest.raises(UnsupportedConversion, match="normalized tokenizer"):
        cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()


def test_family_cannot_claim_exact_native_qualification_before_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "resolve_source", lambda *args, **kwargs: pytest.fail("resolved a qualified family request")
    )
    with pytest.raises(UnsupportedConversion, match="exact recorded"):
        cli.convert(
            tmp_path / "missing",
            tmp_path / "output",
            repo=FINE_TUNE,
            artifact_mode="vision-language",
            require_native_support=True,
        )


@pytest.mark.parametrize(
    "mutation", ["decoder", "tower", "processor", "rope", "operations", "quantization", "remote_code"]
)
def test_family_rejects_unsupported_geometry_and_semantics_before_publication(family_source, tmp_path, mutation):
    path = family_source / ("preprocessor_config.json" if mutation == "processor" else "config.json")
    data = json.loads(path.read_text())
    if mutation == "decoder":
        data["text_config"]["hidden_size"] += 1
    elif mutation == "tower":
        data["vision_config"]["out_hidden_size"] += 1
    elif mutation == "processor":
        data["image_mean"][0] = 0.0
    elif mutation == "rope":
        data["text_config"]["rope_parameters"]["mrope_interleaved"] = False
    elif mutation == "operations":
        data["text_config"]["attention_bias"] = True
    elif mutation == "quantization":
        data["quantization_config"] = {"bits": 4}
    else:
        data["auto_map"] = {"AutoModel": "private.module"}
    path.write_text(json.dumps(data))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


@pytest.mark.parametrize("mutation", ["id", "special", "normalized", "duplicate", "template", "tokenizer_json"])
def test_family_rejects_inconsistent_tokenizer_and_template(family_source, tmp_path, mutation):
    path = family_source / "tokenizer_config.json"
    data = json.loads(path.read_text())
    if mutation == "id":
        data["added_tokens_decoder"]["10"]["id"] = 11
    elif mutation in {"special", "normalized"}:
        data["added_tokens_decoder"]["10"][mutation] = mutation == "normalized"
    elif mutation == "duplicate":
        data["added_tokens_decoder"]["11"]["content"] = "<|vision_start|>"
    elif mutation == "template":
        data["chat_template"] = "private conflicting template"
    else:
        path = family_source / "tokenizer.json"
    path.write_text("private invalid tokenizer" if mutation == "tokenizer_json" else json.dumps(data))
    expected = UnsupportedConversion if mutation == "tokenizer_json" else ConversionError
    with pytest.raises(expected, match="Vision family tokenizer or template"):
        validate_family_source(snapshot(family_source), "bfloat16")


def test_family_metadata_size_bound_precedes_read(tmp_path):
    path = tmp_path / "oversized"
    with path.open("wb") as stream:
        stream.truncate(1024 * 1024 + 1)
    source = SimpleNamespace(resolve_file=lambda _name: path)
    with pytest.raises(ConversionError, match="size limit"):
        vision_family._read_metadata(source, "config.json")


@pytest.mark.parametrize(
    "name,maximum",
    [("preprocessor_config.json", 65536), ("processor_config.json", 65536), ("chat_template.jinja", 262144)],
)
def test_family_sidecar_bounds_match_native_admission(tmp_path, name, maximum):
    path = tmp_path / name
    with path.open("wb") as stream:
        stream.truncate(maximum + 1)
    source = SimpleNamespace(resolve_file=lambda _name: path)
    with pytest.raises(ConversionError, match="size limit"):
        vision_family._read_metadata(source, name)


def test_family_bounds_complete_configuration_before_writing_weights(family_source, tmp_path, monkeypatch):
    path = family_source / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config["eos_token"] = "x" * 1_040_000
    path.write_text(json.dumps(config))
    assert path.stat().st_size < 1024 * 1024
    monkeypatch.setattr(
        cli, "write_plan", lambda *args, **kwargs: pytest.fail("wrote weights for oversized configuration")
    )
    with pytest.raises(UnsupportedConversion, match="configuration exceeds"):
        cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()


def test_family_does_not_copy_source_header_metadata_into_native_output(family_source, tmp_path):
    path = family_source / "model.safetensors"
    data = path.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + size])
    header["__metadata__"] = {"synthetic-extra-metadata": "x" * (1024 * 1024)}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + size :])
    output = tmp_path / "output"
    cli.convert(family_source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    with (output / "model.safetensors").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] <= 1024 * 1024
    metadata, _ = read_safetensors_header(output / "model.safetensors")
    assert "synthetic-extra-metadata" not in metadata


def test_family_rejects_changed_output_payload_before_publication(family_source, tmp_path, monkeypatch):
    original = cli.write_plan

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = args[1]
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        return result

    monkeypatch.setattr(cli, "write_plan", changed)
    with pytest.raises(ConversionError, match="weights changed during conversion"):
        cli.convert(family_source, tmp_path / "output", repo=FINE_TUNE, artifact_mode="vision-language")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("label", ["shards/model-00001.safetensors", "model.safetensors"])
def test_family_permits_bounded_relative_weight_provenance(family_source, label):
    source = snapshot(family_source)
    for record in source.manifest["files"]:
        if record["role"] == "weights":
            record["filename"] = label
    assert label in validate_family_source(source, "bfloat16")["source"]["weight_files"]


@pytest.mark.parametrize("label", ["/abs", "a//b", "a/./b", "a/../b", "a\\b", "has space", "unicodé", "a" * 257])
def test_family_rejects_weight_provenance_labels_outside_native_limits(family_source, label):
    source = snapshot(family_source)
    for record in source.manifest["files"]:
        if record["role"] == "weights":
            record["filename"] = label
    with pytest.raises(ConversionError, match="weight provenance"):
        validate_family_source(source, "bfloat16")


def test_family_rejects_too_many_weight_provenance_records(family_source):
    source = snapshot(family_source)
    weight = next(record for record in source.manifest["files"] if record["role"] == "weights")
    source.manifest["files"] = [record for record in source.manifest["files"] if record["role"] != "weights"] + [
        {**weight, "filename": f"shards/{index}.safetensors"} for index in range(257)
    ]
    with pytest.raises(ConversionError, match="weight provenance"):
        validate_family_source(source, "bfloat16")


def test_official_source_identity_stays_on_its_pinned_path(family_source):
    snapshot(family_source, repo=REPO)
    from stark_convert.vision_contract import validate_vision_source

    # The synthetic fixture pins its own official payloads solely to exercise
    # schema-2 orchestration. A file mutation still cannot relabel it a fine-tune.
    weights = family_source / "model.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)
    with pytest.raises(ConversionError, match="pinned artifact"):
        validate_vision_source(resolve_source(family_source, repo=REPO, artifact_mode="vision-language"), "bfloat16")


def test_family_replay_requires_its_processor_and_profile(family_source, tmp_path):
    source = snapshot(family_source)
    for mutation in ("processor", "profile", "mode"):
        value = json.loads(json.dumps(source.manifest))
        if mutation == "processor":
            value["files"] = [record for record in value["files"] if record["role"] != "processor_config"]
        elif mutation == "profile":
            value.pop("conversion_profile")
        else:
            value["artifact_mode"] = "text"
        manifest = tmp_path / (mutation + ".json")
        manifest.write_text(json.dumps(value))
        with pytest.raises(ConversionError):
            resolve_source(family_source, source_manifest=manifest, artifact_mode="vision-language", offline=True)
