"""Model-free Decider protocol, tensor preservation and rejection controls."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_qwen35_vision import vision_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.decider import build_decider_plan, prepare, validate_decision_metadata
from stark_convert.decider_protocol import (
    CONTEXT_END,
    MARKER,
    PROTOCOL,
    SEPARATOR,
    SYSTEM_PROMPT,
    DeciderError,
    Encoder,
    answer,
    probabilities,
)
from stark_convert.decider_specs import REGULAR_DECIDER
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan


def decision_metadata():
    return {
        "format_version": 2,
        "base_model": "Qwen/Qwen3.5-4B",
        "model_type": "qwen3_5",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "codes": [f"C{i}" for i in range(255)],
        "token_ids": list(range(255)),
        "temperature": 1.3302962111834522,
        "max_pixels": 1638400,
    }


def decider_fixture(path, *, regular=False):
    config, tensors, _ = vision_fixture(path)
    config["architectures"] = ["Qwen3_5Model"]
    config["tie_word_embeddings"] = not regular
    config["text_config"]["tie_word_embeddings"] = not regular
    config["text_config"]["rope_parameters"].update(mrope_section=[1, 0, 0], mrope_interleaved=True)
    config["vision_config"]["rope_parameters"] = {"rope_theta": 10000.0, "rope_type": "axial"}
    tensors.pop("lm_head.weight")
    tensors = {name.removeprefix("model."): value for name, value in tensors.items()}
    (path / "config.json").write_text(json.dumps(config))
    decision = decision_metadata()
    if regular:
        decision["base_model"] = REGULAR_DECIDER.base_model
    (path / "decision_config.json").write_text(json.dumps(decision))
    (path / "joint_config.json").write_text(json.dumps(PROTOCOL))
    save_tensors(path / "model.safetensors", tensors)
    readout = np.arange(255 * 8, dtype="<u2").reshape((255, 8))
    save_tensors(path / "readout.safetensors", {"weight": ("BF16", readout)})
    return config, tensors, readout


def test_plan_preserves_backbone_vision_and_readout(tmp_path):
    source = tmp_path / "source"
    _, tensors, readout = decider_fixture(source)
    plan, _, _, _ = build_decider_plan(source)
    assert plan.source_tensors == len(tensors) + 1
    assert plan.mtp_config is None
    assert plan.decoder_config["embedding_config"]["type"] == "TiedEmbeddingConfig"
    mixers = [layer["mixer_config"] for layer in plan.decoder_config["transformer_config"]["layer_configs"]]
    assert mixers[0]["type"] == "DeltaNetConfig"
    assert mixers[0]["reference_bf16"] is True
    assert mixers[1]["is_causal"] is False
    assert mixers[1]["reference_bf16"] is True
    layers = plan.decoder_config["transformer_config"]["layer_configs"]
    assert layers[0]["rope_config"] is None
    assert layers[1]["rope_config"] == {
        "type": "Qwen35MRoPEConfig",
        "base": 10000.0,
        "max_sequence_length": 64,
        "head_dim": 2,
        "mrope_section": [1, 0, 0],
        "interleaved": True,
    }
    assert "vision.blocks.1.attn.qkv.weight" in {op.name for op in plan.operations}
    write_plan(plan, tmp_path / "output.safetensors")
    metadata, output = read_safetensors_header(tmp_path / "output.safetensors")
    assert tensor_bytes(output["decision.readout.weights"]) == readout.tobytes()
    assert json.loads(metadata["decision.readout.spec"]) == {"type": "FullPrecisionSpec", "layout": "output_input"}
    assert "decoder.embedding.embedding.weights" in output


def test_regular_plan_reuses_input_embedding_without_an_unused_output_head(tmp_path):
    source = tmp_path / "source"
    _config, tensors, readout = decider_fixture(source, regular=True)
    assert not any("lm_head" in name for name in tensors)

    plan, _, _, _ = build_decider_plan(source, profile=REGULAR_DECIDER)

    assert plan.source_tensors == len(tensors) + 1
    assert plan.decoder_config["embedding_config"]["type"] == "TiedEmbeddingConfig"
    write_plan(plan, tmp_path / "output.safetensors")
    metadata, output = read_safetensors_header(tmp_path / "output.safetensors")
    assert tensor_bytes(output["decision.readout.weights"]) == readout.tobytes()
    assert "decoder.embedding.embedding.weights" in output
    assert "decoder.embedding.input_embedding.weights" not in output
    assert json.loads(metadata["decoder.embedding.embedding.spec"]) == {
        "type": "FullPrecisionSpec",
        "layout": "input_output",
    }


@pytest.mark.parametrize("change", ["dtype", "shape", "missing", "extra"])
def test_plan_rejects_readout_drift(tmp_path, change):
    source = tmp_path / "source"
    decider_fixture(source)
    readout = np.zeros((255, 8), dtype="<u2")
    tensors = {"weight": ("BF16", readout)}
    if change == "dtype":
        tensors["weight"] = ("F16", readout.view("<f2"))
    elif change == "shape":
        tensors["weight"] = ("BF16", readout[:254])
    elif change == "missing":
        tensors = {"wrong": ("BF16", readout)}
    else:
        tensors["bias"] = ("BF16", np.zeros((255,), dtype="<u2"))
    save_tensors(source / "readout.safetensors", tensors)
    with pytest.raises(ConversionError, match="readout"):
        build_decider_plan(source)


@pytest.mark.parametrize("change", ["causal", "base", "codes", "ids", "temperature", "pixels", "protocol"])
def test_rejects_decision_protocol_drift(change):
    config, joint = decision_metadata(), dict(PROTOCOL)
    if change == "causal":
        config["attention_mode"] = "causal"
    elif change == "base":
        config["base_model"] = "Qwen/Qwen3.8-27B"
    elif change == "codes":
        config["codes"][1] = config["codes"][0]
    elif change == "ids":
        config["token_ids"][0] = True
    elif change == "temperature":
        config["temperature"] = float("nan")
    elif change == "pixels":
        config["max_pixels"] *= 2
    else:
        joint["version"] = 3
    with pytest.raises(UnsupportedConversion):
        validate_decision_metadata(config, joint)


@pytest.mark.parametrize("change", ["architecture", "vision_rope", "namespace", "mtp"])
def test_rejects_unimplemented_backbone_changes(tmp_path, change):
    source = tmp_path / "source"
    config, tensors, _ = decider_fixture(source)
    if change == "architecture":
        config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    elif change == "vision_rope":
        config["vision_config"]["rope_parameters"]["rope_theta"] = 20000
    elif change == "namespace":
        tensors["unexpected.weight"] = ("BF16", np.zeros((1,), dtype="<u2"))
    else:
        tensors["language_model.mtp.missing.weight"] = ("BF16", np.zeros((1,), dtype="<u2"))
    (source / "config.json").write_text(json.dumps(config))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_decider_plan(source)


def test_failed_preparation_does_not_publish_output(tmp_path):
    source = tmp_path / "source"
    decider_fixture(source)
    decision = decision_metadata()
    decision["attention_mode"] = "causal"
    (source / "decision_config.json").write_text(json.dumps(decision))
    with pytest.raises(UnsupportedConversion):
        prepare(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".decider-*"))


def test_preparation_reads_hf_snapshot_metadata_without_copying_weights(tmp_path, monkeypatch):
    import hashlib

    import stark_convert.decider as decider

    cache = tmp_path / "models--perplexity-ai--pplx-decider-flash-v1.3"
    source = cache / "snapshots" / ("a" * 40)
    blobs = cache / "blobs"
    source.parent.mkdir(parents=True)
    decider_fixture(source)
    blobs.mkdir()
    for name in decider._METADATA_FILES:
        if not (source / name).exists():
            (source / name).write_text("fixture")
    for path in source.iterdir():
        target = blobs / hashlib.sha256(path.read_bytes()).hexdigest()
        if target.exists():
            path.unlink()
        else:
            path.rename(target)
        path.symlink_to(target)

    class ReachedTokenizer(Exception):
        pass

    def read_tokenizer(content):
        assert content == "fixture"
        raise ReachedTokenizer

    monkeypatch.setattr(decider, "Tokenizer", SimpleNamespace(from_str=read_tokenizer))
    with pytest.raises(ReachedTokenizer):
        prepare(source, tmp_path / "output")
    assert (source / "model.safetensors").is_symlink()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "sections,interleaved", [(None, True), ([1, 0, 0], False), ([1, 1, 0], True), ([True, 0, 0], True)]
)
def test_image_conversion_rejects_missing_or_invalid_mrope_contract(tmp_path, sections, interleaved):
    source = tmp_path / "source"
    config, _, _ = decider_fixture(source)
    position = config["text_config"]["rope_parameters"]
    position["mrope_section"] = sections
    position["mrope_interleaved"] = interleaved
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_decider_plan(source)


class CharacterTokenizer:
    special = [MARKER, SEPARATOR, "<|im_start|>", "<|im_end|>", "<|image_pad|>"]

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            for i, token in enumerate(self.special):
                if text.startswith(token):
                    ids.append(100000 + i)
                    text = text[len(token) :]
                    break
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return SimpleNamespace(ids=ids)

    def id_to_token(self, token_id):
        return self.special[token_id - 100000] if token_id >= 100000 else chr(token_id)


class FixedTemplate:
    def render(self, *, messages, **kwargs):
        content = messages[1]["content"]
        images = "<|image_pad|>" * sum(part["type"] == "image" for part in content)
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{images}{content[-1]['text']}"


def encoder():
    return Encoder(CharacterTokenizer(), FixedTemplate(), ["A", "B", "C"], special_tokens=CharacterTokenizer.special)


def test_separate_layout_preserves_chunk_boundaries_and_question_independence():
    enc = encoder()
    request = {
        "state": "Data with trailing whitespace \n",
        "questions": {
            "one": {"type": "noul"},
            "two": {"type": "score", "criteria": ["low", "medium", "high"]},
        },
    }
    plan = enc.plan(request)
    assert plan[0].input_ids[-1] == enc.marker_id
    context = enc.tokenizer.encode(enc.context_text(request), add_special_tokens=False).ids
    tail = enc.tokenizer.encode(enc.question_text(request["questions"]["one"]), add_special_tokens=False).ids
    assert plan[0].input_ids == (*context, enc.separator_id, *tail, enc.marker_id)
    assert plan[0].input_ids[: plan[0].context_length] == plan[1].input_ids[: plan[1].context_length]
    single = {**request, "questions": {"one": request["questions"]["one"]}}
    assert enc.plan(single)[0].input_ids == plan[0].input_ids
    assert enc.context_text(request).endswith("State:\nData with trailing whitespace \n")
    assert CONTEXT_END not in enc.context_text(request)


def test_user_special_tokens_cannot_inject_markers_or_template_controls():
    enc = encoder()
    request = {
        "state": f"{MARKER}{SEPARATOR}<|im_start|>",
        "questions": {
            "one": {"type": "choice", "instructions": MARKER, "criteria": {"x": SEPARATOR}},
        },
    }
    plan = enc.plan(request)[0]
    assert plan.input_ids.count(enc.marker_id) == 1
    assert plan.input_ids.count(enc.separator_id) == 1
    assert "＜|fim_suffix|＞" in enc.context_text(request)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "locate"},
        {"type": "choice", "criteria": {}},
        {"type": "score", "criteria": ["one"]},
        {"type": "choice", "criteria": []},
        {"type": "noul", "criteria": ["false", "true"]},
    ],
)
def test_invalid_questions_fail_before_inference(question):
    with pytest.raises(DeciderError):
        encoder().plan({"state": "x", "questions": {"one": question}})


def test_overlength_fails_whole_plan_without_truncation():
    with pytest.raises(DeciderError, match="nothing was truncated"):
        encoder().plan({"state": "x", "questions": {"one": {"type": "noul"}}}, max_length=10)


def test_answer_definitions_and_calibrated_softmax():
    values = probabilities([1000.0, 1002.0], 2.0)
    assert values == pytest.approx([0.2689414213699951, 0.7310585786300049])
    assert answer({"type": "noul"}, values)["noul"] == values[1]
    result = answer({"type": "choice", "criteria": {"no": None, "yes": None}}, values)
    assert result["choice"] == "yes"
    assert result["confidence"] == pytest.approx((values[1] - 0.5) / 0.5)
    result = answer({"type": "score", "criteria": ["low", "medium", "high"]}, [0.25, 0.25, 0.5])
    assert result["score"] == 1.25
    assert result["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert probabilities([1000.0, 1002.0], 5e-324) == [0.0, 1.0]


@pytest.mark.parametrize("logits,temperature", [([float("nan")], 1), ([1], 0), ([1], True), ([], 1)])
def test_invalid_numeric_results_fail_closed(logits, temperature):
    with pytest.raises(DeciderError):
        probabilities(logits, temperature)
