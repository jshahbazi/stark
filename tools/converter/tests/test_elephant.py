from types import SimpleNamespace

import pytest

from stark_convert import elephant, elephant_install, elephant_protocol, model_catalog, model_support
from stark_convert.desktop_decider import prepared_decider


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(ids=tuple(map(ord, text)))


class Template:
    def __init__(self):
        self.calls = []

    def render(self, **arguments):
        self.calls.append(arguments)
        messages = arguments["messages"]
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join("<image>" if item.get("type") == "image" else item["text"] for item in content)
            parts.append(f"<{message['role']}>\n{content}\n")
        if arguments.get("add_generation_prompt"):
            parts.append("<assistant>\n<think>\n\n</think>\n\n")
        return "".join(parts)


def request():
    return {
        "state": {"ticket": "double charged"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "refunds", "support": "technical"},
            },
            "refund": {"type": "noul"},
        },
    }


def test_onejev_plan_preserves_exact_shared_prefix_and_readout_boundary():
    template = Template()
    encoder = elephant_protocol.Encoder(CharacterTokenizer(), template)
    plan = encoder.plan({**request(), "images": ["private image"]})
    assert len(plan) == 2
    assert [item.option_count for item in plan] == [2, 2]
    assert all(0 < item.block_boundary < len(item.input_ids) for item in plan)
    assert all(item.context_length == item.block_boundary for item in plan)
    prefixes = [item.input_ids[: item.block_boundary] for item in plan]
    assert prefixes[0] == prefixes[1]
    rendered = template.calls[-1]["messages"][1]["content"]
    assert rendered[0] == {"type": "image"}


def test_onejev_answers_keep_yes_first_and_choice_order():
    assert elephant_protocol.answer({"type": "noul"}, [0.75, 0.25])["noul"] == 0.75
    choice = {
        "type": "choice",
        "instructions": "route",
        "criteria": {"billing": None, "support": None},
    }
    result = elephant_protocol.answer(choice, [0.2, 0.8])
    assert result["choice"] == "support"
    assert list(result["probabilities"]) == ["billing", "support"]


def test_onejev_rejects_untrained_option_shapes():
    encoder = elephant_protocol.Encoder(CharacterTokenizer(), Template())
    too_many = request()
    too_many["questions"]["route"]["criteria"] = {str(index): None for index in range(256)}
    with pytest.raises(elephant_protocol.DeciderError):
        encoder.plan(too_many)
    one_level = request()
    one_level["questions"]["route"] = {"type": "score", "criteria": ["one"]}
    with pytest.raises(elephant_protocol.DeciderError):
        encoder.plan(one_level)


def test_elephant_is_a_separate_pinned_decision_family():
    matrix = model_support.support_matrix()
    family = next(item for item in matrix["families"] if item["family_id"] == "elephant-1")
    assert family["model_kind"] == "decision"
    assert family["modes"]["decision"]["status"] == "conditional"
    report = model_catalog.family_catalog("elephant-1")
    assert report["status"] == "ready"
    assert report["items"][0]["repo"] == elephant.SOURCE_REPO
    assert report["items"][0]["revision"] == elephant.SOURCE_REVISION
    assert report["items"][0]["artifact_modes"] == ["decision"]
    assert model_catalog.search_catalog("Elephant 1")["items"][0]["family_id"] == "elephant-1"


def test_elephant_rejects_changed_modeling_contract():
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "image_token_id": 1,
        "vision_start_token_id": 2,
        "vision_end_token_id": 3,
        "text_config": {
            "model_type": "qwen3_5_text",
            "max_position_embeddings": 262144,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
            },
        },
        "vision_config": {"rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"}},
        "modeling_overrides": {
            "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
            "readout": "slot_lm",
            "head_dim": 256,
            "n_slots": 256,
        },
    }
    processor = {
        "processor_class": "Qwen3VLProcessor",
        "image_processor": elephant._EXPECTED_PROCESSOR,
    }
    readout = {"readout": "slot_lm", "head_dim": 256, "n_slots": 256}
    elephant.validate_source_metadata(config, readout, processor)
    changed = {**config, "modeling_overrides": {**config["modeling_overrides"], "readout": "head"}}
    with pytest.raises(elephant.UnsupportedConversion):
        elephant.validate_source_metadata(changed, readout, processor)


def test_elephant_source_detection_is_repo_and_revision_bounded():
    assert elephant_install.is_elephant_source(
        elephant.SOURCE_REPO, kind="huggingface", revision=elephant.SOURCE_REVISION
    )
    assert not elephant_install.is_elephant_source(elephant.SOURCE_REPO, kind="huggingface", revision="0" * 40)


def test_prepared_elephant_admission_keeps_block_attention_and_image_limits(tmp_path):
    for name in ("model.safetensors", "tokenizer.json", "chat_template.jinja", "config.json"):
        (tmp_path / name).write_bytes(b"fixture")
    config = {
        "type": "DeciderModelConfig",
        "decoder_config": {
            "transformer_config": {
                "layer_configs": [
                    {
                        "mixer_config": {"type": "AttentionConfig", "is_causal": True},
                        "rope_config": {
                            "type": "Qwen35MRoPEConfig",
                            "mrope_section": [11, 11, 10],
                            "interleaved": True,
                        },
                    }
                ]
            }
        },
        "vision_config": {},
        "decision_config": {
            "protocol": "onejev-slot-lm-v2",
            "protocol_version": 5,
            "num_options": 256,
            "readout": "slot_lm",
            "attention": "block_bidirectional",
            "temperature": 1.0,
            "max_pixels": elephant.MAX_PIXELS,
            "max_input_tokens": elephant.MAX_INPUT_TOKENS,
            "slot_labels": list(elephant_protocol.SLOT_LABELS),
            "slot_token_ids": list(range(256)),
        },
    }
    assert prepared_decider(tmp_path, config)
