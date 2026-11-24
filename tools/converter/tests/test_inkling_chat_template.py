"""Compare native rendering with the pinned Inkling 2 Mini prompt protocol."""

import hashlib
from pathlib import Path

from stark_convert.chat import _ChatTemplate
from stark_convert.inkling_prepare import TEMPLATE_SHA
from stark_convert.tool_protocols import runtime_template_source

SOURCE = Path(__file__).parent / "fixtures/tool_templates/inkling2_mini.jinja"
END = "<|end_message|>"
STOP = "<|content_model_end_sampling|>"
SYSTEM = "<|message_system|><|content_text|>"
EFFORT = SYSTEM + "Thinking effort level: "
USER = "<|message_user|><|content_text|>"
ASSISTANT = "<|message_model|>"


def renderer():
    return _ChatTemplate(SOURCE.read_text(encoding="utf-8"))


def test_only_the_exact_publisher_source_is_adapted():
    source = SOURCE.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == TEMPLATE_SHA
    adapted = runtime_template_source(source)
    assert "generation -%}" not in adapted
    assert "key | tojson(sort_keys=false)" in adapted
    assert "tool_state.specs | list | tojson" in adapted
    changed = source + "\n"
    assert runtime_template_source(changed) == changed


def test_official_template_text_and_assistant_tracks_match_reference():
    actual = renderer().render(
        messages=[
            {"role": "system", "content": "Guide"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Answer", "reasoning_content": "Reason"},
        ],
        tools=None,
        add_generation_prompt=True,
    )
    assert actual == (
        f"{SYSTEM}Guide{END}{EFFORT}0.9{END}{USER}Hello{END}"
        f"{ASSISTANT}<|content_thinking|>Reason{END}{ASSISTANT}<|content_text|>Answer{END}{STOP}"
    )


def test_official_template_image_uses_its_own_message_frame():
    actual = renderer().render(
        messages=[{"role": "user", "content": ["Before", {"type": "input_image"}, "After"]}],
        tools=None,
        add_generation_prompt=True,
    )
    assert actual == (
        f"{EFFORT}0.9{END}{USER}Before{END}<|message_user|><|content_image|><|unused_200054|>{END}{USER}After{END}"
    )


def test_official_template_tool_json_matches_reference():
    actual = renderer().render(
        messages=[
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "lookup", "arguments": {"z": "café", "A<": {"z": 1.0, "a<": None}}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Found"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Find café <&>",
                    "parameters": {"type": "object", "properties": {"z": {"type": "string"}, "A": {"type": "number"}}},
                },
            }
        ],
        reasoning_effort="none",
        add_generation_prompt=True,
    )
    assert actual == (
        '<|message_system|>tool_declare<|content_xml|>[{"description":"Find café <&>","name":"lookup",'
        '"parameters":{"properties":{"A":{"type":"number"},"z":{"type":"string"}},"type":"object"},'
        '"type":"function"}]'
        f"{END}{EFFORT}0{END}{USER}Hello{END}{ASSISTANT}lookup<|content_invoke_tool_json|>"
        '{"name":"lookup","args":{"A<":{"z":1.0,"a<":null},"z":"café"}}'
        f"{END}{STOP}<|message_tool|>lookup<|content_text|>Found{END}"
    )
