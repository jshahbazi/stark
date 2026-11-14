from pathlib import Path

import pytest

from stark_convert.chat import _ChatTemplate
from stark_convert.reasoning import ReasoningStream, assistant_message, prompt_opens_reasoning, reasoning_options


@pytest.mark.parametrize("suffix", ["<think>", "<think>\n", "<think>\n\n"])
def test_trusted_prompt_accepts_template_whitespace_after_thinking_opener(suffix):
    assert prompt_opens_reasoning("assistant\n" + suffix)


@pytest.mark.parametrize(
    ("text", "initial", "expected"),
    [
        ("Reasoning</think>Answer", True, {"content": "Answer", "reasoning_content": "Reasoning"}),
        ("<think>Reasoning</think>Answer", False, {"content": "Answer", "reasoning_content": "Reasoning"}),
        ("Reasoning</thi", True, {"content": "", "reasoning_content": "Reasoning</thi"}),
        ("<think>unfinished", False, {"content": "", "reasoning_content": "unfinished"}),
        ("<thi", False, {"content": "<thi"}),
        ("literal </think> remains visible", False, {"content": "literal </think> remains visible"}),
        ("Answer <think>literal</think>", False, {"content": "Answer <think>literal</think>"}),
        ("<think>nested</think>Answer", True, {"content": "Answer", "reasoning_content": "<think>nested"}),
    ],
)
def test_buffered_and_streamed_reasoning_match_at_every_chunk_boundary(text, initial, expected):
    assert assistant_message(text, initial_thinking=initial) == {"role": "assistant", **expected}
    for split in range(len(text) + 1):
        emitted = {"content": [], "reasoning_content": []}
        stream = ReasoningStream(
            emitted["content"].append, emitted["reasoning_content"].append, initial_thinking=initial
        )
        stream.feed(text[:split])
        for character in text[split:]:
            stream.feed(character)
        stream.finish(text)
        assert {key: "".join(value) for key, value in emitted.items()} == {
            "content": expected.get("content", ""),
            "reasoning_content": expected.get("reasoning_content", ""),
        }


def test_stream_rejects_terminal_text_that_disagrees_with_delivered_text():
    stream = ReasoningStream(lambda _: None, lambda _: None, initial_thinking=True)
    stream.feed("Reasoning</think>Answer")
    with pytest.raises(ValueError, match="Invalid worker stream"):
        stream.finish("Different</think>Answer")


@pytest.mark.parametrize(("thinking", "effort"), [(False, "low"), (True, "max")])
def test_glm_boolean_maps_to_reviewed_effort_program(thinking, effort):
    source = (Path(__file__).parent / "fixtures" / "glm53_chat_template.jinja").read_text()
    options = reasoning_options({"prompt_template": source}, thinking=thinking)
    assert options == {
        "enable_thinking": thinking,
        "reasoning_effort": effort,
    }
    prompt = _ChatTemplate(source).render(
        messages=[{"role": "user", "content": "Synthetic question"}],
        tools=None,
        add_generation_prompt=True,
        **options,
    )
    assert "<|system|>Reasoning Effort: " + effort.capitalize() in prompt


def test_effort_is_rejected_for_other_templates_even_with_same_tool_protocol():
    source = (Path(__file__).parent / "fixtures" / "flint_chat_template.jinja").read_text()
    with pytest.raises(ValueError):
        reasoning_options({"prompt_template": source}, effort="high")


@pytest.mark.parametrize("thinking", [False, True])
def test_granite_boolean_uses_declared_field_in_native_template(thinking):
    source = (Path(__file__).parent / "fixtures" / "tool_templates" / "granite33.jinja").read_text()
    codec = {
        "prompt_template": source,
        "reasoning_config": {
            "default_reasoning_effort": "no_reasoning",
            "field_name": "thinking",
            "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
        },
    }
    options = reasoning_options(codec, thinking=thinking)
    assert options == {"thinking": thinking}
    prompt = _ChatTemplate(source).render(
        messages=[{"role": "user", "content": "Synthetic question"}],
        tools=None,
        add_generation_prompt=True,
        **options,
    )
    assert ("Write your thoughts between <think></think>" in prompt) is thinking


@pytest.mark.parametrize("thinking", [False, True])
def test_qwen_boolean_default_reaches_native_template(thinking):
    source = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    options = reasoning_options({"prompt_template": source}, thinking=thinking)
    assert options == {"enable_thinking": thinking}
    prompt = _ChatTemplate(source).render(
        messages=[{"role": "user", "content": "Synthetic question"}],
        tools=None,
        add_generation_prompt=True,
        **options,
    )
    assert prompt.endswith("<think>\n\n</think>\n\n") is (not thinking)


@pytest.mark.parametrize("reasoning", [None, "thinking", {}, {"field_name": "unsupported"}, {"field_name": []}])
def test_boolean_field_falls_back_for_unrecognized_metadata(reasoning):
    assert reasoning_options({"reasoning_config": reasoning}, thinking=True) == {"enable_thinking": True}
