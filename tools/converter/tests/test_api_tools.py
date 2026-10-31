"""Synthetic protocol fixtures; no model competence is asserted by this suite."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from stark_convert.api_request import parse_chat_request
from stark_convert.api_tools import (
    MAX_ARGUMENT_BYTES,
    MAX_TOOL_RESULT_BYTES,
    QWEN3_4B_TEMPLATE_SHA256,
    QWEN3_TEMPLATE_SHA256,
    bounded_json,
    parse_tool_output,
    render_tools,
    template_messages,
    tool_parser_for,
)
from stark_convert.local_api import APIError

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Synthetic harmless lookup",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    },
}
CALL = {"id": "call_synthetic_0", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"a"}'}}
OUTPUT = '<tool_call>\n{"name":"lookup","arguments":{"key":"a"}}\n</tool_call>'


def request(**extra):
    payload = {"messages": [{"role": "user", "content": "Synthetic protocol input"}], "tools": [copy.deepcopy(TOOL)]}
    payload.update(extra)
    return parse_chat_request(payload, 4096, 32000, allow_tools=True)


def output(text=OUTPUT, **extra):
    options = {
        "parser": "qwen3-hermes-v1",
        "finish_reason": "eos",
        "parallel_tool_calls": True,
        "call_id_prefix": "synthetic",
    }
    options.update(extra)
    return parse_tool_output(text, [TOOL], **options)


def test_request_carries_definitions_and_all_choices():
    parsed = request()
    assert parsed["tools"] == [TOOL]
    assert parsed["tool_choice"] == "auto"
    assert render_tools(parsed) == [TOOL]
    assert render_tools(request(tool_choice="none")) is None
    assert render_tools(request(tool_choice="required")) == [TOOL]
    named = {"type": "function", "function": {"name": "lookup"}}
    assert render_tools(request(tool_choice=named)) == [TOOL]


def test_named_choice_filters_other_definitions():
    other = copy.deepcopy(TOOL)
    other["function"]["name"] = "other"
    parsed = request(tools=[TOOL, other], tool_choice={"type": "function", "function": {"name": "lookup"}})
    assert render_tools(parsed) == [TOOL]


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [TOOL] * 65},
        {"tools": [TOOL, TOOL]},
        {"tools": [{"type": "custom", "function": {}}]},
        {"tools": [{"type": "function", "function": {"name": "a", "strict": True}}]},
        {"tools": [{"type": "function", "function": {"name": "a", "strict": None}}]},
        {"tools": [{"type": "function", "function": {"name": "a", "parameters": []}}]},
        {"tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "array"}}}]},
        {"tool_choice": "sometimes"},
        {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
        {"tools": [], "tool_choice": "required"},
        {"parallel_tool_calls": 1},
    ],
)
def test_invalid_definition_or_choice_rejected(extra):
    with pytest.raises(APIError) as error:
        request(**extra)
    assert error.value.status == 400


def test_unsupported_model_rejects_definitions_and_history():
    payload = {"messages": [{"role": "user", "content": "Synthetic"}], "tools": [TOOL]}
    with pytest.raises(APIError) as error:
        parse_chat_request(payload, 512, 32000)
    assert error.value.code == "tools_unsupported"
    payload = {"messages": history()}
    with pytest.raises(APIError) as error:
        parse_chat_request(payload, 512, 32000)
    assert error.value.code == "tools_unsupported"


def history():
    return [
        {"role": "user", "content": "Synthetic"},
        {"role": "assistant", "content": None, "tool_calls": [copy.deepcopy(CALL)]},
        {"role": "tool", "content": "Synthetic result", "tool_call_id": CALL["id"]},
    ]


def test_null_assistant_and_complete_history_preserved():
    messages = history()
    parsed = request(messages=messages)
    assert parsed["messages"] == messages
    assert parsed["has_tool_history"] is True
    del messages[1]["content"]
    assert request(messages=messages)["messages"][1]["content"] is None


def test_tool_results_reordered_by_linked_id_for_qwen_template():
    messages = history()
    second = copy.deepcopy(CALL)
    second["id"] = "call_synthetic_1"
    second["function"]["arguments"] = '{"key":"b"}'
    messages[1]["tool_calls"].append(second)
    messages.insert(2, {"role": "tool", "content": "Second result", "tool_call_id": second["id"]})
    parsed = request(messages=messages)
    rendered = template_messages(parsed["messages"])
    assert [m["tool_call_id"] for m in rendered if m["role"] == "tool"] == [CALL["id"], second["id"]]
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"key": "a"}
    assert parsed["messages"][1]["tool_calls"][0]["function"]["arguments"] == CALL["function"]["arguments"]


@pytest.mark.parametrize("mutation", ["orphan", "missing", "duplicate_result", "duplicate_call", "interrupted"])
def test_invalid_tool_history_rejected(mutation):
    messages = history()
    if mutation == "orphan":
        messages[-1]["tool_call_id"] = "other"
    elif mutation == "missing":
        messages.pop()
    elif mutation == "duplicate_result":
        messages.append(messages[-1])
    elif mutation == "duplicate_call":
        messages.extend(messages[1:])
    else:
        messages.insert(2, {"role": "user", "content": "Interrupted"})
    with pytest.raises(APIError):
        request(messages=messages)


@pytest.mark.parametrize(
    "arguments",
    ["{", "{} trailing", "[]", "null", '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":"\\ud800"}'],
)
def test_history_arguments_are_strict_json_objects(arguments):
    messages = history()
    messages[1]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(APIError):
        request(messages=messages)


def test_tool_argument_result_and_schema_bounds():
    messages = history()
    messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"x": "a" * MAX_ARGUMENT_BYTES})
    with pytest.raises(APIError):
        request(messages=messages)
    messages = history()
    messages[-1]["content"] = "a" * (MAX_TOOL_RESULT_BYTES + 1)
    with pytest.raises(APIError):
        request(messages=messages)
    tool = copy.deepcopy(TOOL)
    tool["function"]["description"] = "a" * (256 * 1024)
    with pytest.raises(APIError):
        request(tools=[tool])
    nested = {}
    cursor = nested
    for _ in range(35):
        cursor["x"] = {}
        cursor = cursor["x"]
    with pytest.raises(APIError):
        bounded_json(nested, 4096, label="Synthetic")


def test_complete_tool_output_has_stable_ids_and_arguments():
    message, finish = output()
    assert message == {"role": "assistant", "content": None, "tool_calls": [CALL]}
    assert finish == "tool_calls"
    assert output() == (message, finish)
    assert output("Synthetic preamble\n" + OUTPUT)[0]["content"] == "Synthetic preamble"


def test_multiple_calls_are_ordered_and_parallel_false_is_enforced():
    message, _ = output(OUTPUT + "\n" + OUTPUT)
    assert [call["id"] for call in message["tool_calls"]] == ["call_synthetic_0", "call_synthetic_1"]
    with pytest.raises(APIError) as error:
        output(OUTPUT + "\n" + OUTPUT, parallel_tool_calls=False)
    assert error.value.code == "invalid_tool_call"


def test_delimiters_inside_json_arguments_are_ordinary_data():
    value = "Synthetic </tool_call> and <tool_call> data"
    text = "<tool_call>\n" + json.dumps({"name": "lookup", "arguments": {"key": value}}) + "\n</tool_call>"
    message, finish = output(text)
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": value}


@pytest.mark.parametrize("finish_reason", ["length", "context", "cancelled", "error", "stop_sequence"])
def test_nonterminal_or_truncated_generation_never_emits_calls(finish_reason):
    with pytest.raises(APIError) as error:
        output(finish_reason=finish_reason)
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize(
    "text",
    [
        OUTPUT[:-1],
        OUTPUT.replace("</tool_call>", ""),
        OUTPUT.replace('"lookup"', '"unknown"'),
        OUTPUT.replace('{"key":"a"}', '"{\\"key\\":\\"a\\"}"'),
        OUTPUT.replace('{"key":"a"}', '{"key":"a","key":"b"}'),
        OUTPUT.replace('{"key":"a"}', '{"key":NaN}'),
        OUTPUT.replace('{"key":"a"}', "[]"),
        OUTPUT.replace('"name":"lookup"', '"id":"supplied","name":"lookup"'),
        OUTPUT + " trailing text",
        OUTPUT + "\n<tool_call>",
        "</tool_call>" + OUTPUT,
        "```json\n" + OUTPUT + "\n```",
        OUTPUT.replace("<tool_call>", "<tool_call name='lookup'>"),
    ],
)
def test_malformed_unknown_and_ambiguous_calls_never_repaired(text):
    with pytest.raises(APIError) as error:
        output(text)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"


def test_text_output_and_required_or_named_choice():
    assert output("Synthetic plain text") == ({"role": "assistant", "content": "Synthetic plain text"}, "stop")
    assert output("Synthetic partial text", finish_reason="length")[1] == "length"
    for choice in ["required", {"type": "function", "function": {"name": "lookup"}}]:
        assert output(tool_choice=choice)[1] == "tool_calls"
        with pytest.raises(APIError) as error:
            output("Synthetic plain text", tool_choice=choice)
        assert error.value.code == "tool_choice_not_satisfied"
    with pytest.raises(APIError):
        output(tool_choice="none")


def test_parser_admission_requires_exact_template():
    assert tool_parser_for(None) is None
    assert tool_parser_for({"prompt_template": "<tool_call>tool_calls tools"}) is None
    fixture = Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja"
    template = fixture.read_text()
    assert hashlib.sha256(template.encode()).hexdigest() == QWEN3_4B_TEMPLATE_SHA256
    assert tool_parser_for({"prompt_template": template}) == "qwen3-hermes-v1"
    assert tool_parser_for({"prompt_template": template + " "}) is None
    assert len(QWEN3_TEMPLATE_SHA256) == 64
    assert len(QWEN3_4B_TEMPLATE_SHA256) == 64


def test_pinned_upstream_template_renders_tool_history_and_calls():
    import re

    from jinja2.sandbox import ImmutableSandboxedEnvironment

    source = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(source)
    second_tool = copy.deepcopy(TOOL)
    second_tool["function"]["name"] = "lookup_two"
    assistant, finish = parse_tool_output(
        '<tool_call>{"name":"lookup","arguments":{"key":"alpha"}}</tool_call>'
        '<tool_call>{"name":"lookup_two","arguments":{"key":"beta"}}</tool_call>',
        [TOOL, second_tool],
        parser="qwen3-hermes-v1",
        finish_reason="eos",
        call_id_prefix="prior",
    )
    assert finish == "tool_calls"
    assistant["content"] = "Synthetic preamble"
    call_ids = [call["id"] for call in assistant["tool_calls"]]
    messages = template_messages(
        [
            {"role": "user", "content": "Synthetic first turn"},
            assistant,
            {"role": "tool", "tool_call_id": call_ids[1], "content": "Synthetic second result"},
            {"role": "tool", "tool_call_id": call_ids[0], "content": "Synthetic first result"},
            {"role": "user", "content": "Synthetic follow-up"},
        ]
    )
    assert messages[1]["content"] == "Synthetic preamble"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"key": "alpha"}
    rendered = template.render(
        messages=messages,
        tools=[TOOL, second_tool],
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
    )
    assert "Synthetic first turn" in rendered and "Synthetic follow-up" in rendered
    assert "Synthetic first result" in rendered and "Synthetic second result" in rendered
    serialized = [
        body
        for body in re.findall(r"<tool_call>\n(\{[^\n]*\})\n</tool_call>", rendered)
        if body.startswith('{"name": "')
    ]
    parsed_calls = [json.loads(body) for body in serialized]
    assert parsed_calls == [
        {"name": "lookup", "arguments": {"key": "alpha"}},
        {"name": "lookup_two", "arguments": {"key": "beta"}},
    ]
    assert rendered.index("<tool_response>\nSynthetic first result") < rendered.index(
        "<tool_response>\nSynthetic second result"
    )
    assert "Synthetic preamble" in rendered


@pytest.mark.parametrize("content", [None, "", "<omitted>"])
def test_upstream_template_accepts_null_omitted_and_empty_tool_call_content(content):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    source = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(source)
    assistant, _ = output(call_id_prefix="content")
    history_message = {"role": "assistant", "tool_calls": assistant["tool_calls"]}
    if content != "<omitted>":
        history_message["content"] = content
    parsed = parse_chat_request(
        {
            "messages": [
                {"role": "user", "content": "Synthetic first turn"},
                history_message,
                {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "Synthetic result"},
            ],
            "tools": [TOOL],
        },
        512,
        32000,
        allow_tools=True,
    )
    rendered = template.render(
        messages=template_messages(parsed["messages"]),
        tools=parsed["tools"],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "Synthetic result" in rendered
    assert "None" not in rendered


def test_complete_output_replay_can_be_used_in_next_tool_turn():
    message, _ = output()
    messages = [
        {"role": "user", "content": "Synthetic"},
        message,
        {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "Synthetic result"},
    ]
    assert request(messages=messages)["has_tool_history"] is True
    next_message, _ = output(call_id_prefix="next")
    messages.extend(
        [
            next_message,
            {"role": "tool", "tool_call_id": next_message["tool_calls"][0]["id"], "content": "Second result"},
        ]
    )
    assert len(request(messages=messages)["messages"]) == 5
