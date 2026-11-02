"""Synthetic protocol fixtures; no model competence is asserted by this suite."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from stark_convert.api_request import parse_chat_request
from stark_convert.api_tools import (
    MAX_ARGUMENT_BYTES,
    MAX_TOOL_OUTPUT_BYTES,
    MAX_TOOL_RESULT_BYTES,
    QWEN3_4B_TEMPLATE_SHA256,
    QWEN3_TEMPLATE_SHA256,
    QWEN35_XML_TEMPLATE_SHA256,
    QWEN38_XML_TEMPLATE_SHA256,
    ToolTextStream,
    bounded_json,
    parse_tool_output,
    render_tools,
    template_messages,
    tool_parser_for,
    validate_parser_tools,
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


@pytest.mark.parametrize("text", ["Synthetic text.\n", "  Unicode café and <tag> text  ", " ", "Partial <tool_ca"])
def test_tool_text_stream_plain_text_matches_buffered_for_every_split(text):
    for split in range(len(text) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append)
        stream.feed(text[:split])
        stream.feed(text[split:])
        stream.finish(text)
        assert "".join(delivered) == text


@pytest.mark.parametrize("prefix", ["", "Synthetic preamble", "Synthetic preamble \n\t"])
def test_tool_text_stream_withholds_every_call_marker_split(prefix):
    text = prefix + OUTPUT
    for split in range(len(text) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append)
        stream.feed(text[:split])
        stream.feed(text[split:])
        assert "".join(delivered) == prefix.rstrip()
        message, _ = output(text)
        stream.finish(message["content"])
        assert "".join(delivered) == prefix.rstrip()


def test_tool_text_stream_incremental_text_and_long_whitespace():
    delivered = []
    stream = ToolTextStream(delivered.append)
    stream.feed("First")
    assert delivered == ["First"]  # Visible before the terminal frame.
    for _ in range(2048):
        stream.feed(" \n")
    stream.feed("second")
    assert "".join(delivered) == "First" + " \n" * 2048 + "second"
    stream.finish("First" + " \n" * 2048 + "second")


@pytest.mark.parametrize("suffix", ["<tool_call>{", "</tool_call>", "<tool_callX"])
def test_tool_text_stream_keeps_malformed_wire_private(suffix):
    delivered = []
    stream = ToolTextStream(delivered.append)
    for char in "Preamble \n" + suffix:
        stream.feed(char)
    assert "".join(delivered) == "Preamble"
    with pytest.raises(APIError):
        output("Preamble \n" + suffix)


def test_tool_text_stream_bounds_and_terminal_consistency():
    delivered = []
    stream = ToolTextStream(delivered.append)
    stream.feed(" " * MAX_TOOL_OUTPUT_BYTES)
    stream.feed("overflow")
    assert delivered == [] and stream.whitespace == [] and stream.tail == ""
    with pytest.raises(APIError):
        output(" " * MAX_TOOL_OUTPUT_BYTES + "overflow")
    stream = ToolTextStream(delivered.append)
    stream.feed("First")
    with pytest.raises(ValueError):
        stream.finish("Other")


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


def xml_call(values=None, name="lookup"):
    values = {"key": "a"} if values is None else values
    return (
        "<tool_call>\n<function="
        + name
        + ">\n"
        + "".join("<parameter=" + key + ">\n" + value + "\n</parameter>\n" for key, value in values.items())
        + "</function>\n</tool_call>"
    )


def xml_output(text=None, tools=None, **extra):
    options = {"parser": "qwen38-xml-v1", "finish_reason": "eos", "call_id_prefix": "synthetic"}
    options.update(extra)
    return parse_tool_output(xml_call() if text is None else text, [TOOL] if tools is None else tools, **options)


def xml_property_tool(schema):
    tool = copy.deepcopy(TOOL)
    tool["function"]["parameters"]["properties"]["key"] = schema
    return tool


@pytest.mark.parametrize("parser", ["qwen38-xml-v1", "qwen35-xml-v1"])
def test_xml_complete_call_choices_ids_and_parallel_order(parser):
    message, reason = xml_output(parser=parser)
    assert (message, reason) == ({"role": "assistant", "content": None, "tool_calls": [CALL]}, "tool_calls")
    assert xml_output(parser=parser) == (message, reason)
    for choice in ["required", {"type": "function", "function": {"name": "lookup"}}]:
        assert xml_output(parser=parser, tool_choice=choice)[1] == "tool_calls"
        with pytest.raises(APIError) as error:
            xml_output("Plain reply", parser=parser, tool_choice=choice)
        assert error.value.code == "tool_choice_not_satisfied"
    with pytest.raises(APIError):
        xml_output(parser=parser, tool_choice="none")
    message, reason = xml_output("Preamble \n" + xml_call() + "\n" + xml_call({"key": "b"}), parser=parser)
    assert reason == "tool_calls" and message["content"] == "Preamble"
    assert [call["id"] for call in message["tool_calls"]] == ["call_synthetic_0", "call_synthetic_1"]
    assert [json.loads(call["function"]["arguments"]) for call in message["tool_calls"]] == [
        {"key": "a"},
        {"key": "b"},
    ]
    with pytest.raises(APIError):
        xml_output(xml_call() + xml_call(), parser=parser, parallel_tool_calls=False)


@pytest.mark.parametrize(
    "schema,body,expected",
    [
        ({"type": "string"}, "5", "5"),
        ({"type": "string"}, "null", "null"),
        ({"type": "string"}, "  café <tag> &quot;  ", "  café <tag> &quot;  "),
        ({"type": "string"}, "\nline one\nline two\n", "\nline one\nline two\n"),
        ({"type": "string"}, '"quoted text"', '"quoted text"'),
        ({"type": "string"}, "", ""),
        ({"type": "integer"}, "7", 7),
        ({"type": "number"}, "1.5", 1.5),
        ({"type": "number"}, "7", 7),
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "null"}, "null", None),
        ({"type": "object"}, '{"nested":[true,null,3]}', {"nested": [True, None, 3]}),
        ({"type": "array"}, '[1,"two",false,null]', [1, "two", False, None]),
        ({"type": ["string", "null"]}, "/tmp/synthetic", "/tmp/synthetic"),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "alphabetic", "alphabetic"),
        ({"oneOf": [{"type": "integer"}, {"type": "null"}]}, "null", None),
    ],
)
def test_xml_schema_guided_values_preserve_types_and_raw_string_whitespace(schema, body, expected):
    tool = xml_property_tool(schema)
    validate_parser_tools([tool], "qwen38-xml-v1")
    message, reason = xml_output(xml_call({"key": body}), tools=[tool])
    actual = json.loads(message["tool_calls"][0]["function"]["arguments"])["key"]
    assert reason == "tool_calls" and type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize("parser", ["qwen38-xml-v1", "qwen35-xml-v1"])
@pytest.mark.parametrize("constraint", ["enum", "const"])
@pytest.mark.parametrize(
    "literal,body",
    [
        ("5", "5"),
        ("  raw string\n", "  raw string\n"),
        (7, "7"),
        (1.5, "1.5"),
        (True, "true"),
        (False, "false"),
        (None, "null"),
        ({"nested": [True, None]}, '{"nested":[true,null]}'),
        ([1, "two", False], '[1,"two",false]'),
    ],
)
def test_xml_literal_constraints_conclusively_determine_parameter_representation(parser, constraint, literal, body):
    schema = {constraint: [literal] if constraint == "enum" else literal}
    tool = xml_property_tool(schema)
    validate_parser_tools([tool], parser)
    message, reason = xml_output(xml_call({"key": body}), tools=[tool], parser=parser)
    actual = json.loads(message["tool_calls"][0]["function"]["arguments"])["key"]
    assert reason == "tool_calls" and type(actual) is type(literal) and actual == literal


@pytest.mark.parametrize(
    "schema,body,parser",
    [
        ({"enum": ["5", 5]}, "5", "qwen38-xml-v1"),
        ({"enum": ["null", None]}, "null", "qwen38-xml-v1"),
        ({"enum": ["True", True]}, "True", "qwen35-xml-v1"),
        ({"enum": ['{"nested":true}', {"nested": True}]}, '{"nested":true}', "qwen38-xml-v1"),
    ],
)
def test_xml_mixed_literal_constraints_preserve_ambiguous_value_rejection(schema, body, parser):
    tool = xml_property_tool(schema)
    validate_parser_tools([tool], parser)
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": body}), tools=[tool], parser=parser)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"


def test_xml_unsupported_schema_error_identifies_only_a_fixed_form_category():
    tool = xml_property_tool({"format": "Synthetic schema text must stay private"})
    with pytest.raises(APIError) as error:
        validate_parser_tools([tool], "qwen38-xml-v1")
    assert "untyped_constraint_or_unknown_keyword" in error.value.message
    assert "Synthetic schema text" not in error.value.message and "lookup" not in error.value.message
    tool["function"]["parameters"]["additionalProperties"] = True
    with pytest.raises(APIError) as error:
        validate_parser_tools([tool], "qwen38-xml-v1")
    assert "root_additional_properties" in error.value.message


@pytest.mark.parametrize(
    "body,expected,kind", [("True", True, "boolean"), ("False", False, "boolean"), ("None", None, "null")]
)
def test_qwen35_xml_typed_scalar_lexemes_are_bound_to_its_parser_flavor(body, expected, kind):
    tool = xml_property_tool({"type": kind})
    message, _ = xml_output(xml_call({"key": body}), tools=[tool], parser="qwen35-xml-v1")
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["key"] is expected
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": body}), tools=[tool], parser="qwen38-xml-v1")
    assert error.value.code == "invalid_tool_call"
    string_message, _ = xml_output(xml_call({"key": body}), parser="qwen35-xml-v1")
    assert json.loads(string_message["tool_calls"][0]["function"]["arguments"])["key"] == body


@pytest.mark.parametrize(
    "schema,body,parser",
    [
        ({"type": ["string", "null"]}, "null", "qwen38-xml-v1"),
        ({"type": ["string", "integer"]}, "5", "qwen38-xml-v1"),
        ({"type": ["string", "object"]}, '{"x":1}', "qwen38-xml-v1"),
        ({"type": ["string", "array"]}, "[]", "qwen38-xml-v1"),
        ({"type": ["string", "boolean"]}, "True", "qwen35-xml-v1"),
        ({"type": ["string", "null"]}, "None", "qwen35-xml-v1"),
    ],
)
def test_xml_ambiguous_union_values_never_become_executable_calls(schema, body, parser):
    tool = xml_property_tool(schema)
    validate_parser_tools([tool], parser)
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": body}), tools=[tool], parser=parser)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"


def test_xml_bounded_local_schema_references_and_nested_json():
    tool = xml_property_tool({"$ref": "#/$defs/path~1type"})
    tool["function"]["parameters"]["$defs"] = {"path/type": {"type": "string"}}
    validate_parser_tools([tool], "qwen38-xml-v1")
    message, _ = xml_output(xml_call({"key": "  /tmp/example  "}), tools=[tool])
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": "  /tmp/example  "}
    tool = xml_property_tool({"$ref": "#/definitions/entry"})
    tool["function"]["parameters"]["definitions"] = {"entry": {"type": "object"}}
    message, _ = xml_output(xml_call({"key": '{"nested":[true,null]}'}), tools=[tool], parser="qwen35-xml-v1")
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": {"nested": [True, None]}}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": None},
        {"type": []},
        {"type": "arbitrary"},
        {"type": True},
        {"enum": []},
        {"enum": "not-a-literal-list"},
        {"type": "string", "enum": None},
        {"$ref": "https://example.invalid/type"},
        {"$ref": "#/$defs/missing"},
        {"anyOf": []},
        {"allOf": [{"type": "string"}]},
        {"description": None},
        {"title": 1},
        {"examples": "invalid"},
        {"deprecated": "invalid"},
        {"format": "unknown-without-type"},
    ],
)
def test_xml_unsupported_parameter_schemas_rejected_before_generation(schema):
    with pytest.raises(APIError) as error:
        validate_parser_tools([xml_property_tool(schema)], "qwen38-xml-v1")
    assert error.value.status == 400 and error.value.code == "tool_schema_unsupported"
    # The existing JSON wire path does not need these XML representation rules.
    validate_parser_tools([xml_property_tool(schema)], "qwen3-hermes-v1")


@pytest.mark.parametrize("parser", ["qwen38-xml-v1", "qwen35-xml-v1"])
@pytest.mark.parametrize(
    "unknown_schema",
    [
        {},
        {"description": "Optional annotation-only input", "default": None},
        {"title": "Input", "$comment": "Metadata", "examples": [None], "deprecated": False},
        {"readOnly": True, "writeOnly": False},
        {"anyOf": [{"type": "string"}, {}]},
        {"oneOf": [{"description": "No representation"}, {"type": "integer"}]},
        {"$ref": "#/$defs/untyped"},
    ],
)
def test_xml_unused_unknown_parameters_preserve_chat_and_typed_tool_calls(parser, unknown_schema):
    tool = copy.deepcopy(TOOL)
    tool["function"]["parameters"]["properties"]["hint"] = unknown_schema
    tool["function"]["parameters"]["$defs"] = {"untyped": {}}
    original = copy.deepcopy(tool)
    validate_parser_tools([tool], parser)
    assert tool == original  # Definitions are forwarded without inferred types.
    assert xml_output("Ordinary chat", tools=[tool], parser=parser) == (
        {"role": "assistant", "content": "Ordinary chat"},
        "stop",
    )
    message, reason = xml_output(xml_call(), tools=[tool], parser=parser)
    assert reason == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": "a"}
    # A completed typed call preceding an untyped one is still withheld: the
    # completion is atomic and no generated text is used to guess a type.
    wire = "Preamble\n" + xml_call() + xml_call({"key": "b", "hint": "5"})
    delivered = []
    stream = ToolTextStream(delivered.append)
    for character in wire:
        stream.feed(character)
    with pytest.raises(APIError) as error:
        xml_output(wire, tools=[tool], parser=parser)
    assert error.value.status == 502 and error.value.code == "tool_argument_type_unsupported"
    assert "".join(delivered) == "Preamble"
    # An unknown union branch cannot be bypassed by a value that fits its known
    # branch. This second request must fail for the same fixed reason.
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"hint": "plain text"}), tools=[tool], parser=parser)
    assert error.value.code == "tool_argument_type_unsupported"


def test_xml_unknown_alternative_still_validates_all_remaining_branches():
    tool = xml_property_tool({"anyOf": [{}, {"type": None}]})
    with pytest.raises(APIError) as error:
        validate_parser_tools([tool], "qwen38-xml-v1")
    assert error.value.code == "tool_schema_unsupported" and "declared_type" in error.value.message


@pytest.mark.parametrize("parser", ["qwen38-xml-v1", "qwen35-xml-v1"])
@pytest.mark.parametrize(
    "root_fields",
    [
        {"$ref": "#/$defs/parameters"},
        {"$dynamicRef": "#/$defs/parameters"},
        {"$recursiveRef": "#"},
        {"allOf": [{"properties": {"other": {"type": "string"}}}]},
        {"anyOf": [{"properties": {"other": {"type": "string"}}}]},
        {"oneOf": [{"properties": {"other": {"type": "string"}}}]},
        {"if": {"required": ["key"]}, "then": {"properties": {"other": {"type": "string"}}}},
        {"not": {"required": ["key"]}},
        {"patternProperties": {".*": {"type": "string"}}},
        {"dependentSchemas": {"key": {"properties": {"other": {"type": "string"}}}}},
        {"dependencies": {"key": {"properties": {"other": {"type": "string"}}}}},
        {"unevaluatedProperties": {"type": "string"}},
        {"additionalProperties": True},
        {"additionalProperties": {"type": "string"}},
    ],
)
def test_xml_dynamic_and_composed_root_parameters_are_rejected_early(parser, root_fields):
    tool = xml_property_tool({"type": "string"})
    tool["function"]["parameters"].update(root_fields)
    with pytest.raises(APIError) as error:
        validate_parser_tools([tool], parser)
    assert error.value.status == 400 and error.value.code == "tool_schema_unsupported"
    validate_parser_tools([tool], "qwen3-hermes-v1")


@pytest.mark.parametrize("parser", ["qwen38-xml-v1", "qwen35-xml-v1"])
def test_xml_direct_parameters_allow_explicit_closed_object_schema(parser):
    tool = xml_property_tool({"$ref": "#/$defs/value"})
    tool["function"]["parameters"].update({"additionalProperties": False, "$defs": {"value": {"type": "string"}}})
    validate_parser_tools([tool], parser)
    message, reason = xml_output(xml_call({"key": "unchanged"}), tools=[tool], parser=parser)
    assert reason == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": "unchanged"}


def test_xml_schema_reference_cycles_and_bad_parameter_names_rejected():
    tool = xml_property_tool({"$ref": "#/$defs/cycle"})
    tool["function"]["parameters"]["$defs"] = {"cycle": {"$ref": "#/$defs/cycle"}}
    with pytest.raises(APIError):
        validate_parser_tools([tool], "qwen38-xml-v1")
    tool = copy.deepcopy(TOOL)
    tool["function"]["parameters"]["properties"] = {"not a wire name": {"type": "string"}}
    with pytest.raises(APIError):
        validate_parser_tools([tool], "qwen38-xml-v1")


def test_xml_repeated_local_reference_graph_is_resolved_without_exponential_work():
    tool = xml_property_tool({"$ref": "#/$defs/level15"})
    definitions = {"level0": {"type": "string"}}
    for level in range(1, 16):
        definitions[f"level{level}"] = {
            "anyOf": [{"$ref": f"#/$defs/level{level - 1}"}, {"$ref": f"#/$defs/level{level - 1}"}]
        }
    tool["function"]["parameters"]["$defs"] = definitions
    validate_parser_tools([tool], "qwen38-xml-v1")
    message, _ = xml_output(xml_call({"key": "resolved"}), tools=[tool])
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": "resolved"}


def test_xml_local_reference_expansion_has_a_depth_limit():
    tool = xml_property_tool({"$ref": "#/$defs/level36"})
    definitions = {"level0": {"type": "string"}}
    for level in range(1, 37):
        definitions[f"level{level}"] = {"$ref": f"#/$defs/level{level - 1}"}
    tool["function"]["parameters"]["$defs"] = definitions
    with pytest.raises(APIError) as error:
        validate_parser_tools([tool], "qwen38-xml-v1")
    assert error.value.code == "tool_schema_unsupported"


@pytest.mark.parametrize("parser", ["qwen3-hermes-v1", "qwen38-xml-v1", "qwen35-xml-v1"])
def test_completed_thinking_before_a_call_is_distinct_from_a_call_inside_thinking(parser):
    wire = OUTPUT if parser == "qwen3-hermes-v1" else xml_call()
    prefix = '<think>Completed reasoning containing an open quotation " and ```ordinary code```.</think>\n'
    message, reason = output(prefix + wire, parser=parser)
    assert reason == "tool_calls" and message["content"] == prefix.rstrip()
    for unsafe in [
        "<think>Unclosed reasoning " + wire,
        "<think>Quoted call " + wire + "</think>",
        "<think>Nested <think>reasoning</think></think>" + wire,
        "</think>" + wire,
        "Example: 'I'm describing a quoted call " + wire,
        'Example: "' + wire,
        "Example: '" + wire,
        "Example: “" + wire,
        "Example: `" + wire,
        "```\n" + wire + "\n```",
    ]:
        with pytest.raises(APIError) as error:
            output(unsafe, parser=parser)
        assert error.value.code == "invalid_tool_call"


def test_xml_zero_argument_function_and_named_choice_are_strict():
    empty = {"type": "function", "function": {"name": "empty", "parameters": {"type": "object", "properties": {}}}}
    message, reason = xml_output(xml_call({}, "empty"), tools=[empty])
    assert reason == "tool_calls" and message["tool_calls"][0]["function"] == {"name": "empty", "arguments": "{}"}
    with pytest.raises(APIError):
        xml_output(
            xml_call({}, "empty"),
            tools=[TOOL, empty],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
        )


@pytest.mark.parametrize("finish_reason", ["length", "context", "cancelled", "error", "stop_sequence"])
def test_xml_calls_require_a_normal_terminal_frame(finish_reason):
    with pytest.raises(APIError) as error:
        xml_output(finish_reason=finish_reason)
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize(
    "text",
    [
        xml_call()[:-1],
        xml_call().replace("</function>", ""),
        xml_call().replace("</parameter>", ""),
        xml_call().replace("<function=lookup>", "<function=unknown>"),
        xml_call().replace("<parameter=key>", "<parameter=unknown>"),
        xml_call().replace("</function>", "<parameter=key>\nb\n</parameter>\n</function>"),
        xml_call() + " trailing",
        xml_call() + "\n<tool_call>",
        xml_call().replace("<tool_call>", "<tool_call name=lookup>"),
        xml_call().replace("<parameter=key>\na\n", "<parameter=key>\na"),
        xml_call().replace("<function=lookup>", "<function=" + "a" * 65 + ">"),
        xml_call().replace("<parameter=key>", "<parameter=" + "a" * 65 + ">"),
        "<function=lookup>\n</function>",
        "<think>unfinished reasoning\n" + xml_call(),
        "<think>quoted example " + xml_call() + "</think>",
        "```xml\n" + xml_call() + "\n```",
        'An open quotation "' + xml_call(),
        "An open quotation '" + xml_call(),
        "An open quotation “" + xml_call(),
        "An open code quotation `" + xml_call(),
        xml_call({"key": "x</parameter>y"}),
        xml_call({"key": "x<function=unavailable>y"}),
        xml_call({"key": "x<tool_call>y"}),
    ],
)
def test_xml_malformed_unknown_quoted_or_ambiguous_wire_is_not_repaired(text):
    with pytest.raises(APIError) as error:
        xml_output(text)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"


def test_xml_natural_preamble_and_inline_string_values_remain_lossless():
    text = 'I\'ll look up "weather".\n' + xml_call().replace("<parameter=key>\na\n", "<parameter=key>a")
    message, reason = xml_output(text)
    assert reason == "tool_calls" and message["content"] == 'I\'ll look up "weather".'
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"key": "a"}


@pytest.mark.parametrize(
    "schema,body",
    [
        ({"type": "integer"}, "true"),
        ({"type": "integer"}, "1.5"),
        ({"type": "boolean"}, "1"),
        ({"type": "object"}, "[]"),
        ({"type": "array"}, "{}"),
        ({"type": "number"}, "NaN"),
        ({"type": "number"}, "Infinity"),
        ({"type": "number"}, "1e999"),
        ({"type": "object"}, '{"x":1,"x":2}'),
        ({"type": "array"}, "[1,]"),
        ({"type": "object"}, '{"x":"\\ud800"}'),
    ],
)
def test_xml_nonstring_values_require_strict_bounded_json_of_the_declared_type(schema, body):
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": body}), tools=[xml_property_tool(schema)])
    assert error.value.code == "invalid_tool_call"


def test_xml_call_argument_depth_bytes_and_count_limits():
    for text, tools in [
        (xml_call({"key": "x" * MAX_ARGUMENT_BYTES}), [TOOL]),
        (xml_call({"key": "[" * 33 + "0" + "]" * 33}), [xml_property_tool({"type": "array"})]),
        (xml_call() * 17, [TOOL]),
    ]:
        with pytest.raises(APIError) as error:
            xml_output(text, tools=tools)
        assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("parser", ["qwen38-xml-v1", "qwen35-xml-v1"])
def test_xml_stream_content_matches_terminal_without_exposing_wire_at_any_split(parser):
    source = "Leading text \n" + xml_call() + "\n" + xml_call({"key": "b"})
    message, reason = xml_output(source, parser=parser)
    for split in range(len(source) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append)
        stream.feed(source[:split])
        stream.feed(source[split:])
        stream.finish(message["content"])
        assert reason == "tool_calls" and "".join(delivered) == "Leading text"


def test_xml_template_hash_admission_is_exact_and_flavor_specific(monkeypatch):
    import stark_convert.api_tools as api_tools

    assert QWEN38_XML_TEMPLATE_SHA256 == "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    assert QWEN35_XML_TEMPLATE_SHA256 == "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    # Synthetic programs exercise the exact hash gate without redistributing
    # checkpoint templates whose license terms have not been established here.
    for constant, source, parser in [
        ("QWEN38_XML_TEMPLATE_SHA256", "synthetic Lily template <tool_call>", "qwen38-xml-v1"),
        ("QWEN35_XML_TEMPLATE_SHA256", "synthetic Jev template <tool_call>", "qwen35-xml-v1"),
    ]:
        monkeypatch.setattr(api_tools, constant, hashlib.sha256(source.encode()).hexdigest())
        assert tool_parser_for({"prompt_template": source}) == parser
        assert tool_parser_for({"prompt_template": source + " "}) is None
    assert tool_parser_for({"prompt_template": "a different template with <function= and <parameter="}) is None


def test_xml_tool_result_history_preserves_values_order_and_followup_rendering():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    second = copy.deepcopy(TOOL)
    second["function"]["name"] = "lookup_two"
    message, _ = xml_output(
        xml_call({"key": "  first\n"}) + xml_call({"key": "second"}, "lookup_two"), tools=[TOOL, second]
    )
    calls = message["tool_calls"]
    messages = template_messages(
        [
            {"role": "user", "content": "Synthetic first turn"},
            message,
            {"role": "tool", "tool_call_id": calls[1]["id"], "content": "Second result"},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": "First result"},
            {"role": "user", "content": "Synthetic follow-up"},
        ]
    )
    # A minimal synthetic protocol renderer exercises the public history
    # contract. Model-bound validation separately renders the exact templates.
    template = ImmutableSandboxedEnvironment().from_string(
        "{{ tools|tojson }}\n{% for m in messages %}"
        "{% if m.role == 'assistant' %}{{ m.content }}{% for c in m.tool_calls %}"
        "<tool_call>\n<function={{ c.function.name }}>\n"
        "{% for key,value in c.function.arguments.items() %}<parameter={{ key }}>\n"
        "{{ value }}\n</parameter>\n{% endfor %}"
        "</function>\n</tool_call>{% endfor %}"
        "{% elif m.role == 'tool' %}<tool_response>\n{{ m.content }}\n</tool_response>"
        "{% else %}{{ m.content }}{% endif %}{% endfor %}"
    )
    rendered = template.render(messages=messages, tools=[TOOL, second])
    assert messages[1]["content"] == ""
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"key": "  first\n"}
    assert "Synthetic first turn" in rendered and "Synthetic follow-up" in rendered
    assert rendered.index("<function=lookup>") < rendered.index("<function=lookup_two>")
    assert rendered.index("<tool_response>\nFirst result") < rendered.index("<tool_response>\nSecond result")
    assert "<parameter=key>\n  first\n\n</parameter>" in rendered
