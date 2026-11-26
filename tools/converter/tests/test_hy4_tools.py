"""Regression coverage for the source-bound Hy4 Flash tool protocol."""

import json

import pytest

from stark_convert import api_tools
from stark_convert.api_tools import ToolTextStream, parse_tool_output, tool_parser_for, validate_parser_tools
from stark_convert.local_api import APIError

PARSER = "hy4-flash-xml-v1"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "options": {"type": "object"},
            },
        },
    },
}


def frame(name="lookup", parameters=()):
    body = "".join(
        f"<arg_key:opensource>{key}</arg_key:opensource><arg_value:opensource>{value}</arg_value:opensource>"
        for key, value in parameters
    )
    return f"<tool_call:opensource>{name}{body}</tool_call:opensource>"


def output(text, tools=None, **options):
    return parse_tool_output(
        text,
        [TOOL] if tools is None else tools,
        parser=PARSER,
        finish_reason=options.pop("finish_reason", "eos"),
        call_id_prefix=options.pop("call_id_prefix", "hy4test"),
        **options,
    )


def wrapped(*frames):
    return "<tool_calls:opensource>" + "".join(frames) + "</tool_calls:opensource>"


def test_template_admission_is_bound_to_the_pinned_hy4_source():
    assert api_tools.HY4_TEMPLATE_SHA256 == "11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35"
    assert tool_parser_for({"prompt_template": "synthetic reviewed Hy4 Flash template"}) is None


def test_required_outer_wrapper_suffix_and_strict_glm_typed_arguments():
    validate_parser_tools([TOOL], PARSER)
    text = "<think:opensource>checking inputs</think:opensource>" + wrapped(
        frame(parameters=[("query", "find cafés"), ("limit", "7"), ("options", '{"nested":[true,null,3]}')])
    )
    message, reason = output(text)
    assert reason == "tool_calls"
    assert message["content"] == "<think:opensource>checking inputs</think:opensource>"
    call = message["tool_calls"][0]
    assert call["id"] == "call_hy4test_0"
    assert json.loads(call["function"]["arguments"]) == {
        "query": "find cafés",
        "limit": 7,
        "options": {"nested": [True, None, 3]},
    }


@pytest.mark.parametrize(
    "text",
    [
        "<tool_calls:opensource><tool_call>lookup<arg_key>query</arg_key><arg_value>x</arg_value></tool_call></tool_calls:opensource>",
        "<tool_calls:opensource>" + frame() + "</tool_calls>",
        "<tool_call>lookup:other</tool_call>",
        "<tool_calls:opensource>" + frame() + "",
        frame().replace(":opensource", "", 1),
        frame(parameters=[("query", "unfinished")])[:-1],
    ],
)
def test_missing_or_malformed_hy4_wrapper_and_suffix_are_rejected(text):
    with pytest.raises(APIError) as error:
        output(text)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"


@pytest.mark.parametrize(
    "text",
    [
        "<think:opensource>unfinished reasoning " + wrapped(frame()),
        '<think:opensource>quoted example "' + wrapped(frame()) + "</think:opensource>",
        "Example: '" + wrapped(frame()),
        "> quoted\n" + wrapped(frame()),
        "```xml\n" + wrapped(frame()) + "\n```",
        wrapped(frame(parameters=[("query", "x</arg_value><arg_key>limit</arg_key>")])),
    ],
)
def test_reasoning_quotes_and_argument_markup_cannot_smuggle_calls(text):
    with pytest.raises(APIError) as error:
        output(text)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"
    assert "lookup" not in error.value.message


def test_parallel_calls_order_ids_and_tool_choice_controls():
    second = {"type": "function", "function": {**TOOL["function"], "name": "search"}}
    text = wrapped(frame(parameters=[("query", "one")]), frame("search", [("query", "two")]))
    message, reason = output(text, [TOOL, second])
    assert reason == "tool_calls"
    assert [call["id"] for call in message["tool_calls"]] == ["call_hy4test_0", "call_hy4test_1"]
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["lookup", "search"]
    assert [json.loads(call["function"]["arguments"]) for call in message["tool_calls"]] == [
        {"query": "one"},
        {"query": "two"},
    ]

    with pytest.raises(APIError) as error:
        output(text, [TOOL, second], parallel_tool_calls=False)
    assert error.value.code == "invalid_tool_call"
    with pytest.raises(APIError) as error:
        output(text, [TOOL, second], tool_choice="none")
    assert error.value.code == "invalid_tool_call"
    with pytest.raises(APIError) as error:
        output("ordinary response", tool_choice="required")
    assert error.value.code == "tool_choice_not_satisfied"
    with pytest.raises(APIError) as error:
        output(wrapped(frame()), tool_choice={"type": "function", "function": {"name": "search"}})
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("finish_reason", ["length", "context", "cancelled", "error", "stop_sequence"])
def test_hy4_call_is_not_released_without_a_normal_terminal_frame(finish_reason):
    text = "<tool_calls:opensource>" + frame() + "</tool_calls:opensource>"
    with pytest.raises(APIError) as error:
        output(text, finish_reason=finish_reason)
    assert error.value.code == "invalid_tool_call"


def test_chunked_stream_keeps_outer_and_inner_call_markup_private_at_every_split():
    text = "Visible prefix\n" + wrapped(frame())
    content = "Visible prefix"
    for split in range(len(text) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append, parser=PARSER)
        stream.feed(text[:split])
        stream.feed(text[split:])
        assert "".join(delivered) == content
        stream.finish(content)
        assert "".join(delivered) == content
