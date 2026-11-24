import json

import pytest

from stark_convert.api_tools import parse_tool_output
from stark_convert.inkling_chat import PARSER, ResponseStream, parse_response
from stark_convert.local_api import APIError

_THOUGHT = "<|message_model|><|content_thinking|>Use the lookup.<|end_message|>"
_TEXT = "<|message_model|><|content_text|>I can check. <|end_message|>"
_CALL = (
    "<|message_model|>lookup<|content_invoke_tool_json|>"
    '{"name":"lookup","args":{"term":"café <|end_message|>"}}<|end_message|>'
)
_TOOLS = [{"type": "function", "function": {"name": "lookup"}}]


def _parse(text, finish="eos"):
    return parse_tool_output(text, _TOOLS, parser=PARSER, finish_reason=finish, call_id_prefix="test")


def test_inkling_stream_hides_tool_arguments_across_every_split():
    raw = _THOUGHT + _TEXT + _CALL + _TEXT
    expected, _, _ = parse_response(raw)
    for split in range(len(raw) + 1):
        content, thinking = [], []
        stream = ResponseStream(content.append, thinking.append)
        stream.feed(raw[:split])
        stream.feed(raw[split:])
        assert "café" not in "".join(content + thinking)
        stream.finish(expected)
        assert "".join(content) == expected["content"] == "I can check. I can check. "
        assert "".join(thinking) == expected["reasoning_content"] == "Use the lookup."


def test_inkling_tool_frame_preserves_nested_json_and_controls_inside_strings():
    message, reason = _parse(_THOUGHT + _TEXT + _CALL)
    assert reason == "tool_calls"
    assert message["content"] == "I can check. "
    assert message["reasoning_content"] == "Use the lookup."
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"term": "café <|end_message|>"}


def test_inkling_tool_error_marker_in_visible_text_stays_literal():
    literal = "This is a <|content_tool_error|> marker."
    error = "<|content_tool_error|>hidden error<|end_message|>"
    raw = "<|message_model|><|content_text|>" + literal + "<|end_message|>" + error
    expected, _, _ = parse_response(raw)
    assert expected["content"] == literal
    for split in range(len(raw) + 1):
        text = []
        stream = ResponseStream(text.append, lambda _: None)
        stream.feed(raw[:split])
        stream.feed(raw[split:])
        stream.finish(expected)
        assert "".join(text) == literal


@pytest.mark.parametrize(
    ("raw", "finish"),
    [
        (_CALL.replace("lookup<|content", "other<|content"), "eos"),
        (_CALL.replace("<|message_model|>lookup", ""), "eos"),
        (_CALL.replace('"term":', '"term":0,"term":'), "eos"),
        (_CALL.removesuffix("<|end_message|>"), "eos"),
        (_CALL, "length"),
    ],
)
def test_inkling_rejects_incomplete_or_inconsistent_calls(raw, finish):
    with pytest.raises(APIError) as error:
        _parse(raw, finish)
    assert error.value.code == "invalid_tool_call"
