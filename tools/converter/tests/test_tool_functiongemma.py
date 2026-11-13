"""Strict synthetic tests for FunctionGemma's bounded call wire format."""

import hashlib
from pathlib import Path

import pytest

from stark_convert import tool_functiongemma
from stark_convert.chat import _ChatTemplate
from stark_convert.tool_functiongemma import parse_functiongemma_calls

FIXTURES = Path(__file__).parent / "fixtures" / "tool_templates"
TEMPLATES = {
    "mlx": (
        "functiongemma_mlx_chat_template.jinja",
        "db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b",
    ),
    "unsloth": (
        "functiongemma_unsloth_chat_template.jinja",
        "0dbaf70819c970c9bacd7125be8e9db46b7bc0e6ed3bc303d92bd58fb9dd604a",
    ),
}


class ParseFailure(Exception):
    def __init__(self, category):
        self.category = category


def invalid(category):
    return ParseFailure(category)


def parse(text):
    return parse_functiongemma_calls(text, invalid=invalid)


def assert_invalid(text, category=None):
    with pytest.raises(ParseFailure) as raised:
        parse(text)
    if category is not None:
        assert raised.value.category == category


@pytest.fixture
def small_byte_budget(monkeypatch):
    monkeypatch.setattr(tool_functiongemma, "_MAX_OUTPUT_BYTES", 512 * 1024)
    monkeypatch.setattr(tool_functiongemma, "_MAX_ARGUMENT_BYTES", 64 * 1024)


@pytest.mark.parametrize("fixture,expected", TEMPLATES.values())
def test_fixture_bytes_are_pinned(fixture, expected):
    source = (FIXTURES / fixture).read_bytes()
    assert hashlib.sha256(source).hexdigest() == expected


@pytest.mark.parametrize("fixture,_digest", TEMPLATES.values())
def test_native_templates_render_calls_and_structured_tool_results(fixture, _digest):
    values = {
        "text": " café, colon: braces {} and brackets []\nnext line ",
        "flag": True,
        "none_value": None,
        "integer": -42,
        "decimal": 2.5e-7,
        "nested": {"items": [False, "λ", {"count": 3}]},
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Synthetic harmless lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Input text"},
                        "flag": {"type": "boolean", "description": "Choice"},
                        "none_value": {"type": "null", "description": "Empty value"},
                        "integer": {"type": "integer", "description": "Count"},
                        "decimal": {"type": "number", "description": "Scale"},
                        "nested": {
                            "type": "object",
                            "description": "Nested value",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "description": "Nested items",
                                    "items": {"type": "string"},
                                }
                            },
                            "required": ["items"],
                        },
                    },
                    "required": ["text"],
                },
            },
        }
    ]
    call = {"type": "function", "function": {"name": "lookup", "arguments": values}}
    messages = [
        {"role": "user", "content": "Synthetic request"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "content": {"name": "lookup", "response": {"found": True, "count": 1}}},
        {"role": "assistant", "content": "Synthetic completed answer"},
    ]
    source = (FIXTURES / fixture).read_text()
    rendered = _ChatTemplate(source).render(
        messages=messages,
        tools=tools,
        add_generation_prompt=False,
        bos_token="<bos>",
    )
    assert "<start_function_declaration>" in rendered
    assert "<start_function_response>response:lookup{" in rendered
    assistant_turn = rendered.split("<start_of_turn>model\n", 1)[1].split("<end_function_call>", 1)[0]
    assistant_turn += "<end_function_call>"
    calls = parse(assistant_turn)
    assert calls == [("lookup", values)]


def test_native_template_null_literal_is_none_and_json_null_is_also_decoded():
    # Both pinned templates stringify Python None as `None`; `null` is accepted
    # as the corresponding JSON scalar in generated structured values.
    assert parse("<start_function_call>call:f{value:None}<end_function_call>") == [("f", {"value": None})]
    assert parse("<start_function_call>call:f{value:null}<end_function_call>") == [("f", {"value": None})]


@pytest.mark.parametrize("fixture,_digest", TEMPLATES.values())
def test_template_places_response_marker_after_completed_calls(fixture, _digest):
    tools = [{"type": "function", "function": {"name": "lookup", "description": "d", "parameters": {}}}]
    messages = [
        {"role": "user", "content": "Synthetic request"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"key": "v"}}}],
        },
    ]
    rendered = _ChatTemplate((FIXTURES / fixture).read_text()).render(
        messages=messages, tools=tools, add_generation_prompt=False, bos_token="<bos>"
    )
    assert rendered.endswith(
        "<start_function_call>call:lookup{key:<escape>v<escape>}<end_function_call><start_function_response>"
    )


def test_preserves_string_payload_exactly_and_parses_parallel_calls():
    text = (
        "<start_function_call>call:a{text:<escape> café, {x}: [y]\n\t <escape>}<end_function_call>\n\t"
        "<start_function_call>call:b{count:1e2}<end_function_call>"
    )
    assert parse(text) == [("a", {"text": " café, {x}: [y]\n\t "}), ("b", {"count": 100.0})]


def test_supports_bounded_nested_containers_and_escaped_keys():
    text = (
        "<start_function_call>call:f{outer:{<escape>non identifier key<escape>:[true, null, "
        "<escape>value<escape>]}}<end_function_call>"
    )
    assert parse(text) == [("f", {"outer": {"non identifier key": [True, None, "value"]}})]


@pytest.mark.parametrize(
    "text,category",
    [
        ("", "format"),
        ("ordinary text", "format"),
        ("intro <start_function_call>call:f{}<end_function_call>", "format"),
        ("<start_function_call>call:f{}<end_function_call> trailing", "trailing"),
        (
            "<start_function_call>call:f{}<end_function_call><start_function_call>call:g{}<end_function_call>x",
            "trailing",
        ),
        ("<start_function_call>call:f {}<end_function_call>", "format"),
        ('<start_function_call>call:f{value:"raw"}<end_function_call>', "value"),
        ("<start_function_call>call:f{value:undefined}<end_function_call>", "value"),
        ("<start_function_call>call:f{value:NaN}<end_function_call>", "value"),
        ("<start_function_call>call:f{value:Infinity}<end_function_call>", "value"),
        ("<start_function_call>call:f{value:1e999}<end_function_call>", "value"),
        ("<start_function_call>call:f{value:01}<end_function_call>", "format"),
        ("<start_function_call>call:f{value:<escape>bad <escape> marker<escape>}<end_function_call>", "format"),
        ("<start_function_call>call:f{bad-key:1}<end_function_call>", "format"),
        ("<start_function_call>call:f{x:1,x:2}<end_function_call>", "argument"),
        ("<start_function_call>call:f{x:{y:1,y:2}}<end_function_call>", "argument"),
        ("<start_function_call>call:f{x:[1,]}<end_function_call>", "value"),
        ("<start_function_call>call:f{x:{y:1}<end_function_call>", "format"),
    ],
)
def test_rejects_noncanonical_or_ambiguous_blocks(text, category):
    assert_invalid(text, category)


def test_rejects_invalid_names_and_missing_terminators():
    assert_invalid("<start_function_call>call:has.dot{}<end_function_call>", "format")
    assert_invalid("<start_function_call>call:f{value:1}", "format")
    assert_invalid("<start_function_call>call:f{value:<escape>unfinished}", "format")


def test_rejects_nesting_node_argument_and_output_overlimits(small_byte_budget):
    nested = "[" * 33 + "0" + "]" * 33
    assert_invalid(f"<start_function_call>call:f{{value:{nested}}}<end_function_call>", "limits")
    many = ",".join(f"k{i}:0" for i in range(9000))
    assert_invalid(f"<start_function_call>call:f{{{many}}}<end_function_call>", "limits")
    assert_invalid(
        "<start_function_call>call:f{value:<escape>" + "x" * (64 * 1024 + 1) + "<escape>}<end_function_call>",
        "limits",
    )
    assert_invalid(" " * (512 * 1024 + 1), "limits")


def test_rejects_flat_scalar_node_overlimit_and_excess_call_count():
    values = ",".join(["false"] * 16_384)
    assert_invalid(f"<start_function_call>call:f{{items:[{values}]}}<end_function_call>", "limits")
    calls = " ".join("<start_function_call>call:f{}<end_function_call>" for _ in range(17))
    assert_invalid(calls, "limits")


def test_multibyte_argument_budget_uses_exact_utf8_byte_boundary(small_byte_budget):
    opening = "{value:<escape>"
    closing = "<escape>}"
    fill_bytes = 64 * 1024 - len(opening.encode()) - len(closing.encode())
    assert fill_bytes % 2 == 0
    exact = "é" * (fill_bytes // 2)
    exact_call = f"<start_function_call>call:f{{value:<escape>{exact}<escape>}}<end_function_call>"
    assert parse(exact_call) == [("f", {"value": exact})]
    over = exact + "x"
    over_call = f"<start_function_call>call:f{{value:<escape>{over}<escape>}}<end_function_call>"
    assert_invalid(over_call, "limits")


def test_rejects_non_text_input():
    assert_invalid(None, "format")


def test_shared_budget_accepts_string_larger_than_old_output_and_argument_caps():
    value = "é" * (300 * 1024)
    assert len(value.encode("utf-8")) > 512 * 1024
    text = f"<start_function_call>call:f{{value:<escape>{value}<escape>}}<end_function_call>"
    assert parse(text) == [("f", {"value": value})]
