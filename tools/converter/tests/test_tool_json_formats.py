"""Source-bound tests for Mistral and Llama JSON tool protocols."""

import json
import re
from pathlib import Path

import pytest
from test_desktop import desktop as desktop
from test_desktop import executable as executable
from test_desktop import model as model

from stark_convert.api_tools import (
    ToolTextStream,
    parse_tool_output,
    template_messages,
    template_tools,
    tool_parser_for,
)
from stark_convert.chat import _ChatTemplate
from stark_convert.local_api import APIError

FIXTURES = Path(__file__).parent / "fixtures" / "tool_templates"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": 'Find "quoted" data\\safely\nand retain Unicode café.',
            "parameters": {
                "type": "object",
                "properties": {
                    "null_value": {"type": "null"},
                    "boolean": {"type": "boolean"},
                    "integer": {"type": "integer"},
                    "number": {"type": "number"},
                    "string": {"type": "string"},
                    "array": {"type": "array", "items": {"type": "string"}},
                    "object": {"type": "object"},
                },
            },
        },
    }
]
ARGS = {
    "null_value": None,
    "boolean": True,
    "integer": 7,
    "number": 1.25,
    "string": 'quoted "value"\\path\n雪',
    "array": ["one", "two"],
    "object": {"nested": False},
}


def _render(name, messages, tools=None, **extra):
    source = (FIXTURES / name).read_text()
    template = _ChatTemplate(source)
    values = {
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    values.update(extra)
    return template.render(**values)


@pytest.mark.parametrize(
    ("fixture", "parser"),
    [("mistral_v03.jinja", "mistral-json-v1"), ("llama31.jinja", "llama-json-v1"), ("llama32.jinja", "llama-json-v1")],
)
def test_exact_source_fixtures_are_admitted(fixture, parser):
    source = (FIXTURES / fixture).read_text()
    assert tool_parser_for({"prompt_template": source}) == parser


@pytest.mark.parametrize("parser,text", [("llama-json-v1", "[1, 2]"), ("phi4-json-v1", '{"value": 2}')])
def test_other_json_container_shapes_remain_ordinary_text(parser, text):
    message, reason = parse_tool_output(text, TOOLS, parser=parser, finish_reason="eos", call_id_prefix="synthetic")
    assert message == {"role": "assistant", "content": text} and reason == "stop"
    delivered = []
    stream = ToolTextStream(delivered.append, parser=parser)
    for character in text:
        stream.feed(character)
    stream.finish(message["content"])
    assert "".join(delivered) == text


def test_mistral_render_escapes_definition_and_result_and_preserves_id_linkage():
    call = {
        "id": "external/id with arbitrary length",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps(ARGS, ensure_ascii=False)},
    }
    messages = template_messages(
        [
            {"role": "user", "content": "Synthetic query"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "lookup", "content": 'quote " slash \\ newline\n雪'},
        ],
        parser="mistral-json-v1",
    )
    rendered = _render("mistral_v03.jinja", messages, template_tools(TOOLS, parser="mistral-json-v1"))
    assert "[AVAILABLE_TOOLS] [" in rendered
    assert "Synthetic query" in rendered
    assert 'quote \\" slash \\\\ newline\\n雪' in rendered
    calls = re.findall(r"\[TOOL_CALLS\] \[(.*?)\]</s>", rendered)
    assert len(calls) == 1
    parsed = json.loads("[" + calls[0] + "]")
    assert parsed[0]["name"] == "lookup"
    assert parsed[0]["arguments"] == ARGS
    call_id = parsed[0]["id"]
    assert re.fullmatch(r"[A-Za-z0-9]{9}", call_id)
    assert f'"call_id": "{call_id}"' in rendered
    # Rendered definition JSON must survive descriptions with quotes/newlines.
    definition_text = rendered.split("[AVAILABLE_TOOLS] [", 1)[1].split("[/AVAILABLE_TOOLS]", 1)[0]
    definition = json.loads("[" + definition_text.rstrip("]") + "]")
    assert definition[0]["function"]["description"] == TOOLS[0]["function"]["description"]


def test_mistral_parser_accepts_native_calls_and_assigns_host_ids():
    raw = json.dumps([{"name": "lookup", "arguments": ARGS, "id": "Abc123xyz"}], ensure_ascii=False)
    message, finish = parse_tool_output(
        "[TOOL_CALLS] " + raw,
        TOOLS,
        parser="mistral-json-v1",
        finish_reason="eos",
        call_id_prefix="unit",
    )
    assert finish == "tool_calls"
    assert message["content"] is None
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == ARGS
    assert call["id"] != "Abc123xyz"


@pytest.mark.parametrize(
    "text",
    [
        '[TOOL_CALLS] [{"name":"lookup","arguments":{},"id":"short"}]',
        '[TOOL_CALLS] [{"name":"lookup","arguments":{},"id":"Abc123xy_"}]',
        '[TOOL_CALLS] [{"name":"lookup","arguments":{},"id":"Abc123xyz","id":"Zbc123xyz"}]',
        '[TOOL_CALLS] [{"name":"missing","arguments":{}}]',
        '[TOOL_CALLS] [{"name":"lookup","arguments":{},"extra":1}]',
        '[TOOL_CALLS] [{"name":"lookup","arguments":{}}] trailing',
        '[TOOL_CALLS] [{"name":"lookup","arguments":{',
        '[TOOL_CALLS] [{"name":"lookup","arguments":{},"id":"Abc123xyz"},'
        '{"name":"lookup","arguments":{},"id":"Abc123xyz"}]',
    ],
)
def test_mistral_parser_fails_closed(text):
    with pytest.raises(APIError):
        parse_tool_output(text, TOOLS, parser="mistral-json-v1", finish_reason="eos", call_id_prefix="unit")


@pytest.mark.parametrize("fixture", ["llama31.jinja", "llama32.jinja"])
def test_llama_render_is_plain_json_and_preserves_scalar_arguments(fixture):
    call = {
        "id": "public-call-id",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps(ARGS, ensure_ascii=False)},
    }
    messages = template_messages(
        [
            {"role": "user", "content": "Synthetic query"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "lookup", "content": "synthetic result"},
        ],
        parser="llama-json-v1",
    )
    rendered = _render(fixture, messages, template_tools(TOOLS, parser="llama-json-v1"))
    assert "Given the following functions" in rendered
    assert '"name": "lookup"' in rendered
    assert "<|start_header_id|>ipython" in rendered
    assert "synthetic result" in rendered
    assert "public-call-id" not in rendered
    assert "<|python_tag|>" not in rendered
    match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(\{"name".*?\})<\|eot_id\|>', rendered)
    # The assistant history JSON is present and its arguments retain every JSON type.
    assert match
    assert json.loads(match.group(1))["parameters"] == ARGS


def test_llama_history_rejects_multiple_calls_instead_of_splitting():
    one = {"id": "one", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    two = {"id": "two", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    with pytest.raises(APIError) as error:
        template_messages([{"role": "assistant", "content": None, "tool_calls": [one, two]}], parser="llama-json-v1")
    assert error.value.code == "tool_history_unsupported"


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        ("llama-json-v1", '{"name":"lookup","parameters":' + json.dumps(ARGS) + "}"),
        ("llama-json-v1", '<|python_tag|>{"name":"lookup","parameters":' + json.dumps(ARGS) + "}"),
        ("mistral-json-v1", "[TOOL_CALLS] " + json.dumps([{"name": "lookup", "arguments": ARGS}])),
    ],
)
def test_json_protocol_parser_returns_one_valid_call(parser, text):
    message, reason = parse_tool_output(
        text,
        TOOLS,
        parser=parser,
        finish_reason="eos",
        call_id_prefix="unit",
        parallel_tool_calls=False,
    )
    assert reason == "tool_calls"
    assert len(message["tool_calls"]) == 1
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == ARGS


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        ("llama-json-v1", '{"name":"lookup","parameters":{},"parameters":{}}'),
        ("llama-json-v1", '{"name":"unknown","parameters":{}}'),
        ("llama-json-v1", '{"name":"lookup","parameters":[]}'),
        ("llama-json-v1", '{"name":"lookup","parameters":{}} trailing'),
        ("llama-json-v1", "<|python_tag|>lookup(x=1)"),
        ("mistral-json-v1", '[TOOL_CALLS] [{"name":"lookup","arguments":{},"extra":1}]'),
        ("mistral-json-v1", '[TOOL_CALLS] [{"name":"missing","arguments":{}}]'),
        ("mistral-json-v1", '[TOOL_CALLS] [{"name":"lookup","arguments":{},"arguments":{}}]'),
    ],
)
def test_json_protocol_parser_rejects_bad_shapes_duplicate_keys_and_trailing_data(parser, text):
    with pytest.raises(APIError):
        parse_tool_output(text, TOOLS, parser=parser, finish_reason="eos", call_id_prefix="unit")


@pytest.mark.parametrize("parser,marker", [("mistral-json-v1", "[TOOL_CALLS]"), ("llama-json-v1", "<|python_tag|>")])
def test_json_tool_stream_withholds_possible_call_for_every_marker_split(parser, marker):
    payload = marker + '{"name":"lookup","arguments":{}}'
    for index in range(len(payload) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append, parser=parser)
        stream.feed(payload[:index])
        stream.feed(payload[index:])
        assert delivered == []


def test_json_tool_stream_passes_plain_text_with_every_marker_split():
    text = "Ordinary answer mentioning [TOOL_CALLS] as text only."
    for index in range(len(text) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append, parser="mistral-json-v1")
        stream.feed(text[:index])
        stream.feed(text[index:])
        stream.finish(text)
        assert "".join(delivered) == text


def test_unsupported_llama_history_fails_before_queue_admission(desktop, model, monkeypatch):
    from test_desktop import running

    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["token_codec_config"]["prompt_template"] = (FIXTURES / "llama31.jinja").read_text()
    config_path.write_text(json.dumps(config))
    running(desktop, model)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unsupported history must fail before worker admission")

    monkeypatch.setattr(desktop.worker, "request", forbidden)
    calls = [
        {"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        for identifier in ("first", "second")
    ]
    payload = {
        "messages": [
            {"role": "user", "content": "Synthetic query"},
            {"role": "assistant", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "second", "content": "Synthetic result"},
            {"role": "tool", "tool_call_id": "first", "content": "Synthetic result"},
            {"role": "user", "content": "Continue"},
        ]
    }
    with pytest.raises(APIError) as error:
        desktop.api_complete(payload)
    assert error.value.code == "tool_history_unsupported"
    assert desktop.job is None and desktop.api_queue == []


@pytest.mark.parametrize("streaming", [False, True])
def test_mistral_desktop_buffered_and_streamed_round_trip_keeps_public_call_id_private(
    desktop, model, monkeypatch, streaming
):
    """Exercise the installed Desktop request path with the exact native fixture."""
    from test_desktop import running

    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["token_codec_config"]["prompt_template"] = (FIXTURES / "mistral_v03.jinja").read_text()
    config_path.write_text(json.dumps(config))
    running(desktop, model)
    desktop.worker.streaming = True
    replay_value = '"quoted"\\\n雪'
    first_native = "[TOOL_CALLS] " + json.dumps(
        [{"name": "lookup", "arguments": {"value": replay_value}, "id": "Abc123xyz"}], ensure_ascii=False
    )
    second_text = "Synthetic answer after the result."
    captured_prompts = []
    generation = 0

    def request(payload, **options):
        nonlocal generation
        if payload["operation"] == "tokenize":
            captured_prompts.append(payload["text"])
            return {"token_count": 24, "token_ids": [2] * 24}
        generation += 1
        reply_text = first_native if generation == 1 else second_text
        if on_delta := options.get("on_delta"):
            on_delta(reply_text)
        return {"text": reply_text, "input_tokens": 24, "output_tokens": 12, "finish_reason": "eos"}

    monkeypatch.setattr(desktop.worker, "request", request)
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": 'Description with "quotes" and \\slashes.',
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
    }
    request_payload = {
        "messages": [{"role": "user", "content": "Synthetic query"}],
        "tools": [tool],
        "stream": streaming,
    }
    deltas = []
    first = desktop.api_complete(request_payload, on_delta=deltas.append if streaming else None)
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    public_call = choice["message"]["tool_calls"][0]
    assert json.loads(public_call["function"]["arguments"]) == {"value": replay_value}
    assert deltas == []

    replay = {
        **request_payload,
        "stream": False,
        "messages": request_payload["messages"]
        + [choice["message"], {"role": "tool", "tool_call_id": public_call["id"], "content": 'result "quoted"\\\n雪'}],
    }
    second = desktop.api_complete(replay)
    assert second["choices"][0]["message"]["content"] == second_text
    assert public_call["id"] not in captured_prompts[-1]
    native_ids = re.findall(r'"id": "([A-Za-z0-9]{9})"', captured_prompts[-1])
    assert native_ids and native_ids[-1] == "s00000000"
    assert captured_prompts[-1].count('"call_id": "s00000000"') == 1
    assert desktop.job is None and desktop.api_queue == []
