"""Native render and protocol round trips for reviewed trained tool formats."""

import io
import json
import sys
from pathlib import Path

import pytest

from stark_convert.api_tools import (
    ToolTextStream,
    parse_tool_output,
    template_messages,
    template_tools,
    tool_parser_for,
)
from stark_convert.chat import _ChatTemplate
from stark_convert.desktop import Desktop
from stark_convert.template_compat import normalize_chat_template
from stark_convert.tool_protocols import protocol_stop_ids

FIXTURES = Path(__file__).parent / "fixtures" / "tool_templates"
CASES = [
    ("granite30", "granite30-json-v1"),
    ("granite31", "granite33-json-v1"),
    ("granite33", "granite33-json-v1"),
    ("smollm2", "smollm2-json-v1"),
    ("smollm3", "smollm3-hermes-v1"),
    ("rnj1", "qwen3-hermes-v1"),
    ("phi4mini", "phi4-json-v1"),
    ("lfm2_1_2b", "lfm2-pythonic-v1"),
    ("lfm25_350m", "lfm2-pythonic-v1"),
    ("lfm25_1_2b", "lfm2-pythonic-v1"),
    ("functiongemma_mlx_chat_template", "functiongemma-v1"),
    ("functiongemma_unsloth_chat_template", "functiongemma-v1"),
]


def source_for(name):
    source = (FIXTURES / f"{name}.jinja").read_text()
    return normalize_chat_template(source)


def protocol_for(name, expected):
    parser = tool_parser_for({"prompt_template": source_for(name)})
    assert parser == expected
    return parser


def defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Synthetic lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text"},
                        "flag": {"type": "boolean", "description": "Flag"},
                        "count": {"type": "integer", "description": "Count"},
                        "nested": {
                            "type": "object",
                            "description": "Nested",
                            "properties": {"word": {"type": "string", "description": "Word"}},
                        },
                    },
                    "required": ["text", "flag", "count", "nested"],
                },
            },
        }
    ]


def call():
    args = {"text": "quotes ' \" slash \\ café\nnext", "flag": True, "count": 7, "nested": {"word": "雪"}}
    return {
        "id": "call_test_0",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps(args, ensure_ascii=False)},
    }


def test_reviewed_fixtures_are_admitted_by_exact_program():
    for name, expected in CASES:
        protocol_for(name, expected)


@pytest.mark.parametrize("name,expected", CASES)
def test_native_template_renders_definitions_call_result_history_and_followup(name, expected):
    parser = protocol_for(name, expected)
    tools = defs()
    messages = [
        {"role": "system", "content": "Synthetic system"},
        {"role": "user", "content": "Synthetic request"},
        {"role": "assistant", "content": None, "tool_calls": [call()]},
        {
            "role": "tool",
            "tool_call_id": "call_test_0",
            "name": "lookup",
            "content": {"found": True, "label": "café"} if expected == "functiongemma-v1" else "Synthetic result café",
        },
        {"role": "assistant", "content": "Synthetic follow-up"},
    ]
    adapted_tools = template_tools(tools, parser=parser)
    adapted = template_messages(messages, parser=parser, tools=tools)
    rendered = _ChatTemplate(source_for(name)).render(
        messages=adapted,
        tools=adapted_tools,
        add_generation_prompt=True,
        bos_token="<bos>",
        eos_token="<eos>",
        documents=[],
        controls=[],
        thinking=False,
    )
    assert "lookup" in rendered and "Synthetic request" in rendered and "Synthetic follow-up" in rendered
    assert "café" in rendered or "caf\\u00e9" in rendered
    assert '"text"' in rendered or "text=" in rendered or "text:" in rendered
    if parser == "smollm2-json-v1":
        assert "You have access to the following tools" in rendered
    if parser == "phi4-json-v1":
        assert "<|tool|>" in rendered and "tool_response" in rendered
    if parser == "functiongemma-v1":
        assert "<start_function_response>response:lookup{" in rendered


@pytest.mark.parametrize("name,parser", CASES)
def test_native_output_is_accepted_and_stream_hides_call_marker(name, parser):
    protocol_for(name, parser)
    # Exercise each parser through the same terminal API with a simple exact
    # native payload. FunctionGemma's end response marker is a tokenizer stop.
    payloads = {
        "granite30-json-v1": '<|tool_call|>[{"name":"lookup","arguments":{"value":1}}]',
        "granite33-json-v1": '<|tool_call|>[{"name":"lookup","arguments":{"value":1}}]',
        "smollm2-json-v1": '<tool_call>[{"name":"lookup","arguments":{"value":1}}]</tool_call>',
        "smollm3-hermes-v1": '<tool_call>{"name":"lookup","arguments":{"value":1}}</tool_call>',
        "qwen3-hermes-v1": '<tool_call>{"name":"lookup","arguments":{"value":1}}</tool_call>',
        "phi4-json-v1": '<|tool_call|>[{"name":"lookup","arguments":{"value":1}}]<|/tool_call|>',
        "lfm2-pythonic-v1": "<|tool_call_start|>[lookup(value=1)]<|tool_call_end|>",
    }
    if parser == "functiongemma-v1":
        payloads[parser] = "<start_function_call>call:lookup{value:1}<end_function_call>"
    text = payloads[parser]
    message, finish = parse_tool_output(
        text,
        [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        parser=parser,
        finish_reason="eos",
        call_id_prefix="test",
    )
    assert finish == "tool_calls" and message["tool_calls"][0]["function"]["name"] == "lookup"
    emitted = []
    stream = ToolTextStream(emitted.append, parser=parser)
    stream.feed(text[:5])
    stream.feed(text[5:])
    stream.finish(message["content"])
    assert "".join(emitted) == ""


class TokenizerWorker:
    def __init__(self):
        self.requests = []

    def request(self, payload, **_options):
        self.requests.append(payload)
        return {"token_count": 1, "token_ids": [902]}


def test_functiongemma_native_stop_id_is_added_only_for_tool_generation():
    worker = TokenizerWorker()
    assert protocol_stop_ids(worker, "functiongemma-v1", active=True) == [902]
    assert worker.requests == [
        {"operation": "tokenize", "text": "<start_function_response>", "add_special_tokens": False}
    ]
    before = len(worker.requests)
    assert protocol_stop_ids(worker, "functiongemma-v1", active=False) == []
    assert protocol_stop_ids(worker, "qwen3-hermes-v1", active=True) == []
    assert len(worker.requests) == before


def test_desktop_keeps_client_stops_and_adds_native_functiongemma_stop_id(tmp_path):
    executable = tmp_path / "synthetic-worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "print(json.dumps({'ready':True,'mode':'text-generation-v1','max_context_tokens':8192,"
        "'max_new_tokens':4096,'streaming':'text-delta-v1','stop_sequences':True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if request['operation']=='tokenize':\n"
        "        reply={'token_count':20,'token_ids':[2]*20}\n"
        "    else:\n"
        "        reply={'text':'synthetic answer','input_tokens':20,'output_tokens':2,'finish_reason':'eos'}\n"
        "    print(json.dumps(reply), flush=True)\n"
    )
    executable.chmod(0o755)
    model_dir = tmp_path / "synthetic-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"synthetic fixture")
    config = {
        "decoder_config": {},
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "prompt_template": (FIXTURES / "functiongemma_mlx_chat_template.jinja").read_text(),
            "default_system_prompt": "Synthetic system",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "tokenizer.json").write_text("{}")
    desktop = Desktop(executable, tmp_path / "library", io.StringIO(), timeout=2)

    def finish_job():
        thread = desktop.job
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive()

    try:
        desktop.dispatch({"command": "install", "kind": "local", "source": str(model_dir)})
        finish_job()
        assert desktop.models
        identifier = desktop.models[0]["id"]
        desktop.dispatch({"command": "run", "model_id": identifier})
        finish_job()
        requests = []

        def fake_request(payload, **_options):
            recorded = {"operation": payload["operation"]}
            if payload["operation"] == "tokenize":
                marker = payload.get("text") == "<start_function_response>"
                recorded["marker"] = marker
                requests.append(recorded)
                ids = [902] if marker else [2] * 20
                return {"token_count": len(ids), "token_ids": ids}
            recorded["stop"] = payload.get("stop")
            if "stop_token_ids" in payload:
                recorded["stop_token_ids"] = payload["stop_token_ids"]
            requests.append(recorded)
            return {"text": "synthetic answer", "input_tokens": 20, "output_tokens": 2, "finish_reason": "eos"}

        desktop.worker.request = fake_request
        stops = ["client-stop-a", "client-stop-b", "client-stop-c", "client-stop-d"]
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
            }
        ]
        desktop.api_complete(
            {"messages": [{"role": "user", "content": "Synthetic tool request"}], "tools": tools, "stop": stops}
        )
        desktop.api_complete({"messages": [{"role": "user", "content": "Synthetic ordinary request"}], "stop": stops})
    finally:
        desktop.stop(quiet=True)

    generations = [request for request in requests if request["operation"] == "generate"]
    assert [request["stop"] for request in generations] == [stops, stops]
    assert generations[0]["stop_token_ids"] == [902]
    assert "stop_token_ids" not in generations[1]
    # Tokenization records are intentionally reduced to marker membership; no
    # prompt text or token vectors are retained by the fixture.
    assert sum(request.get("marker") is True for request in requests if request["operation"] == "tokenize") == 1


def test_functiongemma_rejects_unrenderable_parameter_schemas_and_delimiters():
    from stark_convert.local_api import APIError
    from stark_convert.tool_protocols import validate_functiongemma_tools

    parser = "functiongemma-v1"
    bad_tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"properties": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "lookup", "parameters": {"$ref": "#/defs/x"}}},
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"untyped": {"description": "No type"}}},
            },
        },
    ]
    for tools in bad_tools:
        with pytest.raises(APIError):
            validate_functiongemma_tools([tools])
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": json.dumps({"text": "<escape>"})},
                }
            ],
        }
    ]
    with pytest.raises(APIError):
        template_messages(msgs, parser=parser, tools=defs())


@pytest.mark.parametrize("parser", ["granite33-json-v1", "qwen3-hermes-v1"])
def test_parallel_results_follow_call_order_by_call_id(parser):
    tools = defs()
    calls = [
        {"id": "first", "type": "function", "function": {"name": "lookup", "arguments": '{"count":1}'}},
        {"id": "second", "type": "function", "function": {"name": "lookup", "arguments": '{"count":2}'}},
    ]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": "second", "name": "lookup", "content": "result second"},
        {"role": "tool", "tool_call_id": "first", "name": "lookup", "content": "result first"},
    ]
    adapted = template_messages(messages, parser=parser, tools=tools)
    assert [item["content"] for item in adapted[1:]] == ["result first", "result second"]


def test_lfm_pythonic_preserves_trailing_prose_and_end_marker_inside_argument_string():
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    text = "<|tool_call_start|>[lookup(value='inside <|tool_call_end|> marker')]<|tool_call_end|> Follow-up text"
    message, finish = parse_tool_output(
        text,
        tools,
        parser="lfm2-pythonic-v1",
        finish_reason="eos",
        call_id_prefix="lfm",
    )
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "value": "inside <|tool_call_end|> marker"
    }
    assert message["content"] == " Follow-up text"
    for split in range(len(text) + 1):
        emitted = []
        stream = ToolTextStream(emitted.append, parser="lfm2-pythonic-v1")
        stream.feed(text[:split])
        stream.feed(text[split:])
        stream.finish(message["content"])
        assert "".join(emitted) == message["content"]
