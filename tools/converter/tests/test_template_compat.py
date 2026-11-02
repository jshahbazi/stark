"""Program-equivalence admission; synthetic conversations assert no model competence."""

from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from test_desktop import desktop as desktop
from test_desktop import executable as executable
from test_desktop import model as model
from test_desktop import (
    test_tool_http_stream_matches_buffered_calls_and_never_duplicates as assert_http_tool_roundtrip,
)
from test_desktop import tool_desktop as tool_desktop

from stark_convert import api_tools, template_compat
from stark_convert.api_tools import parse_tool_output, template_messages, tool_parser_for
from stark_convert.local_api import APIError
from stark_convert.template_compat import chat_template_environment, template_program_sha256

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja"


@pytest.fixture
def source():
    return FIXTURE.read_text()


def harmless_variants(source):
    return [
        "{# Synthetic maintenance comment. #}" + source,
        "{# Synthetic maintenance comment.\nSecond comment line. #}\n" + source,
        source.replace("if tools", "if ( tools )", 1),
        source.replace("messages[0].role == 'system'", 'messages [ 0 ] . role == "system"', 1),
        source.replace("'<|im_start|>system\\n'", "'<|im_start|>system\\x0a'", 1),
        source.replace("    {%-", "        {%-"),
        source.replace("{%- if tools %}", "{%+ if tools %}", 1),
        source.replace("\n", "\r\n"),
    ]


def render_arguments(*, with_tools=True, history=True, thinking=False):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "system", "content": "Synthetic system turn."},
        {"role": "user", "content": "Synthetic first turn."},
    ]
    if history:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "synthetic_0",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"synthetic"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "synthetic_0", "content": "Synthetic tool result."},
                {"role": "user", "content": "Synthetic follow-up."},
            ]
        )
    return {
        "messages": template_messages(messages),
        "tools": tools if with_tools else None,
        "add_generation_prompt": True,
        "enable_thinking": thinking,
    }


@pytest.mark.parametrize(
    "with_tools,history,thinking", [(True, True, False), (True, False, True), (False, False, False)]
)
def test_real_template_harmless_edits_are_admitted_and_render_identically(source, with_tools, history, thinking):
    environment = chat_template_environment()
    arguments = render_arguments(with_tools=with_tools, history=history, thinking=thinking)
    expected = environment.from_string(source).render(**arguments)
    fingerprint = template_program_sha256(source)
    assert fingerprint is not None
    for variant in harmless_variants(source):
        assert variant != source
        assert template_program_sha256(variant) == fingerprint
        assert tool_parser_for({"prompt_template": variant}) == "qwen3-hermes-v1"
        assert environment.from_string(variant).render(**arguments) == expected


@pytest.mark.parametrize(
    "old,new",
    [
        ("<tool_call>", "<tool_invoke>"),
        ("if tools", "if not tools"),
        ("messages[::-1]", "messages[::1]"),
        ("tool | tojson", "tool | string"),
        ("tool_call.name", "tool_call.other"),
        ("tool_call.arguments | tojson", "tool_call.arguments"),
        ("message.content", "message.reasoning_content"),
        ("'<|im_start|>system\\n'", "'<|im_start|>system\\n\\n'"),
        ("{{- '<|im_start|>system\\n' }}", "{{ '<|im_start|>system\\n' }}"),
    ],
)
def test_real_template_semantic_edits_are_rejected(source, old, new):
    variant = source.replace(old, new, 1)
    assert variant != source
    assert template_program_sha256(variant) != template_program_sha256(source)
    assert tool_parser_for({"prompt_template": variant}) is None


def test_literal_trailing_output_is_not_treated_as_formatting(source):
    assert tool_parser_for({"prompt_template": source + " "}) is None
    assert template_program_sha256(source + " ") != template_program_sha256(source)


@pytest.mark.parametrize(
    "generated,finish_reason",
    [
        ('<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "eos"),
        ('<tool_call>{"name":"lookup","arguments":{"key":"a","key":"b"}}</tool_call>', "eos"),
        ('<tool_call>{"name":"lookup","arguments":{"key":"a"}}', "eos"),
        ('<tool_call>{"name":"lookup","arguments":{"key":"a"}}</tool_call>', "length"),
        ('```\n<tool_call>{"name":"lookup","arguments":{}}</tool_call>\n```', "eos"),
    ],
)
def test_harmless_template_admission_preserves_strict_generated_call_validation(source, generated, finish_reason):
    parser = tool_parser_for({"prompt_template": "{# maintenance comment #}" + source})
    assert parser == "qwen3-hermes-v1"
    tools = render_arguments()["tools"]
    with pytest.raises(APIError) as error:
        parse_tool_output(generated, tools, parser=parser, finish_reason=finish_reason, call_id_prefix="synthetic")
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize(
    "left,right",
    [
        ("prefix {{ x }} suffix", "pre{# note #}fix {{ x }} suf{# second note #}fix"),
        ("{{ 'same' }}", '{{ "same" }}'),
        ("{{ 'line\\n' }}", "{{ 'line\\x0a' }}"),
        ("{{ item }}", "{{ ( item ) }}"),
        ("{% if enabled %}yes{% endif %}", "{% if ( enabled ) %}yes{% endif %}"),
        ("a\n{# note #}\nb", "a\nb"),
    ],
)
def test_program_equivalence_discards_only_nonsemantic_syntax(left, right):
    assert template_program_sha256(left) is not None
    assert template_program_sha256(left) == template_program_sha256(right)


@pytest.mark.parametrize(
    "left,right",
    [
        ("{{ 1 }}", "{{ true }}"),
        ("{{ 1 }}", "{{ 1.0 }}"),
        ("{{ none }}", "{{ 'None' }}"),
        ("{{ first }}{{ second }}", "{{ second }}{{ first }}"),
        ("{{ item.name }}", "{{ item['name'] }}"),
        ("{{ item }}", "{{ item | string }}"),
        ("{{ item }}", "{{ other }}"),
        ("prefix {{ item }}", "prefix{{ item }}"),
        ("prefix {{ item }}", "prefix {{- item }}"),
        ("{{ 'one' }}{{ 'two' }}", "{{ 'onetwo' }}"),
    ],
)
def test_program_fingerprint_preserves_types_order_access_and_output(left, right):
    assert template_program_sha256(left) is not None
    assert template_program_sha256(right) is not None
    assert template_program_sha256(left) != template_program_sha256(right)


@pytest.mark.parametrize("parser", ["qwen3-hermes-v1", "qwen38-xml-v1", "qwen35-xml-v1"])
def test_equivalent_program_keeps_its_registered_parser_flavor(monkeypatch, parser):
    source = "{% if tools %}synthetic {{ tools }}{% endif %}"
    fingerprint = template_program_sha256(source)
    monkeypatch.setitem(api_tools._SUPPORTED_TEMPLATE_PROGRAMS, fingerprint, parser)
    assert tool_parser_for({"prompt_template": "{# harmless comment #}" + source}) == parser
    assert tool_parser_for({"prompt_template": source.replace("tools", "messages")}) is None


@pytest.mark.parametrize("value", [None, False, 5, [], {}, b"synthetic", "{{", "{% if tools %}", "\ud800"])
def test_invalid_templates_fail_closed(value):
    assert template_program_sha256(value) is None
    assert tool_parser_for({"prompt_template": value}) is None


def test_template_source_limit_is_utf8_bytes():
    assert template_program_sha256("é" * (template_compat.MAX_TEMPLATE_BYTES // 2 + 1)) is None
    assert tool_parser_for({"prompt_template": "x" * (template_compat.MAX_TEMPLATE_BYTES + 1)}) is None


def test_wide_and_deep_programs_fail_closed():
    wide = "{{ item }}" * (template_compat.MAX_TEMPLATE_NODES + 1)
    deep = "{% if enabled %}" * (template_compat.MAX_TEMPLATE_DEPTH + 1)
    deep += "synthetic" + "{% endif %}" * (template_compat.MAX_TEMPLATE_DEPTH + 1)
    assert template_program_sha256(wide) is None
    assert template_program_sha256(deep) is None
    assert tool_parser_for({"prompt_template": deep}) is None


def test_fingerprinting_never_compiles_or_executes_a_template(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Fingerprinting must only parse bounded template source.")

    monkeypatch.setattr(ImmutableSandboxedEnvironment, "compile", forbidden)
    monkeypatch.setattr(ImmutableSandboxedEnvironment, "from_string", forbidden)
    assert template_program_sha256("{{ raise_exception('synthetic, never executed') }}") is not None
    assert template_program_sha256("{{ range(999999999) | list }}") is not None


def test_admission_does_not_infer_support_from_names_or_wire_markers():
    assert (
        tool_parser_for(
            {
                "prompt_template": "<tool_call><function=lookup><parameter=key>synthetic",
                "model_type": "qwen3",
                "tool_parser": "qwen38-xml-v1",
            }
        )
        is None
    )


def test_shared_environment_retains_renderer_whitespace_and_loop_control():
    environment = chat_template_environment()
    assert isinstance(environment, ImmutableSandboxedEnvironment)
    assert environment.trim_blocks is True
    assert environment.lstrip_blocks is True
    assert environment.autoescape is False
    assert "jinja2.ext.LoopControlExtension" in environment.extensions
    assert environment.from_string("{% for item in [1,2] %}{{ item }}{% break %}{% endfor %}").render() == "1"


@pytest.mark.parametrize("tool_desktop", ["qwen3-hermes-v1"], indirect=True)
def test_harmless_template_runs_real_desktop_capabilities_history_and_http_stream(tool_desktop, monkeypatch, source):
    desktop, payload, seen, _ = tool_desktop
    variant = "{# harmless integration-test comment #}" + source.replace("if tools", "if ( tools )", 1)
    desktop.codec = {**desktop.codec, "prompt_template": variant}
    desktop.template = chat_template_environment().from_string(variant)
    # The worker fixture selects a synthetic parser. Restore production
    # admission so this exercises the loaded codec through the real API path.
    monkeypatch.setattr("stark_convert.desktop.tool_parser_for", api_tools.tool_parser_for)
    assert desktop.api_models()[0]["tool_calling"] is True
    capabilities = desktop.api_status()["capabilities"]
    assert capabilities["tool_calling"] is True
    assert capabilities["tool_parser"] == "qwen3-hermes-v1"
    response = desktop.api_complete(payload)
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    message = response["choices"][0]["message"]
    call = message["tool_calls"][0]
    assert call["function"] == {"name": "lookup", "arguments": '{"value":3}'}
    payload["messages"] += [
        message,
        {"role": "tool", "tool_call_id": call["id"], "content": "Synthetic integration result."},
    ]
    assert desktop.api_complete(payload)["choices"][0]["finish_reason"] == "tool_calls"
    rendered = seen[2]["text"]
    assert '<tool_call>\n{"name": "lookup", "arguments": {"value": 3}}\n</tool_call>' in rendered
    assert "<tool_response>\nSynthetic integration result.\n</tool_response>" in rendered
    assert_http_tool_roundtrip(tool_desktop)
    assert desktop.job is None and desktop.api_queue == []


def test_program_cache_is_bounded_and_oversize_sources_never_enter_it():
    cache = template_compat._program_sha256
    cache.cache_clear()
    try:
        for index in range(9):
            assert template_program_sha256("{{ value_" + str(index) + " }}") is not None
        before = cache.cache_info()
        assert before.maxsize == 8 and before.currsize == 8
        assert template_program_sha256("x" * (template_compat.MAX_TEMPLATE_BYTES + 1)) is None
        assert template_program_sha256("é" * (template_compat.MAX_TEMPLATE_BYTES // 2 + 1)) is None
        assert cache.cache_info() == before
    finally:
        cache.cache_clear()
