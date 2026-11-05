"""Synthetic resource-boundary regressions; no model or private content is used."""

import json
import threading
import time
from pathlib import Path

import pytest
from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from stark_convert import template_runtime
from stark_convert.chat import ChatCancelled, ChatError, load_template, render_template
from stark_convert.template_compat import chat_template_environment
from stark_convert.template_runtime import IsolatedTemplate, TemplateCancelled, TemplateRuntimeError

BUSY = "{% for a in range(16000) %}{% for b in range(16000) %}{% if a < 0 %}x{% endif %}{% endfor %}{% endfor %}"


def test_compilation_and_rendering_never_execute_in_the_host(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Model programs must not compile in the backend.")

    monkeypatch.setattr(ImmutableSandboxedEnvironment, "compile", forbidden)
    monkeypatch.setattr(ImmutableSandboxedEnvironment, "from_string", forbidden)
    template = IsolatedTemplate("{% for item in messages %}{{ item }}{% endfor %}")
    assert template.render(messages=["synthetic", " control"]) == "synthetic control"


def test_template_evaluation_does_not_launch_or_require_a_python_executable(monkeypatch):
    import sys

    commands = []
    popen = template_runtime.Popen

    def capture(command, **kwargs):
        commands.append(command)
        return popen(command, **kwargs)

    monkeypatch.setattr(sys, "executable", "/nonexistent/template-python")
    monkeypatch.setattr(template_runtime, "Popen", capture)
    assert IsolatedTemplate("{{ value }}").render(value="synthetic control") == "synthetic control"
    assert len(commands) == 2
    assert all(len(command) == 1 and Path(command[0]).name == "stark-template" for command in commands)


def test_missing_native_helper_has_an_actionable_chat_error(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"token_codec_config": {"type": "ChatCodecConfig", "prompt_template": "{{ value }}"}})
    )

    def missing():
        raise template_runtime.TemplateHelperMissing(
            "Native chat-template helper is missing; rebuild or reinstall Stark."
        )

    monkeypatch.setattr(template_runtime, "_template_executable", missing)
    with pytest.raises(ChatError, match="helper is missing; rebuild or reinstall"):
        load_template(tmp_path)


@pytest.mark.parametrize("source", ["{{ 'x' * 1000000000 }}", "{{ 2 ** 1000000000 }}"])
def test_compile_time_constant_expansion_is_guarded(source):
    started = time.monotonic()
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(source).render()
    assert time.monotonic() - started < template_runtime.WALL_SECONDS + 1


@pytest.mark.parametrize(
    "source",
    [
        "{{ messages[0] * 1000000000 }}",
        "{{ [messages[0]] * 1000000000 }}",
        "{{ [messages[0]] * 16000 | join }}",
        "{{ messages[0] | replace('x', messages[0]) }}",
        "{{ messages[0] | tojson(indent=1000000000) }}",
        "{{ '%1000000000s' % messages[0] }}",
        "{{ '{:1000000000}'.format(messages[0]) }}",
        "{{ messages[0].ljust(1000000000) }}",
        "{{ lipsum(1000000000) }}",
        "{% set ns=namespace(x=['x']) %}{% for n in range(64) %}"
        "{% set ns.x=[ns.x, ns.x] %}{% endfor %}{{ ns.x | string }}",
        "{% set ns=namespace(x=['x']) %}{% for n in range(64) %}"
        "{% set ns.x=[ns.x, ns.x] %}{% endfor %}{{ ns.x | striptags }}",
    ],
)
def test_unbounded_primitives_fail_with_fixed_errors_and_leave_control_healthy(source):
    template = IsolatedTemplate(source)
    with pytest.raises(TemplateRuntimeError) as error:
        template.render(messages=["x" * 1025])
    assert str(error.value) == template_runtime.ERROR
    assert source not in str(error.value)
    assert IsolatedTemplate("{{ messages[0] }}").render(messages=["healthy control"]) == "healthy control"


@pytest.mark.parametrize(
    "source",
    [
        "{% set captured %}{% for n in range(16000) %}{{ messages[0] }}{% endfor %}{% endset %}ok",
        "{% macro captured() %}{% for n in range(16000) %}{{ messages[0] }}{% endfor %}{% endmacro %}{{ captured() }}",
        "{% for n in range(16000) %}{{ messages[0] }}{% endfor %}",
    ],
)
def test_all_output_buffers_are_capped(source):
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(source).render(messages=["x" * 1024])


@pytest.mark.parametrize("route", ["capture", "concat", "list", "dict", "call", "method", "filter"])
def test_aggregate_expansion_is_bounded_in_the_native_child(route):
    # Individual results stay below the output cap; retained intermediates
    # exceed the native allocator's cumulative limit before they can escape.
    values = ["p * 16"] * 512
    if route == "capture":
        source = "{% set captured %}" + "".join("{{ " + value + " }}" for value in values) + "{% endset %}ok"
    elif route == "concat":
        source = "{{ " + " ~ ".join(values) + " }}"
    elif route == "list":
        source = "{% set captured = [" + ",".join(values) + "] %}ok"
    elif route == "dict":
        source = "{% set captured = {" + ",".join(f"'k{index}': {value}" for index, value in enumerate(values))
        source += "} %}ok"
    else:
        if route == "method":
            values = ["p.upper()"] * 512
        elif route == "filter":
            values = ["p | upper"] * 512
        source = "{% set captured = dict(" + ",".join(f"k{index}={value}" for index, value in enumerate(values))
        source += ") %}ok"
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(source).render(p="x" * (65536 if route in {"method", "filter"} else 4096))
    assert IsolatedTemplate("{{ value }}").render(value="healthy control") == "healthy control"


@pytest.mark.parametrize("route", ["list", "dict", "call"])
def test_retained_slice_copies_are_bounded_in_the_native_child(route):
    values = ["p[1:]"] * 512
    if route == "list":
        source = "{% set captured = [" + ",".join(values) + "] %}ok"
    elif route == "dict":
        source = "{% set captured = {" + ",".join(f"'k{index}': {item}" for index, item in enumerate(values))
        source += "} %}ok"
    else:
        source = "{% set captured = dict(" + ",".join(f"k{index}={item}" for index, item in enumerate(values))
        source += ") %}ok"
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(source).render(p="x" * 65536)


def test_bounded_slicing_preserves_indexes_steps_and_empty_slices():
    source = "{{ p[1:] }};{{ p[::-1] }};{{ p[::2] }};{{ p[5:1] }};{{ messages[1:] | join(',') }};{{ p[0] }}"
    arguments = {"p": "abcdef", "messages": ["a", "b", "c"]}
    expected = chat_template_environment().from_string(source).render(**arguments)
    assert IsolatedTemplate(source).render(**arguments) == expected


def test_bounded_literal_containers_preserve_empty_values_and_unpacking():
    source = (
        "{% set empty=[] %}{% set record={'a':1, 'b':2} %}{% set pair=['x', 'y'] %}{% set left,right=pair %}"
        "{% for key,value in record.items() %}{{ key }}={{ value }};{% endfor %}"
        "{{ empty | length }};{{ pair | join(',') }};{{ left }}{{ right }};{{ {} | length }}"
    )
    expected = chat_template_environment().from_string(source).render()
    assert IsolatedTemplate(source).render() == expected


@pytest.mark.parametrize("ensure_ascii", [True, False])
@pytest.mark.parametrize("indent", [None, 2])
def test_explicit_json_ascii_policy_matches_python_json_with_html_escaping(ensure_ascii, indent):
    value = {"é": ["é😀<&>'\u0000\n\r\t\b\f\u001f\u007f\u0085\u2028"], "a": 1.0}
    options = f"ensure_ascii={'true' if ensure_ascii else 'false'}"
    if indent is not None:
        options += f", indent={indent}"
    source = "{{ value | tojson(" + options + ") }}"
    expected = json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, indent=indent)
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027")):
        expected = expected.replace(character, escaped)
    assert IsolatedTemplate(source).render(value=value) == expected


@pytest.mark.parametrize("options", ["ensure_ascii=none", "ensure_ascii=1", "ensure_ascii='false'", "unknown=true"])
def test_json_options_reject_unsupported_values_and_keywords(options):
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate("{{ value | tojson(" + options + ") }}").render(value="synthetic")


@pytest.mark.parametrize(
    "source",
    [
        "{{ _args }}{% set _args=1 %}",
        "{% if false %}{% set _args=1 %}{% endif %}{{ _args }}",
        "{% for _args in [] %}{% endfor %}{{ _args }}",
    ],
)
def test_private_root_arguments_do_not_supply_unbound_locals(source):
    assert IsolatedTemplate(source).render(_args="caller-private") == ""


def test_unused_private_root_keys_and_nested_json_keep_their_public_behavior():
    assert IsolatedTemplate("{{ value }}").render(value="public", _unused="ignored") == "public"
    source = "{% set _args=value %}{{ _args['_private'] }}"
    assert IsolatedTemplate(source).render(value={"_private": "nested data"}, _args="caller-private") == "nested data"


@pytest.mark.parametrize(
    "source",
    ["{{ ('a','b') }}", "{% set pair=('a','b') %}{{ pair | join(',') }}", "{% set pair='a','b' %}{{ pair }}"],
)
def test_tuple_literals_fail_closed_instead_of_changing_python_representation(source):
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(source).render()


@pytest.mark.parametrize("role", ["user", "assistant", "unsupported"])
def test_constant_string_membership_tuples_match_jinja(role):
    source = "{% if role not in ('user','assistant','system') %}unsupported{% else %}{{ role }}{% endif %}"
    expected = chat_template_environment().from_string(source).render(role=role)
    assert IsolatedTemplate(source).render(role=role) == expected


def test_busy_template_native_budget_and_cleanup(monkeypatch):
    template = IsolatedTemplate(BUSY)
    children = []
    popen = template_runtime.Popen

    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(template_runtime, "Popen", capture)
    started = time.monotonic()
    with pytest.raises(TemplateRuntimeError):
        template.render()
    assert time.monotonic() - started < template_runtime.WALL_SECONDS + 1
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_wall_deadline_and_cancellation_reap_children(monkeypatch):
    template = IsolatedTemplate("healthy control")
    children = []
    popen = template_runtime.Popen

    def stalled_child(_command, **kwargs):
        # Exercise the parent's deadline independently of native VM fuel.
        child = popen(["/bin/sleep", "30"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(template_runtime, "Popen", stalled_child)
    monkeypatch.setattr(template_runtime, "WALL_SECONDS", 0.1)
    with pytest.raises(TemplateRuntimeError):
        template.render()
    monkeypatch.setattr(template_runtime, "WALL_SECONDS", 5)
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    try:
        started = time.monotonic()
        with pytest.raises(TemplateCancelled):
            template.render_with_cancellation(cancelled)
        assert time.monotonic() - started < 1
    finally:
        timer.cancel()
        timer.join()
    assert len(children) == 2
    assert all(child.poll() is not None and child.stdin.closed and child.stdout.closed for child in children)


def test_source_input_and_ast_limits_fail_closed():
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate("é" * (template_runtime.MAX_SOURCE_BYTES // 2 + 1))
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate("{{ x }}" * 20000)
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate("{{ messages }}").render(messages=["x" * template_runtime.MAX_REQUEST_BYTES])
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate("{{ messages }}").render(messages=[object()])


def test_sandbox_errors_remain_content_free_security_errors():
    with pytest.raises(SecurityError) as error:
        IsolatedTemplate("{{ ''.__class__.__mro__ }}").render()
    assert str(error.value) == "The model's chat template violated the sandbox."


def test_loaded_template_has_the_same_boundary_and_cancellation(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"token_codec_config": {"type": "ChatCodecConfig", "prompt_template": BUSY}})
    )
    _, template = load_template(tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ChatCancelled):
        render_template(template, cancelled=cancelled, messages=[])
    with pytest.raises(ChatError, match="resource limits"):
        template.render(messages=[])


def test_tool_history_matches_the_existing_reviewed_template_renderer():
    source = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    arguments = {
        "messages": [
            {"role": "system", "content": "Synthetic system."},
            {"role": "user", "content": "Synthetic question."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "lookup", "arguments": {"key": "synthetic"}}}],
            },
            {"role": "tool", "content": "Synthetic result."},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    expected = chat_template_environment().from_string(source).render(**arguments)
    assert IsolatedTemplate(source).render(**arguments) == expected


@pytest.mark.parametrize(
    "profile", ["gemma-3", "lfm2.5-instruct", "qwen3", "qwen3-instruct", "qwen3-thinking", "qwen3.5"]
)
@pytest.mark.parametrize("generation_prompt", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_inherited_model_templates_match_jinja_output(profile, generation_prompt, thinking):
    fixture = json.loads((Path(__file__).parent / "fixtures/rendering_templates.json").read_text())
    source = fixture["templates"][profile]
    arguments = {
        "messages": [
            {"role": "system", "content": "Synthetic system."},
            {"role": "user", "content": "Synthetic évidence with literal <think> text."},
        ],
        "tools": None,
        "add_generation_prompt": generation_prompt,
        "enable_thinking": thinking,
        "bos_token": "synthetic-bos",
        "eos_token": "synthetic-eos",
    }
    expected = chat_template_environment().from_string(source).render(**arguments)
    assert IsolatedTemplate(source).render(**arguments) == expected


@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("parallel_results", [False, True])
def test_qwen35_tool_history_matches_jinja_output(followup, parallel_results):
    fixture = json.loads((Path(__file__).parent / "fixtures/rendering_templates.json").read_text())
    source = fixture["templates"]["qwen3.5"]
    calls = [{"function": {"name": "lookup", "arguments": {"key": "synthetic"}}}]
    if parallel_results:
        calls.append({"function": {"name": "lookup", "arguments": {"key": "second"}}})
    messages = [
        {"role": "system", "content": "Synthetic system."},
        {"role": "user", "content": "Synthetic question."},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "content": "Synthetic result."},
    ]
    if parallel_results:
        messages.append({"role": "tool", "content": "Synthetic second result."})
    if followup:
        messages.extend(
            [
                {"role": "assistant", "content": "Synthetic answer."},
                {"role": "user", "content": "Synthetic follow-up."},
            ]
        )
    arguments = {
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    expected = chat_template_environment().from_string(source).render(**arguments)
    rendered = IsolatedTemplate(source).render(**arguments)
    assert rendered == expected
    assert rendered.count("<|im_start|>user\n<tool_response>") == 1
    assert rendered.count("<tool_response>\nSynthetic") == len(calls)
    assert "</tool_response><|im_end|>" in rendered


@pytest.mark.parametrize("messages", [[], ["first"], ["first", "second", "third"]])
def test_adjacent_loop_items_match_jinja_output(messages):
    source = (
        "{% for item in messages %}"
        "{{ loop.previtem is defined }}:{{ loop.previtem }}|{{ item }}|"
        "{{ loop.nextitem is defined }}:{{ loop.nextitem }};"
        "{% endfor %}"
    )
    expected = chat_template_environment().from_string(source).render(messages=messages)
    assert IsolatedTemplate(source).render(messages=messages) == expected


def test_supported_methods_filters_loopcontrols_and_concat():
    source = (
        "{% set ns=namespace(content='') %}"
        "{% for item in messages %}{% if item.startswith('skip') %}{% continue %}{% endif %}"
        "{% set ns.content=ns.content ~ item.strip() %}{% if loop.index == 3 %}{% break %}{% endif %}"
        "{% endfor %}{{ ns.content }};{{ values.items() | list | length }};"
        "{{ messages | map('trim') | reject('equalto', 'skip') | join(',') }}"
    )
    arguments = {"messages": [" first ", "skip", " third ", "last"], "values": {"a": 1}}
    expected = chat_template_environment().from_string(source).render(**arguments)
    assert IsolatedTemplate(source).render(**arguments) == expected
