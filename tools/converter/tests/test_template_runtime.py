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


@pytest.mark.parametrize("source", ["{{ 'x' * 1000000000 }}", "{{ 2 ** 1000000000 }}"])
def test_compile_time_constant_expansion_is_guarded(source):
    started = time.monotonic()
    template = IsolatedTemplate(source)
    with pytest.raises(TemplateRuntimeError):
        template.render()
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


@pytest.mark.parametrize("route", ["capture", "concat", "list", "tuple", "dict", "call", "method", "filter"])
def test_aggregate_expansion_is_charged_before_eager_arguments_accumulate(route):
    # Every expansion is only 256 bytes; a deliberately small private budget
    # exercises the production guard without making a large allocation.
    values = ["p * 16"] * 64
    if route == "capture":
        source = "{% set captured %}" + "".join("{{ " + value + " }}" for value in values) + "{% endset %}ok"
    elif route == "concat":
        source = "{{ " + " ~ ".join(values) + " }}"
    elif route in {"list", "tuple"}:
        left, right = ("[", "]") if route == "list" else ("(", ")")
        source = "{% set captured = " + left + ",".join(values) + right + " %}ok"
    elif route == "dict":
        source = "{% set captured = {" + ",".join(f"'k{index}': {value}" for index, value in enumerate(values))
        source += "} %}ok"
    else:
        if route == "method":
            values = ["p.upper()"] * 64
        elif route == "filter":
            values = ["p | upper"] * 64
        source = "{% set captured = dict(" + ",".join(f"k{index}={value}" for index, value in enumerate(values))
        source += ") %}ok"
    budget = template_runtime._ExpansionBudget(16 * 1024)
    template = template_runtime._child_environment(expansion_budget=budget)(source)
    assert budget.used == 0
    with pytest.raises(TemplateRuntimeError):
        template.render(p="x" * (256 if route in {"method", "filter"} else 16))
    assert budget.used <= budget.limit


def test_output_constant_folding_cannot_execute_filters_while_compiling():
    budget = template_runtime._ExpansionBudget(16 * 1024)
    template = template_runtime._child_environment(expansion_budget=budget)("{{ 'x' | replace('x', 'synthetic') }}")
    assert budget.used == 0
    assert template.render() == "synthetic"
    assert budget.used > 0


@pytest.mark.parametrize("route", ["list", "dict", "call"])
@pytest.mark.parametrize("value", ["x" * 256, ["x"] * 256, tuple(["x"] * 256)])
def test_slices_charge_the_shared_budget_before_copying_into_eager_containers(route, value):
    values = ["p[1:]"] * 64
    if route == "list":
        source = "{% set captured = [" + ",".join(values) + "] %}ok"
    elif route == "dict":
        source = "{% set captured = {" + ",".join(f"'k{index}': {item}" for index, item in enumerate(values))
        source += "} %}ok"
    else:
        source = "{% set captured = dict(" + ",".join(f"k{index}={item}" for index, item in enumerate(values))
        source += ") %}ok"
    budget = template_runtime._ExpansionBudget(16 * 1024)
    template = template_runtime._child_environment(expansion_budget=budget)(source)
    with pytest.raises(TemplateRuntimeError):
        template.render(p=value)
    assert budget.used <= budget.limit


def test_bounded_slicing_preserves_indexes_steps_and_empty_slices():
    source = "{{ p[1:] }};{{ p[::-1] }};{{ p[::2] }};{{ p[5:1] }};{{ messages[1:] | join(',') }};{{ p[0] }}"
    arguments = {"p": "abcdef", "messages": ["a", "b", "c"]}
    expected = chat_template_environment().from_string(source).render(**arguments)
    assert IsolatedTemplate(source).render(**arguments) == expected


@pytest.mark.parametrize("kind", ["list", "tuple", "dict"])
def test_loop_retained_literal_containers_use_the_shared_allocation_budget(kind):
    if kind == "dict":
        value = "{" + ",".join(f"'k{index}': ns.value" for index in range(8)) + "}"
    else:
        left, right = ("[", "]") if kind == "list" else ("(", ")")
        value = left + ",".join(["ns.value"] * 8) + right
    source = "{% set ns=namespace(value=none) %}{% for n in range(100) %}{% set ns.value="
    source += value + " %}{% endfor %}ok"
    budget = template_runtime._ExpansionBudget(800)
    template = template_runtime._child_environment(expansion_budget=budget)(source)
    with pytest.raises(TemplateRuntimeError):
        template.render()
    assert budget.used <= budget.limit


def test_bounded_literal_containers_preserve_empty_values_and_tuple_assignment():
    source = (
        "{% set empty=[] %}{% set record={'a':1, 'b':2} %}{% set pair=('x', 'y') %}"
        "{% for key,value in record.items() %}{{ key }}={{ value }};{% endfor %}"
        "{{ empty | length }};{{ pair | join(',') }};{{ {} | length }};{{ () | length }}"
    )
    expected = chat_template_environment().from_string(source).render()
    assert IsolatedTemplate(source).render() == expected


def test_busy_template_cpu_limit_and_cleanup(monkeypatch):
    import resource
    import signal

    template = IsolatedTemplate(BUSY)
    children = []
    popen = template_runtime.Popen

    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(template_runtime, "Popen", capture)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with pytest.raises(TemplateRuntimeError):
        template.render()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    assert cpu_seconds >= template_runtime.CPU_SECONDS - 0.5
    assert time.monotonic() - started < template_runtime.WALL_SECONDS + 1
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].returncode in {-signal.SIGKILL, -signal.SIGXCPU}
    assert children[0].stdin.closed and children[0].stdout.closed


def test_wall_deadline_and_cancellation_reap_children(monkeypatch):
    template = IsolatedTemplate(BUSY)
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
