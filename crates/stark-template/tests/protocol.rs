//! Native subprocess controls use synthetic inputs and retain no conversation data.

use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

use serde_json::{Value, json};

fn invoke(
    source: &str,
    arguments: Value,
) -> Output {
    bytes(&serde_json::to_vec(&json!({"source": source, "arguments": arguments})).unwrap())
}

fn bytes(data: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_stark-template"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child.stdin.take().unwrap().write_all(data);
    child.wait_with_output().unwrap()
}

fn success(
    source: &str,
    arguments: Value,
    expected: &[u8],
) {
    let result = invoke(source, arguments);
    assert!(result.status.success(), "native renderer rejected synthetic control");
    assert!(result.stderr.is_empty());
    assert_eq!(&result.stdout[..1], b"O");
    assert_eq!(&result.stdout[1..], expected);
}

fn rejected(
    source: &str,
    arguments: Value,
) {
    let result = invoke(source, arguments);
    assert!(!result.status.success(), "native renderer accepted hostile control");
    assert!(result.stderr.is_empty());
    assert!(matches!(result.stdout.as_slice(), b"E" | b"L" | b"S"));
    success("{{ value }}", json!({"value": "healthy control"}), b"healthy control");
}

#[test]
fn self_test_exercises_native_renderer_without_content() {
    let result = Command::new(env!("CARGO_BIN_EXE_stark-template")).arg("--self-test").output().unwrap();
    assert!(result.status.success());
    assert!(result.stderr.is_empty());
    assert_eq!(result.stdout, b"{\"status\":\"ok\",\"protocol\":\"stark-template-v1\"}\n");
}

#[test]
fn compile_only_and_empty_render_are_distinct() {
    success("{{ missing }}", Value::Null, b"");
    success("{{ missing }}", json!({}), b"");
    rejected("{{", Value::Null);
}

#[test]
fn python_method_unicode_indices_and_empty_count_match() {
    success(
        "{{ text.count('') }};{{ text.count('é') }};{{ text.find('b') }};{{ text.rfind('é') }};{{ text.find('missing') }}",
        json!({"text": "ébé"}),
        b"4;2;1;2;-1",
    );
    success("{{ 'a1'.islower() }};{{ 'A1'.isupper() }};{{ ''.isdigit() }}", json!({}), b"True;True;False");
    rejected("{{ text.isdigit() }}", json!({"text": "²"}));
    success("{{ text.strip() }}", json!({"text": "\u{1c}synthetic\u{1f}"}), b"synthetic");
}

#[test]
fn python_json_spacing_sorting_ascii_and_html_escaping_match() {
    success(
        "{{ value | tojson }}",
        json!({"value": {"z": [true, Value::Null, "é😀<&>'"], "a": 1.0}}),
        br#"{"a": 1.0, "z": [true, null, "\u00e9\ud83d\ude00\u003c\u0026\u003e\u0027"]}"#,
    );
    success(
        "{{ value | tojson(indent=2) }}",
        json!({"value": {"b": 2, "a": [1]}}),
        b"{\n  \"a\": [\n    1\n  ],\n  \"b\": 2\n}",
    );
    success("{{ value | tojson }}", json!({"value": [1e-5, 1e16, -0.0]}), b"[1e-05, 1e+16, -0.0]");
    rejected("{{ ('nan' | float) | tojson }}", json!({}));
    rejected("{{ [('inf' | float)] | tojson }}", json!({}));
    success(
        "{{ value | tojson(ensure_ascii=false) }}",
        json!({"value": {"é": "é😀<&>'\u{0}\n\u{85}"}}),
        "{\"é\": \"é😀\\u003c\\u0026\\u003e\\u0027\\u0000\\n\u{85}\"}".as_bytes(),
    );
    success(
        "{{ value | tojson(indent=2, ensure_ascii=false) }}",
        json!({"value": {"é": ["é"]}}),
        "{\n  \"é\": [\n    \"é\"\n  ]\n}".as_bytes(),
    );
    rejected("{{ value | tojson(ensure_ascii='false') }}", json!({"value": "synthetic"}));
    rejected("{{ value | tojson(ensure_ascii=false, unknown=true) }}", json!({"value": "synthetic"}));
}

#[test]
fn python_display_preserves_bool_none_and_container_types() {
    success(
        "{{ true }};{{ none }};{{ values }}",
        json!({"values": [false, Value::Null, "é"]}),
        "True;None;[False, None, 'é']".as_bytes(),
    );
    success("{{ values }}", json!({"values": {"b": 2, "a": 1}}), b"{'b': 2, 'a': 1}");
    success("{{ values }}", json!({"values": ["\u{200b}", "\u{ad}"]}), br#"['\u200b', '\xad']"#);
}

#[test]
fn methods_filters_namespaces_loopcontrols_and_slices_work() {
    success(
        "{% set ns=namespace(content='') %}{% for item in messages %}{% if item.startswith('skip') %}{% continue %}{% endif %}{% set ns.content=ns.content ~ item.strip() %}{% if loop.index == 3 %}{% break %}{% endif %}{% endfor %}{{ ns.content }};{{ values.items() | list | length }};{{ messages | map('trim') | reject('equalto', 'skip') | join(',') }}",
        json!({"messages": [" first ", "skip", " third ", "last"], "values": {"a": 1}}),
        b"firstthird;1;first,third,last",
    );
    success(
        "{{ text[::-1] }};{{ messages[1:] | join(',') }}",
        json!({"text": "abcdef", "messages": ["a", "b", "c"]}),
        b"fedcba;b,c",
    );
}

#[test]
fn adjacent_tool_results_keep_group_boundaries_across_completed_tool_cycles() {
    let source = "{% for message in messages %}{% if message.role == 'tool' %}{% if loop.previtem and loop.previtem.role != 'tool' %}[{% endif %}{{ message.content }}{% if not loop.last and loop.nextitem.role != 'tool' %}]{% elif loop.last %}]{% else %},{% endif %}{% endif %}{% endfor %}";
    success(
        source,
        json!({"messages": [{"role": "user"}, {"role": "assistant"}, {"role": "tool", "content": "first"}, {"role": "assistant"}, {"role": "tool", "content": "second"}, {"role": "assistant"}]}),
        b"[first][second]",
    );
    success(
        source,
        json!({"messages": [{"role": "user"}, {"role": "assistant"}, {"role": "tool", "content": "first"}, {"role": "tool", "content": "second"}, {"role": "assistant"}, {"role": "tool", "content": "third"}]}),
        b"[first,second][third]",
    );
}

#[test]
fn declared_single_underscore_temporaries_remain_bounded_data_only() {
    success(
        "{% set _args=arguments if arguments is mapping else dict() %}{{ _args | tojson }}",
        json!({"arguments": {"key": "synthetic"}}),
        br#"{"key": "synthetic"}"#,
    );
    success(
        "{% for key,_value in arguments.items() %}{{ key }}={{ _value }};{% endfor %}",
        json!({"arguments": {"key": "synthetic"}}),
        b"key=synthetic;",
    );
    success("{% set _args %}synthetic{% endset %}{{ _args }}", json!({}), b"synthetic");
    // String lookup remains a native string operation, with no Python object
    // or introspection attributes behind an ordinary data value.
    success("{% set _args='synthetic' %}{{ _args['__class__'] | default('unavailable') }}", json!({}), b"unavailable");
    for source in [
        "{{ _args }}",
        "{% set __args='synthetic' %}{{ __args }}",
        "{% set _args='synthetic' %}{{ _args.__class__ }}",
        "{% set _args='synthetic' %}{{ _args._private }}",
        "{% set _args=dict() %}{% set _private='synthetic' %}{{ _args._private }}",
        "{% set _args='synthetic' %}{{ _args | attr('__class__') }}",
        "{% set _args='synthetic' %}{{ _args.__getitem__('__class__') }}",
        "{% set ns=namespace() %}{% set ns._args='synthetic' %}",
    ] {
        rejected(source, json!({"_args": "synthetic"}));
    }
}

#[test]
fn private_root_arguments_cannot_supply_unbound_template_locals() {
    for source in [
        "{{ _args }}{% set _args=1 %}",
        "{% if false %}{% set _args=1 %}{% endif %}{{ _args }}",
        "{% for _args in [] %}{% endfor %}{{ _args }}",
    ] {
        success(source, json!({"_args": "caller-private"}), b"");
    }
    success("{{ value }}", json!({"value": "public", "_unused": "ignored"}), b"public");
    success(
        "{% set _args=value %}{{ _args['_private'] }}",
        json!({"value": {"_private": "nested data"}, "_args": "caller-private"}),
        b"nested data",
    );
}

#[test]
fn tuple_literals_fail_closed_while_calls_groups_and_unpacking_remain_supported() {
    for source in [
        "{{ ('a', 'b') }}",
        "{{ () }}",
        "{% set pair=('a', 'b') %}{{ pair | join(',') }}",
        "{% set pair='a','b' %}{{ pair }}",
        "{% set pair='a', %}{{ pair }}",
        "{{ dict(value=('a', 'b')) }}",
    ] {
        rejected(source, json!({}));
    }
    success("{{ dict(a=('x'), b=dict(c='y')['c']) | tojson }}", json!({}), br#"{"a": "x", "b": "y"}"#);
    success(
        "{% set value=dict(a='x', b=['y','z']) %}{{ value | tojson }}",
        json!({}),
        br#"{"a": "x", "b": ["y", "z"]}"#,
    );
    success("{% macro pair(a,b) %}{{ a }}{{ b }}{% endmacro %}{{ pair(('x'),dict(c='y')['c']) }}", json!({}), b"xy");
    success(
        "{% set a,b=['x','y'] %}{{ a }}{{ b }};{% for k,v in values.items() %}{{ k }}={{ v }}{% endfor %}",
        json!({"values": {"a": 1}}),
        b"xy;a=1",
    );
}

#[test]
fn constant_string_membership_tuples_have_only_boolean_outputs() {
    success("{% if role not in ('a','b','c') %}yes{% else %}no{% endif %}", json!({"role": "z"}), b"yes");
    success("{% if role not in ('a','b','c') %}yes{% else %}no{% endif %}", json!({"role": "b"}), b"no");
    success("{% set member=role in ('a','b',) %}{{ member }}", json!({"role": "b"}), b"True");
    success("{{ role in ('a','b') and role=='b' }}", json!({"role": "b"}), b"True");
    success("{{ (role in ('a','b')) }}", json!({"role": "z"}), b"False");
    for source in [
        "{% for role in ('a','b') %}{{ role }}{% endfor %}",
        "{{ role in ('a','b') | join }}",
        "{{ role in ('a','b') ~ '' }}",
        "{{ role in ('a','b')[0] }}",
        "{{ role in ('a','b').items() }}",
        "{{ role in ('a','b')('a') }}",
        "{{ role in ('a','b') == value }}",
        "{{ role in ('a','b') != value }}",
        "{{ role in ('a','b') < value }}",
        "{{ role in ('a','b') in value }}",
        "{{ role in ('a','b') not in value }}",
        "{{ role in ('a','b') is sequence }}",
        "{{ role in ('a','b') + value }}",
        "{{ role in ('a',value) }}",
        "{{ role in (value,'b') }}",
    ] {
        rejected(source, json!({"role": "a", "value": ["a", "b"]}));
    }
}

#[test]
fn qwen_fixture_decision_boundary_and_tool_history_remain_exact() {
    let fixture = include_str!("../../../tools/converter/tests/fixtures/qwen3_4b_chat_template.jinja");
    success(
        fixture,
        json!({"messages": [{"role": "system", "content": "Synthetic system."}, {"role": "user", "content": "Synthetic question."}], "tools": Value::Null, "add_generation_prompt": true, "enable_thinking": false}),
        b"<|im_start|>system\nSynthetic system.<|im_end|>\n<|im_start|>user\nSynthetic question.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    success(
        fixture,
        json!({"messages": [{"role": "user", "content": "Synthetic question."}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"key": "synthetic"}}}]}, {"role": "tool", "content": "Synthetic result."}], "tools": Value::Null, "add_generation_prompt": false, "enable_thinking": false}),
        b"<|im_start|>user\nSynthetic question.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"key\": \"synthetic\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nSynthetic result.\n</tool_response><|im_end|>\n",
    );
}

#[test]
fn forbidden_python_attributes_and_environment_features_fail_closed() {
    for source in [
        "{{ ''.__class__.__mro__ }}",
        "{{ lipsum(10) }}",
        "{{ debug() }}",
        "{% include 'file' %}",
        "{{ value | attr('path') }}",
        "{{ value.format() }}",
    ] {
        rejected(source, json!({"value": "synthetic"}));
    }
}

#[test]
fn huge_expansions_and_cumulative_intermediates_are_bounded() {
    for source in [
        "{{ 'x' * 1000000000 }}",
        "{{ 2 ** 1000000000 }}",
        "{{ ['x'] * 1000000000 }}",
        "{% set x = value | replace('x', value * 100) %}ok",
        "{{ value | tojson(indent=1000000000) }}",
        "{{ '%1000000000s' % value }}",
        "{% set ns=namespace(x=['x']) %}{% for n in range(64) %}{% set ns.x=[ns.x, ns.x] %}{% endfor %}{{ ns.x | string }}",
    ] {
        rejected(source, json!({"value": "x".repeat(1025)}));
    }
    // Each retained slice is permitted individually; the combined allocation
    // exceeds the process-wide cumulative budget before a final output exists.
    rejected(
        "{% set ns=namespace(x=[]) %}{% for n in range(1000) %}{% set ns.x=ns.x + [value[1:]] %}{% endfor %}ok",
        json!({"value": "x".repeat(32768)}),
    );
}

#[test]
fn fuel_limits_busy_templates_and_macro_recursion() {
    rejected(
        "{% for a in range(16000) %}{% for b in range(16000) %}{% if a < 0 %}x{% endif %}{% endfor %}{% endfor %}",
        json!({}),
    );
    rejected("{% macro recur() %}{{ recur() }}{% endmacro %}{{ recur() }}", json!({}));
}

#[test]
fn source_request_output_and_structure_limits_are_enforced() {
    rejected(&"x".repeat(256 * 1024 + 1), Value::Null);
    rejected(&"{{ value }}".repeat(20000), Value::Null);
    rejected("{{ values }}", json!({"values": vec![0; 16385]}));
    rejected("{{ value }}", json!({"value": "x".repeat(1024 * 1024 + 1)}));
    rejected("{% for n in range(2000) %}{{ value }}{% endfor %}", json!({"value": "x".repeat(1024)}));
    let result = bytes(&vec![b' '; 2 * 1024 * 1024 + 1]);
    assert_eq!(result.stdout, b"L");
    assert!(!result.status.success());
}

#[test]
fn unsupported_large_integer_inputs_do_not_silently_round() {
    for data in [
        br#"{"source":"{{ value }}","arguments":{"value":18446744073709551617}}"#.as_slice(),
        br#"{"source":"{{ value }}","arguments":{"value":-9223372036854775809}}"#.as_slice(),
    ] {
        let result = bytes(data);
        assert_eq!(result.stdout, b"E");
        assert!(!result.status.success());
    }
    success("{{ value }}", json!({"value": 18446744073709551615u64}), b"18446744073709551615");
}
