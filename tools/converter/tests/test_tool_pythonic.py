import warnings

import pytest

from stark_convert.tool_pythonic import (
    MAX_ARGUMENT_BYTES,
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_CALLS,
    MAX_PAYLOAD_BYTES,
    parse_pythonic_payload,
)


class Rejected(Exception):
    def __init__(self, category):
        self.category = category


def parse(payload):
    return parse_pythonic_payload(payload, invalid=Rejected)


def test_native_lfm_calls_scalars_and_json_composites():
    assert parse(
        "[get_weather(city='New York', exact=true, count=2, ratio=-1.25e2, "
        'opts={"units": "C"}, tags=["a", null]), clock()]'
    ) == [
        (
            "get_weather",
            {
                "city": "New York",
                "exact": True,
                "count": 2,
                "ratio": -125.0,
                "opts": {"units": "C"},
                "tags": ["a", None],
            },
        ),
        ("clock", {}),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "[f(x=1, x=2)]",
        "[f(**x)]",
        "[f(*x)]",
        "[f(1)]",
        "[f(x=y)]",
        "[f(x=obj.attr)]",
        "[f(x=call())]",
        "[f(x=(1,))]",
        "[f(x={1, 2})]",
        "[f(x=b'hi')]",
        "[f(x=f'{x}')]",
        "[f(x='a' 'b')]",
        "[f(x=r'hi')]",
        "[f(x=0x10)]",
        "[f(x=1_000)]",
        "[f(x=NaN)]",
        "[f(x=Infinity)]",
        "[f(x=1j)]",
        '[f(x={"a": 1, "a": 2})]',
        "[f(x={**y})]",
        "[f(class=1)]",
        "[f(x=1)] # comment",
        "[f(x=...)]",
        "[f(x=[z for z in y])]",
        "[f(x=1)] trailing",
        "f(x=1)",
        "[]",
        "[f(1), g(y=2)]",
    ],
)
def test_rejects_noncanonical_or_executable_python(source):
    with pytest.raises(Rejected):
        parse(source)


@pytest.mark.parametrize("source", ["[f(x=1e999)]", "[f(x=-1e999)]", "[f(x=1e-9999)]"])
def test_rejects_nonfinite_or_underflowed_float(source):
    with pytest.raises(Rejected):
        parse(source)


def test_duplicate_keyword_and_duplicate_json_keys_rejected():
    for source in ("[f(x=1, x=2)]", '[f(x={"k": 1, "k": 2})]'):
        with pytest.raises(Rejected):
            parse(source)


def test_call_and_payload_limits():
    with pytest.raises(Rejected) as err:
        parse("[" + ",".join("f()" for _ in range(MAX_CALLS + 1)) + "]")
    assert err.value.category == "format"
    with pytest.raises(Rejected) as err:
        parse("[f(x='" + "a" * MAX_PAYLOAD_BYTES + "')]")
    assert err.value.category == "limits"


def test_argument_limit():
    with pytest.raises(Rejected) as err:
        parse("[f(x='" + "a" * MAX_ARGUMENT_BYTES + "')]")
    assert err.value.category == "limits"


def test_ast_node_and_depth_limits():
    with pytest.raises(Rejected) as err:
        parse("[f(x=[" + ",".join("0" for _ in range(MAX_AST_NODES)) + "])]")
    assert err.value.category == "limits"
    with pytest.raises(Rejected) as err:
        parse("[f(x=" + "[" * (MAX_AST_DEPTH + 2) + "0" + "]" * (MAX_AST_DEPTH + 2) + ")]")
    assert err.value.category == "limits"


def test_payload_is_data_and_never_executes():
    marker = "stark_tool_pythonic_parser_marker"
    source = f"[f(x=__import__('pathlib').Path('/tmp/{marker}').touch())]"
    with pytest.raises(Rejected):
        parse(source)
    from pathlib import Path

    assert not Path("/tmp/" + marker).exists()


def test_identifier_length_is_bounded():
    assert parse("[" + "a" * 64 + "()]")[0][0] == "a" * 64
    with pytest.raises(Rejected):
        parse("[" + "a" * 65 + "()]")


def test_parse_warnings_are_suppressed_and_rejected():
    payload = "[f(x='\\qSECRET_MODEL_TEXT')]"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Rejected):
            parse(payload)
    assert caught == []


def test_returned_value_depth_and_node_bounds():
    nested = "[f(x=" + "[" * (MAX_AST_DEPTH + 1) + "0" + "]" * (MAX_AST_DEPTH + 1) + ")]"
    with pytest.raises(Rejected):
        parse(nested)
