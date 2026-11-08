"""Pre-parse resource admission with synthetic JSON only."""

import json

import pytest

from stark_convert import json_budget
from stark_convert.json_budget import check_json_structure


def test_large_string_does_not_consume_structural_nodes():
    data = b'{"content":"' + b"x" * (16 * 1024 * 1024) + b'"}'
    check_json_structure(data)


@pytest.mark.parametrize("value", [None, True, False, 12, -1.25, "", "é"])
def test_scalar_json_is_admitted(value):
    check_json_structure(json.dumps(value, ensure_ascii=False).encode())


def test_depth_boundary_is_checked_before_parsing():
    check_json_structure(b"[" * json_budget.MAX_JSON_DEPTH + b"]" * json_budget.MAX_JSON_DEPTH)
    with pytest.raises(ValueError, match="^JSON structure exceeds the resource budget\\.$"):
        check_json_structure(b"[" * (json_budget.MAX_JSON_DEPTH + 1))


def test_compact_array_rejects_excess_nodes():
    allowed_items = json_budget.MAX_JSON_NODES - 1
    check_json_structure(b"[" + b"0," * (allowed_items - 1) + b"0]")
    with pytest.raises(ValueError):
        check_json_structure(b"[" + b"0," * allowed_items + b"0]")


def test_empty_containers_each_consume_one_node():
    with pytest.raises(ValueError):
        check_json_structure(b"[" + b"[]," * json_budget.MAX_JSON_NODES + b"[]]")


def test_object_keys_and_scalar_values_are_counted(monkeypatch):
    monkeypatch.setattr(json_budget, "MAX_JSON_NODES", 5)
    check_json_structure(b'{"first":null,"second":false}')
    with pytest.raises(ValueError):
        check_json_structure(b'{"first":null,"second":false,"third":true}')


def test_structural_characters_and_escaped_quotes_in_strings_are_ignored(monkeypatch):
    monkeypatch.setattr(json_budget, "MAX_JSON_NODES", 3)
    monkeypatch.setattr(json_budget, "MAX_JSON_DEPTH", 1)
    content = "é" + '[{,:}]"\\' * 10000
    data = json.dumps({"content": content}, ensure_ascii=False).encode()
    check_json_structure(data)
    assert json.loads(data)["content"] == content


def test_even_backslash_runs_allow_the_closing_quote_to_end_a_string(monkeypatch):
    monkeypatch.setattr(json_budget, "MAX_JSON_NODES", 2)
    check_json_structure(b'["\\\\"]')
    with pytest.raises(ValueError):
        check_json_structure(b'["\\\\",0]')


@pytest.mark.parametrize("data", [b'"unterminated', b'"bad\\q"', b'"control\x00"', b"[}", b"\xff"])
def test_syntax_and_encoding_errors_are_left_to_the_json_parser(data):
    check_json_structure(data)
    with pytest.raises((ValueError, UnicodeError)):
        json.loads(data)


def test_error_contains_no_input(monkeypatch):
    monkeypatch.setattr(json_budget, "MAX_JSON_DEPTH", 1)
    with pytest.raises(ValueError) as error:
        check_json_structure(b'{"private-value":[[]]}')
    assert str(error.value) == "JSON structure exceeds the resource budget."
