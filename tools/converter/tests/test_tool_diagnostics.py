"""Category-only records stay separate from authorized argument capture."""

import json
import stat
from datetime import datetime

import pytest
from test_api_tools import xml_call, xml_output, xml_property_tool

from stark_convert import api_tools, tool_diagnostics
from stark_convert.local_api import APIError


@pytest.mark.parametrize(
    "schema,body,category",
    [
        ({"type": "integer"}, "private_synthetic_argument", "xml_value_json"),
        ({"type": "integer"}, '"private_synthetic_argument"', "xml_value_type"),
        ({"type": ["string", "integer"]}, "1234567", "xml_value_ambiguous"),
    ],
)
def test_actual_parser_rejection_persists_only_timestamp_and_category(schema, body, category):
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": body}), tools=[xml_property_tool(schema)])
    assert error.value.code == "invalid_tool_call"
    path = tool_diagnostics._log_directory() / "tool-diagnostics.jsonl"
    record = json.loads(path.read_text())
    assert set(record) == {"timestamp", "category"}
    assert record["category"] == category
    assert datetime.fromisoformat(record["timestamp"]).utcoffset().total_seconds() == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert body not in path.read_text()
    captured = json.loads((path.parent / "tool-arguments.jsonl").read_text())
    assert captured["argument"] == body and captured["truncated"] is False
    expected_types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    assert captured["expected_types"] == sorted(expected_types)
    assert captured["parser"] == "qwen38-xml-v1" and captured["category"] == category
    assert set(captured) == {"timestamp", "category", "argument", "truncated", "expected_types", "parser"}


def test_unknown_categories_cannot_write_payloads():
    tool_diagnostics.record_xml_rejection("synthetic private text")
    tool_diagnostics.record_xml_argument("unknown", "synthetic private text", frozenset({"integer"}), "qwen38-xml-v1")
    assert not tool_diagnostics._log_directory().exists()


def test_authorized_argument_capture_is_bounded_and_private(monkeypatch):
    monkeypatch.setattr(tool_diagnostics, "_MAX_CAPTURE_CHARACTERS", 8192)
    monkeypatch.setattr(tool_diagnostics, "_MAX_ARGUMENT_LOG_BYTES", 512 * 1024)
    body = "x" * 9000
    for _ in range(80):
        tool_diagnostics.record_xml_argument("xml_value_json", body, frozenset({"object"}), "qwen38-xml-v1")
    path = tool_diagnostics._log_directory() / "tool-arguments.jsonl"
    assert path.stat().st_size <= 512 * 1024 and stat.S_IMODE(path.stat().st_mode) == 0o600
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records and all(record["argument"] == body[:8192] and record["truncated"] for record in records)


def test_unexpected_logger_failure_preserves_original_rejection(monkeypatch):
    def fail(category):
        raise RuntimeError("synthetic private exception")

    monkeypatch.setattr(api_tools, "record_xml_rejection", fail)
    monkeypatch.setattr(api_tools, "record_xml_argument", lambda *args: fail(None))
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": "bad"}), tools=[xml_property_tool({"type": "integer"})])
    assert error.value.code == "invalid_tool_call" and "[xml_value_json]" in error.value.message
    assert "synthetic private exception" not in error.value.message


def test_log_remains_bounded_and_retains_newest_record(monkeypatch):
    monkeypatch.setattr(tool_diagnostics, "_MAX_BYTES", 200)
    for _ in range(10):
        tool_diagnostics.record_xml_rejection("xml_value_type")
    path = tool_diagnostics._log_directory() / "tool-diagnostics.jsonl"
    assert path.stat().st_size <= 200
    assert all(json.loads(line)["category"] == "xml_value_type" for line in path.read_text().splitlines())


@pytest.mark.parametrize("blocked", ["directory", "file", "symlink", "hardlink"])
def test_unavailable_or_unsafe_log_does_not_change_parser_error(tmp_path, blocked):
    directory = tool_diagnostics._log_directory()
    target = tmp_path / "unrelated"
    target.write_text("untouched")
    if blocked == "directory":
        directory.symlink_to(tmp_path, target_is_directory=True)
    else:
        directory.mkdir()
        path = directory / "tool-diagnostics.jsonl"
        if blocked == "file":
            path.mkdir()
        elif blocked == "symlink":
            path.symlink_to(target)
        else:
            path.hardlink_to(target)
    with pytest.raises(APIError) as error:
        xml_output(xml_call({"key": "bad"}), tools=[xml_property_tool({"type": "integer"})])
    assert "[xml_value_json]" in error.value.message
    assert target.read_text() == "untouched"
