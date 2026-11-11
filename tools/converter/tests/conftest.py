"""Keep synthetic diagnostic records out of the user's app log."""

import pytest

from stark_convert import tool_diagnostics


@pytest.fixture(autouse=True)
def isolated_tool_diagnostic_log(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_diagnostics, "_log_directory", lambda: tmp_path / "tool-logs")
