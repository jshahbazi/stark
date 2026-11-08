import os
import subprocess
import sys
from pathlib import Path

import pytest

from stark_convert import resource_budget


@pytest.mark.parametrize(("available", "expected"), [(0, 16), (2 << 30, 16), (4 << 30, 32), (8 << 30, 64)])
def test_automatic_budget_tracks_available_memory(available, expected):
    assert resource_budget.automatic_budget_mib(available) == expected


def test_resource_budget_formulas():
    size = resource_budget.REQUEST_BUDGET_MIB * 1024 * 1024
    assert resource_budget.INPUT_BYTES == size
    assert resource_budget.WIRE_BYTES == 6 * size + 2 * 1024 * 1024
    assert resource_budget.PENDING_BYTES == 2 * (resource_budget.WIRE_BYTES + 6 * 1024 * 1024)
    assert resource_budget.TEMPLATE_LIVE_BYTES == 128 * 1024 * 1024 + 8 * size
    assert resource_budget.TEMPLATE_CUMULATIVE_BYTES == 32 * 1024 * 1024 + 32 * size


@pytest.mark.parametrize(("setting", "expected"), [("1", 1), ("64", 64), ("01", 1)])
def test_resource_budget_setting_is_read_at_process_start(setting, expected):
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, STARK_REQUEST_BUDGET_MIB=setting)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from stark_convert.resource_budget import REQUEST_BUDGET_MIB; print(REQUEST_BUDGET_MIB)",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(expected)


@pytest.mark.parametrize("setting", ["", "0", "65", "1.0", "+1", " 1", "001", "１２"])
def test_resource_budget_rejects_invalid_settings_without_echoing_them(setting):
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, STARK_REQUEST_BUDGET_MIB=setting)
    result = subprocess.run(
        [sys.executable, "-c", "import stark_convert.resource_budget"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() == "Invalid STARK_REQUEST_BUDGET_MIB; expected a decimal integer from 1 through 64."
