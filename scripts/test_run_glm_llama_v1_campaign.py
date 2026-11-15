from __future__ import annotations

import json
from pathlib import Path

import pytest
import run_glm_llama_v1_campaign as campaign


def cell(input_tokens, budget, *, wall=0.0, decode=0.0, wall_range=0.02, decode_range=0.02,
         wall_wins=8, decode_wins=8):
    return {
        "input_tokens": input_tokens,
        "requested_output_tokens": budget,
        "pairs": 8,
        "median_paired_wall_reduction": wall,
        "median_paired_decode_gain": decode,
        "wall_pair_wins": wall_wins,
        "decode_pair_wins": decode_wins,
        "before_wall_seconds": {"relative_range": wall_range},
        "after_wall_seconds": {"relative_range": wall_range},
        "before_useful_decode_tokens_per_second": {"relative_range": decode_range},
        "after_useful_decode_tokens_per_second": {"relative_range": decode_range},
    }


def test_screen_rejects_positive_wall_signal_inside_same_code_noise():
    result = campaign.screen_decision({"passed": True, "summary": [cell(512, 16, wall=0.06, wall_range=0.07)]})

    assert result["survives"] is False


def test_screen_requires_five_percent_decode_gain_beyond_range():
    noisy = campaign.screen_decision({"passed": True, "summary": [cell(512, 16, decode=0.05, decode_range=0.06)]})
    signal = campaign.screen_decision({"passed": True, "summary": [cell(512, 16, decode=0.06, decode_range=0.05)]})

    assert noisy["survives"] is False
    assert signal["survives"] is True


def test_screen_rejects_another_cell_with_over_ten_percent_wall_regression():
    result = campaign.screen_decision({
        "passed": True,
        "summary": [cell(96, 8, wall=0.08), cell(512, 16, wall=-0.101)],
    })

    assert result["survives"] is False


def test_one_confirmation_workload_is_not_enough():
    result = campaign.confirmation_decision({"passed": True, "summary": [cell(512, 16, wall=0.2, wall_range=0.15)]})

    assert result["functional_passed"] is True
    assert result["performance_confirmed"] is False
    assert result["wall_confirming_cells"] == 1


def test_confirmation_uses_maximum_of_ten_percent_or_observed_range():
    result = campaign.confirmation_decision({
        "passed": True,
        "summary": [
            cell(512, 16, wall=0.21, wall_range=0.20, wall_wins=7),
            cell(3072, 16, wall=0.12, wall_range=0.02, wall_wins=7),
        ],
    })

    assert result["performance_confirmed"] is True
    assert result["wall_confirming_cells"] == 2
    assert result["confirmed_effect"] == "complete_request_latency"


def test_confirmation_rejects_over_five_percent_wall_regression_elsewhere():
    result = campaign.confirmation_decision({
        "passed": True,
        "summary": [
            cell(512, 16, wall=0.15),
            cell(3072, 16, wall=0.15),
            cell(512, 128, wall=-0.051),
        ],
    })

    assert result["performance_confirmed"] is False
    assert result["confirmed_effect"] is None


def long_decode_report(*, sampled=128):
    rows = [
        {"input_tokens": 512, "requested_output_tokens": 128, "warmup": False,
         "sampled_tokens": sampled}
        for _ in range(16)
    ]
    return {
        "passed": True,
        "sessions": [{"requests": rows}],
        "summary": [
            cell(512, 128, decode=0.21, decode_range=0.20, decode_wins=7),
            cell(3072, 16, decode=0.12, decode_range=0.02, decode_wins=7),
        ],
    }


def test_general_decode_confirmation_requires_full_budget_for_all_measured_rows():
    result = campaign.confirmation_decision(long_decode_report())
    early_stop = campaign.confirmation_decision(long_decode_report(sampled=127))

    assert result["performance_confirmed"] is True
    assert result["confirmed_effect"] == "general_useful_decode_throughput"
    assert result["general_decode_long_cell_confirmed"] is True
    assert early_stop["performance_confirmed"] is False
    assert early_stop["general_decode_long_cell_confirmed"] is False


def test_general_decode_confirmation_rejects_decode_regression_in_another_cell():
    report = long_decode_report()
    report["summary"].append(cell(96, 8, decode=-0.051))

    result = campaign.confirmation_decision(report)

    assert result["general_decode_long_cell_confirmed"] is True
    assert result["decode_regression_guard_passed"] is False
    assert result["performance_confirmed"] is False


def test_shorter_decode_signal_cannot_stand_alone_as_general_decode_confirmation():
    result = campaign.confirmation_decision({
        "passed": True,
        "summary": [
            cell(512, 16, decode=0.2, decode_range=0.1, decode_wins=7),
            cell(3072, 16, decode=0.2, decode_range=0.1, decode_wins=7),
        ],
    })

    assert result["decode_confirming_cells"] == 2
    assert result["general_decode_long_cell_confirmed"] is False
    assert result["performance_confirmed"] is False


@pytest.mark.parametrize("valid_configs, usable", [(set(), 0), ({"defaults", "profile", "mma_off"}, 2)])
def test_validation_failures_remain_visible_while_independent_comparisons_continue(
    tmp_path, monkeypatch, valid_configs, usable,
):
    output = tmp_path / "output"
    output.mkdir()
    receipt = tmp_path / "weights.json"
    receipt.write_text(json.dumps({"passed": True}))
    stub = tmp_path / "stub"
    monkeypatch.setattr(campaign, "prepare", lambda _args: (stub, stub, stub, stub, receipt, output))
    monkeypatch.setattr(campaign, "digest", lambda _path: "source-hash")

    class Phase:
        def __init__(self, command, **_kwargs):
            phase = command[2]
            phase_output = Path(command[command.index("--output") + 1])
            phase_output.mkdir()
            report = {"identities": {"source": "same"}, "weights": {"weight": "same"}}
            if phase == "validate":
                report.update(passed=False, validation={
                    name: {"passed": name in valid_configs} for name in campaign.BASE_CONFIGS
                })
                self.code = 1
            else:
                assert phase == "screen"
                report.update(passed=True, summary=[cell(512, 16)])
                self.code = 0
            (phase_output / "report.json").write_text(json.dumps(report))

        def wait(self, **_kwargs):
            return self.code

        def poll(self):
            return self.code

    monkeypatch.setattr(campaign.subprocess, "Popen", Phase)
    args = [item for name in ("model", "worker", "example", "corpus", "verified-weights", "output")
            for item in ("--" + name, str(output if name == "output" else stub))]

    assert campaign.main(args) == 1
    result = json.loads((output / "campaign.json").read_text())
    assert result["status"] == "completed_with_failures"
    assert result["failed_phase_count"] == 1
    assert result["usable_comparison_count"] == usable
