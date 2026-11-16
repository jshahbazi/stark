"""Publish counts, hashes and timing aggregates from the sealed reconciliation."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "runtime/lily-mtp-reconcile-20261115"
RESULTS = PRIVATE / "results-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["completed"] and len(summary["cells"]) == 6
    initial = json.loads((PRIVATE / "results/summary.json").read_text())
    assert initial["cells"][0]["failure_type"] == "StatisticsError"
    assert sha(PRIVATE / "run-as-initial.py") == initial["harness_sha256"]
    assert sha(PRIVATE / "run-v2.py") == summary["harness_sha256"]
    report = dict(summary)
    report["cells"] = []
    measured_count = 0
    setup_count = 0
    recovered_source = PRIVATE / "results/before-input512-top20-out256-seed42.json"
    original_cell = json.loads(recovered_source.read_text())
    original_measured = [r for b in original_cell["blocks"] for r in b["requests"]]
    assert len(original_measured) == 18
    for index, meta in enumerate(summary["cells"]):
        source = RESULTS / (meta["name"] + ".json")
        cell = json.loads(source.read_text())
        assert cell["passed"]
        assert len(cell["warmups"]) == 3
        assert sorted(r["depth"] for r in cell["warmups"]) == [0, 1, 2]
        assert cell["selector_probe"]["selector"] == "enabled"
        setup_count += len(cell["warmups"]) + 1
        measured = [r for block in cell["blocks"] for r in block["requests"]]
        assert len(cell["blocks"]) == (6 if index == 0 else 12)
        assert [b["order"] for b in cell["blocks"]] == summary["plans"][index]
        if index == 0:
            assert cell["recovered_from_sha256"] == sha(recovered_source)
            native = lambda row: {k: v for k, v in row.items() if not k.startswith("clock_")}
            assert [native(r) for r in measured] == [native(r) for r in original_measured]
        measured_count += len(measured)
        assert all(r["reused_input_tokens"] == 0 for r in measured)
        assert all(sorted(block["order"]) == [0, 1, 2] for block in cell["blocks"])
        rows = cell["warmups"] + [cell["selector_probe"]] + measured
        published = dict(meta)
        published["receipt_sha256"] = sha(source)
        published["measurements"] = []
        for block in cell["blocks"]:
            for item in block["requests"]:
                # No generated text, tokens or probabilities exist in these receipts.
                row = {k: v for k, v in item.items() if not k.startswith("clock_")}
                row["block_number"] = block["number"]
                published["measurements"].append(row)
        published["excluded_fixed_setup"] = [
            {**{k: v for k, v in row.items() if not k.startswith("clock_")},
             "exclusion_reason": "fixed warmup" if row["selector"] == "explicit" else "fixed selector probe"}
            for row in cell["warmups"] + [cell["selector_probe"]]
        ]
        published["host_aggregate"] = {
            "measured": len(measured),
            "legacy_eligible": sum(r["host_screen"].get("eligible", False) for r in measured),
            "missing_host_observations": sum(r["host_screen"].get("observation_status") == "missing" for r in measured),
            "external_cpu_max": max(r["host_screen"].get("external_cpu_max", 0) for r in measured),
            "thermal_max": max(r["host_screen"].get("thermal_max", 0) for r in measured),
            "paging_delta": {key: sum(r["host_screen"].get("paging_delta", {}).get(key, 0) for r in measured)
                             for key in ("pageouts", "swapouts")},
        }
        published["stages_by_depth"] = {}
        for depth in (0, 1, 2):
            same = [r for r in rows if r["depth"] == depth]
            assert len({r["output_sha256"] for r in same}) == 1
            if depth:
                active = [r for r in measured if r["depth"] == depth]
                published["stages_by_depth"][str(depth)] = {
                    key: sum(r["mtp"]["timings_seconds"][key] for r in active)
                    for key in ("draft_seconds", "verify_seconds", "rollback_seconds")
                }
        blocks = [{r["depth"]: r for r in b["requests"]} for b in cell["blocks"]]
        published["median_one_vs_two_absolute_wall_saved_seconds"] = statistics.median(
            p[2]["wall_seconds"] - p[1]["wall_seconds"] for p in blocks
        )
        report["cells"].append(published)
    assert measured_count == 198 and setup_count == 24
    for depth in ("0", "1", "2"):
        first, second = report["cells"][0]["arm_summary"][depth], report["cells"][1]["arm_summary"][depth]
        assert first["output_sha256"] == second["output_sha256"]
        assert first["mtp_counts"] == second["mtp_counts"]
    primary = report["cells"][1]["contrasts"]["1_vs_2"]
    held = report["cells"][5]["contrasts"]["1_vs_2"]
    conditions = {
        "primary_at_least_five_percent_request_reduction": primary["paired_wall_change"] <= -0.05,
        "primary_at_least_ten_of_twelve_latency_wins": primary["wall_wins"] >= 10,
        "primary_conditional_interval_excludes_zero": primary["wall_change_moving_block_interval"][1] < 0,
        "heldout_latency_same_direction": held["paired_wall_change"] < 0,
        "before_primary_same_direction": report["cells"][0]["contrasts"]["1_vs_2"]["paired_wall_change"] < 0,
    }
    for cell in report["cells"][2:5]:
        contrast = cell["contrasts"]["1_vs_2"]
        conditions[cell["name"] + "_wall_guard"] = contrast["paired_wall_change"] <= 0.05
        conditions[cell["name"] + "_useful_decode_guard"] = contrast["paired_decode_change"] >= -0.05
    report["decision"] = {"checks": conditions, "keep_one_as_recommendation": all(conditions.values())}
    report["total_measured_requests"] = measured_count
    report["total_fixed_setup_requests"] = setup_count
    report["initial_auxiliary_failure"] = {
        "failure_type": "StatisticsError", "clock_sample_count": 0, "clock_parse_failures": 172,
        "native_measurements_recovered": len(original_measured), "native_measurements_rerun": 0,
        "source_sha256": initial["harness_sha256"],
        "summary_sha256": sha(PRIVATE / "results/summary.json"),
        "clock_sha256": sha(PRIVATE / "results/before-input512-top20-out256-seed42-clock-private.json"),
    }
    report["publication_harness_sha256"] = sha(Path(__file__))
    target = ROOT / "docs/benchmarks/data/lily-mtp-default-reconciliation-2026-11-15.json"
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"measured": measured_count, "decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
