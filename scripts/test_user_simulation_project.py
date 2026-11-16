"""Counterexamples for the independent manual-test project acceptance oracle."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import user_simulation_project as project

REFERENCE = """import argparse, csv, json, sys
VALIDATE = True
p = argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--category')
p.add_argument('--format', choices=['json', 'csv'], default='json')
a = p.parse_args()
try:
    with open(a.input) as f:
        rows = json.load(f)
    if VALIDATE:
        if not isinstance(rows, list):
            raise ValueError()
        ids = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError()
            if not isinstance(row['id'], str) or not isinstance(row['category'], str):
                raise ValueError()
            if type(row['amount_cents']) is not int or row['amount_cents'] < 0 or row['id'] in ids:
                raise ValueError()
            ids.add(row['id'])
    rows = [r for r in rows if a.category is None or r['category'] == a.category]
    totals = {}
    for row in rows:
        totals[row['category']] = totals.get(row['category'], 0) + row['amount_cents']
    if a.format == 'csv':
        w = csv.writer(sys.stdout)
        w.writerow(['category', 'total_cents'])
        w.writerows(sorted(totals.items()))
    else:
        print(json.dumps({'count': len(rows), 'total_cents': sum(totals.values()), 'by_category': totals}))
except (ValueError, KeyError, TypeError):
    sys.exit(2)
"""


class ProjectAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stark-oracle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)

    def verify(self, source=REFERENCE, stage="recover"):
        (self.workspace / "ledger.py").write_text(source)
        return project.verify_project(self.workspace, stage)

    def test_valid_project_passes_all_stages(self):
        for stage in project.STAGES:
            with self.subTest(stage=stage):
                result = self.verify(stage=stage)
                self.assertTrue(result["passed"])
                self.assertEqual(len(result["artifact_sha256"]), 64)
                self.assertEqual(set(result), {"passed", "checks", "artifact_sha256"})

    def test_sample_hardcoding_fails_hidden_case(self):
        expected = project._expected(project.fixture_rows())
        result = self.verify("print(" + repr(json.dumps(expected)) + ")", "build")
        self.assertFalse(result["passed"])
        self.assertTrue(
            next(c["passed"] for c in result["checks"] if c["id"] == "sample_json")
        )
        self.assertFalse(
            next(c["passed"] for c in result["checks"] if c["id"] == "varied_json")
        )

    def test_build_success_does_not_imply_revision_success(self):
        source = REFERENCE.replace(
            "a = p.parse_args()",
            "a = p.parse_args(); a.category = None; a.format = 'json'",
        )
        self.assertTrue(self.verify(source, "build")["passed"])
        self.assertFalse(self.verify(source, "revise")["passed"])

    def test_recovery_gate_rejects_boolean_negative_and_duplicates(self):
        source = REFERENCE.replace("VALIDATE = True", "VALIDATE = False")
        self.assertTrue(self.verify(source, "revise")["passed"])
        result = self.verify(source)
        self.assertFalse(result["passed"])
        checks = {check["id"]: check["passed"] for check in result["checks"]}
        for name in (
            "boolean_amount",
            "negative_amount",
            "duplicate_id",
            "fractional_amount",
        ):
            self.assertFalse(checks[name])

    def test_input_modification_fails_despite_correct_output(self):
        source = REFERENCE + "\nopen(a.input, 'w').write('[]')\n"
        self.assertFalse(self.verify(source, "build")["passed"])

    def test_timeout_and_output_flood_are_bounded_and_content_free(self):
        with patch.object(project, "CASE_TIMEOUT", 0.1):
            result = self.verify("import time; time.sleep(60)", "build")
        self.assertFalse(result["passed"])
        result = self.verify("print('private-output-sentinel' * 100000)", "build")
        self.assertFalse(result["passed"])
        self.assertNotIn("private-output-sentinel", json.dumps(result))

    def test_environment_does_not_forward_credentials(self):
        source = "import os; assert 'STARK_API_KEY' not in os.environ\n" + REFERENCE
        with patch.dict(os.environ, {"STARK_API_KEY": "test-secret"}):
            self.assertTrue(self.verify(source)["passed"])

    def test_background_descendant_is_stopped_after_normal_exit(self):
        marker = self.workspace / "descendant-marker"
        command = (
            "import time, pathlib; time.sleep(.4); pathlib.Path("
            + repr(str(marker))
            + ").write_text('late')"
        )
        source = (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', " + repr(command) + "], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        ) + REFERENCE
        self.assertTrue(self.verify(source, "build")["passed"])
        time.sleep(0.5)
        self.assertFalse(marker.exists())

    def test_deeply_nested_output_is_a_failure_without_exception(self):
        source = "print('[' * 5000 + ']' * 5000)"
        self.assertFalse(self.verify(source, "build")["passed"])

    def test_missing_symlink_and_nonregular_program_fail_without_execution(self):
        self.assertFalse(project.verify_project(self.workspace, "build")["passed"])
        target = self.workspace / "other.py"
        target.write_text(REFERENCE)
        program = self.workspace / "ledger.py"
        program.symlink_to(target)
        self.assertFalse(project.verify_project(self.workspace, "build")["passed"])
        program.unlink()
        os.mkfifo(program)
        self.assertFalse(project.verify_project(self.workspace, "build")["passed"])

    def test_successful_json_on_error_is_rejected(self):
        source = REFERENCE.replace("sys.exit(2)", "print('{}'); sys.exit(2)")
        self.assertFalse(self.verify(source)["passed"])

    def test_boolean_counts_are_not_accepted_as_integers(self):
        source = REFERENCE.replace("'count': len(rows)", "'count': bool(len(rows))")
        self.assertFalse(self.verify(source, "build")["passed"])


if __name__ == "__main__":
    unittest.main()
