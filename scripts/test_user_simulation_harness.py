"""Model-free evidence-integrity counterexamples for the manual desktop harness."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stark_user_test as harness


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stark-user-harness-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patches = [
            patch.object(harness, "RUNS", self.root),
            patch.object(
                harness,
                "installed_identities",
                return_value={"stark": {"version": "test"}},
            ),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.run = harness.prepare(["pi"])
        self.state = harness.load(self.run)
        self.program = self.run / "pi/ledger.py"
        self.program.write_text("# test artifact\n")

    def identity(self):
        harness.record_identity(
            self.state, "pi", "test/model", 32768, None, "off", "loaded", "1.5.2"
        )

    def observations(self):
        for step in harness.COMMON_STEPS + harness.LIFECYCLE_STEPS["pi"]:
            harness.observe(self.state, "pi", step, "pass")

    def accepted(self, stage):
        value = {
            "passed": True,
            "checks": [{"id": "synthetic_acceptance", "passed": True}],
            "artifact_sha256": harness.digest(self.program),
        }
        with patch.object(harness, "verify_project", return_value=value):
            return harness.verify(self.run, self.state, "pi", stage)

    def complete(self):
        self.identity()
        self.observations()
        for stage in harness.STAGES:
            self.accepted(stage)

    def test_prepare_is_private_and_contains_no_solution(self):
        other = harness.prepare(list(harness.CLIENTS))
        self.assertEqual(other.stat().st_mode & 0o777, 0o700)
        self.assertEqual((other / "run.json").stat().st_mode & 0o777, 0o600)
        for client in harness.CLIENTS:
            self.assertEqual(
                {p.name for p in (other / client).iterdir()},
                {"expenses.json", "BRIEF.md"},
            )
        self.assertEqual(
            harness.summarize(other, harness.load(other))["status"], "incomplete"
        )

    def test_no_observations_or_identity_cannot_verify(self):
        with patch.object(harness, "verify_project") as oracle:
            with self.assertRaises(harness.UserTestError):
                harness.verify(self.run, self.state, "pi", "build")
            oracle.assert_not_called()

    def test_model_changes_require_new_run(self):
        self.identity()
        with self.assertRaises(harness.UserTestError):
            harness.record_identity(
                self.state, "pi", "other/model", 32768, None, "off", "loaded", "1.5.2"
            )

    def test_completed_subset_is_explicit_and_final_artifact_changes_invalidate(self):
        self.complete()
        result = harness.report(self.run, self.state)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(set(result["clients"]), {"pi"})
        self.program.write_text("# changed after acceptance\n")
        self.assertEqual(
            harness.summarize(self.run, self.state)["status"], "incomplete"
        )

    def test_rerunning_early_stage_supersedes_later_acceptance(self):
        self.complete()
        self.accepted("build")
        self.assertEqual(
            harness.summarize(self.run, self.state)["status"], "incomplete"
        )
        with self.assertRaises(harness.UserTestError):
            self.accepted("recover")
        self.accepted("revise")
        self.accepted("recover")
        self.assertEqual(harness.summarize(self.run, self.state)["status"], "passed")

    def test_fixture_mutation_cannot_pass_hidden_acceptance(self):
        self.identity()
        self.observations()
        (self.run / "pi/expenses.json").write_text("[]")
        self.assertFalse(self.accepted("build")["passed"])
        self.assertEqual(harness.summarize(self.run, self.state)["status"], "failed")

    def test_recovery_requires_predecessor_and_repeat_cancel_invalidates(self):
        with self.assertRaises(harness.UserTestError):
            harness.observe(self.state, "pi", "cancel_recovery", "pass")
        self.complete()
        harness.observe(self.state, "pi", "cancel", "pass")
        self.assertEqual(
            harness.summarize(self.run, self.state)["status"], "incomplete"
        )
        harness.observe(self.state, "pi", "cancel_recovery", "pass")
        self.assertEqual(harness.summarize(self.run, self.state)["status"], "passed")
        with self.assertRaises(harness.UserTestError):
            harness.observe(self.state, "pi", "disconnect_denied", "pass")

    def test_changed_helpers_or_app_cannot_reuse_a_pass(self):
        self.complete()
        self.state["harness_sha256"] = "0" * 64
        self.assertEqual(
            harness.summarize(self.run, self.state)["status"], "incomplete"
        )
        with self.assertRaises(harness.UserTestError):
            self.accepted("recover")
        self.state["harness_sha256"] = harness.digest(Path(harness.__file__))
        with patch.object(
            harness,
            "installed_identities",
            return_value={"stark": {"version": "changed"}},
        ):
            self.assertEqual(
                harness.summarize(self.run, self.state)["status"], "incomplete"
            )

    def test_repaired_failures_remain_in_history(self):
        self.complete()
        harness.observe(self.state, "pi", "cancel", "fail")
        self.assertEqual(harness.summarize(self.run, self.state)["status"], "failed")
        harness.observe(self.state, "pi", "cancel", "pass")
        harness.observe(self.state, "pi", "cancel_recovery", "pass")
        self.assertEqual(harness.summarize(self.run, self.state)["status"], "passed")
        self.assertEqual(len(self.state["clients"]["pi"]["observations"]["cancel"]), 3)

    def test_foreign_or_public_manifest_rejected(self):
        with self.assertRaises(harness.UserTestError):
            harness.load(self.root)
        (self.run / "run.json").chmod(0o644)
        with self.assertRaises(harness.UserTestError):
            harness.load(self.run)

    def test_cli_exception_does_not_print_error_content(self):
        output = io.StringIO()
        arguments = ["stark_user_test.py", "report", "--run", str(self.run)]
        with (
            patch("sys.argv", arguments),
            patch.object(harness, "load", side_effect=OSError("private-sentinel")),
            contextlib.redirect_stderr(output),
        ):
            self.assertEqual(harness.main(), 2)
        self.assertEqual(output.getvalue().strip(), "user_test_operation_failed")
        self.assertNotIn("private-sentinel", json.dumps(self.state))


if __name__ == "__main__":
    unittest.main()
