"""Model-free evidence-integrity counterexamples for the manual desktop harness."""

from __future__ import annotations

import contextlib
import io
import json
import plistlib
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
                return_value={
                    "stark": {
                        "app_path": "/synthetic/Stark.app",
                        "executable_sha256": "a" * 64,
                        "runtime_sha256": {"libexec/stark-generate": "b" * 64},
                    },
                    "pi": {
                        "executable_path": "/synthetic/pi",
                        "executable_sha256": "c" * 64,
                    },
                },
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

    def test_failure_survives_helper_or_application_identity_drift(self):
        self.complete()
        harness.observe(self.state, "pi", "cancel", "fail")
        for drift in ("helper", "application"):
            with self.subTest(drift=drift):
                self.state["harness_sha256"] = (
                    "0" * 64
                    if drift == "helper"
                    else harness.digest(Path(harness.__file__))
                )
                builds = self.state["installed"] if drift == "helper" else {}
                with patch.object(harness, "installed_identities", return_value=builds):
                    result = harness.report(self.run, self.state)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["evidence_valid"])
                self.assertIn("Outcome: failed", (self.run / "report.md").read_text())

    def test_missing_build_identity_cannot_be_bound_or_pass(self):
        with (
            patch.object(harness, "installed_identities", return_value={}),
            self.assertRaisesRegex(
                harness.UserTestError, "tested_application_unavailable"
            ),
        ):
            self.identity()
        self.complete()
        self.state["installed"]["stark"] = {"executable_sha256": None}
        with patch.object(
            harness, "installed_identities", return_value=self.state["installed"]
        ):
            self.assertEqual(
                harness.summarize(self.run, self.state)["status"], "incomplete"
            )

    def test_changed_running_application_blocks_verification(self):
        self.complete()
        with (
            patch.object(harness, "installed_identities", return_value={}),
            self.assertRaisesRegex(harness.UserTestError, "tested_application_changed"),
        ):
            self.accepted("recover")

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


class ApplicationIdentityTests(unittest.TestCase):
    def test_running_user_install_is_bound_and_worker_changes_are_detected(self):
        with tempfile.TemporaryDirectory(prefix="stark-app-identity-") as temporary:
            root = Path(temporary)
            user_app = root / "user/Applications/Stark.app"
            other_app = root / "Applications/Stark.app"
            for app in (user_app, other_app):
                contents = app / "Contents"
                (contents / "MacOS").mkdir(parents=True)
                (contents / "Info.plist").write_bytes(
                    plistlib.dumps(
                        {
                            "CFBundleExecutable": "Stark",
                            "CFBundleShortVersionString": "test",
                        }
                    )
                )
                (contents / "MacOS/Stark").write_bytes(b"launcher")
                runtime = contents / "Resources/runtime"
                (runtime / "libexec").mkdir(parents=True)
                for name in ("stark-generate", "stark-template"):
                    (runtime / "libexec" / name).write_bytes(b"worker")
                (runtime / "libexec/python/bin").mkdir(parents=True)
                (runtime / "libexec/python/bin/python3.13").write_bytes(b"python")
                backend = runtime / "lib/python/site-packages/stark_convert"
                backend.mkdir(parents=True)
                (backend / "desktop.py").write_text("# synthetic backend")
            with (
                patch.object(
                    harness, "running_app_paths", return_value={"stark": user_app}
                ),
                patch.object(harness.shutil, "which", return_value=None),
            ):
                original = harness.installed_identities()
            self.assertEqual(original["stark"]["app_path"], str(user_app.resolve()))
            self.assertTrue(harness.available_builds(original, []))
            # Rehash the bound bundle even after cleanup closes the application.
            with patch.object(
                harness,
                "running_app_paths",
                side_effect=AssertionError("must use bound path"),
            ):
                unchanged = harness.installed_identities(original)
                self.assertEqual(original, unchanged)
                for relative in (
                    "libexec/stark-generate",
                    "libexec/python/bin/python3.13",
                    "lib/python/site-packages/stark_convert/desktop.py",
                ):
                    with self.subTest(changed=relative):
                        target = user_app / "Contents/Resources/runtime" / relative
                        previous = target.read_bytes()
                        target.write_bytes(b"changed-runtime")
                        changed = harness.installed_identities(original)
                        self.assertEqual(
                            original["stark"]["executable_sha256"],
                            changed["stark"]["executable_sha256"],
                        )
                        self.assertNotEqual(
                            original["stark"]["runtime_sha256"],
                            changed["stark"]["runtime_sha256"],
                        )
                        target.write_bytes(previous)

    def test_running_bundle_discovery_fails_closed_on_ambiguity(self):
        reply = [
            {"id": "com.starkengine.menubar", "path": "/first/Stark.app"},
            {"id": "com.starkengine.menubar", "path": "/second/Stark.app"},
            {
                "id": "com.differentai.openwork",
                "path": "/user/Applications/OpenWork.app",
            },
        ]
        with patch.object(
            harness.subprocess,
            "run",
            return_value=type(
                "Reply", (), {"stdout": json.dumps(reply), "returncode": 0}
            )(),
        ):
            paths = harness.running_app_paths()
        self.assertNotIn("stark", paths)
        self.assertEqual(paths["openwork"], Path("/user/Applications/OpenWork.app"))


if __name__ == "__main__":
    unittest.main()
