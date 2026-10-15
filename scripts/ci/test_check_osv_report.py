"""Model-free regression tests for the advisory policy and report completeness."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import check_dependency_exceptions as exceptions_gate
import check_osv_report as gate
from test_check_dependency_exceptions import TODAY, write_scope


class ReportPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lockfiles = ["Cargo.lock", "tools/uv.lock", "binding/pnpm-lock.yaml"]
        self.document = {"results": [
            {"source": {"path": str(self.root / filename), "type": "lockfile"},
             "packages": [{"package": {"ecosystem": ecosystem, "name": name, "version": "1.0.0"}}]}
            for filename, ecosystem, name in zip(
                self.lockfiles, ["crates.io", "PyPI", "npm"], ["fixture-crate", "fixture-python", "fixture-js"])
        ]}

    def advisory(self, informational="unmaintained", ecosystem="crates.io", name="fixture-crate"):
        specific = {} if informational is None else {"informational": informational}
        return {"id": "RUSTSEC-2099-0001", "schema_version": "1.8.0",
                "modified": "2099-01-01T00:00:00Z", "affected": [
            {"package": {"ecosystem": ecosystem, "name": name}, "database_specific": specific,
             "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}]}]}
        ]}

    def attach(self, advisory, index=0):
        package = self.document["results"][index]["packages"][0]
        package["vulnerabilities"] = [advisory]
        package["groups"] = [{"ids": [advisory["id"]]}]

    def validate(self):
        return gate.validate_report(self.document, self.root, self.lockfiles)

    def assert_invalid(self, document=None):
        with self.assertRaises(gate.InvalidReport):
            gate.validate_report(document if document is not None else self.document, self.root, self.lockfiles)

    def test_clean_report_requires_every_requested_lockfile(self):
        result = self.validate()
        self.assertEqual((result["sources"], result["packages"], result["blocking"]), (3, 3, []))
        self.document["results"].pop()
        self.assert_invalid()

    def test_full_report_accepts_valid_vcs_commit_metadata(self):
        identity = self.document["results"][1]["packages"][0]["package"]
        for commit in ("a" * 40, "A1" * 20):
            with self.subTest(commit=commit):
                identity["commit"] = commit
                result = self.validate()
                self.assertEqual((result["sources"], result["packages"], result["blocking"]), (3, 3, []))

    def test_full_report_rejects_invalid_vcs_commit_metadata(self):
        identity = self.document["results"][1]["packages"][0]["package"]
        for commit in ("", "a" * 39, "a" * 41, "z" * 40, "a" * 40 + "\n", True, 40, None, ["a" * 40]):
            with self.subTest(commit=commit):
                identity["commit"] = commit
                self.assert_invalid()

    def test_full_report_rejects_unknown_package_identity_fields(self):
        identity = self.document["results"][1]["packages"][0]["package"]
        identity["commit"] = "a" * 40
        identity["unexpected"] = "metadata"
        self.assert_invalid()

    def test_full_report_requires_identity_fields_when_commit_is_present(self):
        baseline = copy.deepcopy(self.document)
        for field in ("ecosystem", "name", "version"):
            with self.subTest(field=field):
                self.document = copy.deepcopy(baseline)
                identity = self.document["results"][1]["packages"][0]["package"]
                identity["commit"] = "a" * 40
                identity.pop(field)
                self.assert_invalid()

    def test_maintenance_notice_warns_but_unsound_and_unknown_fail(self):
        self.attach(self.advisory())
        self.assertEqual(len(self.validate()["maintenance"]), 1)
        for classification in ("unsound", None, "notice", "UNMAINTAINED", True, ["unmaintained"]):
            with self.subTest(classification=classification):
                self.attach(self.advisory(classification))
                result = self.validate()
                self.assertEqual((len(result["maintenance"]), len(result["blocking"])), (0, 1))

    def test_unrelated_affected_package_cannot_downgrade_a_finding(self):
        advisory = self.advisory(None)
        advisory["affected"].append(self.advisory(name="other-crate")["affected"][0])
        self.attach(advisory)
        self.assertEqual(len(self.validate()["blocking"]), 1)
        advisory["affected"].pop(0)
        self.assert_invalid()

    def test_mixed_classification_for_same_package_remains_blocking(self):
        advisory = self.advisory()
        advisory["affected"].append(self.advisory("unsound")["affected"][0])
        self.attach(advisory)
        self.assertEqual(len(self.validate()["blocking"]), 1)

    def test_pypi_identity_uses_registry_normalization(self):
        self.document["results"][1]["packages"][0]["package"]["name"] = "Fixture.Python"
        self.attach(self.advisory(ecosystem="PyPI", name="fixture_python"), index=1)
        self.assertEqual(len(self.validate()["maintenance"]), 1)

    def test_versionless_local_project_cannot_carry_a_downgraded_finding(self):
        self.document["results"][1]["packages"][0]["package"]["version"] = ""
        self.validate()
        self.attach(self.advisory(ecosystem="PyPI", name="fixture-python"), index=1)
        self.assert_invalid()

    def test_registry_version_shapes_fail_closed(self):
        baseline = copy.deepcopy(self.document)
        for index, version in ((0, ""), (2, ""), (0, None), (2, 1), (1, "1.0\nprivate")):
            with self.subTest(index=index, version=version):
                document = copy.deepcopy(baseline)
                document["results"][index]["packages"][0]["package"]["version"] = version
                self.assert_invalid(document)

    def test_empty_duplicate_unexpected_and_wrong_ecosystem_sources_fail(self):
        baseline = copy.deepcopy(self.document)
        mutations = [
            lambda doc: doc["results"][0].update(packages=[]),
            lambda doc: doc["results"].append(copy.deepcopy(doc["results"][0])),
            lambda doc: doc["results"][0]["source"].update(path=str(self.root / "other/Cargo.lock")),
            lambda doc: doc["results"][0]["packages"][0]["package"].update(ecosystem="npm"),
            lambda doc: doc.update(error="private exception must not escape"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = copy.deepcopy(baseline)
                mutate(document)
                self.assert_invalid(document)

    def test_missing_advisory_record_referenced_by_group_fails(self):
        self.attach(self.advisory())
        self.document["results"][0]["packages"][0]["vulnerabilities"] = []
        self.assert_invalid()

    def test_malformed_advisory_schema_cannot_become_a_warning(self):
        mutations = [
            lambda advisory: advisory.update(schema_version=1.8),
            lambda advisory: advisory.update(schema_version="2.0.0"),
            lambda advisory: advisory.pop("modified"),
            lambda advisory: advisory.update(modified="2099-02-31T00:00:00Z"),
            lambda advisory: advisory["affected"][0].pop("ranges"),
            lambda advisory: advisory["affected"][0].update(ranges={}),
            lambda advisory: advisory["affected"][0]["ranges"][0].update(type="unknown"),
            lambda advisory: advisory["affected"][0]["ranges"][0].update(events=[]),
            lambda advisory: advisory["affected"][0]["ranges"][0].update(events=[{"introduced": "0", "fixed": "2"}]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                advisory = self.advisory()
                mutate(advisory)
                self.attach(advisory)
                with self.assertRaises((gate.InvalidReport, ValueError)):
                    self.validate()

    def test_duplicate_package_and_bad_identifiers_fail(self):
        self.document["results"][0]["packages"].append(
            copy.deepcopy(self.document["results"][0]["packages"][0]))
        self.assert_invalid()
        self.document["results"][0]["packages"].pop()
        advisory = self.advisory()
        advisory["id"] = "RUSTSEC-2099-0001\n::error::private"
        self.attach(advisory)
        self.assert_invalid()

    def test_malformed_truncated_duplicate_key_and_non_json_constants_fail_without_dumps(self):
        report = self.root / "report.json"
        arguments = [str(report), "--root", str(self.root)]
        for filename in self.lockfiles:
            arguments += ["--lockfile", filename]
        payloads = ["{\"results\":", "{\"results\":[],\"results\":[]}",
                    "{\"results\":NaN}", "{\"error\":\"private exception\"}"]
        for payload in payloads:
            with self.subTest(payload=payload):
                report.write_text(payload, encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    self.assertEqual(gate.main(arguments), 2)
                self.assertEqual(output.getvalue(), "ERROR: osv_report_invalid\n")

    def test_cli_does_not_hide_ordinary_vulnerability_or_withdrawn_record(self):
        advisory = self.advisory(None)
        advisory["withdrawn"] = "2099-01-01T00:00:00Z"
        self.attach(advisory)
        report = self.root / "report.json"
        report.write_text(json.dumps(self.document), encoding="utf-8")
        arguments = [str(report), "--root", str(self.root)]
        for filename in self.lockfiles:
            arguments += ["--lockfile", filename]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(gate.main(arguments), 1)
        self.assertIn("blocking=1", output.getvalue())
        self.assertNotIn(str(self.root), output.getvalue())


class AcceptedRiskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = write_scope(self.root)
        self.policy = exceptions_gate.load_policy(self.path, today=TODAY)
        self.lockfiles = ["Cargo.lock", "crates/legacy/download-manager/examples/wasm/Cargo.lock"]
        advisory = {
            "id": "RUSTSEC-2026-0346", "modified": "2026-10-15T00:00:00Z", "affected": [{
                "package": {"ecosystem": "crates.io", "name": "rustls"},
                "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}]}],
            }],
        }
        self.document = {"results": [{
            "source": {"path": str(self.root / filename), "type": "lockfile"}, "packages": [{
                "package": {"ecosystem": "crates.io", "name": "rustls", "version": "0.23.45"},
                "vulnerabilities": [copy.deepcopy(advisory)],
            }],
        } for filename in self.lockfiles]}

    def validate(self, enabled=True):
        return gate.validate_report(self.document, self.root, self.lockfiles, self.policy if enabled else None)

    def test_exception_requires_explicit_policy_and_keeps_advisories_visible(self):
        default = self.validate(False)
        self.assertEqual((len(default["blocking"]), default["accepted"]), (1, []))
        accepted = self.validate()
        self.assertEqual((accepted["blocking"], len(accepted["accepted"])), ([], 1))
        for result in self.document["results"]:
            advisory = result["packages"][0]["vulnerabilities"][0]
            result["packages"][0]["vulnerabilities"] = [
                {**copy.deepcopy(advisory), "id": identifier} for identifier in sorted(self.policy.advisories)
            ]
        self.assertEqual(len(self.validate()["accepted"]), 4)

    def test_alias_alone_cannot_match_the_accepted_advisory(self):
        for result in self.document["results"]:
            advisory = result["packages"][0]["vulnerabilities"][0]
            advisory.update(id="GHSA-fixture-identifier", aliases=["RUSTSEC-2026-0346"])
        result = self.validate()
        self.assertEqual((len(result["blocking"]), result["accepted"]), (1, []))

    def test_other_advisory_version_package_or_lockfile_remains_blocking(self):
        original = copy.deepcopy(self.document)
        for change in ("advisory", "version", "package", "lockfile"):
            with self.subTest(change=change):
                self.document = copy.deepcopy(original)
                package = self.document["results"][0]["packages"][0]
                if change == "advisory":
                    package["vulnerabilities"][0]["id"] = "RUSTSEC-2099-0001"
                elif change == "version":
                    package["package"]["version"] = "0.23.46"
                elif change == "package":
                    package["package"]["name"] = "other"
                    package["vulnerabilities"][0]["affected"][0]["package"]["name"] = "other"
                else:
                    self.document["results"][0]["source"]["path"] = str(self.root / "other/Cargo.lock")
                    self.lockfiles[0] = "other/Cargo.lock"
                self.assertEqual(len(self.validate()["blocking"]), 1)
                self.lockfiles[0] = "Cargo.lock"

    def test_exception_never_bypasses_advisory_schema_or_source_completeness(self):
        original = copy.deepcopy(self.document)
        changes = [
            lambda advisory: advisory.pop("modified"),
            lambda advisory: advisory.update(modified="2026-02-31T00:00:00Z"),
            lambda advisory: advisory["affected"][0].update(ranges=[]),
            lambda advisory: advisory["affected"][0]["ranges"][0].update(events=[]),
            lambda advisory: advisory["affected"][0]["package"].update(name="unrelated"),
            lambda advisory: advisory.update(aliases=["RUSTSEC-2026-0346", "RUSTSEC-2026-0346"]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.document = copy.deepcopy(original)
                change(self.document["results"][0]["packages"][0]["vulnerabilities"][0])
                with self.assertRaises((gate.InvalidReport, ValueError)):
                    self.validate()
        self.document = copy.deepcopy(original)
        self.document["results"].pop()
        with self.assertRaises(gate.InvalidReport):
            self.validate()

    def test_explicit_cli_policy_warns_and_expired_or_drifted_policy_fails(self):
        report = self.root / "report.json"
        report.write_text(json.dumps(self.document))
        arguments = [str(report), "--root", str(self.root), "--exceptions", str(self.path)]
        for filename in self.lockfiles:
            arguments += ["--lockfile", filename]
        with mock.patch.object(exceptions_gate, "datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 15, tzinfo=timezone.utc)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                self.assertEqual(gate.main(arguments), 0)
            self.assertIn("WARNING accepted-risk: crates.io rustls@0.23.45 RUSTSEC-2026-0346", output.getvalue())
            self.assertIn("accepted=1 blocking=0", output.getvalue())
            self.assertNotIn(str(self.root), output.getvalue())
            clock.now.return_value = datetime(2026, 11, 14, tzinfo=timezone.utc)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                self.assertEqual(gate.main(arguments), 2)
            self.assertEqual(output.getvalue(), "ERROR: osv_report_invalid\n")
            clock.now.return_value = datetime(2026, 10, 15, tzinfo=timezone.utc)
            (self.root / "deny.toml").write_text('[advisories]\nignore = []\n')
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                self.assertEqual(gate.main(arguments), 2)
            self.assertEqual(output.getvalue(), "ERROR: osv_report_invalid\n")


if __name__ == "__main__":
    unittest.main()
