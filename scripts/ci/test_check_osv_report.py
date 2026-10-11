"""Model-free regression tests for the advisory policy and report completeness."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import check_osv_report as gate


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


if __name__ == "__main__":
    unittest.main()
