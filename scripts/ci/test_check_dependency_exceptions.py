"""Scope, expiry and configuration-drift checks for the temporary exception."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import check_dependency_exceptions as gate

TODAY = date(2026, 10, 15)


def write_scope(root: Path) -> Path:
    policy = root / gate.POLICY_PATH
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_bytes(Path(__file__).with_name("dependency-exceptions.toml").read_bytes())
    accepted = gate.load_policy(policy, today=TODAY)
    lock = (f'[[package]]\nname = "rustls"\nversion = "{accepted.version}"\n'
            f'source = "{accepted.source}"\nchecksum = "{accepted.checksum}"\n')
    for filename in accepted.lockfiles:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(lock)
    ignores = ",\n".join(f'{{ id = "{identifier}", reason = "{accepted.rationale}" }}'
                         for identifier in sorted(accepted.advisories))
    (root / "deny.toml").write_text(f"[advisories]\nignore = [\n{ignores}\n]\n")
    return policy


class ExceptionPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = write_scope(self.root)

    def load(self, today=TODAY):
        return gate.load_policy(self.path, today=today)

    def test_exact_scope_passes_and_expiry_is_exclusive_utc(self):
        gate.validate_preflight(self.load(), self.root)
        self.load(date(2026, 11, 13))
        for day in (date(2026, 11, 14), date(2026, 11, 15)):
            with self.subTest(day=day), self.assertRaisesRegex(gate.InvalidPolicy, "expired"):
                self.load(day)

    def test_policy_cannot_expand_package_version_advisories_paths_or_expiry(self):
        original = self.path.read_text()
        changes = [
            ('ecosystem = "crates.io"', 'ecosystem = "PyPI"'),
            ('name = "rustls"', 'name = "other"'),
            ('version = "0.23.45"', 'version = "0.23.46"'),
            ('RUSTSEC-2026-0346', 'RUSTSEC-2026-9999'),
            ('"Cargo.lock",', '"other/Cargo.lock",'),
            ('expires = "2026-11-14"', 'expires = "2027-11-14"'),
            ('schema_version = 1', 'schema_version = true'),
            ('rationale = "Accepted', 'unknown = true\nrationale = "Accepted'),
        ]
        for before, after in changes:
            with self.subTest(before=before):
                self.path.write_text(original.replace(before, after))
                with self.assertRaises(gate.InvalidPolicy):
                    self.load()

    def test_malformed_or_duplicate_policy_is_rejected(self):
        original = self.path.read_text()
        for contents in ('not valid TOML {', original + '\nschema_version = 1\n',
                         original + '\n[[exceptions]]\nname = "rustls"\n'):
            with self.subTest(contents=contents[:20]):
                self.path.write_text(contents)
                with self.assertRaises(ValueError):
                    self.load()

    def test_every_listed_lockfile_and_every_rustls_entry_must_match(self):
        policy = self.load()
        for filename in sorted(policy.lockfiles):
            path = self.root / filename
            original = path.read_text()
            variants = [
                original.replace('version = "0.23.45"', 'version = "0.23.46"'),
                original.replace(gate.REGISTRY, 'git+https://example.invalid/rustls'),
                original.replace(gate.CHECKSUM, '0' * 64),
                original.replace('name = "rustls"', 'name = "other"'),
                original + original.replace('version = "0.23.45"', 'version = "0.23.44"'),
            ]
            for index, contents in enumerate(variants):
                with self.subTest(lockfile=filename, variant=index):
                    path.write_text(contents)
                    with self.assertRaises(gate.InvalidPolicy):
                        gate.validate_preflight(policy, self.root)
            path.write_text(original)
            path.unlink()
            with self.assertRaises(gate.InvalidPolicy):
                gate.validate_preflight(policy, self.root)
            path.write_text(original)

    def test_deny_ignores_and_reasons_must_match_exactly(self):
        policy = self.load()
        path = self.root / "deny.toml"
        original = path.read_text()
        for contents in (
            '[advisories]\nignore = []\n',
            original.replace('RUSTSEC-2026-0346', 'RUSTSEC-2026-9999'),
            original.replace('RUSTSEC-2026-0346', 'RUSTSEC-2026-0347'),
            original.replace(policy.rationale, 'changed reason', 1),
            original.replace('reason =', 'expiry = "P31D", reason =', 1),
        ):
            with self.subTest(contents=contents[:30]):
                path.write_text(contents)
                with self.assertRaises(gate.InvalidPolicy):
                    gate.validate_preflight(policy, self.root)

    def test_cli_defaults_to_policy_under_selected_root_and_sanitizes_errors(self):
        output = io.StringIO()
        with mock.patch.object(gate, "datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 15, tzinfo=timezone.utc)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                self.assertEqual(gate.main(["--root", str(self.root)]), 0)
            self.assertIn("WARNING accepted-risk", output.getvalue())
            self.assertNotIn(str(self.root), output.getvalue())
            self.path.write_text('private fixture content {')
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                self.assertEqual(gate.main(["--root", str(self.root)]), 2)
            self.assertEqual(output.getvalue(), "ERROR: dependency_exception_invalid\n")


if __name__ == "__main__":
    unittest.main()
