"""Distribution version selection and workflow-output contracts."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import release_version


class ReleaseVersionTests(unittest.TestCase):
    def test_bumps_highest_stable_tag_numerically(self):
        tags = ["v0.1.2", "v0.9.9", "v0.10.3", "v9.0.0-rc.1", "latest", "v01.0.0"]
        for bump, expected in (
            ("major", "1.0.0"),
            ("minor", "0.11.0"),
            ("patch", "0.10.4"),
        ):
            with self.subTest(bump=bump):
                self.assertEqual(
                    release_version.next_version("0.1.0", tags, bump), expected
                )

    def test_source_version_is_lower_bound_and_empty_history_is_supported(self):
        self.assertEqual(
            release_version.next_version("2.3.4", ["v1.0.0"], "patch"), "2.3.5"
        )
        self.assertEqual(release_version.next_version("0.1.0", [], "minor"), "0.2.0")

    def test_invalid_inputs_fail(self):
        for version, bump in (
            ("0.1.0-rc.1", "patch"),
            ("01.2.3", "patch"),
            ("1.2.3", "other"),
        ):
            with (
                self.subTest(version=version, bump=bump),
                self.assertRaises(ValueError),
            ):
                release_version.next_version(version, [], bump)

    def test_selection_emits_output_without_creating_a_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*arguments):
                return subprocess.run(
                    ["git", *arguments],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout

            git("init", "--quiet")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-m",
                "test",
            )
            git("tag", "v0.1.2")
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.1.0"\n')
            output = root / "output"
            with (
                patch.object(release_version, "ROOT", root),
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
                patch("sys.argv", ["release_version.py", "--bump", "patch"]),
            ):
                self.assertEqual(release_version.main(), 0)
            self.assertEqual(output.read_text(), "version=0.1.3\n")
            self.assertEqual(git("tag", "--list"), "v0.1.2\n")


if __name__ == "__main__":
    unittest.main()
