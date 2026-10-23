"""Model-free installer checks. Successful installs exercise BSD tar/mv on macOS."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stark install test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "install space"
        self.commands = self.root / "commands"
        self.commands.mkdir()
        self.environment = {
            **os.environ,
            "HOME": str(self.root),
            "PATH": f"{self.commands}:{os.defpath}",
        }
        self.command("uname", 'case "$1" in -s) echo Darwin;; -m) echo arm64;; esac')
        self.command("sw_vers", "echo 26.4")

    def command(self, name, body):
        path = self.commands / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def run_install(self, *arguments, expected=0):
        result = subprocess.run(
            [
                "/bin/sh",
                str(INSTALLER),
                *map(str, arguments),
                "--prefix",
                str(self.prefix),
            ],
            env=self.environment,
            text=True,
            capture_output=True,
            cwd=self.root,
            check=False,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def bundle(self, version="0.1.0-test.1", *, doctor_ok=True, extra=None, app=False):
        archive = (
            self.root / f"bundle-{len(list(self.root.glob('bundle-*.tar.gz')))}.tar.gz"
        )
        members = {
            "stark/share/stark/INSTALL_FORMAT": ("1\n", 0o644),
            "stark/share/stark/VERSION": (version + "\n", 0o644),
            "stark/bin/stark": (
                "#!/bin/sh\nexit " + ("0\n" if doctor_ok else "1\n"),
                0o755,
            ),
            "stark/libexec/stark-score": ("#!/bin/sh\nexit 0\n", 0o755),
            "stark/libexec/stark-generate": ("#!/bin/sh\nexit 0\n", 0o755),
            "stark/libexec/python/bin/python3.13": ("#!/bin/sh\nexit 0\n", 0o755),
        }
        if app:
            members["stark/libexec/python/bin/python3.13"] = (
                "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
                0o755,
            )
            members["stark/share/stark/install_app.py"] = (
                (ROOT / "scripts/install_app.py").read_text(),
                0o644,
            )
            payload = self.root / (archive.stem + ".app-source") / "Stark.app"
            executable = payload / "Contents/MacOS/Stark"
            executable.parent.mkdir(parents=True)
            shutil.copyfile("/usr/bin/true", executable)
            executable.chmod(0o755)
            (payload / "Contents/Info.plist").write_bytes(
                plistlib.dumps(
                    {
                        "CFBundleIdentifier": "com.starkengine.menubar",
                        "CFBundleExecutable": "Stark",
                        "CFBundlePackageType": "APPL",
                        "CFBundleVersion": version,
                    }
                )
            )
            signed = subprocess.run(
                ["/usr/bin/codesign", "--force", "--sign", "-", payload],
                capture_output=True,
                check=False,
            )
            self.assertEqual(signed.returncode, 0, "synthetic app signing failed")
            for path in payload.rglob("*"):
                if path.is_file():
                    members[
                        "stark/app/Stark.app/" + path.relative_to(payload).as_posix()
                    ] = (
                        path.read_bytes(),
                        path.stat().st_mode & 0o777,
                    )
        with tarfile.open(archive, "w:gz") as output:
            for name, (text, mode) in members.items():
                entry = tarfile.TarInfo(name)
                data = text.encode() if isinstance(text, str) else text
                entry.size, entry.mode = len(data), mode
                output.addfile(entry, io.BytesIO(data))
            if extra:
                entry = tarfile.TarInfo(extra[0])
                if len(extra) > 1:
                    entry.type, entry.linkname = tarfile.SYMTYPE, extra[1]
                output.addfile(entry)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        Path(str(archive) + ".sha256").write_text(checksum + "  " + archive.name + "\n")
        return archive

    def test_help_requires_no_platform_or_tools(self):
        self.command("uname", "exit 99")
        self.assertIn("--from-source", self.run_install("--help").stdout)

    def test_rejects_other_platforms_before_installing(self):
        self.command("uname", "echo Linux")
        result = self.run_install("--version", "0.1.0", expected=1)
        self.assertIn("Apple silicon", result.stderr)
        self.assertFalse(self.prefix.exists())

    def test_rejects_intel_or_rosetta(self):
        self.command("uname", 'case "$1" in -s) echo Darwin;; -m) echo x86_64;; esac')
        self.assertIn(
            "Rosetta", self.run_install("--version", "0.1.0", expected=1).stderr
        )

    def test_rejects_older_macos(self):
        self.command("sw_vers", "echo 26.3.9")
        self.assertIn("26.4", self.run_install("--version", "0.1.0", expected=1).stderr)

    def test_rejects_ambiguous_and_malformed_arguments(self):
        for args in [
            ("--archive",),
            ("--prefix",),
            ("--uninstall", "--version", "0.1.0"),
            ("--version", "../0.1.0"),
            ("--version", "0.1.0\n../bad"),
            ("--version", "0.1.0", "--sha256", "x" * 64),
        ]:
            self.run_install(*args, expected=1)

    def test_checksum_required(self):
        archive = self.bundle()
        Path(str(archive) + ".sha256").unlink()
        self.assertIn(
            "checksum required",
            self.run_install("--archive", archive, expected=1).stderr,
        )

    def test_corrupt_download_does_not_replace_installation(self):
        archive = self.bundle()
        archive.write_bytes(archive.read_bytes() + b"corrupted")
        result = self.run_install("--archive", archive, expected=1)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse((self.prefix / "lib/stark/current").exists())

    @unittest.skipUnless(
        platform.system() == "Darwin", "uses BSD mv -h for installation"
    )
    def test_local_archive_replacement_after_hash_cannot_replace_verified_bytes(self):
        archive = self.bundle()
        replacement = self.bundle(version="0.1.0-malicious.1", doctor_ok=False)
        self.environment["TEST_EXTERNAL_ARCHIVE"] = str(archive)
        self.environment["TEST_REPLACEMENT_ARCHIVE"] = str(replacement)
        self.command(
            "shasum",
            '/usr/bin/shasum "$@"\n'
            '/bin/cp "$TEST_REPLACEMENT_ARCHIVE" "$TEST_EXTERNAL_ARCHIVE"',
        )
        self.run_install("--archive", archive)
        self.assertEqual(
            (self.prefix / "lib/stark/current").readlink(),
            Path("releases/0.1.0-test.1"),
        )

    def test_missing_release_is_actionable(self):
        self.command("curl", "exit 22")
        self.assertIn(
            "release bundle unavailable",
            self.run_install("--version", "0.1.0", expected=1).stderr,
        )

    def test_rejects_traversal_and_links_even_with_valid_checksum(self):
        for extra in [("stark/../../escape",), ("stark/alias", "/tmp"), ("outside",)]:
            archive = self.bundle(extra=extra)
            result = self.run_install("--archive", archive, expected=1)
            self.assertIn(
                "unsafe archive" if len(extra) == 1 else "archive links", result.stderr
            )
            self.assertFalse((self.prefix / "escape").exists())

    def test_collision_preserves_existing_command(self):
        (self.prefix / "bin").mkdir(parents=True)
        command = self.prefix / "bin/stark"
        command.write_text("existing command")
        self.assertIn(
            "another installation",
            self.run_install("--archive", self.bundle(), expected=1).stderr,
        )
        self.assertEqual(command.read_text(), "existing command")

    def test_rejects_managed_layout_redirects(self):
        base = self.prefix / "lib/stark"
        base.mkdir(parents=True)
        (base / "INSTALL_FORMAT").write_text("1\n")
        (base / "current").mkdir()
        self.assertIn(
            "managed version symlink",
            self.run_install("--archive", self.bundle(), expected=1).stderr,
        )
        (base / "current").rmdir()
        (base / "current").symlink_to("../../elsewhere")
        self.assertIn(
            "invalid current",
            self.run_install("--archive", self.bundle(), expected=1).stderr,
        )

    def test_invalid_bundle_version_cannot_escape_release_directory(self):
        archive = self.bundle("0.1.0\n../escape")
        self.assertIn(
            "invalid bundle version",
            self.run_install("--archive", archive, expected=1).stderr,
        )

    @unittest.skipUnless(
        platform.system() == "Darwin", "uses BSD mv -h for atomic symlink switching"
    )
    def test_install_reinstall_upgrade_and_uninstall_keep_models_and_other_commands(
        self,
    ):
        archive = self.bundle()
        self.run_install("--archive", archive)
        current = self.prefix / "lib/stark/current"
        replacement = self.bundle(extra=("stark/changed",))
        self.assertIn(
            "already installed with different contents",
            self.run_install("--archive", replacement, expected=1).stderr,
        )
        self.assertEqual(current.readlink(), Path("releases/0.1.0-test.1"))
        self.assertEqual(current.readlink(), Path("releases/0.1.0-test.1"))
        self.assertEqual(
            (self.prefix / "bin/stark").resolve(), current.resolve() / "bin/stark"
        )
        self.run_install("--archive", archive)
        self.run_install(
            "--archive", self.bundle("0.1.0-test.2", doctor_ok=False), expected=1
        )
        self.assertEqual(current.readlink(), Path("releases/0.1.0-test.1"))
        self.run_install("--archive", self.bundle("0.1.0-test.2"))
        self.assertEqual(current.readlink(), Path("releases/0.1.0-test.2"))
        self.assertTrue((self.prefix / "lib/stark/releases/0.1.0-test.1").is_dir())
        model = self.prefix / "models/keep"
        model.parent.mkdir()
        model.write_text("model placeholder")
        other = self.prefix / "bin/other"
        other.write_text("other command")
        self.run_install("--uninstall")
        self.assertFalse(current.parent.exists())
        self.assertEqual(model.read_text(), "model placeholder")
        self.assertEqual(other.read_text(), "other command")

    def test_uninstall_refuses_unmanaged_directory(self):
        base = self.prefix / "lib/stark"
        base.mkdir(parents=True)
        (base / "keep").write_text("user data")
        self.run_install("--uninstall", expected=1)
        self.assertTrue((base / "keep").exists())

    @unittest.skipUnless(platform.system() == "Darwin", "uses signed synthetic app")
    def test_combined_install_upgrade_reinstall_uninstall_preserves_models(self):
        first = self.bundle("0.1.2", app=True)
        self.run_install("--archive", first)
        target = self.root / "Applications/Stark.app"
        self.assertTrue(target.is_dir())
        self.assertFalse(target.is_symlink())
        self.run_install("--archive", first)
        self.run_install("--archive", self.bundle("0.1.3", app=True))
        self.assertEqual(
            (self.prefix / "lib/stark/current").readlink(), Path("releases/0.1.3")
        )
        info = plistlib.loads((target / "Contents/Info.plist").read_bytes())
        self.assertEqual(info["CFBundleVersion"], "0.1.3")
        model = self.root / "Library/Application Support/Stark/models/keep"
        model.parent.mkdir(parents=True)
        model.write_text("private model placeholder")
        self.run_install("--uninstall")
        self.assertFalse(target.exists())
        self.assertFalse((target.parent / ".Stark.app.installation.json").exists())
        self.assertEqual(model.read_text(), "private model placeholder")

    @unittest.skipUnless(platform.system() == "Darwin", "uses signed synthetic app")
    def test_combined_app_directory_override_and_collision(self):
        archive = self.bundle("0.1.2", app=True)
        apps = self.root / "custom apps with spaces"
        target = apps / "Stark.app"
        target.mkdir(parents=True)
        (target / "keep").write_text("unrelated app")
        self.run_install("--archive", archive, "--app-dir", apps, expected=1)
        self.assertFalse((self.prefix / "lib/stark/current").exists())
        self.assertEqual((target / "keep").read_text(), "unrelated app")
        shutil.rmtree(target)
        self.run_install("--archive", archive, "--app-dir", apps)
        self.assertTrue(target.is_dir())
        self.assertFalse((self.root / "Applications/Stark.app").exists())
        self.run_install("--uninstall")
        self.assertFalse(target.exists())

    @unittest.skipUnless(platform.system() == "Darwin", "uses signed synthetic app")
    def test_cli_only_downgrade_does_not_orphan_managed_app(self):
        self.run_install("--archive", self.bundle("0.1.2", app=True))
        result = self.run_install("--archive", self.bundle("0.1.1"), expected=1)
        self.assertIn("CLI-only bundle", result.stderr)
        self.assertEqual(
            (self.prefix / "lib/stark/current").readlink(), Path("releases/0.1.2")
        )
        self.assertTrue((self.root / "Applications/Stark.app").is_dir())
        self.run_install("--uninstall")


if __name__ == "__main__":
    unittest.main()
