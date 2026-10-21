"""Model-free contracts for the relocatable standalone command."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).with_name("stark")
CONVERTER_ENTRYPOINT = 'import sys; from stark_convert.cli import main; sys.argv[0] = "stark-convert"; raise SystemExit(main())'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stark cli ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bundle = self.directory / "extracted bundle"
        (self.bundle / "bin").mkdir(parents=True)
        (self.bundle / "libexec" / "python" / "bin").mkdir(parents=True)
        (self.bundle / "share" / "stark").mkdir(parents=True)
        shutil.copy2(LAUNCHER, self.bundle / "bin" / "stark")
        (self.bundle / "bin" / "stark").chmod(0o755)
        (self.bundle / "share" / "stark" / "VERSION").write_text("0.1.0-preview.1\n")
        self.environment = os.environ.copy()
        self.environment.update(
            PYTHONPATH="/unrelated/python", PYTHONHOME="/unrelated/home"
        )

    def run_cli(self, *arguments, launcher=None, expected=0, stdin=None):
        result = subprocess.run(
            [str(launcher or self.bundle / "bin" / "stark"), *arguments],
            input=stdin,
            cwd=self.directory,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def fake_executable(self, destination, body):
        destination.write_text("#!/bin/sh\nset -eu\n" + body)
        destination.chmod(0o755)

    def fake_argument_echo(self, destination):
        self.fake_executable(destination, "printf '%s\\n' \"$@\"\ncat\n")

    def fake_platform(self, system="Darwin", architecture="arm64", version="26.4"):
        commands = self.directory / "system commands"
        commands.mkdir(exist_ok=True)
        self.fake_executable(
            commands / "uname",
            f'case "$1" in -s) printf "%s\\n" "{system}" ;; -m) printf "%s\\n" "{architecture}" ;; esac\n',
        )
        self.fake_executable(commands / "sw_vers", f'printf "%s\\n" "{version}"\n')
        self.environment["PATH"] = str(commands) + os.pathsep + os.defpath

    def test_help_does_not_require_runtime(self):
        for arguments in ([], ["--help"], ["help"], ["-h"]):
            self.assertIn("Usage: stark COMMAND", self.run_cli(*arguments).stdout)

    def test_version_and_unknown_command(self):
        self.assertEqual(self.run_cli("--version").stdout, "stark 0.1.0-preview.1\n")
        self.assertEqual(self.run_cli("version").stdout, "stark 0.1.0-preview.1\n")
        self.assertIn("unknown command", self.run_cli("unsupported", expected=2).stderr)
        self.run_cli("version", "extra", expected=1)

    def test_converter_uses_isolated_python_and_preserves_arguments(self):
        self.fake_argument_echo(
            self.bundle / "libexec" / "python" / "bin" / "python3.13"
        )
        result = self.run_cli("convert", "local model", "--output", "output folder")
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "-I",
                "-B",
                "-c",
                CONVERTER_ENTRYPOINT,
                "local model",
                "--output",
                "output folder",
            ],
        )

    def test_converter_alias_resolves_relative_and_absolute_links(self):
        self.fake_argument_echo(
            self.bundle / "libexec" / "python" / "bin" / "python3.13"
        )
        (self.bundle / "bin" / "stark-convert").symlink_to("stark")
        link = self.directory / "convert link"
        link.symlink_to(self.bundle / "bin" / "stark-convert")
        direct = self.run_cli("--help", launcher=self.bundle / "bin" / "stark-convert")
        self.assertEqual(
            direct.stdout.splitlines(),
            ["-I", "-B", "-c", CONVERTER_ENTRYPOINT, "--help"],
        )
        # Aliases are selected by the invoked name. An arbitrary symlink keeps
        # the regular command syntax while retaining correct bundle resolution.
        self.assertEqual(
            self.run_cli("version", launcher=link).stdout, "stark 0.1.0-preview.1\n"
        )

    def test_workers_preserve_argument_boundaries_and_stdio(self):
        for command, name in (("score", "stark-score"), ("generate", "stark-generate")):
            self.fake_argument_echo(self.bundle / "libexec" / name)
            result = self.run_cli(
                command, "--model-path", "model folder", stdin="synthetic input\n"
            )
            self.assertEqual(
                result.stdout, "--model-path\nmodel folder\nsynthetic input\n"
            )

    def test_relocation_through_current_and_path_command(self):
        prefix = self.directory / "install prefix"
        release = prefix / "lib" / "stark" / "releases" / "0.1.0-preview.1"
        release.parent.mkdir(parents=True)
        self.bundle.rename(release)
        self.bundle = release
        (prefix / "lib" / "stark" / "current").symlink_to("releases/0.1.0-preview.1")
        (prefix / "bin").mkdir()
        (prefix / "bin" / "stark").symlink_to("../lib/stark/current/bin/stark")
        self.environment["PATH"] = str(prefix / "bin") + os.pathsep + os.defpath
        result = subprocess.run(
            ["sh", "-c", "stark version"],
            cwd=self.directory,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "stark 0.1.0-preview.1\n")

    def test_missing_components_have_actionable_errors(self):
        self.assertIn(
            "bundled Python is missing", self.run_cli("convert", expected=1).stderr
        )
        self.assertIn(
            "native worker is missing", self.run_cli("score", expected=1).stderr
        )
        (self.bundle / "share" / "stark" / "VERSION").unlink()
        self.assertIn(
            "bundle version is missing", self.run_cli("version", expected=1).stderr
        )

    def test_entrypoints_work_with_only_system_commands_on_path(self):
        self.environment["PATH"] = "/usr/bin:/bin"
        self.assertEqual(self.run_cli("version").stdout, "stark 0.1.0-preview.1\n")
        self.fake_argument_echo(
            self.bundle / "libexec" / "python" / "bin" / "python3.13"
        )
        self.assertEqual(
            self.run_cli("convert", "--version").stdout.splitlines(),
            ["-I", "-B", "-c", CONVERTER_ENTRYPOINT, "--version"],
        )

    def test_converter_entrypoint_sets_argv0_and_preserves_user_arguments(self):
        import sys

        runtime = self.directory / "converter runtime"
        package = runtime / "stark_convert"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text(
            "import json, sys\ndef main():\n    print(json.dumps(sys.argv))\n    return 0\n"
        )
        prefix = f"import sys; sys.path.insert(0, {str(runtime)!r}); "
        shim = (
            "import os, sys; args = sys.argv[1:]; "
            f"os.execv({sys.executable!r}, [{sys.executable!r}, '-I', '-B', '-c', "
            + repr(prefix)
            + "+ args[3], *args[4:]])"
        )
        python = self.bundle / "libexec" / "python" / "bin" / "python3.13"
        python.write_text(f"#!{sys.executable} -I\n" + shim + "\n")
        python.chmod(0o755)
        result = self.run_cli(
            "convert", "model folder", "--output", "output folder", "--version"
        )
        self.assertEqual(
            json.loads(result.stdout),
            ["stark-convert", "model folder", "--output", "output folder", "--version"],
        )

    def test_doctor_rejects_platform_and_missing_components(self):
        for system, architecture, version, message in (
            ("Linux", "aarch64", "26.4", "Apple silicon Mac"),
            ("Darwin", "x86_64", "26.4", "outside Rosetta"),
            ("Darwin", "arm64", "26.3.1", "26.4 or newer"),
            ("Darwin", "arm64", "15.7", "26.4 or newer"),
            ("Darwin", "arm64", "26.4", "bundled Python is missing"),
        ):
            with self.subTest(
                system=system, architecture=architecture, version=version
            ):
                self.fake_platform(system, architecture, version)
                self.assertIn(message, self.run_cli("doctor", expected=1).stderr)

    def test_doctor_checks_workers_without_relaying_output(self):
        self.fake_platform()
        python = self.bundle / "libexec" / "python" / "bin" / "python3.13"
        # Run doctor's embedded Python using the test interpreter. A synthetic
        # package on its isolated sys.path supplies only converter metadata.
        import sys

        runtime = self.directory / "doctor runtime"
        runtime.mkdir()
        package = runtime / "stark_convert"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text("")
        metadata = runtime / "stark_convert-0.9.1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: stark-convert\nVersion: 0.9.1\n")
        # The fake interpreter executes the supplied snippet after adding its
        # synthetic runtime; Python environment variables remain ignored.
        prefix = f"import sys; sys.path.insert(0, {str(runtime)!r}); "
        shim = (
            "import json, os, sys; "
            "args = sys.argv[1:]; "
            f"os.execv({sys.executable!r}, [{sys.executable!r}, '-I', '-B', '-c', "
            + repr(prefix)
            + "+ args[3], *args[4:]])"
        )
        # Write a Python shim with an absolute interpreter; -I on the real
        # invocation prevents the deliberately invalid host environment leaking.
        python.write_text(f"#!{sys.executable} -I\n" + shim + "\n")
        python.chmod(0o755)
        for name in ("stark-score", "stark-generate"):
            self.fake_executable(
                self.bundle / "libexec" / name,
                "printf '%s\\n' '{\"ready\":false,\"error\":\"invalid_arguments\"}'\nprintf 'discarded diagnostic' >&2\nexit 1\n",
            )
        result = self.run_cli("doctor")
        self.assertIn("Installation checks passed", result.stdout)
        self.assertNotIn("discarded diagnostic", result.stdout + result.stderr)
        self.fake_executable(
            self.bundle / "libexec" / "stark-generate",
            "printf 'unexpected diagnostic'\nexit 1\n",
        )
        result = self.run_cli("doctor", expected=1)
        self.assertNotIn("unexpected diagnostic", result.stdout + result.stderr)
        self.assertIn("check failed", result.stderr)

    def test_uninstall_requires_installed_layout(self):
        self.assertIn("extracted bundle", self.run_cli("uninstall", expected=1).stderr)

    def test_uninstall_delegates_the_exact_installed_prefix(self):
        prefix = self.directory / "install prefix"
        release = prefix / "lib" / "stark" / "releases" / "0.1.0-preview.1"
        release.parent.mkdir(parents=True)
        self.bundle.rename(release)
        self.bundle = release
        (prefix / "lib" / "stark" / "INSTALL_FORMAT").write_text("1\n")
        self.fake_argument_echo(release / "share" / "stark" / "install.sh")
        self.assertEqual(
            self.run_cli("uninstall").stdout.splitlines(),
            ["--uninstall", "--prefix", str(prefix.resolve())],
        )


if __name__ == "__main__":
    unittest.main()
