"""Check cache compatibility boundaries with a synthetic tracked source tree."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

KEY_SCRIPT = Path(__file__).with_name("rust-cache-key.sh")


class RustCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = {
            **os.environ,
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
            "ImageOS": "ubuntu24",
            "ImageVersion": "image-1",
            "CARGO_INCREMENTAL": "0",
        }
        self.git("init", "--quiet")
        files = {
            "Cargo.toml": "[workspace]\n",
            "Cargo.lock": "version = 4\n",
            "rust-toolchain.toml": '[toolchain]\nchannel = "1.98.1"\n',
            ".cargo/config.toml": "[env]\n",
            ".github/workflows/ci.yml": "name: CI\n",
            ".github/actions/rust-cache/action.yml": "name: Cache\n",
            "scripts/ci/rust-cache-key.sh": KEY_SCRIPT.read_text(),
            "crates/stark-engine/src/lib.rs": "pub fn engine() {}\n",
            "crates/stark-engine/build/main.rs": "fn main() {}\n",
            "crates/stark-engine/src/kernel.metal": "kernel void f() {}\n",
            "vendor/cxx/include/cxx.h": "// bridge\n",
            "crates/legacy/download-manager/examples/wasm/Cargo.lock": "version = 4\n",
        }
        for name, contents in files.items():
            self.write(name, contents)
        self.git("add", ".")
        tools = self.root / "fake-tools"
        tools.mkdir()
        for name in ("rustc", "cargo", "cc", "c++", "xcodebuild", "xcrun"):
            tool = tools / name
            tool.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$0 $*" "${CACHE_TEST_COMPILER:-v1}"\n'
            )
            tool.chmod(0o755)
        self.environment["PATH"] = f"{tools}{os.pathsep}{os.environ['PATH']}"

    def git(self, *arguments):
        subprocess.run(
            ["git", *arguments], cwd=self.root, check=True, capture_output=True
        )

    def write(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def key(self, scope="cxx-test-cpu-grammar", metal="false"):
        result = subprocess.run(
            ["bash", str(KEY_SCRIPT), scope, metal],
            cwd=self.root,
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_unchanged_inputs_reuse_exact_cache(self):
        self.assertEqual(self.key(), self.key())

    def test_rust_edit_reuses_dependencies_with_new_exact_key(self):
        before = self.key()
        self.write("crates/stark-engine/src/lib.rs", "pub fn engine() { } // changed\n")
        after = self.key()
        self.assertEqual(before["restore-prefix"], after["restore-prefix"])
        self.assertNotEqual(before["key"], after["key"])

    def test_native_and_configuration_edits_invalidate_fallback(self):
        for name in (
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            ".github/workflows/ci.yml",
            ".github/actions/rust-cache/action.yml",
            "scripts/ci/rust-cache-key.sh",
            "crates/stark-engine/build/main.rs",
            "crates/stark-engine/src/kernel.metal",
            "vendor/cxx/include/cxx.h",
            "crates/legacy/download-manager/examples/wasm/Cargo.lock",
        ):
            with self.subTest(path=name):
                before = self.key()
                with (self.root / name).open("a") as output:
                    output.write("\nchanged\n")
                self.assertNotEqual(
                    before["restore-prefix"], self.key()["restore-prefix"]
                )

    def test_new_native_file_invalidates_fallback(self):
        before = self.key()
        self.write("vendor/cxx/new.cc", "// new native source\n")
        self.git("add", "vendor/cxx/new.cc")
        self.assertNotEqual(before["restore-prefix"], self.key()["restore-prefix"])

    def test_tracked_directory_symlinks_are_hashed_without_following_them(self):
        link = self.root / "vendor/cxx/gen"
        link.symlink_to("include", target_is_directory=True)
        self.git("add", "vendor/cxx/gen")
        before = self.key()
        link.unlink()
        link.symlink_to("different-target", target_is_directory=True)
        self.assertNotEqual(before["restore-prefix"], self.key()["restore-prefix"])

    def test_docs_and_untracked_runtime_files_do_not_affect_cache(self):
        before = self.key()
        self.write("README.md", "Documentation\n")
        self.git("add", "README.md")
        self.write("artifacts/private.txt", "synthetic untracked fixture\n")
        self.assertEqual(before, self.key())

    def test_job_feature_scopes_do_not_share_fallbacks(self):
        self.assertNotEqual(
            self.key("scoring-release-default")["restore-prefix"],
            self.key("scoring-release-optimized")["restore-prefix"],
        )

    def test_runner_compiler_and_build_flags_invalidate_fallback(self):
        for name in (
            "ImageVersion",
            "RUNNER_ARCH",
            "CACHE_TEST_COMPILER",
            "RUSTFLAGS",
            "CARGO_PROFILE_TEST_OPT_LEVEL",
            "CARGO_BUILD_RUSTFLAGS",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
            "LDFLAGS",
            "BUILD_CLEAN",
        ):
            with self.subTest(variable=name):
                before = self.key()
                self.environment[name] = "changed"
                self.assertNotEqual(
                    before["restore-prefix"], self.key()["restore-prefix"]
                )

    def test_macos_sdk_and_metal_are_fingerprinted(self):
        self.environment["RUNNER_OS"] = "macOS"
        cpu = self.key(metal="false")
        metal = self.key(metal="true")
        self.assertNotEqual(cpu["restore-prefix"], metal["restore-prefix"])
        self.environment["MACOSX_DEPLOYMENT_TARGET"] = "27.0"
        self.assertNotEqual(
            metal["restore-prefix"], self.key(metal="true")["restore-prefix"]
        )

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.key(scope="../invalid")

    def test_missing_tracked_input_fails_closed(self):
        (self.root / "Cargo.lock").unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            self.key()

    def test_failed_compiler_probe_fails_closed(self):
        (self.root / "fake-tools/cc").write_text("#!/bin/sh\nexit 1\n")
        with self.assertRaises(subprocess.CalledProcessError):
            self.key()


if __name__ == "__main__":
    unittest.main()
