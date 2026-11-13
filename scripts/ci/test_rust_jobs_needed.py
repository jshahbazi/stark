"""Check which changes may skip the macOS Rust jobs."""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("rust-jobs-needed.sh")
ROOT = Path(__file__).resolve().parents[2]


def classify(*paths: str) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--classify"],
        input="".join(path + "\0" for path in paths).encode(),
        capture_output=True,
        check=True,
    )
    return result.stdout.decode().strip()


class ClassificationTests(unittest.TestCase):
    def test_rust_inputs_and_unknown_paths_run_the_jobs(self):
        for path in (
            "crates/stark-engine/src/lib.rs",
            "crates/stark-engine/src/kernel.metal",
            "vendor/cxx/include/cxx.h",
            "vendor/lily-core/src/lib.rs",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "clippy.toml",
            ".cargo/config.toml",
            ".config/nextest.toml",
            ".github/workflows/ci.yml",
            ".github/actions/rust-cache/action.yml",
            "scripts/ci/rust-cache-key.sh",
            "scripts/ci/rust-jobs-needed.sh",
            "scripts/test_generation_process.py",
            # Rust sources include converter data files.
            "tools/converter/stark_convert/guided_memory_profiles.json",
            "tools/converter/tests/fixtures/qwen3_4b_chat_template.jinja",
            "new-top-level/file.txt",
            "unknown-root-file",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify(path), "rust=true")

    def test_paths_outside_the_rust_jobs_skip_them(self):
        for path in (
            "docs/CI.md",
            "docs/model-experiments/attempt.md",
            "website/public/index.html",
            "apps/menu-bar/Sources/App.swift",
            "provenance/UPSTREAM.md",
            "README.md",
            "AGENTS.md",
            "LICENSE",
            "LICENSE.mlx-reference",
            "install.sh",
            "scripts/build_menubar.py",
            "scripts/test_menubar_controller.py",
            "scripts/licenses/distribution/manifest.json",
            "tools/converter/stark_convert/convert.py",
            "tools/converter/README.md",
            "tools/converter/pyproject.toml",
            "tools/converter/uv.lock",
            "tools/uv.lock",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify(path), "rust=false")

    def test_one_rust_input_among_other_paths_runs_the_jobs(self):
        self.assertEqual(
            classify("docs/CI.md", "crates/stark-score/src/main.rs", "README.md"),
            "rust=true",
        )
        self.assertEqual(classify("docs/CI.md", "README.md"), "rust=false")

    def test_no_changed_paths_skip_the_jobs(self):
        self.assertEqual(classify(), "rust=false")

    def test_markdown_in_a_rust_directory_runs_the_jobs(self):
        self.assertEqual(classify("crates/stark-engine/README.md"), "rust=true")

    def test_files_rust_sources_include_run_the_jobs(self):
        """A file a crate reads from outside crates/ and vendor/ must not be skippable."""
        literal = re.compile(r'"(/?(?:\.\./)+[^"\n]+)"')
        included = set()
        for source in sorted((ROOT / "crates").rglob("*.rs")):
            crate = next(p for p in source.parents if (p / "Cargo.toml").is_file())
            for reference in literal.findall(source.read_text(errors="replace")):
                # concat!(env!("CARGO_MANIFEST_DIR"), "/../..") is crate-relative.
                start = crate if reference.startswith("/") else source.parent
                target = (start / reference.lstrip("/")).resolve()
                if not target.is_file() or ROOT not in target.parents:
                    continue
                relative = target.relative_to(ROOT).as_posix()
                if not relative.startswith(("crates/", "vendor/")):
                    included.add(relative)
        self.assertTrue(included, "expected Rust sources to include converter data")
        for path in sorted(included):
            with self.subTest(path=path):
                self.assertEqual(classify(path), "rust=true")


class BaseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "work"
        self.root.mkdir()
        self.git("init", "--quiet", "--initial-branch=main")
        self.commit({"docs/a.md": "a\n", "crates/x/src/lib.rs": "pub fn x() {}\n"})

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *arguments,
            ],
            cwd=self.root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

    def commit(self, files: dict[str, str]) -> str:
        for name, contents in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        self.git("add", "--all")
        self.git("commit", "--quiet", "--message", "change")
        return self.git("rev-parse", "HEAD")

    def merge_of(self, files: dict[str, str]) -> None:
        """Leave HEAD on a merge commit shaped like a pull request merge ref."""
        self.git("checkout", "--quiet", "-b", "topic")
        self.commit(files)
        self.git("checkout", "--quiet", "main")
        self.commit({"docs/base-moved.md": "base\n"})
        self.git("merge", "--quiet", "--no-ff", "--message", "merge", "topic")

    def detect(self, **environment: str) -> str:
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=self.root,
            env={**os.environ, **environment},
            capture_output=True,
            check=True,
        )
        return result.stdout.decode().strip()

    def test_pull_request_compares_the_merge_with_the_base_tip(self):
        self.merge_of({"docs/b.md": "b\n"})
        self.assertEqual(self.detect(EVENT_NAME="pull_request"), "rust=false")

    def test_pull_request_with_a_rust_change_runs_the_jobs(self):
        self.merge_of({"crates/x/src/lib.rs": "pub fn y() {}\n"})
        self.assertEqual(self.detect(EVENT_NAME="pull_request"), "rust=true")

    def test_pull_request_without_a_merge_commit_runs_the_jobs(self):
        self.commit({"docs/b.md": "b\n"})
        self.assertEqual(self.detect(EVENT_NAME="pull_request"), "rust=true")

    def test_moving_a_file_out_of_a_rust_directory_runs_the_jobs(self):
        self.git("checkout", "--quiet", "-b", "topic")
        self.git("mv", "crates/x/src/lib.rs", "docs/lib.rs")
        self.git("commit", "--quiet", "--message", "move")
        self.git("checkout", "--quiet", "main")
        self.git("merge", "--quiet", "--no-ff", "--message", "merge", "topic")
        self.assertEqual(self.detect(EVENT_NAME="pull_request"), "rust=true")

    def push_base(self) -> str:
        """Serve the repository as `origin` so the base can be fetched by hash."""
        base = self.git("rev-parse", "HEAD")
        self.git("config", "uploadpack.allowAnySHA1InWant", "true")
        self.git("remote", "add", "origin", self.root.as_uri())
        return base

    def test_push_compares_with_the_last_successful_commit(self):
        base = self.push_base()
        self.commit({"docs/b.md": "b\n"})
        self.commit({"apps/menu-bar/App.swift": "// app\n"})
        self.assertEqual(self.detect(EVENT_NAME="push", BASE_SHA=base), "rust=false")

    def test_push_sees_a_rust_change_from_an_earlier_cancelled_run(self):
        base = self.push_base()
        self.commit({"crates/x/src/lib.rs": "pub fn y() {}\n"})
        self.commit({"docs/b.md": "b\n"})
        self.assertEqual(self.detect(EVENT_NAME="push", BASE_SHA=base), "rust=true")

    def test_push_without_a_usable_base_runs_the_jobs(self):
        self.push_base()
        self.commit({"docs/b.md": "b\n"})
        for base in ("", "not-a-hash", "0" * 40, "abc123"):
            with self.subTest(base=base):
                self.assertEqual(
                    self.detect(EVENT_NAME="push", BASE_SHA=base), "rust=true"
                )

    def test_other_events_run_the_jobs(self):
        self.commit({"docs/b.md": "b\n"})
        self.assertEqual(self.detect(EVENT_NAME="workflow_dispatch"), "rust=true")
        self.assertEqual(self.detect(EVENT_NAME=""), "rust=true")


if __name__ == "__main__":
    unittest.main()
