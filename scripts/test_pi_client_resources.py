"""Model-free checks for locked Pi resources staged into the app bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("pi_client_resources.py")
SPEC = importlib.util.spec_from_file_location("pi_client_resources", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pi_resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pi_resources)


class PiClientResourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stark-pi-resources-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.resources = self.root / "Resources"
        self.source.mkdir(mode=0o755)
        self.resources.mkdir(mode=0o755)
        self.payload = {
            "package.json": b'{"name":"stark-agent-clients","type":"module","private":true}\n',
            "LICENSE": b"MIT License\n",
            "src/config.mjs": b"export const config = true;\n",
            "src/guarded-fetch.mjs": b"export const guard = true;\n",
            "extensions/stark-stream-guard.mjs": b"export default function guard() {}\n",
            "src/desktop-setup.mjs": b"export const setup = true;\n",
            "src/desktop-launch.mjs": b"export const launch = true;\n",
            "src/default-profile.mjs": b"export const defaults = true;\n",
        }
        for name, data in self.payload.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o644)
        self.lock = self.root / "PiClient.lock.json"
        self.write_lock()

    def tearDown(self):
        self.temporary.cleanup()

    def write_lock(self, *, files=None, revision="a" * 40, repository=None):
        document = {
            "schema_version": 1,
            "repository": repository or pi_resources.REPOSITORY,
            "revision": revision,
            "files_sha256": files
            if files is not None
            else {
                name: hashlib.sha256(data).hexdigest()
                for name, data in self.payload.items()
            },
        }
        self.lock.write_text(json.dumps(document), encoding="utf-8")
        self.lock.chmod(0o644)

    def install(self, source=None):
        return pi_resources.install_pi_client(
            self.resources,
            source=self.source if source is None else source,
            lockfile=self.lock,
        )

    def assert_code(self, code, function):
        with self.assertRaises(pi_resources.PiClientResourceError) as caught:
            function()
        self.assertEqual(str(caught.exception), code)

    def test_copies_exact_public_file_closure_and_reports_hashes(self):
        result = self.install()
        self.assertEqual(result["repository"], pi_resources.REPOSITORY)
        self.assertEqual(result["revision"], "a" * 40)
        self.assertEqual(set(result["files_sha256"]), set(pi_resources.FILES))
        self.assertEqual(
            set(result["resources_sha256"]),
            {f"PiClient/{name}" for name in pi_resources.FILES},
        )
        for name, data in self.payload.items():
            staged = self.resources / "PiClient" / name
            self.assertEqual(staged.read_bytes(), data)
            self.assertEqual(staged.stat().st_mode & 0o777, 0o644)
        self.assertFalse((self.resources / "PiClient/node_modules").exists())

    def test_source_hash_mismatch_fails_before_staging(self):
        (self.source / "src/config.mjs").write_text("changed\n", encoding="utf-8")
        self.assert_code("pi_client_hash_mismatch", self.install)
        self.assertFalse((self.resources / "PiClient").exists())

    def test_default_snapshot_is_staged_from_lock_directory(self):
        snapshot = self.lock.parent / "PiClient"
        shutil.copytree(self.source, snapshot)
        result = pi_resources.install_pi_client(
            self.resources,
            source=None,
            lockfile=self.lock,
        )
        self.assertEqual(result["revision"], "a" * 40)
        self.assertEqual(
            {
                name: hashlib.sha256(
                    (self.resources / "PiClient" / name).read_bytes()
                ).hexdigest()
                for name in pi_resources.FILES
            },
            result["files_sha256"],
        )

    def test_missing_default_snapshot_has_fixed_error(self):
        self.assert_code(
            "pi_client_snapshot_missing",
            lambda: pi_resources.install_pi_client(
                self.resources, source=None, lockfile=self.lock
            ),
        )

    def test_committed_snapshot_stages_in_a_clean_consumer_without_source_checkout(
        self,
    ):
        # Exercise the real lock and bytes, then relocate them into a clean
        # consumer that cannot find an adjacent integration checkout.
        consumer = self.root / "clean-stark" / "apps/menu-bar/Resources"
        consumer.mkdir(parents=True)
        shutil.copy2(pi_resources.LOCK_PATH, consumer / "PiClient.lock.json")
        shutil.copytree(
            pi_resources.LOCK_PATH.parent / "PiClient", consumer / "PiClient"
        )
        result = pi_resources.install_pi_client(
            self.resources, source=None, lockfile=consumer / "PiClient.lock.json"
        )
        self.assertEqual(
            result["files_sha256"], pi_resources.read_lock()["files_sha256"]
        )
        self.assertEqual(set(result["files_sha256"]), set(pi_resources.FILES))

    def test_source_symlink_is_rejected(self):
        path = self.source / "LICENSE"
        path.unlink()
        path.symlink_to(self.source / "package.json")
        self.assert_code("pi_client_source_invalid", self.install)
        self.assertFalse((self.resources / "PiClient").exists())

    def test_source_root_symlink_is_rejected(self):
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self.assert_code("pi_client_source_invalid", lambda: self.install(alias))

    def test_writable_or_executable_source_file_is_rejected(self):
        (self.source / "LICENSE").chmod(0o755)
        self.assert_code("pi_client_source_invalid", self.install)

    def test_locked_file_set_must_be_complete(self):
        self.write_lock(files={})
        self.assert_code("pi_client_lock_invalid", self.install)

    def test_lock_repository_and_revision_are_validated(self):
        self.write_lock(repository="https://example.invalid/repo")
        self.assert_code("pi_client_lock_invalid", self.install)
        self.write_lock(revision="not-a-revision")
        self.assert_code("pi_client_lock_invalid", self.install)

    def test_invalid_package_manifest_is_rejected(self):
        self.payload["package.json"] = (
            b'{"name":"other","type":"module","private":true}\n'
        )
        self.write_lock()
        (self.source / "package.json").write_bytes(self.payload["package.json"])
        self.assert_code("pi_client_package_invalid", self.install)

    def test_existing_destination_is_not_overwritten(self):
        target = self.resources / "PiClient"
        target.mkdir()
        marker = target / "marker"
        marker.write_text("keep", encoding="utf-8")
        self.assert_code("pi_client_destination_invalid", self.install)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_symlink_destination_is_not_followed(self):
        target = self.resources / "PiClient"
        outside = self.root / "outside"
        outside.mkdir()
        target.symlink_to(outside, target_is_directory=True)
        self.assert_code("pi_client_destination_invalid", self.install)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
