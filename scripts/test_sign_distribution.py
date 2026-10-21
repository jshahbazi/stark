"""Model-free input-integrity, signing-policy, and publication checks."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "sign_distribution", SCRIPTS / "sign_distribution.py"
)
assert SPEC is not None and SPEC.loader is not None
signing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(signing)

SOURCE = "a" * 40
IDENTITY = {"sha1": "B" * 40, "team_id": "ABCDEFGHIJ"}
UUID = "12345678-1234-1234-1234-123456789abc"
NATIVE_NAMES = (
    "libexec/stark-score",
    "libexec/stark-generate",
    "libexec/python/bin/python3.13",
    "libexec/python/lib/libpython3.13.dylib",
    "libexec/python/lib/python3.13/site-packages/example/native.so",
)


def fixture(root: Path) -> tuple[Path, str, dict]:
    bundle = root / "source/stark"
    for name in NATIVE_NAMES:
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"\xcf\xfa\xed\xfe" + bytes(8) + (2).to_bytes(4, "little") + name.encode()
        )
        path.chmod(0o755)
    share = bundle / "share/stark"
    share.mkdir(parents=True)
    (share / "VERSION").write_text("0.1.0\n")
    (bundle / "module.py").write_text("# synthetic non-native file\n")
    manifest = {
        "schema_version": 1,
        "version": "0.1.0",
        "target": signing.distribution.TARGET,
        "minimum_macos": signing.distribution.MIN_MACOS,
        "source": {"revision": SOURCE, "dirty": False},
        "files_sha256": signing.distribution.file_manifest(bundle),
    }
    (bundle / signing.MANIFEST).write_text(json.dumps(manifest))
    archive = root / "input.tar.gz"
    digest = signing.distribution.create_archive(bundle, archive)
    return archive, digest, manifest


def result(stdout: bytes = b"", stderr: bytes = b""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)


class SigningTests(unittest.TestCase):
    def test_verified_input_retains_its_original_source_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, digest, original = fixture(root)
            extracted, manifest = signing.extract_verified(
                archive, digest, SOURCE, root / "extracted"
            )
            self.assertEqual(manifest, original)
            self.assertEqual((extracted / "share/stark/VERSION").read_text(), "0.1.0\n")

    def test_checksum_is_required_before_archive_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, _, _ = fixture(root)
            with self.assertRaisesRegex(
                signing.SigningError, "input_archive_hash_mismatch"
            ):
                signing.extract_verified(archive, "0" * 64, SOURCE, root / "extracted")
            self.assertFalse((root / "extracted").exists())

    def test_unrecorded_file_and_changed_bytes_are_rejected(self):
        for change in ("extra", "bytes"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                _, _, _ = fixture(root)
                bundle = root / "source/stark"
                path = bundle / ("extra.py" if change == "extra" else NATIVE_NAMES[0])
                path.write_bytes(b"unexpected content")
                archive = root / "changed.tar.gz"
                digest = signing.distribution.create_archive(bundle, archive)
                with self.assertRaisesRegex(
                    signing.SigningError, "input_file_integrity"
                ):
                    signing.extract_verified(
                        archive, digest, SOURCE, root / "extracted"
                    )

    def test_source_revision_and_dirty_inputs_are_rejected(self):
        for change in ("revision", "dirty"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                _, _, manifest = fixture(root)
                manifest["source"][change] = "c" * 40 if change == "revision" else True
                (root / "source/stark" / signing.MANIFEST).write_text(
                    json.dumps(manifest)
                )
                archive = root / "changed.tar.gz"
                digest = signing.distribution.create_archive(
                    root / "source/stark", archive
                )
                with self.assertRaisesRegex(
                    signing.SigningError, "input_source_identity"
                ):
                    signing.extract_verified(
                        archive, digest, SOURCE, root / "extracted"
                    )

    def test_unsafe_members_are_rejected_before_any_extraction(self):
        cases = [
            ("stark/../escape", tarfile.REGTYPE, False),
            ("/stark/escape", tarfile.REGTYPE, False),
            ("stark/link", tarfile.SYMTYPE, False),
            ("stark/hard", tarfile.LNKTYPE, False),
            ("stark/device", tarfile.CHRTYPE, False),
            ("stark/file", tarfile.REGTYPE, True),
        ]
        for name, kind, duplicate in cases:
            with (
                self.subTest(name=name, kind=kind),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                archive = root / "unsafe.tar"
                with tarfile.open(archive, "w") as contents:
                    entry = tarfile.TarInfo(name)
                    entry.type = kind
                    entry.linkname = "escape"
                    contents.addfile(entry, io.BytesIO())
                    if duplicate:
                        contents.addfile(entry, io.BytesIO())
                with self.assertRaises(signing.SigningError):
                    signing.extract_verified(
                        archive,
                        signing.distribution.sha256(archive),
                        SOURCE,
                        root / "extracted",
                    )
                self.assertFalse((root / "extracted").exists())

    def test_only_an_unambiguous_developer_id_application_identity_is_accepted(self):
        valid = f'  1) {IDENTITY["sha1"]} "Developer ID Application: Example ({IDENTITY["team_id"]})"\n'.encode()
        with patch.object(signing, "run", return_value=result(valid)):
            self.assertEqual(
                signing.resolve_identity(IDENTITY["sha1"], None, {}), IDENTITY
            )
        for output in (
            valid.replace(b"Developer ID Application", b"Apple Development"),
            valid + valid,
            b"0 valid identities found\n",
        ):
            with (
                self.subTest(output=output),
                patch.object(signing, "run", return_value=result(output)),
                self.assertRaisesRegex(
                    signing.SigningError, "developer_id_application_identity_required"
                ),
            ):
                signing.resolve_identity(IDENTITY["sha1"], None, {})

    def test_all_native_runtime_files_are_signed_individually(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            bundle = root / "source/stark"
            with (
                patch.object(signing, "run") as command,
                patch.object(signing, "verify_signature") as verify,
            ):
                paths = signing.sign_bundle(
                    bundle, IDENTITY, Path("/private/test.keychain-db"), {}
                )
            self.assertEqual(set(paths), set(NATIVE_NAMES))
            self.assertEqual(command.call_count, len(NATIVE_NAMES))
            self.assertEqual(verify.call_count, len(NATIVE_NAMES))
            identifiers = []
            for call in command.call_args_list:
                arguments = call.args[0]
                self.assertNotIn("--deep", arguments)
                self.assertIn("--timestamp", arguments)
                self.assertEqual(arguments[arguments.index("--options") + 1], "runtime")
                self.assertEqual(
                    arguments[arguments.index("--sign") + 1], IDENTITY["sha1"]
                )
                identifiers.append(arguments[arguments.index("--identifier") + 1])
            self.assertEqual(len(set(identifiers)), len(NATIVE_NAMES))

    def test_non_loadable_objects_are_preserved_without_executable_signatures(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "python.o"
            original = b"\xcf\xfa\xed\xfe" + bytes(8) + (1).to_bytes(4, "little")
            path.write_bytes(original)
            self.assertEqual(signing.macho_kind(path), "object")
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(b"\xcf\xfa\xed\xfe")
            with self.assertRaisesRegex(signing.SigningError, "mach_o_header_invalid"):
                signing.macho_kind(path)

    def test_signature_policy_requires_exact_certificate_runtime_team_and_timestamp(
        self,
    ):
        metadata = f"CodeDirectory v=20500 size=100 flags=0x10000(runtime) hashes=1+2 location=embedded\nTeamIdentifier={IDENTITY['team_id']}\nTimestamp=Oct 21, 2026 at 1:00:00 PM\n".encode()

        def command(_arguments, stage, _env):
            return result(
                stderr=metadata if stage == "code_signature_metadata" else b""
            )

        with patch.object(signing, "run", side_effect=command) as run:
            signing.verify_signature(Path("/binary"), IDENTITY, {}, notarized=True)
        requirement = run.call_args_list[0].args[0]
        self.assertIn("--check-notarization", requirement)
        self.assertIn(
            f'=anchor apple generic and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf = H"{IDENTITY["sha1"]}"',
            requirement,
        )
        for bad, stage in (
            (
                metadata.replace(b"flags=0x10000", b"flags=0x0"),
                "hardened_runtime_missing",
            ),
            (
                metadata.replace(b"Timestamp=Oct", b"Absent=Oct"),
                "secure_timestamp_missing",
            ),
            (
                metadata.replace(IDENTITY["team_id"].encode(), b"ZZZZZZZZZZ"),
                "signing_team_mismatch",
            ),
        ):
            with (
                self.subTest(stage=stage),
                patch.object(
                    signing,
                    "run",
                    side_effect=lambda _arguments, current, _env, bad=bad: result(
                        stderr=bad if current == "code_signature_metadata" else b""
                    ),
                ),
                self.assertRaisesRegex(signing.SigningError, stage),
            ):
                signing.verify_signature(Path("/binary"), IDENTITY, {})

    def test_notarization_must_be_accepted_with_a_real_submission_identifier(self):
        for response in (
            {"status": "Invalid", "id": UUID},
            {"status": "In Progress", "id": UUID},
            {"status": "Accepted", "id": "not-an-id"},
        ):
            with (
                self.subTest(response=response),
                patch.object(
                    signing, "run", return_value=result(json.dumps(response).encode())
                ),
                self.assertRaises(signing.SigningError),
            ):
                signing.notarize(Path("/signed.zip"), "test-profile", None, {})
        with patch.object(
            signing,
            "run",
            return_value=result(
                json.dumps({"status": "Accepted", "id": UUID}).encode()
            ),
        ):
            self.assertEqual(
                signing.notarize(Path("/signed.zip"), "test-profile", None, {}),
                {"status": "accepted", "submission_id": UUID},
            )

    def test_exclusive_publication_preserves_existing_files_and_rolls_back_new_files(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = root / "staged"
            staged.write_text("candidate")
            original = root / "existing"
            original.write_text("published")
            fresh = root / "fresh"
            with self.assertRaises(FileExistsError):
                signing.publish([(staged, fresh), (staged, original)])
            self.assertEqual(original.read_text(), "published")
            self.assertFalse(fresh.exists())

    def test_sign_only_outputs_are_distinct_and_hash_signed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, digest, original = fixture(root)
            options = argparse.Namespace(
                archive=archive,
                input_sha256=digest,
                expected_source=SOURCE,
                version="0.1.1",
                sign_identity=IDENTITY["sha1"],
                notary_profile=None,
                keychain=None,
                sign_only=True,
                output_dir=root / "output",
            )
            recipe = {
                "revision": "d" * 40,
                "dirty": True,
                "files_sha256": {"synthetic": "e" * 64},
            }

            def sign_files(bundle, *_args):
                for name in NATIVE_NAMES:
                    path = bundle / name
                    path.write_bytes(path.read_bytes() + b"signed")
                return list(NATIVE_NAMES)

            with (
                patch.object(signing.platform, "system", return_value="Darwin"),
                patch.object(signing.platform, "machine", return_value="arm64"),
                patch.object(signing, "recipe_provenance", return_value=recipe),
                patch.object(signing, "resolve_identity", return_value=IDENTITY),
                patch.object(signing, "sign_bundle", side_effect=sign_files),
                patch.object(signing.distribution, "verify_relocation"),
                patch.object(signing, "verify_signature"),
                patch.object(signing, "notarize") as notary,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                receipt = signing.sign(options)
            self.assertEqual(receipt["notarization"], {"status": "not_requested"})
            notary.assert_not_called()
            self.assertEqual(len(list(options.output_dir.glob("*"))), 3)
            signed_archive = (
                options.output_dir
                / "stark-0.1.1-aarch64-apple-darwin-signed-unnotarized.tar.gz"
            )
            with tarfile.open(signed_archive) as contents:
                manifest = json.load(contents.extractfile("stark/" + signing.MANIFEST))
                for name in NATIVE_NAMES:
                    signed = contents.extractfile("stark/" + name).read()
                    self.assertTrue(signed.endswith(b"signed"))
                    self.assertNotEqual(
                        manifest["files_sha256"][name], original["files_sha256"][name]
                    )
                    self.assertEqual(
                        manifest["files_sha256"][name],
                        signing.hashlib.sha256(signed).hexdigest(),
                    )
            self.assertEqual(manifest["source"], original["source"])
            self.assertEqual(manifest["version"], "0.1.1")
            self.assertEqual(manifest["distribution_source"], recipe)

    def test_normal_mode_requires_acceptance_and_online_executable_tickets(self):
        for accepted in (False, True):
            with (
                self.subTest(accepted=accepted),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                archive, digest, _ = fixture(root)
                options = argparse.Namespace(
                    archive=archive,
                    input_sha256=digest,
                    expected_source=SOURCE,
                    version="0.1.1",
                    sign_identity=IDENTITY["sha1"],
                    notary_profile="test-profile",
                    keychain=None,
                    sign_only=False,
                    output_dir=root / "output",
                )
                recipe = {"revision": "d" * 40, "dirty": False, "files_sha256": {}}

                def create_zip(arguments, stage, _env):
                    self.assertEqual(stage, "notary_zip_failed")
                    Path(arguments[-1]).write_bytes(b"synthetic notarization ZIP")
                    return result()

                with (
                    patch.object(signing.platform, "system", return_value="Darwin"),
                    patch.object(signing.platform, "machine", return_value="arm64"),
                    patch.object(signing, "recipe_provenance", return_value=recipe),
                    patch.object(signing, "resolve_identity", return_value=IDENTITY),
                    patch.object(
                        signing, "sign_bundle", return_value=list(NATIVE_NAMES)
                    ),
                    patch.object(signing.distribution, "verify_relocation"),
                    patch.object(signing, "verify_signature") as verify,
                    patch.object(signing, "run", side_effect=create_zip),
                    patch.object(
                        signing,
                        "notarize",
                        return_value={"status": "accepted", "submission_id": UUID},
                        side_effect=None
                        if accepted
                        else signing.SigningError("notarization_not_accepted"),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if not accepted:
                        with self.assertRaisesRegex(
                            signing.SigningError, "notarization_not_accepted"
                        ):
                            signing.sign(options)
                        self.assertEqual(list(options.output_dir.iterdir()), [])
                        continue
                    receipt = signing.sign(options)
                self.assertEqual(len(list(options.output_dir.iterdir())), 5)
                self.assertEqual(
                    receipt["notarization"]["standalone_executable_tickets"],
                    "verified_online",
                )
                online = [
                    call
                    for call in verify.call_args_list
                    if call.kwargs.get("notarized")
                ]
                self.assertEqual(len(online), 3)


if __name__ == "__main__":
    unittest.main()
