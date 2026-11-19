"""Model-free disk-image integrity, signing order, and mount cleanup checks."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dmg as dmg
import update_app as updater

SOURCE = "a" * 40
IDENTITY = {"sha1": "B" * 40, "team_id": "ABCDEFGHIJ"}
SUBMISSION = "12345678-1234-1234-1234-123456789abc"
CREATED = 1_793_000_000.0


def read_layout(data: bytes) -> dict[tuple[str, bytes], object]:
    """Decode a .DS_Store and check its buddy allocator and B-tree invariants."""
    magic, signature, root, size, root_again, _ = struct.unpack_from(">I4sIII16s", data)
    assert (magic, signature, root_again) == (1, b"Bud1", root)
    count, _ = struct.unpack_from(">II", data, 4 + root)
    addresses = struct.unpack_from(f">{count}I", data, 4 + root + 8)
    blocks = [(address & ~0x1F, address & 0x1F) for address in addresses]
    assert blocks[0] == (root, size.bit_length() - 1)
    position = 4 + root + 8 + 4 * (-(-count // 256) * 256)
    (entries,) = struct.unpack_from(">I", data, position)
    position += 4
    toc = {}
    for _ in range(entries):
        length = data[position]
        name = data[position + 1 : position + 1 + length]
        (toc[name],) = struct.unpack_from(">I", data, position + 1 + length)
        position += 5 + length
    free = []
    for width in range(32):
        (entries,) = struct.unpack_from(">I", data, position)
        offsets = struct.unpack_from(f">{entries}I", data, position + 4)
        free.extend((offset, width) for offset in offsets)
        position += 4 + 4 * entries
    assert position <= 4 + root + size
    # The allocator header, used blocks and free blocks tile the address space.
    cursor = 0
    for offset, width in sorted([(0, 5), *blocks, *free]):
        assert offset == cursor and offset % (1 << width) == 0
        cursor += 1 << width
    assert cursor == 1 << 31
    end = max(4 + offset + (1 << width) for offset, width in blocks)
    assert len(data) >= end
    superblock = blocks[toc[b"DSDB"]][0]
    node_id, levels, records, nodes, page = struct.unpack_from(
        ">IIIII", data, 4 + superblock
    )
    assert (levels, nodes, page) == (0, 1, 4096)
    node, width = blocks[node_id]
    pointer, count = struct.unpack_from(">II", data, 4 + node)
    assert pointer == 0 and count == records
    position = 4 + node + 8
    result, order = {}, []
    for _ in range(count):
        (length,) = struct.unpack_from(">I", data, position)
        name = data[position + 4 : position + 4 + 2 * length].decode("utf-16-be")
        position += 4 + 2 * length
        code, kind = data[position : position + 4], data[position + 4 : position + 8]
        position += 8
        if kind == b"blob":
            (length,) = struct.unpack_from(">I", data, position)
            value = data[position + 4 : position + 4 + length]
            position += 4 + length
        elif kind == b"long":
            (value,) = struct.unpack_from(">I", data, position)
            position += 4
        else:
            assert kind == b"type"
            value = data[position : position + 4]
            position += 4
        result[(name, code)] = value
        order.append((name.lower(), code))
    assert order == sorted(order) and position <= 4 + node + (1 << width)
    return result


def read_alias(data: bytes) -> tuple[tuple, dict[int, bytes]]:
    """Decode a version 2 alias record into its fixed fields and tags."""
    _, length, version = struct.unpack_from(">4shh", data)
    assert (length, version) == (len(data), 2)
    layout = ">h28pI2shI64pII4s4shhI2s10s"
    fields = struct.unpack_from(layout, data, 8)
    position = 8 + struct.calcsize(layout)
    tags = {}
    while True:
        code, length = struct.unpack_from(">hh", data, position)
        position += 4
        if code == -1:
            break
        tags[code] = data[position : position + length]
        position += length + length % 2
    assert position == len(data)
    return fields, tags


class DMGTests(unittest.TestCase):
    def test_independent_payload_copies_are_rejected(self):
        launcher = self.root / "Independent.app"
        shutil.copytree(self.app, launcher / "Contents/Helpers/Stark.app")
        with self.assertRaisesRegex(dmg.signing.SigningError, "dmg_payload_not_shared"):
            dmg.verify_shared_payload(self.app, launcher)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "source/stark"
        self.app = self.bundle / dmg.signing.APPLICATION
        executable = self.app / "Contents/MacOS/Stark"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"synthetic executable")
        (self.app / "Contents/Info.plist").write_bytes(
            plistlib.dumps(
                {
                    "CFBundleIdentifier": dmg.signing.APPLICATION_ID,
                    "CFBundleExecutable": "Stark",
                    "CFBundleShortVersionString": "0.3.0",
                }
            )
        )
        runtime = self.app / "Contents/Resources/runtime/share/stark"
        runtime.mkdir(parents=True)
        (runtime / "manifest.json").write_text(
            json.dumps({"source": {"revision": SOURCE}, "version": "0.3.0"})
        )
        share = self.bundle / "share/stark"
        share.mkdir(parents=True)
        (share / "VERSION").write_text("0.3.0\n")
        self.manifest = {
            "schema_version": 1,
            "version": "0.3.0",
            "target": dmg.signing.distribution.TARGET,
            "minimum_macos": dmg.signing.distribution.MIN_MACOS,
            "source": {"revision": SOURCE, "dirty": False},
            "files_sha256": dmg.signing.distribution.file_manifest(self.bundle),
        }
        (self.bundle / dmg.signing.MANIFEST).write_text(json.dumps(self.manifest))
        self.archive = self.root / "input.tar.gz"
        self.digest = dmg.signing.distribution.create_archive(self.bundle, self.archive)
        self.options = argparse.Namespace(
            archive=self.archive,
            input_sha256=self.digest,
            expected_source=SOURCE,
            version="0.3.0",
            sign_identity=IDENTITY["sha1"],
            keychain=Path("/private/synthetic.keychain"),
            notary_profile="synthetic-profile",
            output_dir=self.root / "output",
        )
        self.commands = []
        self.fail_stage = None
        self.tamper = False
        self.tamper_installer = False
        self.notarization_status = "Accepted"
        self.image_app = self.app
        self.image_installer = None
        self.installer_snapshot = self.root / "installer-snapshot.app"
        self.metadata = (
            f"CodeDirectory v=20500 flags=0x10000(runtime)\nTeamIdentifier={IDENTITY['team_id']}\nTimestamp=Oct 23, 2026\n"
        ).encode()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(dmg.platform, "system", return_value="Darwin")
        )
        self.stack.enter_context(
            patch.object(dmg.platform, "machine", return_value="arm64")
        )
        self.stack.enter_context(
            patch.object(dmg.signing, "resolve_identity", return_value=IDENTITY)
        )
        self.stack.enter_context(
            patch.object(dmg.signing, "run", side_effect=self.run_command)
        )
        # Creation times and Finder flags are macOS metadata; CI runs on Linux.
        self.stack.enter_context(patch.object(dmg, "created", return_value=CREATED))
        self.hidden = []
        self.stack.enter_context(
            patch.object(dmg, "hide", side_effect=self.hidden.append)
        )
        self.render_background = True
        self.layout = None
        self.layout_mount = None

    def run_command(self, arguments, stage, env, **kwargs):
        arguments = list(map(str, arguments))
        self.commands.append((stage, arguments))
        if stage == self.fail_stage or (
            isinstance(self.fail_stage, set) and stage in self.fail_stage
        ):
            raise dmg.signing.SigningError(stage)
        stdout, stderr = b"", b""
        if stage == "dmg_recipe_revision":
            stdout = (SOURCE + "\n").encode()
        elif stage == "dmg_installer_icon_package_failed":
            Path(arguments[-1]).write_bytes(b"synthetic package icon")
        elif stage == "dmg_installer_compilation_failed":
            Path(arguments[arguments.index("-o") + 1]).write_bytes(
                b"synthetic installer"
            )
        elif stage == "dmg_background_render_failed":
            json.loads(arguments[-1])
            if self.render_background:
                Path(arguments[-2]).write_bytes(b"synthetic background")
        elif stage == "dmg_creation_failed":
            self.assertEqual(arguments[arguments.index("-format") + 1], "UDRW")
            self.image_app = (
                Path(arguments[arguments.index("-srcfolder") + 1]) / "Stark.app"
            )
            self.image_installer = self.image_app.parent / dmg.INSTALLER_NAME
            shutil.rmtree(self.installer_snapshot, ignore_errors=True)
            shutil.copytree(self.image_installer, self.installer_snapshot)
            Path(arguments[-1]).write_bytes(b"synthetic writable image")
        elif stage == "dmg_layout_mount_failed":
            self.assertIn("-readwrite", arguments)
            self.assertIn("-nobrowse", arguments)
            self.assertIn("-noautoopen", arguments)
            self.layout_mount = Path(arguments[arguments.index("-mountpoint") + 1])
            shutil.copytree(
                self.image_app.parent,
                self.layout_mount,
                symlinks=True,
                dirs_exist_ok=True,
                copy_function=os.link,
            )
            (self.layout_mount / ".fseventsd").mkdir()
        elif stage == "dmg_layout_detach_failed":
            background = self.layout_mount / dmg.BACKGROUND
            self.layout = (self.layout_mount / dmg.LAYOUT_NAME).read_bytes()
            self.layout_state = {
                "fseventsd": (self.layout_mount / ".fseventsd").exists(),
                "folder_id": background.parent.stat().st_ino,
                "file_id": background.stat().st_ino,
            }
        elif stage == "dmg_compression_failed":
            self.assertEqual(arguments[arguments.index("-format") + 1], "UDZO")
            Path(arguments[-1]).write_bytes(b"synthetic compressed image")
        elif stage in {"code_signature_metadata", "dmg_signature_metadata"}:
            stderr = self.metadata
        elif stage == "notarization_failed":
            stdout = json.dumps(
                {"status": self.notarization_status, "id": SUBMISSION}
            ).encode()
        elif stage == "dmg_mount_failed":
            self.assertIn("-readonly", arguments)
            self.assertIn("-nobrowse", arguments)
            self.assertIn("-noautoopen", arguments)
            destination = Path(arguments[arguments.index("-mountpoint") + 1])
            shutil.copytree(
                self.image_app, destination / "Stark.app", copy_function=os.link
            )
            if self.image_installer is not None:
                shutil.copytree(
                    self.image_installer,
                    destination / dmg.INSTALLER_NAME,
                    copy_function=os.link,
                )
            (destination / "Applications").symlink_to("/Applications")
            if self.tamper:
                (destination / "Stark.app/Contents/MacOS/Stark").write_bytes(b"changed")
            if self.tamper_installer:
                (
                    destination / dmg.INSTALLER_NAME / "Contents/MacOS/InstallStark"
                ).write_bytes(b"changed")
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=stderr)

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return dmg.build(self.options)

    def stages(self):
        return [stage for stage, _ in self.commands]

    def test_success_preserves_application_and_verifies_before_publication(self):
        before = self.archive.read_bytes()
        receipt = self.build()
        self.assertEqual(receipt["payload_storage"], "shared_hardlinks")
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertEqual(receipt["input_archive_sha256"], self.digest)
        self.assertEqual(
            receipt["application_inventory_sha256"], dmg.inventory_digest(self.app)
        )
        self.assertTrue(receipt["application_bytes_preserved"])
        self.assertEqual(receipt["installer_bundle_id"], dmg.INSTALLER_ID)
        self.assertEqual(
            receipt["installer_inventory_sha256"],
            dmg.inventory_digest(self.installer_snapshot),
        )
        self.assertEqual(
            dmg.inventory_digest(
                self.installer_snapshot / "Contents/Helpers/Stark.app"
            ),
            dmg.inventory_digest(self.app),
        )
        metadata = plistlib.loads(
            (self.installer_snapshot / "Contents/Info.plist").read_bytes()
        )
        self.assertNotEqual(metadata["CFBundleIdentifier"], dmg.signing.APPLICATION_ID)
        self.assertEqual(metadata["CFBundleIconFile"], "InstallStark")
        self.assertEqual(
            (
                self.installer_snapshot / "Contents/Resources/InstallStark.icns"
            ).read_bytes(),
            b"synthetic package icon",
        )
        for name in ("install_app.py", "update_app.py"):
            self.assertEqual(
                (self.installer_snapshot / "Contents/Resources" / name).read_bytes(),
                (dmg.ROOT / "scripts" / name).read_bytes(),
            )
        self.assertEqual(receipt["notarization"]["status"], "accepted")
        self.assertEqual(receipt["distribution_source"]["revision"], SOURCE)
        self.assertIn(
            "scripts/build_dmg.py", receipt["distribution_source"]["files_sha256"]
        )
        stages = self.stages()
        self.assertLess(
            stages.index("application_ticket_invalid"),
            stages.index("dmg_creation_failed"),
        )
        self.assertLess(
            stages.index("dmg_installer_icon_package_failed"),
            stages.index("dmg_installer_signing_failed"),
        )
        self.assertLess(
            stages.index("dmg_installer_signing_failed"),
            stages.index("dmg_creation_failed"),
        )
        self.assertIn("dmg_installer_gatekeeper_rejected", stages)
        self.assertLess(
            stages.index("dmg_signing_failed"), stages.index("notarization_failed")
        )
        self.assertLess(
            stages.index("notarization_failed"), stages.index("dmg_stapling_failed")
        )
        self.assertLess(
            stages.index("dmg_stapling_failed"), stages.index("dmg_mount_failed")
        )
        self.assertEqual(stages[-1], "dmg_detach_failed")
        names = sorted(path.name for path in self.options.output_dir.iterdir())
        self.assertEqual(len(names), 3)
        self.assertTrue(
            all(
                name.startswith("stark-0.3.0-aarch64-apple-darwin.dmg")
                for name in names
            )
        )

    def test_window_layout_is_written_before_compression_and_signing(self):
        self.build()
        stages = self.stages()
        order = [
            "dmg_installer_signing_failed",
            "dmg_background_render_failed",
            "dmg_creation_failed",
            "dmg_layout_mount_failed",
            "dmg_layout_detach_failed",
            "dmg_compression_failed",
            "dmg_signing_failed",
        ]
        self.assertEqual(
            [stages.index(stage) for stage in order],
            sorted(stages.index(stage) for stage in order),
        )
        self.assertIn(
            "apps/installer/Resources/RenderBackground.swift", dmg.RECIPE_FILES
        )
        self.assertEqual(self.hidden, [self.layout_mount / ".background"])
        self.assertFalse(self.layout_state["fseventsd"])
        self.assertEqual(len(self.layout), dmg.LAYOUT_SIZE)
        records = read_layout(self.layout)
        for name, (x, y) in dmg.ICON_CENTRES.items():
            self.assertEqual(
                struct.unpack(">IIII", records[(name, b"Iloc")]),
                (x, y, 0xFFFFFFFF, 0xFFFF0000),
            )
        self.assertEqual(records[(".", b"vstl")], b"icnv")
        window = plistlib.loads(records[(".", b"bwsp")])
        self.assertFalse(window["ShowToolbar"] or window["ShowSidebar"])
        self.assertEqual(window["WindowBounds"], "{{320, 200}, {640, 572}}")
        view = plistlib.loads(records[(".", b"icvp")])
        self.assertEqual(
            (view["backgroundType"], view["arrangeBy"], view["iconSize"]),
            (2, "none", 128.0),
        )
        fields, tags = read_alias(view["backgroundImageAlias"])
        self.assertEqual(fields[1], b"Stark 0.3.0")
        self.assertEqual(fields[2], int(CREATED) + dmg.MAC_EPOCH_OFFSET)
        self.assertEqual(fields[5], self.layout_state["folder_id"])
        self.assertEqual(fields[6], b"background.tiff")
        self.assertEqual(fields[7], self.layout_state["file_id"])
        self.assertEqual(tags[18], b"/.background/background.tiff")
        self.assertEqual(tags[19], b"/Volumes/Stark 0.3.0")
        self.assertEqual(tags[15][2:].decode("utf-16-be"), "Stark 0.3.0")
        self.assertNotIn(str(self.root).encode(), self.layout)

    def test_alias_rejects_volume_names_finder_cannot_record(self):
        for name in ("", "Stark:0.3.0", "Stark " + "9" * 22):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    dmg.signing.SigningError, "dmg_volume_name_invalid"
                ),
            ):
                dmg.background_alias(name, CREATED, 2, 3, CREATED)

    def test_missing_background_stops_before_image_creation(self):
        self.render_background = False
        with self.assertRaisesRegex(dmg.signing.SigningError, "dmg_background_missing"):
            self.build()
        self.assertNotIn("dmg_creation_failed", self.stages())
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_layout_failures_detach_the_writable_image_and_publish_nothing(self):
        for failure in ("dmg_layout_item_missing", "dmg_layout_force_detach_failed"):
            with self.subTest(failure=failure):
                self.commands.clear()
                if failure == "dmg_layout_item_missing":
                    self.fail_stage = None
                    context = patch.dict(dmg.ICON_CENTRES, {"Missing.app": (1, 1)})
                else:
                    self.fail_stage = {
                        "dmg_layout_detach_failed",
                        "dmg_layout_force_detach_failed",
                    }
                    context = contextlib.nullcontext()
                with context, self.assertRaisesRegex(dmg.signing.SigningError, failure):
                    self.build()
                stages = self.stages()
                self.assertIn("dmg_layout_detach_failed", stages)
                self.assertNotIn("dmg_compression_failed", stages)
                self.assertNotIn("dmg_signing_failed", stages)
                self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_busy_writable_mount_is_force_detached(self):
        self.fail_stage = "dmg_layout_detach_failed"
        self.build()
        stages = self.stages()
        self.assertLess(
            stages.index("dmg_layout_force_detach_failed"),
            stages.index("dmg_compression_failed"),
        )

    def test_hash_mismatch_does_not_access_signing_identity_or_publish(self):
        self.options.input_sha256 = "0" * 64
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "input_archive_hash_mismatch"
        ):
            self.build()
        self.assertNotIn("dmg_signing_failed", self.stages())
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_wrong_source_or_version_cannot_be_repackaged(self):
        for field, value, message in (
            ("expected_source", "c" * 40, "input_source_identity"),
            ("version", "0.4.0", "dmg_input_version_mismatch"),
        ):
            with self.subTest(field=field):
                previous = getattr(self.options, field)
                setattr(self.options, field, value)
                with self.assertRaisesRegex(dmg.signing.SigningError, message):
                    self.build()
                setattr(self.options, field, previous)

    def test_existing_output_and_dangling_symlinks_are_never_replaced(self):
        self.options.output_dir.mkdir()
        path = self.options.output_dir / "stark-0.3.0-aarch64-apple-darwin.dmg"
        path.symlink_to("/nonexistent/synthetic")
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_output_already_exists"
        ):
            self.build()
        self.assertTrue(path.is_symlink())
        self.assertNotIn("dmg_creation_failed", self.stages())

    def test_unsigned_input_is_rejected_before_image_creation(self):
        self.fail_stage = "code_signature_invalid"
        with self.assertRaisesRegex(dmg.signing.SigningError, "code_signature_invalid"):
            self.build()
        self.assertNotIn("dmg_creation_failed", self.stages())

    def test_notarization_rejection_publishes_nothing(self):
        self.notarization_status = "Invalid"
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "notarization_not_accepted"
        ):
            self.build()
        self.assertNotIn("dmg_stapling_failed", self.stages())
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_mount_verification_failure_detaches_and_publishes_nothing(self):
        self.tamper = True
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_application_bytes_changed"
        ):
            self.build()
        self.assertEqual(self.stages()[-1], "dmg_detach_failed")
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_app_signature_failure_after_mount_always_detaches(self):
        self.fail_stage = "dmg_application_seal_invalid"
        with self.assertRaisesRegex(dmg.signing.SigningError, self.fail_stage):
            self.build()
        self.assertEqual(self.stages()[-1], "dmg_detach_failed")

    def test_installer_tampering_detaches_and_publishes_nothing(self):
        self.tamper_installer = True
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_installer_bytes_changed"
        ):
            self.build()
        self.assertEqual(self.stages()[-1], "dmg_detach_failed")
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_installer_compilation_and_signing_failure_publish_nothing(self):
        for stage in (
            "dmg_installer_icon_render_failed",
            "dmg_installer_icon_resize_failed",
            "dmg_installer_icon_package_failed",
            "dmg_installer_compilation_failed",
            "dmg_installer_signing_failed",
        ):
            with self.subTest(stage=stage):
                self.fail_stage = stage
                with self.assertRaisesRegex(dmg.signing.SigningError, stage):
                    self.build()
                self.assertNotIn("dmg_creation_failed", self.stages())
                self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_disk_image_requires_secure_timestamp(self):
        self.metadata = f"TeamIdentifier={IDENTITY['team_id']}\n".encode()
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_secure_timestamp_missing"
        ):
            dmg.verify_image_signature(self.archive, IDENTITY, {})

    def test_busy_private_mount_is_force_detached(self):
        self.fail_stage = "dmg_detach_failed"
        self.build()
        self.assertEqual(self.stages()[-1], "dmg_force_detach_failed")
        self.assertIn("-force", self.commands[-1][1])

    def test_failed_detach_never_publishes_an_image(self):
        self.fail_stage = {"dmg_detach_failed", "dmg_force_detach_failed"}
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_force_detach_failed"
        ):
            self.build()
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_receiving_verification_never_imports_signing_credentials(self):
        receipt = self.build()
        self.image_app = self.installer_snapshot / "Contents/Helpers/Stark.app"
        self.image_installer = self.installer_snapshot
        image = self.options.output_dir / "stark-0.3.0-aarch64-apple-darwin.dmg"
        receipt_path = image.with_suffix(".dmg.notarization.json")
        options = argparse.Namespace(
            verify_dmg=image,
            receipt=receipt_path,
            expected_source=SOURCE,
            expected_signer=IDENTITY["sha1"],
            version="0.3.0",
            output=self.root / "receiving.json",
        )
        with (
            patch.object(
                dmg.signing,
                "resolve_identity",
                side_effect=AssertionError("must not read credentials"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = dmg.verify_download(options)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["dmg_sha256"], receipt["dmg_sha256"])
        self.assertIn("installer_gatekeeper", result["checks"])
        # Historical receipts describe two independent payload copies.
        receipt.pop("payload_storage")
        receipt_path.write_text(json.dumps(receipt))
        self.image_app = self.app
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dmg.verify_download(options)["status"], "passed")
        image.write_bytes(b"different downloaded bytes")
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_download_hash_mismatch"
        ):
            dmg.verify_download(options)


@unittest.skipUnless(sys.platform == "darwin", "Xcode Swift compiler required")
class NativeInstallerTests(unittest.TestCase):
    def test_hfs_image_shares_payload_and_foundation_copies_survive_unmount(self):
        with tempfile.TemporaryDirectory(prefix="stark shared dmg ") as temporary:
            root = Path(temporary)
            staging = root / "image"
            app = staging / "Stark.app"
            (app / "Contents/MacOS").mkdir(parents=True)
            shutil.copy("/bin/echo", app / "Contents/MacOS/Stark")
            (app / "Contents/Info.plist").write_bytes(
                plistlib.dumps(
                    {
                        "CFBundleIdentifier": "com.starkengine.test-payload",
                        "CFBundleExecutable": "Stark",
                        "CFBundlePackageType": "APPL",
                    }
                )
            )
            (app / "Contents/Resources").mkdir()
            payload = app / "Contents/Resources/payload"
            payload.write_bytes(os.urandom(2 * 1024 * 1024))
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(app)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            launcher = staging / dmg.INSTALLER_NAME
            shutil.copytree(
                app, launcher / "Contents/Helpers/Stark.app", copy_function=os.link
            )
            (launcher / "Contents/MacOS").mkdir()
            shutil.copy("/bin/echo", launcher / "Contents/MacOS/InstallStark")
            (launcher / "Contents/Info.plist").write_bytes(
                plistlib.dumps(
                    {
                        "CFBundleIdentifier": "com.starkengine.test-installer",
                        "CFBundleExecutable": "InstallStark",
                        "CFBundlePackageType": "APPL",
                    }
                )
            )
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(launcher)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            image = root / "shared.dmg"
            subprocess.run(
                [
                    "hdiutil",
                    "create",
                    "-srcfolder",
                    str(staging),
                    "-fs",
                    "HFS+",
                    "-format",
                    "UDZO",
                    str(image),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            # Incompressible fixture data must occupy one payload, not two.
            self.assertLess(image.stat().st_size, 3 * 1024 * 1024)
            mount = root / "mount"
            mount.mkdir()
            subprocess.run(
                [
                    "hdiutil",
                    "attach",
                    "-readonly",
                    "-nobrowse",
                    "-mountpoint",
                    str(mount),
                    str(image),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            try:
                dmg.verify_shared_payload(
                    mount / "Stark.app", mount / dmg.INSTALLER_NAME
                )
                for signed in (mount / "Stark.app", mount / dmg.INSTALLER_NAME):
                    subprocess.run(
                        ["codesign", "--verify", "--deep", "--strict", str(signed)],
                        check=True,
                        capture_output=True,
                        timeout=60,
                    )
                # Use the same Foundation operation as the unchanged installer
                # and existing in-app updaters, then detach before using copies.
                driver = root / "copy.swift"
                driver.write_text(
                    "import Foundation\n"
                    "for i in stride(from: 1, to: CommandLine.arguments.count, by: 2) {\n"
                    "try FileManager.default.copyItem(atPath: CommandLine.arguments[i], "
                    "toPath: CommandLine.arguments[i+1])\n}\n"
                )
                subprocess.run(
                    [
                        "xcrun",
                        "swift",
                        str(driver),
                        str(mount / "Stark.app"),
                        str(root / "updated.app"),
                        str(mount / dmg.INSTALLER_NAME / "Contents/Helpers/Stark.app"),
                        str(root / "installed.app"),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
            finally:
                subprocess.run(
                    ["hdiutil", "detach", str(mount)],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
            for name in ("updated.app", "installed.app"):
                copied = root / name / "Contents/Resources/payload"
                self.assertEqual(copied.read_bytes(), payload.read_bytes())
                self.assertFalse(copied.samefile(payload))
                subprocess.run(
                    ["codesign", "--verify", "--deep", "--strict", str(root / name)],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )

    def test_native_workspace_passes_helper_validation_and_reports_early_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Harness.swift"
            source.write_text(
                """import Foundation
@main struct Harness {
    static func main() throws {
        let workspace = try InstallerFiles.createWorkspace()
        let updateWorkspace = try AppUpdateFiles.createWorkspace(prefix: "stark-update-")
        let early = Data(#"{"error":"installation workspace is invalid"}"#.utf8)
        do {
            _ = try InstallerFiles.helperResult(status: 1, output: early, workspace: workspace)
            preconditionFailure("early helper failure was hidden")
        } catch let failure as InstallerFailure {
            precondition(failure.message == "installation workspace is invalid")
        }
        try Data(#"{"error":"recorded failure"}"#.utf8).write(to: workspace.appendingPathComponent("result.json"))
        do {
            _ = try InstallerFiles.helperResult(status: 1, output: early, workspace: workspace)
            preconditionFailure("recorded helper failure was hidden")
        } catch let failure as InstallerFailure {
            precondition(failure.message == "recorded failure")
        }
        try FileManager.default.removeItem(at: workspace.appendingPathComponent("result.json"))
        for output in [Data("invalid".utf8), Data(repeating: 65, count: 16_385)] {
            do {
                _ = try InstallerFiles.helperResult(status: 1, output: output, workspace: workspace)
                preconditionFailure("invalid helper reply was accepted")
            } catch InstallerError.install {}
        }
        let installed = try InstallerFiles.helperResult(status: 0,
            output: Data(#"{"path":"/Applications/Stark.app","version":"0.6.1"}"#.utf8), workspace: workspace)
        precondition(installed.version == "0.6.1")
        let paths = ["installer": workspace, "updater": updateWorkspace].mapValues { directory in
            ["workspace": directory.path,
             "candidate": directory.appendingPathComponent("Stark.app").path,
             "helper": directory.appendingPathComponent("update_app.py").path]
        }
        let data = try JSONSerialization.data(withJSONObject: paths)
        FileHandle.standardOutput.write(data)
    }
}
"""
            )
            executable = root / "harness"
            compiled = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-O",
                    "-D",
                    "STARK_INSTALLER_LIBRARY",
                    "-target",
                    "arm64-apple-macos26.4",
                    "-framework",
                    "AppKit",
                    "-framework",
                    "SwiftUI",
                    str(dmg.ROOT / "apps/menu-bar/Sources/AppUpdate.swift"),
                    str(dmg.ROOT / "apps/installer/Sources/InstallerApp.swift"),
                    str(source),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                compiled.returncode, 0, "native regression fixture compilation failed"
            )
            result = subprocess.run(
                [str(executable)], capture_output=True, timeout=30, check=False
            )
            self.assertEqual(
                result.returncode, 0, "native installer regression fixture failed"
            )
            paths = json.loads(result.stdout)
            for record in paths.values():
                self.addCleanup(
                    shutil.rmtree, Path(record["workspace"]), ignore_errors=True
                )
            self.assertEqual(set(paths), {"installer", "updater"})
            for name, record in paths.items():
                with self.subTest(workspace=name):
                    workspace = Path(record["workspace"])
                    self.assertTrue(workspace.is_dir())
                    self.assertEqual(
                        updater.physical_path(workspace, "invalid workspace"),
                        workspace,
                    )
                    self.assertEqual(workspace.stat().st_mode & 0o777, 0o700)
                    self.assertEqual(workspace.stat().st_uid, os.getuid())
                    for field, filename in (
                        ("candidate", "Stark.app"),
                        ("helper", "update_app.py"),
                    ):
                        child = Path(record[field])
                        self.assertEqual(child, workspace / filename)
                        self.assertEqual(
                            updater.physical_path(child, "invalid child path"), child
                        )

    def test_installer_compiles_with_shared_release_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-O",
                    "-target",
                    "arm64-apple-macos26.4",
                    "-framework",
                    "AppKit",
                    "-framework",
                    "SwiftUI",
                    str(dmg.ROOT / "apps/menu-bar/Sources/AppUpdate.swift"),
                    str(dmg.ROOT / "apps/installer/Sources/InstallerApp.swift"),
                    "-o",
                    str(Path(temporary) / "InstallStark"),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, "native installer compilation failed"
            )


if __name__ == "__main__":
    unittest.main()
