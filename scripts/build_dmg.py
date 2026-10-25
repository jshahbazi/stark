#!/usr/bin/env python3
"""Package an unchanged signed Stark.app into a signed, notarized disk image."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import plistlib
import re
import shutil
import sys
import tempfile
from pathlib import Path

import sign_distribution as signing

ROOT = Path(__file__).resolve().parents[1]
RECIPE_FILES = (
    "scripts/build_dmg.py",
    "scripts/sign_distribution.py",
    "scripts/build_distribution.py",
    "scripts/update_app.py",
    "scripts/install_app.py",
    "apps/installer/Sources/InstallerApp.swift",
    "apps/menu-bar/Sources/AppUpdate.swift",
)
INSTALLER_NAME = "Install Stark.app"
INSTALLER_ID = "com.starkengine.installer"


def build_installer(
    staging: Path,
    version: str,
    identity: dict[str, str],
    keychain: Path | None,
    env: dict[str, str],
) -> Path:
    """Give DMG installation its own identity so an older Stark cannot intercept it."""
    app = staging / INSTALLER_NAME
    contents = app / "Contents"
    executable = contents / "MacOS/InstallStark"
    executable.parent.mkdir(parents=True)
    resources = contents / "Resources"
    resources.mkdir()
    info = {
        "CFBundleIdentifier": INSTALLER_ID,
        "CFBundleExecutable": "InstallStark",
        "CFBundleName": "Install Stark",
        "CFBundleDisplayName": "Install Stark",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": signing.distribution.MIN_MACOS,
        "LSMultipleInstancesProhibited": True,
        "NSHighResolutionCapable": True,
    }
    icon = staging / "Stark.app/Contents/Resources/Stark.icns"
    if icon.is_file():
        shutil.copyfile(icon, resources / "Stark.icns")
        info["CFBundleIconFile"] = "Stark"
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    for name in ("update_app.py", "install_app.py"):
        shutil.copyfile(ROOT / "scripts" / name, resources / name)
    # The installer must survive Gatekeeper App Translocation. Keep its payload
    # inside its own seal; the root copy remains for existing in-app updaters.
    helpers = contents / "Helpers"
    helpers.mkdir()
    shutil.copytree(staging / "Stark.app", helpers / "Stark.app")
    signing.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            "-O",
            "-target",
            "arm64-apple-macos" + signing.distribution.MIN_MACOS,
            "-framework",
            "AppKit",
            "-framework",
            "SwiftUI",
            ROOT / "apps/menu-bar/Sources/AppUpdate.swift",
            ROOT / "apps/installer/Sources/InstallerApp.swift",
            "-o",
            executable,
        ],
        "dmg_installer_compilation_failed",
        env,
        timeout=300,
    )
    arguments = [
        "/usr/bin/codesign",
        "--sign",
        identity["sha1"],
        "--timestamp",
        "--options",
        "runtime",
        "--identifier",
        INSTALLER_ID,
    ]
    if keychain is not None:
        arguments.extend(["--keychain", str(keychain)])
    signing.run([*arguments, app], "dmg_installer_signing_failed", env)
    signing.verify_signature(app, identity, env)
    return app


def inventory_digest(app: Path) -> str:
    inventory = signing.distribution.file_manifest(app)
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recipe_provenance(env: dict[str, str]) -> dict:
    revision = (
        signing.run(["/usr/bin/git", "rev-parse", "HEAD"], "dmg_recipe_revision", env)
        .stdout.decode()
        .strip()
    )
    signing.require(
        signing.REVISION.fullmatch(revision) is not None, "dmg_recipe_revision_invalid"
    )
    status = signing.run(
        [
            "/usr/bin/git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *RECIPE_FILES,
        ],
        "dmg_recipe_status",
        env,
    ).stdout
    signing.require(not status, "dmg_recipe_has_changes")
    return {
        "revision": revision,
        "dirty": False,
        "files_sha256": {
            name: signing.distribution.sha256(ROOT / name) for name in RECIPE_FILES
        },
    }


def verify_image_signature(
    image: Path, identity: dict[str, str], env: dict[str, str]
) -> None:
    # A disk image is a container, not executable code: hardened-runtime flags
    # apply to its already-signed app, not to the disk image's signature.
    requirement = (
        "=anchor apple generic and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        f'certificate leaf = H"{identity["sha1"]}"'
    )
    signing.run(
        ["/usr/bin/codesign", "--verify", "--strict", "-R", requirement, image],
        "dmg_signature_invalid",
        env,
    )
    detail = signing.run(
        ["/usr/bin/codesign", "--display", "--verbose=4", image],
        "dmg_signature_metadata",
        env,
    ).stderr.decode(errors="replace")
    signing.require(
        re.search(rf"(?m)^TeamIdentifier={re.escape(identity['team_id'])}$", detail)
        is not None,
        "dmg_signing_team_mismatch",
    )
    timestamp = re.search(r"(?m)^Timestamp=(.+)$", detail)
    signing.require(
        timestamp is not None
        and timestamp[1].strip().lower() not in {"none", "not set"},
        "dmg_secure_timestamp_missing",
    )


def verify_image(
    image: Path,
    app_digest: str,
    identity: dict[str, str],
    env: dict[str, str],
    mountpoint: Path,
    source: str,
    version: str,
    installer_digest: str | None = None,
) -> None:
    """Inspect an attached read-only image without opening or installing its app."""
    verify_image_signature(image, identity, env)
    signing.run(["/usr/bin/hdiutil", "verify", image], "dmg_integrity_invalid", env)
    signing.run(
        ["/usr/bin/xcrun", "stapler", "validate", image], "dmg_ticket_invalid", env
    )
    signing.run(
        [
            "/usr/sbin/spctl",
            "--assess",
            "--type",
            "open",
            "--context",
            "context:primary-signature",
            image,
        ],
        "dmg_gatekeeper_rejected",
        env,
    )
    mountpoint.mkdir()
    attached = False
    try:
        signing.run(
            [
                "/usr/bin/hdiutil",
                "attach",
                "-readonly",
                "-nobrowse",
                "-noautoopen",
                "-mountpoint",
                mountpoint,
                image,
            ],
            "dmg_mount_failed",
            env,
        )
        attached = True
        link = mountpoint / "Applications"
        signing.require(
            link.is_symlink() and link.readlink() == Path("/Applications"),
            "dmg_applications_link_invalid",
        )
        app = mountpoint / "Stark.app"
        signing.require(
            app.is_dir() and not app.is_symlink(), "dmg_application_missing"
        )
        signing.require(
            inventory_digest(app) == app_digest, "dmg_application_bytes_changed"
        )
        app_manifest = json.loads(
            (app / "Contents/Resources/runtime/share/stark/manifest.json").read_bytes()
        )
        signing.require(
            app_manifest.get("source", {}).get("revision") == source
            and app_manifest.get("version") == version,
            "dmg_application_source_mismatch",
        )
        signing.verify_signature(app, identity, env, notarized=True)
        signing.run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", app],
            "dmg_application_seal_invalid",
            env,
        )
        signing.run(
            ["/usr/bin/xcrun", "stapler", "validate", app],
            "dmg_application_ticket_invalid",
            env,
        )
        if installer_digest is not None:
            launcher = mountpoint / INSTALLER_NAME
            signing.require(
                launcher.is_dir()
                and not launcher.is_symlink()
                and inventory_digest(launcher) == installer_digest,
                "dmg_installer_bytes_changed",
            )
            metadata = plistlib.loads((launcher / "Contents/Info.plist").read_bytes())
            signing.require(
                metadata.get("CFBundleIdentifier") == INSTALLER_ID
                and metadata.get("CFBundleShortVersionString") == version,
                "dmg_installer_identity_invalid",
            )
            signing.verify_signature(launcher, identity, env, notarized=True)
            signing.run(
                ["/usr/bin/codesign", "--verify", "--deep", "--strict", launcher],
                "dmg_installer_seal_invalid",
                env,
            )
            signing.run(
                ["/usr/sbin/spctl", "--assess", "--type", "execute", launcher],
                "dmg_installer_gatekeeper_rejected",
                env,
            )
    finally:
        if attached or mountpoint.is_mount():
            try:
                signing.run(
                    ["/usr/bin/hdiutil", "detach", mountpoint], "dmg_detach_failed", env
                )
            except signing.SigningError:
                signing.run(
                    ["/usr/bin/hdiutil", "detach", "-force", mountpoint],
                    "dmg_force_detach_failed",
                    env,
                )


def build(options: argparse.Namespace) -> dict:
    signing.require(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "dmg_requires_apple_silicon_macos",
    )
    signing.distribution.validate_version(options.version)
    signing.require(bool(options.notary_profile), "notary_profile_required")
    env = signing.environment()
    recipe = recipe_provenance(env)
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"stark-{options.version}-{signing.distribution.TARGET}"
    names = [f"{stem}.dmg", f"{stem}.dmg.sha256", f"{stem}.dmg.notarization.json"]
    signing.require(
        all(
            not (output / name).exists() and not (output / name).is_symlink()
            for name in names
        ),
        "dmg_output_already_exists",
    )
    with tempfile.TemporaryDirectory(prefix=".stark-dmg-", dir=output) as temporary:
        work = Path(temporary)
        payload = work / "payload"
        payload.mkdir()
        bundle, manifest = signing.extract_verified(
            options.archive.resolve(),
            options.input_sha256,
            options.expected_source,
            payload,
        )
        signing.require(
            manifest["version"] == options.version, "dmg_input_version_mismatch"
        )
        app = signing.application(bundle)
        signing.require(app is not None, "dmg_application_missing")
        metadata = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        signing.require(
            metadata.get("CFBundleShortVersionString") == options.version,
            "dmg_application_version_mismatch",
        )
        identity = signing.resolve_identity(
            options.sign_identity, options.keychain, env
        )
        signing.verify_application(bundle, identity, env, notarized=True)
        app_digest = inventory_digest(app)
        staging = work / "image"
        staging.mkdir()
        shutil.copytree(app, staging / "Stark.app")
        launcher = build_installer(
            staging, options.version, identity, options.keychain, env
        )
        installer_digest = inventory_digest(launcher)
        (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
        image = work / names[0]
        signing.run(
            [
                "/usr/bin/hdiutil",
                "create",
                "-volname",
                f"Stark {options.version}",
                "-srcfolder",
                staging,
                "-fs",
                "HFS+",
                "-format",
                "UDZO",
                image,
            ],
            "dmg_creation_failed",
            env,
            timeout=600,
        )
        arguments = [
            "/usr/bin/codesign",
            "--sign",
            identity["sha1"],
            "--timestamp",
            "--identifier",
            "com.starkengine.diskimage",
        ]
        if options.keychain is not None:
            arguments.extend(["--keychain", str(options.keychain)])
        signing.run([*arguments, image], "dmg_signing_failed", env)
        verify_image_signature(image, identity, env)
        submission_hash = signing.distribution.sha256(image)
        notarization = signing.notarize(
            image, options.notary_profile, options.keychain, env
        )
        signing.run(
            ["/usr/bin/xcrun", "stapler", "staple", image], "dmg_stapling_failed", env
        )
        verify_image(
            image,
            app_digest,
            identity,
            env,
            work / "mounted",
            options.expected_source,
            options.version,
            installer_digest,
        )
        digest = signing.distribution.sha256(image)
        receipt = {
            "schema_version": 1,
            "version": options.version,
            "source_revision": options.expected_source,
            "distribution_source": recipe,
            "input_archive_sha256": options.input_sha256,
            "dmg_sha256": digest,
            "application_inventory_sha256": app_digest,
            "application_bytes_preserved": True,
            "installer_inventory_sha256": installer_digest,
            "installer_bundle_id": INSTALLER_ID,
            "application_ticket": "stapled_and_validated",
            "certificate_sha1": identity["sha1"],
            "team_id": identity["team_id"],
            "secure_timestamp": True,
            "notarization_submission_sha256": submission_hash,
            "notarization": {
                **notarization,
                "disk_image_ticket": "stapled_and_validated",
            },
            "gatekeeper_assessment": "accepted",
            "read_only_mount_verified": True,
            "models_loaded": 0,
            "native_inference_calls": 0,
        }
        (work / names[1]).write_text(f"{digest}  {names[0]}\n")
        (work / names[2]).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        signing.publish([(work / name, output / name) for name in names])
    print(
        json.dumps(
            {"status": "passed", "version": options.version, "dmg_sha256": digest},
            sort_keys=True,
        )
    )
    return receipt


def verify_download(options: argparse.Namespace) -> dict:
    signing.require(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "dmg_requires_apple_silicon_macos",
    )
    signing.require(
        signing.REVISION.fullmatch(options.expected_source) is not None,
        "dmg_source_invalid",
    )
    signing.require(
        re.fullmatch(r"[0-9A-Fa-f]{40}", options.expected_signer) is not None,
        "dmg_expected_signer_invalid",
    )
    receipt = json.loads(options.receipt.read_bytes())
    signing.require(
        receipt.get("schema_version") == 1
        and receipt.get("version") == options.version
        and receipt.get("source_revision") == options.expected_source,
        "dmg_receipt_identity_mismatch",
    )
    certificate = options.expected_signer.upper()
    signing.require(
        receipt.get("certificate_sha1") == certificate, "dmg_receipt_signer_mismatch"
    )
    team = receipt.get("team_id")
    signing.require(
        isinstance(team, str) and re.fullmatch(r"[A-Z0-9]{10}", team) is not None,
        "dmg_receipt_team_invalid",
    )
    digest = signing.distribution.sha256(options.verify_dmg)
    signing.require(receipt.get("dmg_sha256") == digest, "dmg_download_hash_mismatch")
    app_digest = receipt.get("application_inventory_sha256")
    signing.require(
        isinstance(app_digest, str)
        and signing.SHA256.fullmatch(app_digest) is not None,
        "dmg_receipt_application_hash_invalid",
    )
    installer_digest = receipt.get("installer_inventory_sha256")
    if installer_digest is not None:
        signing.require(
            isinstance(installer_digest, str)
            and signing.SHA256.fullmatch(installer_digest) is not None
            and receipt.get("installer_bundle_id") == INSTALLER_ID,
            "dmg_receipt_installer_hash_invalid",
        )
    signing.require(
        receipt.get("notarization", {}).get("status") == "accepted",
        "dmg_receipt_notarization_missing",
    )
    with tempfile.TemporaryDirectory(prefix="stark-dmg-verification-") as temporary:
        verify_image(
            options.verify_dmg.resolve(),
            app_digest,
            {"sha1": certificate, "team_id": team},
            signing.environment(),
            Path(temporary) / "mounted",
            options.expected_source,
            options.version,
            installer_digest,
        )
    result = {
        "status": "passed",
        "version": options.version,
        "source_revision": options.expected_source,
        "dmg_sha256": digest,
        "certificate_sha1": certificate,
        "application_inventory_sha256": app_digest,
        "checks": [
            "download_hash",
            "developer_id_signature",
            "secure_timestamp",
            "stapled_dmg_ticket",
            "gatekeeper_assessment",
            "read_only_mount",
            "applications_shortcut",
            "application_file_hashes",
            "application_source",
            "application_seal",
            "stapled_application_ticket",
        ],
        "models_loaded": 0,
        "native_inference_calls": 0,
    }
    if installer_digest is not None:
        result["installer_inventory_sha256"] = installer_digest
        result["checks"].extend(
            ["installer_file_hashes", "installer_signature", "installer_gatekeeper"]
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "passed", "version": options.version, "dmg_sha256": digest},
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--expected-source", required=True)
    parser.add_argument(
        "--version", type=signing.distribution.validate_version, required=True
    )
    parser.add_argument("--sign-identity")
    parser.add_argument("--notary-profile")
    parser.add_argument("--keychain", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/distributions"
    )
    parser.add_argument("--verify-dmg", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-signer")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.verify_dmg is not None:
        if not all((options.receipt, options.expected_signer, options.output)):
            parser.error(
                "--verify-dmg requires --receipt, --expected-signer and --output"
            )
    elif not all(
        (
            options.archive,
            options.input_sha256,
            options.sign_identity,
            options.notary_profile,
        )
    ):
        parser.error(
            "building requires --archive, --input-sha256, --sign-identity and --notary-profile"
        )
    try:
        if options.verify_dmg is not None:
            verify_download(options)
        else:
            build(options)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
        message = (
            str(error)
            if isinstance(error, signing.SigningError)
            else "dmg_build_failed"
        )
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
