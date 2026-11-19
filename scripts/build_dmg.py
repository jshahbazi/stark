#!/usr/bin/env python3
"""Package an unchanged signed Stark.app into a signed, notarized disk image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import stat
import struct
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
    "apps/installer/Resources/RenderIcon.swift",
    "apps/installer/Resources/RenderBackground.swift",
    "apps/menu-bar/Resources/Stark.jpg",
    "apps/menu-bar/Sources/AppUpdate.swift",
)
INSTALLER_NAME = "Install Stark.app"
INSTALLER_ID = "com.starkengine.installer"

# Finder window layout, in points from the top-left of the window content.
# RenderBackground.swift draws the panel, arrow and guidance around these
# icon centres, so the two always agree.
WINDOW_SIZE = (640, 540)
# Finder's saved bounds cover the whole frame, including the macOS 26 title
# bar, and measure the origin from the bottom-left of the screen.
TITLE_BAR_HEIGHT = 32
WINDOW_ORIGIN = (320, 200)
ICON_SIZE = 128
ICON_CENTRES = {
    "Stark.app": (200, 216),
    "Applications": (440, 216),
    INSTALLER_NAME: (200, 427),
}
BACKGROUND = Path(".background/background.tiff")
LAYOUT_NAME = ".DS_Store"
# Finder's own layout files for one small B-tree are 8196 bytes.
LAYOUT_SIZE = 8196
# Seconds from the classic Mac OS epoch (1904-01-01) to the Unix epoch.
MAC_EPOCH_OFFSET = 2_082_844_800
# Classic alias records store the volume name as a 27-byte Pascal string.
MAX_VOLUME_NAME = 27


def build_installer_icon(resources: Path, env: dict[str, str]) -> None:
    """Render the package artwork with the existing cat logo before signing."""
    with tempfile.TemporaryDirectory(prefix="stark-installer-icon-") as temporary:
        work = Path(temporary)
        artwork = work / "InstallStark.png"
        signing.run(
            [
                "/usr/bin/xcrun",
                "swift",
                ROOT / "apps/installer/Resources/RenderIcon.swift",
                ROOT / "apps/menu-bar/Resources/Stark.jpg",
                artwork,
            ],
            "dmg_installer_icon_render_failed",
            env,
            timeout=300,
        )
        iconset = work / "InstallStark.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                filename = (
                    f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
                )
                pixels = str(size * scale)
                signing.run(
                    [
                        "/usr/bin/sips",
                        "-z",
                        pixels,
                        pixels,
                        artwork,
                        "--out",
                        iconset / filename,
                    ],
                    "dmg_installer_icon_resize_failed",
                    env,
                )
        signing.run(
            [
                "/usr/bin/iconutil",
                "-c",
                "icns",
                iconset,
                "-o",
                resources / "InstallStark.icns",
            ],
            "dmg_installer_icon_package_failed",
            env,
        )


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
        "CFBundleIconFile": "InstallStark",
    }
    build_installer_icon(resources, env)
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    for name in ("update_app.py", "install_app.py"):
        shutil.copyfile(ROOT / "scripts" / name, resources / name)
    # The installer must survive Gatekeeper App Translocation. Keep its payload
    # inside its own seal; the root copy remains for existing in-app updaters.
    helpers = contents / "Helpers"
    helpers.mkdir()
    # HFS+ disk images preserve file hard links. Keep both real app directories
    # for old updaters and App Translocation, but store their file data once.
    shutil.copytree(staging / "Stark.app", helpers / "Stark.app", copy_function=os.link)
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


def render_background(staging: Path, env: dict[str, str]) -> None:
    """Draw the window background around the configured icon positions."""
    destination = staging / BACKGROUND
    destination.parent.mkdir()

    def centre(name: str) -> dict[str, int]:
        x, y = ICON_CENTRES[name]
        return {"x": x, "y": y}

    layout = {
        "width": WINDOW_SIZE[0],
        "height": WINDOW_SIZE[1],
        "iconSize": ICON_SIZE,
        "app": centre("Stark.app"),
        "applications": centre("Applications"),
        "installer": centre(INSTALLER_NAME),
    }
    signing.run(
        [
            "/usr/bin/xcrun",
            "swift",
            ROOT / "apps/installer/Resources/RenderBackground.swift",
            destination,
            json.dumps(layout, sort_keys=True, separators=(",", ":")),
        ],
        "dmg_background_render_failed",
        env,
        timeout=300,
    )
    signing.require(destination.is_file(), "dmg_background_missing")
    # Reserve the layout file's blocks so the writable image cannot run out
    # of space when the real layout replaces it after mounting.
    (staging / LAYOUT_NAME).write_bytes(bytes(LAYOUT_SIZE))


def created(path: Path) -> float:
    return path.stat().st_birthtime


def hide(path: Path) -> None:
    os.chflags(path, path.lstat().st_flags | stat.UF_HIDDEN)


def background_alias(
    volume_name: str,
    volume_created: float,
    folder_id: int,
    file_id: int,
    file_created: float,
) -> bytes:
    """Encode the version 2 alias record Finder resolves for the background."""
    volume = volume_name.encode()
    signing.require(
        0 < len(volume) <= MAX_VOLUME_NAME and b":" not in volume,
        "dmg_volume_name_invalid",
    )
    folder, filename = BACKGROUND.parts
    volume_date = volume_created + MAC_EPOCH_OFFSET
    file_date = file_created + MAC_EPOCH_OFFSET
    record = struct.pack(
        ">h28pI2shI64pII4s4shhI2s10s",
        0,  # file
        volume,
        int(volume_date),
        b"H+",
        0,  # fixed disk
        folder_id,
        filename.encode(),
        file_id,
        int(file_date),
        bytes(4),
        bytes(4),
        -1,
        -1,
        0,
        bytes(2),
        bytes(10),
    )

    def tag(code: int, value: bytes) -> bytes:
        return struct.pack(">hh", code, len(value)) + value + bytes(len(value) % 2)

    def text(code: int, value: str) -> bytes:
        encoded = value.encode("utf-16-be")
        return tag(code, struct.pack(">h", len(encoded) // 2) + encoded)

    # Tag order follows Finder's own records.
    record += b"".join(
        (
            tag(0, folder.encode()),
            tag(16, struct.pack(">Q", int(volume_date * 65536))),
            tag(17, struct.pack(">Q", int(file_date * 65536))),
            tag(1, struct.pack(">I", folder_id)),
            tag(2, b":".join((volume, folder.encode(), filename.encode()))),
            text(14, filename),
            text(15, volume_name),
            tag(18, f"/{BACKGROUND}".encode()),
            # Users mount the image under /Volumes. Recording the private
            # build mount point would leak a local path into the release.
            tag(19, f"/Volumes/{volume_name}".encode()),
            struct.pack(">hh", -1, 0),
        )
    )
    return struct.pack(">4shh", bytes(4), 8 + len(record), 2) + record


def layout_store(alias: bytes) -> bytes:
    """Encode Finder's window layout as a single-leaf .DS_Store B-tree."""
    x, y = WINDOW_ORIGIN
    width, height = WINDOW_SIZE[0], WINDOW_SIZE[1] + TITLE_BAR_HEIGHT
    window = {
        "ContainerShowSidebar": False,
        "PreviewPaneVisibility": False,
        "ShowPathbar": False,
        "ShowSidebar": False,
        "ShowStatusBar": False,
        "ShowTabView": False,
        "ShowToolbar": False,
        "SidebarWidth": 0,
        # An NSRect string: {{x, y}, {width, height}}.
        "WindowBounds": f"{{{{{x}, {y}}}, {{{width}, {height}}}}}",
    }
    view = {
        "arrangeBy": "none",
        "backgroundColorBlue": 1.0,
        "backgroundColorGreen": 1.0,
        "backgroundColorRed": 1.0,
        "backgroundImageAlias": alias,
        "backgroundType": 2,
        "gridOffsetX": 0.0,
        "gridOffsetY": 0.0,
        "gridSpacing": 100.0,
        "iconSize": float(ICON_SIZE),
        "labelOnBottom": True,
        "scrollPositionX": 0.0,
        "scrollPositionY": 0.0,
        "showIconPreview": False,
        "showItemInfo": False,
        "textSize": 13.0,
        "viewOptionsVersion": 1,
    }

    def blob(value: bytes) -> bytes:
        return b"blob" + struct.pack(">I", len(value)) + value

    records = [
        (".", b"bwsp", blob(plistlib.dumps(window, fmt=plistlib.FMT_BINARY))),
        (".", b"icvp", blob(plistlib.dumps(view, fmt=plistlib.FMT_BINARY))),
        (".", b"vSrn", b"long" + struct.pack(">I", 1)),
        (".", b"vstl", b"type" + b"icnv"),
    ]
    for name, (x, y) in ICON_CENTRES.items():
        location = struct.pack(">IIII", x, y, 0xFFFFFFFF, 0xFFFF0000)
        records.append((name, b"Iloc", blob(location)))
    # Finder keeps records in case-insensitive name order, then by code.
    records.sort(key=lambda record: (record[0].lower(), record[1]))
    node = struct.pack(">II", 0, len(records))
    for name, code, value in records:
        encoded = name.encode("utf-16-be")
        node += struct.pack(">I", len(encoded) // 2) + encoded + code + value

    # Buddy-allocated blocks, as (offset, log2 size) after the 4-byte magic:
    # the 32-byte allocator header sits at 0, then the B-tree superblock,
    # allocator root and one leaf. Every other buddy block remains free.
    superblock, root, leaf = (0x20, 5), (0x800, 11), (0x1000, 12)
    signing.require(len(node) <= 1 << leaf[1], "dmg_layout_too_large")
    free = {width: [1 << width] for width in (6, 7, 8, 9, 10, *range(13, 31))}
    addresses = [offset | width for offset, width in (root, superblock, leaf)]
    allocator = struct.pack(">II", len(addresses), 0)
    allocator += struct.pack(">256I", *addresses, *[0] * (256 - len(addresses)))
    allocator += struct.pack(">IB4sI", 1, 4, b"DSDB", 1)
    for width in range(32):
        blocks = free.get(width, [])
        allocator += struct.pack(f">I{len(blocks)}I", len(blocks), *blocks)
    signing.require(len(allocator) <= 1 << root[1], "dmg_layout_too_large")

    data = bytearray(LAYOUT_SIZE)
    data[:36] = struct.pack(
        ">I4sIII16s", 1, b"Bud1", root[0], 1 << root[1], root[0], bytes(16)
    )
    for (offset, _), payload in (
        (root, allocator),
        # Root node block 2, no internal levels, one node of 4096 bytes.
        (superblock, struct.pack(">IIIII", 2, 0, len(records), 1, 1 << leaf[1])),
        (leaf, node),
    ):
        data[4 + offset : 4 + offset + len(payload)] = payload
    return bytes(data)


def write_layout(volume: Path, volume_name: str) -> None:
    """Point Finder at the background and place each icon on the mounted image."""
    background = volume / BACKGROUND
    signing.require(
        background.is_file() and not background.is_symlink(),
        "dmg_background_missing",
    )
    # The Applications shortcut is checked as a link; its target belongs to
    # whoever mounts the image, not to the image itself.
    for name in ICON_CENTRES:
        signing.require(os.path.lexists(volume / name), "dmg_layout_item_missing")
    alias = background_alias(
        volume_name,
        created(volume),
        background.parent.stat().st_ino,
        background.stat().st_ino,
        created(background),
    )
    (volume / LAYOUT_NAME).write_bytes(layout_store(alias))
    hide(background.parent)
    # A writable mount may start an fsevents log; it is not part of the image.
    shutil.rmtree(volume / ".fseventsd", ignore_errors=True)


def apply_layout(
    image: Path, volume_name: str, mountpoint: Path, env: dict[str, str]
) -> None:
    """Write the Finder layout into the writable image before compression."""
    mountpoint.mkdir()
    attached = False
    try:
        signing.run(
            [
                "/usr/bin/hdiutil",
                "attach",
                "-readwrite",
                "-nobrowse",
                "-noautoopen",
                "-noverify",
                "-mountpoint",
                mountpoint,
                image,
            ],
            "dmg_layout_mount_failed",
            env,
        )
        attached = True
        write_layout(mountpoint, volume_name)
    finally:
        if attached or mountpoint.is_mount():
            try:
                signing.run(
                    ["/usr/bin/hdiutil", "detach", mountpoint],
                    "dmg_layout_detach_failed",
                    env,
                )
            except signing.SigningError:
                signing.run(
                    ["/usr/bin/hdiutil", "detach", "-force", mountpoint],
                    "dmg_layout_force_detach_failed",
                    env,
                )


def inventory_digest(app: Path) -> str:
    inventory = signing.distribution.file_manifest(app)
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_shared_payload(app: Path, installer: Path) -> None:
    """Reject images that expanded the compatibility paths into two payloads."""
    embedded = installer / "Contents/Helpers/Stark.app"
    signing.require(
        inventory_digest(app) == inventory_digest(embedded),
        "dmg_shared_payload_bytes_changed",
    )
    for path in app.rglob("*"):
        if path.is_file():
            other = embedded / path.relative_to(app)
            signing.require(
                not path.is_symlink()
                and not other.is_symlink()
                and path.samefile(other),
                "dmg_payload_not_shared",
            )


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
    shared_payload: bool = False,
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
            if shared_payload:
                verify_shared_payload(app, launcher)
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
        render_background(staging, env)
        volume_name = f"Stark {options.version}"
        writable = work / "writable.dmg"
        signing.run(
            [
                "/usr/bin/hdiutil",
                "create",
                "-volname",
                volume_name,
                "-srcfolder",
                staging,
                "-fs",
                "HFS+",
                "-format",
                "UDRW",
                writable,
            ],
            "dmg_creation_failed",
            env,
            timeout=600,
        )
        apply_layout(writable, volume_name, work / "layout", env)
        image = work / names[0]
        signing.run(
            [
                "/usr/bin/hdiutil",
                "convert",
                writable,
                "-format",
                "UDZO",
                "-o",
                image,
            ],
            "dmg_compression_failed",
            env,
            timeout=600,
        )
        writable.unlink()
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
            shared_payload=True,
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
            "payload_storage": "shared_hardlinks",
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
            shared_payload=receipt.get("payload_storage") == "shared_hardlinks",
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
