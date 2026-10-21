#!/usr/bin/env python3
"""Build a relocatable Apple-silicon Stark bundle without loading a model.

Run with the repository's pinned uv and Python. All generated files remain in
ignored directories; subprocess diagnostics are intentionally not published.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "aarch64-apple-darwin"
MIN_MACOS = "26.4"
PYTHON_VERSION = "3.13.13"
UV_VERSION = "0.12.24"
RUST_VERSION = "1.98.1"
RECIPE_INPUTS = (
    "scripts/build_distribution.py",
    "scripts/stark",
    "install.sh",
    "scripts/install-from-source.sh",
    "Cargo.lock",
    "rust-toolchain.toml",
    "tools/converter/uv.lock",
)
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)*\Z")
SOURCE_PATHS = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo",
    "crates",
    "vendor/cxx",
    "tools/converter",
    "LICENSE",
    "provenance/UPSTREAM.md",
    "scripts/licenses/distribution",
    *RECIPE_INPUTS,
)


class BuildError(Exception):
    """A content-free build failure suitable for terminal output."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_version(value: str) -> str:
    if VERSION.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "version must have three numeric components followed by optional dot/hyphen letter-or-digit components"
        )
    return value


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # Do not let a developer environment supply Python modules, native flags or
    # credentials to the build or to the model-free installation checks.
    for name in tuple(environment):
        if name.startswith(
            ("STARK_CAPTURE_", "UZU_CAPTURE_", "CARGO_PROFILE_", "DYLD_")
        ) or name in {
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_TARGET",
            "RUSTUP_TOOLCHAIN",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_TOKEN_PATH",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "UV_PYTHON",
            "UV_PYTHON_INSTALL_DIR",
            "UV_PROJECT",
            "UV_WORKING_DIR",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        }:
            environment.pop(name, None)
    environment.update(
        CARGO_TARGET_DIR=str(ROOT / "target"),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        UV_NO_PROGRESS="1",
        MACOSX_DEPLOYMENT_TARGET=MIN_MACOS,
    )
    return environment


def run(
    command: list[str],
    stage: str,
    environment: dict[str, str],
    *,
    cwd: Path = ROOT,
    expected: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            command, cwd=cwd, env=environment, capture_output=True, check=False
        )
    except OSError as error:
        raise BuildError(f"{stage}: command_unavailable") from error
    if result.returncode != expected:
        raise BuildError(f"{stage}: command_failed (exit {result.returncode})")
    return result


def source_provenance(
    environment: dict[str, str], allow_dirty: bool
) -> dict[str, object]:
    revision = (
        run(["git", "rev-parse", "HEAD"], "source_revision", environment)
        .stdout.decode()
        .strip()
    )
    diff = run(
        ["git", "diff", "--binary", "HEAD", "--"], "source_diff", environment
    ).stdout
    for name in RECIPE_INPUTS:
        if not (ROOT / name).is_file():
            raise BuildError("missing_distribution_recipe_input")
    untracked_source = run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS],
        "source_recipe_status",
        environment,
    ).stdout
    file_names = (
        run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *SOURCE_PATHS,
            ],
            "source_file_inventory",
            environment,
        )
        .stdout.decode()
        .split("\0")
    )
    source_files = {
        name: sha256(ROOT / name)
        for name in sorted(set(file_names))
        if name and (ROOT / name).is_file()
    }
    dirty = bool(diff or untracked_source)
    if dirty and not allow_dirty:
        raise BuildError(
            "source_tree_has_changes: commit the recipe or use --allow-dirty for a development bundle"
        )
    return {
        "repository": "https://github.com/jshahbazi/stark",
        "revision": revision,
        "dirty": dirty,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files_sha256": source_files,
        "files_manifest_sha256": hashlib.sha256(
            json.dumps(source_files, sort_keys=True).encode()
        ).hexdigest(),
        "recipe_sha256": {name: sha256(ROOT / name) for name in RECIPE_INPUTS},
    }


def verify_native(binary: Path, environment: dict[str, str]) -> None:
    architecture = run(
        ["/usr/bin/lipo", "-archs", str(binary)], "native_architecture", environment
    ).stdout
    if architecture.strip() != b"arm64":
        raise BuildError("native_architecture_must_be_arm64")
    linkage = run(
        ["/usr/bin/otool", "-L", str(binary)], "native_linkage", environment
    ).stdout.decode()
    for line in linkage.splitlines()[1:]:
        dependency = line.strip().split(" (", 1)[0]
        if dependency and not dependency.startswith(("/System/Library/", "/usr/lib/")):
            raise BuildError("native_non_system_dependency")
    load_commands = run(
        ["/usr/bin/otool", "-l", str(binary)], "native_deployment_target", environment
    ).stdout.decode()
    if re.search(r"\bminos\s+26\.4(?:\.0)?\s", load_commands) is None:
        raise BuildError("native_deployment_target_must_be_macos_26_4")
    result = run(
        [str(binary)], "native_invalid_arguments_smoke", environment, expected=1
    )
    try:
        reply = json.loads(result.stdout)
    except (UnicodeDecodeError, ValueError) as error:
        raise BuildError("native_invalid_arguments_reply") from error
    if reply != {"ready": False, "error": "invalid_arguments"} or result.stderr:
        raise BuildError("native_invalid_arguments_reply")


def build_native(bundle: Path, cargo: str, environment: dict[str, str]) -> None:
    libexec = bundle / "libexec"
    libexec.mkdir(parents=True)
    for name, features in (
        ("stark-score", ["--features", "optimized"]),
        ("stark-generate", []),
    ):
        print(f"Building {name} with locked dependencies", flush=True)
        # Keep these invocations separate: Cargo feature unification must not
        # enable scoring-only optimizations in the generation executable.
        run(
            [
                cargo,
                f"+{RUST_VERSION}",
                "build",
                "--release",
                "--locked",
                "-p",
                name,
                "--bin",
                name,
                *features,
            ],
            f"build_{name}",
            environment,
        )
        binary = ROOT / "target" / "release" / name
        verify_native(binary, environment)
        shutil.copy2(binary, libexec / name)


def copy_runtime(source: Path, destination: Path) -> None:
    """Copy only an internally linked runtime, dereferencing its file aliases."""
    source = source.resolve()
    for entry in source.rglob("*"):
        if entry.is_symlink() and not entry.resolve().is_relative_to(source):
            raise BuildError("python_runtime_external_symlink")
    shutil.copytree(source, destination, symlinks=False)


def build_python(
    bundle: Path, temporary: Path, uv: str, environment: dict[str, str]
) -> None:
    print("Bundling isolated Python and locked converter dependencies", flush=True)
    runtimes = temporary / "managed-python"
    run(
        [
            uv,
            "python",
            "install",
            "--install-dir",
            str(runtimes),
            "--no-bin",
            PYTHON_VERSION,
        ],
        "python_download",
        environment,
    )
    interpreters = sorted(
        {path.resolve() for path in runtimes.glob("cpython-*/bin/python3.13")}
    )
    if len(interpreters) != 1:
        raise BuildError("python_runtime_layout")
    runtime = bundle / "libexec" / "python"
    copy_runtime(interpreters[0].parent.parent, runtime)
    python = runtime / "bin" / "python3.13"
    version = (
        run([str(python), "-I", "-B", "--version"], "python_version", environment)
        .stdout.decode()
        .strip()
    )
    if version != f"Python {PYTHON_VERSION}":
        raise BuildError("python_version_mismatch")
    requirements = temporary / "requirements.txt"
    project = ROOT / "tools" / "converter"
    run(
        [
            uv,
            "export",
            "--project",
            str(project),
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ],
        "converter_locked_requirements",
        environment,
    )
    # Install into this copied interpreter's private module directory. --target
    # deliberately avoids modifying uv's externally-managed interpreter marker
    # and never touches the builder's system Python or virtual environment.
    site_packages = runtime / "lib" / "python3.13" / "site-packages"
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--target",
            str(site_packages),
            "--require-hashes",
            "--only-binary",
            ":all:",
            "-r",
            str(requirements),
        ],
        "converter_locked_dependencies",
        environment,
    )
    wheels = temporary / "wheels"
    run(
        [
            uv,
            "build",
            str(project),
            "--wheel",
            "--python",
            str(python),
            "--out-dir",
            str(wheels),
            "--no-build-logs",
        ],
        "converter_wheel",
        environment,
    )
    wheel_files = list(wheels.glob("stark_convert-*.whl"))
    if len(wheel_files) != 1:
        raise BuildError("converter_wheel_layout")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--target",
            str(site_packages),
            "--no-deps",
            str(wheel_files[0]),
        ],
        "converter_install",
        environment,
    )
    # The wheel's local staging URL is not useful after relocation. Source and
    # artifact hashes are carried in the bundle and converter receipts instead.
    for entry in runtime.glob(
        "lib/python3.13/site-packages/stark_convert-*.dist-info/direct_url.json"
    ):
        entry.unlink()
    if (site_packages / "bin").exists():
        shutil.rmtree(site_packages / "bin")
    # uv-generated console scripts contain a staging path. The bundle launcher
    # invokes modules through the interpreter at its relocated path instead.
    for entry in (runtime / "bin").iterdir():
        if entry.name != "python3.13":
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    for entry in list(runtime.rglob("__pycache__")):
        if entry.is_dir():
            shutil.rmtree(entry)
    for entry in runtime.rglob("*.pyc"):
        entry.unlink()
    shutil.copyfile(
        requirements, bundle / "share" / "stark" / "converter-requirements.txt"
    )


def dependency_licenses(bundle: Path, cargo: str, environment: dict[str, str]) -> None:
    """Include upstream native notices without relying on an extra Cargo tool.

    Workspace metadata may over-include optional dependencies; that is recorded
    rather than claiming this inventory proves which object files were linked.
    """
    print("Collecting upstream and dependency license notices", flush=True)
    notices = bundle / "share" / "stark" / "licenses"
    notices.mkdir()
    for name in (
        "LICENSE",
        "provenance/UPSTREAM.md",
        "tools/converter/LICENSE.lalamo",
        "vendor/cxx/LICENSE-MIT",
        "vendor/cxx/LICENSE-APACHE",
    ):
        destination = notices / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    sysroot = Path(
        run(
            ["rustc", f"+{RUST_VERSION}", "--print", "sysroot"],
            "rust_standard_library_notices",
            environment,
        )
        .stdout.decode()
        .strip()
    )
    rust_notices = sysroot / "share" / "doc" / "rust"
    if (
        not (rust_notices / "COPYRIGHT-library.html").is_file()
        or not (rust_notices / "licenses").is_dir()
    ):
        raise BuildError("rust_standard_library_license_text_missing")
    (notices / "rust").mkdir()
    shutil.copyfile(
        rust_notices / "COPYRIGHT-library.html",
        notices / "rust" / "COPYRIGHT-library.html",
    )
    shutil.copytree(
        rust_notices / "licenses", notices / "rust" / "licenses", symlinks=False
    )
    supplemental_directory = ROOT / "scripts" / "licenses" / "distribution"
    supplemental_provenance = notices / "supplemental-provenance"
    supplemental_provenance.mkdir()
    for name in ("manifest.json", "README.md"):
        if not (supplemental_directory / name).is_file():
            raise BuildError("supplemental_license_provenance_missing")
        shutil.copyfile(supplemental_directory / name, supplemental_provenance / name)
    metadata = json.loads(
        run(
            [
                cargo,
                f"+{RUST_VERSION}",
                "metadata",
                "--locked",
                "--format-version",
                "1",
                "--filter-platform",
                TARGET,
            ],
            "cargo_license_metadata",
            environment,
        ).stdout
    )
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    pending = [
        key
        for key, package in packages.items()
        if package["name"] in {"stark-score", "stark-generate"}
    ]
    selected: set[str] = set()
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        selected.add(key)
        pending.extend(
            dependency["pkg"]
            for dependency in nodes[key]["deps"]
            if any(kind["kind"] is None for kind in dependency["dep_kinds"])
        )
    inventory = []
    for key in sorted(selected):
        package = packages[key]
        if package["source"] is None:
            continue
        directory = Path(package["manifest_path"]).parent
        notice_files = []
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.upper().startswith(
                ("LICENSE", "COPYING", "NOTICE", "COPYRIGHT")
            ):
                notice_files.append(entry)
        declared = package.get("license_file")
        if declared:
            declared_path = directory / declared
            if declared_path.is_file() and declared_path not in notice_files:
                notice_files.append(declared_path)
        # Some packages carry notices in a LICENSES subdirectory.
        for subdirectory in ("LICENSES", "licenses"):
            candidate = directory / subdirectory
            if candidate.is_dir():
                notice_files.extend(
                    entry for entry in candidate.rglob("*") if entry.is_file()
                )
        supplement = None
        if not notice_files:
            notice_files, supplement = supplemental_notices(package, directory)
        destination = notices / "cargo" / f"{package['name']}-{package['version']}"
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for entry in sorted(set(notice_files)):
            relative = entry.relative_to(
                directory
                if supplement is None
                else ROOT / "scripts/licenses/distribution"
            )
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, output)
            files.append(relative.as_posix())
        inventory.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package["license"],
                "source": package["source"],
                "notices": files,
                **(
                    {"supplemental_provenance": supplement}
                    if supplement is not None
                    else {}
                ),
            }
        )
    (notices / "cargo-inventory.json").write_text(
        json.dumps(
            {
                "scope": "normal dependencies reachable in locked workspace metadata; may include unused optional code",
                "packages": inventory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def supplemental_notices(
    package: dict[str, object], source_directory: Path
) -> tuple[list[Path], dict[str, object]]:
    """Verify source-bound notices omitted from a published crates.io archive."""
    directory = ROOT / "scripts" / "licenses" / "distribution"
    manifest = directory / "manifest.json"
    key = f"{package['name']}@{package['version']}"
    if not manifest.is_file():
        raise BuildError(f"dependency_license_text_missing:{package['name']}")
    document = json.loads(manifest.read_text())
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("packages"), dict)
    ):
        raise BuildError("supplemental_license_schema_invalid")
    supplement = document.get("packages", {}).get(key)
    if not isinstance(supplement, dict):
        raise BuildError(f"dependency_license_text_missing:{package['name']}")
    vcs_path = source_directory / ".cargo_vcs_info.json"
    if not vcs_path.is_file():
        raise BuildError("supplemental_license_vcs_missing")
    vcs = json.loads(vcs_path.read_text())
    if (
        supplement.get("repository") != package.get("repository")
        or supplement.get("source_revision") != vcs.get("git", {}).get("sha1")
        or supplement.get("path_in_vcs") != vcs.get("path_in_vcs")
    ):
        raise BuildError("supplemental_license_source_mismatch")
    files = supplement.get("files")
    if not isinstance(files, dict) or not files:
        raise BuildError("supplemental_license_files_missing")
    paths = []
    for name, digest in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise BuildError("supplemental_license_path_invalid")
        if sha256(path) != digest:
            raise BuildError("supplemental_license_hash_mismatch")
        paths.append(path)
    return paths, supplement


def verify_relocation(bundle: Path, environment: dict[str, str]) -> None:
    """Move the completed tree before checks so checkout paths cannot mask bugs."""
    print("Checking relocated bundle without a model", flush=True)
    relocated = bundle.parent / "relocated bundle" / "stark"
    relocated.parent.mkdir()
    bundle.rename(relocated)
    try:
        python = relocated / "libexec" / "python" / "bin" / "python3.13"
        inspection = (
            "import sys; from pathlib import Path; import stark_convert; "
            "assert sys.version.split()[0] == '3.13.13'; "
            "assert Path(stark_convert.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
            "import numpy, tokenizers, ml_dtypes, huggingface_hub, psutil; "
            "assert not any(n.split('.')[0] in {'lalamo','jax','torch'} for n in sys.modules)"
        )
        run(
            [str(python), "-I", "-B", "-c", inspection],
            "relocated_python_imports",
            environment,
            cwd=relocated,
        )
        run(
            [str(python), "-I", "-B", "-m", "stark_convert.cli", "--help"],
            "relocated_converter_help",
            environment,
            cwd=relocated,
        )
        run(
            [str(python), "-I", "-B", "-m", "stark_convert.cli", "--version"],
            "relocated_converter_version",
            environment,
            cwd=relocated,
        )
        launcher = relocated / "bin" / "stark"
        run(
            [str(launcher), "--help"],
            "relocated_launcher_help",
            environment,
            cwd=relocated,
        )
        run(
            [str(launcher), "--version"],
            "relocated_launcher_version",
            environment,
            cwd=relocated,
        )
        converter_version = tomllib.loads(
            (ROOT / "tools" / "converter" / "pyproject.toml").read_text()
        )["project"]["version"]
        version_reply = run(
            [str(launcher), "convert", "--version"],
            "relocated_launcher_converter_version",
            environment,
            cwd=relocated,
        )
        if (
            version_reply.stdout.decode().strip()
            != f"stark-convert {converter_version}"
            or version_reply.stderr
        ):
            raise BuildError("relocated_launcher_converter_version_identity")
        help_reply = run(
            [str(launcher), "convert", "--help"],
            "relocated_launcher_converter_help",
            environment,
            cwd=relocated,
        )
        if (
            re.search(r"(?m)^usage: stark-convert(?:\s|$)", help_reply.stdout.decode())
            is None
            or help_reply.stderr
        ):
            raise BuildError("relocated_launcher_converter_help_identity")
        for name in ("stark-score", "stark-generate"):
            verify_native(relocated / "libexec" / name, environment)
    finally:
        relocated.rename(bundle)


def file_manifest(bundle: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise BuildError("distribution_contains_symlink")
        if path.is_file():
            hashes[path.relative_to(bundle).as_posix()] = sha256(path)
        elif not path.is_dir():
            raise BuildError("distribution_contains_special_file")
    return hashes


def create_archive(bundle: Path, output: Path) -> str:
    """Emit regular files/directories only, with neutral ownership metadata."""

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        if not (info.isfile() or info.isdir()):
            raise BuildError("archive_contains_non_regular_entry")
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode &= ~0o022
        info.pax_headers = {}
        return info

    # A failed or interrupted build must not replace an earlier valid archive.
    partial = output.with_name(output.name + ".partial")
    try:
        with (
            partial.open("wb") as output_stream,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=output_stream, mtime=0
            ) as compressed,
            tarfile.open(
                fileobj=compressed,
                mode="w",
                dereference=True,
                format=tarfile.GNU_FORMAT,
            ) as archive,
        ):
            archive.add(bundle, arcname="stark", filter=normalize)
        digest = sha256(partial)
        partial.replace(output)
        output.with_name(output.name + ".sha256").write_text(
            f"{digest}  {output.name}\n"
        )
        return digest
    finally:
        partial.unlink(missing_ok=True)


def build(options: argparse.Namespace) -> Path:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise BuildError("build_requires_native_apple_silicon_macos")
    environment = clean_environment()
    macos = (
        run(["/usr/bin/sw_vers", "-productVersion"], "macos_version", environment)
        .stdout.decode()
        .strip()
    )
    if tuple(map(int, macos.split(".")[:2])) < (26, 4):
        raise BuildError("build_requires_macos_26_4_or_newer")
    uv, cargo = shutil.which("uv"), shutil.which("cargo")
    if not uv or not cargo:
        raise BuildError(
            "missing_build_tools: uv 0.12.24 and rustup/Cargo 1.98.1 are required"
        )
    uv_version = (
        run([uv, "--version"], "uv_version", environment).stdout.decode().split()
    )
    if uv_version[:2] != ["uv", UV_VERSION]:
        raise BuildError("build_requires_uv_0_12_24")
    source = source_provenance(environment, options.allow_dirty)
    output_directory = options.output_dir.resolve()
    # Keep generated artifacts in an ignored repository tree by default. An
    # explicit path is useful to maintainers preparing a local release handoff.
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"stark-{options.version}-{TARGET}.tar.gz"
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise BuildError(
            "distribution_version_already_exists: choose a new version or output directory"
        )
    (ROOT / "target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="stark-build-", dir=ROOT / "target"
    ) as temporary_name:
        temporary = Path(temporary_name)
        bundle = temporary / "stark"
        share = bundle / "share" / "stark"
        share.mkdir(parents=True)
        (bundle / "bin").mkdir()
        shutil.copyfile(ROOT / "scripts" / "stark", bundle / "bin" / "stark")
        (bundle / "bin" / "stark").chmod(0o755)
        shutil.copyfile(ROOT / "install.sh", share / "install.sh")
        (share / "VERSION").write_text(options.version + "\n")
        (share / "INSTALL_FORMAT").write_text("1\n")
        build_native(bundle, cargo, environment)
        build_python(bundle, temporary, uv, environment)
        dependency_licenses(bundle, cargo, environment)
        verify_relocation(bundle, environment)
        current_source = source_provenance(environment, options.allow_dirty)
        if (
            current_source["revision"] != source["revision"]
            or current_source["files_sha256"] != source["files_sha256"]
        ):
            raise BuildError("source_changed_during_build")
        manifest = {
            "schema_version": 1,
            "version": options.version,
            "target": TARGET,
            "minimum_macos": MIN_MACOS,
            "source": source,
            "build": {
                "rust": RUST_VERSION,
                "uv": UV_VERSION,
                "python": PYTHON_VERSION,
                "macos": macos,
                "scoring_features": ["optimized"],
                "generation_features": ["metal"],
                "native_builds_separate": True,
                "macosx_deployment_target": MIN_MACOS,
            },
            "validation": {
                "relocation": "passed",
                "native_invalid_arguments": "passed",
                "model_loaded": False,
            },
            "files_sha256": file_manifest(bundle),
        }
        (share / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        digest = create_archive(bundle, output)
    print(f"Bundle: {output}\nSHA-256: {digest}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        type=validate_version,
        help="immutable bundle version, e.g. 0.1.0-preview.1",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "distributions"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build a development bundle with recorded tracked diff hash",
    )
    options = parser.parse_args()
    try:
        build(options)
    except Exception as error:  # noqa: BLE001 - content-free command-line failure boundary
        message = (
            str(error) if isinstance(error, BuildError) else "distribution_build_failed"
        )
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
