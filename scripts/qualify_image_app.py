#!/usr/bin/env python3
"""Opt-in, real-model image check of a clean, isolated packaged installation.

Uses only the copied application's interpreter, package and workers at runtime.
The result contains aggregate checks and delivered-file hashes, never API keys,
image bytes, prompts, generated text, model paths or exception details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import select
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import time
import uuid
import zlib
from pathlib import Path

PYTHON = Path("libexec/python/bin/python3.13")
PACKAGE = Path("libexec/python/lib/python3.13/site-packages/stark_convert")
EVIDENCE_LEVELS = ("reference_verified", "locally_tested")
MAX_REPLY_BYTES = 65536
LOCAL_CHECKS = {
    "text_generation",
    "image_generation",
    "image_token_accounting",
    "recoverable_image_error",
    "fresh_image_replay",
    "staging_cleanup",
}
ADAPTATION_NOTICES = (
    "pytorch-LICENSE",
    "ggml-LICENSE",
    "sleef-LICENSE.txt",
    "sleef-ATTRIBUTION.txt",
    "arm-optimized-routines-LICENSE.txt",
    "arm-optimized-routines-ATTRIBUTION.txt",
    "transformers-LICENSE",
    "transformers-ATTRIBUTION.txt",
    "LICENSE.mlx-reference",
)


class QualificationError(Exception):
    """Fixed, content-free failure stage."""


def require(value: object, stage: str) -> None:
    if not value:
        raise QualificationError(stage)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(app: Path) -> dict[str, str]:
    result = {}
    for path in sorted(app.rglob("*")):
        metadata = path.lstat()
        require(
            stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode),
            "bundle_entry_invalid",
        )
        if stat.S_ISREG(metadata.st_mode):
            result[path.relative_to(app).as_posix()] = digest(path)
    require(result, "bundle_empty")
    return result


def verify_native_notices(runtime: Path, manifest: dict) -> None:
    record = manifest.get("native_license_notices", {})
    require(record.get("rust_version") == "1.98.1", "native_notice_toolchain_missing")
    require(
        record.get("cargo_lock_sha256")
        == manifest.get("source", {}).get("files_sha256", {}).get("Cargo.lock")
        and isinstance(record.get("cargo_lock_sha256"), str),
        "native_notice_source_mismatch",
    )
    overlays = manifest.get(
        "signed_overlays_sha256", manifest.get("overlays_sha256", {})
    )
    for name in (*ADAPTATION_NOTICES, "cargo-inventory.json"):
        relative = "share/stark/licenses/" + name
        require(
            relative in overlays and digest(runtime / relative) == overlays[relative],
            "native_notice_missing_or_changed",
        )
    document = json.loads(
        (runtime / "share/stark/licenses/cargo-inventory.json").read_bytes()
    )
    names = {package["name"] for package in document["packages"]}
    require(
        {"image", "jpeg-decoder", "png", "fdeflate"} <= names,
        "image_decoder_dependency_notices_missing",
    )


def run(arguments: list[str | Path], environment: dict, cwd: Path) -> bytes:
    try:
        reply = subprocess.run(
            arguments,
            env=environment,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise QualificationError("package_command_unavailable") from None
    require(reply.returncode == 0, "package_command_failed")
    return reply.stdout


def circle_png() -> bytes:
    """Public synthetic 256-pixel black circle; native Rust remains the decoder."""
    side = 256
    rows = bytearray()
    for y in range(side):
        rows.append(0)
        for x in range(side):
            value = 0 if (x - 128) ** 2 + (y - 128) ** 2 <= 72**2 else 255
            rows.extend((value, value, value))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


def metadata_bytes(path: Path, maximum: int, stage: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, stage)
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(
        len(data) == before.st_size
        and all(
            getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        stage,
    )
    return data


class InstallObservation:
    """Observe checking before the first published model-library event."""

    def __init__(self, schema: int):
        self.schema = schema
        self.checking = False
        self.published = False
        self.installed = False

    def observe(self, event: dict) -> None:
        if event.get("event") == "image_check" and event.get("state") == "checking":
            require(not self.published, "image_check_after_model_publication")
            self.checking = True
        if event.get("event") == "models" and event.get("models"):
            require(self.schema != 3 or self.checking, "install_image_check_missing")
            self.published = True
        if event.get("event") == "installed":
            require(self.schema != 3 or self.checking, "install_image_check_missing")
            self.installed = True


def verify_readiness(ready: dict, evidence_level: str) -> None:
    require(ready.get("images") is True, "packaged_worker_unqualified")
    require(
        ready.get("image_verification") == evidence_level,
        "image_evidence_level_mismatch",
    )
    require(
        ready.get("image_attachment_transport") == "private-file-v1",
        "attachment_transport_invalid",
    )


def generation_limits(settings: dict) -> tuple[int, int]:
    context, maximum = settings.get("context_tokens"), settings.get("max_new_tokens")
    require(
        type(context) is int and 2 <= context <= 16384, "generation_context_invalid"
    )
    require(
        type(maximum) is int and 0 < maximum <= min(4096, context - 1),
        "generation_limit_invalid",
    )
    return context, maximum


def verify_reply(reply: dict, evidence_level: str, context: int, maximum: int) -> None:
    text = reply.get("text")
    require(
        isinstance(text, str)
        and bool(text.strip())
        and len(text.encode("utf-8")) <= MAX_REPLY_BYTES,
        "generation_missing_or_oversize",
    )
    require(
        type(reply.get("input_tokens")) is int
        and reply["input_tokens"] > 0
        and type(reply.get("output_tokens")) is int
        and 0 < reply["output_tokens"] <= maximum
        and reply["input_tokens"] + reply["output_tokens"] <= context,
        "usage_missing_or_out_of_bounds",
    )
    require(
        reply.get("finish_reason")
        in {"eos", "stop_token", "stop_sequence", "length", "context"},
        "generation_finish_invalid",
    )
    if evidence_level == "reference_verified":
        require("circle" in text.lower(), "synthetic_shape_mismatch")


def receipt_snapshot(root: Path) -> dict:
    """Keep receipt contents private; compare immutable identity between runs."""
    require(root.is_dir(), "local_receipt_missing")
    paths = sorted(root.iterdir())
    require(len(paths) == 1, "local_receipt_inventory_invalid")
    path = paths[0]
    require(
        re.fullmatch(r"[0-9a-f]{64}\.json", path.name), "local_receipt_name_invalid"
    )
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode)
        and before.st_uid == os.getuid()
        and stat.S_IMODE(before.st_mode) == 0o600
        and before.st_nlink == 1,
        "local_receipt_permissions_invalid",
    )
    data = metadata_bytes(path, 65536, "local_receipt_invalid")
    receipt = json.loads(data)
    require(
        isinstance(receipt, dict)
        and type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("evidence_level") == "locally_tested"
        and isinstance(receipt.get("identity"), dict)
        and receipt["identity"]
        and isinstance(receipt.get("checks"), dict)
        and set(receipt["checks"]) == LOCAL_CHECKS
        and all(value is True for value in receipt["checks"].values()),
        "local_receipt_invalid",
    )
    after = path.lstat()
    require(
        all(
            getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "local_receipt_changed",
    )
    return {
        "name": path.name,
        "identity": tuple(
            getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class Backend:
    def __init__(self, arguments: list[str | Path], environment: dict, cwd: Path):
        self.process = subprocess.Popen(
            arguments,
            env=environment,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.pending = b""

    def send(self, value: dict) -> None:
        assert self.process.stdin is not None
        frame = json.dumps(value).encode() + b"\n"
        require(len(frame) <= 1024 * 1024, "command_frame_oversize")
        self.process.stdin.write(frame)
        self.process.stdin.flush()

    def until(
        self,
        predicate,
        *,
        timeout: float = 180,
        allow_error: bool = False,
        observe=None,
    ) -> dict:
        assert self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                require(len(line) <= 1024 * 1024, "backend_frame_oversize")
                event = json.loads(line)
                require(isinstance(event, dict), "backend_frame_invalid")
                require(
                    allow_error or event.get("event") != "error",
                    "backend_rejected_request",
                )
                if observe is not None:
                    observe(event)
                if predicate(event):
                    return event
                continue
            if not select.select(
                [self.process.stdout],
                [],
                [],
                max(0, min(1, deadline - time.monotonic())),
            )[0]:
                continue
            block = os.read(self.process.stdout.fileno(), 65536)
            require(block, "backend_exited")
            self.pending += block
            require(len(self.pending) <= 2 * 1024 * 1024, "backend_frame_oversize")
        raise QualificationError("backend_deadline")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send({"command": "quit"})
                self.process.wait(timeout=10)
            except (OSError, BrokenPipeError, subprocess.TimeoutExpired):
                pass
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


def qualify(app: Path, model: Path, evidence_level: str = "reference_verified") -> dict:
    require(evidence_level in EVIDENCE_LEVELS, "image_evidence_level_invalid")
    app, model = app.resolve(strict=True), model.resolve(strict=True)
    config_bytes = metadata_bytes(
        model / "config.json", 1024 * 1024, "model_config_invalid"
    )
    config = json.loads(config_bytes)
    require(
        isinstance(config, dict) and isinstance(config.get("artifact_contract"), dict),
        "image_model_schema_invalid",
    )
    schema = config["artifact_contract"].get("schema_version")
    require(
        config.get("type") == "Qwen35VisionLanguageModelConfig"
        and type(schema) is int
        and schema in (2, 3),
        "image_model_schema_invalid",
    )
    model_config_hash = hashlib.sha256(config_bytes).hexdigest()
    metadata = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    require(
        metadata.get("CFBundleIdentifier") == "com.starkengine.menubar",
        "bundle_identity_invalid",
    )
    source_inventory = inventory(app)
    with tempfile.TemporaryDirectory(prefix="stark-image-app-qualification-") as name:
        work = Path(name).resolve()
        work.chmod(0o700)
        installed = work / "clean installation with spaces/Applications/Stark.app"
        installed.parent.mkdir(parents=True)
        shutil.copytree(app, installed)
        require(inventory(installed) == source_inventory, "installed_bundle_changed")
        home, temporary, attachment_root = (
            work / "home",
            work / "temporary",
            work / "attachments",
        )
        for path in (home, temporary, attachment_root):
            path.mkdir(mode=0o700)
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "HF_HOME": str(work / "empty-hub"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/invalid-developer-python",
            "PYTHONHOME": "/invalid-developer-python",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C",
        }
        run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", installed],
            environment,
            work,
        )
        runtime = installed / "Contents/Resources/runtime"
        manifest = json.loads(
            (installed / "Contents/Resources/menu-bar-manifest.json").read_bytes()
        )
        verify_native_notices(runtime, manifest)
        inspection = (
            "import importlib,json,sys;from pathlib import Path;"
            "root=Path(sys.argv[1]).resolve();"
            "mods=[importlib.import_module('stark_convert.'+n) for n in "
            "['desktop','desktop_media','media','vision_runtime','chat','image_checks','vision_family']];"
            "assert all(Path(m.__file__).resolve().is_relative_to(root) for m in mods);"
            "assert Path(sys.executable).resolve().is_relative_to(root);"
            "assert (Path(mods[0].__file__).parent/'vision_runtime_qualifications.json').is_file();"
            "print(json.dumps({'isolated_package_imports':True}))"
        )
        result = json.loads(
            run(
                [runtime / PYTHON, "-I", "-B", "-c", inspection, runtime],
                environment,
                work,
            )
        )
        require(
            result == {"isolated_package_imports": True},
            "package_import_origin_invalid",
        )
        backend = Backend(
            [
                runtime / PYTHON,
                "-I",
                "-B",
                "-m",
                "stark_convert.desktop",
                "--worker",
                runtime / "libexec/stark-generate",
                "--score-worker",
                runtime / "libexec/stark-score",
                "--data-dir",
                home / "model-library",
                "--attachment-root",
                attachment_root,
            ],
            environment,
            work,
        )
        try:
            backend.until(lambda event: event.get("event") == "models")
            installation = InstallObservation(schema)
            backend.send({"command": "install", "kind": "local", "source": str(model)})
            models = backend.until(
                lambda event: event.get("event") == "models" and event.get("models"),
                observe=installation.observe,
            )
            backend.until(
                lambda event: (
                    event.get("event") == "status"
                    and event.get("message") == "Model installed"
                ),
                observe=installation.observe,
            )
            require(
                installation.installed and installation.published,
                "install_publication_missing",
            )
            identifier = models["models"][0]["id"]
            registry_path = home / "model-library/models.json"
            registry = json.loads(
                metadata_bytes(registry_path, 1024 * 1024, "model_registry_invalid")
            )
            require(
                isinstance(registry, list)
                and len(registry) == 1
                and registry[0].get("id") == identifier,
                "model_registry_invalid",
            )
            if schema == 3:
                require(
                    registry[0].get("image_verification") == "locally_tested",
                    "registry_image_evidence_invalid",
                )
            receipt = None
            if evidence_level == "locally_tested" and schema == 3:
                receipt = receipt_snapshot(home / "model-library/image-checks")
                reinstallation = InstallObservation(schema)
                backend.send(
                    {"command": "install", "kind": "local", "source": str(model)}
                )
                repeated = backend.until(
                    lambda event: event.get("event") == "installed",
                    observe=reinstallation.observe,
                )
                require(
                    repeated.get("model", {}).get("id") == identifier,
                    "reinstall_model_identity_changed",
                )
                backend.until(
                    lambda event: (
                        event.get("event") == "status"
                        and event.get("message") == "Model already installed"
                    )
                )
                require(
                    receipt_snapshot(home / "model-library/image-checks") == receipt,
                    "reinstall_receipt_replaced",
                )
            backend.send({"command": "run", "model_id": identifier, "mode": "chat"})
            ready = backend.until(lambda event: event.get("event") == "ready")
            verify_readiness(ready, evidence_level)
            backend.until(
                lambda event: (
                    event.get("event") == "status" and event.get("state") == "ready"
                )
            )
            if receipt is not None:
                require(
                    receipt_snapshot(home / "model-library/image-checks") == receipt,
                    "run_receipt_replaced",
                )
            if evidence_level == "locally_tested":
                context_limit = ready.get("max_context_tokens")
                output_limit = ready.get("max_output_tokens")
                require(
                    type(context_limit) is int and context_limit >= 512,
                    "generation_context_invalid",
                )
                require(
                    type(output_limit) is int and output_limit > 0,
                    "generation_limit_invalid",
                )
                context = min(2048, context_limit)
                maximum = min(32, output_limit, context - 1)
                backend.send(
                    {
                        "command": "configure",
                        "context_tokens": context,
                        "max_new_tokens": maximum,
                        "system_prompt": None,
                        "thinking": False,
                    }
                )
                ready = backend.until(lambda event: event.get("event") == "configured")
                verify_readiness(ready, evidence_level)
                require(
                    generation_limits(ready) == (context, maximum),
                    "generation_configuration_mismatch",
                )
            context, maximum = generation_limits(ready)
            image = circle_png()
            outputs = []
            for turn in range(2):
                media_id = uuid.uuid4().hex
                descriptor = os.open(
                    attachment_root / media_id,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(image)
                backend.send(
                    {
                        "command": "generate",
                        "text": "Name the black shape in one word."
                        if turn == 0
                        else "Name that same shape again in one word.",
                        "image": {
                            "id": media_id,
                            "media_type": "image/png",
                            "byte_length": len(image),
                        },
                    }
                )
                reply = backend.until(lambda event: event.get("event") == "reply")
                outputs.append(reply)
                backend.until(
                    lambda event: (
                        event.get("event") == "status" and event.get("state") == "ready"
                    )
                )
                require(not list(attachment_root.iterdir()), "attachment_file_retained")
                verify_reply(reply, evidence_level, context, maximum)
        finally:
            backend.close()
        require(not list(attachment_root.iterdir()), "attachment_cleanup_failed")
        require(
            not any(
                path.name.startswith("stark-media-") for path in temporary.iterdir()
            ),
            "native_media_cleanup_failed",
        )
        require(inventory(installed) == source_inventory, "installed_bundle_mutated")
        require(
            digest(model / "config.json") == model_config_hash, "model_config_changed"
        )
        canonical = json.dumps(
            source_inventory, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            "schema_version": 2,
            "passed": True,
            "evidence_level": evidence_level,
            "artifact_schema_version": schema,
            "model_config_sha256": model_config_hash,
            "bundle_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
            "bundle_file_count": len(source_inventory),
            "worker_sha256": digest(runtime / "libexec/stark-generate"),
            "swift_executable_sha256": digest(installed / "Contents/MacOS/Stark"),
            "catalog_sha256": digest(
                runtime / PACKAGE / "vision_runtime_qualifications.json"
            ),
            "checks": {
                "clean_install_copy_and_signature": True,
                "isolated_embedded_interpreter_and_imports": True,
                "ordinary_catalog_qualification": evidence_level
                == "reference_verified",
                "local_execution_checks": evidence_level == "locally_tested",
                "ready_evidence_level_matches": True,
                "checking_before_model_publication": installation.checking
                if schema == 3
                else None,
                "registry_image_evidence_matches": True if schema == 3 else None,
                "repeat_install_preserves_model_identity": True
                if receipt is not None
                else None,
                "local_receipt_unchanged_on_reinstall_and_run": True
                if receipt is not None
                else None,
                "current_native_and_adaptation_notices": True,
                "image_turns": len(outputs),
                "synthetic_circle_matches": len(outputs)
                if evidence_level == "reference_verified"
                else None,
                "followup_resupplies_image": True,
                "usage_present": True,
                "attachment_and_worker_media_cleanup": True,
                "installed_bundle_unchanged": True,
            },
            "limits": {
                "swift_controller": "separate_harness",
                "notarization": "not_checked",
                "generated_answer_quality": "synthetic_circle_predicate"
                if evidence_level == "reference_verified"
                else "not_checked",
            },
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-level", choices=EVIDENCE_LEVELS, default="reference_verified"
    )
    args = parser.parse_args(argv)
    try:
        result = qualify(args.app, args.model, args.evidence_level)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps({"passed": True, "checks": result["checks"]}, sort_keys=True))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        QualificationError,
        subprocess.SubprocessError,
    ) as error:
        stage = (
            str(error)
            if isinstance(error, QualificationError)
            else "qualification_failed"
        )
        print(json.dumps({"passed": False, "stage": stage}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
