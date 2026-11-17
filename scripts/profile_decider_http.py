"""Profile the local HTTP Decision route with aggregate-only output.

The caller supplies an already prepared native Decider artifact and installed
worker executables. Request text, token IDs, logits, probabilities, API keys,
and exception details are never written to the output file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import plistlib
import re
import secrets
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

METRIC_KEYS = (
    "state_creation_seconds",
    "prefill_seconds",
    "readout_seconds",
    "total_seconds",
)
MODEL_MARKERS = (
    "stark-decide",
    "stark-generate",
    "stark-score",
    "mlx",
    "qwen",
    "llama",
    "tensorfold",
    "vllm",
    "ollama",
)
UI_GPU_NAMES = {
    "dock",
    "controlcenter",
    "windowserver",
    "windowmanager",
    "loginwindow",
    "screencapture",
    "iterm2",
    "codex",
    "codex desktop",
    "codex (aperitif)",
    "codex (aperitif renderer)",
    "google chrome helper (gpu)",
}


class ProfileError(Exception):
    """Fixed, content-free profile error."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ProfileError("empty_measurements")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_rows(rows: list[dict], warmups: int) -> list[dict]:
    """Keep only safe counters and phase timing summaries."""
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["requested_words"], []).append(row)
    result = []
    for requested_words, group in sorted(grouped.items()):
        measured = group[warmups:]
        if not measured:
            raise ProfileError("missing_measurements")
        metrics = {}
        for key in (
            "http_wall_seconds",
            "encode_seconds",
            "native_roundtrip_seconds",
            *METRIC_KEYS,
        ):
            values = [row[key] for row in measured if row.get(key) is not None]
            metrics[key] = summarize(values) if values else None
        result.append(
            {
                "requested_words": requested_words,
                "samples": len(measured),
                "input_tokens": summarize(
                    [float(row["input_tokens"]) for row in measured]
                ),
                "metrics": metrics,
            }
        )
    return result


def process_snapshot() -> dict[int, tuple[int, str]]:
    try:
        raw = subprocess.check_output(
            ["/bin/ps", "-A", "-o", "pid=,ppid=,comm="],
            text=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        raise ProfileError("process_snapshot_unavailable") from None
    result = {}
    for line in raw.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3:
            continue
        try:
            result[int(fields[0])] = (int(fields[1]), Path(fields[2]).name.lower())
        except ValueError:
            continue
    if not result:
        raise ProfileError("process_snapshot_unavailable")
    return result


def gpu_counters() -> dict[int, int]:
    try:
        raw = subprocess.check_output(
            ["/usr/sbin/ioreg", "-r", "-c", "IOGPUDeviceUserClient", "-a"], timeout=8
        )
        devices = plistlib.loads(raw)
    except (
        OSError,
        subprocess.SubprocessError,
        plistlib.InvalidFileException,
        ValueError,
        TypeError,
    ):
        raise ProfileError("gpu_attribution_unavailable") from None
    if not isinstance(devices, list):
        raise ProfileError("gpu_attribution_unavailable")
    counters: dict[int, int] = {}
    for device in devices:
        if not isinstance(device, dict):
            continue
        creator = device.get("IOUserClientCreator", "")
        prefix, _, _ = creator.partition(",")
        parts = prefix.split()
        if len(parts) != 2 or parts[0] != "pid" or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        usage_rows = device.get("AppUsage", [])
        if not isinstance(usage_rows, list):
            continue
        for usage in usage_rows:
            if (
                isinstance(usage, dict)
                and usage.get("API") == "Metal"
                and type(usage.get("accumulatedGPUTime")) is int
                and usage["accumulatedGPUTime"] >= 0
            ):
                counters[pid] = counters.get(pid, 0) + usage["accumulatedGPUTime"]
    return counters


def reject_foreign_compute(owned_pids: set[int]) -> None:
    """Fail closed on competing model processes or observed foreign GPU work."""
    processes = process_snapshot()
    own_tree = set(owned_pids)
    changed = True
    while changed:
        changed = False
        for pid, (parent, _name) in processes.items():
            if parent in own_tree and pid not in own_tree:
                own_tree.add(pid)
                changed = True
    for pid, (_parent, name) in processes.items():
        if pid in own_tree:
            continue
        if any(marker in name for marker in MODEL_MARKERS):
            raise ProfileError("foreign_model_process_detected")

    before = gpu_counters()
    time.sleep(1.0)
    after = gpu_counters()
    current = process_snapshot()
    for pid, counter in after.items():
        if pid in own_tree:
            continue
        previous = before.get(pid)
        if previous is None or counter <= previous:
            continue
        name = current.get(pid, (0, ""))[1]
        if name in UI_GPU_NAMES:
            continue
        raise ProfileError("foreign_gpu_activity_detected")


def generated_state(words: int) -> str:
    # Distinct deterministic words prevent a constant repeated-token prompt.
    return " ".join(f"record{index:04d}" for index in range(words))


def generated_png(width: int = 32, height: int = 32) -> bytes:
    """Create a small deterministic RGB PNG using only the standard library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\x00"
        + bytes(
            component
            for x in range(width)
            for component in (x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256)
        )
        for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=6))
        + chunk(b"IEND", b"")
    )


def api_call(
    base_url: str, api_key: str, model_id: str, payload: dict, timeout: float
) -> None:
    request = urllib.request.Request(
        f"{base_url}/decisions",
        data=json.dumps({"model": model_id, **payload}, ensure_ascii=False).encode(
            "utf-8"
        ),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ProfileError("http_request_failed")
            # Parse but deliberately discard response content, including probabilities.
            json.load(response)
    except urllib.error.HTTPError as error:
        try:
            code = json.load(error).get("error", {}).get("code", "unknown")
        except (ValueError, AttributeError):
            code = "unknown"
        if not isinstance(code, str) or re.fullmatch(r"[a-z_]{1,80}", code) is None:
            code = "unknown"
        raise ProfileError(f"http_{error.code}_{code}") from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise ProfileError("http_request_failed") from None


def private_write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args) -> dict:
    if sys.platform != "darwin":
        raise ProfileError("macos_required")
    model = args.model.resolve(strict=True)
    worker_path = args.decide_worker.resolve(strict=True)
    generate_worker = shutil.which("stark-generate")
    template_helper = shutil.which("stark-template")
    if (
        not model.is_dir()
        or not worker_path.is_file()
        or not generate_worker
        or not template_helper
    ):
        raise ProfileError("input_path_invalid")
    if args.repetitions < 1 or args.repetitions > 20 or not args.word_counts:
        raise ProfileError("invalid_profile_options")
    if any(count < 0 or count > 256 for count in args.word_counts):
        raise ProfileError("word_count_out_of_range")

    from stark_convert.desktop import Desktop

    hashes = {
        name: file_hash(model / filename)
        for name, filename in (
            ("config_sha256", "config.json"),
            ("weights_sha256", "model.safetensors"),
            ("tokenizer_sha256", "tokenizer.json"),
            ("template_sha256", "chat_template.jinja"),
        )
    }
    template_helper_hash = file_hash(Path(template_helper))
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        timeout=5,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            timeout=5,
        ).strip()
    )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="stark-decider-http-") as temporary:
        private_root = Path(temporary)
        private_root.chmod(0o700)
        registry_record = [
            {
                "id": "decider-profile",
                "name": "Decider profile artifact",
                "path": str(model),
                "source": "private-profile-input",
                "artifact_mode": "decision",
            }
        ]
        registry = private_root / "models.json"
        registry.write_text(json.dumps(registry_record), encoding="utf-8")
        registry.chmod(0o600)
        desktop = Desktop(
            Path(generate_worker),
            private_root,
            output=io.StringIO(),
            timeout=args.timeout,
            decide_executable=worker_path,
        )
        try:
            # Reject competing model jobs before loading this artifact.
            reject_foreign_compute({os.getpid()})
            load_started = time.perf_counter()
            desktop.run(desktop.epoch, "decider-profile", "decision")
            load_seconds = time.perf_counter() - load_started
            worker = desktop.worker
            if worker is None or getattr(worker, "backend", None) != "decision-native":
                raise ProfileError("native_decider_not_ready")

            encode_timings: list[float] = []
            native_rows: list[dict] = []
            original_plan = worker.encoder.plan
            original_request = worker.request

            def measured_plan(*plan_args, **plan_kwargs):
                encode_start = time.perf_counter()
                result = original_plan(*plan_args, **plan_kwargs)
                encode_timings.append(time.perf_counter() - encode_start)
                return result

            def measured_request(payload, *request_args, **request_kwargs):
                request_start = time.perf_counter()
                result = original_request(payload, *request_args, **request_kwargs)
                elapsed = time.perf_counter() - request_start
                metrics = result.get("metrics")
                if not isinstance(metrics, dict) or any(
                    type(metrics.get(key)) not in (int, float) or metrics[key] < 0
                    for key in METRIC_KEYS
                ):
                    raise ProfileError("native_metrics_missing")
                native_rows.append(
                    {
                        "native_roundtrip_seconds": elapsed,
                        **{key: float(metrics[key]) for key in METRIC_KEYS},
                        "input_tokens": result.get("input_tokens"),
                    }
                )
                return result

            worker.encoder.plan = measured_plan
            worker.request = measured_request
            key = secrets.token_hex(32)
            desktop.start_api(key, 0)
            if desktop.api is None or desktop.api_port is None:
                raise ProfileError("local_api_start_failed")
            model_id = desktop._api_models()[0]["id"]
            base_url = f"http://127.0.0.1:{desktop.api_port}/v1"
            owned_pids = {os.getpid(), worker.process.pid}
            rows = []
            for word_count in args.word_counts:
                for repetition in range(args.repetitions + 1):
                    reject_foreign_compute(owned_pids)
                    encode_before, native_before = len(encode_timings), len(native_rows)
                    payload = {
                        "state": generated_state(word_count),
                        "questions": {
                            "decision": {
                                "type": "choice",
                                "instructions": "Select the option associated with the recorded context.",
                                "criteria": {
                                    "A": "The first measured option.",
                                    "B": "The second measured option.",
                                },
                            }
                        },
                    }
                    http_start = time.perf_counter()
                    api_call(base_url, key, model_id, payload, args.timeout)
                    http_elapsed = time.perf_counter() - http_start
                    if (
                        len(encode_timings) != encode_before + 1
                        or len(native_rows) != native_before + 1
                    ):
                        raise ProfileError("unexpected_worker_call_count")
                    phase = native_rows[-1]
                    if type(phase.get("input_tokens")) is not int:
                        raise ProfileError("input_count_missing")
                    rows.append(
                        {
                            "requested_words": word_count,
                            "http_wall_seconds": http_elapsed,
                            "encode_seconds": encode_timings[-1],
                            **phase,
                        }
                    )
                    print(
                        f"Decider HTTP sample completed: size {word_count}, call {repetition + 1}.",
                        flush=True,
                    )
            measured = aggregate_rows(rows, warmups=1)
            image_smoke = None
            if args.image_smoke:
                reject_foreign_compute(owned_pids)
                image_bytes = generated_png()
                image_payload = {
                    "state": "Classify the simple geometric image.",
                    "questions": {
                        "decision": {
                            "type": "choice",
                            "instructions": "Select the option associated with the image.",
                            "criteria": {
                                "A": "The first measured option.",
                                "B": "The second measured option.",
                            },
                        }
                    },
                    "images": [
                        {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        }
                    ],
                }
                encode_before, native_before = len(encode_timings), len(native_rows)
                http_start = time.perf_counter()
                api_call(base_url, key, model_id, image_payload, args.timeout)
                http_elapsed = time.perf_counter() - http_start
                if (
                    len(encode_timings) != encode_before + 1
                    or len(native_rows) != native_before + 1
                ):
                    raise ProfileError("unexpected_image_worker_call_count")
                phase = native_rows[-1]
                if type(phase.get("input_tokens")) is not int:
                    raise ProfileError("image_input_count_missing")
                image_smoke = {
                    "http_success": True,
                    "samples": 1,
                    "expanded_input_tokens": phase["input_tokens"],
                    "metrics": {
                        "http_wall_seconds": summarize([http_elapsed]),
                        "encode_seconds": summarize([encode_timings[-1]]),
                        **{
                            key: summarize([phase[key]])
                            for key in ("native_roundtrip_seconds", *METRIC_KEYS)
                        },
                    },
                }
            worker_sha = file_hash(worker_path)
        finally:
            if desktop.api is not None:
                desktop.api.close()
            desktop.stop(quiet=True)
            desktop.output.close()

    return {
        "schema_version": 1,
        "scope": "local HTTP /v1/decisions; single loaded native Decider; text only",
        "source_revision": git_revision,
        "source_dirty": dirty,
        "worker_sha256": worker_sha,
        "template_helper_sha256": template_helper_hash,
        "artifact": hashes,
        "platform": platform.platform(),
        "load_seconds_excluded": load_seconds,
        "warmups_per_input_length": 1,
        "repetitions_per_input_length": args.repetitions,
        "word_counts": args.word_counts,
        "samples": measured,
        "image_smoke": image_smoke,
        "elapsed_seconds_including_load": time.perf_counter() - started,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decide-worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--word-counts", type=int, nargs="+", default=[0, 64, 256])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--image-smoke",
        action="store_true",
        help="add one generated 32x32 PNG API request",
    )
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    try:
        record = run(args)
        private_write(args.output, record)
    except ProfileError as error:
        print(f"Decider HTTP profile stopped: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - never print exception details or request contents.
        print("Decider HTTP profile stopped: profile_failed", file=sys.stderr)
        return 2
    print("Decider HTTP aggregate profile written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
