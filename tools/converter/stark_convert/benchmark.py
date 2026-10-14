"""Measure streaming conversions in fresh processes against frozen outputs.

Model artifacts and the aggregate JSON report remain beneath a private output
directory. No downloads, inference, tensor contents, or exception dumps belong
to the timed runs or report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

CHUNK_BYTES = 8 * 1024 * 1024
RSS_SAMPLE_SECONDS = 0.01
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C64": 8,
}


class BenchmarkError(Exception):
    """A deliberately terse benchmark failure category."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(source).parts)
    )


def _source_digest(source: Path, files: list[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        digest.update(_json_bytes([str(path.relative_to(source)), size, _file_sha256(path)]))
        total += size
    return digest.hexdigest(), total


def _warm_source(files: list[Path]) -> None:
    for path in files:
        with path.open("rb") as handle:
            while handle.read(CHUNK_BYTES):
                pass


def _tensor_manifest(path: Path) -> tuple[dict[str, dict[str, object]], object]:
    """Independently validate headers and hash each tensor in bounded memory."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise BenchmarkError("invalid_output_header")
        header_bytes = struct.unpack("<Q", length_bytes)[0]
        if header_bytes > 100_000_000 or header_bytes > file_size - 8:
            raise BenchmarkError("invalid_output_header")
        try:
            header = json.loads(handle.read(header_bytes), object_pairs_hook=_reject_duplicate_keys)
        except (ValueError, TypeError):
            raise BenchmarkError("invalid_output_header") from None
        if not isinstance(header, dict):
            raise BenchmarkError("invalid_output_header")
        data_start = 8 + header_bytes
        data_bytes = file_size - data_start
        result = {}
        occupied = []
        for name in sorted(key for key in header if key != "__metadata__"):
            spec = header[name]
            if not name or not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}:
                raise BenchmarkError("invalid_output_header")
            dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in DTYPE_BYTES or not isinstance(shape, list):
                raise BenchmarkError("invalid_output_header")
            expected_bytes = DTYPE_BYTES[dtype]
            for dimension in shape:
                if type(dimension) is not int or dimension < 0:
                    raise BenchmarkError("invalid_output_header")
                expected_bytes *= dimension
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > data_bytes
                or offsets[1] - offsets[0] != expected_bytes
            ):
                raise BenchmarkError("invalid_output_header")
            if expected_bytes:
                occupied.append(tuple(offsets))
            digest = hashlib.sha256()
            handle.seek(data_start + offsets[0])
            remaining = expected_bytes
            while remaining:
                chunk = handle.read(min(remaining, CHUNK_BYTES))
                if not chunk:
                    raise BenchmarkError("truncated_output_tensor")
                digest.update(chunk)
                remaining -= len(chunk)
            result[name] = {"dtype": dtype, "shape": shape, "sha256": digest.hexdigest()}
        cursor = 0
        for start, end in sorted(occupied):
            if start != cursor:
                raise BenchmarkError("invalid_output_offsets")
            cursor = end
        if cursor != data_bytes:
            raise BenchmarkError("invalid_output_offsets")
    return result, header.get("__metadata__", {})


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError("invalid_output_json")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> object:
    raise BenchmarkError("invalid_output_json")


def _read_json(path: Path) -> object:
    try:
        with path.open() as handle:
            result = json.load(
                handle, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite_number
            )
        if not isinstance(result, dict):
            raise BenchmarkError("invalid_output_json")
        return result
    except (OSError, ValueError, TypeError):
        raise BenchmarkError("invalid_output_json") from None


def _tensor_specs(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in metadata.items()
    ):
        raise BenchmarkError("invalid_output_metadata")
    specs = {}
    try:
        for name, value in metadata.items():
            if name.endswith(".spec"):
                parsed = json.loads(
                    value, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite_number
                )
                if not isinstance(parsed, dict):
                    raise BenchmarkError("invalid_output_tensor_spec")
                specs[name] = parsed
    except (ValueError, TypeError):
        raise BenchmarkError("invalid_output_tensor_spec") from None
    return specs


def compare_outputs(streaming: Path, reference: Path) -> dict[str, object]:
    """Compare tensor bytes, matrix specifications, and runtime configuration.

    The reference output is generated independently before measurement. Its
    conversion time is not available here, so this comparison establishes
    output parity without producing a performance ratio.
    """
    fast_tensors, fast_metadata = _tensor_manifest(streaming / "model.safetensors")
    reference_tensors, reference_metadata = _tensor_manifest(reference / "model.safetensors")
    fast_specs, reference_specs = _tensor_specs(fast_metadata), _tensor_specs(reference_metadata)
    fast_config, reference_config = _read_json(streaming / "config.json"), _read_json(reference / "config.json")
    fast_tokenizer = _read_json(streaming / "tokenizer.json")
    reference_tokenizer = _read_json(reference / "tokenizer.json")
    tensor_hash = hashlib.sha256(_json_bytes(fast_tensors)).hexdigest()
    reference_hash = hashlib.sha256(_json_bytes(reference_tensors)).hexdigest()
    result = {
        "tensors_match": fast_tensors == reference_tensors,
        "tensor_metadata_match": fast_metadata == reference_metadata,
        "tensor_specs_match": fast_specs == reference_specs,
        "config_match": fast_config == reference_config,
        "tokenizer_match": fast_tokenizer == reference_tokenizer,
        "tensor_count": len(fast_tensors),
        "streaming_tensor_manifest_sha256": tensor_hash,
        "reference_tensor_manifest_sha256": reference_hash,
        "streaming_tensor_specs_sha256": hashlib.sha256(_json_bytes(fast_specs)).hexdigest(),
        "reference_tensor_specs_sha256": hashlib.sha256(_json_bytes(reference_specs)).hexdigest(),
        "streaming_config_sha256": hashlib.sha256(_json_bytes(fast_config)).hexdigest(),
        "reference_config_sha256": hashlib.sha256(_json_bytes(reference_config)).hexdigest(),
        "streaming_tokenizer_sha256": hashlib.sha256(_json_bytes(fast_tokenizer)).hexdigest(),
        "reference_tokenizer_sha256": hashlib.sha256(_json_bytes(reference_tokenizer)).hexdigest(),
        "streaming_weights_sha256": _file_sha256(streaming / "model.safetensors"),
        "reference_weights_sha256": _file_sha256(reference / "model.safetensors"),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "tensors_match",
            "tensor_metadata_match",
            "tensor_specs_match",
            "config_match",
            "tokenizer_match",
        )
    )
    return result


def _tree_rss(process: psutil.Process) -> int:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    total = 0
    for member in (process, *children):
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _stop_measured_child(child: subprocess.Popen, reader: threading.Thread | None) -> None:
    """Reap the owned process tree and receipt reader before propagating errors."""
    try:
        descendants = psutil.Process(child.pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        descendants = []

    def signal_group(value: int) -> None:
        if os.name == "posix":
            try:
                os.killpg(child.pid, value)
            except (ProcessLookupError, PermissionError):
                pass
        elif child.poll() is None:
            child.terminate() if value == signal.SIGTERM else child.kill()

    signal_group(signal.SIGTERM)
    for descendant in descendants:
        try:
            descendant.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGKILL)
    # A launcher may exit while descendants still hold stdout open. Force
    # their shutdown as well so joining the receipt reader cannot leak it.
    signal_group(signal.SIGKILL)
    for descendant in descendants:
        try:
            descendant.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    child.wait()
    if reader is not None:
        reader.join()


def _measure(command: list[str]) -> dict[str, object]:
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    started = time.perf_counter()
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=os.name == "posix",
    ) as child:
        received: dict[str, object] = {}
        reader = None

        def collect_receipt() -> None:
            try:
                received["stdout"], _ = child.communicate()
            except BaseException:
                # Exceptions can include child output; retain no diagnostic
                # text and suppress thread exception dumps.
                received["failed"] = True

        try:
            monitored = psutil.Process(child.pid)
            peak = _tree_rss(monitored)
            samples = 1
            reader = threading.Thread(target=collect_receipt, name="stark-benchmark-receipt")
            reader.start()
            while child.poll() is None:
                if received.get("failed"):
                    raise BenchmarkError("backend_receipt_failed")
                peak = max(peak, _tree_rss(monitored))
                samples += 1
                time.sleep(RSS_SAMPLE_SECONDS)
            reader.join()
            if received.get("failed"):
                raise BenchmarkError("backend_receipt_failed")
            stdout = received.get("stdout")
        except BaseException:
            _stop_measured_child(child, reader)
            raise
    elapsed = time.perf_counter() - started
    if child.returncode != 0:
        raise BenchmarkError("backend_conversion_failed")
    try:
        conversion = json.loads(
            stdout, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite_number
        )
    except (ValueError, TypeError, BenchmarkError):
        raise BenchmarkError("invalid_backend_metrics") from None
    if (
        not isinstance(conversion, dict)
        or conversion.get("backend") != "streaming"
        or conversion.get("status") != "ok"
    ):
        raise BenchmarkError("invalid_backend_metrics")
    seconds = conversion.get("elapsed_seconds")
    if type(seconds) not in (int, float) or seconds < 0 or seconds > sys.float_info.max:
        raise BenchmarkError("invalid_backend_metrics")
    if not math.isfinite(seconds):
        raise BenchmarkError("invalid_backend_metrics")
    for name in ("total_rss_bytes", "output_bytes", "tensor_count"):
        if type(conversion.get(name)) is not int or conversion[name] < 1:
            raise BenchmarkError("invalid_backend_metrics")
    return {
        "backend": conversion["backend"],
        "end_to_end_seconds": elapsed,
        "cli_conversion_seconds": seconds,
        "peak_process_tree_rss_bytes": peak,
        "rss_samples": samples,
        "reported_max_rss_bytes": conversion["total_rss_bytes"],
        "output_bytes": conversion["output_bytes"],
        "tensor_count": conversion["tensor_count"],
    }


def _summary(runs: list[dict[str, object]], backend: str) -> dict[str, object]:
    selected = [run for run in runs if run["backend"] == backend]
    times = [float(run["end_to_end_seconds"]) for run in selected]
    conversions = [
        float(run["cli_conversion_seconds"]) for run in selected if run["cli_conversion_seconds"] is not None
    ]
    rss = [int(run["peak_process_tree_rss_bytes"]) for run in selected]
    return {
        "runs": len(selected),
        "median_end_to_end_seconds": statistics.median(times),
        "min_end_to_end_seconds": min(times),
        "max_end_to_end_seconds": max(times),
        "median_cli_conversion_seconds": statistics.median(conversions) if conversions else None,
        "median_peak_process_tree_rss_bytes": statistics.median(rss),
        "max_peak_process_tree_rss_bytes": max(rss),
    }


def _write_report(output: Path, report: dict[str, object]) -> None:
    """Replace a complete private report; preserve the previous one on failure."""
    temporary = None
    try:
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output, prefix=".report-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "report.json")
    except (OSError, ValueError, TypeError):
        raise BenchmarkError("report_write_failed") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    from .cli import toolchain_provenance

    source = args.source.resolve()
    if not source.is_dir():
        raise BenchmarkError("missing_source_directory")
    files = _source_files(source)
    if not files:
        raise BenchmarkError("empty_source_directory")
    reference = getattr(args, "reference_output", None)
    reference_files = []
    reference_hash = None
    if reference is not None:
        reference = reference.resolve()
        if not reference.is_dir():
            raise BenchmarkError("missing_reference_directory")
        reference_files = _source_files(reference)
        reference_hash, _ = _source_digest(reference, reference_files)
        # Validate the independently generated reference before timed runs.
        compare_outputs(reference, reference)
    source_manifest = getattr(args, "source_manifest", None)
    source_manifest_hash = _file_sha256(source_manifest) if source_manifest is not None else None
    if args.output.resolve().is_relative_to(source):
        raise BenchmarkError("output_inside_source")
    if reference is not None and args.output.resolve().is_relative_to(reference):
        raise BenchmarkError("output_inside_reference")
    if args.output.exists():
        raise BenchmarkError("output_exists")
    # Hashing and provenance validation precede output creation and timing.
    tooling = toolchain_provenance()
    source_hash, source_bytes = _source_digest(source, files)
    report: dict[str, object] = {
        "schema_version": 3,
        "status": "running",
        "repo": args.repo,
        "dtype": args.dtype,
        "chunk_mib": args.chunk_mib,
        "model_kind": getattr(args, "model_kind", "language"),
        "architecture": getattr(args, "architecture", None),
        "metadata_profile": getattr(args, "metadata_profile", None),
        **tooling,
        "source_manifest_sha256": source_hash,
        "source_bytes": source_bytes,
        "source_replay_manifest_sha256": source_manifest_hash,
        "repeats": args.repeats,
        "reference_comparison_requested": reference is not None,
        "reference_output_manifest_sha256": reference_hash,
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "measurement": {
            "launcher": "current_python",
            "fresh_process_per_run": True,
            "source_cache_warmed_before_each_run": True,
            "setup_downloads_excluded": True,
            "backend": "streaming",
            "reference_timing_measured": False,
            "rss": "sampled sum of RSS for launcher and descendants; shared pages may be counted more than once",
            "rss_sample_interval_seconds": RSS_SAMPLE_SECONDS,
            "end_to_end": (
                "CLI startup, conversion, output verification and receipt hashing; output comparison excluded"
            ),
            "cli_conversion": "elapsed_seconds reported by the converter CLI",
        },
        "runs": [],
        "parity": [],
    }
    runs: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    report["runs"] = runs
    report["parity"] = comparisons
    args.output.mkdir(parents=True, mode=0o700)
    args.output.chmod(0o700)
    _write_report(args.output, report)
    try:
        _run_benchmark(args, report, files, source, source_manifest, reference)
        # Checkpoint changes during the benchmark invalidate source-bound evidence.
        if _source_digest(source, files)[0] != source_hash or _source_files(source) != files:
            report["status"] = "source_changed"
        if source_manifest is not None and _file_sha256(source_manifest) != source_manifest_hash:
            report["status"] = "source_changed"
        if reference is not None and (
            _source_digest(reference, reference_files)[0] != reference_hash
            or _source_files(reference) != reference_files
        ):
            report["status"] = "reference_changed"
        if toolchain_provenance() != tooling:
            report["status"] = "tooling_changed"
        if report["status"] == "running":
            report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_category"] = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else str(error)
            if isinstance(error, BenchmarkError)
            else "benchmark_failed"
        )
        report["failed_repeat"] = report.get("active_repeat")
    report.pop("active_repeat", None)
    _write_report(args.output, report)
    return report


def _run_benchmark(args, report, files, source, source_manifest, reference):
    runs = report["runs"]
    comparisons = report["parity"]
    for repeat in range(args.repeats):
        report["active_repeat"] = repeat + 1
        _write_report(args.output, report)
        _warm_source(files)
        destination = args.output / f"repeat-{repeat + 1}" / "streaming"
        command = [
            sys.executable,
            "-m",
            "stark_convert.cli",
            str(source),
            "--offline",
            "--output",
            str(destination),
            "--dtype",
            args.dtype,
            "--chunk-mib",
            str(args.chunk_mib),
        ]
        if args.repo is not None:
            command.extend(("--repo", args.repo))
        for name in ("model_kind", "architecture", "metadata_profile"):
            value = getattr(args, name, None)
            if value is not None:
                command.extend(("--" + name.replace("_", "-"), value))
        if source_manifest is not None:
            command.extend(("--source-manifest", str(source_manifest)))
        metrics = _measure(command)
        if metrics["backend"] != "streaming":
            raise BenchmarkError("unexpected_backend_selection")
        metrics.update(repeat=repeat + 1)
        runs.append(metrics)
        report["summary"] = {"streaming": _summary(runs, "streaming")}
        _write_report(args.output, report)
        if reference is not None:
            parity = compare_outputs(destination, reference)
            parity["repeat"] = repeat + 1
            comparisons.append(parity)
            if not parity["passed"]:
                report["status"] = "parity_failed"
                break


def main() -> int:
    from .cli import _Parser
    from .registry import CONFIG_ADAPTERS

    parser = _Parser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", help="Catalog identity for a local checkpoint")
    parser.add_argument("--metadata-profile", choices=("hf-generic",))
    parser.add_argument("--architecture", choices=sorted(set(CONFIG_ADAPTERS.values())))
    parser.add_argument("--model-kind", choices=("language", "classifier", "tts"), default="language")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk_mib < 1:
        print('{"status":"failed","error_category":"invalid_benchmark_options"}')
        return 1
    try:
        report = benchmark(args)
    except BaseException as error:
        category = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else str(error)
            if isinstance(error, BenchmarkError)
            else "benchmark_failed"
        )
        print(json.dumps({"status": "failed", "error_category": category}, separators=(",", ":")))
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if report.get("error_category") == "interrupted":
        return 130
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
