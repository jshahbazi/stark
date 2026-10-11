"""Compare local streaming and pinned Lalamo conversions in fresh processes.

Model artifacts and the aggregate JSON report remain beneath a private output
directory. No downloads, inference, tensor contents, or exception dumps belong
to the timed runs or report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import struct
import subprocess
import time
from pathlib import Path

import psutil

from .lalamo_runner import LALAMO_REVISION

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
        header = json.loads(handle.read(header_bytes))
        if not isinstance(header, dict):
            raise BenchmarkError("invalid_output_header")
        data_start = 8 + header_bytes
        data_bytes = file_size - data_start
        result = {}
        occupied = []
        for name in sorted(key for key in header if key != "__metadata__"):
            spec = header[name]
            dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
            if dtype not in DTYPE_BYTES or not isinstance(shape, list):
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


def _read_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def compare_outputs(streaming: Path, lalamo: Path) -> dict[str, object]:
    """Compare every emitted tensor byte and all runtime configuration."""
    fast_tensors, fast_metadata = _tensor_manifest(streaming / "model.safetensors")
    reference_tensors, reference_metadata = _tensor_manifest(lalamo / "model.safetensors")
    fast_config, reference_config = _read_json(streaming / "config.json"), _read_json(lalamo / "config.json")
    fast_tokenizer = _read_json(streaming / "tokenizer.json")
    reference_tokenizer = _read_json(lalamo / "tokenizer.json")
    tensor_hash = hashlib.sha256(_json_bytes(fast_tensors)).hexdigest()
    reference_hash = hashlib.sha256(_json_bytes(reference_tensors)).hexdigest()
    result = {
        "tensors_match": fast_tensors == reference_tensors,
        "tensor_metadata_match": fast_metadata == reference_metadata,
        "config_match": fast_config == reference_config,
        "tokenizer_match": fast_tokenizer == reference_tokenizer,
        "tensor_count": len(fast_tensors),
        "streaming_tensor_manifest_sha256": tensor_hash,
        "lalamo_tensor_manifest_sha256": reference_hash,
        "streaming_config_sha256": hashlib.sha256(_json_bytes(fast_config)).hexdigest(),
        "lalamo_config_sha256": hashlib.sha256(_json_bytes(reference_config)).hexdigest(),
        "streaming_tokenizer_sha256": hashlib.sha256(_json_bytes(fast_tokenizer)).hexdigest(),
        "lalamo_tokenizer_sha256": hashlib.sha256(_json_bytes(reference_tokenizer)).hexdigest(),
        "streaming_weights_sha256": _file_sha256(streaming / "model.safetensors"),
        "lalamo_weights_sha256": _file_sha256(lalamo / "model.safetensors"),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "tensors_match",
            "tensor_metadata_match",
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


def _measure(command: list[str]) -> dict[str, object]:
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    started = time.perf_counter()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment) as child:
        try:
            monitored = psutil.Process(child.pid)
            peak = _tree_rss(monitored)
            samples = 1
            while child.poll() is None:
                peak = max(peak, _tree_rss(monitored))
                samples += 1
                time.sleep(RSS_SAMPLE_SECONDS)
            stdout, _ = child.communicate()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    elapsed = time.perf_counter() - started
    try:
        conversion = json.loads(stdout)
    except (ValueError, TypeError):
        raise BenchmarkError("invalid_backend_metrics") from None
    if child.returncode != 0:
        raise BenchmarkError("backend_conversion_failed")
    if not isinstance(conversion, dict) or conversion.get("backend") not in ("streaming", "lalamo"):
        raise BenchmarkError("invalid_backend_metrics")
    return {
        "backend": conversion["backend"],
        "end_to_end_seconds": elapsed,
        "cli_conversion_seconds": conversion.get("elapsed_seconds"),
        "peak_process_tree_rss_bytes": peak,
        "rss_samples": samples,
        "reported_max_rss_bytes": conversion.get("max_rss_bytes"),
        "output_bytes": conversion.get("output_bytes"),
        "tensor_count": conversion.get("tensor_count", conversion.get("tensors")),
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


def _sync_environment(project: Path) -> None:
    result = subprocess.run(
        ["uv", "sync", "--project", str(project), "--locked"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise BenchmarkError("dependency_setup_failed")


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    from .cli import toolchain_provenance

    project = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    if not source.is_dir():
        raise BenchmarkError("missing_source_directory")
    files = _source_files(source)
    if not files:
        raise BenchmarkError("empty_source_directory")
    if args.output.exists():
        raise BenchmarkError("output_exists")
    args.output.mkdir(parents=True, mode=0o700)
    args.output.chmod(0o700)
    # Installation and checkpoint hashing are completed before any timed run.
    _sync_environment(project)
    _sync_environment(project / "lalamo")
    tooling = toolchain_provenance()
    source_hash, source_bytes = _source_digest(source, files)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "repo": args.repo,
        "dtype": args.dtype,
        "chunk_mib": args.chunk_mib,
        "lalamo_revision": LALAMO_REVISION,
        **tooling,
        "source_manifest_sha256": source_hash,
        "source_bytes": source_bytes,
        "repeats_per_backend": args.repeats,
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
            "fresh_process_per_run": True,
            "source_cache_warmed_before_each_run": True,
            "setup_downloads_excluded": True,
            "backend_order": "streaming,lalamo on even repeats; lalamo,streaming on odd repeats",
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
    for repeat in range(args.repeats):
        order = ("streaming", "lalamo") if repeat % 2 == 0 else ("lalamo", "streaming")
        outputs = {}
        for backend in order:
            _warm_source(files)
            destination = args.output / f"repeat-{repeat + 1}" / backend
            outputs[backend] = destination
            command = [
                "uv",
                "run",
                "--project",
                str(project),
                "--locked",
                "--no-sync",
                "stark-convert",
                str(source),
                "--repo",
                args.repo,
                "--backend",
                backend,
                "--output",
                str(destination),
                "--dtype",
                args.dtype,
                "--chunk-mib",
                str(args.chunk_mib),
            ]
            metrics = _measure(command)
            if metrics["backend"] != backend:
                raise BenchmarkError("unexpected_backend_selection")
            metrics.update(repeat=repeat + 1, order=order.index(backend) + 1)
            runs.append(metrics)
        parity = compare_outputs(outputs["streaming"], outputs["lalamo"])
        parity["repeat"] = repeat + 1
        comparisons.append(parity)
        if not parity["passed"]:
            report["status"] = "parity_failed"
            break
    report["summary"] = {backend: _summary(runs, backend) for backend in ("streaming", "lalamo")}
    fast = report["summary"]["streaming"]
    reference = report["summary"]["lalamo"]
    report["ratios"] = {
        "end_to_end_speedup": reference["median_end_to_end_seconds"] / fast["median_end_to_end_seconds"],
        "sampled_rss_reduction": (
            reference["median_peak_process_tree_rss_bytes"] / max(fast["median_peak_process_tree_rss_bytes"], 1)
        ),
    }
    if report["status"] == "running":
        report["status"] = "passed"
    # Checkpoint changes during the benchmark invalidate source-bound evidence.
    if _source_digest(source, files)[0] != source_hash or _source_files(source) != files:
        report["status"] = "source_changed"
    if toolchain_provenance() != tooling:
        report["status"] = "tooling_changed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk_mib < 1:
        print('{"status":"failed","error_category":"invalid_benchmark_options"}')
        return 1
    try:
        report = benchmark(args)
    except BaseException as error:
        category = str(error) if isinstance(error, BenchmarkError) else "benchmark_failed"
        print(json.dumps({"status": "failed", "error_category": category}, separators=(",", ":")))
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    report_path = args.output / "report.json"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
