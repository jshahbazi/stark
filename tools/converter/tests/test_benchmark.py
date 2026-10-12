"""Independent checks for conversion parity and benchmark measurements."""

import json
import os
import signal
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark_convert import benchmark as benchmark_module
from stark_convert import cli
from stark_convert.benchmark import BenchmarkError, _measure, _tensor_manifest, compare_outputs


def write_weights(path, tensors, order=None, metadata=None):
    order = list(tensors) if order is None else order
    header = {} if metadata is None else {"__metadata__": metadata}
    payload = b""
    for name in order:
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


@pytest.fixture
def outputs(tmp_path):
    fast, reference = tmp_path / "streaming", tmp_path / "reference"
    for output in (fast, reference):
        output.mkdir()
        write_weights(
            output / "model.safetensors",
            {"a": ("BF16", [2], b"\x00\x00\x80\x3f"), "b": ("F32", [1], b"\x00\x00\x80\x3f")},
        )
        (output / "config.json").write_text('{"generation_config":{"stop_token_ids":[2]}}')
        (output / "tokenizer.json").write_text('{"model":{"type":"BPE","vocab":{"a":0}}}')
    return fast, reference


def test_parity_compares_tensor_content_independently_of_file_layout(outputs):
    fast, reference = outputs
    write_weights(
        reference / "model.safetensors",
        {"a": ("BF16", [2], b"\x00\x00\x80\x3f"), "b": ("F32", [1], b"\x00\x00\x80\x3f")},
        order=["b", "a"],
    )
    result = compare_outputs(fast, reference)
    assert result["passed"]
    assert result["streaming_tensor_manifest_sha256"] == result["reference_tensor_manifest_sha256"]
    assert result["streaming_weights_sha256"] != result["reference_weights_sha256"]


@pytest.mark.parametrize("difference", ["tensor_byte", "dtype", "shape", "config", "tokenizer", "metadata", "spec"])
def test_parity_detects_material_conversion_differences(outputs, difference):
    fast, reference = outputs
    tensors = {"a": ("BF16", [2], b"\x00\x00\x80\x3f"), "b": ("F32", [1], b"\x00\x00\x80\x3f")}
    if difference == "tensor_byte":
        tensors["a"] = ("BF16", [2], b"\x00\x00\x00\x3f")
    elif difference == "dtype":
        tensors["a"] = ("F16", [2], tensors["a"][2])
    elif difference == "shape":
        tensors["a"] = ("BF16", [1, 2], tensors["a"][2])
    elif difference in ("config", "tokenizer"):
        (reference / f"{difference}.json").write_text('{"changed":true}')
    metadata = {"format": "changed"} if difference == "metadata" else None
    if difference == "spec":
        metadata = {"a.spec": '{"type":"FullPrecisionSpec","layout":"column_major"}'}
    write_weights(reference / "model.safetensors", tensors, metadata=metadata)
    result = compare_outputs(fast, reference)
    assert not result["passed"]
    key = {
        "config": "config_match",
        "tokenizer": "tokenizer_match",
        "metadata": "tensor_metadata_match",
        "spec": "tensor_specs_match",
    }.get(difference, "tensors_match")
    assert not result[key]


@pytest.mark.parametrize(
    ("shape", "offsets", "dtype", "payload"),
    [
        ([2], [0, 4], "BF16", b"\x00" * 3),
        ([2], [1, 5], "BF16", b"\x00" * 5),
        ([True], [0, 2], "BF16", b"\x00" * 2),
        ([1], [False, 2], "BF16", b"\x00" * 2),
        ([1], [0, 2], "UNKNOWN", b"\x00" * 2),
        ([2], [0, 2], "BF16", b"\x00" * 2),
    ],
)
def test_independent_reader_rejects_malformed_tensor_headers(tmp_path, shape, offsets, dtype, payload):
    header = json.dumps({"a": {"dtype": dtype, "shape": shape, "data_offsets": offsets}}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    with pytest.raises(BenchmarkError):
        _tensor_manifest(path)


def test_independent_reader_rejects_overlapping_tensors(tmp_path):
    header = json.dumps(
        {name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for name in ("a", "b")}
    ).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    with pytest.raises(BenchmarkError, match="invalid_output_offsets"):
        _tensor_manifest(path)


def test_measure_includes_fresh_process_wall_time_and_tree_rss():
    child = (
        "import json,time; time.sleep(0.08); "
        "print(json.dumps({'status':'ok','backend':'streaming','elapsed_seconds':0.01,"
        "'total_rss_bytes':1,'output_bytes':2,'tensor_count':1}))"
    )
    measured = _measure([sys.executable, "-c", child])
    assert measured["backend"] == "streaming"
    assert measured["end_to_end_seconds"] >= 0.08
    assert measured["cli_conversion_seconds"] == 0.01
    assert measured["reported_max_rss_bytes"] == 1
    assert measured["peak_process_tree_rss_bytes"] > 0
    assert measured["rss_samples"] >= 2


def test_measure_rejects_unsuccessful_backend_without_exposing_output():
    child = "import json,sys; print(json.dumps({'status':'failed','detail':'must stay private'})); sys.exit(1)"
    with pytest.raises(BenchmarkError, match="^backend_conversion_failed$"):
        _measure([sys.executable, "-c", child])


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("elapsed_seconds", None),
        ("elapsed_seconds", True),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", 10**400),
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
        ("tensor_count", None),
        ("tensor_count", 0),
        ("tensor_count", True),
        ("output_bytes", "2"),
        ("output_bytes", -1),
        ("total_rss_bytes", 1.5),
    ],
)
def test_measure_rejects_invalid_or_missing_metrics(field, value):
    receipt = {
        "status": "ok",
        "backend": "streaming",
        "elapsed_seconds": 0.1,
        "total_rss_bytes": 1,
        "output_bytes": 2,
        "tensor_count": 1,
    }
    if value is None:
        receipt.pop(field)
    else:
        receipt[field] = value
    child = "print(" + repr(json.dumps(receipt)) + ")"
    with pytest.raises(BenchmarkError, match="^invalid_backend_metrics$"):
        _measure([sys.executable, "-c", child])


@pytest.mark.parametrize(
    "failure",
    [
        BenchmarkError("backend_conversion_failed"),
        KeyboardInterrupt("private details"),
        RuntimeError("private details"),
    ],
)
def test_failed_repeat_preserves_completed_observations(tmp_path, monkeypatch, failure):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    args = SimpleNamespace(
        source=source,
        repo=None,
        output=tmp_path / "output",
        repeats=3,
        dtype="bfloat16",
        chunk_mib=8,
        metadata_profile="hf-generic",
        architecture="qwen3",
    )
    calls = []

    def measure(command):
        calls.append(command)
        assert command[:3] == [sys.executable, "-m", "stark_convert.cli"]
        assert "--repo" not in command
        assert command[command.index("--metadata-profile") + 1] == "hf-generic"
        if len(calls) == 2:
            raise failure
        return {
            "backend": "streaming",
            "end_to_end_seconds": 1.0,
            "cli_conversion_seconds": 0.5,
            "peak_process_tree_rss_bytes": 100,
            "rss_samples": 2,
        }

    monkeypatch.setattr(benchmark_module, "_measure", measure)
    report = benchmark_module.benchmark(args)
    assert report == json.loads((args.output / "report.json").read_text())
    assert report["status"] == "failed" and report["failed_repeat"] == 2
    assert len(report["runs"]) == 1 and report["summary"]["streaming"]["runs"] == 1
    assert "private details" not in json.dumps(report)
    if isinstance(failure, KeyboardInterrupt):
        assert report["error_category"] == "interrupted"
    assert (args.output / "report.json").stat().st_mode & 0o777 == 0o600
    assert args.output.stat().st_mode & 0o777 == 0o700


def test_report_replacement_failure_retains_previous_complete_json(tmp_path, monkeypatch):
    original = {"status": "running", "runs": []}
    benchmark_module._write_report(tmp_path, original)

    def fail(*args):
        raise OSError("private details")

    monkeypatch.setattr(benchmark_module.os, "replace", fail)
    with pytest.raises(BenchmarkError, match="^report_write_failed$"):
        benchmark_module._write_report(tmp_path, {"status": "passed"})
    assert json.loads((tmp_path / "report.json").read_text()) == original
    assert not list(tmp_path.glob(".report-*"))


def test_report_write_error_is_sanitized_by_main(tmp_path, monkeypatch, capsys):
    def fail(args):
        raise BenchmarkError("report_write_failed")

    monkeypatch.setattr(benchmark_module, "benchmark", fail)
    monkeypatch.setattr(
        sys, "argv", ["stark-convert-benchmark", "--source", str(tmp_path), "--output", str(tmp_path / "output")]
    )
    assert benchmark_module.main() == 1
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "error_category": "report_write_failed"}


def test_final_write_failure_leaves_running_report_and_unsuccessful_command(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    output = tmp_path / "output"
    write_report = benchmark_module._write_report

    def write(directory, report):
        if report["status"] == "failed":
            raise BenchmarkError("report_write_failed")
        write_report(directory, report)

    def fail(command):
        raise BenchmarkError("backend_conversion_failed")

    monkeypatch.setattr(benchmark_module, "_write_report", write)
    monkeypatch.setattr(benchmark_module, "_measure", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stark-convert-benchmark",
            "--source",
            str(source),
            "--output",
            str(output),
            "--metadata-profile",
            "hf-generic",
        ],
    )
    assert benchmark_module.main() == 1
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "error_category": "report_write_failed"}
    assert json.loads((output / "report.json").read_text())["status"] == "running"


def test_benchmark_interrupt_returns_130_and_saves_failure(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    output = tmp_path / "output"

    def interrupt(command):
        raise KeyboardInterrupt("private details")

    monkeypatch.setattr(benchmark_module, "_measure", interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stark-convert-benchmark",
            "--source",
            str(source),
            "--output",
            str(output),
            "--metadata-profile",
            "hf-generic",
        ],
    )
    assert benchmark_module.main() == 130
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed" and report["error_category"] == "interrupted"
    assert report == json.loads((output / "report.json").read_text())
    assert "private details" not in json.dumps(report)


@pytest.fixture
def bounded_measurement(monkeypatch):
    """Kill only test-owned processes if a receipt-draining regression hangs."""
    created = []
    expired = threading.Event()
    original_popen = benchmark_module.subprocess.Popen

    def tracked_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if "start_new_session" in kwargs:
            created.append((child, kwargs["start_new_session"]))
        return child

    def stop_created():
        expired.set()
        for child, isolated_group in created:
            if os.name == "posix" and isolated_group:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.kill()

    monkeypatch.setattr(benchmark_module.subprocess, "Popen", tracked_popen)
    deadline = threading.Timer(5, stop_created)
    deadline.start()
    try:
        yield created, expired
    finally:
        deadline.cancel()
        deadline.join()
        for child, isolated_group in created:
            if os.name == "posix" and isolated_group:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.kill()
            child.wait()


def receipt_reader_threads():
    return {thread.ident for thread in threading.enumerate() if thread.name == "stark-benchmark-receipt"}


def test_measure_drains_receipts_larger_than_pipe_capacity_with_bounded_deadline(bounded_measurement):
    _, expired = bounded_measurement
    readers_before = receipt_reader_threads()
    # This valid aggregate receipt exceeds macOS/Linux pipe capacity by orders
    # of magnitude. Waiting for child exit before reading stdout deadlocks.
    child = (
        "import json; "
        "print(json.dumps({'status':'ok','backend':'streaming','elapsed_seconds':0.125,'total_rss_bytes':27,"
        "'output_bytes':4096,'tensor_count':2048,"
        "'source_files':[{'sha256':'a'*64,'size':1}]*20000}))"
    )
    measured = _measure([sys.executable, "-c", child])
    assert not expired.is_set()
    assert measured["backend"] == "streaming"
    assert measured["cli_conversion_seconds"] == 0.125
    assert measured["reported_max_rss_bytes"] == 27
    assert measured["output_bytes"] == 4096
    assert measured["tensor_count"] == 2048
    assert measured["rss_samples"] >= 1
    assert measured["peak_process_tree_rss_bytes"] > 0
    assert receipt_reader_threads() == readers_before


def test_measure_interrupt_reaps_child_and_joins_receipt_reader(bounded_measurement, monkeypatch):
    created, expired = bounded_measurement
    readers_before = receipt_reader_threads()

    original_sleep = benchmark_module.time.sleep
    interrupted = False

    def interrupted_sleep(seconds):
        nonlocal interrupted
        if not interrupted and seconds == benchmark_module.RSS_SAMPLE_SECONDS:
            interrupted = True
            raise KeyboardInterrupt()
        return original_sleep(seconds)

    monkeypatch.setattr(benchmark_module.time, "sleep", interrupted_sleep)
    with pytest.raises(KeyboardInterrupt):
        _measure([sys.executable, "-c", "import time; time.sleep(30)"])
    assert not expired.is_set()
    assert all(child.poll() is not None for child, _ in created)
    assert receipt_reader_threads() == readers_before


@pytest.mark.skipif(os.name != "posix", reason="isolated process-group cleanup is a POSIX contract")
def test_interrupt_cleans_descendant_holding_receipt_pipe_open(tmp_path, bounded_measurement, monkeypatch):
    created, expired = bounded_measurement
    readers_before = receipt_reader_threads()
    marker = tmp_path / "owned-descendant.pid"
    descendant_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    launcher_code = (
        f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{descendant_code!r}]); time.sleep(30)"
    )
    original_sleep = benchmark_module.time.sleep
    interrupted = False

    def interrupt_after_descendant_starts(seconds):
        nonlocal interrupted
        if not interrupted and marker.is_file():
            interrupted = True
            raise KeyboardInterrupt()
        return original_sleep(seconds)

    monkeypatch.setattr(benchmark_module.time, "sleep", interrupt_after_descendant_starts)
    with pytest.raises(KeyboardInterrupt):
        _measure([sys.executable, "-c", launcher_code])
    assert not expired.is_set()
    assert all(child.poll() is not None for child, _ in created)
    assert receipt_reader_threads() == readers_before
    try:
        descendant = benchmark_module.psutil.Process(int(marker.read_text()))
        assert descendant.status() in {benchmark_module.psutil.STATUS_ZOMBIE, benchmark_module.psutil.STATUS_DEAD}
    except benchmark_module.psutil.NoSuchProcess:
        pass


def test_receipt_reader_failure_is_private_and_reaps_child(bounded_measurement, monkeypatch, capsys):
    created, expired = bounded_measurement
    readers_before = receipt_reader_threads()

    def broken_communication(*args, **kwargs):
        raise RuntimeError("synthetic private receipt exception")

    tracked_popen = benchmark_module.subprocess.Popen

    def broken_reader_popen(*args, **kwargs):
        child = tracked_popen(*args, **kwargs)
        if "start_new_session" in kwargs:
            monkeypatch.setattr(child, "communicate", broken_communication)
        return child

    monkeypatch.setattr(benchmark_module.subprocess, "Popen", broken_reader_popen)
    with pytest.raises(BenchmarkError, match="^backend_receipt_failed$"):
        _measure([sys.executable, "-c", "import time; time.sleep(30)"])
    assert not expired.is_set()
    assert all(child.poll() is not None for child, _ in created)
    assert receipt_reader_threads() == readers_before
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize(
    "changed", [None, "converter_sources_sha256", "fast_lock_sha256", "source", "reference", "source_manifest"]
)
def test_benchmark_binds_results_to_unchanged_source_and_tooling(tmp_path, monkeypatch, changed):
    source = tmp_path / "source"
    source.mkdir()
    source_config = source / "config.json"
    source_config.write_text("{}")
    reference = tmp_path / "reference"
    reference.mkdir()
    write_weights(reference / "model.safetensors", {"a": ("BF16", [1], b"\x00\x00")})
    (reference / "config.json").write_text("{}")
    (reference / "tokenizer.json").write_text("{}")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}")
    args = SimpleNamespace(
        source=source,
        repo="primary/repo",
        output=tmp_path / "private",
        repeats=1,
        dtype="bfloat16",
        chunk_mib=8,
        reference_output=reference,
        source_manifest=source_manifest,
    )
    provenance = {name: "a" * 64 for name in ("converter_sources_sha256", "fast_lock_sha256")}
    snapshots = []

    def toolchain():
        result = provenance.copy()
        if snapshots and changed in result:
            result[changed] = "b" * 64
        snapshots.append(result)
        return result

    def measure(command):
        assert "--backend" not in command
        assert "--offline" in command
        assert command[command.index("--source-manifest") + 1] == str(source_manifest)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        write_weights(output / "model.safetensors", {"a": ("BF16", [1], b"\x00\x00")})
        (output / "config.json").write_text("{}")
        (output / "tokenizer.json").write_text("{}")
        if changed == "source":
            source_config.write_text('{"changed":true}')
        elif changed == "reference":
            (reference / "extra-file").write_text("changed")
        elif changed == "source_manifest":
            source_manifest.write_text('{"changed":true}')
        return {
            "backend": "streaming",
            "end_to_end_seconds": 1.0,
            "cli_conversion_seconds": 0.5,
            "peak_process_tree_rss_bytes": 100,
            "reported_max_rss_bytes": 100,
            "output_bytes": 2,
            "tensor_count": 1,
            "rss_samples": 2,
        }

    monkeypatch.setattr(benchmark_module, "_measure", measure)
    monkeypatch.setattr(cli, "toolchain_provenance", toolchain)
    report = benchmark_module.benchmark(args)
    assert report["converter_sources_sha256"] == "a" * 64
    expected_status = (
        "source_changed"
        if changed in ("source", "source_manifest")
        else "reference_changed"
        if changed == "reference"
        else "tooling_changed"
        if changed
        else "passed"
    )
    assert report["status"] == expected_status
    assert report["measurement"]["reference_timing_measured"] is False
    assert "ratios" not in report
    assert report["reference_comparison_requested"]
    assert len(report["parity"]) == 1


def test_benchmark_without_reference_measures_streaming_repeats_only(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    args = SimpleNamespace(
        source=source,
        repo="primary/repo",
        output=tmp_path / "private",
        repeats=3,
        dtype="float16",
        chunk_mib=2,
        reference_output=None,
        source_manifest=None,
    )
    commands = []

    def measure(command):
        commands.append(command)
        return {
            "backend": "streaming",
            "end_to_end_seconds": 1.0,
            "cli_conversion_seconds": 0.5,
            "peak_process_tree_rss_bytes": 100,
            "rss_samples": 2,
        }

    monkeypatch.setattr(benchmark_module, "_measure", measure)
    monkeypatch.setattr(cli, "toolchain_provenance", lambda: {"converter_sources_sha256": "a" * 64})
    report = benchmark_module.benchmark(args)
    assert report["status"] == "passed"
    assert report["summary"]["streaming"]["runs"] == 3
    assert len({command[command.index("--output") + 1] for command in commands}) == 3
    assert all("--offline" in command and "--source-manifest" not in command for command in commands)
    assert report["parity"] == []
    assert not report["reference_comparison_requested"]
    assert report["reference_output_manifest_sha256"] is None
    assert "ratios" not in report


def test_measure_rejects_an_unrequested_backend():
    child = "import json; print(json.dumps({'backend':'external','elapsed_seconds':0.01}))"
    with pytest.raises(BenchmarkError, match="^invalid_backend_metrics$"):
        _measure([sys.executable, "-c", child])


@pytest.mark.parametrize("contents", ['{"a":1,"a":2}', "not-json", "[]", '{"a":NaN}'])
def test_parity_rejects_malformed_runtime_json(outputs, contents):
    fast, reference = outputs
    (reference / "config.json").write_text(contents)
    with pytest.raises(BenchmarkError, match="^invalid_output_json$"):
        compare_outputs(fast, reference)


@pytest.mark.parametrize("spec", [None, {}, {"dtype": ["BF16"], "shape": [1], "data_offsets": [0, 2]}])
def test_independent_reader_rejects_incomplete_tensor_records(tmp_path, spec):
    header = json.dumps({"a": spec}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    with pytest.raises(BenchmarkError, match="^invalid_output_header$"):
        _tensor_manifest(path)


@pytest.mark.parametrize("extra_field", [True, False])
def test_independent_reader_rejects_empty_names_and_extra_spec_fields(tmp_path, extra_field):
    spec = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
    if extra_field:
        spec["unexpected"] = "private-fixture-marker"
    header = json.dumps({"a" if extra_field else "": spec}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    with pytest.raises(BenchmarkError, match="^invalid_output_header$"):
        _tensor_manifest(path)


def test_benchmark_argument_errors_do_not_echo_supplied_values(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stark-convert-benchmark", "--dtype", "private-fixture-marker"])
    with pytest.raises(SystemExit) as error:
        benchmark_module.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "private-fixture-marker" not in captured.out + captured.err
    assert json.loads(captured.err) == {"status": "error", "error_category": "invalid_arguments"}


@pytest.mark.parametrize("spec", ["[]", "not-json"])
def test_parity_rejects_malformed_tensor_specifications(outputs, spec):
    fast, reference = outputs
    write_weights(reference / "model.safetensors", {"a": ("BF16", [1], b"\x00\x00")}, metadata={"a.spec": spec})
    with pytest.raises(BenchmarkError, match="^invalid_output_tensor_spec$"):
        compare_outputs(fast, reference)
