"""Independent checks for conversion parity and benchmark measurements."""

import json
import struct
import sys
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
    fast, reference = tmp_path / "streaming", tmp_path / "lalamo"
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
    assert result["streaming_tensor_manifest_sha256"] == result["lalamo_tensor_manifest_sha256"]
    assert result["streaming_weights_sha256"] != result["lalamo_weights_sha256"]


@pytest.mark.parametrize("difference", ["tensor_byte", "dtype", "shape", "config", "tokenizer", "metadata"])
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
    write_weights(reference / "model.safetensors", tensors, metadata=metadata)
    result = compare_outputs(fast, reference)
    assert not result["passed"]
    key = {"config": "config_match", "tokenizer": "tokenizer_match", "metadata": "tensor_metadata_match"}.get(
        difference, "tensors_match"
    )
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
        "print(json.dumps({'backend':'streaming','elapsed_seconds':0.01,'max_rss_bytes':1}))"
    )
    measured = _measure([sys.executable, "-c", child])
    assert measured["backend"] == "streaming"
    assert measured["end_to_end_seconds"] >= 0.08
    assert measured["cli_conversion_seconds"] == 0.01
    assert measured["peak_process_tree_rss_bytes"] > 0
    assert measured["rss_samples"] >= 2


def test_measure_rejects_unsuccessful_backend_without_exposing_output():
    child = "import json,sys; print(json.dumps({'status':'failed','detail':'must stay private'})); sys.exit(1)"
    with pytest.raises(BenchmarkError, match="^backend_conversion_failed$"):
        _measure([sys.executable, "-c", child])


@pytest.mark.parametrize(
    "changed", [None, "converter_sources_sha256", "fast_lock_sha256", "lalamo_lock_sha256", "source"]
)
def test_benchmark_binds_results_to_unchanged_source_and_tooling(tmp_path, monkeypatch, changed):
    source = tmp_path / "source"
    source.mkdir()
    source_config = source / "config.json"
    source_config.write_text("{}")
    args = SimpleNamespace(
        source=source,
        repo="primary/repo",
        output=tmp_path / "private",
        repeats=1,
        dtype="bfloat16",
        chunk_mib=8,
    )
    provenance = {name: "a" * 64 for name in ("converter_sources_sha256", "fast_lock_sha256", "lalamo_lock_sha256")}
    snapshots = []

    def toolchain():
        result = provenance.copy()
        if snapshots and changed in result:
            result[changed] = "b" * 64
        snapshots.append(result)
        return result

    def measure(command):
        backend = command[command.index("--backend") + 1]
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        write_weights(output / "model.safetensors", {"a": ("BF16", [1], b"\x00\x00")})
        (output / "config.json").write_text("{}")
        (output / "tokenizer.json").write_text("{}")
        if changed == "source":
            source_config.write_text('{"changed":true}')
        return {
            "backend": backend,
            "end_to_end_seconds": 1.0,
            "cli_conversion_seconds": 0.5,
            "peak_process_tree_rss_bytes": 100,
            "reported_max_rss_bytes": 100,
            "output_bytes": 2,
            "tensor_count": 1,
            "rss_samples": 2,
        }

    monkeypatch.setattr(benchmark_module, "_sync_environment", lambda project: None)
    monkeypatch.setattr(benchmark_module, "_measure", measure)
    monkeypatch.setattr(cli, "toolchain_provenance", toolchain)
    report = benchmark_module.benchmark(args)
    assert report["converter_sources_sha256"] == "a" * 64
    expected_status = "source_changed" if changed == "source" else "tooling_changed" if changed else "passed"
    assert report["status"] == expected_status
