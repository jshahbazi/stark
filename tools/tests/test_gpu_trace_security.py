"""Synthetic imported-capture and trace JSON security regressions."""

import struct
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from gpu_trace import gputrace
from gpu_trace.formatting import _bar
from gpu_trace.models import GpuUtilization, TraceExport, TraceMetadata
from gpu_trace.serialization import from_dict, to_dict


def trace_data():
    metadata = TraceMetadata("synthetic", "synthetic", "synthetic", "synthetic", 1, 100, "", "", "")
    utilization = GpuUtilization(100, 60, 40, 100, 40, 20, 80, 1, 1)
    return to_dict(TraceExport(metadata, utilization))


def test_ordinary_trace_json_and_bars_remain_supported():
    data = trace_data()
    assert asdict(from_dict(data).utilization) == data["utilization"]
    assert _bar(50, 10) == "█" * 5 + "░" * 5


@pytest.mark.parametrize("value", [1e300, -1e300, 2**4096, float("inf"), float("-inf"), float("nan")])
def test_bar_allocation_is_always_bounded(value):
    assert len(_bar(value, 20)) == 20


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_active_ns", 2**62),
        ("target_active_ns", 101),
        ("run_duration_ns", 0),
        ("window_end_ns", 101),
        ("global_idle_ns", -1),
        ("global_active_ns", float("nan")),
        ("target_active_ns", True),
        ("window_start_ns", 90),
    ],
)
def test_inconsistent_imported_trace_is_rejected(field, value):
    data = trace_data()
    data["utilization"][field] = value
    with pytest.raises(ValueError):
        from_dict(data)


def test_capture_labels_and_reads_require_real_resource_basenames(tmp_path):
    capture = tmp_path / "capture.gputrace"
    capture.mkdir()
    resource = capture / "MTLBuffer-1-2"
    resource.write_bytes(struct.pack("<f", 3.5))
    (capture / "device-resources-1").write_bytes(b"MTLBuffer-1-2\0synthetic buffer\0")
    labels = gputrace.extract_buffer_labels(capture)
    assert labels == {"MTLBuffer-1-2": "synthetic buffer"}
    assert gputrace.resolve_buffer_filename(capture, "synthetic buffer", labels) == resource.name
    assert gputrace.read_buffer(capture, resource.name, "float")[0].fields == (("field0", 3.5),)
    assert gputrace.buffer_element_count(capture, resource.name, "float") == 1
    assert gputrace.buffer_summary(capture, resource.name, "synthetic").float_count == 1


def test_capture_metadata_uses_verified_bytes_for_plutil(tmp_path, monkeypatch):
    (tmp_path / "metadata").write_bytes(b"synthetic metadata")

    def convert(command, *, input, capture_output):
        assert command[-1] == "-" and input == b"synthetic metadata" and capture_output
        return SimpleNamespace(returncode=0, stdout=b'{"DYCaptureSession.graphics_api": 1}')

    monkeypatch.setattr(gputrace.subprocess, "run", convert)
    assert gputrace.parse_metadata(tmp_path).graphics_api == "Metal"


def test_capture_derived_traversal_and_symlinks_cannot_select_outside_data(tmp_path):
    capture = tmp_path / "capture.gputrace"
    capture.mkdir()
    (capture / "MTLBuffer-1-2").mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(struct.pack("<f", 9.5))
    attack = "MTLBuffer-1-2/../../outside"
    (capture / "MTLBuffer-2-3").symlink_to(outside)
    (capture / "metadata").symlink_to(outside)
    (capture / "device-resources-1").write_bytes(
        attack.encode() + b"\0synthetic outside label\0MTLBuffer-2-3\0synthetic link label\0"
    )
    assert gputrace.extract_buffer_labels(capture) == {}
    with pytest.raises(OSError):
        gputrace.parse_metadata(capture)
    for name in (attack, "../outside", "MTLBuffer-2-3", str(outside)):
        assert gputrace.resolve_buffer_filename(capture, name, {name: "synthetic"}) is None
        with pytest.raises((ValueError, OSError)):
            gputrace.read_buffer(capture, name, "float")
        with pytest.raises((ValueError, OSError)):
            gputrace.buffer_element_count(capture, name, "float")
        with pytest.raises((ValueError, OSError)):
            gputrace.buffer_summary(capture, name, "synthetic")


def test_capture_matching_uses_ordered_linear_merge_and_metadata_budgets(tmp_path, monkeypatch):
    count = 2000
    names = {f"MTLBuffer-1-{index}" for index in range(count)}
    monkeypatch.setattr(gputrace, "_resource_names", lambda _: names)
    data = b"\0".join(f"MTLBuffer-1-{i}\0synthetic label {i}".encode() for i in range(count))
    (tmp_path / "device-resources-1").write_bytes(data)
    labels = gputrace.extract_buffer_labels(tmp_path)
    assert len(labels) == count
    assert labels["MTLBuffer-1-1999"] == "synthetic label 1999"
    monkeypatch.setattr(gputrace, "_MAX_METADATA_BYTES", 10)
    with pytest.raises(ValueError, match="byte budget"):
        gputrace.extract_buffer_labels(tmp_path)
    monkeypatch.setattr(gputrace, "_MAX_METADATA_BYTES", len(data))
    monkeypatch.setattr(gputrace, "_MAX_STRINGS", 3)
    with pytest.raises(ValueError, match="string budget"):
        gputrace.extract_buffer_labels(tmp_path)
