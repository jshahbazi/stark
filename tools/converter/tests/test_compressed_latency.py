"""Sequential coefficient export under bounded and full-plane workspaces."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from test_streaming import save_tensors

from stark_convert import compressed, streaming
from stark_convert.streaming import ConversionError, ConversionPlan, TensorOperation, read_safetensors_header


class SequentialOutput(io.BytesIO):
    def seek(self, *args):
        raise AssertionError("Coefficient export must write sequentially")


def fused_operation(tmp_path):
    infos, arrays = [], []
    for index, rows in enumerate((3, 7, 3)):
        values = (np.arange(rows * 5, dtype="<f4") + index * 128).reshape(rows, 5)
        path = tmp_path / f"plane-{index}.safetensors"
        save_tensors(path, {"params": ("F32", values)})
        infos.append(read_safetensors_header(path)[1]["params"])
        arrays.append(values)
    operation = TensorOperation("matrix.scales", tuple(infos), "F32", (5, 16), "mlx_params")
    expected = np.zeros(operation.shape, dtype="<f4")
    expected[:, :13] = np.concatenate(arrays).T
    return operation, expected


@pytest.mark.parametrize("workspace", [16, 96, 256, 4096])
def test_sequential_fused_export_is_identical_across_workspace_strategies(tmp_path, workspace):
    operation, expected = fused_operation(tmp_path)
    output = SequentialOutput()
    output.write(b"existing prefix")
    peak = compressed.write_mlx_params(operation, output, workspace)
    assert output.getvalue() == b"existing prefix" + expected.tobytes()
    assert peak <= workspace


@pytest.mark.parametrize("workspace", [16, 256, 4096])
def test_sequential_params_preserve_inline_whole_file_hash(tmp_path, monkeypatch, workspace):
    operation, _ = fused_operation(tmp_path)
    plan = ConversionPlan({}, (operation,), {}, sum(source.nbytes for source in operation.sources), 3)

    def unexpected_reread(*args):
        raise AssertionError("Sequential coefficient export must retain inline hashing")

    monkeypatch.setattr(streaming.hashlib, "file_digest", unexpected_reread)
    path = tmp_path / "output.safetensors"
    stats = streaming.write_plan(plan, path, chunk_bytes=workspace)
    assert stats["output_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert stats["peak_buffer_bytes"] <= workspace


def test_full_group_band_reads_each_source_plane_contiguously(tmp_path, monkeypatch):
    operation, expected = fused_operation(tmp_path)
    original_open = compressed._open_tensor
    reads = []

    class RecordingReader:
        def __init__(self, source):
            self.stream = original_open(source)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def seek(self, offset):
            return self.stream.seek(offset)

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(compressed, "_open_tensor", RecordingReader)
    output = SequentialOutput()
    peak = compressed.write_mlx_params(operation, output, 4096)
    assert output.getvalue() == expected.tobytes()
    assert reads == [source.nbytes for source in operation.sources]
    assert peak <= 4096


@pytest.mark.parametrize("source_dtype", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("target", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("workspace", [256, 65536])
def test_group_bands_preserve_frozen_exceptional_float_casts(tmp_path, source_dtype, target, workspace):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "dense_casts.json").read_text())["cases"][source_dtype]
    values = np.array(fixture["source_bits"], dtype="<u4" if source_dtype == "F32" else "<u2").reshape(-1, 2)
    path = tmp_path / "source.safetensors"
    save_tensors(path, {"params": (source_dtype, values)})
    info = read_safetensors_header(path)[1]["params"]
    rows = info.shape[0]
    operation = TensorOperation("matrix.scales", (info,), target, (2, rows + (-rows % 4)), "mlx_params")
    output = SequentialOutput()
    peak = compressed.write_mlx_params(operation, output, workspace)
    stored = np.frombuffer(output.getvalue(), dtype="<u4" if target == "F32" else "<u2").reshape(operation.shape)
    assert np.all(stored[:, rows:] == 0)
    restored = np.ascontiguousarray(stored[:, :rows].T)
    assert hashlib.sha256(restored.tobytes()).hexdigest() == fixture["output_sha256"][target]
    assert peak <= workspace


@pytest.mark.parametrize("workspace", [16, 256, 4096])
def test_each_sequential_strategy_rejects_truncated_planes(tmp_path, monkeypatch, workspace):
    operation, _ = fused_operation(tmp_path)

    def truncated_open(source):
        return io.BytesIO(source.path.read_bytes()[:-1])

    monkeypatch.setattr(compressed, "_open_tensor", truncated_open)
    with pytest.raises(ConversionError, match="truncated"):
        compressed.write_mlx_params(operation, SequentialOutput(), workspace)


def test_full_plane_strategy_rechecks_source_signature(tmp_path):
    operation, _ = fused_operation(tmp_path)
    source = operation.sources[0]
    source.path.write_bytes(source.path.read_bytes() + b"mutation")
    with pytest.raises(ConversionError, match="Checkpoint changed"):
        compressed.write_mlx_params(operation, SequentialOutput(), 4096)
