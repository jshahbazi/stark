"""Synthetic container mutations; no model payload or network is required."""

import hashlib
import struct
from dataclasses import replace

import pytest

from stark_convert import edge0_source as edge
from stark_convert.streaming import ConversionError, UnsupportedConversion


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def gguf(tmp_path, *, offset=0, kind=30, shape=(4,), metadata=None, payload=b"12345678"):
    metadata = metadata or []
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, value in metadata:
        data += string(key) + struct.pack("<II", 4, value)
    data += string("test.weight") + struct.pack("<I", len(shape))
    data += b"".join(struct.pack("<Q", d) for d in shape)
    data += struct.pack("<IQ", kind, offset)
    data += b"\0" * (-len(data) % 32)
    path = tmp_path / "test.gguf"
    path.write_bytes(data + payload)
    return path


def test_gguf_bf16_span_and_partial_header(tmp_path):
    path = gguf(tmp_path)
    source = edge.read_gguf(path)
    assert source.tensors["test.weight"] == edge.Tensor("test.weight", (4,), "bf16", source.data_start, 8)
    full_size = path.stat().st_size
    path.write_bytes(path.read_bytes()[: source.data_start])
    assert edge.read_gguf(path, expected_size=full_size) == source
    with pytest.raises(ConversionError, match="spans"):
        edge.read_gguf(path)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"offset": 32}, "spans"),
        ({"kind": 11}, "encoding"),
        ({"shape": (0,)}, "dimensions"),
        ({"shape": (400000001,)}, "dimensions"),
        ({"shape": (31,), "kind": 8}, "alignment"),
        ({"metadata": [("same", 1), ("same", 2)]}, "Duplicate"),
        ({"metadata": [("general.alignment", 64)]}, "alignment"),
        ({"payload": b"123456789"}, "trailing"),
    ],
)
def test_gguf_rejects_malformed_layouts(tmp_path, change, message):
    with pytest.raises((ConversionError, UnsupportedConversion), match=message):
        edge.read_gguf(gguf(tmp_path, **change))


def test_gguf_oversized_metadata_does_not_allocate(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 1, 1, 2**63))
    with pytest.raises(UnsupportedConversion, match="string"):
        edge.read_gguf(path)


def index(tmp_path, rows=None, *, pool=64):
    rows = rows or [
        "test.weight 0 4 0 8 0 8 4 0 0 0 1 0 0 0 0 0 0 0",
        "native.weight 0 0 0 0 64 0 32 2 8 0 32 0 0 0 0 0 0 0",
    ]
    path = tmp_path / "index.txt"
    path.write_text(
        "# strata pack index v3 -- independent synthetic fixture\n"
        f"# align 64 pool {pool} tensors {len(rows)}\n" + "\n".join(rows) + "\n"
    )
    return path


def test_pack_native_tensors_consume_no_dense_bytes(tmp_path):
    tensors = edge.read_pack_index(index(tmp_path), 64)
    assert tensors["test.weight"].nbytes == 8
    assert tensors["native.weight"].nbytes == 68


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("0 4 0 8", "1 4 0 8", "layout"),
        ("0 4 0 8", "0 4 64 8", "span"),
        ("0 4 0 8", "0 4 0 10", "span"),
        ("64 0 32 2", "0 0 32 2", "pool offset"),
        ("32 2 8 0 32", "31 2 8 0 32", "native tensor"),
        ("test.weight", "../outside", "tensor"),
        ("0 4 0 8", "0 4 -1 8", "integer"),
    ],
)
def test_pack_rejects_mutated_records(tmp_path, old, new, message):
    path = index(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises((ConversionError, UnsupportedConversion), match=message):
        edge.read_pack_index(path, 64)


def test_pack_rejects_extra_dense_bytes_and_duplicate_names(tmp_path):
    with pytest.raises(ConversionError, match="pool size"):
        edge.read_pack_index(index(tmp_path), 128)
    path = index(tmp_path)
    path.write_text(path.read_text().replace("native.weight", "test.weight"))
    with pytest.raises(ConversionError, match="duplicate"):
        edge.read_pack_index(path, 64)


def native_manifest(tmp_path):
    path = tmp_path / "native_experts.txt"
    path.write_text(
        "# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)\n"
        + "".join(f"{i} 11 20 {i * 834928640} 1630720\n" for i in range(48))
    )
    return path


def test_native_expert_record_formula(tmp_path):
    result = edge.read_native_experts(native_manifest(tmp_path), 40076574720)
    assert result["record_bytes"] == 1630720
    assert result["gate_bytes"] == 492800
    assert result["down_bytes"] == 645120


@pytest.mark.parametrize(
    "old,new", [("v6:", "v5:"), ("0 11 20 0", "0 11 11 0"), ("1 11 20 834928640", "1 11 20 0"), ("1630720", "1630784")]
)
def test_native_manifest_rejects_other_arithmetic_or_overlaps(tmp_path, old, new):
    path = native_manifest(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        edge.read_native_experts(path, 40076574720)


def test_duplicate_tensor_checks_payload_and_geometry(tmp_path):
    path = gguf(tmp_path)
    skeleton = edge.read_gguf(path)
    dense = tmp_path / "dense.bin"
    dense.write_bytes(b"12345678")
    tensor = replace(skeleton.tensors["test.weight"], offset=0)
    pack = {tensor.name: tensor}
    assert edge.compare_duplicate_tensors(skeleton, pack, path, dense)["bytes"] == 8
    dense.write_bytes(b"12345679")
    with pytest.raises(ConversionError, match="payload mismatch"):
        edge.compare_duplicate_tensors(skeleton, pack, path, dense)
    dense.write_bytes(b"1234567")
    with pytest.raises(ConversionError, match="payload mismatch"):
        edge.compare_duplicate_tensors(skeleton, pack, path, dense)
    pack[tensor.name] = replace(tensor, shape=(8,))
    with pytest.raises(ConversionError, match="geometry"):
        edge.compare_duplicate_tensors(skeleton, pack, path, dense)


def test_duplicate_comparison_is_chunk_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(edge, "_CHUNK", 3)
    path = gguf(tmp_path)
    source = edge.read_gguf(path)
    dense = tmp_path / "dense.bin"
    dense.write_bytes(b"12345678")
    tensor = replace(source.tensors["test.weight"], offset=0)
    assert edge.compare_duplicate_tensors(source, {tensor.name: tensor}, path, dense)["tensors"] == 1


@pytest.mark.parametrize("bad", [8, 0xFFFFFFFF])
def test_projection_labels_reject_out_of_range_and_signed_aliases(tmp_path, bad):
    path = tmp_path / "labels.bin"
    path.write_bytes(struct.pack("<I", 7) * (48 * 512))
    edge.validate_labels(path)
    with path.open("r+b") as stream:
        stream.seek(4 * 511)
        stream.write(struct.pack("<I", bad))
    with pytest.raises(ConversionError, match="0..7"):
        edge.validate_labels(path)
    path.write_bytes(b"\0" * (48 * 512 * 4 - 1))
    with pytest.raises(ConversionError, match="count"):
        edge.validate_labels(path)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_column_scales_require_positive_finite_values(tmp_path, bad):
    path = tmp_path / "l0.f32"
    path.write_bytes(struct.pack("<f", 1.0) * 1792)
    edge.validate_column_scales(path)
    with path.open("r+b") as stream:
        stream.write(struct.pack("<f", bad))
    with pytest.raises(ConversionError, match="finite and positive"):
        edge.validate_column_scales(path)


def test_mtp_dense_index_rejects_offsets_and_wrong_row_encoding(tmp_path):
    path = tmp_path / "dense.txt"
    lines = [f"norm.{i}.weight f32 1 4 {i * 16} 16" for i in range(29)]
    path.write_text("\n".join(lines))
    assert len(edge.read_mtp_index(path, 29 * 16)) == 29
    path.write_text("\n".join(lines).replace("norm.1.weight f32 1 4 16 16", "norm.1.weight f32 1 4 0 16"))
    with pytest.raises(ConversionError, match="span"):
        edge.read_mtp_index(path, 29 * 16)
    path.write_text("\n".join(lines).replace("f32 1 4", "q8_0 1 4"))
    with pytest.raises(ConversionError, match="span"):
        edge.read_mtp_index(path, 29 * 16)
    path.write_text("\n".join(lines))
    with pytest.raises(ConversionError, match="file size"):
        edge.read_mtp_index(path, 29 * 16 + 1)


@pytest.mark.parametrize("name", ["../outside", "/outside", "a/../outside", "a//outside", "./outside"])
def test_source_paths_reject_traversal(tmp_path, name):
    with pytest.raises(ConversionError, match="Unsafe"):
        edge.source_path(tmp_path, name)


def test_source_paths_reject_directory_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "data").write_bytes(b"data")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(ConversionError, match="symlinks"):
        edge.source_path(tmp_path, "alias/data")


@pytest.mark.parametrize("value", [49, True, 48.0])
def test_skeleton_rejects_wrong_geometry_or_metadata_types(value):
    metadata = {"general.architecture": "qwen4exp", **edge.EXPECTED_GEOMETRY}
    metadata["qwen4exp.block_count"] = value
    with pytest.raises(ConversionError, match="geometry"):
        edge.validate_skeleton(edge.GGUFSource(metadata, {}, 0))


def test_header_preflight_offline_requires_verified_capture(tmp_path):
    with pytest.raises(ConversionError, match="unavailable offline"):
        edge.fetch_skeleton_metadata(tmp_path, offline=True)
    (tmp_path / ".cache" / "edge0-skeleton-header.gguf").write_bytes(b"invalid")
    with pytest.raises(ConversionError, match="unavailable offline"):
        edge.fetch_skeleton_metadata(tmp_path, offline=True)


def test_header_preflight_rejects_server_ignoring_range(tmp_path, monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            pytest.fail("server body must not be read when Range was ignored")

    monkeypatch.setattr(edge.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(ConversionError, match="bounded byte range"):
        edge.fetch_skeleton_metadata(tmp_path)


def test_header_preflight_hash_and_atomic_cache(tmp_path, monkeypatch):
    data = b"synthetic header"
    monkeypatch.setattr(edge, "SKELETON_HEADER_BYTES", len(data))
    monkeypatch.setattr(edge, "SKELETON_HEADER_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(edge, "read_gguf", lambda *a, **kw: edge.GGUFSource({"checked": True}, {}, 0))
    monkeypatch.setattr(edge, "validate_skeleton", lambda source: None)

    class Response:
        status = 206
        headers = {"Content-Range": f"bytes 0-{len(data) - 1}/5991569248"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            assert n == len(data) + 1
            return data

    monkeypatch.setattr(edge.urllib.request, "urlopen", lambda *a, **kw: Response())
    assert edge.fetch_skeleton_metadata(tmp_path) == {"checked": True}
    assert edge.fetch_skeleton_metadata(tmp_path, offline=True) == {"checked": True}
    assert list((tmp_path / ".cache").iterdir()) == [tmp_path / ".cache" / "edge0-skeleton-header.gguf"]
