"""Merge four private singleton captures into the Stark slice fixture ABI."""

import json
import math
import os
import struct
import sys
from pathlib import Path

WIDTH = 2304
KV = 1024
TOP_K = 8
MID = 1152


def read(path):
    data = path.read_bytes()
    (length,) = struct.unpack_from("<Q", data)
    header = json.loads(data[8 : 8 + length])
    arrays = {}
    for name, field in header.items():
        if name == "__metadata__":
            continue
        start, end = field["data_offsets"]
        arrays[name] = (
            field["dtype"],
            field["shape"],
            data[8 + length + start : 8 + length + end],
        )
    return arrays


def write(path, arrays):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in sorted(arrays.items()):
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def merge(directory):
    rows = [read(directory / f"row-{row}.safetensors") for row in range(4)]
    first = rows[0]
    out = {"start": first["start"]}
    (start,) = struct.unpack("<I", first["start"][2])
    for name, (dtype, shape, data) in first.items():
        if name == "start" or ".history." in name:
            continue
        if name.endswith((".keys", ".values")):
            out[name] = dtype, shape, data
            continue
        combined = b"".join(row[name][2] for row in rows)
        if name in {"blocks", "previous"}:
            new_shape = [4, 2, WIDTH]
        elif name.endswith(".messages"):
            new_shape = [4, 4, WIDTH]
        elif name.endswith((".mid", ".down_inner")):
            new_shape = [4, TOP_K, MID]
        else:
            new_shape = [4, math.prod(shape)]
        if math.prod(shape) != math.prod(new_shape) // 4:
            raise ValueError(f"{name} has unexpected oracle shape {shape}")
        if name.endswith(".routes") and dtype == "I32":
            dtype = "U32"  # MLX route indices are nonnegative vocabulary integers.
        if name in {"0.trace.key_raw", "0.trace.value_raw"}:
            name = name.replace("key_raw", "k_raw").replace("value_raw", "v_raw")
        out[name] = dtype, new_shape, combined
    for layer in (10, 11):
        channels = {"key": KV, "value": KV, "attn": WIDTH, "mlp": WIDTH}
        history = [first[f"{layer}.history.{kind}"] for kind in channels]
        if any(
            dtype != "BF16" or math.prod(shape) != 3 * width
            for (dtype, shape, _), width in zip(history, channels.values())
        ):
            raise ValueError(f"invalid layer {layer} BF16 convolution history")
        width = sum(channels.values())
        ring = []
        for (_, _, data), size in zip(history, channels.values()):
            # Oldest state is absolute position start-3; Stark indexes by position % 3.
            split = ((-start) % 3) * size * 2
            ring.append(data[split:] + data[:split])
        out[f"{layer}.conv"] = "BF16", [3 * width], b"".join(ring)
        raw = []
        for row in rows:
            for kind, size in channels.items():
                dtype, shape, data = row[f"{layer}.trace.{kind}_raw"]
                if dtype != "BF16" or math.prod(shape) != size:
                    raise ValueError(f"invalid layer {layer} {kind} convolution input")
                raw.append(data)
        out[f"{layer}.conv_rows"] = "BF16", [4, width], b"".join(raw)
        for kind in ("key", "value"):
            out[f"{layer}.trace.{kind[0]}_raw"] = out.pop(f"{layer}.trace.{kind}_raw")
    write(directory / "fixture.safetensors", out)


if __name__ == "__main__":
    os.umask(0o077)
    merge(Path(sys.argv[1]))
