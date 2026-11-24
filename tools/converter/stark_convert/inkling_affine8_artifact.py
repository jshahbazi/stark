"""Build Stark's native Inkling 2 Mini artifact with 8-bit affine dense weights.

The artifact is derived only from the pinned original checkpoint
(``thinkingmachines/Inkling-2-Mini`` at revision
``6e7c8248f75265bdd4fbab5273304243b5d8c2cb``, Apache-2.0). It keeps Stark's
(original) tensor names. The 178 tensors that MLX-Serve quantizes in its
published 8-bit pack (every attention projection, the dynamic convolution
weight projections of the four global layers, the token embedding and the
untied output head) are replaced by MLX affine 8-bit codes with group size 64:
``value = code * scale + bias`` per row and per 64 consecutive inputs, with an
unsigned 8-bit code and BF16 scale and bias. The codes are produced by
``mlx_affine.quantize_rows``, an independent bit-exact transcription of MLX's
``affine_quantize`` (MIT License, Copyright (c) 2023 Apple Inc., commit
58d91ee81a86dd9710123ec42e7ff91f6d1d21b1). Everything else (expert banks,
router, gates, norms, convolutions, vision tower) is copied byte for byte and
the draft head ``mtp.safetensors`` is copied unchanged.

For a quantized original tensor ``NAME`` of shape ``[rows, columns]`` the
artifact stores, in place of ``NAME``:

    NAME.affine8_codes   U8   [rows, columns]
    NAME.affine8_scales  BF16 [rows, columns / 64]
    NAME.affine8_biases  BF16 [rows, columns / 64]

Command line (aggregates only; never prints tensor values)::

    python -m stark_convert.inkling_affine8_artifact build ORIGINAL_DIR OUTPUT_DIR [--tensors-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

from . import inkling_prepare
from ._build_metadata import file_hash
from .inkling_affine8 import GROUP_SIZE, quantized_mapping
from .mlx_affine import iter_quantized_bf16
from .streaming import DTYPE_BYTES, TensorInfo, _open_tensor, read_safetensors_header

# Published by the default catalog choice, inkling_prepare.CHOICE.
LAYOUT = inkling_prepare.AFFINE8_LAYOUT
QUANTIZATION = inkling_prepare.AFFINE8_QUANTIZATION
SUFFIXES = ("affine8_codes", "affine8_scales", "affine8_biases")
QUANTIZED_TENSORS = 178
PROVENANCE = "affine8-provenance.json"
ALGORITHM = {
    "name": "mlx-affine-8bit",
    "bits": 8,
    "group_size": GROUP_SIZE,
    "mode": "affine",
    "value": "code * scale + bias, code u8, scale and bias BF16 per row and group",
    "reference": "ml-explore/mlx mlx/backend/metal/kernels/quantized.h affine_quantize",
    "reference_revision": "58d91ee81a86dd9710123ec42e7ff91f6d1d21b1",
    "reference_license": "MIT, Copyright (c) 2023 Apple Inc.",
    "implementation": "stark_convert.mlx_affine (independent NumPy transcription; nothing imported from MLX)",
    "tensor_selection": "ddalcu/mlx-serve tests/convert_inkling2.py",
    "tensor_selection_revision": "2fe9be834a601c70ef1a597563026340f0e722fe",
    "tensor_selection_license": "Apache-2.0",
}
COPIED_METADATA = (
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "special_tokens_map.json",
    "README.md",
)


def quantized_names(original: dict[str, TensorInfo]) -> list[str]:
    """Original names of the tensors replaced by affine codes."""
    names = sorted(quantized_mapping(original))
    if len(names) != QUANTIZED_TENSORS:
        raise ValueError("Unexpected number of Inkling affine tensors")
    for name in names:
        tensor = original[name]
        if tensor.dtype != "BF16" or len(tensor.shape) != 2 or tensor.shape[1] % GROUP_SIZE:
            raise ValueError("Inkling affine source must be a BF16 matrix with whole groups")
    return names


def plan(original: dict[str, TensorInfo], tensors_only: bool = False) -> list[tuple[str, str, tuple, str]]:
    """Ordered output entries: (output name, dtype, shape, original source name)."""
    quantized = set(quantized_names(original))
    entries = []
    for name, tensor in sorted(original.items(), key=lambda item: item[1].offset):
        if name in quantized:
            rows, columns = tensor.shape
            groups = columns // GROUP_SIZE
            entries.append((f"{name}.{SUFFIXES[0]}", "U8", (rows, columns), name))
            entries.append((f"{name}.{SUFFIXES[1]}", "BF16", (rows, groups), name))
            entries.append((f"{name}.{SUFFIXES[2]}", "BF16", (rows, groups), name))
        elif not tensors_only:
            entries.append((name, tensor.dtype, tensor.shape, name))
    return entries


def _nbytes(dtype: str, shape: tuple) -> int:
    count = DTYPE_BYTES[dtype]
    for dim in shape:
        count *= dim
    return count


def _header(entries, metadata: dict[str, str]) -> bytes:
    header: dict = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, _ in entries:
        size = _nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    return raw + b" " * (-len(raw) % 8)


class _Rows:
    def __init__(self, tensor: TensorInfo):
        self.tensor = tensor
        self.row_bytes = tensor.nbytes // tensor.shape[0]
        self.stream = _open_tensor(tensor)

    def __call__(self, start: int, count: int) -> bytes:
        self.stream.seek(self.tensor.offset + start * self.row_bytes)
        return self.stream.read(count * self.row_bytes)

    def close(self) -> None:
        self.stream.close()


def _copy(tensor: TensorInfo, output) -> None:
    with _open_tensor(tensor) as stream:
        remaining = tensor.nbytes
        while remaining:
            chunk = stream.read(min(64 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Inkling source changed during the build")
            output.write(chunk)
            remaining -= len(chunk)


def _quantize(tensor: TensorInfo, output) -> None:
    rows, columns = tensor.shape
    groups = columns // GROUP_SIZE
    scales = np.empty((rows, groups), dtype="<u2")
    biases = np.empty((rows, groups), dtype="<u2")
    reader = _Rows(tensor)
    try:
        for start, quantized in iter_quantized_bf16(reader, tensor.shape, GROUP_SIZE, 4096):
            count = quantized.codes.shape[0]
            output.write(np.ascontiguousarray(quantized.codes).tobytes())
            scales[start : start + count] = quantized.scales
            biases[start : start + count] = quantized.biases
    finally:
        reader.close()
    output.write(scales.tobytes())
    output.write(biases.tobytes())


def write_target(original: dict[str, TensorInfo], path: Path, metadata: dict[str, str], tensors_only: bool) -> None:
    entries = plan(original, tensors_only)
    header = _header(entries, metadata)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        index = 0
        while index < len(entries):
            name, _, _, source = entries[index]
            if name.endswith(SUFFIXES[0]):
                _quantize(original[source], output)
                index += 3
            else:
                _copy(original[source], output)
                index += 1
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def _installed_config(raw: bytes, template: bytes) -> bytes:
    config = json.loads(raw)
    config["stark_inkling"] = {"source": inkling_prepare.REPO, "revision": inkling_prepare.REVISION}
    config["stark_inkling_affine8"] = {"layout": LAYOUT, "group_size": GROUP_SIZE, "bits": 8}
    config["token_codec_config"] = {"type": "ChatCodecConfig", "prompt_template": template.decode("utf-8")}
    return (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()


def build(original_dir: Path, output_dir: Path, tensors_only: bool = False) -> dict:
    """Verify the pinned original and write the derived artifact (or only its affine tensors)."""
    original_dir = original_dir.resolve()
    target = original_dir / inkling_prepare.TARGET
    size, digest = inkling_prepare.WEIGHTS[inkling_prepare.TARGET]
    if target.stat().st_size != size or file_hash(target) != digest:
        raise ValueError("Inkling original target identity changed")
    _, original = read_safetensors_header(target)
    names = quantized_names(original)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "stark_layout": LAYOUT,
        "source_repo": inkling_prepare.REPO,
        "source_revision": inkling_prepare.REVISION,
        "source_target_sha256": digest,
        "algorithm": ALGORITHM["name"],
        "algorithm_reference_revision": ALGORITHM["reference_revision"],
        "group_size": str(GROUP_SIZE),
    }
    began = time.monotonic()
    name = "affine8-tensors.safetensors" if tensors_only else inkling_prepare.TARGET
    write_target(original, output_dir / name, metadata, tensors_only)
    result = {
        "layout": LAYOUT,
        "quantized_tensors": len(names),
        "target": name,
        "target_bytes": (output_dir / name).stat().st_size,
        "target_sha256": file_hash(output_dir / name),
        "seconds": round(time.monotonic() - began, 1),
    }
    if tensors_only:
        return result
    draft_size, draft_digest = inkling_prepare.WEIGHTS[inkling_prepare.DRAFT]
    draft = original_dir / inkling_prepare.DRAFT
    if draft.stat().st_size != draft_size or file_hash(draft) != draft_digest:
        raise ValueError("Inkling draft identity changed")
    shutil.copyfile(draft, output_dir / inkling_prepare.DRAFT)
    for item in COPIED_METADATA:
        if (original_dir / item).is_file():
            shutil.copyfile(original_dir / item, output_dir / item)
    # A folder prepared from the original keeps the upstream config beside the installed one.
    source_config = inkling_prepare.CONFIG if (original_dir / inkling_prepare.CONFIG).is_file() else "config.json"
    raw = (original_dir / source_config).read_bytes()
    if hashlib.sha256(raw).hexdigest() != inkling_prepare.CONFIG_SHA:
        raise ValueError("Inkling original config identity changed")
    template = (output_dir / "chat_template.jinja").read_bytes()
    if hashlib.sha256(template).hexdigest() != inkling_prepare.TEMPLATE_SHA:
        raise ValueError("Inkling template identity changed")
    (output_dir / inkling_prepare.CONFIG).write_bytes(raw)
    encoded = _installed_config(raw, template)
    (output_dir / "config.json").write_bytes(encoded)
    _, written = read_safetensors_header(output_dir / inkling_prepare.TARGET)
    index = {
        "metadata": {"total_size": sum(t.nbytes for t in written.values())},
        "weight_map": dict.fromkeys(sorted(written), inkling_prepare.TARGET),
    }
    (output_dir / inkling_prepare.INDEX).write_text(json.dumps(index, indent=2) + "\n")
    provenance = {
        "schema_version": 1,
        "layout": LAYOUT,
        "source": {
            "repo": inkling_prepare.REPO,
            "revision": inkling_prepare.REVISION,
            "license": "Apache-2.0",
            "target_sha256": digest,
            "target_bytes": size,
            "draft_sha256": draft_digest,
            "config_sha256": inkling_prepare.CONFIG_SHA,
        },
        "algorithm": ALGORITHM,
        "quantized": {"count": len(names), "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()},
        "unchanged": "every other target tensor byte for byte; mtp.safetensors byte for byte",
        "target_sha256": result["target_sha256"],
        "target_bytes": result["target_bytes"],
    }
    (output_dir / PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")
    receipt = _receipt(encoded, result["target_bytes"], result["target_sha256"])
    (output_dir / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


def _receipt(encoded: bytes, target_bytes: int, target_sha256: str) -> dict:
    draft_size, draft_digest = inkling_prepare.WEIGHTS[inkling_prepare.DRAFT]
    return {
        "schema_version": 7,
        "layout": LAYOUT,
        "architecture": inkling_prepare.FAMILY,
        "artifact_mode": "vision-language",
        "quantization": QUANTIZATION,
        "weights_source": {"repo": inkling_prepare.REPO, "revision": inkling_prepare.REVISION},
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "weight_bytes": target_bytes + draft_size,
        "artifact_sha256": target_sha256,
        "draft_sha256": draft_digest,
        "processor_sha256": inkling_prepare.PROCESSOR_SHA,
        "tokenizer_sha256": inkling_prepare.TOKENIZER_SHA,
        "template_sha256": inkling_prepare.TEMPLATE_SHA,
        "native_loading": "unverified",
        "execution": "unverified",
    }


def prepared_identity(root: Path, config: dict, receipt: dict) -> dict | None:
    """Recognize a built 8-bit artifact directory by bounded inspection.

    Like the original layout's check, this reads metadata and file sizes only;
    the native loader verifies the pinned target hash. It does not mark the
    artifact executed or grant image qualification.
    """
    if config.get("stark_inkling_affine8") != {"layout": LAYOUT, "group_size": GROUP_SIZE, "bits": 8}:
        return None
    raw = inkling_prepare._read(root, inkling_prepare.CONFIG, 1024 * 1024, inkling_prepare.CONFIG_SHA)
    template = inkling_prepare._read(root, "chat_template.jinja", 256 * 1024, inkling_prepare.TEMPLATE_SHA)
    encoded = inkling_prepare._read(root, "config.json", 1024 * 1024)
    expected = _receipt(encoded, inkling_prepare.AFFINE8_TARGET_BYTES, inkling_prepare.AFFINE8_TARGET_SHA)
    if encoded != _installed_config(raw, template) or receipt != expected:
        return None
    inkling_prepare._read(root, "processor_config.json", 128 * 1024, inkling_prepare.PROCESSOR_SHA)
    inkling_prepare._read(root, "hf_quant_config.json", 1024 * 1024, inkling_prepare.QUANT_SHA)
    draft_size = inkling_prepare.WEIGHTS[inkling_prepare.DRAFT][0]
    for name, size in (
        (inkling_prepare.TARGET, inkling_prepare.AFFINE8_TARGET_BYTES),
        (inkling_prepare.DRAFT, draft_size),
    ):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            return None
    tokenizer = root / "tokenizer.json"
    if tokenizer.is_symlink() or not tokenizer.is_file() or not 0 < tokenizer.stat().st_size <= 32 * 1024 * 1024:
        return None
    return {
        "repo": inkling_prepare.REPO,
        "revision": inkling_prepare.REVISION,
        "architecture": inkling_prepare.FAMILY,
        "weight_bytes": inkling_prepare.AFFINE8_TARGET_BYTES + draft_size,
        "quantization": QUANTIZATION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("build", help="Write the derived artifact from the pinned original")
    command.add_argument("original", help="Directory of the pinned original checkpoint")
    command.add_argument("output", help="Output directory (must be ignored by Git)")
    command.add_argument("--tensors-only", action="store_true", help="Write only the affine tensors")
    args = parser.parse_args(argv)
    print(json.dumps(build(Path(args.original), Path(args.output), args.tensors_only)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
