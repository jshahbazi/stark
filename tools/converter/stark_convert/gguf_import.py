"""Experimental Qwen3 Q8_0 GGUF importer into Stark's compressed native layout.

GGUF v3 and Q8_0 contracts follow ggml/llama.cpp (MIT licensed). The importer
implements those contracts independently; upstream revision/provenance belongs
in each experiment's source-bound report. It never computes new quantization.
Original Hugging Face metadata is required and checked against GGUF metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from tokenizers import Tokenizer

from .cli import _publish
from .metadata import write_metadata
from .provenance import file_hash, toolchain_provenance
from .qwen3 import build_qwen3_plan
from .registry import get_model
from .streaming import (
    DTYPE_BYTES,
    ConversionError,
    TensorInfo,
    TensorOperation,
    UnsupportedConversion,
    _cast,
    _cast_f32,
    _cast_values,
    _copy,
    _HashingWriter,
    _signature,
    read_json,
    read_safetensors_header,
    validate_planning_layers,
)

GGML_REFERENCE_REVISION = "1b70350c92563449576bcbbf552994b1d1ed847d"
_MAX_COUNT = 2_000_000
_MAX_STRING = 64 * 1024 * 1024
_MAX_HEADER = 100_000_000
_TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34)}
_VALUES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    kind: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUFSource:
    path: Path
    signature: tuple[int, int, int, int]
    metadata: dict
    tensors: dict[str, GGUFTensor]


def _read(stream: BinaryIO, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ConversionError("Truncated GGUF source")
    return value


class _HeaderReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def read(self, count: int) -> bytes:
        if count < 0 or self.stream.tell() + count > _MAX_HEADER:
            raise UnsupportedConversion("GGUF header exceeds experiment limits")
        return self.stream.read(count)

    def tell(self) -> int:
        return self.stream.tell()


def _number(stream: BinaryIO, code: str):
    return struct.unpack("<" + code, _read(stream, struct.calcsize(code)))[0]


def _count(stream: BinaryIO) -> int:
    value = _number(stream, "Q")
    if value > _MAX_COUNT:
        raise UnsupportedConversion("GGUF inventory exceeds experiment limits")
    return value


def _string(stream: BinaryIO) -> str:
    length = _number(stream, "Q")
    if length > _MAX_STRING:
        raise UnsupportedConversion("GGUF string exceeds experiment limits")
    try:
        return _read(stream, length).decode("utf-8")
    except UnicodeError:
        raise ConversionError("Invalid GGUF text encoding") from None


def _value(stream: BinaryIO, kind: int):
    if kind == 8:
        return _string(stream)
    if kind == 9:
        subtype, count = _number(stream, "I"), _count(stream)
        if subtype == 9:
            raise UnsupportedConversion("Nested GGUF arrays are outside experiment coverage")
        return [_value(stream, subtype) for _ in range(count)]
    if kind not in _VALUES:
        raise UnsupportedConversion("GGUF metadata type is outside experiment coverage")
    if kind == 7:
        value = _number(stream, "B")
        if value not in {0, 1}:
            raise ConversionError("Invalid GGUF boolean")
        return bool(value)
    value = _number(stream, _VALUES[kind])
    if isinstance(value, float) and not math.isfinite(value):
        raise ConversionError("Nonfinite GGUF metadata")
    return value


def read_gguf(path: Path) -> GGUFSource:
    """Validate one little-endian GGUF v3 file, without reading weight payloads."""
    path = path.resolve()
    with path.open("rb") as stream:
        signature = _signature(os.fstat(stream.fileno()))
        stream = _HeaderReader(stream)
        if _read(stream, 4) != b"GGUF" or _number(stream, "I") != 3:
            raise UnsupportedConversion("Experiment requires little-endian GGUF version 3")
        tensor_count, metadata_count = _count(stream), _count(stream)
        if not tensor_count:
            raise ConversionError("GGUF source has no tensors")
        metadata = {}
        for _ in range(metadata_count):
            key = _string(stream)
            if not key or key in metadata:
                raise ConversionError("Duplicate or empty GGUF metadata key")
            metadata[key] = _value(stream, _number(stream, "I"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise ConversionError("Invalid GGUF alignment")
        tensors = {}
        for _ in range(tensor_count):
            name, rank = _string(stream), _number(stream, "I")
            if not name or name in tensors or rank not in {1, 2}:
                raise UnsupportedConversion("GGUF tensor name or rank is outside experiment coverage")
            dimensions = tuple(_number(stream, "Q") for _ in range(rank))
            if any(not 0 < value <= _MAX_COUNT for value in dimensions):
                raise ConversionError("Invalid GGUF tensor dimensions")
            type_id, offset = _number(stream, "I"), _number(stream, "Q")
            if type_id not in _TYPES:
                raise UnsupportedConversion("Experiment supports Q8_0, F16 and F32 tensors only")
            kind, block_size, block_bytes = _TYPES[type_id]
            if dimensions[0] % block_size or offset % alignment:
                raise ConversionError("Invalid GGUF block or tensor alignment")
            nbytes = math.prod(dimensions) // block_size * block_bytes
            tensors[name] = GGUFTensor(name, dimensions[::-1], kind, offset, nbytes)
        data_start = stream.tell() + (-stream.tell() % alignment)
        spans = sorted((tensor.offset, tensor.offset + tensor.nbytes) for tensor in tensors.values())
        end = 0
        for start, stop in spans:
            if start < end or start != end + (-end % alignment) or data_start + stop > signature[2]:
                raise ConversionError("Invalid GGUF tensor spans")
            end = stop
        if data_start + end != signature[2]:
            raise ConversionError("GGUF contains unindexed trailing data")
        tensors = {
            name: GGUFTensor(name, tensor.shape, tensor.kind, data_start + tensor.offset, tensor.nbytes)
            for name, tensor in tensors.items()
        }
    return GGUFSource(path, signature, metadata, tensors)


def _hf_names(config: dict) -> dict[str, str]:
    validate_planning_layers(config.get("num_hidden_layers"))
    names = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight"}
    if not config.get("tie_word_embeddings", False):
        names["output.weight"] = "lm_head.weight"
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        for gguf, hf in {
            "attn_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
            "attn_q_norm": "self_attn.q_norm",
            "attn_k_norm": "self_attn.k_norm",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_down": "mlp.down_proj",
        }.items():
            names[f"blk.{layer}.{gguf}.weight"] = f"{prefix}.{hf}.weight"
    return names


def validate_metadata(source: GGUFSource, hf_source: Path, config: dict) -> None:
    """Bind the existing HF metadata route to the GGUF architecture/vocabulary."""
    metadata = source.metadata
    if metadata.get("general.architecture") != "qwen3" or config.get("model_type") != "qwen3":
        raise UnsupportedConversion("Experiment supports Qwen3 only")
    if config.get("attention_bias", False) or config.get("use_sliding_window", False):
        raise UnsupportedConversion("Experiment requires unbiased full Qwen3 attention")
    if any(
        key.startswith("qwen3.rope.") and key not in {"qwen3.rope.freq_base", "qwen3.rope.dimension_count"}
        for key in metadata
    ):
        raise UnsupportedConversion("GGUF RoPE variant is outside experiment coverage")
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    checks = {
        "qwen3.embedding_length": config["hidden_size"],
        "qwen3.block_count": config["num_hidden_layers"],
        "qwen3.feed_forward_length": config["intermediate_size"],
        "qwen3.attention.head_count": config["num_attention_heads"],
        "qwen3.attention.head_count_kv": config.get("num_key_value_heads", config["num_attention_heads"]),
        "qwen3.attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        "qwen3.rope.freq_base": float(config["rope_theta"]),
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if type(expected) is int:
            valid = type(actual) is int and actual == expected
        else:
            valid = type(actual) in {int, float} and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0)
        if not valid:
            raise ConversionError("GGUF and HF architecture metadata disagree")
    for key, expected in {
        "qwen3.attention.key_length": head_dim,
        "qwen3.attention.value_length": head_dim,
        "qwen3.rope.dimension_count": head_dim,
        "qwen3.context_length": config["max_position_embeddings"],
    }.items():
        if key in metadata and metadata[key] != expected:
            raise ConversionError("GGUF and HF attention metadata disagree")
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != "qwen2":
        raise UnsupportedConversion("Experiment requires the Qwen2 byte BPE tokenizer convention")
    tokens = metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or len(tokens) != config["vocab_size"]:
        raise ConversionError("GGUF and HF vocabulary dimensions disagree")
    try:
        tokenizer = Tokenizer.from_file(str(hf_source / "tokenizer.json"))
        vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    except Exception:
        raise ConversionError("Cannot inspect original HF tokenizer") from None
    if any(
        type(index) is not int or index >= len(tokens) or tokens[index] != token for token, index in vocabulary.items()
    ):
        raise ConversionError("GGUF and HF tokenizer vocabulary disagree")
    tokenizer_json = read_json(hf_source / "tokenizer.json")
    if tokenizer_json.get("model", {}).get("type") != "BPE":
        raise UnsupportedConversion("Experiment requires original Qwen3 HF BPE metadata")
    merges = tokenizer_json.get("model", {}).get("merges")
    if isinstance(merges, list):
        normalized_merges = [" ".join(merge) if isinstance(merge, list) else merge for merge in merges]
        if metadata.get("tokenizer.ggml.merges") != normalized_merges:
            raise ConversionError("GGUF and HF tokenizer merges disagree")
    eos = config.get("eos_token_id")
    eos_ids = eos if isinstance(eos, list) else [eos]
    if metadata.get("tokenizer.ggml.eos_token_id") not in eos_ids:
        raise ConversionError("GGUF and HF stop metadata disagree")
    for name in ("bos", "pad"):
        key = f"tokenizer.ggml.{name}_token_id"
        if key in metadata and config.get(f"{name}_token_id") is not None:
            if metadata[key] != config[f"{name}_token_id"]:
                raise ConversionError("GGUF and HF special token metadata disagree")


def build_import_plan(source: GGUFSource, hf_source: Path):
    config = read_json(hf_source / "config.json")
    validate_metadata(source, hf_source, config)
    layers = config.get("num_hidden_layers")
    validate_planning_layers(layers, source.tensors)
    expected_count = 2 + (not config.get("tie_word_embeddings", False)) + 11 * layers
    if expected_count != len(source.tensors):
        raise ConversionError("GGUF tensor inventory disagrees with Qwen3 HF config")
    names = _hf_names(config)
    if set(source.tensors) != set(names):
        raise ConversionError("GGUF tensor inventory disagrees with Qwen3 HF config")
    catalog = {}
    gguf_by_offset = {}
    for name, tensor in source.tensors.items():
        if tensor.kind == "Q8_0" and len(tensor.shape) != 2:
            raise UnsupportedConversion("Q8_0 experiment requires matrix tensors")
        # The shared planner validates logical shapes and architecture mapping.
        # Q8_0 entries use a planning-only F16 descriptor; custom writer operations
        # below read their actual block payloads, never these logical descriptors.
        dtype = "F16" if tensor.kind == "Q8_0" else tensor.kind
        descriptor = TensorInfo(source.path, dtype, tensor.shape, tensor.offset, tensor.nbytes, source.signature)
        catalog[names[name]] = descriptor
        gguf_by_offset[tensor.offset] = tensor
    plan = build_qwen3_plan(hf_source, tensors=catalog)
    operations = []
    metadata = dict(plan.metadata)
    for op in plan.operations:
        representations = [gguf_by_offset[item.offset].kind == "Q8_0" for item in op.sources]
        if any(representations):
            if not all(representations) or len(op.shape) != 2 or not op.name.endswith(".weights"):
                raise UnsupportedConversion("Mixed or nonmatrix Q8_0 fusion is outside experiment coverage")
            rows, columns = op.shape
            if columns % 32:
                raise ConversionError("Q8_0 matrix width must be divisible by 32")
            prefix = op.name.removesuffix(".weights")
            layout = json.loads(metadata[prefix + ".spec"])["layout"]
            shape = (columns // 32, rows + (-rows % 4)) if layout == "output_input" else (rows, columns // 32)
            operations.append(TensorOperation(op.name, op.sources, "U8", (rows, columns), "gguf_q8_codes"))
            scale_kind = "gguf_q8_scales_transpose" if layout == "output_input" else "gguf_q8_scales"
            operations.append(TensorOperation(prefix + ".scales", op.sources, "BF16", shape, scale_kind))
            metadata[prefix + ".spec"] = json.dumps(
                {"type": "IntSpec", "bits": 8, "group_size": 32, "is_symmetric": True, "layout": layout}
            )
        else:
            operations.append(op)
    if not any(op.kind == "gguf_q8_codes" for op in operations):
        raise ConversionError("GGUF source has no compressed Q8_0 matrices")
    return plan, tuple(operations), metadata, config


def _q8_blocks(source: GGUFSource, tensor: TensorInfo, chunk_bytes: int):
    blocks_per_chunk = max(1, chunk_bytes // 512)
    with source.path.open("rb") as stream:
        if _signature(os.fstat(stream.fileno())) != source.signature:
            raise ConversionError("GGUF source changed")
        stream.seek(tensor.offset)
        remaining = tensor.nbytes // 34
        while remaining:
            count = min(remaining, blocks_per_chunk)
            raw = _read(stream, count * 34)
            yield np.frombuffer(raw, dtype="u1").reshape(count, 34)
            remaining -= count


def _write_codes(source: GGUFSource, op: TensorOperation, output: BinaryIO, chunk_bytes: int) -> int:
    peak = 0
    for tensor in op.sources:
        for blocks in _q8_blocks(source, tensor, chunk_bytes):
            codes = np.bitwise_xor(blocks[:, 2:], np.uint8(128))
            output.write(memoryview(codes).cast("B"))
            peak = max(peak, blocks.nbytes + codes.nbytes)
    return peak


def _write_scales(source: GGUFSource, op: TensorOperation, output: BinaryIO, chunk_bytes: int, error: dict) -> int:
    import ml_dtypes

    rows = sum(tensor.shape[0] for tensor in op.sources)
    groups = op.sources[0].shape[1] // 32
    transposed = op.kind == "gguf_q8_scales_transpose"
    peak = 0

    def cast(blocks: np.ndarray):
        values = np.ascontiguousarray(blocks[:, :2]).view("<f2").reshape(-1)
        if not np.isfinite(values).all():
            raise ConversionError("Q8_0 scale is nonfinite")
        converted = _cast_values(values, "F16", "BF16")
        exact = values.astype("<f8")
        difference = np.abs(converted.astype("<f8") - exact)
        error["count"] += values.size
        error["changed"] += int(np.count_nonzero(difference))
        error["squared_error_sum"] += float(np.dot(difference, difference))
        error["max_absolute_error"] = max(error["max_absolute_error"], float(difference.max(initial=0)))
        relative = np.divide(difference, np.abs(exact), out=np.zeros_like(exact), where=exact != 0)
        error["max_relative_error"] = max(error["max_relative_error"], float(relative.max(initial=0)))
        # Closed-form weight error uses block code moments and never materializes
        # dequantized tensors. Signed Q8_0 is d*q; native U8 symmetric is d*(u-128).
        codes = blocks[:, 2:].view("i1").astype("<i2")
        moments = np.sum(codes * codes, axis=1, dtype="<f8")
        error["weight_squared_error_sum"] += float(np.dot(moments, difference * difference))
        error["weight_squared_reference_sum"] += float(np.dot(moments, exact * exact))
        maximum_codes = np.max(np.abs(codes), axis=1).astype("<f8")
        error["weight_max_absolute_error"] = max(
            error["weight_max_absolute_error"], float(np.max(maximum_codes * difference, initial=0))
        )
        error["weight_count"] += values.size * 32
        return converted

    if not transposed:
        for tensor in op.sources:
            for blocks in _q8_blocks(source, tensor, chunk_bytes):
                values = cast(blocks)
                output.write(values.tobytes())
                peak = max(peak, blocks.nbytes + 384 * values.size)
        return peak
    # A bounded output group band permits one pass per source matrix for the
    # usual small coefficient planes. Narrower bands preserve the same bound.
    padded_rows = op.shape[1]
    band_width = min(groups, chunk_bytes // (padded_rows * 2 + 512))
    if not band_width:
        # Keep writes sequential even when one full output group cannot fit.
        for group in range(groups):
            for tensor in op.sources:
                with source.path.open("rb") as stream:
                    if _signature(os.fstat(stream.fileno())) != source.signature:
                        raise ConversionError("GGUF source changed")
                    for row in range(tensor.shape[0]):
                        stream.seek(tensor.offset + (row * groups + group) * 34)
                        blocks = np.frombuffer(_read(stream, 34), dtype="u1").reshape(1, 34)
                        output.write(cast(blocks).tobytes())
                        peak = max(peak, 418)
            output.write(bytes((padded_rows - rows) * 2))
        return peak
    for group in range(0, groups, band_width):
        width = min(band_width, groups - group)
        transposed_values = np.zeros((width, padded_rows), dtype=ml_dtypes.bfloat16)
        available = chunk_bytes - transposed_values.nbytes
        height_limit = max(1, available // (width * 512))
        fused_row = 0
        for tensor in op.sources:
            with source.path.open("rb") as stream:
                if _signature(os.fstat(stream.fileno())) != source.signature:
                    raise ConversionError("GGUF source changed")
                for row in range(0, tensor.shape[0], height_limit):
                    height = min(height_limit, tensor.shape[0] - row)
                    if width == groups:
                        stream.seek(tensor.offset + row * groups * 34)
                        raw = _read(stream, height * width * 34)
                        blocks = np.frombuffer(raw, dtype="u1").reshape(height * width, 34)
                        read_bytes = len(raw)
                    else:
                        blocks = np.empty((height, width, 34), dtype="u1")
                        for index in range(height):
                            stream.seek(tensor.offset + ((row + index) * groups + group) * 34)
                            raw = _read(stream, width * 34)
                            blocks[index] = np.frombuffer(raw, dtype="u1").reshape(width, 34)
                        blocks = blocks.reshape(height * width, 34)
                        read_bytes = blocks.nbytes + len(raw)
                    values = cast(blocks).reshape(height, width)
                    transposed_values[:, fused_row + row : fused_row + row + height] = values.T
                    peak = max(peak, transposed_values.nbytes + read_bytes + 384 * height * width)
                    del blocks, raw, values
            fused_row += tensor.shape[0]
        output.write(memoryview(transposed_values.view("<u2")).cast("B"))
        del transposed_values
    return peak


def write_import(
    source: GGUFSource, operations: tuple[TensorOperation, ...], metadata: dict, path: Path, *, chunk_bytes: int
) -> dict:
    if type(chunk_bytes) is not int or chunk_bytes < 4096:
        raise ConversionError("GGUF workspace must be at least 4096 bytes")
    operations = sorted(operations, key=lambda op: (-DTYPE_BYTES[op.dtype], op.name))
    header, offset = {"__metadata__": metadata}, 0
    for op in operations:
        if op.name in header:
            raise ConversionError("Duplicate GGUF output operation")
        header[op.name] = {"dtype": op.dtype, "shape": list(op.shape), "data_offsets": [offset, offset + op.nbytes]}
        offset += op.nbytes
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    error = {
        "count": 0,
        "changed": 0,
        "squared_error_sum": 0.0,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "weight_squared_error_sum": 0.0,
        "weight_squared_reference_sum": 0.0,
        "weight_max_absolute_error": 0.0,
        "weight_count": 0,
    }
    start, peak = time.perf_counter(), 0
    with path.open("xb") as stream:
        output = _HashingWriter(stream)
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for op in operations:
            before = output.tell()
            if op.kind == "gguf_q8_codes":
                peak = max(peak, _write_codes(source, op, output, chunk_bytes))
            elif op.kind in {"gguf_q8_scales", "gguf_q8_scales_transpose"}:
                peak = max(peak, _write_scales(source, op, output, chunk_bytes, error))
            elif op.kind in {"copy", "fuse"}:
                for item in op.sources:
                    peak = max(peak, _copy(item, output, chunk_bytes))
            elif op.kind == "cast_f32" and len(op.sources) == 1:
                peak = max(peak, _cast_f32(op.sources[0], output, chunk_bytes))
            elif op.kind in {"cast", "fuse_cast"}:
                for item in op.sources:
                    peak = max(peak, _cast(item, output, op.dtype, chunk_bytes))
            else:
                raise UnsupportedConversion("GGUF operation is outside experiment coverage")
            if output.tell() - before != op.nbytes or peak > chunk_bytes:
                raise ConversionError("GGUF writer violated byte or workspace contract")
        output.flush()
        os.fsync(output.fileno())
    error["rms_absolute_error"] = math.sqrt(error.pop("squared_error_sum") / max(error["count"], 1))
    weight_error = error.pop("weight_squared_error_sum")
    weight_reference = error.pop("weight_squared_reference_sum")
    error["weight_relative_rms_error"] = math.sqrt(weight_error / weight_reference) if weight_reference else 0.0
    error["weight_rms_absolute_error"] = math.sqrt(weight_error / max(error["weight_count"], 1))
    read_safetensors_header(path)
    return {
        "output_sha256": output.digest.hexdigest(),
        "output_bytes": path.stat().st_size,
        "output_tensor_bytes": offset,
        "tensors": len(operations),
        "peak_buffer_bytes": peak,
        "chunk_bytes": chunk_bytes,
        "write_seconds": time.perf_counter() - start,
        "scale_f16_to_bf16": error,
    }


def validate_codes(source: GGUFSource, operations: tuple[TensorOperation, ...], path: Path, chunk_bytes: int) -> dict:
    """Independently read back every compressed code, coefficient and padding."""
    tensors = read_safetensors_header(path)[1]
    count, mismatches, scales_count, scales_mismatches, padding_count = 0, 0, 0, 0, 0
    with path.open("rb") as output:
        for op in operations:
            if op.kind != "gguf_q8_codes":
                continue
            output.seek(tensors[op.name].offset)
            for tensor in op.sources:
                for blocks in _q8_blocks(source, tensor, chunk_bytes):
                    raw = _read(output, blocks.shape[0] * 32)
                    native = np.frombuffer(raw, dtype="u1").reshape(-1, 32)
                    restored = np.bitwise_xor(native, np.uint8(128))
                    mismatches += int(np.count_nonzero(restored != blocks[:, 2:]))
                    count += restored.size
        for op in operations:
            if op.kind not in {"gguf_q8_scales", "gguf_q8_scales_transpose"}:
                continue
            native_info = tensors[op.name]
            groups = op.sources[0].shape[1] // 32
            if op.kind == "gguf_q8_scales":
                output.seek(native_info.offset)
                for tensor in op.sources:
                    for blocks in _q8_blocks(source, tensor, chunk_bytes):
                        values = np.ascontiguousarray(blocks[:, :2]).view("<f2").reshape(-1)
                        expected = _cast_values(values, "F16", "BF16").view("<u2")
                        native_bits = np.frombuffer(_read(output, expected.size * 2), dtype="<u2")
                        scales_mismatches += int(np.count_nonzero(native_bits != expected))
                        scales_count += expected.size
                continue
            # Read bounded complete source rows, then independently gather each
            # native group column by file offset. No writer transpose routine is
            # reused, so a systematic fusion/axis error cannot pass this check.
            padded_rows = op.shape[1]
            height_limit = max(1, chunk_bytes // (groups * 512))
            fused_row = 0
            for tensor in op.sources:
                with source.path.open("rb") as stream:
                    if _signature(os.fstat(stream.fileno())) != source.signature:
                        raise ConversionError("GGUF source changed")
                    for row in range(0, tensor.shape[0], height_limit):
                        height = min(height_limit, tensor.shape[0] - row)
                        stream.seek(tensor.offset + row * groups * 34)
                        raw = _read(stream, height * groups * 34)
                        blocks = np.frombuffer(raw, dtype="u1").reshape(height, groups, 34)
                        values = np.ascontiguousarray(blocks[:, :, :2]).view("<f2").reshape(-1)
                        expected = _cast_values(values, "F16", "BF16").view("<u2").reshape(height, groups)
                        for group in range(groups):
                            output.seek(native_info.offset + (group * padded_rows + fused_row + row) * 2)
                            native_bits = np.frombuffer(_read(output, height * 2), dtype="<u2")
                            scales_mismatches += int(np.count_nonzero(native_bits != expected[:, group]))
                        scales_count += height * groups
                        del raw, blocks, values, expected
                fused_row += tensor.shape[0]
            for group in range(groups):
                output.seek(native_info.offset + (group * padded_rows + fused_row) * 2)
                padding = np.frombuffer(_read(output, (padded_rows - fused_row) * 2), dtype="<u2")
                scales_mismatches += int(np.count_nonzero(padding))
                padding_count += padding.size
    if mismatches:
        raise ConversionError("GGUF compressed code preservation failed")
    if scales_mismatches:
        raise ConversionError("GGUF coefficient layout preservation failed")
    return {
        "compared_codes": count,
        "mismatched_codes": mismatches,
        "compared_scales": scales_count,
        "mismatched_scale_bits": scales_mismatches,
        "compared_padding_scales": padding_count,
    }


def import_qwen3(
    source_path: Path,
    hf_source: Path,
    output: Path,
    *,
    repo: str = "Qwen/Qwen3-0.6B",
    source_revision: str,
    hf_revision: str,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict:
    """Stage, fully reverify and atomically publish one private experiment tuple."""
    source_path, hf_source, output = source_path.resolve(), hf_source.resolve(), output.resolve()
    if not get_model(repo) or get_model(repo)["adapter"] != "qwen3" or get_model(repo).get("compressed_format"):
        raise UnsupportedConversion("Original registered dense Qwen3 metadata identity is required")
    if any(
        len(value) != 40 or any(character not in "0123456789abcdef" for character in value)
        for value in (source_revision, hf_revision)
    ):
        raise ConversionError("Immutable source revisions are required")
    if (
        output.exists()
        or source_path == output
        or hf_source == output
        or output in source_path.parents
        or output in hf_source.parents
    ):
        raise ConversionError("GGUF output exists or overlaps input")
    start = time.perf_counter()
    source = read_gguf(source_path)
    input_files = [source_path] + [
        hf_source / filename
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
    ]
    hashes = {str(path): file_hash(path) for path in input_files}
    tooling = toolchain_provenance()
    plan, operations, metadata, config = build_import_plan(source, hf_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gguf-staging-", dir=output.parent))
    try:
        stats = write_import(source, operations, metadata, staging / "model.safetensors", chunk_bytes=chunk_bytes)
        stats["code_preservation"] = validate_codes(source, operations, staging / "model.safetensors", chunk_bytes)
        normalized = write_metadata(hf_source, staging, {**config, "_stark_model_repo": repo})
        (staging / "config.json").write_text(
            json.dumps({"type": "LanguageModelConfig", "decoder_config": plan.decoder_config, **normalized}, indent=4)
        )
        if (
            any(file_hash(path) != hashes[str(path)] for path in input_files)
            or _signature(source_path.stat()) != source.signature
        ):
            raise ConversionError("GGUF or original metadata changed during conversion")
        if tooling != toolchain_provenance():
            raise ConversionError("GGUF conversion tooling changed")
        receipt = {
            "schema_version": 1,
            "experiment": "qwen3_gguf_q8_0",
            "repo": repo,
            "source_revision": source_revision,
            "hf_metadata_revision": hf_revision,
            "input_hashes": hashes,
            "toolchain": tooling,
            "ggml_format_reference_revision": GGML_REFERENCE_REVISION,
            "stats": stats,
            "native_loading": "unverified",
            "native_execution": "unverified",
            "compression": {
                "codes": "Q8_0 signed codes xor 0x80; no requantization",
                "scales": "F16 to BF16",
                "group_size": 32,
            },
            "outputs": {
                filename: file_hash(staging / filename)
                for filename in ("config.json", "tokenizer.json", "model.safetensors")
            },
            "total_seconds": time.perf_counter() - start,
        }
        (staging / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for path in staging.iterdir():
            path.chmod(0o600)
        _publish(staging, output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--hf-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    try:
        receipt = import_qwen3(
            args.source,
            args.hf_source,
            args.output,
            repo=args.repo,
            source_revision=args.source_revision,
            hf_revision=args.hf_revision,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
        )
    except Exception:
        print(json.dumps({"status": "failed", "reason": "unsupported_or_invalid_gguf_experiment"}))
        return 1
    print(
        json.dumps(
            {
                "status": "converted",
                "native_loading": "unverified",
                **receipt["stats"],
                "total_seconds": receipt["total_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
