"""Inkling 2 Mini affine 8-bit name mapping and byte-for-byte pack verification.

The rename rules and the quantized-tensor pattern are transcribed as data from
MLX-Serve ``tests/convert_inkling2.py`` at commit
2fe9be834a601c70ef1a597563026340f0e722fe (https://github.com/ddalcu/mlx-serve,
Apache-2.0), which converts ``thinkingmachines/Inkling-2-Mini`` (Apache-2.0)
into mlx-lm names. Nothing is imported or executed from MLX-Serve or MLX.

Command line (aggregates only; never prints tensor values)::

    python -m stark_convert.inkling_affine8 verify ORIGINAL_DIR PACK_DIR [--start N --count M]
    python -m stark_convert.inkling_affine8 inventory ORIGINAL_DIR PACK_DIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mlx_affine import bf16_bits_to_f32, iter_quantized_bf16
from .streaming import TensorInfo, _open_tensor, load_checkpoint

GROUP_SIZE = 64

# Upstream rule order matters; ``attn_res.attn_queries`` is handled first.
_ATTENTION_QUERY = re.compile(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$")
RENAME_RULES: tuple[tuple[str, str], ...] = (
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.embed\.", "model.language_model.embed_tokens."),
    (r"^model\.llm\.", "model.language_model."),
    (r"\.attn_res\.mlp_queries\.(\d+)$", r".layers.\1.mlp_input.query"),
    (r"\.attn_res\.output_query$", ".final_input.query"),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.gate_attn\.", ".self_attn.gate_proj."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.(k|v|mlp)\.", r".\1_message_gate."),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"sconv\.weight$", "sconv.conv1d.weight"),
    (r"convolution\.weight$", "convolution.conv1d.weight"),
    (r"sconv\.static_w$", "sconv.static_weight"),
    (r"sconv\.weight_proj$", "sconv.weight_proj.weight"),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"^model\.visual\.backbone\.ln_pre\.", "model.vision_tower.pre_layernorm."),
    (r"^model\.visual\.backbone\.", "model.vision_tower."),
    (r"^model\.visual\.proj_in\.", "model.vision_tower.merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "model.vision_tower.merger.linear_2."),
    (r"embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"(vision_tower\.layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(vision_tower\.layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$", r"\1.lambda_\2"),
)
QUANTIZED = re.compile(
    r"^(lm_head|model\.language_model\.(embed_tokens|layers\.\d+\.self_attn\."
    r"(\w+_proj|[kv]_sconv\.weight_proj)))\.weight$"
)
FLOAT32_SUFFIXES = (".mlp.gate.weight", ".forget_gate_proj.weight", ".message_gate_proj.weight")
_EXPERT = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
_CROSS_PASS = ("model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries")


def pack_name(original: str) -> str:
    """mlx-lm name of an original tensor under the upstream rename rules."""
    match = _ATTENTION_QUERY.match(original)
    if match:
        return f"model.language_model.layers.{int(match.group(1)) + 1}.attention_input.query"
    name = original
    for pattern, replacement in RENAME_RULES:
        name = re.sub(pattern, replacement, name)
    return name


def quantized_mapping(original: dict[str, TensorInfo]) -> dict[str, str]:
    """Map original tensor name to pack module prefix for every quantized tensor."""
    result = {}
    for name in original:
        if _EXPERT.match(name) or name in _CROSS_PASS:
            continue
        renamed = pack_name(name)
        if QUANTIZED.match(renamed):
            result[name] = renamed.removesuffix(".weight")
    return dict(sorted(result.items(), key=lambda item: _order(item[1])))


def _order(prefix: str) -> tuple:
    match = re.search(r"\.layers\.(\d+)\.", prefix)
    return (0 if match is None else 1, int(match.group(1)) if match else 0, prefix)


@dataclass
class _PlaneDiff:
    elements: int = 0
    differing: int = 0
    max_abs: float = 0.0

    def add(self, ours: np.ndarray, theirs: np.ndarray, as_bf16: bool) -> None:
        self.elements += ours.size
        mask = ours != theirs
        count = int(np.count_nonzero(mask))
        if not count:
            return
        self.differing += count
        if as_bf16:
            delta = np.abs(bf16_bits_to_f32(ours[mask]) - bf16_bits_to_f32(theirs[mask]))
        else:
            delta = np.abs(ours[mask].astype(np.int32) - theirs[mask].astype(np.int32))
        self.max_abs = max(self.max_abs, float(delta.max()))

    def summary(self) -> dict:
        return {"elements": self.elements, "differing": self.differing, "max_abs_diff": self.max_abs}


class _RowReader:
    """Reads row ranges of a 2-D tensor through a validated file handle."""

    def __init__(self, tensor: TensorInfo):
        self.tensor = tensor
        self.row_bytes = tensor.nbytes // tensor.shape[0]
        self.stream = _open_tensor(tensor)

    def __call__(self, start: int, count: int) -> bytes:
        self.stream.seek(self.tensor.offset + start * self.row_bytes)
        return self.stream.read(count * self.row_bytes)

    def close(self) -> None:
        self.stream.close()


def _expect(tensor: TensorInfo, dtype: str, shape: tuple[int, ...]) -> None:
    if tensor.dtype != dtype or tensor.shape != shape:
        raise ValueError("Pack tensor dtype or shape does not match the affine layout")


def verify_tensor(source: TensorInfo, pack: dict[str, TensorInfo], prefix: str, rows_per_chunk: int) -> dict:
    """Quantize ``source`` and compare codes, scales and biases with the pack."""
    if source.dtype != "BF16" or len(source.shape) != 2:
        raise ValueError("Quantized source tensors are expected to be BF16 matrices")
    rows, columns = source.shape
    groups = columns // GROUP_SIZE
    weight, scales, biases = (pack[f"{prefix}.{part}"] for part in ("weight", "scales", "biases"))
    _expect(weight, "U32", (rows, columns // 4))
    _expect(scales, "BF16", (rows, groups))
    _expect(biases, "BF16", (rows, groups))
    diffs = {"codes": _PlaneDiff(), "scales": _PlaneDiff(), "biases": _PlaneDiff()}
    readers = {"source": _RowReader(source), "codes": _RowReader(weight)}
    readers["scales"], readers["biases"] = _RowReader(scales), _RowReader(biases)
    try:
        for start, quantized in iter_quantized_bf16(readers["source"], source.shape, GROUP_SIZE, rows_per_chunk):
            count = quantized.codes.shape[0]
            theirs_codes = np.frombuffer(readers["codes"](start, count), dtype=np.uint8).reshape(count, columns)
            diffs["codes"].add(quantized.codes, theirs_codes, as_bf16=False)
            for plane in ("scales", "biases"):
                theirs = np.frombuffer(readers[plane](start, count), dtype="<u2").reshape(count, groups)
                diffs[plane].add(getattr(quantized, plane), theirs, as_bf16=True)
    finally:
        for reader in readers.values():
            reader.close()
    planes = {key: value.summary() for key, value in diffs.items()}
    return {
        "tensor": prefix,
        "shape": [rows, columns],
        "byte_identical": all(plane["differing"] == 0 for plane in planes.values()),
        **planes,
    }


def check_mapping(original: dict[str, TensorInfo], pack: dict[str, TensorInfo]) -> dict[str, str]:
    """Confirm the mapped quantized set equals the pack's quantized set."""
    mapping = quantized_mapping(original)
    pack_prefixes = {name.removesuffix(".scales") for name in pack if name.endswith(".scales")}
    pack_quantized = {
        prefix for prefix in pack_prefixes if f"{prefix}.biases" in pack and pack[f"{prefix}.weight"].dtype == "U32"
    }
    # Expert banks carry ``.scales`` too but no ``.biases``; they are not affine.
    if set(mapping.values()) != pack_quantized:
        missing = len(pack_quantized - set(mapping.values()))
        extra = len(set(mapping.values()) - pack_quantized)
        raise ValueError(f"Quantized name mapping mismatch: {missing} pack-only, {extra} mapping-only")
    return mapping


def run_verify(args: argparse.Namespace) -> int:
    original = load_checkpoint(Path(args.original))
    pack = load_checkpoint(Path(args.pack))
    mapping = check_mapping(original, pack)
    items = list(mapping.items())
    selected = items[args.start : args.start + args.count] if args.count else items[args.start :]
    print(json.dumps({"mapping_confirmed": True, "quantized_tensors": len(items), "selected": len(selected)}))
    failures = 0
    output = open(args.output, "a", encoding="utf-8") if args.output else None
    try:
        for name, prefix in selected:
            began = time.monotonic()
            record = {"source": name, **verify_tensor(original[name], pack, prefix, args.rows_per_chunk)}
            record["seconds"] = round(time.monotonic() - began, 2)
            failures += not record["byte_identical"]
            line = json.dumps(record)
            print(line, flush=True)
            if output:
                output.write(line + "\n")
                output.flush()
    finally:
        if output:
            output.close()
    print(json.dumps({"checked": len(selected), "byte_identical": len(selected) - failures, "different": failures}))
    return 1 if failures else 0


def inventory(original: dict[str, TensorInfo], pack: dict[str, TensorInfo]) -> dict:
    """Header-level classification of how every pack tensor relates to the original."""
    categories: dict[str, list[str]] = {}

    def put(category: str, name: str) -> None:
        categories.setdefault(category, []).append(name)

    produced = set()
    mapping = quantized_mapping(original)
    for name, info in original.items():
        if name in mapping:
            for part in ("weight", "scales", "biases"):
                produced.add(f"{mapping[name]}.{part}")
            put("affine8_quantized", name)
            continue
        if name in _CROSS_PASS:
            produced.add("model.language_model.message_input.query")
            put("stacked_into_message_input_query", name)
            continue
        expert = _EXPERT.match(name)
        if expert:
            base = pack_name(expert.group(1)) + (".gate_up_proj" if expert.group(2) == "w13" else ".down_proj")
            suffix = expert.group(3)
            if suffix == "":
                target = f"{base}.weight"
                put("expert_codes_retyped_u8_to_u32" if pack[target].dtype == "U32" else "expert_codes_other", name)
            elif suffix == "_scale":
                target = f"{base}.scales"
                put(f"expert_scales_retyped_{info.dtype}_to_{pack[target].dtype}", name)
            else:
                target = f"{base}.global_scale"
                put("expert_amax_and_denominator_folded_into_global_scale", name)
            produced.add(target)
            continue
        target = pack_name(name)
        if target not in pack:
            put("absent_from_pack", name)
            continue
        produced.add(target)
        theirs = pack[target]
        if target.endswith("conv1d.weight"):
            same = theirs.dtype == info.dtype and theirs.shape == (info.shape[0], info.shape[2], info.shape[1])
            put("conv_swapaxes_1_2" if same else "conv_other", name)
        elif target.endswith(FLOAT32_SUFFIXES):
            put(f"upcast_{info.dtype}_to_{theirs.dtype}", name)
        elif theirs.dtype == info.dtype and theirs.shape == info.shape:
            put("renamed_same_dtype_shape" if target != name else "unchanged_name_dtype_shape", name)
        else:
            put(f"changed_{info.dtype}{list(info.shape)}_to_{theirs.dtype}{list(theirs.shape)}", name)
    unexplained = sorted(set(pack) - produced)
    return {
        "original_tensors": len(original),
        "pack_tensors": len(pack),
        "categories": {key: len(value) for key, value in sorted(categories.items())},
        "pack_tensors_unexplained": unexplained,
    }


def run_inventory(args: argparse.Namespace) -> int:
    result = inventory(load_checkpoint(Path(args.original)), load_checkpoint(Path(args.pack)))
    print(json.dumps(result, indent=2))
    return 0 if not result["pack_tensors_unexplained"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="Quantize original tensors and compare with the pack")
    inv = sub.add_parser("inventory", help="Classify pack tensors against the original headers")
    for command in (verify, inv):
        command.add_argument("original", help="Directory of the original checkpoint")
        command.add_argument("pack", help="Directory of the published MLX 8-bit pack")
    verify.add_argument("--start", type=int, default=0, help="Index of the first quantized tensor to check")
    verify.add_argument("--count", type=int, default=0, help="Number of tensors to check (0 means all)")
    verify.add_argument("--rows-per-chunk", type=int, default=2048)
    verify.add_argument("--output", help="Append per-tensor aggregate JSON lines to this file")
    args = parser.parse_args(argv)
    return run_verify(args) if args.command == "verify" else run_inventory(args)


if __name__ == "__main__":
    sys.exit(main())
