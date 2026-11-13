"""Independent binary64 synthetic oracle for the pinned Edge0 MTP head.

No checkpoint, native implementation, GPU, NumPy, or activation quantizer is
used. Source equations are traced to Edge0 revision SOURCE_REVISION below;
Strata reference provenance and MIT notice remain in NOTICE.edge0-reference
and LICENSE.edge0-reference.
Generated synthetic payloads are private, ignored runtime artifacts. The CLI
prints hashes and counts only. Each new output directory is registered before
the oracle evaluates it; an existing directory is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "b7900cbb322926ccebdcd0c0c5a1280497ecff22"
SOURCE_ROOT = REPO / "runtime/edge0-mtp-2026-11-13/source"
SCHEMA = "edge0-mtp-synthetic-v1"
SEED = 0xED6E0501
GEOMETRY = {
    "hidden": 64,
    "intermediate": 64,
    "streams": 4,
    "low_rank": 8,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 16,
    "rotary": 8,
    "experts": 4,
    "active": 2,
    "index_heads": 4,
    "index_dim": 8,
    "vocabulary": 128,
    "context": 16,
    "cells": 4,
}
POLICY = {
    "arithmetic": "binary64 scalar products with math.fsum",
    "candidate": "F32 activations, accumulation, cache and output",
    "hidden_norm": "global streams*hidden RMS; stored final gamma",
    "hc_norm": "per-stream RMS; stored final gamma; F64 activations",
    "weights": "unchanged Q8_0/F32/BF16 dense and four-plane Q2_0 experts",
    "activation_quantization": "none",
    "rope": "half-split, unscaled base 10000, cell position",
    "attention": "dense causal, own unquantized K/V, contiguous grouped-query heads",
    "router": "source generic ordering and gather/clamp/renormalize promoted to F64, no source F32 probability rounding",
    "readout": "own final HC mixer then complete main Q8_0 head; no extra norm",
}
GATES = {
    "max_absolute_error": 3e-5,
    "max_scaled_error": 3e-5,
    "comparison": "abs(actual-reference) <= max_absolute_error + max_scaled_error*max(abs(reference)) per stage",
    "routed_ids": "exact",
    "finite": "all decoded inputs and expected stages",
    "final_max_absolute_error": 0.002,
    "final_relative_l2_error": 0.0002,
    "isolated_q2_dot_error": "2e-5*max(sum(abs(products)),1)",
    "router_minimum_rank_margin": 0.0001,
    "scope": "synthetic arithmetic only; no actual artifact, sampling, performance or lifecycle qualification",
}
CASES = {
    "serial": {
        "front_only_cells": [],
        "full_cells": [0, 1, 2, 3],
        "residual": "explicit target rows",
    },
    "catchup": {
        "front_only_cells": [0, 1],
        "full_cells": [2, 3],
        "residual": "explicit target rows",
    },
    "chain": {
        "front_only_cells": [],
        "full_cells": [0, 1, 2, 3],
        "residual": "explicit target rows 0..2; cell 3 uses cell 2 final_residual",
    },
}
SOURCE_FILES = (
    "edge0-strata/tools/mtp_rt.py",
    "edge0-strata/tools/mtp_pack.py",
    "edge0-strata/include/strata/core/mtp.hpp",
    "edge0-strata/src/core/mtp.cpp",
    "edge0-strata/include/strata/kernels/gr.hpp",
    "edge0-strata/include/strata/kernels/fused_gr.hpp",
    "edge0-strata/include/strata/kernels/cpu/expert.hpp",
    "edge0-strata/src/kernels/cpu/expert.cpp",
    "edge0-strata/src/kernels/cuda/fused_gr.cu",
    "edge0-strata/src/kernels/cuda/native_qsa.cu",
    "edge0-strata/src/kernels/cuda/native_rope.cu",
    "edge0-strata/src/kernels/cuda/qsa_decode_attn.cu",
    "edge0-strata/src/kernels/cuda/router_top10.cu",
    "edge0-strata/src/kernels/cuda/native_router.cu",
    "edge0-strata/src/kernels/cuda/shared_expert.cu",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def silu(x):
    return x * sigmoid(x)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def rms(vector, gamma, eps=1e-6):
    scale = 1.0 / math.sqrt(dot(vector, vector) / len(vector) + eps)
    return [x * scale * g for x, g in zip(vector, gamma, strict=True)]


def rows(vector, width):
    if len(vector) % width:
        raise ValueError("non-integral row geometry")
    return [vector[i : i + width] for i in range(0, len(vector), width)]


def flatten(matrix):
    return [x for row in matrix for x in row]


def decode_dense(raw, kind, nrows, ncols):
    """Scalar unpacking, independent of the logical-code fixture encoder."""
    if kind == "q8_0":
        if ncols % 32 or len(raw) != nrows * (ncols // 32) * 34:
            raise ValueError("bad Q8_0 geometry")
        out = []
        for start in range(0, len(raw), 34):
            scale = struct.unpack_from("<e", raw, start)[0]
            out.extend(scale * q for q in struct.unpack_from("<32b", raw, start + 2))
    elif kind == "bf16":
        if len(raw) != nrows * ncols * 2:
            raise ValueError("bad BF16 geometry")
        out = [
            struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            for (bits,) in struct.iter_unpack("<H", raw)
        ]
    elif kind == "f32":
        if len(raw) != nrows * ncols * 4:
            raise ValueError("bad F32 geometry")
        out = [x for (x,) in struct.iter_unpack("<f", raw)]
    else:
        raise ValueError("unsupported dense encoding")
    if not all(math.isfinite(x) for x in out):
        raise ValueError("nonfinite dense value")
    return rows(out, ncols)


def encode_q2_matrix(codes, scales):
    """Logical adjacent 2-bit codes -> separate code and F16 scale planes."""
    packed = bytearray()
    for row in codes:
        if len(row) % 64 or any(c not in range(4) for c in row):
            raise ValueError("invalid logical Q2 codes")
        for i in range(0, len(row), 4):
            packed.append(sum(row[i + j] << (2 * j) for j in range(4)))
    flat_scales = flatten(scales)
    if len(flat_scales) != sum(len(row) // 64 for row in codes):
        raise ValueError("invalid logical Q2 scales")
    return bytes(packed), struct.pack("<" + "e" * len(flat_scales), *flat_scales)


def decode_expert(blob, hidden, intermediate, mutation=None):
    """Decode physical [GU codes][D codes][GU scales][D scales] planes."""
    if hidden % 64 or intermediate % 64:
        raise ValueError("invalid expert geometry")
    gu_codes = 2 * intermediate * hidden // 4
    down_codes = hidden * intermediate // 4
    gu_scales = 2 * intermediate * (hidden // 64) * 2
    if len(blob) != 3 * hidden * intermediate * 18 // 64:
        raise ValueError("invalid expert blob bytes")
    down_scale_start = gu_codes + down_codes + gu_scales

    def decode(start, scale_start, count, width):
        result = []
        for r in range(count):
            row = []
            for c in range(width):
                code = (blob[start + r * (width // 4) + c // 4] >> (2 * (c % 4))) & 3
                scale = struct.unpack_from(
                    "<e", blob, scale_start + 2 * (r * (width // 64) + c // 64)
                )[0]
                row.append((code - 1) * scale)
            result.append(row)
        return result

    scale_start = gu_codes + down_codes
    if mutation == "scale_plane":
        scale_start = 0
    gu = decode(0, scale_start, 2 * intermediate, hidden)
    down = decode(gu_codes, down_scale_start, hidden, intermediate)
    gate, up = gu[::2], gu[1::2]
    if mutation == "row_interleave":
        gate, up = gu[:intermediate], gu[intermediate:]
    if any(
        not math.isfinite(x)
        for matrix in (gate, up, down)
        for row in matrix
        for x in row
    ):
        raise ValueError("nonfinite expert value")
    return gate, up, down


def hc_read(residual, norm, down, up, inject, hidden, streams):
    """fused_gr.cu gr_down_kernel/gr_up_kernel: grouped norm, not wide RMS."""
    normalized = flatten(
        [
            rms(r, g)
            for r, g in zip(rows(residual, hidden), rows(norm, hidden), strict=True)
        ]
    )
    low = [silu(x / streams) for x in matvec(down, normalized)]
    gates = [sigmoid(x) for x in matvec(up, low)]
    mixed = [
        math.fsum(
            normalized[c * hidden + d] * gates[c * hidden + d] for c in range(streams)
        )
        / streams
        for d in range(hidden)
    ]
    injection = matvec(inject, normalized) if inject is not None else []
    return normalized, low, injection, mixed


def hc_write(residual, block, injection, hidden, streams):
    return [
        residual[c * hidden + d] + block[d] * 2 * sigmoid(injection[c] / streams)
        for c in range(streams)
        for d in range(hidden)
    ]


def rope(vector, position, rotary, base=10000.0):
    out = vector.copy()
    for i in range(rotary // 2):
        angle = position * base ** (-2.0 * i / rotary)
        a, b = vector[i], vector[i + rotary // 2]
        out[i] = a * math.cos(angle) - b * math.sin(angle)
        out[i + rotary // 2] = a * math.sin(angle) + b * math.cos(angle)
    return out


def route(logits, active):
    maximum = max(logits)
    ex = [math.exp(x - maximum) for x in logits]
    total = math.fsum(ex)
    probabilities = [x / total for x in ex]
    ids = sorted(range(len(logits)), key=lambda i: (-probabilities[i], i))[:active]
    divisor = max(math.fsum(probabilities[i] for i in ids), 2.0**-14)
    return ids, [probabilities[i] / divisor for i in ids]


def dense_specs(g):
    h, ff, s, lr = (g[k] for k in ("hidden", "intermediate", "streams", "low_rank"))
    nh, nk, hd = (g[k] for k in ("heads", "kv_heads", "head_dim"))
    specs = [("fc_embedding.weight", "q8_0", h, h), ("fc_hidden.weight", "q8_0", h, h)]
    for prefix, injecting in (
        ("hyper_connection_mixer", False),
        ("attn_hyper_connection", True),
        ("mlp_hyper_connection", True),
    ):
        if injecting:
            specs.append((prefix + ".block_inject_weight.weight", "bf16", s, s * h))
        specs.extend(
            [
                (prefix + ".hc_norm.weight", "f32", 1, s * h),
                (prefix + ".input_mix_weight_down.weight", "bf16", lr, s * h),
                (prefix + ".input_mix_weight_up.weight", "bf16", s * h, lr),
            ]
        )
    specs.extend(
        [
            ("mlp.gate.weight", "bf16", g["experts"], h),
            ("mlp.shared_expert.down_proj.weight", "q8_0", h, ff),
            ("mlp.shared_expert.gate_proj.weight", "q8_0", ff, h),
            ("mlp.shared_expert.up_proj.weight", "q8_0", ff, h),
            ("mlp.shared_expert_gate.weight", "bf16", 1, h),
            (
                "self_attn.indexer.index_qk_proj.weight",
                "q8_0",
                (g["index_heads"] + 1) * g["index_dim"],
                h,
            ),
            ("self_attn.indexer.k_layernorm.weight", "f32", 1, g["index_dim"]),
            ("self_attn.indexer.q_layernorm.weight", "f32", 1, g["index_dim"]),
            ("self_attn.k_norm.weight", "f32", 1, hd),
            ("self_attn.k_proj.weight", "q8_0", nk * hd, h),
            ("self_attn.o_proj.weight", "q8_0", h, nh * hd),
            ("self_attn.q_norm.weight", "f32", 1, hd),
            ("self_attn.q_proj.weight", "q8_0", 2 * nh * hd, h),
            ("self_attn.v_proj.weight", "q8_0", nk * hd, h),
            ("pre_fc_norm_embedding.weight", "f32", 1, h),
            ("pre_fc_norm_hidden.weight", "f32", 1, s * h),
        ]
    )
    return specs


def expected_layout(g=None):
    """Register fixed stage names, widths, dtypes and cell indices before evaluation."""
    g = GEOMETRY if g is None else g
    h, ff, s, lr, nh, nk, hd, k = (
        g[key]
        for key in (
            "hidden",
            "intermediate",
            "streams",
            "low_rank",
            "heads",
            "kv_heads",
            "head_dim",
            "active",
        )
    )
    widths = {
        "embedding_norm": h,
        "hidden_norm": s * h,
        "embedding_projected": h,
        "hidden_projected": s * h,
        "input_residual": s * h,
        "query_full": 2 * nh * hd,
        "query": nh * hd,
        "key": nk * hd,
        "value": nk * hd,
        "attention_scores": nh * g["cells"],
        "attention_probabilities": nh * g["cells"],
        "attention": nh * hd,
        "gated_attention": nh * hd,
        "attention_output": h,
        "post_attention": s * h,
        "router_logits": g["experts"],
        "router_weights": k,
        "routed_gate": k * ff,
        "routed_up": k * ff,
        "routed_activation": k * ff,
        "routed_parts": k * h,
        "routed_gate_sumabs": k * ff,
        "routed_up_sumabs": k * ff,
        "routed_down_sumabs": k * h,
        "shared_gate": ff,
        "shared_up": ff,
        "shared_activation": ff,
        "shared_scalar": 1,
        "shared_output": h,
        "moe_output": h,
        "final_residual": s * h,
        "readout": g["vocabulary"],
        "routed_ids": k,
    }
    for prefix in ("attn_hc", "mlp_hc", "final_hc"):
        widths.update(
            {prefix + "_normalized": s * h, prefix + "_low": lr, prefix + "_mixed": h}
        )
        if prefix != "final_hc":
            widths[prefix + "_inject"] = s
    front_names = {
        "embedding_norm",
        "hidden_norm",
        "embedding_projected",
        "hidden_projected",
        "input_residual",
        "attn_hc_normalized",
        "attn_hc_low",
        "attn_hc_inject",
        "attn_hc_mixed",
        "key",
        "value",
    }
    result = []
    for case, policy in CASES.items():
        for name, width in widths.items():
            indices = (
                list(range(g["cells"])) if name in front_names else policy["full_cells"]
            )
            result.append(
                {
                    "name": case + "/" + name,
                    "dtype": "u32" if name == "routed_ids" else "f64",
                    "shape": [len(indices), width],
                    "cell_indices": indices,
                }
            )
    return result


def make_payload(g=None):
    """Make synthetic raw bytes only; no head evaluation or writes."""
    g = dict(GEOMETRY if g is None else g)
    rng = random.Random(SEED)

    def matrix_bytes(kind, nr, nc):
        if kind == "q8_0":
            raw = bytearray()
            for _ in range(nr * nc // 32):
                # The independently generated logical signed codes are never quantized.
                raw.extend(struct.pack("<e", rng.choice((1, 2, 3, 4)) / 1024))
                raw.extend(
                    struct.pack("<32b", *(rng.randrange(-24, 25) for _ in range(32)))
                )
            return bytes(raw)
        if kind == "bf16":
            # Dyadic values are exactly representable in BF16, including signed values.
            vals = [rng.randrange(-24, 25) / 512 for _ in range(nr * nc)]
            bits = [struct.unpack("<I", struct.pack("<f", x))[0] >> 16 for x in vals]
            return struct.pack("<" + "H" * len(bits), *bits)
        vals = [1 + rng.randrange(-32, 33) / 256 for _ in range(nr * nc)]
        return struct.pack("<" + "f" * len(vals), *vals)

    dense, records, lines = bytearray(), [], []
    for name, kind, nr, nc in dense_specs(g):
        dense.extend(b"\0" * (-len(dense) % 256))
        raw = matrix_bytes(kind, nr, nc)
        if name == "mlp.gate.weight":
            # Predesigned separated rows: ranking depends on a fixed nonzero
            # mixed coordinate, avoiding incidental random near-ties. The
            # source router still receives a real BF16 matrix dot product.
            vals = [
                ((2 * r - nr + 1) / 2 if c == 0 else 0.0)
                for r in range(nr)
                for c in range(nc)
            ]
            raw = struct.pack(
                "<" + "H" * len(vals),
                *(struct.unpack("<I", struct.pack("<f", x))[0] >> 16 for x in vals),
            )
        record = {
            "name": name,
            "file": "dense.bin",
            "dtype": kind,
            "shape": [nr, nc],
            "offset": len(dense),
            "bytes": len(raw),
            "sha256": sha256(raw),
        }
        records.append(record)
        lines.append(f"{name} {kind} {nr} {nc} {len(dense)} {len(raw)}")
        dense.extend(raw)
    experts = bytearray()
    h, ff = g["hidden"], g["intermediate"]
    for _ in range(g["experts"]):
        gate = [[rng.randrange(4) for _ in range(h)] for _ in range(ff)]
        up = [[rng.randrange(4) for _ in range(h)] for _ in range(ff)]
        gu = [row for pair in zip(gate, up, strict=True) for row in pair]
        down = [[rng.randrange(4) for _ in range(ff)] for _ in range(h)]
        gc, gs = encode_q2_matrix(
            gu, [[rng.choice((1, 2, 3, 4)) / 128 for _ in range(h // 64)] for _ in gu]
        )
        dc, ds = encode_q2_matrix(
            down,
            [[rng.choice((1, 2, 3, 4)) / 128 for _ in range(ff // 64)] for _ in down],
        )
        experts.extend(gc + dc + gs + ds)
    inputs, input_records = bytearray(), []
    for name, kind, shape, raw in (
        (
            "target_residual",
            "f32",
            [g["cells"], g["streams"], h],
            struct.pack(
                "<" + "f" * (g["cells"] * g["streams"] * h),
                *(
                    f32((rng.random() * 2 - 1) * (0.3, 0.8, 1.6, 2.7)[c])
                    for _ in range(g["cells"])
                    for c in range(g["streams"])
                    for _ in range(h)
                ),
            ),
        ),
        (
            "token_embedding",
            "q8_0",
            [g["vocabulary"], h],
            matrix_bytes("q8_0", g["vocabulary"], h),
        ),
        (
            "main_head",
            "q8_0",
            [g["vocabulary"], h],
            matrix_bytes("q8_0", g["vocabulary"], h),
        ),
        ("next_token_ids", "u32", [g["cells"]], struct.pack("<4I", 7, 19, 71, 106)),
    ):
        input_records.append(
            {
                "name": name,
                "file": "inputs.bin",
                "dtype": kind,
                "shape": shape,
                "offset": len(inputs),
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
        inputs.extend(raw)
    payload = {
        "dense.bin": bytes(dense),
        "dense.txt": ("\n".join(lines) + "\n").encode(),
        "experts.bin": bytes(experts),
        "inputs.bin": bytes(inputs),
    }
    return payload, records + input_records


def evaluate(payload, records, g=None, mutation=None, full_from=0, chain=False):
    """Evaluate forced synthetic cells with independent scalar F64 equations."""
    g = dict(GEOMETRY if g is None else g)
    w, inp = {}, {}
    for rec in records:
        raw = payload[rec["file"]][rec["offset"] : rec["offset"] + rec["bytes"]]
        if rec["dtype"] == "u32":
            inp[rec["name"]] = [x for (x,) in struct.iter_unpack("<I", raw)]
        else:
            nr, nc = math.prod(rec["shape"][:-1]), rec["shape"][-1]
            matrix = decode_dense(raw, rec["dtype"], nr, nc)
            (w if rec["file"] == "dense.bin" else inp)[rec["name"]] = matrix
    h, ff, s, hd, nh, nk, cells = (
        g[k]
        for k in (
            "hidden",
            "intermediate",
            "streams",
            "head_dim",
            "heads",
            "kv_heads",
            "cells",
        )
    )
    if mutation == "gamma_plus_one":
        w = {
            name: [[x + 1 for x in row] for row in val] if "norm" in name else val
            for name, val in w.items()
        }
    blob_size = 3 * h * ff * 18 // 64
    experts = [
        decode_expert(
            payload["experts.bin"][i * blob_size : (i + 1) * blob_size], h, ff, mutation
        )
        for i in range(g["experts"])
    ]
    cache_k, cache_v, result, route_ids, previous_wide = [], [], {}, [], None

    def save(name, values):
        result.setdefault(name, []).append(values.copy())

    def hc(prefix, residual, label):
        normalized, low, inject, mixed = hc_read(
            residual,
            w[prefix + ".hc_norm.weight"][0],
            w[prefix + ".input_mix_weight_down.weight"],
            w[prefix + ".input_mix_weight_up.weight"],
            w.get(prefix + ".block_inject_weight.weight"),
            h,
            s,
        )
        for suffix, values in (
            ("normalized", normalized),
            ("low", low),
            ("inject", inject),
            ("mixed", mixed),
        ):
            if values:
                save(label + "_" + suffix, values)
        return inject, mixed

    for cell in range(cells):
        embedding = inp["token_embedding"][inp["next_token_ids"][cell]]
        en = rms(embedding, w["pre_fc_norm_embedding.weight"][0])
        residual = flatten(inp["target_residual"][cell * s : (cell + 1) * s])
        if chain and cell == 3:
            if previous_wide is None:
                raise ValueError("chain requires previous full cell")
            residual = previous_wide.copy()
        gamma = w["pre_fc_norm_hidden.weight"][0]
        hn = rms(residual, gamma)
        if mutation == "stream_hidden_norm":
            hn = flatten(
                [
                    rms(r, gn)
                    for r, gn in zip(rows(residual, h), rows(gamma, h), strict=True)
                ]
            )
        ep = matvec(w["fc_embedding.weight"], en)
        hp = flatten([matvec(w["fc_hidden.weight"], row) for row in rows(hn, h)])
        residual = [x + ep[i % h] for i, x in enumerate(hp)]
        for name, values in (
            ("embedding_norm", en),
            ("hidden_norm", hn),
            ("embedding_projected", ep),
            ("hidden_projected", hp),
            ("input_residual", residual),
        ):
            save(name, values)
        inject, mixed = hc("attn_hyper_connection", residual, "attn_hc")
        pos = cell + (mutation == "rope_position")
        key = [
            rope(rms(row, w["self_attn.k_norm.weight"][0]), pos, g["rotary"])
            for row in rows(matvec(w["self_attn.k_proj.weight"], mixed), hd)
        ]
        value = rows(matvec(w["self_attn.v_proj.weight"], mixed), hd)
        cache_k.append(key)
        cache_v.append(value)
        if cell < full_from:
            # Prompt catch-up stores only the same independently computed K/V.
            # It never evaluates attention output, MLP, final mixer or readout.
            save("key", flatten(key))
            save("value", flatten(value))
            continue
        qfull = matvec(w["self_attn.q_proj.weight"], mixed)
        qheads = rows(qfull, 2 * hd)
        q = [
            rope(rms(row[:hd], w["self_attn.q_norm.weight"][0]), pos, g["rotary"])
            for row in qheads
        ]
        attention, score_rows, prob_rows = [], [], []
        for head in range(nh):
            kv = head // (nh // nk)
            scores = [
                dot(q[head], cache_k[t][kv]) / math.sqrt(hd) for t in range(cell + 1)
            ]
            ex = [math.exp(x - max(scores)) for x in scores]
            probs = [x / math.fsum(ex) for x in ex]
            attention.extend(
                math.fsum(probs[t] * cache_v[t][kv][d] for t in range(cell + 1))
                for d in range(hd)
            )
            score_rows.append(scores + [0.0] * (cells - cell - 1))
            prob_rows.append(probs + [0.0] * (cells - cell - 1))
        gated = [
            x * sigmoid(qheads[head][hd + d])
            for head in range(nh)
            for d, x in enumerate(attention[head * hd : (head + 1) * hd])
        ]
        output = matvec(w["self_attn.o_proj.weight"], gated)
        for name, values in (
            ("query_full", qfull),
            ("query", flatten(q)),
            ("key", flatten(key)),
            ("value", flatten(value)),
            ("attention_scores", flatten(score_rows)),
            ("attention_probabilities", flatten(prob_rows)),
            ("attention", attention),
            ("gated_attention", gated),
            ("attention_output", output),
        ):
            save(name, values)
        residual = hc_write(residual, output, inject, h, s)
        save("post_attention", residual)
        inject, mixed = hc("mlp_hyper_connection", residual, "mlp_hc")
        logits = matvec(w["mlp.gate.weight"], mixed)
        ids, weights = route(logits, g["active"])
        ordered = sorted(logits, reverse=True)
        if (
            min(a - b for a, b in itertools.pairwise(ordered))
            < GATES["router_minimum_rank_margin"]
        ):
            raise ValueError("synthetic router rank margin below registered gate")
        route_ids.append(ids)
        parts, routed_gate, routed_up, routed_activation = [], [], [], []
        gate_sumabs, up_sumabs, down_sumabs = [], [], []
        for expert in ids:
            gate, up, down = experts[expert]
            gv, uv = matvec(gate, mixed), matvec(up, mixed)
            activation = [silu(a) * b for a, b in zip(gv, uv, strict=True)]
            parts.append(matvec(down, activation))
            gate_sumabs.extend(
                math.fsum(abs(a * b) for a, b in zip(row, mixed, strict=True))
                for row in gate
            )
            up_sumabs.extend(
                math.fsum(abs(a * b) for a, b in zip(row, mixed, strict=True))
                for row in up
            )
            down_sumabs.extend(
                math.fsum(abs(a * b) for a, b in zip(row, activation, strict=True))
                for row in down
            )
            routed_gate.extend(gv)
            routed_up.extend(uv)
            routed_activation.extend(activation)
        sg = matvec(w["mlp.shared_expert.gate_proj.weight"], mixed)
        su = matvec(w["mlp.shared_expert.up_proj.weight"], mixed)
        sa = [silu(a) * b for a, b in zip(sg, su, strict=True)]
        scalar = sigmoid(dot(w["mlp.shared_expert_gate.weight"][0], mixed))
        shared = [
            x * scalar for x in matvec(w["mlp.shared_expert.down_proj.weight"], sa)
        ]
        moe = [
            shared[d] + math.fsum(weights[k] * parts[k][d] for k in range(g["active"]))
            for d in range(h)
        ]
        for name, values in (
            ("router_logits", logits),
            ("router_weights", weights),
            ("routed_gate", routed_gate),
            ("routed_up", routed_up),
            ("routed_activation", routed_activation),
            ("routed_parts", flatten(parts)),
            ("routed_gate_sumabs", gate_sumabs),
            ("routed_up_sumabs", up_sumabs),
            ("routed_down_sumabs", down_sumabs),
            ("shared_gate", sg),
            ("shared_up", su),
            ("shared_activation", sa),
            ("shared_scalar", [scalar]),
            ("shared_output", shared),
            ("moe_output", moe),
        ):
            save(name, values)
        residual = hc_write(residual, moe, inject, h, s)
        previous_wide = residual.copy()
        save("final_residual", residual)
        _, mixed = hc("hyper_connection_mixer", residual, "final_hc")
        save("readout", matvec(inp["main_head"], mixed))
    if any(
        not math.isfinite(x) for stage in result.values() for row in stage for x in row
    ):
        raise ValueError("nonfinite expected stage")
    return result, route_ids


def source_bindings():
    tree = json.loads((Path("/tmp/edge0-intake/github-tree.json")).read_text())
    if tree.get("sha") != SOURCE_REVISION:
        raise ValueError("source tree revision mismatch")
    entries = {entry["path"]: entry for entry in tree["tree"]}
    result = []
    for name in SOURCE_FILES:
        data = (SOURCE_ROOT / name).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if entries[name]["sha"] != blob:
            raise ValueError("source Git blob mismatch")
        result.append(
            {"path": name, "git_blob": blob, "bytes": len(data), "sha256": sha256(data)}
        )
    return result


def generate(out):
    runtime = (REPO / "runtime").resolve()
    out = out.resolve()
    if not out.is_relative_to(runtime) or out == runtime or out.exists():
        raise ValueError("output must be a new private runtime subdirectory")
    payload, records = make_payload()
    registration = {
        "schema": SCHEMA,
        "source_revision": SOURCE_REVISION,
        "sources": source_bindings(),
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "seed": SEED,
        "tests_sha256": sha256(
            Path(__file__).with_name("test_edge0_mtp_fixture.py").read_bytes()
        ),
        "geometry": GEOMETRY,
        "policy": POLICY,
        "gates": GATES,
        "inputs": records,
        "cases": CASES,
        "expected_layout": expected_layout(),
        "files": {
            name: {"bytes": len(raw), "sha256": sha256(raw)}
            for name, raw in payload.items()
        },
    }
    out.mkdir(parents=True, mode=0o700)
    for name, raw in payload.items():
        (out / name).write_bytes(raw)
        (out / name).chmod(0o600)
    registration_raw = canonical(registration)
    (out / "registration.json").write_bytes(registration_raw)
    (out / "registration.json").chmod(0o600)
    # Evaluation occurs only after the immutable raw bytes and gates are on disk.
    cases = {}
    cases["serial"] = evaluate(payload, records)
    cases["catchup"] = evaluate(payload, records, full_from=2)
    cases["chain"] = evaluate(payload, records, chain=True)
    # The registered catch-up must reproduce the two full-cell suffix outputs.
    for name in cases["catchup"][0]:
        actual = cases["catchup"][0][name]
        reference = cases["serial"][0][name]
        if actual != (reference if len(actual) == GEOMETRY["cells"] else reference[2:]):
            raise ValueError("catch-up equation mismatch")
    if cases["catchup"][1] != cases["serial"][1][2:]:
        raise ValueError("catch-up routing mismatch")
    raw, expected_records = bytearray(), []
    for case, (expected, ids) in cases.items():
        for name, stage in [*expected.items(), ("routed_ids", ids)]:
            values = flatten(stage)
            dtype, fmt = ("u32", "I") if name == "routed_ids" else ("f64", "d")
            data = struct.pack("<" + fmt * len(values), *values)
            indices = list(range(GEOMETRY["cells"] - len(stage), GEOMETRY["cells"]))
            expected_records.append(
                {
                    "name": case + "/" + name,
                    "file": "expected.bin",
                    "dtype": dtype,
                    "shape": [len(stage), len(stage[0])],
                    "cell_indices": indices,
                    "offset": len(raw),
                    "bytes": len(data),
                    "sha256": sha256(data),
                }
            )
            raw.extend(data)
    actual_layout = {
        r["name"]: {key: r[key] for key in ("name", "dtype", "shape", "cell_indices")}
        for r in expected_records
    }
    if actual_layout != {r["name"]: r for r in registration["expected_layout"]}:
        raise ValueError("expected output disagrees with registered schema")
    (out / "expected.bin").write_bytes(raw)
    (out / "expected.bin").chmod(0o600)
    manifest = dict(
        registration,
        registration_sha256=sha256(registration_raw),
        expected=expected_records,
        expected_file={"bytes": len(raw), "sha256": sha256(raw)},
    )
    (out / "manifest.json").write_bytes(canonical(manifest))
    (out / "manifest.json").chmod(0o600)
    return {
        "schema": SCHEMA,
        "cells": GEOMETRY["cells"],
        "stages": len(expected_records),
        "registration_sha256": sha256(registration_raw),
        "manifest_sha256": sha256((out / "manifest.json").read_bytes()),
        "expected_sha256": sha256(raw),
        "payload_bytes": sum(len(v) for v in payload.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = generate(args.out)
    except (ValueError, OSError, KeyError):
        parser.exit(
            1, "synthetic fixture generation failed; no payload details logged\n"
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
