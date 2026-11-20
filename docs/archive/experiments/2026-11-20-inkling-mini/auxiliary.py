"""Time isolated original-weight components on Apple MLX. Only aggregates leave the process.

The inputs, historical KV and expert routes are deterministic synthetic controls.
The output is a component screen. It does not exercise a Stark graph or a full
layer. The projection arithmetic follows the publisher manifest and the Ollama
reference pinned in the milestone record. Use mlx==0.31.2, numpy==2.5.3.
"""

import argparse
import json
import math
import statistics
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


class Weights:
    def __init__(self, path):
        self.path = path
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            self.specs = json.loads(stream.read(size))
        self.base = 8 + size

    def read(self, name, first=None):
        spec = self.specs[name]
        dtype = {"BF16": "<u2", "F32": "<f4", "F8_E4M3": "u1", "U8": "u1"}[spec["dtype"]]
        array = np.memmap(
            self.path,
            mode="r",
            offset=self.base + spec["data_offsets"][0],
            dtype=dtype,
            shape=tuple(spec["shape"]),
        )
        return array if first is None else array[:first]

    def metal(self, name):
        value = mx.array(np.asarray(self.read(name)))
        return value.view(mx.bfloat16) if self.specs[name]["dtype"] == "BF16" else value


def timed(operation, cases, rows):
    times = []
    for index in range(20):
        start = time.perf_counter_ns()
        value = operation(*cases[index % len(cases)])
        mx.eval(value)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if index >= 4:
            times.append(elapsed)
    result = {
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "timed_samples": len(times),
        "warmups": 4,
        "finite": bool(mx.all(mx.isfinite(value)).item()),
    }
    if rows == 1:
        group_times = []
        for index in range(20):
            start = time.perf_counter_ns()
            value = None
            for step in range(8):
                first, *rest = cases[(index + step) % len(cases)]
                if value is not None:
                    first = mx.depends(first, value)
                value = operation(first, *rest)
            mx.eval(value)
            if index >= 4:
                group_times.append((time.perf_counter_ns() - start) / 1e6)
        result["eight_ordered_singletons_one_wait_ms"] = statistics.median(group_times)
    return result


def activation_fp8(x):
    block = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 32)
    maximum = mx.max(mx.abs(block), axis=-1, keepdims=True)
    raw_scale = maximum * (1.0 / 448)
    bits = raw_scale.view(mx.int32)
    exponent = (bits >> 23) + ((bits & ((1 << 23) - 1)) > 0).astype(mx.int32)
    exponent = mx.where(raw_scale > 2.0**-127, mx.minimum(exponent, 254), 0)
    scale = mx.maximum((exponent << 23).view(mx.float32), 2.0**-127)
    inverse = mx.where(exponent == 0, 2.0**127, 1.0 / scale)
    reconstructed = mx.from_fp8(mx.to_fp8(mx.clip(block * inverse, -448, 448)), mx.float32)
    return (reconstructed * scale).reshape(x.shape)


def fp8_table():
    i = np.arange(256, dtype=np.int32)
    exp, mantissa = (i & 127) >> 3, (i & 7)
    mag = np.where(exp == 0, mantissa * 2.0**-9, (1 + mantissa / 8.0) * np.exp2(exp - 7))
    mag[(i & 127) == 127] = np.nan
    return np.where(i & 128, -mag, mag).astype(np.float32)


def numpy_activation(x, table):
    blocks = x.reshape(-1, 32)
    scale = np.exp2(np.ceil(np.log2(np.maximum(np.abs(blocks).max(1) / 448, 2.0**-127))))
    scaled = blocks / scale[:, None]
    # Nearest E4M3 value, ties to even in the encoded mantissa.
    pos = np.abs(scaled)
    hi = np.minimum(np.searchsorted(table[:127], pos), 126)
    lo = np.maximum(0, hi - 1)
    below, above = pos - table[lo], table[hi] - pos
    code = np.where((below < above) | ((below == above) & (lo % 2 == 0)), lo, hi)
    return (np.copysign(table[code], scaled) * scale[:, None]).reshape(x.shape)


def mxfp8(store, rng):
    report = {}
    table = fp8_table()
    for part in ("w13_weight", "w2_weight"):
        name = "model.llm.layers.20.mlp.experts." + part
        raw = np.asarray(store.read(name, 16))
        exponents = np.asarray(store.read(name + "_scale", 16))
        _, outputs, width = raw.shape
        weight = mx.array(raw.view(np.uint32))
        scales = mx.array(exponents)
        mx.eval(weight, scales)

        def project(x, lhs, rhs, weight=weight, scales=scales):
            quantized = activation_fp8(x)
            return mx.gather_qmm(
                quantized,
                weight,
                scales,
                lhs_indices=lhs,
                rhs_indices=rhs,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )

        for rows in (1, 32):
            cases = []
            input_rows = rows * 8 if part == "w2_weight" else rows
            lhs = np.arange(input_rows, dtype=np.uint32)
            if part == "w13_weight":
                lhs = np.repeat(lhs, 8)
            for _ in range(4):
                values = rng.standard_normal((input_rows, 1, width), dtype=np.float32)
                indices = np.stack([rng.choice(16, size=8, replace=False) for _ in range(rows)])
                x = mx.array(values).astype(mx.bfloat16)
                right = mx.array(indices.flatten().astype(np.uint32))
                cases.append((x, mx.array(lhs), right))
                mx.eval(*cases[-1])
            result = timed(project, cases, rows)
            x, lhs, rhs = cases[-1]
            x_cpu = np.array(x.astype(mx.float32))
            reference_x = numpy_activation(x_cpu, table)
            result["activation_unequal"] = int(np.count_nonzero(reference_x != np.array(activation_fp8(x))))
            actual = np.array(project(x, lhs, rhs))
            sampled = np.linspace(0, outputs - 1, 32, dtype=int)
            errors = []
            references = []
            for route, expert in enumerate(np.asarray(rhs)[:8]):
                coefficients = table[raw[expert, sampled]].reshape(32, -1, 32)
                exps = exponents[expert, sampled].astype(np.int32) - 127
                decoded = np.ldexp(coefficients, exps[..., None]).reshape(32, width)
                source = reference_x[int(lhs[route].item()), 0]
                ref = decoded.astype(np.float64) @ source.astype(np.float64)
                references.append(ref)
                errors.append(actual[route, 0, sampled] - ref)
            result["sampled_values"] = int(np.size(errors))
            result["relative_l2"] = float(np.linalg.norm(errors) / np.linalg.norm(references))
            result["max_scaled_error"] = float(np.max(np.abs(errors)) / np.max(np.abs(references)))
            report[f"layer20_{part}_{rows}rows"] = result
    return report


def cross_pass(store, rng):
    q = store.metal("model.llm.middle_cycle_cross_pass_queries")
    kv = store.metal("model.llm.middle_cycle_cross_pass_kv_queries")
    query = mx.stack([q[0], kv[0], kv[1], q[1]]) / math.sqrt(2304)
    stem = "model.llm.layers.6.controllers."
    weights = []
    for key in ("attn", "k", "v", "mlp"):
        part = stem + key + "."
        down = store.metal(part + "down_weight")
        expand = mx.concatenate([store.metal(part + "gain_weight"), store.metal(part + "forget_weight")])
        weights.append((down.T, expand.astype(mx.float32).T))
    mx.eval(query, *(weight for pair in weights for weight in pair))

    def step(x, bank):
        sources = mx.fast.rms_norm(bank.astype(mx.float32), None, 1e-6)
        messages = mx.softmax(query @ sources.swapaxes(-1, -2), axis=-1) @ sources
        outs = []
        for slot, (down, expand) in enumerate(weights):
            msg = messages[:, slot]
            pair = mx.stack([x, msg], axis=1)
            inputs = mx.fast.rms_norm(pair, None, 1e-6).astype(mx.bfloat16).reshape(x.shape[0], -1)
            compressed = (inputs @ down).astype(mx.float32)
            gain, forget = mx.split(mx.tanh(compressed @ expand), 2, axis=-1)
            outs.append((1 - forget) * x + gain * msg)
        return mx.stack(outs, axis=1)

    report = {}
    for rows in (1, 32):
        cases = [
            (
                mx.array(rng.standard_normal((rows, 2304), dtype=np.float32)),
                mx.array(rng.standard_normal((rows, 2, 2304), dtype=np.float32)),
            )
            for _ in range(4)
        ]
        for case in cases:
            mx.eval(*case)
        report[f"layer6_message_read_four_gates_{rows}rows"] = timed(step, cases, rows)
    return report


def attention_layer(store, rng, physical):
    # Full attention sublayer on synthetic pre-normalized hidden inputs, KV
    # history and convolution tails. The chronological KV view uses a concat.
    # This deliberately excludes ring mutation, outer residuals and post-conv.
    local = physical == 6
    base = f"model.llm.layers.{physical}.attn."
    projectors = {
        name: store.metal(base + name + ".weight").T
        for name in (
            "wq_du",
            "wk_dv",
            "wv_dv",
            "wr_du",
            "wo_ud",
            "gate_attn",
            "attn_sink_proj",
        )
    }
    relative = store.metal(base + "rel_logits_proj.proj")
    q_clip = store.metal(base + "q_l2_clip.scale")
    r_clip = store.metal(base + "rel_l2_clip.scale")
    conv = []
    for side in ("k", "v"):
        prefix = base + side + "_sconv."
        if local:
            conv.append((store.metal(prefix + "weight")[:, 0, :].T, None))
        else:
            conv.append(
                (
                    store.metal(prefix + "static_w"),
                    store.metal(prefix + "weight_proj").T,
                )
            )
    for matrix in projectors.values():
        mx.eval(matrix)
    mx.eval(relative, q_clip, r_clip)
    for static, dynamic in conv:
        mx.eval(static)
        if dynamic is not None:
            mx.eval(dynamic)

    history_length = 1023 if local else 4095
    history = [
        mx.array(rng.standard_normal((1, 8, history_length, 128), dtype=np.float32)).astype(mx.bfloat16)
        for _ in range(2)
    ]
    tails = [mx.array(rng.standard_normal((3, 1024), dtype=np.float32)).astype(mx.bfloat16) for _ in range(2)]
    mx.eval(*history, *tails)

    def convolve(x, source, side):
        static, dynamic = conv[side]
        taps = static.astype(mx.float32)
        rows = x.shape[0]
        past = mx.concatenate([tails[side], x]).astype(mx.float32)
        if dynamic is None:
            value = sum(past[j : j + rows] * taps[j] for j in range(4))
            return ((value + x) / math.sqrt(5)).astype(mx.bfloat16)
        coeff = (source @ dynamic).astype(mx.float32).reshape(rows, 8, 4)
        value = sum(
            past[3 - j : 3 - j + rows].reshape(rows, 8, 128) * (taps[j].reshape(8, 128) + coeff[:, :, j : j + 1])
            for j in range(4)
        )
        return ((value.reshape(rows, 1024) + x) / 3).astype(mx.bfloat16)

    def clip_l2(x, threshold, weight=None):
        value = x.astype(mx.float32)
        if weight is not None:
            value = value * weight.astype(mx.float32)
        norm = mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))
        return (value * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(mx.bfloat16)

    def measure_rows(rows):
        width = history_length + rows
        distance = mx.arange(history_length, width)[:, None] - mx.arange(width)
        in_table = (distance >= 0) & (distance < relative.shape[-1])
        ids = mx.clip(distance, 0, relative.shape[-1] - 1)[None, None]
        visible = (distance >= 0) & ((distance < 1024) if local else True)
        sinks = mx.zeros((24,), dtype=mx.float32)
        mx.eval(in_table, visible, ids, sinks)

        def step(qin, kin, vin):
            q = clip_l2(
                (qin @ projectors["wq_du"]).reshape(rows, 24, 128),
                math.sqrt(512),
                q_clip,
            )
            k = convolve(kin @ projectors["wk_dv"], kin, 0)
            k = clip_l2(k.reshape(rows, 8, 128), math.sqrt(512))
            v = convolve(vin @ projectors["wv_dv"], vin, 1)
            v = mx.fast.rms_norm(v.reshape(rows, 8, 128), None, 1e-6)
            r = clip_l2((qin @ projectors["wr_du"]).reshape(rows, 24, 16), 8, r_clip)
            table = (r @ relative).transpose(1, 0, 2)[None]
            bias = mx.take_along_axis(table, ids, axis=-1)
            sink = (qin @ projectors["attn_sink_proj"]).T[None, :, :, None].astype(mx.float32)
            bias = mx.where(
                visible,
                mx.where(in_table, bias.astype(mx.float32), 0) - sink,
                -float("inf"),
            )
            k_all = mx.concatenate([history[0], k.transpose(1, 0, 2)[None]], axis=2).astype(mx.float32)
            v_all = mx.concatenate([history[1], v.transpose(1, 0, 2)[None]], axis=2).astype(mx.float32)
            y = mx.fast.scaled_dot_product_attention(
                q.transpose(1, 0, 2)[None].astype(mx.float32),
                k_all,
                v_all,
                scale=128**-0.5,
                mask=bias,
                sinks=sinks,
            )
            y = y[0].transpose(1, 0, 2).reshape(rows, 3072)
            gated = (y * mx.sigmoid((qin @ projectors["gate_attn"]).astype(mx.float32))).astype(mx.bfloat16)
            return gated @ projectors["wo_ud"]

        cases = []
        for _ in range(4):
            inputs = tuple(
                mx.array(rng.standard_normal((rows, 2304), dtype=np.float32)).astype(mx.bfloat16) for _ in range(3)
            )
            mx.eval(*inputs)
            cases.append(inputs)
        return timed(step, cases, rows)

    return {f"layer{physical}_attention_sublayer_{rows}rows": measure_rows(rows) for rows in (1, 32)}


def attention_probe(store, rng):
    return attention_layer(store, rng, 6) | attention_layer(store, rng, 11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    store = Weights(args.model)
    rng = np.random.default_rng(20261120)
    report = {
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "synthetic_activations": True,
    }
    for label, function in [
        ("mxfp8", mxfp8),
        ("cross_pass", cross_pass),
        ("attention", attention_probe),
    ]:
        report[label] = function(store, rng)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(label, "complete", flush=True)


if __name__ == "__main__":
    main()
