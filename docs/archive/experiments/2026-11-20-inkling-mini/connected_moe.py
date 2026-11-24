"""Measure an original-checkpoint, GPU-resident Inkling expert sublayer on Apple MLX."""

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from auxiliary import Weights
from nvfp4_probe import E4M3, SIGNED_FP4, bf16, cpu_activation, metrics
from original_layer0 import layer0_fixtures


def activation_kernel(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != "8a69afd0c9c24a239af288802ea32028c6be3fa06ca1a730959e76a3743fdfc2":
        raise ValueError("unexpected Ollama activation shader")
    header, body = re.search(
        r"metal: gpuSource\{\s*header: `(.*?)`,\s*source: `(.*?)`,", source.decode(), re.DOTALL
    ).groups()
    fn = mx.fast.metal_kernel(
        name="inkling_activation",
        input_names=["x"],
        output_names=["inner", "amax"],
        source=body,
        header=header,
    )

    def activate(x):
        rows, width = x.shape
        group = (width // 16 + 31) // 32 * 32
        return fn(
            inputs=[x],
            template=[("W", width), ("G", group), ("MaxProduct", 1536)],
            grid=(group, rows, 1),
            threadgroup=(group, 1, 1),
            output_shapes=[x.shape, (rows, 1)],
            output_dtypes=[mx.float32, mx.float32],
        )

    return activate


class Bank:
    def __init__(self, weights, part):
        prefix = f"model.llm.layers.0.mlp.experts.{part}_weight"
        self.raw = weights.read(prefix)
        self.scales = weights.read(prefix + "_scale")
        self.maxima = weights.read(prefix + "_amax")
        self.denom = weights.read(prefix + "_outer_scale_denominator")[0]
        self.packed = mx.array(np.asarray(self.raw).view(np.uint32))
        self.blocks = mx.array(np.asarray(self.scales))
        self.maxima_gpu = mx.array(np.asarray(self.maxima))
        mx.eval(self.packed, self.blocks, self.maxima_gpu)

    def project(self, activate, x, lhs, rhs):
        inner, amax = activate(x)
        product = mx.gather_qmm(
            inner[:, None, :],
            self.packed,
            self.blocks,
            lhs_indices=lhs,
            rhs_indices=rhs,
            group_size=16,
            bits=4,
            mode="nvfp4",
        ).squeeze(-2)
        outer = amax[lhs] * self.maxima_gpu[rhs][..., None] / float(1536 * self.denom)
        return (product * outer).astype(mx.bfloat16)

    def reference(self, x, expert, rows):
        raw = self.raw[expert, rows]
        codes = np.stack((raw & 15, raw >> 4), axis=-1).reshape(len(raw), -1, 16)
        weight = SIGNED_FP4[codes] * E4M3[self.scales[expert, rows]][..., None]
        inner, amax = cpu_activation(x.reshape(1, -1))
        prod = weight.reshape(len(raw), -1).astype(np.float64) @ inner[0].astype(np.float64)
        outer = amax[0] * self.maxima[expert] / (np.float32(1536) * self.denom)
        return bf16(prod.astype(np.float32) * outer)[1]


def make_forward(weights, up, down, activate):
    router = weights.metal("model.llm.layers.0.mlp.gate.weight").T.astype(mx.float32)
    bias = weights.metal("model.llm.layers.0.mlp.gate.bias")
    mx.eval(router, bias)

    def forward(x):
        rows = x.shape[0]
        logits = x.astype(mx.float32) @ router
        ids = mx.argpartition(-(mx.sigmoid(logits) + bias), 7)[:, :8]
        chosen = mx.take_along_axis(logits, ids, axis=-1)
        scores = 8 * mx.softmax(-mx.logaddexp(0, -chosen), axis=-1)
        lhs = mx.broadcast_to(mx.arange(rows, dtype=mx.uint32)[:, None], ids.shape)
        paired = up.project(activate, x, lhs, ids).astype(mx.float32)
        gate, value = paired[..., ::2], paired[..., 1::2]
        intermediate = (4 * mx.tanh(gate / 4)) * mx.sigmoid(gate) * (25 * mx.tanh(value / 25))
        intermediate = (intermediate * scores[..., None]).astype(mx.bfloat16).reshape(rows * 8, -1)
        out = down.project(activate, intermediate, mx.arange(rows * 8, dtype=mx.uint32), ids.reshape(-1)).reshape(
            rows, 8, -1
        )
        return mx.sum(out.astype(mx.float32), axis=1).astype(mx.bfloat16), ids

    return mx.compile(forward)


def check_outputs(weights, up, down, inputs, ids, measured):
    raw_router = weights.read("model.llm.layers.0.mlp.gate.weight")
    router = (raw_router.astype(np.uint32) << 16).view(np.float32)
    selected = np.linspace(0, inputs.shape[-1] - 1, 48, dtype=int)
    expected = []
    for pos in (0, 15, 31):
        x = inputs[pos]
        logits = router[ids[pos]].astype(np.float64) @ x.astype(np.float64)
        scores = 1 / (1 + np.exp(-logits))
        scores = (8 * scores / scores.sum()).astype(np.float32)
        routes = []
        for score, expert in zip(scores, ids[pos], strict=True):
            paired = up.reference(x, expert, slice(None))
            gate, value = paired[::2], paired[1::2]
            middle = 4 * np.tanh(gate / 4) * (1 / (1 + np.exp(-gate))) * (25 * np.tanh(value / 25))
            routes.append(down.reference(bf16(middle * score)[1], expert, selected))
        expected.append(bf16(np.sum(routes, axis=0, dtype=np.float32))[1])
    actual = measured[[0, 15, 31]][:, selected]
    return metrics(actual, np.array(expected)) | {
        "elements": int(actual.size),
        "unequal_bf16_outputs": int(np.count_nonzero(actual != expected)),
    }


def measure(forward, cases, *, ordered=False):
    elapsed = []
    for i in range(40):
        start = time.perf_counter_ns()
        # Each input retains its original matrix shape. Multiple independent
        # singleton graphs share a single terminal wait in the dispatch control.
        outputs = []
        for x in cases[i % len(cases)]:
            # Preserve the real input while requiring each sublayer to finish
            # before the next router can consume its own row.
            if ordered and outputs:
                x = mx.depends(x, outputs[-1])
            outputs.append(forward(x)[0])
        mx.eval(*outputs)
        ms = (time.perf_counter_ns() - start) / 1e6
        if i >= 8:
            elapsed.append(ms)
    return {
        "warmups": 8,
        "samples": len(elapsed),
        "median_ms": statistics.median(elapsed),
        "p10_ms": float(np.percentile(elapsed, 10)),
        "p90_ms": float(np.percentile(elapsed, 90)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("shader", type=Path)
    parser.add_argument("workloads", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    weights = Weights(args.model)
    inputs, cpu_ids = layer0_fixtures(args.model, args.workloads, token_count=32)
    inputs = inputs[:, 0]
    up, down = Bank(weights, "w13"), Bank(weights, "w2")
    forward = make_forward(weights, up, down, activation_kernel(args.shader))
    tile = mx.array(inputs).astype(mx.bfloat16)
    singles = [tile[i : i + 1] for i in range(len(tile))]
    mx.eval(tile, *singles)
    tile_output, tile_ids = forward(tile)
    measured, ids = np.array(tile_output.astype(mx.float32)), np.array(tile_ids)
    single_outputs, single_ids = zip(*(forward(x) for x in singles), strict=True)
    singleton = np.array(mx.concatenate(single_outputs).astype(mx.float32))
    single_ids = np.array(mx.concatenate(single_ids))
    reference = np.sort(cpu_ids, axis=-1)
    result = {
        "mlx": mx.__version__,
        "physical_layer": 0,
        "original_input_positions": 32,
        "unique_experts": len(np.unique(cpu_ids)),
        "full_bank_experts": len(up.raw),
        "tile_route_set_mismatches": int(np.any(np.sort(ids, axis=-1) != reference, axis=1).sum()),
        "singleton_route_set_mismatches": int(np.any(np.sort(single_ids, axis=-1) != reference, axis=1).sum()),
        "all_finite": bool(np.isfinite(measured).all() and np.isfinite(singleton).all()),
        "tile_vs_reference": check_outputs(weights, up, down, inputs, ids, measured),
        "singleton_vs_reference": check_outputs(weights, up, down, inputs, single_ids, singleton),
        "tile_vs_singleton": metrics(measured, singleton),
        "resident_one_row": measure(forward, [[row] for row in singles]),
        "resident_32_rows": measure(forward, [[tile]]),
        "eight_independent_singletons_one_wait": measure(
            forward, [singles[start : start + 8] for start in range(0, 32, 8)]
        ),
        "eight_ordered_singletons_one_wait": measure(
            forward, [singles[start : start + 8] for start in range(0, 32, 8)], ordered=True
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
