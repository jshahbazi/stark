"""Run a source-bound NVFP4 operator experiment. Only aggregate results leave RAM."""

import argparse
import ctypes as ct
import hashlib
import json
import mmap
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
SIGNED_FP4 = np.concatenate([FP4, -FP4])
CODES = np.arange(127, dtype=np.int32)
E4M3 = np.where(CODES < 8, CODES * 2.0**-9, (1 + (CODES % 8) / 8) * np.exp2(CODES // 8 - 7)).astype(np.float32)


def nearest_even(x, grid):
    """Nearest code on an ascending positive grid, breaking ties on code parity."""
    upper = np.minimum(np.searchsorted(grid, x, side="left"), len(grid) - 1)
    lower = np.maximum(upper - 1, 0)
    low_error, high_error = x - grid[lower], grid[upper] - x
    take_low = (low_error < high_error) | ((low_error == high_error) & (lower % 2 == 0))
    return grid[np.where(take_low, lower, upper)]


def bf16(x):
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    halves = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded = (halves.astype(np.uint32) << 16).view(np.float32)
    return halves, rounded


def cpu_activation(x):
    """Independent E4M3 and E2M1 table lookup, including 4-over-6 MAE scale selection."""
    blocks = x.reshape(len(x), -1, 16)
    block_maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    row_maximum = np.max(block_maximum, axis=1, keepdims=True)
    normalizer = np.divide(
        np.float32(1536), row_maximum, where=row_maximum != 0, out=np.full_like(row_maximum, np.finfo(np.float32).max)
    )
    candidates, errors = [], []
    for divisor in (6, 4):
        scales = nearest_even(np.minimum(block_maximum * (normalizer * np.float32(1 / divisor)), 448), E4M3)
        denom = scales * (1 / normalizer)
        inverse = np.divide(np.float32(1), denom, where=denom != 0, out=np.full_like(denom, np.finfo(np.float32).max))
        quantized = np.copysign(nearest_even(np.abs(blocks) * inverse, FP4), blocks) * scales
        candidates.append(quantized)
        errors.append(
            np.add.accumulate(np.abs(blocks - quantized * (row_maximum * np.float32(1 / 1536))), axis=-1)[..., -1:]
        )
    return np.where(errors[1] < errors[0], candidates[1], candidates[0]).reshape(x.shape), row_maximum.ravel()


def activation_shader(go_source, width):
    source = go_source.read_bytes()
    if hashlib.sha256(source).hexdigest() != "8a69afd0c9c24a239af288802ea32028c6be3fa06ca1a730959e76a3743fdfc2":
        raise ValueError("unexpected Ollama activation source")
    match = re.search(r"metal: gpuSource\{\s*header: `(.*?)`,\s*source: `(.*?)`,", source.decode(), re.DOTALL)
    header, body = match.groups()
    signature = """kernel void quantize_activation(
        device const bfloat* x [[buffer(0)]], device float* inner [[buffer(1)]], device float* amax [[buffer(2)]],
        uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
        uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
        uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
        uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {"""
    workers = ((width // 16 + 31) // 32) * 32
    defines = f"#define W {width}\n#define G {workers}\n#define MaxProduct 1536"
    return f"{header}\n{defines}\n{signature}\n{body}\n}}\n".encode()


def stats(times):
    return {k: float(np.percentile(times, q) * 1000) for k, q in (("p10_ms", 10), ("median_ms", 50), ("p90_ms", 90))}


def metrics(got, ref):
    a = got.astype(np.float64)
    b = ref.astype(np.float64)
    delta = a - b
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-100)),
        "max_abs_over_ref_max": float(np.max(np.abs(delta)) / max(np.max(np.abs(b)), 1e-100)),
    }


class Official:
    def __init__(self, path):
        with path.open("rb") as source:
            self.backing = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        size = struct.unpack_from("<Q", self.backing)[0]
        self.start = size + 8
        self.header = json.loads(self.backing[8 : self.start])

    def tensor(self, name):
        tensor = self.header[name]
        dtype = np.float32 if tensor["dtype"] == "F32" else np.uint8
        return np.ndarray(tensor["shape"], dtype, self.backing, self.start + tensor["data_offsets"][0])


class Reference:
    def __init__(self, layer, down, official, gguf, block6_screen):
        gg_name = f"blk.{layer}.ffn_{'down' if down else 'gate_up'}_exps"
        hf_name = f"model.llm.layers.{layer}.mlp.experts.{'w2' if down else 'w13'}_weight"
        self.gbank = gguf[gg_name + ".weight"]
        self.scales = gguf[gg_name + ".scale"].data.reshape(128)
        self.original = official.tensor(hf_name)
        self.block_scales = official.tensor(hf_name + "_scale")
        self.outer = official.tensor(hf_name + "_amax") / official.tensor(hf_name + "_outer_scale_denominator")[0]
        if not np.array_equal(self.outer, self.scales):
            raise ValueError("expert outer scales differ")
        width, rows, experts = map(int, self.gbank.shape)
        if int(self.gbank.tensor_type) != 40 or experts != 128:
            raise ValueError("unexpected GGUF descriptor")
        self.width, self.rows = width, rows
        self.indices = np.unique(
            np.r_[
                0,
                1,
                rows // 2 - 1,
                rows // 2,
                rows - 1,
                np.random.default_rng(71 + layer).choice(rows, size=43, replace=False),
            ]
        )
        if block6_screen:
            self.indices = np.linspace(0, rows - 1, 65, dtype=int)
        self.original_indices = (
            self.indices if down else 2 * (self.indices % (rows // 2)) + self.indices // (rows // 2)
        )
        self.checked = set()
        self.reference_rows = {}

    def rows_for_expert(self, expert):
        if expert in self.reference_rows:
            return self.reference_rows[expert]
        raw = self.original[expert, self.original_indices]
        codes = np.stack([raw & 15, raw >> 4], axis=-1).reshape(len(self.indices), -1, 16)
        original_scales = self.block_scales[expert, self.original_indices]
        # Compare exact bytes for all sampled scale and weight codes before floating-point validation.
        packed = self.gbank.data[expert, self.indices].reshape(len(self.indices), -1, 36)
        if not np.array_equal(packed[..., :4].reshape(original_scales.shape), original_scales):
            raise ValueError("GGUF block scales differ from the official shard")
        if not np.array_equal(
            packed[..., 4:].reshape(len(self.indices), -1, 8), codes[:, :, :8] | (codes[:, :, 8:] << 4)
        ):
            raise ValueError("GGUF weight codes differ from the official shard")
        self.checked.add(expert)
        rows = (
            (SIGNED_FP4[codes] * E4M3[original_scales][..., None] * self.outer[expert])
            .reshape(len(self.indices), self.width)
            .astype(np.float64)
        )
        self.reference_rows[expert] = rows
        return rows

    def project(self, inputs, ids):
        expected = np.zeros((*ids.shape, len(self.indices)), np.float64)
        for expert in np.unique(ids):
            token, lane = np.where(ids == expert)
            chosen = inputs[token, np.zeros_like(lane) if inputs.shape[1] == 1 else lane]
            expected[token, lane] = chosen.astype(np.float64) @ self.rows_for_expert(int(expert)).T
        return expected


def load_probe(path):
    lib = ct.CDLL(str(path.resolve()))
    integer, ptr = ct.c_int, ct.c_void_p
    lib.capture_visits.argtypes = [ct.c_char_p, ct.c_char_p, integer]
    lib.capture_meta.argtypes = [integer, ptr]
    lib.capture_copy.argtypes = [integer, ptr, ptr]
    lib.read_span.argtypes = [integer, ptr, ptr]
    lib.read_span.restype = ct.c_double
    lib.probe_projection.argtypes = [integer] * 5 + [ptr, ct.c_size_t] + [ptr] * 5 + [integer, integer, ptr, integer]
    lib.probe_activation.argtypes = [ct.c_char_p, integer, integer] + [ptr] * 5 + [integer, integer]
    return lib


def activation(lib, shader, x, repeats, dispatch_groups=1):
    x = np.ascontiguousarray(x)
    width = x.shape[-1]
    halves, rounded = bf16(x.reshape(-1, width))
    cpu_inner, cpu_max = cpu_activation(rounded)
    gpu_inner, gpu_max = np.empty_like(cpu_inner), np.empty_like(cpu_max)
    wall, gpu = np.zeros(repeats), np.zeros(repeats)
    rc = lib.probe_activation(
        shader,
        width,
        len(halves),
        halves.ctypes.data,
        gpu_inner.ctypes.data,
        gpu_max.ctypes.data,
        wall.ctypes.data,
        gpu.ctypes.data,
        repeats,
        dispatch_groups,
    )
    if rc:
        raise RuntimeError(f"Metal activation failure {rc}")
    result = {
        "max_mismatches": int(np.sum(gpu_max != cpu_max)),
        "inner_mismatches": int(np.sum(gpu_inner != cpu_inner)),
        "checked": gpu_inner.size,
        "dispatches_per_submission": dispatch_groups,
        "wall": stats(wall / dispatch_groups),
        "gpu": stats(gpu / dispatch_groups),
    }
    gpu_outer = (gpu_max * np.float32(1 / 1536)).reshape(*x.shape[:-1], 1)
    return (
        gpu_inner.reshape(x.shape) * gpu_outer,
        (cpu_inner * (cpu_max[:, None] * np.float32(1 / 1536))).reshape(x.shape),
        result,
        gpu_inner.reshape(x.shape),
        gpu_outer,
    )


def request_cases(lib, gguf, tokens):
    # Public deterministic synthetic prose; activations and routes are computed by the actual GGUF.
    fixture = (
        "The observatory measured stellar brightness through several filters. "
        "A software engineer compared the readings, reviewed the controls, and repeated the experiment. "
    ) * 12
    n = lib.capture_visits(str(gguf).encode(), fixture.encode(), tokens)
    if n != 24:
        raise RuntimeError(f"expected 24 real graph input captures, found {n}")
    found = defaultdict(list)
    for index in range(n):
        meta = np.empty(5, np.int32)
        lib.capture_meta(index, meta.ctypes.data)
        layer, down, t, c, k = map(int, meta)
        arr, ids = np.empty((t, c, k), np.float32), np.empty((t, 8), np.int32)
        lib.capture_copy(index, arr.ctypes.data, ids.ctypes.data)
        if not np.isfinite(arr).all() or (ids < 0).any() or (ids >= 128).any():
            raise ValueError("invalid captured projections")
        found[layer, down, t].append((arr, ids))
    for layer in (6, 11):
        for down in (0, 1):
            singles, batches = found[layer, down, 1], found[layer, down, tokens]
            if len(singles) != 3 or len(batches) != 3:
                raise ValueError("invalid visit count")
            for visit, ((x, ids), (bx, bids)) in enumerate(zip(singles, batches, strict=True), start=1):
                # Vary singleton routes/working sets across timed repetitions. Check the true decode row.
                yield layer, down, visit, np.concatenate([x, bx])[:, None], np.concatenate([ids, bids])[:, None]
                yield layer, down, visit, bx[None], bids[None]


def main():
    from gguf import GGUFReader

    ap = argparse.ArgumentParser()
    for name in ("gguf", "official", "lib", "activation-source", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--layer0-only", action="store_true", help="Run the original-checkpoint ingress fixtures only")
    ap.add_argument(
        "--block6-screen",
        action="store_true",
        help="Replay the early seeded eight-expert control on the complete pinned artifacts",
    )
    args = ap.parse_args()
    lib = load_probe(args.lib)
    official = Official(args.official)
    gguf = {t.name: t for t in GGUFReader(args.gguf).tensors}
    result = {"cases": [], "instrumented_spans": []}

    def checkpoint():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    ref_cache = {}
    shaders = {k: activation_shader(args.activation_source, k) for k in (1152, 2304)}

    def fixtures():
        if args.block6_screen:
            rng = np.random.default_rng(324)
            for down in (0, 1):
                width, channels = (1152, 8) if down else (2304, 1)
                for tokens in (1, 32):
                    x = rng.standard_normal((1, tokens, channels, width)).astype(np.float32)
                    ids = np.stack([rng.permutation(8) for _ in range(tokens)]).astype(np.int32)[None]
                    yield 6, down, 1, x, ids
            return
        from original_layer0 import layer0_fixtures

        x, ids = layer0_fixtures(args.official)
        yield 0, 0, 1, x[:, None], ids[:, None]
        yield 0, 0, 1, x[None], ids[None]
        if not args.layer0_only:
            yield from request_cases(lib, args.gguf, args.tokens)

    for layer, down, visit, inputs, ids in fixtures():
        key = (layer, down)
        if key not in ref_cache:
            ref_cache[key] = Reference(layer, down, official, gguf, args.block6_screen)
        ref = ref_cache[key]
        fixture_count, tokens, channels, width = inputs.shape
        # Quantizer correctness over every input. Standalone latency separately for deployment shape.
        quant, cpu_quant, q_stats, inner, outer = activation(lib, shaders[width], inputs, args.repeats)
        if fixture_count > 1:
            _, _, shape_stats, _, _ = activation(lib, shaders[width], inputs[0], args.repeats)
        else:
            shape_stats = q_stats
        source = "original_layer0_cpu" if layer == 0 else "gguf_graph_callback"
        case = {
            "physical": layer,
            "bank": "down" if down else "gate_up",
            "pass": visit,
            "tokens": tokens,
            "input_source": "seeded_block6" if args.block6_screen else source,
            "fixtures": fixture_count,
            "distinct_experts": len(np.unique(ids)),
            "activation": q_stats,
            "activation_deployed_shape": shape_stats,
            "projection": {},
        }
        refs = {}
        arms = [
            ("float", inputs, inputs, None, ids, 0),
            ("four_over_six_early", quant, cpu_quant, None, ids, 0),
            ("four_over_six", inner, cpu_quant, outer, ids, 0),
        ]
        if tokens > 1:
            arms.append(("four_over_six_serial_replay", inner, cpu_quant, outer, ids, 1))
        if tokens == 1 and fixture_count >= 8:
            _, _, group_stats, _, _ = activation(lib, shaders[width], inputs[:8], args.repeats, 8)
            case["activation_eight_singletons_per_submission"] = group_stats

            def grouped(value):
                groups = len(value) // 8
                return value[: groups * 8, 0].reshape(groups, 8, *value.shape[2:])

            arms.append(
                ("four_over_six_eight_singletons", grouped(inner), grouped(cpu_quant), grouped(outer), grouped(ids), 1)
            )
        for mode, src, reference_input, scale, route_ids, replay in arms:
            arm_fixtures, arm_tokens = src.shape[:2]
            output = np.empty((arm_tokens, 8, ref.rows), np.float32)
            timing = np.zeros(args.repeats, np.float64)
            bank = ref.gbank.data[:8] if args.block6_screen else ref.gbank.data
            rc = lib.probe_projection(
                width,
                ref.rows,
                len(bank),
                arm_tokens,
                channels,
                bank.ctypes.data,
                bank.nbytes,
                src.ctypes.data,
                route_ids.ctypes.data,
                ref.scales.ctypes.data,
                output.ctypes.data,
                timing.ctypes.data,
                args.repeats,
                arm_fixtures,
                scale.ctypes.data if scale is not None else None,
                replay,
            )
            if rc:
                raise RuntimeError(f"Metal projection failed {rc}")
            if not np.isfinite(output).all():
                raise ValueError("nonfinite output")
            expected = ref.project(reference_input[0], route_ids[0])
            comparisons = metrics(output[:, :, ref.indices], expected)
            divisor = 8 if mode == "four_over_six_eight_singletons" else 1
            case["projection"][mode] = {
                **comparisons,
                "compared": expected.size,
                "wall_divisor": divisor,
                "wall": stats(timing / divisor),
            }
            refs[mode] = expected
        case["float_vs_four_over_six"] = metrics(refs["float"], refs["four_over_six"])
        result["cases"].append(case)
        checkpoint()
        print(f"Completed layer {layer}, pass {visit}, {len(case['projection'])} arms, {tokens} rows", flush=True)
    result["weight_checks"] = [
        {
            "physical": key[0],
            "down": key[1],
            "expert_count": len(ref.checked),
            "sampled_rows_per_expert": len(ref.indices),
            "channels": ref.width,
        }
        for key, ref in ref_cache.items()
    ]
    for i in range(lib.span_count()):
        visit, stage = ct.c_int(), ct.c_int()
        dt = lib.read_span(i, ct.byref(visit), ct.byref(stage))
        result["instrumented_spans"].append({"visit": visit.value, "stage": stage.value, "ms": dt * 1000})
    checkpoint()


if __name__ == "__main__":
    main()
