"""Independent original-weight reference for a bounded Inkling Mini slice.

The equations follow Ollama 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8
mlxrunner/model/inkling2/{model,residual,conv,attention,experts,weights}.go.
Intermediate values and checkpoint tensors are private.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-11-20-inkling-mini"))
from auxiliary import Weights
from connected_moe import Bank, activation_kernel
from original_layer0 import bf16_round

MODEL_PREFIX = "model.llm."
WIDTH = 2304
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
WINDOW = 1024
EPSILON = np.float32(1e-6)
LOCAL_LAYERS = frozenset(
    (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
)
CONV_WIDTHS = (1024, 1024, WIDTH, WIDTH)


def rms(value, weight=None):
    normalized = value / np.sqrt(
        np.mean(value * value, axis=-1, keepdims=True) + EPSILON
    )
    return normalized if weight is None else bf16_round(normalized * weight)


def linear(value, weight, *, rounded=True):
    projected = value @ weight.T
    return bf16_round(projected) if rounded else projected


def mlx_linear(value, weight, *, rounded=True):
    """Use the pinned short-target eight-row projection geometry with Python MLX."""
    dtype = mx.bfloat16 if rounded else mx.float32
    padded = np.pad(value, ((0, (-len(value)) % 8), (0, 0))).reshape(
        -1, 8, value.shape[-1]
    )
    weights = mx.array(weight).astype(dtype).T
    rows = [mx.array(tile).astype(dtype) @ weights for tile in padded]
    return np.asarray(mx.concatenate(rows, axis=0)[: len(value)].astype(mx.float32))


def softmax(value):
    exponentials = np.exp(value - np.max(value, axis=-1, keepdims=True))
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def mix(blocks, partial, query):
    sources = rms(np.concatenate((blocks, partial[:, None, :]), axis=1))
    scores = sources @ query.T / np.float32(math.sqrt(WIDTH))
    if query.ndim == 1:
        return np.sum(sources * softmax(scores)[..., None], axis=1)
    weights = softmax(scores.swapaxes(-1, -2))
    return weights @ sources


def source_messages(previous, query):
    sources = rms(previous)
    scores = query @ sources.swapaxes(-2, -1) / np.float32(math.sqrt(WIDTH))
    return softmax(scores) @ sources


def controller(value, message, down, gain, forget, project=linear):
    normalized = bf16_round(rms(np.stack((value, message), axis=1))).reshape(
        len(value), -1
    )
    compressed = project(normalized, down)
    if project is mlx_linear:
        expanded = project(compressed, np.concatenate((gain, forget)), rounded=False)
        gain, forget = np.split(expanded, 2, axis=-1)
    else:
        gain = project(compressed, gain, rounded=False)
        forget = project(compressed, forget, rounded=False)
    retain = 1 - np.tanh(forget)
    admit = np.tanh(gain)
    return value * retain + message * admit


def causal_conv(value, tail, taps, conditioning=None, projection=None, project=linear):
    row_count, channels = value.shape
    history = np.concatenate((tail, value), axis=0)
    if projection is None:
        total = sum(history[tap : tap + row_count] * taps[tap] for tap in range(4))
        divisor = math.sqrt(5)
    else:
        coeff = project(conditioning, projection).reshape(row_count, KV_HEADS, 4)
        total = sum(
            history[3 - lag : 3 - lag + row_count].reshape(row_count, KV_HEADS, -1)
            * (taps[lag].reshape(KV_HEADS, -1) + coeff[:, :, lag, None])
            for lag in range(4)
        ).reshape(row_count, channels)
        divisor = 3
    return bf16_round((value + total) / np.float32(divisor)), history[-3:]


def clip_l2(value, threshold, weight=None, temperature=None):
    if weight is not None:
        value = value * weight
    norm = np.sqrt(np.sum(value * value, axis=-1, keepdims=True))
    value = value * np.minimum(np.float32(threshold) / (norm + EPSILON), 1)
    if temperature is not None:
        value = value * temperature
    return bf16_round(value)


def relative_attention(q, k, v, relative, sink, history_length, window):
    positions = history_length + np.arange(len(q))
    distances = positions[:, None] - np.arange(len(k))[None, :]
    valid = distances >= 0
    if window:
        valid &= distances < window
    within_table = (distances >= 0) & (distances < relative.shape[-1])
    indices = np.clip(distances, 0, relative.shape[-1] - 1)
    bias = np.take_along_axis(relative, indices[:, None, :], axis=-1)
    bias = np.where(within_table[:, None, :], bias, 0)
    queries = q.reshape(len(q), KV_HEADS, HEADS // KV_HEADS, HEAD_DIM)
    dot = np.einsum("qagd,kad->qagk", queries, k).reshape(len(q), HEADS, len(k))
    score = dot / np.float32(math.sqrt(HEAD_DIM)) + bias
    masked = np.where(valid[:, None, :], score, -np.inf)
    maximum = np.maximum(np.max(masked, axis=-1), sink)
    numerators = np.exp(masked - maximum[..., None])
    denominator = numerators.sum(axis=-1) + np.exp(sink - maximum)
    probs = (numerators / denominator[..., None]).reshape(
        len(q), KV_HEADS, HEADS // KV_HEADS, len(k)
    )
    return np.einsum("qagk,kad->qagd", probs, v).reshape(len(q), HEADS * HEAD_DIM)


class Layer:
    def __init__(self, weights, number, activate, project=linear):
        self.weights = weights
        self.number = number
        self.prefix = f"{MODEL_PREFIX}layers.{number}."
        self.local = number in LOCAL_LAYERS
        self.activate = activate
        self.project = project

    def tensor(self, name):
        return self.model_tensor(self.prefix + name)

    def model_tensor(self, name):
        value = np.asarray(self.weights.read(name))
        if self.weights.specs[name]["dtype"] == "BF16":
            return (value.astype(np.uint32) << 16).view(np.float32)
        return value

    def attention_inputs(self, blocks, partial, messages):
        mixed = partial
        if blocks.shape[1]:
            residual_query = self.model_tensor(
                f"{MODEL_PREFIX}attn_res.attn_queries.{self.number - 1}"
            )
            mixed = mix(blocks, partial, residual_query)
        if mixed.ndim == 2:
            mixed = np.broadcast_to(mixed[:, None], (len(partial), 3, WIDTH))
        names = (
            ["attn_norm"] * 3
            if self.local
            else ["q_input_norm", "k_input_norm", "v_input_norm"]
        )
        return [
            rms(
                self.apply_controller(
                    mixed[:, index],
                    None if messages is None else messages[:, index],
                    part,
                ),
                self.tensor(name + ".weight"),
            )
            for index, (part, name) in enumerate(
                zip(("attn", "k", "v"), names, strict=True)
            )
        ]

    def apply_controller(self, value, message, part):
        if message is None:
            return value
        prefix = f"controllers.{part}."
        return controller(
            value,
            message,
            *(
                self.tensor(prefix + key + "_weight")
                for key in ("down", "gain", "forget")
            ),
            project=self.project,
        )

    def messages(self, previous):
        q = self.model_tensor(MODEL_PREFIX + "middle_cycle_cross_pass_queries")
        kv = self.model_tensor(MODEL_PREFIX + "middle_cycle_cross_pass_kv_queries")
        index = 2 * (self.number - 6)
        return source_messages(
            previous, np.stack([q[index], kv[index], kv[index + 1], q[index + 1]])
        )

    def attention(self, qin, kin, vin, keys, values, tails, offset):
        def project(value, name):
            return self.project(value, self.tensor("attn." + name + ".weight"))

        def convolve(value, source, side, tail):
            prefix = f"attn.{side}_sconv."
            if self.local:
                return causal_conv(value, tail, self.tensor(prefix + "weight")[:, 0].T)[
                    0
                ]
            return causal_conv(
                value,
                tail,
                self.tensor(prefix + "static_w"),
                source,
                self.tensor(prefix + "weight_proj"),
                project=self.project,
            )[0]

        rows = len(qin)
        k_raw = project(kin, "wk_dv")
        v_raw = project(vin, "wv_dv")
        k = convolve(k_raw, kin, "k", tails[0]).reshape(rows, KV_HEADS, HEAD_DIM)
        v = convolve(v_raw, vin, "v", tails[1]).reshape(rows, KV_HEADS, HEAD_DIM)
        tau = None
        if not self.local:
            lengths = np.arange(offset + 1, offset + rows + 1, dtype=np.float32)
            tau = (
                1 + np.float32(0.1) * np.log(np.maximum(lengths / np.float32(8192), 1))
            )[:, None, None]
        q = clip_l2(
            project(qin, "wq_du").reshape(rows, HEADS, HEAD_DIM),
            math.sqrt(512),
            self.tensor("attn.q_l2_clip.scale"),
            tau,
        )
        k = clip_l2(k, math.sqrt(512))
        v = bf16_round(rms(v))
        relative = clip_l2(
            project(qin, "wr_du").reshape(rows, HEADS, 16),
            8,
            self.tensor("attn.rel_l2_clip.scale"),
            tau,
        )
        position_projection = self.tensor("attn.rel_logits_proj.proj")
        if self.project is mlx_linear:
            table = mx.array(relative).astype(mx.bfloat16) @ mx.array(
                position_projection
            ).astype(mx.bfloat16)
            relative = np.asarray(table.astype(mx.float32))
        else:
            relative = bf16_round(relative @ position_projection)
        sink = project(qin, "attn_sink_proj")
        k_all = np.concatenate((keys.reshape(-1, KV_HEADS, HEAD_DIM), k), axis=0)
        v_all = np.concatenate((values.reshape(-1, KV_HEADS, HEAD_DIM), v), axis=0)
        attended = relative_attention(
            q, k_all, v_all, relative, sink, len(keys), WINDOW if self.local else 0
        )
        gate = project(qin, "gate_attn")
        output = project(bf16_round(attended / (1 + np.exp(-gate))), "wo_ud")
        return output, k.reshape(rows, -1), v.reshape(rows, -1), k_raw, v_raw

    def experts(self, value):
        up, down = (
            LayerBank(self.weights, self.prefix, part) for part in ("w13", "w2")
        )
        router = self.tensor("mlp.gate.weight")
        logits = self.project(value, router, rounded=False)
        log_weights = -np.logaddexp(np.float32(0), -logits)
        selection = np.exp(log_weights) + self.tensor("mlp.gate.bias")
        routes = np.argsort(-selection, axis=-1, kind="stable")[:, :8].astype(np.uint32)
        weights = np.float32(8) * softmax(
            np.take_along_axis(log_weights, routes, axis=-1)
        )
        rows = len(value)
        lhs = mx.broadcast_to(mx.arange(rows, dtype=mx.uint32)[:, None], routes.shape)
        indices = mx.array(routes)
        ingress = mx.array(value).astype(mx.bfloat16)
        up_inner, up_max = self.activate(ingress)
        paired = up.project(self.activate, ingress, lhs, indices).astype(mx.float32)
        gate, linear_up = paired[..., ::2], paired[..., 1::2]
        swi_glu = (
            (4 * mx.tanh(gate / 4)) * mx.sigmoid(gate) * (25 * mx.tanh(linear_up / 25))
        )
        weighted = (
            (swi_glu * mx.array(weights)[..., None])
            .astype(mx.bfloat16)
            .reshape(rows * 8, -1)
        )
        down_inner, _ = self.activate(weighted)
        projected = down.project(
            self.activate,
            weighted,
            mx.arange(rows * 8, dtype=mx.uint32),
            indices.reshape(-1),
        )
        first_expert = int(routes[0, 0])
        up.check_projection(value[0], first_expert, paired[0, 0])
        down.check_projection(
            np.asarray(weighted[0].astype(mx.float32)), first_expert, projected[0]
        )
        result = mx.sum(
            projected.reshape(rows, 8, -1).astype(mx.float32), axis=1
        ).astype(mx.bfloat16)
        return {
            "routes": routes,
            "route_weights": weights,
            "experts": np.asarray(result.astype(mx.float32)),
            "mlp_input": value,
            "up_inner": np.asarray(up_inner),
            "up_max": np.asarray(up_max),
            "mid": np.asarray(weighted.astype(mx.float32)).reshape(rows, 8, -1),
            "down_inner": np.asarray(down_inner).reshape(rows, 8, -1),
        }

    def visit(self, blocks, partial, previous, keys, values, tails, offset):
        messages = None if previous is None else self.messages(previous)
        qin, kin, vin = self.attention_inputs(blocks, partial, messages)
        raw_attn, new_keys, new_values, raw_keys, raw_values = self.attention(
            qin, kin, vin, keys, values, tails, offset
        )
        attn, _ = causal_conv(
            raw_attn, tails[2], self.tensor("attn_sconv.weight")[:, 0].T
        )
        if self.number % 6 == 0:
            blocks = np.concatenate((blocks, partial[:, None, :]), axis=1)
            subtotal = attn
        else:
            subtotal = partial + attn
        ffn_query = self.model_tensor(
            f"{MODEL_PREFIX}attn_res.mlp_queries.{self.number}"
        )
        ffn = self.apply_controller(
            mix(blocks, subtotal, ffn_query),
            None if messages is None else messages[:, 3],
            "mlp",
        )
        ffn = rms(ffn, self.tensor("mlp_norm.weight"))
        expert_trace = self.experts(ffn)
        experts = expert_trace["experts"]
        post_moe, _ = causal_conv(
            experts, tails[3], self.tensor("mlp_sconv.weight")[:, 0].T
        )
        return {
            "key_rows": new_keys,
            "value_rows": new_values,
            "conv_rows": np.concatenate(
                (raw_keys, raw_values, raw_attn, experts), axis=-1
            ),
            "messages": messages,
            "attention": attn,
            **expert_trace,
            "partial": bf16_round(subtotal + post_moe),
        }


class LayerBank(Bank):
    def __init__(self, weights, layer_prefix, part):
        prefix = layer_prefix + f"mlp.experts.{part}_weight"
        self.raw = weights.read(prefix)
        self.scales = weights.read(prefix + "_scale")
        self.maxima = weights.read(prefix + "_amax")
        self.denom = weights.read(prefix + "_outer_scale_denominator")[0]
        self.packed = mx.array(np.asarray(self.raw).view(np.uint32))
        self.blocks = mx.array(np.asarray(self.scales))
        self.maxima_gpu = mx.array(np.asarray(self.maxima))

    def check_projection(self, source, expert, actual):
        indices = np.linspace(0, self.raw.shape[1] - 1, 32, dtype=np.int32)
        expected = self.reference(source, expert, indices)
        measured = np.asarray(actual.astype(mx.float32))[indices]
        relative_l2 = np.linalg.norm(measured - expected) / max(
            np.linalg.norm(expected), EPSILON
        )
        if not relative_l2 <= 0.005:
            raise ValueError(
                f"Independent CPU packed projection check failed (relative L2 {relative_l2:.6g})"
            )


def input_state(fixture, layer, keys, values, tails, offset):
    capacity = WINDOW if layer.local else 4352
    physical_positions = np.arange(offset - len(keys), offset) % capacity
    for name, value in (("keys", keys), ("values", values)):
        physical = np.zeros((capacity, KV_HEADS * HEAD_DIM), np.float32)
        physical[physical_positions] = value
        fixture[f"{layer.number}.{name}"] = physical
    physical_tails = [tail[(np.arange(3) - offset) % 3].reshape(-1) for tail in tails]
    fixture[f"{layer.number}.conv"] = np.concatenate(physical_tails)


def make_fixture(weights, activate, project=linear):
    rng = np.random.default_rng(20261121)
    rows, offset = 4, 4096

    def sample(shape):
        return rng.standard_normal(shape, dtype=np.float32)

    fixture = {
        "start": np.array([offset], dtype=np.uint32),
        "blocks": bf16_round(sample((rows, 2, WIDTH))),
        "partial": bf16_round(sample((rows, WIDTH))),
        "previous": bf16_round(sample((rows, 2, WIDTH))),
    }
    partial = fixture["partial"]
    for number in (10, 11):
        layer = Layer(weights, number, activate, project)
        capacity = WINDOW if layer.local else 4352
        length = min(offset, capacity)
        history_shape = (length, KV_HEADS, HEAD_DIM)
        keys = clip_l2(sample(history_shape), math.sqrt(512)).reshape(length, -1)
        values = bf16_round(rms(sample(history_shape))).reshape(length, -1)
        tails = [bf16_round(sample((3, width))) for width in CONV_WIDTHS]
        input_state(fixture, layer, keys, values, tails, offset)
        expected = layer.visit(
            fixture["blocks"], partial, fixture["previous"], keys, values, tails, offset
        )
        fixture.update((f"{number}.{name}", value) for name, value in expected.items())
        partial = expected["partial"]
    return fixture


def trace_prompt(weights, activate, workloads, project=linear):
    """Teacher-force eight real tokens through pass 1 and up to pass 2 layer 11."""
    prompt = json.loads(workloads.read_text())["cases"][0]["prompt"]
    tokenizer = Tokenizer.from_file(str(weights.path.parent / "tokenizer.json"))
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids[:8]
    if len(tokens) != 8:
        raise ValueError("teacher-forced fixture requires eight model tokens")
    offset = 4
    fixture = {"start": np.array([offset], dtype=np.uint32)}
    first = Layer(weights, 0, activate)
    embedding = weights.read(MODEL_PREFIX + "embed.weight")[tokens]
    embedding = (embedding.astype(np.uint32) << 16).view(np.float32)
    partial = rms(embedding, first.model_tensor(MODEL_PREFIX + "embed_norm.weight"))
    blocks = np.empty((len(tokens), 0, WIDTH), dtype=np.float32)
    previous = None
    empty_cache = np.empty((0, KV_HEADS * HEAD_DIM), dtype=np.float32)
    empty_tails = [np.zeros((3, width), dtype=np.float32) for width in CONV_WIDTHS]
    for number in (*range(18), *range(6, 12)):
        if number == 6:
            if blocks.shape[1] == 1:
                prelude = blocks, partial
            else:
                previous = np.concatenate(
                    (blocks[:, prelude[0].shape[1] + 1 :], partial[:, None, :]), axis=1
                )
                blocks, partial = prelude
        layer = Layer(weights, number, activate, project)
        result = layer.visit(
            blocks, partial, previous, empty_cache, empty_cache, empty_tails, 0
        )
        if previous is not None and number in (10, 11):
            if number == 10:
                fixture["blocks"] = blocks[offset:]
                fixture["partial"] = partial[offset:]
                fixture["previous"] = bf16_round(previous[offset:])
            keys, values = (
                result[name][:offset] for name in ("key_rows", "value_rows")
            )
            tails = np.split(
                result["conv_rows"][offset - 3 : offset],
                np.cumsum(CONV_WIDTHS)[:-1],
                axis=1,
            )
            input_state(fixture, layer, keys, values, tails, offset)
            fixture.update(
                (f"{number}.{name}", value[offset:]) for name, value in result.items()
            )
        if number % 6 == 0:
            blocks = np.concatenate((blocks, partial[:, None, :]), axis=1)
        partial = result["partial"]
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("activation_source", type=Path)
    parser.add_argument(
        "--workloads",
        type=Path,
        help="Trace the first private workload through eight teacher-forced tokens",
    )
    parser.add_argument(
        "--mlx-projections",
        action="store_true",
        help="Use eight-row Python MLX dense projections as a diagnostic control",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Private .safetensors output path in an ignored directory",
    )
    args = parser.parse_args()
    os.umask(0o077)
    weights = Weights(args.model)
    activate = activation_kernel(args.activation_source)
    project = mlx_linear if args.mlx_projections else linear
    fixture = (
        trace_prompt(weights, activate, args.workloads, project)
        if args.workloads
        else make_fixture(weights, activate, project)
    )
    for name, value in fixture.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"nonfinite fixture stage {name}")
    tensors = {
        name: (
            mx.array(value)
            if name == "start"
            or name.endswith(
                (
                    ".routes",
                    ".messages",
                    ".up_inner",
                    ".up_max",
                    ".down_inner",
                    ".route_weights",
                )
            )
            else mx.array(value).astype(mx.bfloat16)
        )
        for name, value in fixture.items()
    }
    mx.save_safetensors(str(args.output), tensors)


if __name__ == "__main__":
    main()
