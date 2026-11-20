"""Independent CPU reconstruction of the first original-checkpoint routed inputs.

Follows the local-layer equations in the pinned Ollama Inkling2 implementation.
The CPU matmuls and reductions need not be bit-identical to Metal. All numerical
data remains in memory. The only return values are expert inputs and route IDs.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def bf16_round(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def rms(value, weight=None):
    unit = value / np.sqrt((value * value).mean(axis=-1, keepdims=True) + np.float32(1e-6))
    return unit if weight is None else bf16_round(unit * weight)


def clip_l2(value, limit, weight=None):
    scaled = value if weight is None else value * weight
    norm = np.sqrt((scaled * scaled).sum(axis=-1, keepdims=True))
    return bf16_round(scaled * np.minimum(1, limit / (norm + np.float32(1e-6))))


def short_conv(value, weight):
    taps = weight.shape[-1]
    padded = np.pad(value, ((taps - 1, 0), (0, 0)))
    update = np.zeros_like(value)
    for tap in range(taps):
        update += padded[tap:tap + len(value)] * weight[:, 0, tap]
    return bf16_round((value + update) / np.float32(math.sqrt(taps + 1)))


def layer0_fixtures(official_safetensors_path: Path, workloads_path=None, *, token_count=8):
    if workloads_path is None:
        workloads_path = Path(__file__).resolve().parents[4] / ".inkling-local/workloads.json"
    payload = json.loads(Path(workloads_path).read_text())["cases"][0]["prompt"]
    tokenizer = Tokenizer.from_file(str(official_safetensors_path.parent / "tokenizer.json"))
    all_ids = tokenizer.encode(payload, add_special_tokens=False).ids
    if not 1 <= token_count <= min(len(all_ids), 1024):
        raise ValueError("fixture length must fit the prompt and first layer's local attention window")
    ids = all_ids[:token_count]
    with official_safetensors_path.open("rb") as source:
        header_bytes = struct.unpack("<Q", source.read(8))[0]
        tensors = json.loads(source.read(header_bytes))

    def weight(name, selected=None):
        entry = tensors["model.llm." + name]
        dtype = {"BF16": "<u2", "F32": "<f4"}[entry["dtype"]]
        matrix = np.memmap(official_safetensors_path, dtype=dtype, mode="r",
                           offset=8 + header_bytes + entry["data_offsets"][0],
                           shape=tuple(entry["shape"]))
        if selected is not None:
            matrix = matrix[selected]
        if dtype == "<u2":
            return (np.asarray(matrix, dtype=np.uint32) << 16).view(np.float32)
        return np.asarray(matrix)

    def layer(name):
        return weight("layers.0." + name)

    def projection(inputs, name):
        return bf16_round(np.einsum("ni,oi->no", inputs, layer("attn." + name + ".weight")))

    n = len(ids)
    embedded = rms(weight("embed.weight", ids), weight("embed_norm.weight"))
    normalized = rms(embedded, layer("attn_norm.weight"))
    q = clip_l2(projection(normalized, "wq_du").reshape(n, 24, 128),
                np.float32(math.sqrt(512)), layer("attn.q_l2_clip.scale"))
    k = short_conv(projection(normalized, "wk_dv"), layer("attn.k_sconv.weight"))
    k = clip_l2(k.reshape(n, 8, 128), np.float32(math.sqrt(512))).repeat(3, axis=1)
    v = short_conv(projection(normalized, "wv_dv"), layer("attn.v_sconv.weight"))
    v = bf16_round(rms(v.reshape(n, 8, 128))).repeat(3, axis=1)
    relative = clip_l2(projection(normalized, "wr_du").reshape(n, 24, 16),
                       np.float32(8), layer("attn.rel_l2_clip.scale"))
    biases = bf16_round(np.einsum("thr,rd->thd", relative, layer("attn.rel_logits_proj.proj")[:, :n]))
    sinks = projection(normalized, "attn_sink_proj")
    attended = np.empty_like(q)
    for pos in range(n):
        scores = np.einsum("hd,khd->kh", q[pos], k[:pos + 1]) / np.float32(math.sqrt(128))
        scores += biases[pos, :, :pos + 1].T[::-1] - sinks[pos]
        peaks = np.maximum(scores.max(axis=0), np.float32(0))
        probabilities = np.exp(scores - peaks)
        denominator = probabilities.sum(axis=0) + np.exp(-peaks)
        attended[pos] = np.einsum("kh,khd->hd", probabilities / denominator, v[:pos + 1])
    gate = projection(normalized, "gate_attn")
    gated = bf16_round(attended.reshape(n, -1) / (1 + np.exp(-gate)))
    attn = short_conv(projection(gated, "wo_ud"), layer("attn_sconv.weight"))
    keys = rms(np.stack((embedded, attn), axis=1))
    query = weight("attn_res.mlp_queries.0") / np.float32(math.sqrt(2304))
    logits = np.einsum("nhd,d->nh", keys, query)
    mixing = np.exp(logits - logits.max(axis=1, keepdims=True))
    mixing /= mixing.sum(axis=1, keepdims=True)
    ffn = rms((keys * mixing[:, :, None]).sum(axis=1), layer("mlp_norm.weight"))
    route_logits = np.einsum("ni,ei->ne", ffn, layer("mlp.gate.weight"))
    selection = 1 / (1 + np.exp(-route_logits)) + layer("mlp.gate.bias")
    top = np.argsort(-selection, axis=1, kind="stable")[:, :8]
    if not np.isfinite(ffn).all() or not np.isfinite(selection).all():
        raise ValueError("nonfinite first-layer CPU reconstruction")
    return ffn.reshape(n, 1, -1), top.astype(np.int32)
