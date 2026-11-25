#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resident MLX draft process for the Stark Inkling DFlash 2 pilot.

The model graph is imported from z-lab/dflash at the pinned revision below.
Target embedding, candidate scoring, and the feature contract follow the
source-bound Inkling DFlash 2 trace experiment. Standard I/O carries only
the private worker protocol.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import mmap
import resource
import struct
import subprocess
import sys
from pathlib import Path


REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"
MLX_VERSION = "0.31.2"
MLX_LM_VERSION = "0.31.3"
HIDDEN = 2304
TAPS = (5, 23, 41, 48)
FEATURE_WORDS = len(TAPS) * HIDDEN
VOCAB = 200_058
PADDED_VOCAB = 201_024
WINDOW = 2047
BLOCK_SIZE = 8
RESET = 1
PROPOSE = 2
QUIT = 3
HEADER = struct.Struct("<4I")
STATUS = struct.Struct("<I")
PROPOSALS = struct.Struct("<7I")


def bind_target(draft, target: Path):
    import mlx.core as mx
    import numpy as np
    from mlx import nn

    config_path = target.with_name("source-config.json")
    if not config_path.is_file():
        config_path = target.with_name("config.json")
    config = json.loads(config_path.read_bytes())
    config = config.get("text_config", config)
    if int(config.get("unpadded_vocab_size", config["vocab_size"])) != VOCAB:
        raise ValueError("invalid target vocabulary")

    def tensor(key, shape):
        entry = tensors[key]
        if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape:
            raise ValueError("invalid target tensor")
        start, end = entry["data_offsets"]
        if not 0 <= start < end <= len(mapping) - payload_start:
            raise ValueError("invalid target tensor offsets")
        if end - start != 2 * int(np.prod(shape)):
            raise ValueError("invalid target tensor size")
        words = np.frombuffer(mapping, dtype="<u2", count=(end - start) // 2, offset=payload_start + start)
        return mx.array(words.reshape(shape)).view(mx.bfloat16)

    class InklingEmbedding(nn.Module):
        def __init__(self, weights, norm):
            super().__init__()
            self.tokens = nn.Embedding(PADDED_VOCAB, HIDDEN)
            self.tokens.weight = weights
            self.norm = nn.RMSNorm(HIDDEN, eps=float(config["rms_norm_eps"]))
            self.norm.weight = norm

        def __call__(self, ids):
            return self.norm(self.tokens(ids))

    with target.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_size <= 100_000_000:
            raise ValueError("invalid target header size")
        tensors = json.loads(handle.read(header_size))
        payload_start = 8 + header_size
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            weights = tensor("model.llm.embed.weight", (PADDED_VOCAB, HIDDEN))
            norm = tensor("model.llm.embed_norm.weight", (HIDDEN,))
            head = tensor("model.llm.unembed.weight", (PADDED_VOCAB, HIDDEN))
            mx.eval(weights, norm, head)
    draft.embed_tokens = InklingEmbedding(weights, norm)
    draft.lm_head = nn.Linear(HIDDEN, PADDED_VOCAB, bias=False)
    draft.lm_head.weight = head


def load_model(directory: Path, target: Path, repository: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("invalid DFlash source revision")
    subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet", REVISION, "--", "dflash"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    sys.path.insert(0, str(repository.resolve()))

    import mlx.core as mx
    from dflash.model_mlx import DFlash2DraftModel, DFlashConfig

    if (
        importlib.metadata.version("mlx") != MLX_VERSION
        or importlib.metadata.version("mlx-lm") != MLX_LM_VERSION
        or not mx.metal.is_available()
    ):
        raise ValueError("Inkling DFlash requires the pinned MLX GPU runtime")
    mx.set_default_device(mx.gpu)
    config = json.loads((directory / "config.json").read_bytes())
    draft_config = config["dflash_config"]
    if (
        config["hidden_size"] != HIDDEN
        or config["vocab_size"] != PADDED_VOCAB
        or config["sliding_window"] != WINDOW + 1
        or config["layer_types"] != ["sliding_attention"] * config["num_hidden_layers"]
        or tuple(draft_config["target_layer_ids"]) != TAPS
        or draft_config["block_size"] != BLOCK_SIZE
        or float(draft_config.get("output_multiplier", 1.0)) != 1.0
        or draft_config["selector_softcap_mode"] != "after_correction"
    ):
        raise ValueError("invalid DFlash checkpoint geometry")
    model = DFlash2DraftModel(DFlashConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
        vocab_size=PADDED_VOCAB,
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=config["rope_parameters"]["rope_theta"],
        max_position_embeddings=config["max_position_embeddings"],
        block_size=BLOCK_SIZE,
        target_layer_ids=TAPS,
        num_target_layers=config["num_target_layers"],
        mask_token_id=draft_config["mask_token_id"],
        rope_scaling=config["rope_parameters"],
        layer_types=tuple(config["layer_types"]),
        sliding_window=WINDOW + 1,
        final_logit_softcapping=draft_config["final_logit_softcapping"],
        conv_kernel_size=draft_config["conv_kernel_size"],
        conv_group_size=draft_config["conv_group_size"],
        selector_rank=draft_config["selector_rank"],
        selector_top_k=draft_config["selector_top_k"],
        is_causal=config["is_causal"],
    ))
    weights = mx.load(str(directory / "draft.safetensors"))
    if any(weight.dtype != mx.bfloat16 for weight in weights.values()):
        raise ValueError("Inkling DFlash checkpoint tensors must use BF16")
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        weights[key + ".weight"] = weights.pop(key)
    model.load_weights(list(weights.items()))
    bind_target(model, target)
    model.eval()
    mx.eval(model.parameters())
    return model


class DraftContext:
    """Per-layer chronological K/V over committed target features only."""

    def __init__(self, offset):
        self.offset = offset
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        count = keys.shape[2]
        if self.keys is None:
            self.keys, self.values = keys, values
        elif count:
            keep = max(0, self.keys.shape[2] + count - WINDOW)
            self.keys = mx.concatenate((self.keys[:, :, keep:], keys), axis=2)
            self.values = mx.concatenate((self.values[:, :, keep:], values), axis=2)
        self.offset += count
        return self.keys, self.values


def select_path(model, hidden, anchor):
    import mlx.core as mx

    selector = model.candidate_selector
    logits = model.lm_head(hidden)[..., :VOCAB].astype(mx.float32)
    top_k = selector.top_k
    candidates = mx.argpartition(logits, -top_k, axis=-1)[..., -top_k:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden).astype(mx.float32)
    cap = float(model.config.final_logit_softcapping)
    predecessor = anchor
    path = []
    for position in range(BLOCK_SIZE - 1):
        pred = selector.predecessor_codebook(predecessor).astype(mx.float32)[:, None, :]
        succ = selector.successor_codebook(candidates[:, position]).astype(mx.float32)
        edge = mx.sum(pred * projected[:, position, None, :] * succ, axis=-1)
        scores = cap * mx.tanh((unary[:, position] + edge) / cap)
        chosen = mx.argmax(scores, axis=-1)
        predecessor = mx.take_along_axis(candidates[:, position], chosen[:, None], axis=-1)[:, 0]
        path.append(predecessor)
    return mx.stack(path, axis=1)


def serve(model, source, sink):
    import mlx.core as mx
    import numpy as np

    cache = None
    sink.write(STATUS.pack(0))
    sink.flush()
    while message := source.read(HEADER.size):
        opcode, start, rows, anchor = HEADER.unpack(message)
        if opcode in (RESET, QUIT) and (start or rows or anchor):
            raise ValueError("invalid draft control message")
        if opcode == QUIT:
            return
        if opcode == RESET:
            cache = None
            sink.write(STATUS.pack(0))
            sink.flush()
            continue
        if opcode != PROPOSE or rows > WINDOW or anchor >= VOCAB:
            raise ValueError("invalid draft message")
        if start + rows + BLOCK_SIZE > model.config.max_position_embeddings:
            raise ValueError("draft position exceeds configured maximum")
        if cache is None or cache[0].offset != start:
            if start and rows != WINDOW:
                raise ValueError("draft restart requires a complete context window")
            cache = [DraftContext(start) for _ in model.layers]
        payload = source.read(rows * FEATURE_WORDS * 2)
        features = np.frombuffer(payload, dtype="<u2").reshape(1, rows, FEATURE_WORDS)
        context = mx.array(features).view(mx.bfloat16)
        block = mx.array([[anchor] + [model.config.mask_token_id] * (BLOCK_SIZE - 1)], dtype=mx.int32)
        hidden = model.hidden_states(block, context, cache, logits_start=1)
        path = select_path(model, hidden, block[:, 0])[0].tolist()
        sink.write(STATUS.pack(0) + PROPOSALS.pack(*path))
        sink.flush()


def fail(sink, status, message):
    try:
        sink.write(STATUS.pack(status))
        sink.flush()
    except OSError:
        pass
    print(message, file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dflash-repo", type=Path, required=True)
    args = parser.parse_args()
    sink = sys.stdout.buffer
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        model = load_model(args.draft, args.target, args.dflash_repo)
    except Exception:
        return fail(sink, 1, "Inkling DFlash 2 companion failed to load")
    try:
        serve(model, sys.stdin.buffer, sink)
    except (BrokenPipeError, KeyboardInterrupt):
        return 1
    except Exception:
        return fail(sink, 2, "Inkling DFlash 2 companion request failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
