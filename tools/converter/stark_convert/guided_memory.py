"""Metadata-only allocation inventories for guided hybrid setup.

The native inventories follow engine/hybrid/{flint,glm,qwen}.rs and the
Stark-authored request inventory in vendor/lily-core/src/qwen4exp/request_memory.rs.
Keep source-bound profile results and update their validation when these change.
These are planning bounds, not measured RSS or task-quality evidence.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from math import prod

GIB = 1024**3
MIB = 1024**2


def config_identity(config):
    """Preparation changes the codec, marker and EOS union, not execution geometry."""
    config = deepcopy(config)
    config.pop("token_codec_config", None)
    config.pop("stark_native_hybrid", None)
    config.pop("eos_token_id", None)
    if isinstance(config.get("text_config"), dict):
        config["text_config"].pop("eos_token_id", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def qwen35_geometry(text):
    """Fail closed on incomplete DeltaNet schedules and native head geometry."""
    keys = ("linear_num_value_heads", "linear_num_key_heads", "linear_conv_kernel_dim")
    if any(type(text.get(key)) is not int or not 1 <= text[key] <= 256 for key in keys):
        return None
    if any(text.get(key, 128) != 128 for key in ("linear_key_head_dim", "linear_value_head_dim")):
        return None
    count = text.get("num_hidden_layers")
    if type(count) is not int or not 1 <= count <= 512:
        return None
    types = text.get("layer_types")
    if types is None:
        interval = text.get("full_attention_interval")
        if type(interval) is not int or not 1 <= interval <= count:
            return None
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    if (
        not isinstance(types, list)
        or len(types) != count
        or any(not isinstance(kind, str) for kind in types)
        or not set(types) <= {"linear_attention", "full_attention"}
    ):
        return None
    attention = types.count("full_attention")
    if not attention or attention == count:
        return None
    mtp = text.get("mtp_num_hidden_layers", 0)
    if type(mtp) is not int or not 0 <= mtp <= 1:
        return None
    heads, groups = (text[key] for key in keys[:2])
    if heads % groups:
        return None
    conv = (2 * groups + heads) * 128
    # F32 recurrence/convolution, replaced prefix snapshots and the four-row
    # MTP verifier. Scratch's 2 GiB cached-page allowance is added by Recipe.
    recurrent = (count - attention) * 8 * (heads * 128 * 128 + conv * (text[keys[2]] - 1)) * 4
    return {
        "attention_layers": attention + mtp,
        "recurrent_state_bytes": recurrent,
        "recurrent_workspace_bytes": 2 * GIB,
    }


def _host(context, vocab):
    return context * 48 + vocab * 64 + 8 * MIB


def _flint(c, context):
    rows = sum(
        min(context, c["sliding_window"]) if kind == "sliding_attention" else context for kind in c["layer_types"]
    )
    cache = rows * (2 * c["num_key_value_heads"] * c["head_dim"] * 4 + 48) * 2
    width = max(
        c["hidden_size"],
        c["intermediate_size"],
        c["moe_intermediate_size"] * c["num_experts_per_tok"],
        c["shared_expert_intermediate_size"],
        c["num_attention_heads"] * c["head_dim"],
    )
    return cache + width * 256 + 512 * MIB + _host(context, c["vocab_size"])


def _glm(c, context):
    layers = c["num_hidden_layers"]
    kda_layers = c["layer_types"].count("linear_attention")
    attention_layers = layers - kda_layers
    hidden, streams = c["hidden_size"], c["hc_mult"]
    rows, columns = (2 + streams) * streams, hidden * streams
    controls = 2 * layers * (rows * columns + columns + 2 * hidden + 34 * rows + 3)
    linear = c["linear_attn_config"]
    inner, dim, kernel = linear["num_heads"] * linear["head_dim"], linear["head_dim"], linear["short_conv_kernel_size"]
    kda = kda_layers * (
        3 * inner * (kernel - 1)
        + inner * dim
        + 3 * inner * kernel
        + inner
        + dim
        + linear["num_heads"]
        + 6 * inner
        + linear["num_heads"]
        + 1
    )
    pool, index = c["index_kpool"], c["index_head_dim"]
    pooled_rows = (context + pool - 1) // pool
    attention = attention_layers * (
        2 * (context * c["kv_lora_rank"] + pooled_rows * index) + 3 * pool * index + context
    )
    outside = hidden * streams + hidden + (layers - c["first_k_dense_replace"]) * c["n_routed_experts"]
    # Cover two graph owners and 8-row external score planes, including the
    # overlap policy. We do not discount mapped weights using a per-encoder cap.
    device = 4 * (controls + kda + attention + outside) + 2 * 512 * MIB + 8 * MIB
    host = attention_layers * context * (c["kv_lora_rank"] * 4 + 72) * 2
    host += attention_layers * pooled_rows * index * 8
    external = 2 * attention_layers * 8 * c["num_attention_heads"] * context * 4
    return device + host + external + _host(context, c["vocab_size"])


def _qwen_scalar(c, context):
    layers, attention = c["num_hidden_layers"], c["layer_types"].count("full_attention")
    key_width, index = c["num_key_value_heads"] * c["head_dim"], c["indexer_head_dim"]
    host_row = 2 * key_width * 4 + index * 4 + 72
    graph_row = 2 * key_width * 4 + index * 4 + 2 * c["num_attention_heads"] * 4 + 4
    cache = attention * context * (host_row + graph_row) * 2
    pooled = attention * max(1, context // c["indexer_compress_ratio"]) * (index + 1) * 8
    recurrent = layers * c["linear_num_value_heads"] * c["linear_value_head_dim"] * c["linear_key_head_dim"] * 16
    conv_width = (
        2 * c["linear_num_key_heads"] * c["linear_key_head_dim"]
        + c["linear_num_value_heads"] * c["linear_value_head_dim"]
    )
    conv = layers * c["linear_conv_kernel_dim"] * conv_width * 16
    ple = layers * c["hidden_size"] * c["hc_count"] * c["ple_conv_kernel_size"] * c["ngram_size"] * 16
    width = max(
        c["hidden_size"] * c["hc_count"],
        c["moe_intermediate_size"] * c["num_experts_per_tok"],
        c["shared_expert_intermediate_size"],
        c["num_attention_heads"] * c["head_dim"],
        c["indexer_n_heads"] * index,
    )
    return cache + pooled + recurrent + conv + ple + width * 256 + 512 * MIB + _host(context, c["vocab_size"])


def _qwen_mixed(c, context):
    """Inventory the loaded MTP head as well as live/replaced prefix snapshots.

    Uses the larger Q8 scale allowance for both supported KV layouts, the
    production 4096-row prefill chunk and the native inventory's growth peak.
    """
    h, streams, low = c["hidden_size"], c["hc_count"], c["hc_lowrank"]
    wide = h * streams
    nq, nkv, hd, nh = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["indexer_n_heads"]
    heads, hk = c["linear_num_value_heads"], c["linear_num_key_heads"]
    dim_v = heads * c["linear_value_head_dim"]
    conv = 2 * hk * c["linear_key_head_dim"] + dim_v
    inter, expert, active = c["shared_expert_intermediate_size"], c["moe_intermediate_size"], c["num_experts_per_tok"]
    ratio, budget, idx = c["indexer_compress_ratio"], c["indexer_budget"], 128
    mtp = bool(c.get("mtp"))
    attention = c["layer_types"].count("full_attention") + int(mtp)
    gdn = c["layer_types"].count("linear_attention")
    m = min(4096, context)
    total = 0

    def add(shape, element=2, copies=1):
        nonlocal total
        size = prod(shape) * element
        total += ((size + 255) // 256 * 256) * copies

    def hc(rows):
        for width, copies in ((wide, 3), (low, 2), (h, 1), (streams, 1)):
            add([rows, width], copies=copies)
        add([rows, streams], 4)
        add([min(rows, 4), wide], 4)

    def ple(rows):
        add([rows, (c["ngram_size"] - 1) * c["heads_per_ngram"]], 4, 3)
        add([rows, c["ple_embed_dim"]])
        add([rows, wide], copies=5)
        add([rows, h])
        add([rows, c["ple_embed_dim"]], 4, 2)

    def qsa(rows):
        splits = max(rows * ((budget + ratio - 1 + 255) // 256), min(rows, 4) * ((budget + ratio - 1 + 31) // 32))
        add([rows, max(1, (context + ratio - 1) // ratio)], 4)
        add([rows, budget // ratio], 4)
        add([rows], 4)
        add([splits, nq, hd], 4)
        add([splits, nq, 2], 4)

    def moe(rows, prefill):
        experts, routed = c["num_experts"], rows * active
        add([rows, experts], 4)
        add([rows, active], 4, 4)
        add([rows, h])
        add([rows], 4)
        add([65536])
        if prefill:
            add([min(rows, 4), experts], 4)
            add([min(rows, 4)], 4)
            add([experts + 1], 4, 4)
            add([routed], 4)
            tiles = routed + min(experts, routed)
            add([tiles, (expert + 63) // 64, 4], 4)
            add([tiles, (h + 63) // 64, 4], 4)
            add([routed, h], copies=2)
            add([routed, expert], copies=3)
        else:
            add([active, expert])

    # Persistent state, with live state plus two retained/replaced snapshots.
    add([nkv, context, hd], 1, 2 * attention)
    add([nkv, context, hd // 32], 2, 2 * attention)
    add([context, idx], copies=attention)
    add([max(1, context // ratio), idx], copies=attention)
    add([heads, 128, 128], 4, 3 * gdn)
    add([conv, c["linear_conv_kernel_dim"] - 1], copies=4 * gdn)
    add([h, streams, (c["ple_conv_kernel_size"] - 1) * c["ngram_size"]], copies=4)
    if mtp:
        add([h, streams], copies=3)
    for width, copies in (
        (h, 2),
        (wide, 1),
        (conv + dim_v + 2 * heads, 1),
        ((2 * nq + 2 * nkv) * hd, 1),
        (2 * inter, 1),
        (inter, 1),
        (conv, 1),
        (dim_v, 2),
        (nq * hd, 4),
        ((2 * nh + 1) * idx, 1),
    ):
        add([width], copies=copies)
    hc(1)
    ple(1)
    qsa(1)
    moe(1, False)
    add([c["vocab_size"]], 4, 3)
    # Upper-bound dense SDPA split scratch independently of kernel selection.
    add([nq, budget + ratio, hd + 2], 4)
    add([MIB], 1)
    largest = max(
        conv * h,
        dim_v * h,
        heads * h,
        h * dim_v,
        2 * nq * hd * h,
        nkv * hd * h,
        h * nq * hd,
        c["num_experts"] * h,
        expert * h,
        inter * h,
        low * wide,
        streams * wide,
        (nh + 1) * idx * h,
        wide * c["ple_embed_dim"],
        h * c["ple_embed_dim"],
    )
    add([largest])
    hc(m)
    ple(m)
    moe(m, True)
    qsa(min(256, m))
    for width, copies in (
        (h, 2),
        (wide, 3 if mtp else 1),
        (inter, 3),
        (conv, 2),
        (dim_v, 3),
        (heads, 2),
        (2 * hk * 128, 1),
        (2 * nq * hd, 1),
        (nq * hd, 4),
        (nkv * hd, 2),
        ((2 * nh + 1) * idx, 1),
    ):
        add([m, width], copies=copies)
    add([m, heads], 4, 3)
    add([m, heads, 128], copies=2)
    add([m, heads, 64])
    add([m, max(conv + dim_v + 2 * heads, (2 * nq + 2 * nkv) * hd, 2 * inter)])
    add([m], 4)
    add([m + ratio, 3], 4)
    add([nkv, budget + ratio - 1, hd], copies=2)
    if mtp:
        add([3, heads, 128, 128], 4, gdn)
        add([4, conv], copies=gdn)
        add([4, wide])
        add([4, c["vocab_size"]], 4)
        add([4 * MIB], 1)
    return 2 * total + _host(context, c["vocab_size"]) + context * 32 + c["vocab_size"] * 64 + 8 * MIB


def native_request_bytes(choice, geometry, context=32768):
    # Fixed, revision-bound geometries only; this is not generic source admission.
    if context <= 0 or context > geometry["max_position_embeddings"]:
        raise ValueError("invalid native planning context")
    context = 1 << (context - 1).bit_length()
    if choice == "flint-int4":
        value = _flint(geometry, context)
    elif choice == "glm53-q2":
        value = _glm(geometry, context)
    elif choice == "qwen38-mixed":
        value = _qwen_scalar(geometry, context)
    elif choice in {"qwen38-lily-q4-xl", "qwen38-mlx-serve-mixed"}:
        value = _qwen_mixed(geometry, context)
    else:
        raise ValueError("unknown native memory policy")
    if not 0 < value < 2**63:
        raise ValueError("invalid native memory inventory")
    return value
