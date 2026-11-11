"""Pinned installed-only candidates for the beginner model guide.

This module only admits the already prepared Lily Qwen3.8 text artifact. It
does not discover, download, convert, or qualify another model/backend.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

from ._build_metadata import open_regular
from .hybrid_prepare import SOURCES, prepared_identity
from .model_inventory import _small_file, model_tool_support
from .streaming import MAX_HEADER_BYTES, ConversionError, read_safetensors_header

GIB = 1024**3
TARGET_CONTEXT = 32768
MIN_CONTEXT = 32768
WORKSPACE_RESERVE = 8 * GIB

_SOURCE = SOURCES["qwen38-lily-q4-xl"]
_EXPECTED_LAYERS = [kind for _ in range(12) for kind in ("linear_attention",) * 3 + ("full_attention",)]


def _configuration_matches(config):
    """Require the exact source geometry covered by the planner below."""
    if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
        return False
    if not isinstance(config.get("lily"), dict) or config["lily"].get("format") != "qwen4_exp-affine-v1":
        return False
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "qwen4_exp_text":
        return False
    expected = {
        "hidden_size": 2560,
        "vocab_size": 248320,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "num_hidden_layers": 48,
    }
    if any(type(text.get(key)) is not int or text[key] != value for key, value in expected.items()):
        return False
    if type(text.get("max_position_embeddings")) is not int or not (
        MIN_CONTEXT <= text["max_position_embeddings"] <= 1_048_576
    ):
        return False
    if text.get("layer_types") != _EXPECTED_LAYERS or text.get("ple_layer_ids") != [2]:
        return False
    if text.get("split_ngram_parts") != 128:
        return False
    lily = config.get("lily")
    ple = lily.get("ple") if isinstance(lily, dict) else None
    ngram_quant = lily.get("quantization", {}).get("ngram_embedding") if isinstance(lily, dict) else None
    if (
        not isinstance(ple, dict)
        or ple.get("padded_vocab_size") != 320_001_536
        or not isinstance(ngram_quant, dict)
        or ngram_quant != {"bits": 4, "group_size": 32, "mode": "affine"}
    ):
        return False
    mtp = text.get("mtp")
    if (
        not isinstance(mtp, dict)
        or mtp.get("layer_types") != ["full_attention"]
        or type(mtp.get("num_hidden_layers")) is not int
        or mtp["num_hidden_layers"] != 1
    ):
        return False
    return True


def _count_paged_table_bytes(tensors, config):
    """Validate the PagedTable's three planes per shard and sum their bytes."""
    text = config["text_config"]
    lily = config["lily"]
    ple = lily["ple"]
    shards = text["split_ngram_parts"]
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    if heads == 0 or text["ple_embed_dim"] % heads:
        raise ValueError()
    width = text["ple_embed_dim"] // heads
    group_size = lily["quantization"]["ngram_embedding"]["group_size"]
    if width % group_size or width % 8:
        raise ValueError()
    rows_per_shard, remainder = divmod(ple["padded_vocab_size"], shards)
    if remainder or rows_per_shard == 0:
        raise ValueError()
    prefix = ".layers.1.ple.ple_embedding.ngram_embedding.shard_"
    pattern = re.compile(
        r"^(.*\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_)([0-9]+)\.(weight|scales|biases)$"
    )
    inventory = {}
    for name, tensor in tensors.items():
        if ".ple_embedding.ngram_embedding.shard_" not in name:
            continue
        match = pattern.fullmatch(name)
        if match is None or not match[1].endswith(prefix):
            raise ValueError()
        shard, plane = int(match[2]), match[3]
        if not 0 <= shard < shards or (shard, plane) in inventory:
            raise ValueError()
        inventory[(shard, plane)] = tensor
    if len(inventory) != shards * 3:
        raise ValueError()
    table_bytes = 0
    expected_words, expected_groups = width // 8, width // group_size
    for shard in range(shards):
        codes, scales, biases = (inventory[(shard, plane)] for plane in ("weight", "scales", "biases"))
        if (
            codes.dtype != "U32"
            or scales.dtype != "BF16"
            or biases.dtype != "BF16"
            or len(codes.shape) != 2
            or codes.shape != (rows_per_shard, expected_words)
            or scales.shape != (rows_per_shard, expected_groups)
            or biases.shape != scales.shape
        ):
            raise ValueError()
        table_bytes += codes.nbytes + scales.nbytes + biases.nbytes
    expected_bytes = ple["padded_vocab_size"] * (width // 2 + expected_groups * 4)
    if table_bytes != expected_bytes:
        raise ValueError()
    return table_bytes


def _paged_table_weight_bytes(root, config):
    """Read bounded safetensors headers only; never touch the n-gram payload."""
    files = sorted(root.glob("*.safetensors"))
    if len(files) != _SOURCE.get("weight_shards"):
        raise ValueError()
    total_header_bytes = 0
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError()
        with open_regular(path) as stream:
            prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError()
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= MAX_HEADER_BYTES:
            raise ValueError()
        total_header_bytes += header_size
        if total_header_bytes > 16 * 1024 * 1024:
            raise ValueError()
    table_tensors = {}
    for path in files:
        _metadata, tensors = read_safetensors_header(path)
        for name, tensor in tensors.items():
            if ".ple_embedding.ngram_embedding.shard_" in name:
                if name in table_tensors:
                    raise ValueError()
                table_tensors[name] = tensor
    return _count_paged_table_bytes(table_tensors, config)


def _runtime_memory(weight_bytes, text, context_tokens, *, gpu_weight_bytes=None):
    """Conservative host-memory bound for Lily's Q8, paged-table path.

    We count all prepared safetensors bytes as resident, including the paged
    n-gram table. Cache planning deliberately charges F32 for K/V, index keys,
    and compressed block keys, then doubles the cache to cover growth. GDN
    recurrent state is F32; two prefix snapshots are charged alongside live
    state to cover replacement while an older snapshot is retained.
    """
    layers = text["layer_types"]
    # Account for the optional MTP head even when the current UI preference is
    # off: the worker can enable it after load.
    attention_layers = sum(kind == "full_attention" for kind in layers) + 1
    recurrent_layers = sum(kind == "linear_attention" for kind in layers)
    kv_heads, head_dim = text["num_key_value_heads"], text["head_dim"]
    index_dim, ratio = text["indexer_head_dim"], text["indexer_compress_ratio"]

    # F32 KV rows, BF16 index rows charged as F32, and compressed index rows.
    # Charge all of these as 4-byte elements to avoid relying on current policy.
    cache_per_token = 2 * kv_heads * head_dim * 4 + index_dim * 4
    compressed_rows = (context_tokens + ratio - 1) // ratio
    compressed_bytes = compressed_rows * index_dim * 4
    one_cache = attention_layers * (context_tokens * cache_per_token + compressed_bytes)
    cache_bytes = 2 * one_cache  # Old and grown capacities may coexist.

    # F32 recurrent matrices, plus BF16 convolution windows charged at F32.
    value_heads = text["linear_num_value_heads"]
    value_dim, key_dim = text["linear_value_head_dim"], text["linear_key_head_dim"]
    conv_channels = 2 * text["linear_num_key_heads"] * key_dim + value_heads * value_dim
    conv_bytes = conv_channels * (text["linear_conv_kernel_dim"] - 1) * 4
    state_bytes = value_heads * value_dim * key_dim * 4
    gdn_state = recurrent_layers * (state_bytes + 2 * conv_bytes)

    # Prefix snapshot captures one recurrent matrix and one convolution window
    # per GDN layer, plus the PLE window. Two snapshots cover replacement peak.
    ple_layers = len(text["ple_layer_ids"])
    ple_window = text["hc_count"] * text["hidden_size"] * ((text["ple_conv_kernel_size"] - 1) * text["ngram_size"]) * 4
    one_snapshot = recurrent_layers * (state_bytes + conv_bytes) + ple_layers * ple_window
    mtp_hidden = text["hc_count"] * text["hidden_size"] * 2
    recurrent_and_snapshots = gdn_state + 2 * one_snapshot + 2 * ple_layers * ple_window + 3 * mtp_hidden

    # MTP verifies at most three draft tokens. Charge three extra F32 recurrent
    # matrices and four input rows of GDN/PLE convolution captures per layer.
    draft_state = 3 * recurrent_layers * state_bytes
    draft_conv = 4 * recurrent_layers * conv_channels * 4
    draft_ple = 4 * ple_layers * text["hc_count"] * text["hidden_size"] * 4
    recurrent_and_snapshots += draft_state + draft_conv + draft_ple

    variable = cache_bytes + recurrent_and_snapshots + WORKSPACE_RESERVE
    physical_total = weight_bytes + variable
    gpu_total = (weight_bytes if gpu_weight_bytes is None else gpu_weight_bytes) + variable
    return (
        ((physical_total + GIB - 1) // GIB) * GIB,
        ((gpu_total + GIB - 1) // GIB) * GIB,
    )


def installed_qwen38_candidate(path):
    """Return a planner record only for an exact, tool-compatible pinned install.

    ``None`` means the artifact is absent, malformed, unsupported, or not the
    one pinned Lily source. The lightweight receipt/config checks do not
    replace native model loading or qualify model quality/task success.
    """
    try:
        root = Path(path)
        identity = prepared_identity(root)
        if identity is None or any(
            identity.get(key) != _SOURCE[value]
            for key, value in (
                ("repo", "weights_repo"),
                ("revision", "weights_revision"),
                ("architecture", "architecture"),
                ("weight_bytes", "weight_bytes"),
                ("quantization", "quantization"),
            )
        ):
            return None
        tool_support = model_tool_support(root)
        if not isinstance(tool_support, dict) or (
            tool_support.get("status") != "available" or tool_support.get("parser") != "qwen38-xml-v1"
        ):
            return None
        config_bytes = _small_file(root, "config.json")
        config = json.loads(config_bytes)
        if not _configuration_matches(config):
            return None
        text = config["text_config"]
        paged_table_bytes = _paged_table_weight_bytes(root, config)
        if not 0 < paged_table_bytes < identity["weight_bytes"]:
            return None
        context_tokens = min(TARGET_CONTEXT, text["max_position_embeddings"])
        # Guided OpenWork prompts need the initial 32K context budget.
        if context_tokens < MIN_CONTEXT:
            return None
        logical_parameters = (
            3 * text["hidden_size"] * text["moe_intermediate_size"] * text["num_experts"] * text["num_hidden_layers"]
        )
        runtime_memory, gpu_runtime_memory = _runtime_memory(
            identity["weight_bytes"],
            text,
            context_tokens,
            gpu_weight_bytes=identity["weight_bytes"] - paged_table_bytes,
        )
        return {
            "repo": identity["repo"],
            "revision": identity["revision"],
            "architecture": identity["architecture"],
            "weight_bytes": identity["weight_bytes"],
            "quantization": identity["quantization"],
            "context_tokens": context_tokens,
            "runtime_memory": runtime_memory,
            "gpu_runtime_memory": gpu_runtime_memory,
            "paged_table_bytes": paged_table_bytes,
            "logical_parameters": logical_parameters,
            "logical_parameters_basis": "lower bound: routed expert gate/up/down matrices only",
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "evidence": "pinned_installed_source_and_metadata",
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, ConversionError):
        return None


def inspect_installed(root):
    """Try the pinned hybrid planner, then the separate dense planner."""
    candidate = installed_qwen38_candidate(root)
    if candidate is None:
        try:
            from .guided_dense_installed import inspect_dense

            candidate = inspect_dense(Path(root))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return None
    if candidate is not None:
        candidate["capacity_is_lower_bound"] = True
    return candidate
