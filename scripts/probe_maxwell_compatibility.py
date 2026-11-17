#!/usr/bin/env python3
"""Metadata-only Maxwell prototype; never imports upstream Python or downloads tensor data.

Run with the locked converter environment. Raw upstream metadata stays under runtime/;
the report contains hashes, structural counts, admission results and payload estimates.
This is not a converter, native execution test, or model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from stark_convert.model_support import preflight_model

REPO = "maxwell-model/Maxwell-V1-Flash"
REVISION = "ca536c4380e0600d6aebdb808d2ce65ec724161b"
MAX_METADATA = 16 * 1024 * 1024
FILES = (
    "config.json",
    "configuration_maxwell_moe.py",
    "modeling_maxwell_moe.py",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "generation_config.json",
    "LICENSE",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str, byte_range: tuple[int, int] | None = None) -> bytes:
    if "/" in name or name in {".", ".."}:
        raise ValueError("expected a repository basename")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    headers = {}
    if byte_range is not None:
        start, end = byte_range
        # Separate cache keys prevent a cached first range satisfying the second.
        url += f"?stark_header_range={start}-{end}"
        headers["Range"] = f"bytes={start}-{end}"
        limit = end - start + 1
    else:
        limit = MAX_METADATA
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        if byte_range is not None:
            content_range = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if (
                response.status != 206
                or content_range is None
                or tuple(map(int, content_range.groups()[:2])) != (start, end)
                or int(content_range[3]) <= end
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or int(response.headers.get("Content-Length", limit)) != limit
            ):
                raise ValueError("server did not honor exact header byte range")
        data = response.read(limit + 1)
    if len(data) > limit or (byte_range is not None and len(data) != limit):
        raise ValueError("metadata response size mismatch")
    return data


def header_length(prefix: bytes) -> int:
    if len(prefix) != 8:
        raise ValueError("invalid safetensors length prefix")
    size = struct.unpack("<Q", prefix)[0]
    if not 2 <= size <= MAX_METADATA:
        raise ValueError("header exceeds metadata-only budget")
    return size


def tensor_inventory(headers: dict, weight_map: dict) -> dict:
    tensors = {}
    payload_bytes = 0
    dtype_elements = Counter()
    for shard, header in headers.items():
        spans = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or weight_map.get(name) != shard:
                raise ValueError("index/header ownership mismatch")
            shape, dtype = tensor["shape"], tensor["dtype"]
            if dtype not in {"BF16", "F32"} or any(
                type(x) is not int or x < 0 for x in shape
            ):
                raise ValueError("unexpected source dtype/shape")
            elements = math.prod(shape)
            start, end = tensor["data_offsets"]
            if type(start) is not int or type(end) is not int or start < 0:
                raise ValueError("invalid tensor offsets")
            if end - start != elements * {"BF16": 2, "F32": 4}[dtype]:
                raise ValueError("tensor byte count mismatch")
            spans.append((start, end))
            tensors[name] = tensor
            dtype_elements[dtype] += elements
            payload_bytes += end - start
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("non-contiguous or overlapping tensor spans")
            cursor = end
    if tensors.keys() != weight_map.keys():
        raise ValueError("incomplete shard coverage")
    return {
        "tensors": tensors,
        "tensor_count": len(tensors),
        "payload_bytes": payload_bytes,
        "elements_by_dtype": dict(sorted(dtype_elements.items())),
    }


def geometry_checks(config: dict, tensors: dict) -> dict:
    expected = {
        "model_type": "maxwell_moe",
        "num_hidden_layers": 45,
        "hidden_size": 2048,
        "num_residual_streams": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "attention_head_size": 128,
        "num_experts": 256,
        "num_experts_per_tok": 12,
        "mlp_only_layers": [0, 1, 2],
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 1024,
        "intermediate_size": 6144,
        "mtp_num_layers": 1,
        "block_sparse_gqa_indexer_head_dim": 128,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("probe only supports the pinned Maxwell geometry")
    checks = {}

    def shape(name, expected, dtype="BF16"):
        tensor = tensors.get(name, {})
        return tensor.get("shape") == expected and tensor.get("dtype") == dtype

    h, streams = config["hidden_size"], config["num_residual_streams"]
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        checks[f"layer_{layer}_attention"] = all(
            (
                shape(f"{p}.self_attn.q_proj.weight", [4096, h]),
                shape(f"{p}.self_attn.k_proj.weight", [512, h]),
                shape(f"{p}.self_attn.v_proj.weight", [512, h]),
                shape(f"{p}.self_attn.o_proj.weight", [h, 4096]),
                shape(f"{p}.self_attn.indexer.linear_q_proj.weight", [512, h]),
                shape(f"{p}.self_attn.indexer.linear_k_proj.weight", [128, h]),
            )
        )
        checks[f"layer_{layer}_hc"] = all(
            shape(
                f"{p}.{part}_hyper_connection.mapping_proj.weight",
                [streams * streams + 2 * streams, streams * h],
            )
            for part in ("self_attn", "mlp")
        )
        if layer in config["mlp_only_layers"]:
            checks[f"layer_{layer}_dense"] = all(
                shape(f"{p}.block_sparse_moe.{part}.weight", expected)
                for part, expected in (
                    ("w1", [6144, h]),
                    ("w3", [6144, h]),
                    ("w2", [h, 6144]),
                )
            )
        else:
            p += ".block_sparse_moe"
            checks[f"layer_{layer}_router"] = shape(
                f"{p}.gate.weight", [256, h]
            ) and shape(f"{p}.expert_bias", [256], "F32")
            checks[f"layer_{layer}_experts"] = all(
                shape(f"{p}.experts.{e}.{part}.weight", expected)
                for e in range(256)
                for part, expected in (
                    ("w1", [512, h]),
                    ("w3", [512, h]),
                    ("w2", [h, 512]),
                )
            )
            checks[f"layer_{layer}_shared"] = all(
                shape(f"{p}.shared_expert.{part}.weight", expected)
                for part, expected in (
                    ("w1", [1024, h]),
                    ("w3", [1024, h]),
                    ("w2", [h, 1024]),
                )
            )
    checks["mtp_present"] = any(name.startswith("mtp_layers.0.") for name in tensors)
    checks["mtp_no_indexer"] = not any(
        name.startswith("mtp_layers.") and ".indexer." in name for name in tensors
    )
    checks["no_vision_tensors"] = not any(
        any(term in name.lower() for term in ("vision", "visual", "image", "projector"))
        for name in tensors
    )
    return {
        "passed": sum(checks.values()),
        "total": len(checks),
        "failed": [name for name, passed in checks.items() if not passed],
    }


def memory_payload(config: dict, payload_bytes: int) -> dict:
    # Batch=1, BF16 main K/V and indexer K. Excludes MTP, scratch and allocations.
    per_token = (
        config["num_hidden_layers"]
        * 2
        * (
            2 * config["num_key_value_heads"] * config["attention_head_size"]
            + config["block_sparse_gqa_indexer_head_dim"]
        )
    )
    return {
        "source_weight_bytes": payload_bytes,
        "bf16_backbone_cache_bytes_per_token": per_token,
        "bf16_backbone_cache_bytes_256k": per_token * 262144,
        "scope": "analytical payload only; excludes MTP caches, scratch, allocator and transient copies",
    }


def equation_controls() -> dict:
    """Synthetic equations transcribed from inspected sources, not native execution.

    Maxwell HyperConnectionModule vs glm.rs hc_pre: epsilon is outside vs inside
    the square root. Only the normalization factor, Sinkhorn operation order and
    no-tie router selection order are studied; projections and combination are not.
    No model activations, tokens or probabilities are retained or printed.
    """
    rng = np.random.default_rng(20261117)
    eps = 1e-6
    errors = []
    sinkhorn_errors = []
    router_cases = 0
    for scale in (1e-5, 1e-3, 1.0, 100.0):
        x = rng.normal(size=8192) * scale
        maxwell = x / (np.sqrt(np.mean(x * x)) + eps)
        glm = x / np.sqrt(np.mean(x * x) + eps)
        errors.append(float(np.max(np.abs(maxwell - glm))))
        logits = rng.normal(size=(4, 4))
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        upstream = probs + eps
        upstream /= upstream.sum(axis=-2, keepdims=True) + eps
        for _ in range(19):
            upstream /= upstream.sum(axis=-1, keepdims=True) + eps
            upstream /= upstream.sum(axis=-2, keepdims=True) + eps
        native_equation = probs + eps
        for iteration in range(20):
            if iteration:
                native_equation /= native_equation.sum(axis=1, keepdims=True) + eps
            native_equation /= native_equation.sum(axis=0, keepdims=True) + eps
        sinkhorn_errors.append(float(np.max(np.abs(upstream - native_equation))))
        scores = 1 / (1 + np.exp(-rng.normal(size=256)))
        bias = rng.normal(size=256) * 0.1
        # Compare vectorized selection with scalar stable sort; random no-tie inputs.
        selected = np.argsort(-(scores + bias), kind="stable")[:12]
        native_selected = sorted(range(256), key=lambda i: (-(scores[i] + bias[i]), i))[
            :12
        ]
        if list(selected) == native_selected:
            router_cases += 1
    return {
        "scope": "float64 synthetic equation study only; no actual Rust/Metal/PyTorch functions executed",
        "seed": 20261117,
        "cases": 4,
        "hc_normalization_max_abs_differences": errors,
        "sinkhorn_max_abs_error": max(sinkhorn_errors),
        "router_selection_matches": router_cases,
        "sparse_max_vs_mean_counterexample": max(8, -8) > max(1, 1)
        and (8 - 8) / 2 < (1 + 1) / 2,
        "conclusion": "Sinkhorn/order and no-tie selection order match; normalization factors differ; full HC/MoE untested",
    }


def run(output: Path) -> dict:
    runtime_root = Path(__file__).resolve().parents[1] / "runtime"
    if not output.resolve().is_relative_to(runtime_root.resolve()):
        raise ValueError(
            "raw metadata must remain under this worktree's ignored runtime directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {name: fetch(name) for name in FILES}
    for name, data in metadata.items():
        (output / name).write_bytes(data)
    config = json.loads(metadata["config.json"])
    index = json.loads(metadata["model.safetensors.index.json"])
    shards = sorted(set(index["weight_map"].values()))

    def read_header(shard):
        size = header_length(fetch(shard, (0, 7)))
        raw = fetch(shard, (8, size + 7))
        (output / f"{shard}.header.json").write_bytes(raw)
        return shard, raw

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_headers = dict(pool.map(read_header, shards))
    inventory = tensor_inventory(
        {s: json.loads(raw) for s, raw in raw_headers.items()}, index["weight_map"]
    )
    tensors = inventory.pop("tensors")
    if inventory["payload_bytes"] != index["metadata"]["total_size"]:
        raise ValueError("index total_size differs from header tensor spans")
    admission = {
        mode: preflight_model(config, mode=mode)
        for mode in ("text", "decision", "vision-language")
    }
    report = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "scope": "metadata and header structure only; no tensor payload read or upstream Python executed",
        "metadata_sha256": {name: digest(data) for name, data in metadata.items()},
        "header_manifest_sha256": digest(
            json.dumps(
                {s: digest(raw) for s, raw in raw_headers.items()}, sort_keys=True
            ).encode()
        ),
        "shards": len(shards),
        "header_bytes": sum(map(len, raw_headers.values())),
        "inventory": inventory,
        "geometry": geometry_checks(config, tensors),
        "admission": admission,
        "memory": memory_payload(config, inventory["payload_bytes"]),
        "equation_controls": equation_controls(),
        "mtp_tensor_count": sum(name.startswith("mtp_layers.") for name in tensors),
        "qualification": "not run; no converter, runtime capability, native build or installed artifact added",
    }
    (output / "aggregate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="private ignored runtime directory"
    )
    args = parser.parse_args()
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 - CLI privacy boundary: never dump remote error bodies.
        # No exception dumps or remote response bodies.
        raise SystemExit(f"metadata probe failed ({type(error).__name__})") from None
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["geometry"]["failed"]:
        raise SystemExit(1)
