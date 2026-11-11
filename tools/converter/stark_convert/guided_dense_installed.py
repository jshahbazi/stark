"""Metadata-only planning for installed, source-bound dense text artifacts.

Uses the converted attention schedule, not online catalogue availability. The
F32 cache/scratch allowance intentionally covers BF16 and reference F32 paths.
Native admission remains authoritative; no tensors or model code are executed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .model_inventory import _small_file, _weights, model_display, model_tool_support
from .streaming import ConversionError, read_safetensors_header

GIB = 1024**3


def _integer(value, low=1, high=1_048_576):
    if type(value) is not int or not low <= value <= high:
        raise ValueError()
    return value


def inspect_dense(root: Path):
    """Return a conservative local plan, or None for an unknown layout."""
    try:
        from .guided_models import DENSE_ARCHITECTURES, GUIDED_TOOL_PARSERS
        from .model_catalog import _model_group
        from .registry import get_model

        raw = _small_file(root, "config.json")
        config = json.loads(raw)
        display = model_display({"id": "local", "name": "local", "path": str(root)})
        support = model_tool_support(root)
        if (
            config.get("type") != "LanguageModelConfig"
            or display is None
            or display.get("family_id") not in DENSE_ARCHITECTURES
            or display.get("source_revision") is None
            or not display.get("group_id", "").startswith("repo:")
            or support.get("status") != "available"
            or support.get("parser") not in GUIDED_TOOL_PARSERS
        ):
            return None
        decoder = config["decoder_config"]
        transformer = decoder["transformer_config"]
        layers = transformer["layer_configs"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= 512 or decoder.get("ple_model_config"):
            return None
        model_dim = _integer(transformer["model_dim"], high=262144)
        hidden_dim = _integer(transformer["hidden_dim"], high=1_048_576)
        context = 32768
        geometries = []
        for layer in layers:
            mixer = layer["mixer_config"]
            if (
                mixer.get("type") != "AttentionConfig"
                or layer["mlp_config"].get("type") != "DenseMLPConfig"
                or layer.get("mixer_conv_config")
                or layer.get("mlp_conv_config")
                or layer.get("ple_config")
            ):
                return None
            heads = _integer(mixer["num_heads"], high=512)
            groups = _integer(mixer["num_groups"], high=256)
            dim = _integer(mixer["head_dim"], high=1024)
            if heads % groups:
                return None
            # A missing trained limit cannot establish a guided context size.
            limit = _integer(layer["rope_config"]["max_sequence_length"])
            context = min(context, limit)
            geometries.append((heads, groups, dim))
            hidden_dim = max(hidden_dim, _integer(layer.get("hidden_dim") or hidden_dim))
        if context < 32768:
            return None
        weight_bytes, quantization = _weights(root)
        if quantization is None:
            return None
        # Sum only known learned matrix planes. Excluding biases/norms is a
        # lower bound; packed scalar counts are expanded using each exact spec.
        metadata, tensors = read_safetensors_header(root / "model.safetensors")
        parameters = 0
        for name, encoded in metadata.items():
            if not name.endswith(".spec"):
                continue
            spec = json.loads(encoded)
            base = name.removesuffix(".spec")
            if spec.get("type") == "FullPrecisionSpec":
                tensor = tensors[base + ".weights"]
                if tensor.dtype not in {"BF16", "F16", "F32"}:
                    return None
                parameters += math.prod(tensor.shape)
            elif spec.get("type") in {"MLXSpec", "IntSpec"}:
                tensor = tensors[base + ".weights"]
                bits = _integer(spec["bits"], high=8)
                word_bits = {"U8": 8, "I8": 8, "U32": 32, "I32": 32}.get(tensor.dtype)
                if word_bits is None or math.prod(tensor.shape) * word_bits % bits:
                    return None
                parameters += math.prod(tensor.shape) * word_bits // bits
            else:
                # Row-partitioned/expert formats need their own planner.
                return None
        if not 0 < parameters < 2**63:
            return None
        cache = sum(2 * (context + 1024) * groups * dim * 4 for _, groups, dim in geometries)
        # Native sequential scratch retains distinct allocation size classes.
        # Charge every class for heterogeneous attention schedules rather than
        # assuming all layers share the largest buffer.
        score_classes = {
            1 << (1024 * (context + 1024) * width * 4 - 1).bit_length()
            for heads, groups, _ in geometries
            for width in (heads, heads // groups)
        }
        scores = sum(score_classes)
        workspace = max(8 * GIB, scores + 1024 * (model_dim + hidden_dim) * 4 * 12)
        runtime = ((weight_bytes + 128 * 1024**2 + cache + workspace + GIB - 1) // GIB) * GIB
        repo = display["group_id"].removeprefix("repo:")
        group = _model_group(display["family_id"], repo, repo.split("/")[-1], quantization, get_model(repo))
        return {
            "display_name": group["name"],
            "repo": repo,
            "revision": display["source_revision"],
            "architecture": display["family_id"],
            "weight_bytes": weight_bytes,
            "quantization": quantization,
            "context_tokens": context,
            "runtime_memory": runtime,
            "logical_parameters": parameters,
            "capacity_is_lower_bound": True,
            "config_sha256": hashlib.sha256(raw).hexdigest(),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError, ConversionError):
        return None
