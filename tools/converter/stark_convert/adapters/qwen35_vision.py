"""Qwen3.5 vision weight planner.

The pinned Transformers reference (2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397)
declares a Conv3d patch embed, learned position embedding, ``depth`` vision
blocks (LayerNorm, fused QKV attention, projection, and two bias-bearing MLP
linears), and a LayerNorm/linear patch merger.  This module copies those
official wrapper tensors without layout or kernel transformations.
"""

from __future__ import annotations

from ..streaming import ConversionError, TensorInfo, TensorOperation, UnsupportedConversion


def _positive(config: dict, name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ConversionError(f"Qwen3.5 vision {name} must be a positive integer")
    return value


def build_qwen35_vision_plan(vision_config: dict, tensors: dict[str, TensorInfo], model_dim: int):
    """Return copy operations and consumed source names for official HF vision weights."""
    if not isinstance(vision_config, dict) or vision_config.get("model_type") != "qwen3_5":
        raise ConversionError("Qwen3.5 vision_config must declare qwen3_5")
    allowed = {
        "model_type",
        "hidden_size",
        "intermediate_size",
        "depth",
        "num_heads",
        "patch_size",
        "temporal_patch_size",
        "in_channels",
        "num_position_embeddings",
        "spatial_merge_size",
        "out_hidden_size",
        "hidden_act",
        "deepstack_visual_indexes",
        "initializer_range",
        "dtype",
        "torch_dtype",
    }
    if set(vision_config) - allowed:
        raise UnsupportedConversion("Qwen3.5 vision config contains unsupported fields")
    if any(vision_config.get(key, "bfloat16") != "bfloat16" for key in ("dtype", "torch_dtype")):
        raise UnsupportedConversion("Qwen3.5 vision dtype hints must declare BF16")
    hidden = _positive(vision_config, "hidden_size")
    intermediate = _positive(vision_config, "intermediate_size")
    depth = _positive(vision_config, "depth")
    if depth > 4096 or len(tensors) < depth * 12:
        raise ConversionError("Qwen3.5 vision depth is unreasonable for this checkpoint")
    heads = _positive(vision_config, "num_heads")
    patch = _positive(vision_config, "patch_size")
    temporal = _positive(vision_config, "temporal_patch_size")
    channels = _positive(vision_config, "in_channels")
    positions = _positive(vision_config, "num_position_embeddings")
    merge = _positive(vision_config, "spatial_merge_size")
    output = _positive(vision_config, "out_hidden_size")
    if hidden % heads or output != model_dim:
        raise ConversionError("Qwen3.5 vision dimensions disagree with the model")
    if vision_config.get("deepstack_visual_indexes", []) not in ([], None):
        raise ConversionError("Qwen3.5 vision deepstack weights are unsupported")
    if vision_config.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise ConversionError("Qwen3.5 vision requires GELU with tanh approximation")
    merger_dim = hidden * merge * merge
    shapes = {
        "patch_embed.proj.weight": (hidden, channels, temporal, patch, patch),
        "patch_embed.proj.bias": (hidden,),
        "pos_embed.weight": (positions, hidden),
        "merger.norm.weight": (hidden,),
        "merger.norm.bias": (hidden,),
        "merger.linear_fc1.weight": (merger_dim, merger_dim),
        "merger.linear_fc1.bias": (merger_dim,),
        "merger.linear_fc2.weight": (output, merger_dim),
        "merger.linear_fc2.bias": (output,),
    }
    for index in range(depth):
        base = f"blocks.{index}"
        shapes.update(
            {
                f"{base}.norm1.weight": (hidden,),
                f"{base}.norm1.bias": (hidden,),
                f"{base}.norm2.weight": (hidden,),
                f"{base}.norm2.bias": (hidden,),
                f"{base}.attn.qkv.weight": (3 * hidden, hidden),
                f"{base}.attn.qkv.bias": (3 * hidden,),
                f"{base}.attn.proj.weight": (hidden, hidden),
                f"{base}.attn.proj.bias": (hidden,),
                f"{base}.mlp.linear_fc1.weight": (intermediate, hidden),
                f"{base}.mlp.linear_fc1.bias": (intermediate,),
                f"{base}.mlp.linear_fc2.weight": (hidden, intermediate),
                f"{base}.mlp.linear_fc2.bias": (hidden,),
            }
        )
    operations = []
    expected = set()
    for suffix, shape in shapes.items():
        source_name = "model.visual." + suffix
        info = tensors.get(source_name)
        if info is None:
            raise ConversionError(f"Qwen3.5 vision checkpoint is missing {source_name}")
        if info.dtype != "BF16" or info.shape != shape:
            raise ConversionError(f"Qwen3.5 vision tensor {source_name} has invalid dtype or shape")
        expected.add(source_name)
        operations.append(TensorOperation("vision." + suffix, (info,), "BF16", shape, "copy"))
    return tuple(operations), frozenset(expected)
