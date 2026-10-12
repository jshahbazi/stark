"""Bounded dense and affine MLX Mamba2 mixer planning for pinned Llamba.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
This module does not implement Mamba1 or import any upstream runtime.
"""

from __future__ import annotations

import json

from ..compressed import MLXQuantization, plan_mlx_matrix, read_mlx_matrix
from ..streaming import ConversionError, TensorInfo, TensorOperation, UnsupportedConversion
from .dense import _DTYPES, _bool, _positive_int


def mamba2_config(model_dim: int, ssm_config: dict, unsupported: list[str]) -> dict:
    """Validate the Llamba SSM structure and derive its actual Mamba2 export."""
    if type(model_dim) is not int or model_dim <= 0 or not isinstance(ssm_config, dict):
        raise ConversionError("Mamba2 model/SSM dimensions are malformed")
    required = {"d_state", "n_v_heads", "n_qk_heads", "expand", "activation", "bias"}
    if not required <= set(ssm_config):
        raise ConversionError("Mamba2 SSM config is missing structural fields")
    state_dim = _positive_int(ssm_config, "d_state")
    heads = _positive_int(ssm_config, "n_v_heads")
    groups = _positive_int(ssm_config, "n_qk_heads")
    expansion = _positive_int(ssm_config, "expand")
    kernel = _positive_int(ssm_config, "d_conv", 4)
    inner_dim = model_dim * expansion
    if inner_dim % heads or heads % groups:
        raise ConversionError("Mamba2 head/group dimensions disagree")
    biases = _bool(ssm_config, "bias", False)
    conv_bias = _bool(ssm_config, "conv_bias", True)
    if "chunk_size" in ssm_config:
        _positive_int(ssm_config, "chunk_size")
    if not isinstance(ssm_config["activation"], str) or not ssm_config["activation"]:
        raise ConversionError("Mamba2 activation metadata is malformed")
    if ssm_config["activation"] not in ("identity", "silu"):
        unsupported.append("Mamba2 activation is outside implemented coverage")
    if set(ssm_config) - required - {"d_conv", "conv_bias", "chunk_size"}:
        unsupported.append("Mamba2 SSM config options are outside implemented coverage")
    activation = {"type": "Identity"} if ssm_config["activation"] == "identity" else {"type": "SiLU", "alpha": 1.0}
    return {
        "type": "Mamba2Config",
        "in_projection_config": {},
        "out_projection_config": {},
        "conv_config": {"has_biases": conv_bias},
        "activation": activation,
        "kernel_size": kernel,
        "num_heads": heads,
        "num_groups": groups,
        "head_dim": inner_dim // heads,
        "state_dim": state_dim,
        "has_in_biases": biases,
        "has_out_biases": biases,
    }


def plan_mamba2(
    tensors: dict[str, TensorInfo],
    source_prefix: str,
    output_prefix: str,
    model_dim: int,
    ssm_config: dict,
    target_dtype: str,
    expected: set[str],
    unsupported: list[str],
    *,
    quantization: MLXQuantization | None = None,
) -> tuple[list[TensorOperation], dict[str, str], dict]:
    """Plan dense/MLX projections and F32 recurrence from checkpoint headers.

    Every parameter required by the selected config must be present. The pinned
    importer initializes an empty template: missing D, z_bias, convolution, or
    configured biases leave dummy parameters rather than valid model values.
    The caller checks full checkpoint consumption after all layers are planned.
    """
    if target_dtype not in _DTYPES.values():
        raise ConversionError("Mamba2 target dtype is outside dense coverage")
    config = mamba2_config(model_dim, ssm_config, unsupported)
    inner_dim = config["num_heads"] * config["head_dim"]
    conv_dim = inner_dim + 2 * config["num_groups"] * config["state_dim"]
    projection_dim = conv_dim + inner_dim + config["num_heads"]
    operations = []
    metadata = {}

    def tensor(name: str, shapes: tuple[tuple[int, ...], ...]) -> TensorInfo:
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Mamba2 checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape not in shapes:
            raise ConversionError("Mamba2 tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("Mamba2 checkpoint dtype is outside dense coverage")
        return info

    def select(suffixes: tuple[str, ...]) -> str:
        present = [source_prefix + suffix for suffix in suffixes if source_prefix + suffix in tensors]
        if len(present) != 1:
            raise ConversionError("Mamba2 checkpoint has missing or ambiguous parameter aliases")
        return present[0]

    def output(name: str, source_name: str, shape: tuple[int, ...], dtype: str, *, shapes=None):
        info = tensor(source_name, shapes or (shape,))
        kind = "copy" if info.dtype == dtype else "cast"
        operations.append(TensorOperation(output_prefix + name, (info,), dtype, shape, kind))

    for source_name, output_name, shape in (
        (".in_proj", ".in_projection", (projection_dim, model_dim)),
        (".out_proj", ".out_projection", (model_dim, inner_dim)),
    ):
        name = source_prefix + source_name + ".weight"
        matrix_prefix = name.removesuffix(".weight")
        packed = any(matrix_prefix + suffix in tensors for suffix in (".scales", ".biases")) or (
            name in tensors and tensors[name].dtype in {"I32", "U32"}
        )
        destination = output_prefix + output_name + ".weights"
        if packed:
            if quantization is None:
                raise UnsupportedConversion("Mamba2 MLX matrix requires a declared quantization representation")
            matrix = read_mlx_matrix(tensors, name, shape, quantization)
            expected.update((name, matrix_prefix + ".scales", matrix_prefix + ".biases"))
            operations.extend(
                plan_mlx_matrix(
                    destination, (matrix,), dtype=target_dtype, layout="output_input", quantization=quantization
                )
            )
            spec = {
                "type": "MLXSpec",
                "bits": quantization.bits,
                "group_size": quantization.group_size,
                "layout": "output_input",
            }
        else:
            output(output_name + ".weights.weights", name, shape, target_dtype)
            spec = {"type": "FullPrecisionSpec", "layout": "output_input"}
        metadata[destination + ".spec"] = json.dumps(spec)
        if config["has_in_biases"]:
            output(output_name + ".biases", source_prefix + source_name + ".bias", (shape[0],), target_dtype)
    conv_name = select((".conv1d.weight", ".conv_weight", ".conv.weight"))
    output(
        ".conv.weights",
        conv_name,
        (conv_dim, config["kernel_size"]),
        "F32",
        shapes=(
            (conv_dim, config["kernel_size"]),
            (conv_dim, 1, config["kernel_size"]),
            (conv_dim, config["kernel_size"], 1),
        ),
    )
    if config["conv_config"]["has_biases"]:
        bias_name = select((".conv1d.bias", ".conv_bias", ".conv.bias"))
        output(".conv.biases", bias_name, (conv_dim,), "F32")
    output(".skip_connection_weight", source_prefix + ".D", (config["num_heads"],), "F32")
    output(".gate_bias", source_prefix + ".z_bias", (inner_dim,), "F32")
    return operations, metadata, config
