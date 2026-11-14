"""Header-only native control and draft admission for retained Lily checkpoints.

Typed text conversion intentionally omits the draft head. Retained image
preparation loads it, so its metadata and complete storage need separate
checks matching the native Lily loader.
"""

import math
import struct

from .adapters.qwen38 import architecture_config, describe_text_weights
from .streaming import UnsupportedConversion


def _contract(config):
    architecture, _ = architecture_config(config)
    lily = config["lily"]
    if any(type(lily["quantization"].get(flag, False)) is not bool for flag in ("q8_dense", "q8_embed")):
        raise UnsupportedConversion("The Lily Q8 storage flags must be native booleans.")
    groups = set(architecture["quantization"]["q8_groups"])
    draft_q4 = lily["quantization"].get("draft_q4", False)
    if type(draft_q4) is not bool or draft_q4 and not groups.intersection({"attn", "shared", "head"}):
        raise UnsupportedConversion("The Lily draft precision policy is outside the native storage contract.")
    mtp = lily.get("mtp")
    if mtp is not None:
        valid_theta = False
        theta = mtp.get("rope_theta") if isinstance(mtp, dict) else None
        if type(theta) in (int, float):
            try:
                rounded = struct.unpack("<f", struct.pack("<f", theta))[0]
                valid_theta = math.isfinite(rounded) and rounded > 0
            except (OverflowError, struct.error):
                pass
        if (
            not isinstance(mtp, dict)
            or type(mtp.get("layers")) is not int
            or mtp["layers"] != 1
            or mtp.get("layer_types") != ["full_attention"]
            or not valid_theta
        ):
            raise UnsupportedConversion("Lily images require a supported one-layer full-attention draft head.")
    return architecture, groups, draft_q4, mtp


def validate_mtp_config(config):
    """Check native head metadata even before tensor headers are selected."""
    _contract(config)


def validate_trunk_control_dtypes(config, tensors):
    """Match native control inputs while leaving typed text F32 references intact."""
    architecture = architecture_config(config)[0]

    def control(name, _shape, integers=False):
        info = tensors.get(name)
        if info is None:
            raise UnsupportedConversion("The Lily checkpoint is missing native trunk controls.")
        dtype = info.dtype if hasattr(info, "dtype") else info[0]
        # load_gdn explicitly upcasts these two fields through tensor_f32.
        # The other native RMS/HC/attention/PLE gains, dt_bias and convolution
        # weights are consumed by BF16-only kernels/readers.
        allowed = (
            {"I64", "U64"}
            if integers
            else {"BF16", "F32"}
            if name.endswith((".linear_attn.A_log", ".linear_attn.norm.weight"))
            else {"BF16"}
        )
        if dtype not in allowed:
            raise UnsupportedConversion("A retained Lily trunk control has a dtype unsupported by its native kernel.")

    describe_text_weights(architecture, tensors, lambda *_args, **_kwargs: None, control)


def validate_mtp_layout(config, tensors):
    """Require every declared draft plane and reject unconsumed draft tensors."""
    architecture, groups, draft_q4, mtp = _contract(config)
    actual = {name: info for name, info in tensors.items() if name.startswith("mtp.")}
    if mtp is None:
        if actual:
            raise UnsupportedConversion("The Lily checkpoint contains undeclared draft tensors.")
        return
    required = set()

    def record(name):
        info = actual.get(name)
        if info is None:
            raise UnsupportedConversion("The Lily checkpoint is missing required draft tensors.")
        required.add(name)
        return (info.dtype, tuple(info.shape)) if hasattr(info, "dtype") else info

    def control(name, shape, integers=False):
        if not name.startswith("mtp."):
            return
        dtype, actual_shape = record(name)
        # Full-attention draft controls are RMS/HC/attention gains. Their
        # execution kernels require BF16, even though the raw tensor reader
        # also accepts F32 for the trunk's separately upcast GDN controls.
        if dtype != "BF16" or actual_shape != shape:
            raise UnsupportedConversion("A Lily draft control has an unsupported dtype or shape.")

    def matrix(stem, shape, *, group=None, fixed_q8=False, embedding=False, ngram=False):
        if not stem.startswith("mtp."):
            return
        bits = 8 if fixed_q8 or group in groups and not draft_q4 else 4
        columns, rows = shape[-1], math.prod(shape[:-1])
        if columns % 64:
            raise UnsupportedConversion("The Lily draft affine groups are outside the native geometry.")
        for plane, dtype, width in (
            ("weight", "U32", columns * bits // 32),
            ("scales", "BF16", columns // 64),
            ("biases", "BF16", columns // 64),
        ):
            actual_dtype, actual_shape = record(stem + "." + plane)
            # The native row concatenator flattens leading expert dimensions.
            # A flat bank and its logical [expert, row, column] view have the
            # same stored row order; only the row count and packed width vary.
            if (
                actual_dtype != dtype
                or len(actual_shape) < 2
                or actual_shape[-1] != width
                or math.prod(actual_shape[:-1]) != rows
            ):
                raise UnsupportedConversion("A Lily draft affine plane disagrees with its native storage policy.")

    describe_text_weights(architecture, tensors, matrix, control, include_mtp=True)
    if draft_q4 and "head" in groups:
        matrix("mtp.lm_head", (architecture["vocab_size"], architecture["hidden_size"]), group="head")
    if set(actual) != required:
        raise UnsupportedConversion("The Lily checkpoint contains draft tensors outside the declared native head.")
