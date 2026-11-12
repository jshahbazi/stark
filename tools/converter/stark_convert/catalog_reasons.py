"""Wording for catalogue rows that cannot be downloaded.

These helpers only describe why a row is unavailable. They never decide that
a row is downloadable; the catalogue's pinned checks still do that.
"""

from __future__ import annotations

import re

# The desktop backend accepts a reason of at most this many UTF-8 bytes.
MAX_REASON_BYTES = 240

UNSUPPORTED_LAYOUT = "The checkpoint's tensor names, shapes or packing do not match Stark's importer for this family."
GATED = (
    "This model needs access approval on Hugging Face. Sign in to Hugging Face, "
    "accept the model's terms on its page, then refresh the catalog."
)
_FOREIGN = (
    "This checkpoint is quantized with {label} for GPU inference servers. "
    "Stark imports this model's original weights or its MLX exports."
)

# Display names for Hugging Face ``quantization_config.quant_method`` values.
# The badge names the method only: several methods (compressed-tensors,
# modelopt, torchao) cover more than one scheme or bit width.
_FOREIGN_LABELS = {
    "aqlm": "AQLM",
    "auto-round": "AutoRound",
    "auto_round": "AutoRound",
    "awq": "AWQ",
    "bitnet": "BitNet",
    "bitsandbytes": "bitsandbytes",
    "compressed-tensors": "compressed-tensors",
    "compressed_tensors": "compressed-tensors",
    "eetq": "EETQ",
    "exl2": "EXL2",
    "exl3": "EXL3",
    "fbgemm_fp8": "FBGEMM FP8",
    "fp8": "FP8",
    "gptq": "GPTQ",
    "hqq": "HQQ",
    "higgs": "HIGGS",
    "modelopt": "ModelOpt",
    "mxfp4": "MXFP4",
    "quanto": "Quanto",
    "quark": "Quark",
    "spqr": "SpQR",
    "torchao": "torchao",
    "vptq": "VPTQ",
}
_METHOD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,31}\Z")


def _merged(config):
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        return config
    return {**text, **{key: value for key, value in config.items() if key != "text_config"}}


def foreign_format(config):
    """Return ``(badge, reason)`` for a GPU-server quantization export, else ``None``.

    A ``quantization_config`` whose ``quant_method`` names another method is
    foreign unless the config also carries an MLX-style ``quantization`` block,
    which MLX exports write alongside a matching ``quantization_config``.
    """
    if not isinstance(config, dict):
        return None
    merged = _merged(config)
    declared = merged.get("quantization_config")
    if not isinstance(declared, dict) or merged.get("quantization") is not None:
        return None
    method = declared.get("quant_method")
    if not isinstance(method, str) or method.casefold() == "mlx":
        return None
    label = _FOREIGN_LABELS.get(method.casefold())
    if label is None:
        label = method if _METHOD.fullmatch(method) else "another method"
    badge = label if label != "another method" else "Other format"
    return badge, _FOREIGN.format(label=label)


def bounded_reason(message, fallback=UNSUPPORTED_LAYOUT):
    """Return a displayable reason built from a developer-written message.

    Control characters fall back to ``fallback``; a long message is cut at a
    character boundary so it fits the backend's byte limit.
    """
    if not isinstance(message, str):
        return fallback
    text = " ".join(message.split())
    if not text or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in text):
        return fallback
    if text[-1] not in ".!?":
        text += "."
    if len(text.encode("utf-8")) <= MAX_REASON_BYTES:
        return text
    ellipsis = "…"
    budget = MAX_REASON_BYTES - len(ellipsis.encode("utf-8"))
    cut = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip()
    return cut + ellipsis if cut else fallback


def importer_reason(error):
    """Describe a planner or header rejection by the importer's own message.

    ``ConversionError`` and ``UnsupportedConversion`` carry fixed messages
    written in the importer (config field names and rule descriptions, never
    file paths or tensor contents). Every other type keeps the generic text.
    """
    from .streaming import ConversionError, UnsupportedConversion

    if type(error) not in (ConversionError, UnsupportedConversion) or len(error.args) != 1:
        return UNSUPPORTED_LAYOUT
    return bounded_reason(error.args[0])
