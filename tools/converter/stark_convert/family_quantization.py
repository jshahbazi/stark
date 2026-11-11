"""Shared checkpoint weight-format admission for catalogues and conversion."""

from .compressed import MLXQuantization, parse_mlx_quantization
from .streaming import ConversionError, UnsupportedConversion


def gemma4_quantization(config: dict, *, dtype: str = "bfloat16") -> MLXQuantization | None:
    """Keep wrapper quantization and the decoder precision policy consistent."""
    quantization = parse_mlx_quantization(config)
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ConversionError("Gemma 4 wrapper requires a text configuration")
    if text is not config and parse_mlx_quantization(text) not in (None, quantization):
        raise ConversionError("Gemma 4 wrapper and text quantization disagree")
    if text.get("attention_k_eq_v") is True and (quantization is not None or dtype != "bfloat16"):
        raise UnsupportedConversion("Gemma 4 no-PLE global K=V currently requires canonical BF16")
    return quantization


def qwen35_quantization(config: dict) -> MLXQuantization | None:
    """Require MLX's baked-normalization marker for packed Qwen3.5 sources."""
    if config.get("quantization_config") is not None and config.get("quantization") is None:
        raise UnsupportedConversion("Qwen3.5 MLX requires the quantization field for baked normalization weights")
    return parse_mlx_quantization(config, allow_mixed=True)
