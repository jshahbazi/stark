"""Source-bound Gemma 3 SigLIP image conversion, without a model framework.

Equations and original checkpoint names follow Google's Apache-2.0 Gemma 3
and Transformers v4.53.3 (models/gemma3 and models/siglip). The 1B/270M text
models have no image encoder. All three image sizes use the same SigLIP tower.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..streaming import ConversionError, TensorOperation, UnsupportedConversion, load_checkpoint, read_json
from .gemma import build_gemma_text_plan

FAMILY = {"profile": "gemma3-siglip-bf16-v1", "version": 1}
TOKENS = {"image": 262144, "vision_start": 255999, "vision_end": 256000}
VISION = {
    "model_type": "siglip_vision_model", "hidden_size": 1152,
    "image_size": 896, "intermediate_size": 4304, "num_attention_heads": 16,
    "num_hidden_layers": 27, "patch_size": 14, "vision_use_head": False,
}
VISION_DEFAULTS = {
    "num_channels": 3, "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh", "attention_dropout": 0.0,
}
# hidden, intermediate, layers, heads, KV heads, head width, query scalar.
TEXT_GEOMETRIES = {
    2560: (10240, 34, 8, 4, 256, 256),
    3840: (15360, 48, 16, 8, 256, 256),
    5376: (21504, 62, 32, 16, 128, 168),
}
_ANNOTATIONS = {"_name_or_path", "torch_dtype", "dtype", "transformers_version", "initializer_range"}


def gemma3_config(config: dict) -> tuple[dict, dict]:
    """Admit supported original HF image geometry before tensor planning."""
    allowed = _ANNOTATIONS | {"model_type", "architectures", "text_config", "vision_config", "mm_tokens_per_image",
        "image_token_index", "boi_token_index", "eoi_token_index", "eos_token_id", "bos_token_id", "pad_token_id"}
    if (config.get("model_type") != "gemma3"
        or config.get("architectures") != ["Gemma3ForConditionalGeneration"]
        or set(config) - allowed
        or config.get("mm_tokens_per_image", 256) != 256):
        raise UnsupportedConversion("Gemma 3 images require the original multimodal wrapper")
    if any(config.get(key, token) != token for key, token in (
        ("image_token_index", TOKENS["image"]), ("boi_token_index", TOKENS["vision_start"]),
        ("eoi_token_index", TOKENS["vision_end"]))):
        raise UnsupportedConversion("Gemma 3 image token IDs differ from the implemented profile")
    vision, text = config.get("vision_config"), config.get("text_config")
    if not isinstance(vision, dict) or not isinstance(text, dict):
        raise ConversionError("Gemma 3 requires text and vision configurations")
    if (any(type(vision.get(key)) is not type(value) or vision.get(key) != value for key, value in VISION.items())
        or any(type(vision.get(key, value)) is not type(value) or vision.get(key, value) != value
               for key, value in VISION_DEFAULTS.items())
        or set(vision) - (set(VISION) | set(VISION_DEFAULTS) | _ANNOTATIONS)):
        raise UnsupportedConversion("Gemma 3 vision geometry or operations are outside the implemented profile")
    model_dim = text.get("hidden_size")
    if type(model_dim) is not int or model_dim not in TEXT_GEOMETRIES:
        raise UnsupportedConversion("Gemma 3 image execution supports the 4B, 12B, and 27B decoders")
    keys = ("intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "query_pre_attn_scalar")
    defaults = dict(zip(keys, (None, None, 8, 4, 256, 256), strict=True))
    if any(type(text.get(key, defaults[key])) is not int or text.get(key, defaults[key]) != value
           for key, value in zip(keys, TEXT_GEOMETRIES[model_dim], strict=True)):
        raise UnsupportedConversion("Gemma 3 decoder geometry disagrees with its image size profile")
    normalized = {**text, "architectures": ["Gemma3ForCausalLM"]}
    # The complete decoder planner independently admits all remaining fields.
    return normalized, {**VISION, **VISION_DEFAULTS}


def vision_shapes(model_dim: int, *, hidden: int = 1152, intermediate: int = 4304, layers: int = 27,
                  image_size: int = 896, patch: int = 14) -> dict[str, tuple[int, ...]]:
    """Complete consumed source inventory; optional dimensions serve tiny fixtures."""
    shapes = {
        "vision_tower.vision_model.embeddings.patch_embedding.weight": (hidden, 3, patch, patch),
        "vision_tower.vision_model.embeddings.patch_embedding.bias": (hidden,),
        "vision_tower.vision_model.embeddings.position_embedding.weight": ((image_size // patch) ** 2, hidden),
        "vision_tower.vision_model.post_layernorm.weight": (hidden,),
        "vision_tower.vision_model.post_layernorm.bias": (hidden,),
        "multi_modal_projector.mm_soft_emb_norm.weight": (hidden,),
        "multi_modal_projector.mm_input_projection_weight": (hidden, model_dim),
    }
    for index in range(layers):
        base = f"vision_tower.vision_model.encoder.layers.{index}"
        for norm in ("layer_norm1", "layer_norm2"):
            shapes[f"{base}.{norm}.weight"] = (hidden,)
            shapes[f"{base}.{norm}.bias"] = (hidden,)
        for projection, shape in (
            ("self_attn.q_proj", (hidden, hidden)), ("self_attn.k_proj", (hidden, hidden)),
            ("self_attn.v_proj", (hidden, hidden)), ("self_attn.out_proj", (hidden, hidden)),
            ("mlp.fc1", (intermediate, hidden)), ("mlp.fc2", (hidden, intermediate))):
            shapes[f"{base}.{projection}.weight"] = shape
            shapes[f"{base}.{projection}.bias"] = (shape[0],)
    return shapes


def plan_vision_tensors(tensors: dict, model_dim: int, *, shapes: dict | None = None):
    shapes = vision_shapes(model_dim) if shapes is None else shapes
    operations = []
    if set(tensors) != set(shapes):
        raise ConversionError("Gemma 3 vision checkpoint tensor inventory differs from its configuration")
    for name, shape in shapes.items():
        info = tensors[name]
        if info.dtype != "BF16" or info.shape != shape:
            raise ConversionError("Gemma 3 vision tensor requires matching BF16 storage and shape")
        destination = name.removeprefix("vision_tower.vision_model.")
        if name.startswith("multi_modal_projector."):
            destination = name.removeprefix("multi_modal_projector.")
        if ".self_attn." in name and name.rsplit(".", 2)[-2] in {"q_proj", "k_proj", "v_proj"}:
            continue
        if name == "multi_modal_projector.mm_soft_emb_norm.weight":
            operations.append(TensorOperation("vision.mm_soft_emb_norm.scales", (info,), "F32", shape, "cast_f32"))
        else:
            operations.append(TensorOperation("vision." + destination, (info,), "BF16", shape))
    prefixes = sorted({name.rsplit(".self_attn.", 1)[0] for name in shapes if ".self_attn.q_proj." in name})
    for prefix in prefixes:
        for suffix in ("weight", "bias"):
            inputs = tuple(tensors[f"{prefix}.self_attn.{name}_proj.{suffix}"] for name in ("q", "k", "v"))
            shape = (sum(info.shape[0] for info in inputs), *inputs[0].shape[1:])
            destination = prefix.removeprefix("vision_tower.vision_model.") + f".self_attn.qkv.{suffix}"
            operations.append(TensorOperation("vision." + destination, inputs, "BF16", shape, "fuse"))
    return tuple(operations)


def build_gemma3_vision_plan(source: Path, *, dtype: str = "bfloat16"):
    if dtype != "bfloat16":
        raise UnsupportedConversion("Gemma 3 native image execution requires merged BF16 weights")
    config = read_json(source / "config.json")
    text, _ = gemma3_config(config)
    tensors = load_checkpoint(source)
    # Original Transformers checkpoints and their newer canonical namespace are
    # distinct, complete representations. A mixture is never silently accepted.
    original = any(name.startswith("language_model.model.") for name in tensors)
    prefix = "language_model." if original else "model.language_model."
    decoder = {name.removeprefix(prefix): info for name, info in tensors.items() if name.startswith(prefix)}
    if not original:
        decoder = {"model." + name: info for name, info in decoder.items()}
    remainder = {name: info for name, info in tensors.items() if not name.startswith(prefix)}
    if not original:
        remainder = {name.removeprefix("model."): info for name, info in remainder.items()}
    base = build_gemma_text_plan(text, decoder, family="gemma3_text", dtype=dtype)
    operations = plan_vision_tensors(remainder, text["hidden_size"])
    if any(info.dtype != "BF16" for info in tensors.values()):
        raise UnsupportedConversion("Gemma 3 images require complete BF16 source storage")
    return replace(base, operations=tuple(sorted((*base.operations, *operations), key=lambda operation: operation.name)),
        decoder_config={**base.decoder_config, "scalar_image_embeddings": True, "bidirectional_image_attention": True},
        metadata={**base.metadata, "stark.vision.family": FAMILY["profile"]},
        input_bytes=sum(info.nbytes for info in tensors.values()), source_tensors=len(tensors),
        input_tensors=tuple(tensors.values()))
