"""Bounded native Gemma 4 E2B/E4B BF16 image-tower conversion.

Independently follows Google's Apache-2.0 checkpoint geometry and MIT MLX-VLM
53ebd1647cb83549b482a031ea70ba9d9ea60a04. No model framework is imported.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..streaming import ConversionError, TensorOperation, UnsupportedConversion, load_checkpoint, read_json
from .gemma4 import build_gemma4_plan

REPO = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
FAMILY = {"id": "gemma4-e2b-bf16-v1", "version": 1}
PROFILES = {
    REPO: {"revision": REVISION, "model_dim": 1536, "family": FAMILY},
    "google/gemma-4-E4B-it": {
        "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
        "model_dim": 2560,
        "family": {"id": "gemma4-e4b-bf16-v1", "version": 1},
    },
    "google/gemma-4-31B-it": {
        "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "model_dim": 5376,
        "family": {"id": "gemma4-31b-bf16-v1", "version": 1},
    },
}
for _profile in PROFILES.values():
    _profile["template_sha256"] = (
        "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
        if _profile["model_dim"] == 5376
        else "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
    )
TOKENS = {"image": 258880, "vision_start": 255999, "vision_end": 258882, "video": 258884, "audio": 258881}
GEOMETRY = {
    "model_type": "gemma4_vision",
    "num_hidden_layers": 16,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_key_value_heads": 12,
    "head_dim": 64,
    "global_head_dim": 64,
    "intermediate_size": 3072,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "default_output_length": 280,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "standardize": False,
    "use_clipped_linears": True,
    "hidden_activation": "gelu_pytorch_tanh",
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
}
LARGE_GEOMETRY = {
    **GEOMETRY,
    "num_hidden_layers": 27,
    "hidden_size": 1152,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 72,
    "global_head_dim": 72,
    "intermediate_size": 4304,
    "standardize": True,
    "use_clipped_linears": False,
}

_VISION_ANNOTATIONS = {
    "_name_or_path",
    "architectures",
    "dtype",
    "id2label",
    "initializer_range",
    "label2id",
    "output_attentions",
    "output_hidden_states",
    "problem_type",
    "return_dict",
}
_VISION_DEFAULTS = {
    "attention_dropout": 0.0,
    "chunk_size_feed_forward": 0,
    "is_encoder_decoder": False,
    "max_position_embeddings": 131072,
}


def gemma4_vision_contract(source: Path, repo: str | None = None, revision: str | None = None) -> dict:
    config = read_json(source / "config.json")
    if any(config.get(key) is not None for key in ("quantization", "quantization_config")):
        raise UnsupportedConversion("Native Gemma image conversion requires merged BF16 source weights")
    vision = config.get("vision_config")
    geometry = LARGE_GEOMETRY if config.get("text_config", {}).get("hidden_size") == 5376 else GEOMETRY
    if not isinstance(vision, dict) or any(vision.get(key) != value for key, value in geometry.items()):
        raise UnsupportedConversion("Native Gemma images require the declared BF16 vision geometry")
    if set(vision) - (set(GEOMETRY) | _VISION_ANNOTATIONS | set(_VISION_DEFAULTS)):
        raise UnsupportedConversion("Gemma vision configuration declares unimplemented options")
    if any(vision.get(key, default) != default for key, default in _VISION_DEFAULTS.items()):
        raise UnsupportedConversion("Gemma vision configuration changes an unsupported operation")
    profile = next(
        (
            value
            for value in PROFILES.values()
            if value["model_dim"] == config.get("text_config", {}).get("hidden_size")
        ),
        None,
    )
    if config.get("model_type") != "gemma4" or profile is None:
        raise UnsupportedConversion("Native Gemma images require an implemented dense text decoder")
    if repo is not None and (PROFILES.get(repo) != profile or revision != profile["revision"]):
        raise UnsupportedConversion("Native Gemma image source differs from its pinned official profile")
    for key, token in (
        ("image_token_id", TOKENS["image"]),
        ("boi_token_id", TOKENS["vision_start"]),
        ("eoi_token_id", TOKENS["vision_end"]),
        ("video_token_id", TOKENS["video"]),
        ("audio_token_id", TOKENS["audio"]),
    ):
        if config.get(key) != token:
            raise UnsupportedConversion("Gemma image token IDs differ from the admitted profile")
    processor = read_json(source / "processor_config.json").get("image_processor")
    required = {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_processor_type": "Gemma4ImageProcessor",
        "image_seq_length": 280,
        "max_soft_tokens": 280,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1 / 255,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
    }
    if not isinstance(processor, dict) or processor != required:
        raise UnsupportedConversion("Gemma image processing metadata differs from the admitted profile")
    return {
        "schema_version": 4,
        "transport": "mediaref-v1",
        "input_modalities": ["text", "image"],
        "runtime_supported": True,
        "family": dict(profile["family"]),
        "source": {"repo": repo, "revision": revision},
        "vision_config": dict(geometry),
        "token_ids": dict(TOKENS),
        "processor": processor,
    }


def build_gemma4_vision_plan(source: Path, *, dtype: str = "bfloat16"):
    if dtype != "bfloat16":
        raise UnsupportedConversion("Native Gemma images require BF16 activations and weights")
    contract = gemma4_vision_contract(source)
    model_dim = read_json(source / "config.json")["text_config"]["hidden_size"]
    base = build_gemma4_plan(source, dtype=dtype)
    tensors = load_checkpoint(source)
    operations = list(base.operations)
    geometry = contract["vision_config"]
    width, head_dim, intermediate = (geometry[key] for key in ("hidden_size", "head_dim", "intermediate_size"))
    clipped = geometry["use_clipped_linears"]
    expected = {
        "model.vision_tower.patch_embedder.input_proj.weight": (width, 768),
        "model.vision_tower.patch_embedder.position_embedding_table": (2, 10240, width),
        "model.embed_vision.embedding_projection.weight": (model_dim, width),
    }
    if geometry["standardize"]:
        expected.update({"model.vision_tower.std_bias": (width,), "model.vision_tower.std_scale": (width,)})
    for index in range(geometry["num_hidden_layers"]):
        prefix = f"model.vision_tower.encoder.layers.{index}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            expected[f"{prefix}.{name}.weight"] = (width,)
        for name in ("q_norm", "k_norm"):
            expected[f"{prefix}.self_attn.{name}.weight"] = (head_dim,)
        for name, shape in (
            ("self_attn.q_proj", (width, width)),
            ("self_attn.k_proj", (width, width)),
            ("self_attn.v_proj", (width, width)),
            ("self_attn.o_proj", (width, width)),
            ("mlp.gate_proj", (intermediate, width)),
            ("mlp.up_proj", (intermediate, width)),
            ("mlp.down_proj", (width, intermediate)),
        ):
            expected[f"{prefix}.{name}.{'linear.weight' if clipped else 'weight'}"] = shape
            if clipped:
                for bound in ("input_min", "input_max", "output_min", "output_max"):
                    expected[f"{prefix}.{name}.{bound}"] = ()
    actual = {key for key in tensors if key.startswith(("model.vision_tower.", "model.embed_vision."))}
    if actual != set(expected):
        raise UnsupportedConversion("Gemma vision tensor inventory is outside the admitted E2B/E4B profile")
    for name, shape in expected.items():
        tensor = tensors[name]
        if tensor.shape != shape or tensor.dtype != "BF16":
            raise ConversionError("Gemma vision tensor shape or BF16 storage disagrees with its profile")
        output_name = name.removeprefix("model.vision_tower.")
        if name.startswith("model.embed_vision."):
            output_name = name.removeprefix("model.")
        operations.append(TensorOperation(f"vision.{output_name}", (tensor,), "BF16", shape))
    decoder = {**base.decoder_config, "gemma4_image_embeddings": model_dim != 5376}
    if model_dim == 5376:
        decoder.update(scalar_image_embeddings=True, bidirectional_image_attention=True)
    metadata = {**base.metadata, "stark.vision.family": contract["family"]["id"]}
    return replace(
        base,
        decoder_config=decoder,
        operations=tuple(sorted(operations, key=lambda operation: operation.name)),
        metadata=metadata,
    )
