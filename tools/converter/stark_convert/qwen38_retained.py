"""Repository-independent admission for retained Flash Next image checkpoints.

The BF16 tower layout follows the pinned Lily implementation recorded in
vendor/lily-core/UPSTREAM.md. Header checks establish structural compatibility;
the installed worker must still pass local image execution checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import qwen38_external, qwen38_kquant
from .adapters.qwen38 import architecture_config, build_qwen38_plan
from .checkpoint_metadata import CheckpointMetadata
from .hub_headers import HeaderAccessError, checkpoint_headers, read_header
from .profiles import generic_files
from .streaming import ConversionError, UnsupportedConversion
from .vision_family import _PROCESSOR, family_processor_filenames, normalize_family_processor

LAYOUT = "qwen38-retained-affine"
SUPPORTED_TEMPLATE_HASHES = frozenset(
    {
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    }
)


def is_retained_affine(config: dict) -> bool:
    """Identify implemented storage contracts independently of publisher names."""
    return (
        qwen38_kquant.is_kquant_affine(config)
        or qwen38_external.is_external_affine(config)
        or (
            config.get("model_type") == "qwen4_exp"
            and isinstance(config.get("lily"), dict)
            and config["lily"].get("format") == "qwen4_exp-affine-v1"
        )
    )


def quantization_name(config: dict) -> str:
    if qwen38_kquant.is_kquant_affine(config):
        return qwen38_kquant.QUANTIZATION
    return qwen38_external.QUANTIZATION if qwen38_external.is_external_affine(config) else "Lily affine 4/8-bit"


def auxiliary_metadata_filenames(config):
    return (qwen38_kquant.PLE_MANIFEST,) if qwen38_kquant.is_kquant_affine(config) else ()


def retained_files(repo, names, config):
    """Bind all processing sidecars and auxiliary storage to this source."""
    if not is_retained_affine(config):
        raise UnsupportedConversion("This Flash Next weight format has no retained image loader.")
    base = (
        qwen38_kquant.kquant_files(repo, names, config)
        if qwen38_kquant.is_kquant_affine(config)
        else (
            qwen38_external.external_files(repo, names)
            if qwen38_external.is_external_affine(config)
            else (
                *generic_files(repo, names),
                *(
                    {"role": "source_notice", "repo": repo, "filename": name}
                    for name in ("LICENSE", "LICENSE.txt", "README.md")
                    if name in names
                ),
            )
        )
    )
    return (
        *base,
        *({"role": "processor_config", "repo": repo, "filename": name} for name in family_processor_filenames(names)),
    )


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate image processing metadata key")
        value[key] = item
    return value


def validate_processing_metadata(names, read):
    """Require the checkpoint's supported recipe before downloading weights."""
    for name in family_processor_filenames(names):
        try:
            value = json.loads(read(name, 65536), object_pairs_hook=_unique_object)
            if normalize_family_processor(value, name) != _PROCESSOR:
                raise ValueError()
        except (ValueError, TypeError, UnicodeError, RecursionError, AttributeError) as error:
            raise UnsupportedConversion(
                "Flash Next image processing metadata is outside the supported recipe."
            ) from error
    if qwen38_kquant.PLE_MANIFEST in names:
        try:
            config = json.loads(read("config.json", 1048576), object_pairs_hook=_unique_object)
            manifest = json.loads(read(qwen38_kquant.PLE_MANIFEST, 1048576), object_pairs_hook=_unique_object)
            qwen38_kquant.validate_manifest(config, manifest)
        except (ValueError, TypeError, UnicodeError, RecursionError, AttributeError) as error:
            raise UnsupportedConversion("Flash Next PLE source metadata is invalid.") from error
    try:
        tokenizer = json.loads(read("tokenizer_config.json", 2 * 1024 * 1024), object_pairs_hook=_unique_object)
        inline = tokenizer.get("chat_template")
        if isinstance(inline, list):
            inline = next((item.get("template") for item in inline if item.get("name") == "default"), None)
        if "chat_template.jinja" in names:
            payload = read("chat_template.jinja", 262144)
            template = payload.decode() if isinstance(payload, bytes) else payload
            if inline is not None and inline != template:
                raise ValueError()
        else:
            template = inline
        if (
            not isinstance(template, str)
            or not 0 < len(template.encode()) <= 262144
            or hashlib.sha256(template.encode()).hexdigest() not in SUPPORTED_TEMPLATE_HASHES
        ):
            raise ValueError()
    except (ValueError, TypeError, UnicodeError, RecursionError, AttributeError) as error:
        raise UnsupportedConversion("Flash Next images require a complete supported chat-template source.") from error


def validate_vision_config(config):
    """Validate the declared tower and image/text fusion before weight I/O."""
    if not is_retained_affine(config):
        raise UnsupportedConversion("This Flash Next weight format has no retained image loader.")
    if not isinstance(config.get("vision_config"), dict):
        raise UnsupportedConversion("This Flash Next checkpoint does not declare an image tower.")
    if qwen38_kquant.is_kquant_affine(config):
        architecture = qwen38_kquant.validate_config(config)
    elif qwen38_external.is_external_affine(config):
        architecture = qwen38_external._architecture(config)
    else:
        architecture = architecture_config(config)[0]
    validate_native_geometry(architecture)
    qwen38_external._validate_vision_metadata(config, architecture)
    if isinstance(config.get("lily"), dict):
        lily = config["lily"]
        vision = lily.get("vision")
        if (
            not isinstance(vision, dict)
            or vision.get("dtype") != "bf16"
            or type(vision.get("tensors")) is not int
            or vision["tensors"] != 9 + 12 * config["vision_config"]["depth"]
            or not isinstance(lily.get("dropped", []), list)
            or "model.visual." in lily.get("dropped", [])
        ):
            raise UnsupportedConversion("The Lily checkpoint does not declare its retained BF16 image tower.")
    return architecture


def validate_native_geometry(architecture):
    """Match the native Flash Next kernel geometry before any model payload."""
    layers = architecture.get("layer_types", [])
    rope = architecture.get("rope_parameters", {})
    if (
        any(architecture.get(key) != value for key, value in qwen38_external._NATIVE_TEXT_GEOMETRY.items())
        or not 1 <= len(layers) <= 48
        or any(
            layer != ("full_attention" if index % 4 == 3 else "linear_attention") for index, layer in enumerate(layers)
        )
        or architecture.get("num_hidden_layers") != len(layers)
        or architecture["head_dim"] * rope.get("partial_rotary_factor", 0) != 64
        or rope.get("mrope_section") != [11, 11, 10]
        or rope.get("mrope_interleaved") is not True
        or architecture.get("ple_layer_ids") not in ([], [2])
        or architecture.get("ple_layer_ids")
        and architecture.get("ngram_size") != 3
    ):
        raise UnsupportedConversion("The checkpoint geometry is outside Stark's native Flash Next image kernels.")


def vision_tensor_shapes(config):
    """Mirror the exact BF16 names and shapes consumed by vision_weights.rs."""
    validate_vision_config(config)
    v = config["vision_config"]
    h, intermediate, out = v["hidden_size"], v["intermediate_size"], v["out_hidden_size"]
    merge = h * v["spatial_merge_size"] ** 2
    shapes = {
        "patch_embed.proj.weight": (h, v["in_channels"], v["temporal_patch_size"], v["patch_size"], v["patch_size"]),
        "patch_embed.proj.bias": (h,),
        "pos_embed.weight": (v["num_position_embeddings"], h),
        "merger.norm.weight": (h,),
        "merger.norm.bias": (h,),
        "merger.linear_fc1.weight": (merge, merge),
        "merger.linear_fc1.bias": (merge,),
        "merger.linear_fc2.weight": (out, merge),
        "merger.linear_fc2.bias": (out,),
    }
    for layer in range(v["depth"]):
        shapes.update(
            {
                f"blocks.{layer}.{name}": shape
                for name, shape in {
                    "norm1.weight": (h,),
                    "norm1.bias": (h,),
                    "attn.qkv.weight": (3 * h, h),
                    "attn.qkv.bias": (3 * h,),
                    "attn.proj.weight": (h, h),
                    "attn.proj.bias": (h,),
                    "norm2.weight": (h,),
                    "norm2.bias": (h,),
                    "mlp.linear_fc1.weight": (intermediate, h),
                    "mlp.linear_fc1.bias": (intermediate,),
                    "mlp.linear_fc2.weight": (h, intermediate),
                    "mlp.linear_fc2.bias": (h,),
                }.items()
            }
        )
    return {"model.visual." + name: shape for name, shape in shapes.items()}


def validate_vision_layout(config, tensors):
    expected = vision_tensor_shapes(config)
    actual = {}
    for name, info in tensors.items():
        alias = qwen38_external._alias(name)
        if alias.startswith("model.visual."):
            if alias in actual:
                raise UnsupportedConversion("Flash Next image tensors have conflicting names.")
            actual[alias] = (info.dtype, info.shape) if hasattr(info, "dtype") else info
    if set(actual) != set(expected):
        raise UnsupportedConversion("Flash Next requires a complete BF16 image tower with supported tensor names.")
    if qwen38_kquant.is_kquant_affine(config):
        # Indexed MLX exports use a channels-last Conv3d patch kernel. The
        # native indexed loader transposes these BF16 encodings once at load.
        patch = "model.visual.patch_embed.proj.weight"
        out, channels, frames, height, width = expected[patch]
        if actual[patch] == ("BF16", (out, frames, height, width, channels)):
            actual[patch] = ("BF16", expected[patch])
    if any(actual[name] != ("BF16", shape) for name, shape in expected.items()):
        raise UnsupportedConversion(
            "Flash Next image tensors require the supported BF16 shapes; quantized towers are unavailable."
        )


def validate_layout(config, tensors, table=None):
    """Validate both the native text storage and complete image tower headers."""
    if qwen38_kquant.is_kquant_affine(config):
        qwen38_kquant.validate_layout(config, tensors)
    elif qwen38_external.is_external_affine(config):
        if table is None:
            raise UnsupportedConversion("External Flash Next requires its n-gram table.")
        qwen38_external.validate_layout(
            config,
            {name: (info.dtype, info.shape) for name, info in tensors.items()},
            {name: (info.dtype, info.shape) for name, info in table.items()},
        )
    else:
        build_qwen38_plan(Path("."), checkpoint_metadata=CheckpointMetadata(config, tensors))
    validate_vision_layout(config, tensors)


def verify_hub_layout(hub, repo, revision, config, shards, index, check_deadline, sizes=None):
    """Admit image tensor headers with pinned, bounded range reads only."""
    try:
        tensors = checkpoint_headers(hub, repo, revision, shards, index, sizes or {}, check_deadline)
        table = (
            read_header(
                hub,
                repo,
                revision,
                qwen38_external.NGRAM_TABLE,
                check_deadline,
                (sizes or {}).get(qwen38_external.NGRAM_TABLE),
            )
            if qwen38_external.is_external_affine(config)
            else None
        )
        validate_layout(config, tensors, table)
        if qwen38_kquant.is_kquant_affine(config):
            from .model_support import check_hub_metadata_size, read_metadata_bytes

            name = qwen38_kquant.PLE_MANIFEST
            check_deadline()
            check_hub_metadata_size(hub, repo, name, revision, 1048576)
            path = Path(hub.hf_hub_download(repo_id=repo, filename=name, revision=revision))
            manifest = json.loads(
                read_metadata_bytes(path, root=path.parent, maximum=1048576), object_pairs_hook=_unique_object
            )
            check_deadline()
            qwen38_kquant.validate_manifest(config, manifest, tensors)
    except (ConversionError, UnsupportedConversion):
        raise
    except HeaderAccessError:
        raise
    except Exception as error:
        raise HeaderAccessError() from error
