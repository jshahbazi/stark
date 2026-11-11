"""The external-table affine dialect handled by the native Qwen3.8 loader."""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from .adapters.qwen38 import describe_text_weights, text_architecture
from .hub_headers import HeaderAccessError
from .hub_headers import read_range as _read_range
from .profiles import generic_files
from .streaming import ConversionError, UnsupportedConversion, parse_safetensors_header

NGRAM_TABLE = "ngram_table.bin"
INDEX = "model.safetensors.index.json"
QUANTIZATION = "MLX-Serve mixed 4/8-bit"
UNVERIFIED = "Could not verify the tensor headers. Refresh the catalog to retry."
UNSUPPORTED = "The tensor layout does not match Stark’s Qwen3.8 text loader."
_MAX_HEADER = 128 * 1024
_NGRAM_STEM = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
# Keep these geometry constraints in sync with Qwen4ExpConfig::validate_flash_next.
_NATIVE_TEXT_GEOMETRY = {
    "hidden_size": 2560,
    "vocab_size": 248320,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_n_heads": 4,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "heads_per_ngram": 8,
}


class LayoutError(UnsupportedConversion):
    """A bounded, user-facing reason produced by the layout checker."""


def is_external_affine(config: dict) -> bool:
    """Recognize the storage dialect from config, independent of the repository."""
    if config.get("model_type") != "qwen4_exp" or "lily" in config:
        return False
    text = config.get("text_config")
    quantization = config.get("quantization")
    table = config.get("ngram_table")
    return (
        isinstance(text, dict)
        and text.get("model_type") == "qwen4_exp_text"
        and text.get("ngram_size", 3) == 3
        and text.get("ple_layer_ids") == [2]
        and config.get("qwen4_norm_convention", "folded") == "folded"
        and isinstance(quantization, dict)
        and set(quantization) <= {"mode", "bits", "group_size", "calibration"}
        and quantization.get("mode") == "affine"
        and type(quantization.get("bits")) is int
        and quantization["bits"] == 4
        and type(quantization.get("group_size")) is int
        and quantization["group_size"] == 64
        and config.get("quantization_config", quantization) == quantization
        and isinstance(table, dict)
        and table.get("file") == NGRAM_TABLE
        and type(table.get("bits")) is int
        and table["bits"] == 4
        and type(table.get("group_size")) is int
        and table["group_size"] == 32
    )


def external_files(repo, names):
    if NGRAM_TABLE not in names:
        raise UnsupportedConversion("External Qwen3.8 requires ngram_table.bin")
    if INDEX not in names and "model.safetensors" not in names:
        raise UnsupportedConversion("External Qwen3.8 requires native checkpoint shards")
    return (
        *generic_files(repo, names),
        {"role": "ngram_table", "repo": repo, "filename": NGRAM_TABLE},
        *(
            {"role": "source_notice", "repo": repo, "filename": name}
            for name in ("LICENSE", "LICENSE.txt", "README.md")
            if name in names
        ),
    )


def supported_shards(shards, indexed):
    # The native loader accepts the standard single-file name or a flat index.
    return (
        1 <= len(shards) <= 512
        and all(
            PurePosixPath(name).name == name and "\\" not in name and name.endswith(".safetensors") for name in shards
        )
        and (indexed or tuple(shards) == ("model.safetensors",))
    )


def _architecture(config):
    text = config["text_config"]
    architecture = text_architecture(
        {
            **config,
            "text_config": {
                "heads_per_ngram": 8,
                "ngram_size": 3,
                "ple_conv_kernel_size": 4,
                **text,
                "ple_embed_dim": text.get("hidden_size")
                if text.get("ple_embed_dim") is None
                else text["ple_embed_dim"],
                "split_ngram_parts": 1,
            },
        }
    )
    layers = architecture["layer_types"]
    if (
        not 2 <= len(layers) <= 48
        or any(architecture[key] != value for key, value in _NATIVE_TEXT_GEOMETRY.items())
        or any(
            layer != ("full_attention" if index % 4 == 3 else "linear_attention") for index, layer in enumerate(layers)
        )
        or architecture["rope_parameters"]["partial_rotary_factor"] != 0.25
    ):
        raise LayoutError("The checkpoint geometry is outside Stark’s Qwen3.8 Flash Next loader.")
    base = text.get("ngram_vocab_size_base", 20_000_000)
    divisor = text.get("make_ngram_vocab_size_divisible_by", 128)
    seed = text.get("ngram_seed", 1234)
    if (
        type(base) is not int
        or not 0 < base <= 1_000_000_000
        or type(divisor) is not int
        or not 0 < divisor <= 65536
        or type(seed) is not int
        or not 0 < seed <= 2**64 - 1
    ):
        raise LayoutError("The checkpoint’s n-gram table configuration is unsupported.")

    def prime(value):
        return value == 2 or value > 2 and value % 2 and all(value % i for i in range(3, math.isqrt(value) + 1, 2))

    # The native external loader pads the sum of the next 16 prime vocabulary sizes.
    value = base - 1
    rows = 0
    for _ in range(2 * architecture["heads_per_ngram"]):
        value += 1
        while not prime(value):
            value += 1
        rows += value
    rows = (rows + divisor - 1) // divisor * divisor
    if rows > 2**32 - 1:
        raise LayoutError("The checkpoint’s n-gram table is too large for Stark’s loader.")
    architecture["ple"] = {"padded_vocab_size": rows}
    return architecture


def _validate_vision_metadata(config, architecture):
    # Qwen4ExpConfig::from_json checks declared vision geometry even for text.
    vision = config.get("vision_config")
    if vision is None:
        return
    bounds = (
        ("hidden_size", 4096),
        ("intermediate_size", 32768),
        ("num_heads", 64),
        ("depth", 64),
        ("num_position_embeddings", 16384),
    )
    reason = "The declared vision configuration is incompatible with Stark’s Qwen3.8 loader."
    if not isinstance(vision, dict) or any(
        type(vision.get(key)) is not int or not 0 < vision[key] <= maximum for key, maximum in bounds
    ):
        raise LayoutError(reason)
    h, intermediate, heads, depth, positions = (vision[key] for key, _ in bounds)
    merger = architecture["hidden_size"]
    elements = (
        h * 1536
        + positions * h
        + depth * (4 * h * h + 2 * h * intermediate + 9 * h + intermediate)
        + 16 * h * h
        + 4 * h * merger
        + 6 * h
        + merger
    )
    rope = architecture["rope_parameters"]
    if (
        any(
            type(vision.get(key)) is not int or vision[key] != expected
            for key, expected in (
                ("patch_size", 16),
                ("in_channels", 3),
                ("out_hidden_size", merger),
                ("temporal_patch_size", 2),
                ("spatial_merge_size", 2),
            )
        )
        or elements * 2 > 4 * 1024**3
        or h % (2 * heads)
        or math.isqrt(positions) ** 2 != positions
        or vision.get("hidden_act") != "gelu_pytorch_tanh"
        or vision.get("deepstack_visual_indexes", []) != []
        or rope["mrope_section"] != [11, 11, 10]
        or any(type(value) is not int for value in rope["mrope_section"])
        or rope["mrope_interleaved"] is not True
        or any(
            type(config.get(key)) is not int or not 0 <= config[key] <= 2**32 - 1
            for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
        )
    ):
        raise LayoutError(reason)


def _alias(name):
    for source, native in (
        ("language_model.model.", "model.language_model."),
        ("language_model.lm_head.", "lm_head."),
        ("language_model.mtp.", "mtp."),
        ("vision_tower.", "model.visual."),
    ):
        if name.startswith(source):
            name = native + name.removeprefix(source)
            break
    return name.replace(".mlp.switch_mlp.", ".mlp.experts.")


def validate_layout(config, tensors, table):
    """Check the complete text and declared draft tensor layouts, without payloads."""
    architecture = _architecture(config)
    _validate_vision_metadata(config, architecture)
    if set(table) != {"weight", "scales", "biases"}:
        raise LayoutError("The external n-gram table must contain weight, scale, and bias planes.")
    canonical = {}
    for name, record in tensors.items():
        alias = _alias(name)
        if alias in canonical or alias.startswith(_NGRAM_STEM + "."):
            raise LayoutError("The checkpoint contains conflicting tensor names.")
        canonical[alias] = record
    canonical.update({f"{_NGRAM_STEM}.{name}": record for name, record in table.items()})
    required = set()

    def control(name, shape, integers=False):
        required.add(name)
        actual = canonical.get(name)
        if actual is None:
            raise LayoutError("The checkpoint is missing required Qwen3.8 tensors.")
        dtype, actual_shape = actual
        if integers:
            allowed = {"I64", "U64"}
        elif name.endswith((".linear_attn.norm.weight", ".linear_attn.A_log")):
            allowed = {"BF16", "F32"}
        else:
            allowed = {"BF16"}
        transposed = name.endswith(".conv1d.weight") and actual_shape == (shape[0], shape[2], shape[1])
        if dtype not in allowed or (actual_shape != shape and not transposed):
            raise LayoutError(UNSUPPORTED)

    def matrix(stem, shape, *, group=None, fixed_q8=False, embedding=False, ngram=False):
        if stem.endswith((".block_inject_weight", ".mlp.shared_expert_gate")):
            control(stem + ".weight", shape)
            return
        bits = 8 if fixed_q8 or group in {"attn", "gdn", "shared", "head"} else 4
        group_size = 32 if ngram else 64
        *leading, columns = shape
        weight, scales, biases = (stem + "." + plane for plane in ("weight", "scales", "biases"))
        required.update((weight, scales, biases))
        if any(name not in canonical for name in (weight, scales, biases)):
            raise LayoutError("The checkpoint is missing required Qwen3.8 affine tensor planes.")
        packed = canonical[weight]
        params = canonical[scales]
        parameter_shape = (*leading, columns // group_size)
        if packed == ("U32", (*leading, columns * bits // 32)) and params == canonical[biases] == (
            "BF16",
            parameter_shape,
        ):
            return
        # Read the actual expert packing rather than the config's default width.
        if (
            ".mlp.experts." in stem
            and packed[0] == "U32"
            and packed[1][:-1] == tuple(leading)
            and params[0] == "BF16"
            and params[1][:-1] == tuple(leading)
            and packed[1]
            and params[1]
            and params[1][-1] > 0
            and (packed[1][-1] * 32) % columns == 0
            and columns % params[1][-1] == 0
            and params == canonical[biases]
            and (packed[1][-1], params[1][-1]) != (columns * bits // 32, columns // group_size)
        ):
            actual_bits = packed[1][-1] * 32 // columns
            actual_group = columns // params[1][-1]
            raise LayoutError(
                f"Expert weights use {actual_bits}-bit packing with groups of {actual_group}. "
                "Stark’s loader requires 4-bit packing with groups of 64 for these weights."
            )
        raise LayoutError(UNSUPPORTED)

    mtp = config["text_config"].get("mtp")
    if mtp is not None and (
        not isinstance(mtp, dict)
        or mtp.get("hybrid") is not True
        or type(mtp.get("num_hidden_layers")) is not int
        or mtp["num_hidden_layers"] != 1
        or mtp.get("layer_types") != ["full_attention"]
        or mtp.get("mtp_use_hidden_state_from_layer") is not None
        or type(mtp.get("rope_theta")) not in (int, float)
        or not 0 < mtp["rope_theta"] < math.inf
    ):
        raise LayoutError("The checkpoint’s draft head is outside Stark’s Qwen3.8 loader.")
    describe_text_weights(architecture, canonical, matrix, control, include_mtp=mtp is not None)
    if any(not name.startswith("model.visual.") for name in canonical.keys() - required):
        raise LayoutError("The checkpoint contains tensors outside Stark’s Qwen3.8 text layout.")


def verify_hub_layout(hub, repo, revision, config, shards, index, check_deadline, sizes=None):
    """Check the pinned files using HTTP ranges that end at the header boundary."""

    def read_header(name):
        url = hub.hf_hub_url(repo_id=repo, filename=name, revision=revision)
        prefix, total = _read_range(hub, url, 0, 7, check_deadline, (sizes or {}).get(name))
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= _MAX_HEADER or 8 + length >= total:
            raise LayoutError("The checkpoint contains an invalid or oversized tensor header.")
        data, _ = _read_range(hub, url, 8, 7 + length, check_deadline, total)
        if not data.startswith(b"{"):
            raise LayoutError(UNSUPPORTED)
        _, headers = parse_safetensors_header(data, total)
        return {name: (dtype, shape) for name, (dtype, shape, _, _) in headers.items()}

    pool = ThreadPoolExecutor(max_workers=8)
    try:
        tensors = {}
        names = (*shards, NGRAM_TABLE)
        for name, header in zip(names, pool.map(read_header, names), strict=True):
            if name == NGRAM_TABLE:
                table = header
                break
            if tensors.keys() & header.keys() or index is not None and any(index.get(key) != name for key in header):
                raise LayoutError("The checkpoint’s shard index disagrees with the tensor headers.")
            tensors.update(header)
        if index is not None and tensors.keys() != index.keys():
            raise LayoutError("The checkpoint’s shard index disagrees with the tensor headers.")
        validate_layout(config, tensors, table)
    except LayoutError:
        raise
    except (ConversionError, UnsupportedConversion) as error:
        raise LayoutError(UNSUPPORTED) from error
    except Exception as error:
        raise HeaderAccessError() from error
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
