"""Indexed affine Flash Next storage with a range-sharded PLE table.

Admission depends on actual tensor headers and declared per-projection storage.
The adapter changes names and file views only; it never requantizes payloads.
"""

import math
import re

from . import qwen38_external
from .adapters.qwen38 import describe_text_weights
from .profiles import generic_files
from .streaming import UnsupportedConversion

PLE_MANIFEST = "ple_manifest.json"
QUANTIZATION = "MLX mixed 4/8-bit with indexed PLE"
_PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."


def is_kquant_affine(config):
    text = config.get("text_config")
    quant = config.get("quantization")
    return (
        config.get("model_type") == "qwen4_exp"
        and "lily" not in config
        and isinstance(text, dict)
        and text.get("model_type") == "qwen4_exp_text"
        and isinstance(text.get("ple_storage"), dict)
        and text["ple_storage"].get("manifest") == PLE_MANIFEST
        and isinstance(quant, dict)
        and quant.get("mode") == "affine"
        and type(quant.get("bits")) is int
        and quant["bits"] == 4
        and type(quant.get("group_size")) is int
        and quant["group_size"] == 64
    )


def alias(name):
    name = qwen38_external._alias(name + ".")[:-1]
    return re.sub(r"\.ngram_embedding\.shards\.([0-9]+)(?=\.|$)", r".ngram_embedding.shard_\1", name)


def architecture(config):
    if not is_kquant_affine(config) or config.get("qwen4_norm_convention", "delta") != "delta":
        raise UnsupportedConversion("Unsupported indexed Flash Next affine storage configuration.")
    result = qwen38_external._architecture(config)
    parts = config["text_config"].get("split_ngram_parts")
    if type(parts) is not int or not 1 <= parts <= 128 or result["ple"]["padded_vocab_size"] % parts:
        raise UnsupportedConversion("The Flash Next PLE shard count is outside the supported layout.")
    result["split_ngram_parts"] = parts
    return result


def _descriptors(config):
    result = architecture(config)
    specs = {}

    def control(_name, _shape, integers=False):
        pass

    def matrix(stem, shape, *, group=None, fixed_q8=False, embedding=False, ngram=False):
        if stem.endswith(".block_inject_weight"):
            specs[stem] = False
            return
        bits = 4 if ngram else 8 if fixed_q8 or group in {"attn", "gdn", "shared", "head", "embed"} else 4
        size = 32 if ngram else 128 if group == "shared" and stem.startswith("model.language_model.") else 64
        specs[stem] = {"bits": bits, "group_size": size, "mode": "affine"}

    describe_text_weights(result, {}, matrix, control, include_mtp=config["text_config"].get("mtp") is not None)
    return result, specs


def validate_config(config):
    result, specs = _descriptors(config)
    quant = config["quantization"]
    if config.get("quantization_config", quant) != quant:
        raise UnsupportedConversion("Flash Next quantization metadata disagrees.")
    supplied = {}
    for name, spec in quant.items():
        if name in {"bits", "group_size", "mode"}:
            continue
        canonical = alias(name)
        if canonical in supplied or canonical not in specs or spec != specs[canonical]:
            raise UnsupportedConversion("A Flash Next projection declares unsupported affine precision or group size.")
        if spec is not False and (
            not isinstance(spec, dict) or type(spec.get("bits")) is not int or type(spec.get("group_size")) is not int
        ):
            raise UnsupportedConversion("Invalid Flash Next affine projection metadata.")
        supplied[canonical] = spec
    default = {"bits": 4, "group_size": 64, "mode": "affine"}
    if any(spec != default and supplied.get(name) != spec for name, spec in specs.items()):
        raise UnsupportedConversion("Flash Next per-projection precision metadata is incomplete.")
    qwen38_external._validate_vision_metadata(config, result)
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
        raise UnsupportedConversion("Unsupported indexed Flash Next draft-head configuration.")
    return result


def kquant_files(repo, names, config):
    validate_config(config)
    if PLE_MANIFEST not in names or qwen38_external.INDEX not in names:
        raise UnsupportedConversion("Indexed Flash Next requires its PLE manifest and tensor index.")
    return (
        *generic_files(repo, names),
        {"role": "ple_manifest", "repo": repo, "filename": PLE_MANIFEST},
        *(
            {"role": "source_notice", "repo": repo, "filename": name}
            for name in ("LICENSE", "LICENSE.txt", "README.md")
            if name in names
        ),
    )


files = kquant_files


def validate_layout(config, tensors):
    result = validate_config(config)
    _, specs = _descriptors(config)
    canonical = {}
    for name, info in tensors.items():
        name = alias(name)
        if name in canonical:
            raise UnsupportedConversion("Indexed Flash Next tensor names collide.")
        canonical[name] = (info.dtype, info.shape) if hasattr(info, "dtype") else info
    required = set()

    def control(name, shape, integers=False):
        required.add(name)
        dtype, actual = canonical.get(name, (None, None))
        allowed = (
            {"I64", "U64"}
            if integers
            else {"BF16", "F32"}
            if name.endswith((".linear_attn.norm.weight", ".linear_attn.A_log"))
            else {"BF16"}
        )
        transposed = name.endswith(".conv1d.weight") and actual == (shape[0], shape[2], shape[1])
        if dtype not in allowed or actual != shape and not transposed:
            raise UnsupportedConversion("An indexed Flash Next control tensor has an unsupported shape or dtype.")

    def matrix(stem, shape, **_options):
        spec = specs[stem]
        if spec is False:
            control(stem + ".weight", shape)
            return
        *leading, columns = shape
        bits, group = spec["bits"], spec["group_size"]
        if columns % group:
            raise UnsupportedConversion("Invalid indexed Flash Next affine group geometry.")
        for plane, dtype, width in (
            ("weight", "U32", columns * bits // 32),
            ("scales", "BF16", columns // group),
            ("biases", "BF16", columns // group),
        ):
            name = stem + "." + plane
            required.add(name)
            if canonical.get(name) != (dtype, (*leading, width)):
                raise UnsupportedConversion("An indexed Flash Next affine tensor disagrees with its declared storage.")

    describe_text_weights(result, canonical, matrix, control, include_mtp=config["text_config"].get("mtp") is not None)
    if any(not name.startswith("model.visual.") for name in canonical.keys() - required):
        raise UnsupportedConversion("The indexed Flash Next tensor inventory contains unsupported tensors.")


def validate_manifest(config, manifest, tensors=None):
    result = validate_config(config)
    parts = result["split_ngram_parts"]
    rows = result["ple"]["padded_vocab_size"]
    width = result["ple_embed_dim"] // ((result["ngram_size"] - 1) * result["heads_per_ngram"])
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("version")) is not int
        or manifest.get("version") != 2
        or manifest.get("source_root") != "."
        or manifest.get("layout") != "safetensors_ranges"
        or manifest.get("row_width") != width
        or type(manifest.get("row_width")) is not int
        or manifest.get("row_count") != rows
        or type(manifest.get("row_count")) is not int
        or manifest.get("quantization") != {"bits": 4, "group_size": 32, "mode": "affine"}
        or not isinstance(manifest.get("shards"), list)
        or len(manifest["shards"]) != parts
    ):
        raise UnsupportedConversion("The Flash Next PLE range manifest is outside the supported storage contract.")
    canonical = {alias(name): info for name, info in (tensors or {}).items()}
    for index, shard in enumerate(manifest["shards"]):
        count = rows // parts
        if (
            not isinstance(shard, dict)
            or type(shard.get("row_start")) is not int
            or type(shard.get("row_count")) is not int
            or shard.get("row_start") != index * count
            or shard.get("row_count") != count
        ):
            raise UnsupportedConversion("The Flash Next PLE range manifest has missing or overlapping rows.")
        for plane, dtype, columns in (
            ("weight", "U32", width // 8),
            ("scales", "BF16", width // 32),
            ("biases", "BF16", width // 32),
        ):
            value = shard.get(plane)
            if (
                not isinstance(value, dict)
                or value.get("file") != "ple.safetensors"
                or value.get("dtype") != dtype
                or value.get("shape") != [count, columns]
                or type(value.get("offset")) is not int
                or not 8 <= value["offset"] <= 2**63 - 1
            ):
                raise UnsupportedConversion("The Flash Next PLE range points outside its supported tensor file.")
            if tensors is not None:
                info = canonical.get(f"{_PLE}shard_{index}.{plane}")
                if info is None or info.path.name != value["file"] or info.offset != value["offset"]:
                    raise UnsupportedConversion(
                        "The Flash Next PLE ranges disagree with the tensor index and headers."
                    )
