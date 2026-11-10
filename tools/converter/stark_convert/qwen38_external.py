"""The external-table affine dialect handled by the native Qwen3.8 loader."""

from pathlib import PurePosixPath

from .profiles import generic_files
from .streaming import UnsupportedConversion

NGRAM_TABLE = "ngram_table.bin"
INDEX = "model.safetensors.index.json"
QUANTIZATION = "MLX-Serve mixed 4/8-bit"


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
