"""Read Stark's attributed inventory without importing a model framework."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from .streaming import ConversionError, UnsupportedConversion

REFERENCE_REVISION = "831accb8252a6b6606c378048caeb44178f1a050"
CONFIG_ADAPTERS = {
    "HFQwen3Config": "qwen3",
    "HFQwen2Config": "qwen2",
    "HFLlamaConfig": "llama",
    "HFMistralConfig": "mistral",
    "HFGemma2Config": "gemma2",
    "HFGemma3TextConfig": "gemma3_text",
    "HFGemma3Config": "gemma3",
    "HFPhi3Config": "phi3",
    "HFGraniteConfig": "granite",
    "HFSmolLM3Config": "smollm3",
    "HFQwen35Config": "qwen3_5",
    "HFLFM2Config": "lfm2",
    "HFLlambaConfig": "llamba",
    "ModernBERTConfig": "modernbert",
}
ARCHITECTURE_CONFIGS = {adapter: config for config, adapter in CONFIG_ADAPTERS.items()}
# The upstream config class selects distinct dense and MoE tensor contracts.
ARCHITECTURE_CONFIGS["qwen3_5_moe"] = "HFQwen35Config"


def model_kind_for_architecture(architecture: str) -> str:
    """Architecture dispatch also binds the accepted root model kind."""
    if architecture not in ARCHITECTURE_CONFIGS:
        raise UnsupportedConversion("Checkpoint architecture is outside implemented coverage")
    return "classifier" if architecture == "modernbert" else "language"


def architecture_for_config(config: dict, expected: str | None = None) -> str:
    """Select a reusable adapter; its planner still validates every semantic field."""
    architecture = config.get("model_type")
    if not isinstance(architecture, str) or architecture not in ARCHITECTURE_CONFIGS:
        raise UnsupportedConversion("Checkpoint architecture is outside implemented coverage")
    if expected is not None and architecture != expected:
        raise ConversionError("Checkpoint architecture disagrees with the requested architecture")
    return architecture


def load_catalog() -> dict:
    """Return a fresh catalog so callers cannot mutate process-global dispatch."""
    try:
        value = json.loads(files("stark_convert").joinpath("catalog.json").read_text())
        entries = value["models"]
        if value["schema_version"] != 1 or value["reference"]["revision"] != REFERENCE_REVISION:
            raise ValueError()
        if not isinstance(entries, list) or len(entries) != value["entry_count"]:
            raise ValueError()
        extensions = value["extensions"]
        if not isinstance(extensions, list) or len(extensions) != value["extension_count"]:
            raise ValueError()
        all_entries = entries + extensions
        if any(not isinstance(entry, dict) or not isinstance(entry.get("repo"), str) for entry in all_entries):
            raise ValueError()
        if len({entry["repo"] for entry in all_entries}) != len(all_entries):
            raise ValueError()
    except (OSError, KeyError, ValueError, TypeError):
        raise ConversionError("Invalid conversion catalog") from None
    return value


def catalog_sha256() -> str:
    return hashlib.sha256(files("stark_convert").joinpath("catalog.json").read_bytes()).hexdigest()


def models() -> tuple[dict, ...]:
    catalog = load_catalog()
    return tuple(catalog["models"] + catalog["extensions"])


def get_model(repo: str | None) -> dict | None:
    return next((entry for entry in models() if entry["repo"] == repo), None)


def adapter_for_model(repo: str, model_type: str | None = None) -> str:
    entry = get_model(repo)
    if entry is None or entry["adapter"] is None:
        raise UnsupportedConversion("Model tuple is outside implemented conversion coverage")
    adapter = entry["adapter"]
    if model_type is not None and model_type != adapter:
        raise UnsupportedConversion("Model architecture disagrees with its catalog profile")
    return adapter


def required_files(entry: dict) -> tuple[dict, ...]:
    """Return sidecar rules; weights are described separately by container."""
    return tuple({"role": role, **spec} for role, spec in sorted(entry["files"].items()))
