"""Source-bound starting choices for the optional beginner guide.

Recipes are deliberately small and deterministic. Source checkpoints have
recorded native chat evidence; this is neither a quality ranking nor proof of
OpenWork task competence. Resource numbers are conservative planning estimates,
not measurements or a runtime admission decision. The actual converter and
worker still validate every artifact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil

from ._build_metadata import open_regular
from .streaming import authorize_source_path

GIB = 1024**3
MIB = 1024**2
TASKS = frozenset({"everyday", "text", "files", "code", "unsure"})
MAX_EXCLUSIONS = 64


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    repo: str
    revision: str
    architecture: str
    metadata_profile: str | None
    quantization: str
    minimum_memory: int
    context_tokens: int
    attention_layers: int
    kv_heads: int
    head_dim: int
    qualification: str
    files: tuple[tuple[str, int, str], ...]

    @property
    def download_bytes(self):
        return sum(size for _name, size, _digest in self.files)

    @property
    def installed_bytes(self):
        # Repacking preserves the selected weight payload; allow an additional
        # 128 MiB for output headers, tokenizer, config and receipts.
        return sum(size for name, size, _digest in self.files if name.endswith(".safetensors")) + 128 * MIB

    @property
    def setup_bytes(self):
        # Keep source and converted copies, plus a 512 MiB setup reserve.
        return self.download_bytes + self.installed_bytes + 512 * MIB

    @property
    def runtime_memory(self):
        # The current dense generation route uses BF16 activation/KV state.
        # Budget both full-context KV buffers, the 1024-row attention suffix,
        # selected source weights and a 3 GiB workspace/tokenizer allowance.
        # Rounded-up planning values still do not grant native admission.
        weights = sum(size for name, size, _digest in self.files if name.endswith(".safetensors"))
        cache = 2 * (self.context_tokens + 1024) * self.attention_layers * self.kv_heads * self.head_dim * 2
        return ((weights + cache + 3 * GIB + GIB - 1) // GIB) * GIB


# Metadata checked without downloading weights on 2026-11-04 against the named
# Hugging Face revision APIs (blobs=true). Small files were SHA-256 checked;
# weight/tokenizer SHA-256 values are the Hub's LFS identities. Both repositories
# were public, ungated, and declared Apache-2.0 in their model cards.
RECIPES = (
    Recipe(
        "qwen3-4b-mlx4-4dcb3d10",
        "Qwen3 4B",
        "mlx-community/Qwen3-4B-4bit",
        "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
        "qwen3",
        "hf-generic",
        "4-bit",
        16 * GIB,
        32768,
        36,
        8,
        128,
        "This exact source recipe has recorded native chat and OpenWork protocol evidence; "
        "the current build still checks conversion and loading.",
        (
            ("config.json", 937, "b5efdcf3b0035a3638e7228dad4d85f5c4a23f156eb7cdb0b44c8366a5d34d9b"),
            (
                "model.safetensors",
                2263022529,
                "e240c0bdc0ebb0681bf0da0f98d9719fd6ebe269a3633f81542c13e81345651d",
            ),
            (
                "model.safetensors.index.json",
                63924,
                "f7825defe5865d179c3b593173d37056be5f202dcb7153985cf74e75ecf1628b",
            ),
            ("tokenizer.json", 11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            ("tokenizer_config.json", 9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
        ),
    ),
    Recipe(
        "qwen3-06b-bf16-c1899de2",
        "Qwen3 0.6B",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "qwen3",
        None,
        "BF16",
        16 * GIB,
        32768,
        28,
        8,
        128,
        "Source checkpoint exercised in native chat and bounded tool tests; "
        "general agent competence is not established.",
        (
            ("config.json", 726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
            ("generation_config.json", 239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            (
                "model.safetensors",
                1503300328,
                "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
            ),
            ("tokenizer.json", 11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            ("tokenizer_config.json", 9732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
        ),
    ),
)


def recipe_for(identifier):
    return next((recipe for recipe in RECIPES if recipe.id == identifier), None)


def _free_bytes(path):
    path = Path(path).expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def host_resources(data_dir):
    """Read aggregate local resources only; never return private paths."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        memory = psutil.virtual_memory()
        return {
            "supported": sys.platform == "darwin" and platform.machine() == "arm64",
            "physical_memory_bytes": memory.total,
            "available_memory_bytes": memory.available,
            "free_disk_bytes": _free_bytes(data_dir),
            "download_free_disk_bytes": _free_bytes(HF_HUB_CACHE),
        }
    except (OSError, ValueError, AttributeError):
        return {
            "supported": False,
            "physical_memory_bytes": None,
            "available_memory_bytes": None,
            "free_disk_bytes": None,
            "download_free_disk_bytes": None,
        }


def _small_file(root, name):
    with open_regular(authorize_source_path(root, root / name)) as stream:
        value = stream.read(2 * MIB + 1)
    if len(value) > 2 * MIB:
        raise ValueError()
    return value


def matches_recipe(path, recipe):
    """Require the complete exact source selection and config receipt binding.

    This metadata check does not replace safetensors/template/native admission.
    It never hashes or reads tensor payloads to populate the guide.
    """
    try:
        root = Path(path)
        config = _small_file(root, "config.json")
        receipt = json.loads(_small_file(root, "conversion.json"))
        sources = [{"repo": recipe.repo, "revision": recipe.revision}]
        profile = (
            {"architecture": recipe.architecture, "metadata_profile": recipe.metadata_profile}
            if recipe.metadata_profile is not None
            else None
        )
        source_files = receipt.get("source_files")
        expected = {(recipe.repo, recipe.revision, size, digest) for _name, size, digest in recipe.files}
        if (
            receipt.get("sources") != sources
            or receipt.get("conversion_profile") != profile
            or profile is None
            and "conversion_profile" in receipt
            or not isinstance(source_files, list)
            or len(source_files) != len(expected)
            or any(
                not isinstance(item, dict) or set(item) != {"repo", "revision", "size", "sha256"}
                for item in source_files
            )
            or {(item["repo"], item["revision"], item["size"], item["sha256"]) for item in source_files} != expected
        ):
            return False
        identity = {"sources": sources, "files": source_files}
        if profile is not None:
            identity["conversion_profile"] = profile
        config_hash = receipt.get("config_sha256") or receipt.get("outputs", {}).get("config.json")
        return (
            receipt.get("source_manifest_sha256")
            == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            and config_hash == hashlib.sha256(config).hexdigest()
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False


def installation_allowed(recipe, host, *, installed=False):
    if host.get("supported") is not True:
        return False
    physical = host.get("physical_memory_bytes")
    if type(physical) is not int or physical < recipe.minimum_memory:
        return False
    if installed:
        return True
    available = host.get("available_memory_bytes")
    space, download_space = host.get("free_disk_bytes"), host.get("download_free_disk_bytes")
    return (
        type(available) is int
        and available >= 2 * GIB
        and type(space) is int
        and space >= recipe.setup_bytes
        and type(download_space) is int
        and download_space >= recipe.download_bytes
    )


def recommendation_report(host, installed, excluded_recipe_ids=()):
    items = []
    for recipe in RECIPES:
        ids = list(installed.get(recipe.id, ()))
        if recipe.id in excluded_recipe_ids or not installation_allowed(recipe, host, installed=bool(ids)):
            continue
        items.append(
            {
                "id": recipe.id,
                "name": recipe.name,
                "repo": recipe.repo,
                "revision": recipe.revision,
                "artifact_mode": "text",
                "quantization": recipe.quantization,
                "download_bytes": recipe.download_bytes,
                "installed_bytes_estimate": recipe.installed_bytes,
                "setup_disk_bytes_estimate": recipe.setup_bytes,
                "runtime_memory_bytes_estimate": recipe.runtime_memory,
                "context_tokens": recipe.context_tokens,
                "installed_ids": ids,
                "license": "Apache-2.0",
                "qualification": recipe.qualification,
                "caveats": [
                    "Memory and setup storage are conservative planning estimates, not measured fit guarantees.",
                    "Start the model to check it on this Mac. Speed, answer quality, and OpenWork task success vary.",
                ],
            }
        )
    # Reuse an eligible installed recipe before offering a new download.
    items.sort(key=lambda item: not bool(item["installed_ids"]))
    physical = host.get("physical_memory_bytes")
    message = "Choose a starting model."
    if not items:
        message = (
            "No checked starting option is available for these resources. Check again or browse models in Advanced."
        )
        if type(physical) is int and physical < min(recipe.minimum_memory for recipe in RECIPES):
            message = (
                "This guided OpenWork setup currently needs at least 16 GB of memory. "
                "You can browse smaller chat models in Advanced."
            )
    return {
        "schema_version": 1,
        "status": "ready" if items else "unavailable",
        "message": message,
        "host": dict(host),
        "items": items,
    }
