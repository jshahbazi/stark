"""Source-bound starting choices for the optional beginner guide.

Pinned starters have recorded native chat evidence. Additional candidates come
from metadata-preflighted catalogue revisions with complete dense memory plans.
Ranking is an explainable resource/capacity heuristic, not measured answer
quality or proof of OpenWork task competence. The converter and worker still
validate every artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import struct
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import psutil

from ._build_metadata import _signature, open_regular
from .model_inventory import MAX_HEADER_BYTES, _weights
from .registry import get_model
from .streaming import ConversionError, UnsupportedConversion, authorize_source_path, read_safetensors_header

GIB = 1024**3
MIB = 1024**2
TASKS = frozenset({"everyday", "text", "files", "code", "unsure"})
MAX_EXCLUSIONS = 64
PREFERENCES = frozenset({"balanced", "speed", "answers", "task", "unsure"})
DENSE_ARCHITECTURES = frozenset(
    {"qwen3", "qwen2", "llama", "mistral", "phi3", "granite", "gemma2", "gemma3_text", "smollm3"}
)


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
    catalog_item: dict | None = None
    estimated_weight_bytes: int | None = None
    logical_parameters: int | None = None
    task_tags: tuple[str, ...] = ()
    query_heads: int | None = None

    @property
    def weight_bytes(self):
        if self.estimated_weight_bytes is not None:
            return self.estimated_weight_bytes
        return sum(size for name, size, _digest in self.files if name.endswith(".safetensors"))

    @property
    def download_bytes(self):
        if self.catalog_item is not None:
            return self.catalog_item["planning"]["download_bytes"]
        return sum(size for _name, size, _digest in self.files)

    @property
    def installed_bytes(self):
        # Repacking preserves the selected weight payload; allow an additional
        # 128 MiB for output headers, tokenizer, config and receipts.
        return self.weight_bytes + 128 * MIB

    @property
    def setup_bytes(self):
        # Keep source and converted copies, plus a 512 MiB setup reserve.
        return self.download_bytes + self.installed_bytes + 512 * MIB

    @property
    def runtime_memory(self):
        # Guided installs use the converter's BF16 default (cli.py), matching
        # the dense generation route's activation/KV state policy.
        # Budget both full-context KV buffers, the 1024-row attention suffix,
        # converted weights/header overhead and a 3 GiB workspace/tokenizer allowance.
        # Rounded-up planning values still do not grant native admission.
        cache = 2 * (self.context_tokens + 1024) * self.attention_layers * self.kv_heads * self.head_dim * 2
        workspace = 3 * GIB
        if self.query_heads is not None:
            # Include the fallback attention score tensors for one full prefill
            # chunk, plus a GiB for other activations. The score tensors are
            # described by backends/metal/kernel/attention/fallback.rs.
            heads = self.query_heads
            scores = 1024 * (self.context_tokens + 1024) * (heads + heads // self.kv_heads) * 2
            workspace = max(workspace, scores + GIB)
        return ((self.installed_bytes + cache + workspace + GIB - 1) // GIB) * GIB


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
        logical_parameters=4_000_000_000,
        query_heads=32,
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
        logical_parameters=600_000_000,
        query_heads=16,
    ),
)


def recipe_for(identifier):
    return next((recipe for recipe in RECIPES if recipe.id == identifier), None)


def recipe_from_catalog(item):
    """Snapshot a pinned dense candidate only when its full memory plan is known.

    A conditional metadata preflight is not native qualification. Hybrid routes
    need their own recurrent/expert workspace planner before entering this flow.
    """
    try:
        return _recipe_from_catalog(item)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        # One malformed remote metadata document must not discard the other
        # candidates or turn the entire catalogue search into a fallback.
        return None


def _recipe_from_catalog(item):
    if not isinstance(item, dict) or not isinstance(item.get("planning"), dict):
        return None
    plan = item["planning"]
    architecture = plan.get("architecture")
    modes = item.get("artifact_modes")
    if (
        plan.get("status") != "available"
        or not isinstance(architecture, str)
        or architecture not in DENSE_ARCHITECTURES
        or item.get("downloadable") is not True
        or not isinstance(modes, (list, tuple))
        or "text" not in modes
        or plan.get("chat_template_available") is not True
        or plan.get("supports_tools") is not True
        or plan.get("tool_parser") not in ("qwen3-hermes-v1", "qwen38-xml-v1", "qwen35-xml-v1")
    ):
        return None
    bounds = {
        "context_limit": (32768, 1_048_576),
        "attention_layers": (1, 512),
        "kv_heads": (1, 256),
        "query_heads": (1, 512),
        "head_dim": (1, 1024),
        "hidden_size": (1, 262144),
        "download_bytes": (1, 2**63 - 1),
    }
    if any(type(plan.get(key)) is not int or not low <= plan[key] <= high for key, (low, high) in bounds.items()):
        return None
    if plan["query_heads"] % plan["kv_heads"]:
        return None
    weights = item.get("weight_bytes")
    if type(weights) is not int or not 0 < weights <= plan["download_bytes"]:
        return None
    if (
        not isinstance(item.get("id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", item["id"]) is None
        or not isinstance(item.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", item["revision"]) is None
        or not isinstance(item.get("repo"), str)
        or len(item["repo"].encode()) > 256
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", item["repo"]) is None
        or any(
            not isinstance(item.get(key), str)
            or not 0 < len(item[key].encode()) <= limit
            or any(ord(char) < 32 or ord(char) == 127 for char in item[key])
            for key, limit in (("name", 200), ("quantization", 80))
        )
    ):
        return None
    parameters = item.get("parameter_count")
    if type(parameters) is not int or not 0 < parameters < 2**63:
        parameters = None
        label = item.get("size_label")
        match = (
            re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMBT])", label, re.IGNORECASE)
            if isinstance(label, str) and len(label) <= 64
            else None
        )
        if match:
            count = float(match[1]) * {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}[match[2].upper()]
            if 0 < count < 2**63:
                parameters = int(count)
    tags = plan.get("task_tags", ())
    if not isinstance(tags, (list, tuple)) or any(
        not isinstance(tag, str) or not 0 < len(tag) <= 80 or any(ord(char) < 32 or ord(char) == 127 for char in tag)
        for tag in tags
    ):
        return None
    code_tags = {"code", "coding", "programming", "coder", "code-generation", "code_generation", "code-completion"}
    tags = {"code" if tag.casefold() in code_tags else tag for tag in tags}
    tags.intersection_update(TASKS)
    entry = get_model(item["repo"])
    if entry is not None and any(rule.get("repo") != item["repo"] for rule in entry.get("files", {}).values()):
        # Every sidecar needs an exact revision binding before cross-repository
        # metadata can establish this candidate's installation identity.
        return None
    recipe = Recipe(
        item["id"],
        item["name"],
        item["repo"],
        item["revision"],
        architecture,
        "hf-generic" if entry is None else None,
        item["quantization"],
        0,
        32768,
        plan["attention_layers"],
        plan["kv_heads"],
        plan["head_dim"],
        "Catalogue metadata supports this conversion route; native loading, answer quality and "
        "OpenWork task competence remain to be checked for this exact artifact.",
        (),
        catalog_item=deepcopy(item),
        estimated_weight_bytes=weights,
        logical_parameters=parameters,
        task_tags=tuple(sorted(tags)),
        query_heads=plan["query_heads"],
    )
    return replace(recipe, minimum_memory=recipe.runtime_memory + 2 * GIB)


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


def _artifact_precision_matches(root, recipe, receipt):
    """Check native BF16 storage and its size allowance using only headers.

    Source identity alone also matches F16/F32 reference conversions. Those
    outputs cannot inherit the guide's BF16 load/memory policy. Normalization
    scales remain F32 in the supported dense route; quantization parameters
    and floating matrices must be BF16.
    """
    capability = receipt.get("capability")
    if (
        receipt.get("status") != "ok"
        or not isinstance(capability, dict)
        or capability.get("adapter") != recipe.architecture
        or capability.get("model_kind") != "language"
        or capability.get("conversion") != "implemented"
        or capability.get("source_validation") != "complete"
        or capability.get("native_loading")
        not in {"qualified_tuple", "supported_layout_unverified_tuple", "supported_family_unverified", "unverified"}
    ):
        return False
    path = authorize_source_path(root, root / "model.safetensors")
    with open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        signature = _signature(before)
        if not 10 <= before.st_size <= recipe.installed_bytes:
            return False
        prefix = stream.read(8)
        if len(prefix) != 8 or not 2 <= struct.unpack("<Q", prefix)[0] <= MAX_HEADER_BYTES:
            return False
    size, label = _weights(root)
    if size != before.st_size or label != recipe.quantization:
        return False
    if type(receipt.get("output_bytes")) is not int or receipt["output_bytes"] != size:
        return False
    metadata, tensors = read_safetensors_header(path)
    if not tensors or any(tensor.file_signature != signature for tensor in tensors.values()):
        return False
    quantized = set()
    for name, raw in metadata.items():
        if not name.endswith(".spec"):
            continue
        spec = json.loads(raw)
        if not isinstance(spec, dict):
            return False
        if spec.get("type") in {"MLXSpec", "IntSpec"}:
            quantized.add(name.removesuffix(".spec"))
    for name, tensor in tensors.items():
        base, _, plane = name.rpartition(".")
        normalization = plane == "scales" and base.rsplit(".", 1)[-1].endswith("_norm")
        if normalization:
            if tensor.dtype != "F32" or len(tensor.shape) != 1:
                return False
        elif tensor.dtype == "BF16":
            continue
        elif tensor.dtype in {"U8", "I8", "U32", "I32"} and base in quantized and plane in {"weights", "zero_points"}:
            continue
        else:
            return False
    return signature == _signature(path.stat())


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
            or not 0 < len(source_files) <= 16384
            or recipe.catalog_item is None
            and len(source_files) != len(expected)
            or any(
                not isinstance(item, dict) or set(item) != {"repo", "revision", "size", "sha256"}
                for item in source_files
            )
            or recipe.catalog_item is None
            and {(item["repo"], item["revision"], item["size"], item["sha256"]) for item in source_files} != expected
            or recipe.catalog_item is not None
            and any(
                item["repo"] != recipe.repo
                or item["revision"] != recipe.revision
                or type(item["size"]) is not int
                or not 0 < item["size"] < 2**63
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                for item in source_files
            )
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
            and _artifact_precision_matches(root, recipe, receipt)
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        ConversionError,
        UnsupportedConversion,
    ):
        return False


def installation_allowed(recipe, host, *, installed=False):
    if host.get("supported") is not True:
        return False
    physical = host.get("physical_memory_bytes")
    if type(physical) is not int or physical < recipe.minimum_memory:
        return False
    if "gpu_available" in host or "gpu_memory_budget_bytes" in host:
        budget = host.get("gpu_memory_budget_bytes")
        if (
            host.get("gpu_available") is not True
            or host.get("unified_memory") is not True
            or type(budget) is not int
            or budget <= 0
            or recipe.runtime_memory * 100 > budget * 85
            or recipe.runtime_memory + 2 * GIB > physical
        ):
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


def _fit_tier(recipe, host):
    """A comfortable fit leaves 10% beyond the admission planning reserves."""
    physical = host["physical_memory_bytes"] - 2 * GIB
    budget = host.get("gpu_memory_budget_bytes")
    planned = min(physical, budget * 85 // 100) if type(budget) is int and budget > 0 else physical
    return int(recipe.runtime_memory * 110 <= planned * 100)


def _ranking_key(recipe, host, installed, task, preference):
    common = (-_fit_tier(recipe, host), -int(task in recipe.task_tags))
    if preference == "speed":
        objective = (recipe.weight_bytes, recipe.runtime_memory)
    elif preference == "answers":
        objective = (-(recipe.logical_parameters or 0), recipe.runtime_memory)
    else:
        # A memory-cost knee keeps balanced from always choosing the largest
        # fitting checkpoint. This policy is a heuristic, not a quality score.
        capacity_per_cost = Fraction((recipe.logical_parameters or 0) * GIB**2, recipe.runtime_memory**2)
        objective = (-capacity_per_cost, recipe.runtime_memory)
    return (
        *common,
        *objective,
        not bool(installed.get(recipe.id)),
        recipe.repo.casefold(),
        recipe.revision,
        recipe.id,
    )


def _recommendation_reason(recipe, host, task, preference):
    reasons = [
        "The memory plan fits this Mac's RAM and GPU budget."
        if "gpu_available" in host
        else "The memory plan fits this Mac's physical memory."
    ]
    if task in recipe.task_tags:
        reasons.append(
            "The source explicitly identifies coding as an intended use."
            if task == "code"
            else "The source identifies your selected task as an intended use."
        )
    if preference == "speed":
        reasons.append("Favors a model with less weight data to read; response speed has not been measured.")
    elif recipe.logical_parameters:
        reasons.append(
            "Favors a larger model among options that leave memory headroom; answer quality has not been measured."
            if preference == "answers"
            else "Balances model size with estimated memory use; answer quality has not been measured."
        )
    else:
        reasons.append("Parameter capacity is unknown, so the choice uses resource fit and deterministic tie-breaks.")
    available = host.get("available_memory_bytes")
    if type(available) is not int or available < recipe.runtime_memory:
        reasons.append("More memory must be available when starting; switching away from a running model may free it.")
    return " ".join(reasons)


def _license_label(value):
    try:
        if (
            isinstance(value, str)
            and 0 < len(value.encode()) <= 160
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            return value
    except UnicodeError:
        pass
    return "Review the source license"


def recommendation_report(
    host, installed, excluded_recipe_ids=(), *, task="unsure", preference="balanced", candidates=None
):
    candidates = tuple(RECIPES if candidates is None else candidates)
    task = task if task in TASKS else "unsure"
    preference = preference if preference in PREFERENCES else "balanced"
    eligible = [
        recipe
        for recipe in candidates
        if recipe.id not in excluded_recipe_ids
        and installation_allowed(recipe, host, installed=bool(installed.get(recipe.id)))
    ]
    eligible.sort(key=lambda recipe: _ranking_key(recipe, host, installed, task, preference))
    items = []
    for recipe in eligible[:3]:
        ids = list(installed.get(recipe.id, ()))
        license_name = recipe.catalog_item.get("license") if recipe.catalog_item is not None else "Apache-2.0"
        license_name = _license_label(license_name)
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
                "license": license_name,
                "qualification": recipe.qualification,
                "recommendation_reason": _recommendation_reason(recipe, host, task, preference),
                "evidence": "catalogue_metadata" if recipe.catalog_item is not None else "recorded_native_checks",
                "caveats": [
                    "Memory and setup storage are conservative planning estimates, not measured fit guarantees.",
                    "Start the model to check it on this Mac. Speed, answer quality, and OpenWork task success vary.",
                ],
            }
        )
    physical = host.get("physical_memory_bytes")
    message = "Choose a model selected for this Mac's resources and your preferences."
    if not items:
        message = (
            "No model with a complete memory plan fits these resources. Check again or browse models in Advanced."
        )
        if (
            candidates == RECIPES
            and type(physical) is int
            and physical < min(recipe.minimum_memory for recipe in RECIPES)
        ):
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
