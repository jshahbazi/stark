"""Source-bound starting choices for the optional beginner guide.

Pinned starters have recorded native chat evidence. Additional candidates come
from metadata-preflighted catalogue revisions and pinned native source profiles
with complete route-specific memory plans.
Ranking combines resource/capacity heuristics with comparable, attributed
upstream benchmark references. These are not measurements of Stark artifacts
or proof of OpenWork task competence. The converter and worker still validate
every artifact.
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
from .api_tools import TOOL_PARSERS
from .guided_memory import config_identity, native_request_bytes
from .guided_memory_profiles import ResidentProfile, qualified_profile, source_profile
from .model_inventory import MAX_HEADER_BYTES, _weights
from .recommendation_benchmarks import load_registry
from .recommendation_leaderboard import load_snapshot, speed_policy
from .registry import get_model
from .streaming import ConversionError, UnsupportedConversion, authorize_source_path, read_safetensors_header

GIB = 1024**3
MIB = 1024**2
TASKS = frozenset({"general", "coding", "everyday", "text", "files", "code", "unsure"})
MAX_EXCLUSIONS = 64
PREFERENCES = frozenset({"quality", "balanced", "speed", "answers", "task", "unsure"})
DENSE_ARCHITECTURES = frozenset(
    {"qwen3", "qwen2", "llama", "mistral", "phi3", "granite", "gemma2", "gemma3_text", "smollm3"}
)
# FunctionGemma requires a task-specific workflow and native loading qualification.
GUIDED_TOOL_PARSERS = TOOL_PARSERS - {"functiongemma-v1"}
# These official checkpoints are pretrained completion models even though their
# tokenizers contain a chat/tool template. Qwen recommends post-training before
# conversational use: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B
# Keep them in Advanced; a template alone does not establish assistant suitability.
GUIDED_NON_CHAT_SOURCES = frozenset(f"Qwen/Qwen2.5-Coder-{size}B" for size in ("0.5", "1.5", "3", "7", "14", "32"))


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
    installed_only: bool = False
    installed_config_sha256: str | None = None
    estimated_runtime_bytes: int | None = None
    estimated_gpu_runtime_bytes: int | None = None
    estimated_startup_bytes: int | None = None
    capacity_is_lower_bound: bool = False
    display_name: str | None = None
    native_choice: str | None = None
    native_profile: dict | None = None
    recurrent_state_bytes: int = 0
    recurrent_workspace_bytes: int = 0
    resident_profile: ResidentProfile | None = None

    @property
    def weight_bytes(self):
        if self.estimated_weight_bytes is not None:
            return self.estimated_weight_bytes
        return sum(size for name, size, _digest in self.files if name.endswith(".safetensors"))

    @property
    def download_bytes(self):
        if self.installed_only:
            return 0
        if self.native_choice is not None:
            return self.native_profile["download_bytes"]
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
        if self.installed_only:
            return self.installed_bytes
        if self.native_choice is not None:
            # Native preparation publishes downloaded weights by rename.
            return self.installed_bytes + 4 * GIB
        # Keep source and converted copies, plus a 512 MiB setup reserve.
        return self.download_bytes + self.installed_bytes + 512 * MIB

    @property
    def startup_memory(self):
        return self.runtime_memory if self.estimated_startup_bytes is None else self.estimated_startup_bytes

    @property
    def runtime_memory(self):
        if self.estimated_runtime_bytes is not None:
            return self.estimated_runtime_bytes
        if self.native_profile is not None:
            state = native_request_bytes(self.native_choice, self.native_profile["geometry"], self.context_tokens)
            if self.resident_profile is not None:
                policy = self.resident_profile
                resident = self.installed_bytes - policy.paged_weight_bytes
                total = resident + state * policy.request_headroom_divisor
                total += policy.paged_working_set_bytes + policy.host_reserve_bytes
                return ((total + GIB - 1) // GIB) * GIB
            return ((self.installed_bytes + state + 3 * GIB + GIB - 1) // GIB) * GIB
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
        workspace += self.recurrent_workspace_bytes
        return ((self.installed_bytes + cache + workspace + self.recurrent_state_bytes + GIB - 1) // GIB) * GIB

    @property
    def gpu_runtime_memory(self):
        if self.estimated_gpu_runtime_bytes is not None:
            return self.estimated_gpu_runtime_bytes
        # GPU planning is independent of qualified resident-system accounting.
        # Only the admitted Lily/MLX-Serve paths stage selected rows on the GPU.
        paged = self.native_profile["paged_weight_bytes"] if self.native_profile is not None else 0
        if self.native_profile is not None:
            state = native_request_bytes(self.native_choice, self.native_profile["geometry"], self.context_tokens)
            return ((self.installed_bytes + state + 3 * GIB - paged + GIB - 1) // GIB) * GIB
        return self.runtime_memory


def with_resident_profile(recipe, model):
    # Always revalidate; an offer does not authorize a changed artifact at start.
    base = replace(recipe, resident_profile=None)
    profile = qualified_profile(model, recipe.native_choice) if recipe.native_choice is not None else None
    if (
        profile is None
        or profile != source_profile(recipe.native_choice, recipe.native_profile)
        or recipe.context_tokens > profile.maximum_context_tokens
        or profile.paged_weight_bytes != recipe.native_profile["paged_weight_bytes"]
        or (recipe.repo, recipe.revision, recipe.weight_bytes)
        != (profile.repo, profile.revision, profile.weight_bytes)
    ):
        return replace(base, minimum_memory=base.runtime_memory + 2 * GIB) if recipe.native_choice else base
    planned = replace(base, resident_profile=profile)
    return replace(planned, minimum_memory=planned.runtime_memory + 2 * GIB)


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


def native_recipes():
    """Fixed native routes with revision-bound geometry, template and storage plans."""
    from .hybrid_prepare import SOURCES
    from .native_catalog import allowed_choice, catalog_item

    profiles = json.loads(Path(__file__).with_name("guided_native_profiles.json").read_bytes())
    recipes = []
    for choice, profile in profiles["sources"].items():
        source = SOURCES.get(choice)
        if (
            not allowed_choice(choice)
            or source is None
            or profile["weights_source"] != {"repo": source["weights_repo"], "revision": source["weights_revision"]}
            or profile["metadata_source"] != {"repo": source["metadata_repo"], "revision": source["metadata_revision"]}
            or profile["weight_bytes"] != source["weight_bytes"]
            or profile["tool_parser"] not in GUIDED_TOOL_PARSERS
        ):
            continue
        item = catalog_item(choice)
        recipe = Recipe(
            id=item["id"],
            name=item["name"],
            repo=item["repo"],
            revision=item["revision"],
            architecture=source["architecture"],
            metadata_profile=None,
            quantization=item["quantization"],
            minimum_memory=0,
            context_tokens=profile["context_tokens"],
            attention_layers=0,
            kv_heads=0,
            head_dim=0,
            qualification=(
                "This exact native artifact has recorded bounded chat/tool workflow checks; "
                "coding benchmark performance on Stark has not been measured."
                if profile["native_workflow_checked"]
                else "This pinned source uses an implemented native preparation/load route. "
                "Its task quality and tool workflow still need exact-artifact evaluation."
            ),
            files=(),
            estimated_weight_bytes=item["weight_bytes"],
            logical_parameters=profile["parameter_count"],
            task_tags=tuple(profile["task_tags"]),
            native_choice=choice,
            native_profile=deepcopy(profile),
        )
        planned = replace(recipe, resident_profile=source_profile(choice, profile))
        recipes.append(replace(planned, minimum_memory=planned.runtime_memory + 2 * GIB))
    return tuple(recipes)


def recipe_from_catalog(item):
    """Snapshot a catalogue candidate only when its complete memory plan is known.

    A conditional metadata preflight is not native qualification. Dense and
    Qwen3.5 DeltaNet layouts have planners here; fixed native routes use their
    separately source-bound profiles in native_recipes().
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
        or architecture not in DENSE_ARCHITECTURES | {"qwen3_5"}
        or item.get("downloadable") is not True
        or not isinstance(modes, (list, tuple))
        or "text" not in modes
        or plan.get("chat_template_available") is not True
        or plan.get("supports_tools") is not True
        or plan.get("tool_parser") not in GUIDED_TOOL_PARSERS
        or item.get("repo") in GUIDED_NON_CHAT_SOURCES
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
    recurrent = plan.get("recurrent_memory") if architecture == "qwen3_5" else None
    if architecture == "qwen3_5" and (
        not isinstance(recurrent, dict)
        or set(recurrent) != {"attention_layers", "recurrent_state_bytes", "recurrent_workspace_bytes"}
        or any(type(value) is not int or not 0 < value < 2**63 for value in recurrent.values())
        or recurrent["attention_layers"] != plan["attention_layers"]
    ):
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
        recurrent_state_bytes=recurrent["recurrent_state_bytes"] if recurrent else 0,
        recurrent_workspace_bytes=recurrent["recurrent_workspace_bytes"] if recurrent else 0,
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
        elif spec.get("type") == "MLXRowPartitionedSpec":
            prefix = name.removesuffix(".spec")
            segments = spec.get("segments")
            if not isinstance(segments, list) or not 2 <= len(segments) <= 128:
                return False
            quantized.update(f"{prefix}.segment_{index}" for index in range(len(segments)))
    for name, tensor in tensors.items():
        base, _, plane = name.rpartition(".")
        delta_control = recipe.architecture == "qwen3_5" and re.fullmatch(
            r"decoder\.transformer\.layers\.[0-9]+\.mixer\.(?:a_log|dt_bias|conv\.weights)", name
        )
        if delta_control:
            expected_dimensions = 2 if name.endswith(".conv.weights") else 1
            if tensor.dtype != "F32" or len(tensor.shape) != expected_dimensions:
                return False
            continue
        normalization = plane == "scales" and (
            base.rsplit(".", 1)[-1].endswith("_norm")
            or recipe.architecture == "qwen3_5"
            and base.endswith(".mixer.norm")
        )
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
        if recipe.installed_only:
            from .guided_installed import inspect_installed

            metadata = inspect_installed(root)
            return (
                metadata is not None
                and (metadata.get("gpu_runtime_memory", metadata["runtime_memory"]) == recipe.gpu_runtime_memory)
                and (metadata.get("startup_memory", metadata["runtime_memory"]) == recipe.startup_memory)
                and all(
                    metadata[key] == expected
                    for key, expected in (
                        ("repo", recipe.repo),
                        ("revision", recipe.revision),
                        ("weight_bytes", recipe.weight_bytes),
                        ("quantization", recipe.quantization),
                        ("config_sha256", recipe.installed_config_sha256),
                        ("runtime_memory", recipe.runtime_memory),
                    )
                )
            )
        config = _small_file(root, "config.json")
        if recipe.native_choice is not None:
            from .api_tools import tool_parser_for
            from .hybrid_prepare import prepared_identity
            from .vision_runtime import model_image_family

            profile = recipe.native_profile
            receipt = json.loads(_small_file(root, "native-source.json"))
            parsed = json.loads(config)
            template = parsed.get("token_codec_config", {}).get("prompt_template")
            return (
                model_image_family(parsed, root) is None
                and prepared_identity(root)
                == {
                    "repo": recipe.repo,
                    "revision": recipe.revision,
                    "architecture": recipe.architecture,
                    "weight_bytes": recipe.weight_bytes,
                    "quantization": recipe.quantization,
                }
                and receipt.get("metadata_source") == profile["metadata_source"]
                and receipt.get("quantization") == recipe.quantization
                and config_identity(parsed) == profile["config_identity_sha256"]
                and hashlib.sha256(json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                == profile["prepared_config_sha256"]
                and isinstance(template, str)
                and hashlib.sha256(template.encode()).hexdigest() == profile["template_sha256"]
                and tool_parser_for({"prompt_template": template}) == profile["tool_parser"]
            )
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


def setup_credit(recipe, host):
    values = host.get("native_setup_credit_bytes")
    value = values.get(recipe.native_choice, 0) if isinstance(values, dict) else 0
    return (
        min(value, recipe.weight_bytes) if recipe.native_choice is not None and type(value) is int and value > 0 else 0
    )


def installation_allowed(recipe, host, *, installed=False):
    if recipe.installed_only and not installed:
        return False
    if host.get("supported") is not True:
        return False
    physical = host.get("physical_memory_bytes")
    if type(physical) is not int or physical < recipe.minimum_memory:
        return False
    if recipe.native_choice is not None and "gpu_available" not in host:
        return False
    if "gpu_available" in host or "gpu_memory_budget_bytes" in host:
        budget = host.get("gpu_memory_budget_bytes")
        if (
            host.get("gpu_available") is not True
            or host.get("unified_memory") is not True
            or type(budget) is not int
            or budget <= 0
            or recipe.gpu_runtime_memory * 100 > budget * 85
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
        and space + setup_credit(recipe, host) >= recipe.setup_bytes
        and (
            recipe.native_choice is not None or type(download_space) is int and download_space >= recipe.download_bytes
        )
    )


def _planning_budget(host):
    physical = host["physical_memory_bytes"] - 2 * GIB
    budget = host.get("gpu_memory_budget_bytes")
    return min(physical, budget * 85 // 100) if type(budget) is int and budget > 0 else physical


def execution_allowed(recipe, host):
    """Shared current-memory gate for recommendations and fresh worker starts."""
    available = host.get("available_memory_bytes")
    return (
        installation_allowed(recipe, host, installed=True)
        and type(available) is int
        and available >= recipe.startup_memory
    )


def _fit_tier(recipe, host):
    """A comfortable fit leaves 10% beyond the admission planning reserves."""
    physical = host["physical_memory_bytes"] - 2 * GIB
    budget = host.get("gpu_memory_budget_bytes")
    physical_fit = recipe.runtime_memory * 110 <= physical * 100
    gpu_fit = recipe.gpu_runtime_memory * 110 <= budget * 85 if type(budget) is int and budget > 0 else True
    return int(physical_fit and gpu_fit)


def _ranking_key(recipe, host, installed, task, preference):
    common = (-_fit_tier(recipe, host),)
    task_match = -int(task in recipe.task_tags)
    if preference == "speed":
        objective = (recipe.weight_bytes, task_match, recipe.runtime_memory)
    elif preference == "answers":
        objective = (-(recipe.logical_parameters or 0), task_match, recipe.runtime_memory)
    else:
        # Below half the host planning budget, prioritize capacity with a
        # modest memory penalty; above it the penalty grows quadratically.
        # Published comparable evidence is applied after this fallback order.
        target = _planning_budget(host) // 2
        capacity_per_cost = Fraction(
            (recipe.logical_parameters or 0) * target**2, target**2 + recipe.runtime_memory**2
        )
        objective = (task_match, -capacity_per_cost, recipe.runtime_memory)
    return (
        *common,
        *objective,
        not bool(installed.get(recipe.id)),
        recipe.repo.casefold(),
        recipe.revision,
        recipe.id,
    )


def _recommendation_reason(recipe, host, task, preference, *, benchmark_used=False, coding_evidence_available=False):
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
    if task == "code" and preference not in {"speed", "answers"}:
        if benchmark_used:
            reasons.append(
                "Prioritizes published coding benchmark evidence among models that fit; "
                "this Stark artifact has not been benchmarked."
            )
        elif coding_evidence_available:
            reasons.append(
                "No result in the selected coding benchmark cohort is recorded for this model. "
                "It follows models with that evidence; its coding quality is unknown."
            )
        else:
            reasons.append(
                "No eligible coding benchmark references are available. "
                "This is a resource-based fallback; coding quality is unknown."
            )
    elif benchmark_used:
        evaluation = {
            "code": "coding benchmark",
            "files": "tool-use benchmark",
        }.get(task, "instruction-following benchmark")
        reasons.append(
            f"Published upstream {evaluation} results informed comparison with other scored models; "
            "this Stark artifact has not been benchmarked."
        )
    elif preference == "speed":
        reasons.append("Favors a model with less weight data to read; response speed has not been measured.")
    elif recipe.logical_parameters:
        reasons.append(
            "Favors a larger model among options that leave memory headroom; answer quality has not been measured."
            if preference == "answers"
            else "Balances model size with this Mac's memory budget; answer quality has not been measured."
        )
    else:
        reasons.append("Parameter capacity is unknown, so the choice uses resource fit and deterministic tie-breaks.")
    available = host.get("available_memory_bytes")
    if type(available) is not int or available < recipe.startup_memory:
        reasons.append("More memory must be available when starting; switching away from a running model may free it.")
    return " ".join(reasons)


def recipe_from_installed(model):
    """Offer a checked local artifact independently of bounded online discovery."""
    from .guided_installed import inspect_installed

    metadata = inspect_installed(Path(model["path"]))
    if metadata is None:
        return None
    name = metadata.get("name", model["name"])
    if (
        not isinstance(name, str)
        or not 0 < len(name.encode()) <= 200
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        name = metadata["repo"].split("/")[-1]
    identifier = (
        "installed-"
        + hashlib.sha256(
            json.dumps(
                {key: metadata[key] for key in ("repo", "revision", "config_sha256", "quantization", "weight_bytes")},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:32]
    )
    return Recipe(
        id=identifier,
        name=name,
        repo=metadata["repo"],
        revision=metadata["revision"],
        architecture=metadata["architecture"],
        metadata_profile=None,
        quantization=metadata["quantization"],
        minimum_memory=metadata["runtime_memory"] + 2 * GIB,
        context_tokens=metadata["context_tokens"],
        attention_layers=0,
        kv_heads=0,
        head_dim=0,
        qualification="Installed artifact with a local metadata memory plan; native loading and task success "
        "are checked separately. No download is needed.",
        files=(),
        estimated_weight_bytes=metadata["weight_bytes"],
        logical_parameters=metadata["logical_parameters"],
        installed_only=True,
        installed_config_sha256=metadata["config_sha256"],
        estimated_runtime_bytes=metadata["runtime_memory"],
        estimated_gpu_runtime_bytes=metadata.get("gpu_runtime_memory"),
        estimated_startup_bytes=metadata.get("startup_memory"),
        capacity_is_lower_bound=metadata.get("capacity_is_lower_bound", False),
        display_name=metadata.get("display_name"),
    )


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
    host,
    installed,
    excluded_recipe_ids=(),
    *,
    task="general",
    preference="quality",
    candidates=None,
    benchmarks=None,
    leaderboard=None,
):
    candidates = tuple(RECIPES if candidates is None else candidates)
    task = task if task in TASKS else "general"
    preference = preference if preference in PREFERENCES else "quality"
    guide_task, guide_preference = task, preference
    # Older command clients retain their existing cohort semantics. The two
    # new guide paths use the cached leaderboard after the same hard admission.
    task = {"general": "everyday", "coding": "code"}.get(task, task)
    preference = "balanced" if preference == "quality" else preference
    eligible = [
        recipe
        for recipe in candidates
        if recipe.id not in excluded_recipe_ids
        and installation_allowed(recipe, host, installed=bool(installed.get(recipe.id)))
        and execution_allowed(recipe, host)
    ]
    eligible.sort(key=lambda recipe: _ranking_key(recipe, host, installed, task, preference))
    benchmarks = load_registry() if benchmarks is None else benchmarks
    eligible, cohort, benchmark_used = benchmarks.rank(
        eligible, task, preference, lambda recipe: _fit_tier(recipe, host)
    )
    leaderboard = load_snapshot() if leaderboard is None else leaderboard
    speed_limits = speed_policy(guide_task, guide_preference, host.get("physical_memory_bytes"))
    eligible, bucket, records, leaderboard_used = leaderboard.rank(
        eligible, guide_task, guide_preference, host.get("physical_memory_bytes"), benchmarks.model_id
    )
    if leaderboard_used:
        # Historical benchmark evidence stays visible as reference information;
        # it cannot claim to have determined a different final ordering.
        benchmark_used = set()
    items = []
    for recipe in eligible[:3]:
        evidence, benchmark_details = benchmarks.evidence(recipe, task, cohort, benchmark_used)
        cached_evidence, cached_details = leaderboard.evidence(
            recipe, benchmarks.model_id(recipe), guide_task, bucket, records, leaderboard_used
        )
        evidence = cached_evidence + evidence
        benchmark_details = (cached_details + benchmark_details)[:16]
        reason = _recommendation_reason(
            recipe,
            host,
            task,
            preference,
            benchmark_used=recipe.id in benchmark_used,
            coding_evidence_available=bool(benchmark_used),
        )
        if leaderboard_used:
            reason = (
                "Selected using the cached whatmodel.sh "
                + ("Coding" if guide_task == "coding" else "General Work")
                + " leaderboard among models admitted by Stark's RAM, GPU, current memory and disk checks. "
                + (
                    "This model has a published parent-model score; this Stark artifact has not been benchmarked."
                    if recipe.id in leaderboard_used
                    else "No matching cached score is recorded for this model; it follows scored options. "
                    "Its quality is unknown."
                )
            )
        if speed_limits["applied"]:
            reason = (
                "Selected by cached task score within a weight budget of 25% of this Mac's RAM, "
                f"with a minimum parent-model reference score of {speed_limits['minimum_score']}. "
                "Exact export weights determine the budget; quantized answer quality and response speed "
                "have not been measured by this ranking. Stark's RAM, GPU, current memory and disk checks also pass."
            )
        ids = list(installed.get(recipe.id, ()))
        license_name = (
            recipe.native_profile.get("license")
            if recipe.native_profile is not None
            else recipe.catalog_item.get("license")
            if recipe.catalog_item is not None
            else ("Review the installed source license" if recipe.installed_only else "Apache-2.0")
        )
        license_name = _license_label(license_name)
        items.append(
            {
                "id": recipe.id,
                "name": recipe.name,
                "capacity_is_lower_bound": recipe.capacity_is_lower_bound,
                "display_name": recipe.display_name
                or (
                    recipe.catalog_item.get("model_group", {}).get("name", recipe.name)
                    if isinstance(recipe.catalog_item, dict)
                    and isinstance(recipe.catalog_item.get("model_group"), dict)
                    else recipe.name
                ),
                "repo": recipe.repo,
                "revision": recipe.revision,
                "artifact_mode": "text",
                "quantization": recipe.quantization,
                "download_bytes": recipe.download_bytes,
                "installed_bytes_estimate": recipe.installed_bytes,
                "setup_disk_bytes_estimate": recipe.setup_bytes,
                "setup_disk_bytes_remaining_estimate": recipe.setup_bytes - setup_credit(recipe, host),
                "runtime_memory_bytes_estimate": recipe.runtime_memory,
                "startup_memory_bytes_estimate": recipe.startup_memory,
                "gpu_memory_bytes_estimate": recipe.gpu_runtime_memory,
                "context_tokens": recipe.context_tokens,
                "installed_ids": ids,
                "license": license_name,
                "qualification": recipe.qualification,
                "recommendation_reason": reason,
                "selection_preference": guide_preference,
                "speed_policy": speed_limits,
                "benchmarks": evidence,
                "benchmark_details": benchmark_details,
                "evidence": (
                    "installed_metadata"
                    if recipe.installed_only
                    else "recorded_native_checks"
                    if recipe.native_profile is not None and recipe.native_profile["native_workflow_checked"]
                    else "native_source_metadata"
                    if recipe.native_choice is not None
                    else "catalogue_metadata"
                    if recipe.catalog_item is not None
                    else "recorded_native_checks"
                ),
                "caveats": [
                    "Memory and setup storage are conservative planning estimates, not measured fit guarantees.",
                    "Start the model to check it on this Mac. Speed, answer quality, and OpenWork task success vary.",
                ],
            }
        )
    physical = host.get("physical_memory_bytes")
    message = "Choose a model selected for this Mac's resources and your preferences."
    native = [recipe for recipe in candidates if recipe.native_choice is not None]
    if native:
        message += " Native hybrid models were also checked for memory and setup storage."
    if items and speed_limits["applied"]:
        message += (
            " Speed selects the highest cached task score within a RAM-scaled weight budget and "
            "parent-model score floor. Quantized quality and Stark speed remain unmeasured by this ranking."
        )
    elif items and leaderboard_used:
        message += (
            f" Quality uses the offline whatmodel.sh {guide_task} snapshot captured {leaderboard.captured}; "
            "missing scores remain unknown."
        )
    elif items and task == "code" and preference not in {"speed", "answers"}:
        message += (
            " Coding choices prioritize published benchmark references; missing scores remain unknown."
            if benchmark_used
            else " No eligible coding benchmark references are available; these choices use resource fit."
        )
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
                "These guided starting models currently need at least 16 GB of memory. "
                "You can browse smaller chat models in Advanced."
            )
        if speed_limits["applied"]:
            message = (
                "No supported model meets Speed's cached task-score floor and 25%-of-RAM weight budget "
                "within this Mac's current resources. Choose Quality or browse models in Advanced."
            )
    return {
        "schema_version": 1,
        "status": "ready" if items else "unavailable",
        "message": message,
        "host": dict(host),
        "items": items,
        "speed_policy": speed_limits,
        "leaderboard_ranking": {
            "captured": leaderboard.captured,
            "task": guide_task,
            "ram_gb": bucket["ram_gb"] if bucket else None,
            "applied": bool(leaderboard_used),
            "source": bucket["url"] if bucket else None,
            "evidence_kind": "upstream_reference",
        },
        "benchmark_ranking": {
            "cohort": cohort.id if cohort is not None else None,
            "applied": bool(benchmark_used),
            "evidence_kind": "upstream_reference",
        },
    }
