"""Bounded, metadata-only discovery of downloadable model-family candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import islice
from pathlib import Path, PurePosixPath

from .model_support import (
    check_vision_processing_metadata,
    normalize_architecture,
    preflight_model,
    read_metadata_bytes,
    support_matrix,
)
from .registry import get_model, models
from .sources import huggingface_repository
from .vision_contract import REPO as EXACT_VISION_REPO
from .vision_contract import REVISION as EXACT_VISION_REVISION

MAX_ITEMS = 30
MAX_CANDIDATES = 45
MAX_INVENTORY = 16384
MAX_CONFIG_BYTES = 1024 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SIZE = re.compile(r"[0-9]+(?:\.[0-9]+)?[KMBT]\Z", re.IGNORECASE)
_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_GENERATION_TASKS = {"text-generation", "image-text-to-text", "image-to-text", "conversational", "any-to-any"}
_OTHER_TASK_TAGS = {
    "embedding",
    "embeddings",
    "feature-extraction",
    "reranker",
    "reranking",
    "sentence-similarity",
    "sentence-transformers",
    "text-embedding",
    "text-embeddings-inference",
    "text-ranking",
}


class _Unavailable(Exception):
    pass


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _string(value, maximum=256):
    try:
        return (
            isinstance(value, str)
            and 0 < len(value.encode()) <= maximum
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)
        )
    except UnicodeError:
        return False


def _repo(value):
    if not _string(value):
        return None
    try:
        return huggingface_repository(value)
    except ValueError:
        return None


def _report(family, status, message, items=(), *, has_more=False):
    return {
        "schema_version": 1,
        "family_id": family,
        "status": status,
        "message": message,
        "items": list(items),
        "has_more": has_more,
    }


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError()
        value[key] = item
    return value


def _json(payload):
    return json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
    )


def _parameters(info, quantized):
    # Hub totals for packed tensors count packed scalar slots, not logical parameters.
    if quantized:
        return None
    summary = _field(info, "safetensors")
    values, total = _field(summary, "parameters"), _field(summary, "total")
    if (
        not isinstance(values, dict)
        or not values
        or not set(values) <= _FLOAT_DTYPES
        or any(type(count) is not int or not 0 <= count <= 2**63 - 1 for count in values.values())
        or type(total) is not int
        or not 0 < total <= 2**63 - 1
        or sum(values.values()) != total
    ):
        return None
    return total


def _size_label(entry, parameters):
    declared = entry.get("size") if entry else None
    if isinstance(declared, str) and _SIZE.fullmatch(declared):
        return declared
    if parameters is None:
        return None
    for divisor, suffix in ((10**12, "T"), (10**9, "B"), (10**6, "M"), (10**3, "K")):
        if parameters >= divisor:
            return f"{parameters / divisor:.2f}".rstrip("0").rstrip(".") + suffix
    return str(parameters)


def _size_order(item):
    label = item["size_label"]
    if isinstance(label, str) and _SIZE.fullmatch(label):
        count = float(label[:-1]) * {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}[label[-1].upper()]
    else:
        count = item["parameter_count"] or float("inf")
    return count, item["name"].casefold(), item["repo"].casefold(), item["revision"]


def _select_items(items, limit):
    """Keep available sizes and formats represented before filling extra fine-tunes."""
    representatives, seen = [], set()
    for item in items:
        key = item["size_label"], item["quantization"]
        if key not in seen:
            representatives.append(item)
            seen.add(key)
    if len(representatives) > limit:
        chosen = (
            [representatives[0]]
            if limit == 1
            else [representatives[round(index * (len(representatives) - 1) / (limit - 1))] for index in range(limit)]
        )
    else:
        chosen = representatives
    ids = {item["id"] for item in chosen}
    for item in items:
        if len(chosen) == limit:
            break
        if item["id"] not in ids:
            chosen.append(item)
            ids.add(item["id"])
    return sorted(chosen, key=_size_order)


def _weight_inventory(info):
    siblings = _field(info, "siblings")
    if not isinstance(siblings, list) or not 1 <= len(siblings) <= MAX_INVENTORY:
        raise _Unavailable()
    names, sizes = [], {}
    for sibling in siblings:
        name = _field(sibling, "rfilename")
        if not _string(name, 1024):
            raise _Unavailable()
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or "\\" in name:
            raise _Unavailable()
        if name in sizes:
            raise _Unavailable()
        names.append(name)
        sizes[name] = _field(sibling, "size")
    return names, sizes


def _selected_weights(names, read):
    if "model.safetensors.index.json" in names:
        index = _json(read("model.safetensors.index.json", MAX_CONFIG_BYTES))
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(key, str) for key in mapping):
            return None
        if any(not isinstance(name, str) for name in mapping.values()):
            return None
        selected = sorted(set(mapping.values()))
        if any(
            name not in names
            or not name.endswith(".safetensors")
            or PurePosixPath(name).name.startswith("adapter_model")
            for name in selected
        ):
            return None
        return selected
    if "model.safetensors" in names:
        return ["model.safetensors"]
    selected = [
        name
        for name in names
        if name.endswith(".safetensors")
        and len(PurePosixPath(name).parts) == 1
        and not name.startswith("adapter_model")
    ]
    return selected if len(selected) == 1 else None


def _weight_bytes(selected, sizes):
    if all(type(sizes[name]) is int and 0 < sizes[name] <= 2**63 - 1 for name in selected):
        total = sum(sizes[name] for name in selected)
        if total <= 2**63 - 1:
            return total
    return None


def _quantization(config, info, entry):
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        return None
    merged = {**text, **{key: value for key, value in config.items() if key != "text_config"}}
    declared = any(merged.get(key) is not None for key in ("quantization", "quantization_config"))
    packing = entry.get("compressed_format") if entry else None
    if declared:
        from .compressed import parse_mlx_quantization

        try:
            quantization = parse_mlx_quantization(merged)
        except Exception:
            return None
        if quantization is None:
            return None
        bits = quantization.bits
        if packing is not None and (
            not isinstance(packing, dict) or packing.get("scheme") != "mlx_affine" or packing.get("bits") != bits
        ):
            return None
        return f"{bits}-bit"
    if packing is not None:
        if isinstance(packing, dict) and packing.get("scheme") == "mlx_affine" and packing.get("bits") in {4, 8}:
            return f"{packing['bits']}-bit"
        return None
    parameters = _field(_field(info, "safetensors"), "parameters")
    if isinstance(parameters, dict) and parameters and not set(parameters) <= _FLOAT_DTYPES:
        return None
    dtype = merged.get("torch_dtype", merged.get("dtype"))
    if dtype is not None and dtype not in {"bfloat16", "float16", "float32", "float64"}:
        return None
    # Native app floating-point conversion writes BF16, including FP16/F32 sources.
    return "BF16"


def _catalog_seeds(family):
    return [
        entry["repo"]
        for entry in models()
        if entry.get("adapter") == family and entry.get("container") == ".safetensors"
    ]


def _application_source(info, repo):
    task = _field(info, "pipeline_tag")
    if task is not None and task not in _GENERATION_TASKS:
        return False
    tags = _field(info, "tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        return False
    if _OTHER_TASK_TAGS.intersection(tag.lower() for tag in tags):
        return False
    owner, name = repo.lower().split("/", 1)
    if owner.endswith("-internal-testing") or re.search(r"(?:^|[_.-])tiny[_.-]random(?:$|[_.-])", name):
        return False
    return not any(re.fullmatch(r"tiny[-_]random(?:[-_]model)?", tag.lower()) for tag in tags)


def _generation_architecture(config, family):
    declared = config.get("architectures")
    if declared is None:
        return True  # A missing hint leaves full tensor validation to installation.
    if not isinstance(declared, list) or not 1 <= len(declared) <= 16:
        return False
    return all(
        isinstance(name, str)
        and (name.endswith("ForCausalLM") or (family == "qwen3_5" and name == "Qwen3_5ForConditionalGeneration"))
        for name in declared
    )


def family_catalog(family_id, *, hub=None, limit=MAX_ITEMS, offline=False):
    """Return candidate downloads; actual installation must repeat pinned preflight."""
    family = normalize_architecture(family_id)
    definition = next((item for item in support_matrix()["families"] if item["family_id"] == family), None)
    if definition is None or definition["modes"]["text"]["status"] != "conditional":
        return _report(family or "", "error", "This model family is not available in Stark.")
    if type(limit) is not int or not 1 <= limit <= MAX_ITEMS:
        return _report(family, "error", "The model catalogue request is invalid.")
    if offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return _report(family, "offline", "Connect to the internet to browse models.")
    if hub is None:
        import huggingface_hub

        hub = huggingface_hub
    deadline = time.monotonic() + 80

    def remaining():
        if time.monotonic() >= deadline:
            raise _Unavailable()

    try:
        api = hub.HfApi()
        seeds = _catalog_seeds(family)
        discovered = []
        discovery_failed = False
        for alias in definition["architecture_aliases"]:
            try:
                for item in islice(
                    api.list_models(filter=alias, sort="downloads", limit=MAX_CANDIDATES, expand=["sha"]),
                    MAX_CANDIDATES,
                ):
                    remaining()
                    repo = _repo(_field(item, "id"))
                    if repo is not None:
                        discovered.append(repo)
            except Exception:
                discovery_failed = True
        candidates = []
        # Keep known sizes/quantizations visible while also inspecting live fine-tunes.
        for index in range(max(len(seeds), len(discovered))):
            for collection in (seeds, discovered):
                if index < len(collection) and collection[index] not in candidates:
                    candidates.append(collection[index])
        omitted = len(candidates) > MAX_CANDIDATES
        candidates = candidates[:MAX_CANDIDATES]
        if not candidates:
            return _report(
                family,
                "error" if discovery_failed else "ready",
                "Models could not be loaded. Check your connection and try again."
                if discovery_failed
                else "No compatible models were found in this family.",
            )

        @lru_cache(maxsize=256)
        def read(repo, filename, revision, maximum):
            remaining()
            try:
                metadata = hub.get_hf_file_metadata(
                    hub.hf_hub_url(repo_id=repo, filename=filename, revision=revision), timeout=5
                )
                if type(metadata.size) is not int or not 0 < metadata.size <= maximum:
                    raise _Unavailable()
                if _field(metadata, "commit_hash") not in (None, revision):
                    raise _Unavailable()
                selected = Path(
                    hub.hf_hub_download(
                        repo_id=repo,
                        filename=filename,
                        revision=revision,
                        local_files_only=False,
                        etag_timeout=5,
                    )
                )
                authority = selected
                for _part in PurePosixPath(filename).parts:
                    authority = authority.parent
                return read_metadata_bytes(selected, root=authority, maximum=maximum)
            except Exception:
                raise _Unavailable() from None

        def inspect(repo):
            remaining()
            try:
                requested_revision = EXACT_VISION_REVISION if repo == EXACT_VISION_REPO else None
                info = api.model_info(repo_id=repo, revision=requested_revision, files_metadata=True, timeout=5)
                if not _application_source(info, repo):
                    return None
                revision = _field(info, "sha")
                if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
                    raise _Unavailable()
                if requested_revision is not None and revision != requested_revision:
                    raise _Unavailable()
                names, sizes = _weight_inventory(info)
                selected = _selected_weights(names, lambda filename, maximum: read(repo, filename, revision, maximum))
                if selected is None:
                    return None
                weight_bytes = _weight_bytes(selected, sizes)
                entry = get_model(repo)
                if entry is not None and entry.get("adapter") != family:
                    return None
                rule = (
                    entry["files"]["model_config"] if entry is not None else {"repo": repo, "filename": "config.json"}
                )
                config_repo = rule["repo"]
                config_revision = revision
                if config_repo != repo:
                    config_revision = _field(api.model_info(repo_id=config_repo, timeout=5), "sha")
                    if not isinstance(config_revision, str) or _REVISION.fullmatch(config_revision) is None:
                        raise _Unavailable()
                config = _json(read(config_repo, rule["filename"], config_revision, MAX_CONFIG_BYTES))
                if not isinstance(config, dict) or not _generation_architecture(config, family):
                    return None
                verdict = preflight_model(config, mode="text", expected_architecture=family)
                if verdict["status"] != "conditional":
                    return None
                quantization = _quantization(config, info, entry)
                if quantization is None:
                    return None
                modes = ["text"]
                if family == "qwen3_5" and quantization == "BF16":
                    vision = preflight_model(config, mode="vision-language", expected_architecture=family)
                    if vision["status"] == "conditional":
                        try:
                            check_vision_processing_metadata(
                                names, lambda filename, maximum: read(repo, filename, revision, maximum)
                            )
                            modes.append("vision-language")
                        except Exception:
                            pass
                extra_weights = any(name.endswith(".safetensors") and name not in selected for name in names)
                parameters = _parameters(info, quantization != "BF16" or extra_weights)
                return {
                    "id": hashlib.sha256((repo + "@" + revision).encode()).hexdigest()[:32],
                    "repo": repo,
                    "revision": revision,
                    "name": repo.split("/")[-1],
                    "size_label": _size_label(entry, parameters),
                    "parameter_count": parameters,
                    "weight_bytes": weight_bytes,
                    "quantization": quantization,
                    "artifact_modes": modes,
                    "default_artifact_mode": modes[-1],
                    "downloadable": True,
                }
            except _Unavailable:
                raise
            except Exception:
                raise _Unavailable() from None

        accepted, failures = [], 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(inspect, repo) for repo in candidates]
            for future in as_completed(futures):
                try:
                    item = future.result()
                    if item is not None:
                        accepted.append(item)
                except Exception:
                    failures += 1
        accepted.sort(key=_size_order)
        if not accepted and failures == len(candidates):
            return _report(family, "error", "Models could not be loaded. Check your connection and try again.")
        return _report(
            family,
            "ready",
            "Choose a model to download." if accepted else "No compatible models were found in this family.",
            _select_items(accepted, limit),
            has_more=omitted or len(accepted) > limit,
        )
    except Exception:
        return _report(family, "error", "Models could not be loaded. Check your connection and try again.")
