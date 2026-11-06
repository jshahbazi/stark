"""Bounded, metadata-only discovery of downloadable model-family candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
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
MAX_CANDIDATES = MAX_ITEMS
MAX_INVENTORY = 16384
MAX_CONFIG_BYTES = 1024 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SIZE = re.compile(r"[0-9]+(?:\.[0-9]+)?[KMBT]\Z", re.IGNORECASE)
_LINE_SIZE = re.compile(r"(?i)(?<![a-z0-9])[0-9]+(?:\.[0-9]+)?[mbt](?![a-z0-9])")
_SEPARATOR = re.compile(r"[-_. ]{2,}")
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
_PLANNING_DENSE_FAMILIES = {
    "qwen3",
    "qwen2",
    "llama",
    "mistral",
    "phi3",
    "granite",
    "gemma2",
    "gemma3_text",
    "smollm3",
}
_CONTEXT_FIELDS = ("max_position_embeddings", "n_positions", "max_seq_len")
_CODE_TASK_TAGS = {"code", "coding", "programming", "coder"}
_GENERATION_BLOCK_TAG = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
_MAX_CHAT_TEMPLATE_BYTES = 256 * 1024


class _Unavailable(Exception):
    pass


class _Inaccessible(Exception):
    """A source repo or metadata payload cannot be admitted to the catalogue."""


class _MalformedMetadata(Exception):
    """A bounded metadata document is invalid and cannot be retried usefully."""


def _hub_source_unavailable(error):
    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:
        return False
    return isinstance(error, (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError))


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


def _report(family, status, message, items=(), *, offset=0, next_offset=None):
    return {
        "schema_version": 1,
        "family_id": family,
        "status": status,
        "message": message,
        "items": list(items),
        "offset": offset,
        "has_more": next_offset is not None,
        "next_offset": next_offset,
    }


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError()
        value[key] = item
    return value


def _json(payload):
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, UnicodeError, RecursionError):
        raise _MalformedMetadata() from None


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


def _variant_stem(name, quantization):
    """Remove only a terminal export suffix that agrees with validated format."""
    name = _without_quantization_suffix(name, quantization)
    stem = _SEPARATOR.sub("-", _LINE_SIZE.sub("", name, count=1)).strip("-_. ")
    return (stem or name).casefold()


def _without_quantization_suffix(name, quantization):
    suffixes = {
        "4-bit": ("-mlx-4bit", "-4bit", "-4-bit"),
        "8-bit": ("-mlx-8bit", "-8bit", "-8-bit"),
        "BF16": ("-bf16", "-bfloat16"),
        "F16": ("-f16", "-fp16", "-float16"),
    }.get(quantization, ())
    for suffix in suffixes:
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _model_group(family, repo, name, quantization, entry):
    owner, _basename = repo.split("/", 1)
    group_owner = owner
    stem = _variant_stem(name, quantization)
    base_name = _without_quantization_suffix(name, quantization)
    label = (
        _SEPARATOR.sub("-", _LINE_SIZE.sub("", base_name, count=1)).strip("-_. ") or base_name.strip("-_. ") or name
    )
    if entry is not None:
        tokenizer = entry.get("files", {}).get("tokenizer_config", {})
        base_repo = tokenizer.get("repo") if isinstance(tokenizer, dict) else None
        if isinstance(base_repo, str) and base_repo != repo:
            base = next(
                (
                    candidate
                    for candidate in models()
                    if candidate.get("adapter") == family and candidate.get("repo") == base_repo
                ),
                None,
            )
            compressed = entry.get("compressed_format") if isinstance(entry, dict) else None
            expected_bits = {"4-bit": 4, "8-bit": 8}.get(quantization)
            if (
                base is not None
                and expected_bits is not None
                and isinstance(compressed, dict)
                and compressed.get("scheme") == "mlx_affine"
                and compressed.get("bits") == expected_bits
            ):
                base_owner, base_name = base_repo.split("/", 1)
                if (
                    _without_quantization_suffix(name, quantization).casefold()
                    == _without_quantization_suffix(base_name, "BF16").casefold()
                ):
                    group_owner = base_owner
                    stem = _variant_stem(base_name, "BF16")
                    base_label = _without_quantization_suffix(base_name, "BF16")
                    label = (
                        _SEPARATOR.sub("-", _LINE_SIZE.sub("", base_label, count=1)).strip("-_. ")
                        or base_label.strip("-_. ")
                        or base_name
                    )
    # The family and owner are part of the group identity; source revision remains
    # on the downloadable row, where it pins the actual artifact.
    identity = f"{family}/{group_owner}/{stem}"
    group_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return {"id": group_id, "name": label[:256], "publisher": group_owner}


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


def _positive_int(value):
    return value if type(value) is int and 0 < value <= 2**31 - 1 else None


def _source_chat_template(entry, repo, names, read):
    """Read only same-revision metadata used by Stark's chat conversion path."""
    files = entry.get("files", {}) if isinstance(entry, dict) else {}
    tokenizer_spec = files.get("tokenizer_config") or {"repo": repo, "filename": "tokenizer_config.json"}
    template_spec = files.get("chat_template")
    if template_spec is None and entry is None and "chat_template.jinja" in names:
        template_spec = {"repo": repo, "filename": "chat_template.jinja"}
    tokenizer_repo = tokenizer_spec.get("repo") if isinstance(tokenizer_spec, dict) else None
    tokenizer_file = tokenizer_spec.get("filename") if isinstance(tokenizer_spec, dict) else None
    if tokenizer_repo != repo:
        return None, "tokenizer_metadata_source_not_pinned"
    if not isinstance(tokenizer_file, str) or tokenizer_file not in names:
        return None, "tokenizer_config_missing"
    try:
        tokenizer_config = _json(read(tokenizer_file, MAX_CONFIG_BYTES))
    except (_Unavailable, _MalformedMetadata, _Inaccessible):
        return None, "tokenizer_config_unavailable"
    if not isinstance(tokenizer_config, dict):
        return None, "tokenizer_config_invalid"

    template = tokenizer_config.get("chat_template")
    inline_id = entry.get("metadata", {}).get("inline_sources", {}).get("chat_template") if entry else None
    if template is not None and (template_spec is not None or inline_id is not None):
        return None, "chat_template_metadata_conflict"
    if template_spec is not None:
        template_repo = template_spec.get("repo") if isinstance(template_spec, dict) else None
        template_file = template_spec.get("filename") if isinstance(template_spec, dict) else None
        if template_repo != repo:
            return None, "chat_template_source_not_pinned"
        if not isinstance(template_file, str) or template_file not in names:
            return None, "chat_template_file_missing"
        try:
            payload = read(template_file, MAX_CONFIG_BYTES)
            if "field" in template_spec:
                document = _json(payload)
                template = document.get(template_spec["field"]) if isinstance(document, dict) else None
            else:
                template = payload.decode("utf-8")
        except (UnicodeError, _Unavailable, _MalformedMetadata, _Inaccessible):
            return None, "chat_template_file_invalid"
    elif inline_id is not None:
        try:
            from .metadata import _inline_text

            template = _inline_text("chat_template", inline_id)
        except Exception:
            return None, "chat_template_inline_source_unavailable"
    if not isinstance(template, str):
        return None, "chat_template_missing"
    template = _GENERATION_BLOCK_TAG.sub("", template)
    try:
        if not template or len(template.encode("utf-8")) > _MAX_CHAT_TEMPLATE_BYTES:
            return None, "chat_template_invalid"
    except UnicodeEncodeError:
        return None, "chat_template_invalid"
    return template, None


def _planning_metadata(family, config, info, names, sizes, *, chat_template=None, chat_template_blocker=None):
    """Return bounded runtime-planning inputs without implying measured fit."""
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        text = {}
    merged = {**text, **{key: value for key, value in config.items() if key != "text_config"}}
    context = next((_positive_int(merged.get(key)) for key in _CONTEXT_FIELDS if _positive_int(merged.get(key))), None)
    layers = _positive_int(merged.get("num_hidden_layers"))
    heads = _positive_int(merged.get("num_attention_heads"))
    kv_heads = _positive_int(merged.get("num_key_value_heads")) if "num_key_value_heads" in merged else heads
    hidden = _positive_int(merged.get("hidden_size"))
    head_dim = _positive_int(merged.get("head_dim"))
    if head_dim is None and hidden is not None and heads is not None and hidden % heads == 0:
        head_dim = _positive_int(hidden // heads)

    blockers = []
    if family not in _PLANNING_DENSE_FAMILIES:
        blockers.append("runtime_layout_not_covered")
    if context is None:
        blockers.append("context_limit_unknown")
    if layers is None or heads is None or kv_heads is None or head_dim is None or kv_heads > heads:
        blockers.append("attention_geometry_incomplete")
    from .api_tools import tool_parser_for
    from .template_compat import template_program_sha256

    chat_template_available = isinstance(chat_template, str) and template_program_sha256(chat_template) is not None
    tool_parser = tool_parser_for({"prompt_template": chat_template}) if chat_template_available else None
    supports_tools = tool_parser is not None
    if not chat_template_available:
        blockers.append(chat_template_blocker or "chat_template_unavailable")
    if not supports_tools:
        blockers.append("tool_parser_unavailable")

    inventory_bytes = None
    byte_values = [sizes.get(name) for name in names]
    if byte_values and all(type(size) is int and 0 < size <= 2**63 - 1 for size in byte_values):
        total = sum(byte_values)
        if total <= 2**63 - 1:
            inventory_bytes = total

    raw_tags = _field(info, "tags", [])
    task_tags = (
        sorted(
            {
                tag.casefold()
                for tag in raw_tags
                if isinstance(tag, str) and len(tag) <= 64 and tag.casefold() in _CODE_TASK_TAGS
            }
        )
        if isinstance(raw_tags, list)
        else []
    )
    card = _field(info, "cardData", {})
    license_id = card.get("license") if isinstance(card, dict) else None
    if not isinstance(license_id, str) or not 0 < len(license_id) <= 80 or any(ord(char) < 32 for char in license_id):
        license_id = None
    return {
        "status": "available" if not blockers else "unavailable",
        "architecture": family,
        "context_limit": context,
        "attention_layers": layers,
        "query_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "download_bytes": inventory_bytes,
        "task_tags": task_tags,
        "license": license_id,
        "chat_template_available": chat_template_available,
        "supports_tools": supports_tools,
        "tool_parser": tool_parser,
        "blockers": blockers,
    }


def _quantization(config, info, entry, family=None):
    if (
        config.get("model_type") == "qwen4_exp"
        and isinstance(config.get("lily"), dict)
        and config["lily"].get("format") == "qwen4_exp-affine-v1"
    ):
        # Pinned preflight validates this format before catalog display. Its
        # routed experts are Q4 and its fixed small projections are Q8 even
        # when the optional Q8 group list is empty. The generic top-level
        # quantization block describes only the default width.
        return "4-bit / 8-bit"
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        return None
    merged = {**text, **{key: value for key, value in config.items() if key != "text_config"}}
    declared = any(merged.get(key) is not None for key in ("quantization", "quantization_config"))
    packing = entry.get("compressed_format") if entry else None
    if declared:
        from .compressed import parse_mlx_quantization

        try:
            quantization = parse_mlx_quantization(merged, allow_mixed=family == "qwen3_5")
        except Exception:
            return None
        if quantization is None:
            return None
        bits = quantization.bits
        if packing is not None and (
            not isinstance(packing, dict) or packing.get("scheme") != "mlx_affine" or packing.get("bits") != bits
        ):
            return None
        widths = {bits}
        if family == "qwen3_5":
            widths.update(
                override.bits for _, override in quantization.overrides if override is not None
            )
        if packing is not None and not widths <= set(packing.get("mixed_bits", [bits])):
            return None
        if len(widths) > 1:
            return f"{'/'.join(str(width) for width in sorted(widths))}-bit"
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


def _catalog_pins(family):
    if family == "qwen4_exp":
        from .hybrid_prepare import SOURCES

        # The supported Lily export can fall outside the bounded Hub search.
        # Reuse its reviewed source identity and keep generic conversion checks.
        source = SOURCES["qwen38-lily-q4-xl"]
        return {source["weights_repo"]: source["weights_revision"]}
    if family == "qwen3_5":
        return {EXACT_VISION_REPO: EXACT_VISION_REVISION}
    return {}


def _catalog_seeds(family):
    from .native_catalog import catalog_items

    registered = [
        entry["repo"]
        for entry in models()
        if entry.get("adapter") == family and entry.get("container") == ".safetensors"
    ]
    return list(
        dict.fromkeys([*registered, *_catalog_pins(family), *(item["repo"] for item in catalog_items(family))])
    )


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
        and (
            name.endswith("ForCausalLM")
            or (family == "qwen3_5" and name == "Qwen3_5ForConditionalGeneration")
            or (family == "qwen4_exp" and name == "Qwen4ExpForConditionalGeneration")
        )
        for name in declared
    )


def family_catalog(
    family_id, *, hub=None, limit=MAX_ITEMS, offline=False, offset=0, planning=False, deadline_seconds=80
):
    """Return candidate downloads; actual installation must repeat pinned preflight."""
    from .native_catalog import FAMILY_ID as NATIVE_FAMILY_ID
    from .native_catalog import catalog_item as native_catalog_item
    from .native_catalog import catalog_items as native_catalog_items

    valid_offset = type(offset) is int and 0 <= offset <= 10_000_000

    def report(*args, **kwargs):
        return _report(*args, offset=offset if valid_offset else 0, **kwargs)

    if not valid_offset:
        return report(family_id or "", "error", "The model catalogue request is invalid.")
    if type(planning) is not bool or type(deadline_seconds) not in (int, float) or not 0 < deadline_seconds <= 600:
        return report(family_id or "", "error", "The model catalogue request is invalid.")

    if family_id == NATIVE_FAMILY_ID:
        if type(limit) is not int or not 1 <= limit <= MAX_ITEMS or offset != 0:
            return report(NATIVE_FAMILY_ID, "error", "The model catalogue request is invalid.")
        if offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
            return report(NATIVE_FAMILY_ID, "offline", "Connect to the internet to browse models.")
        return report(
            NATIVE_FAMILY_ID,
            "ready",
            "Choose a model to download.",
            [
                {
                    **native_catalog_item(),
                    **(
                        {
                            "planning": {
                                "status": "unavailable",
                                "architecture": NATIVE_FAMILY_ID,
                                "context_limit": None,
                                "attention_layers": None,
                                "query_heads": None,
                                "kv_heads": None,
                                "head_dim": None,
                                "hidden_size": None,
                                "download_bytes": native_catalog_item()["weight_bytes"],
                                "task_tags": [],
                                "license": None,
                                "chat_template_available": False,
                                "supports_tools": False,
                                "tool_parser": None,
                                "blockers": ["runtime_layout_not_covered"],
                            }
                        }
                        if planning
                        else {}
                    ),
                }
            ],
        )

    family = normalize_architecture(family_id)
    definition = next((item for item in support_matrix()["families"] if item["family_id"] == family), None)
    if definition is None or definition["modes"]["text"]["status"] != "conditional":
        return report(family or "", "error", "This model family is not available in Stark.")
    if type(limit) is not int or not 1 <= limit <= MAX_ITEMS:
        return report(family, "error", "The model catalogue request is invalid.")
    if offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return report(family, "offline", "Connect to the internet to browse models.")
    if hub is None:
        import huggingface_hub

        hub = huggingface_hub
    deadline = time.monotonic() + deadline_seconds

    def remaining():
        if time.monotonic() >= deadline:
            raise _Unavailable()

    try:
        api = hub.HfApi()
        seeds = _catalog_seeds(family)
        pins = _catalog_pins(family)
        native_items = {item["repo"]: item for item in native_catalog_items(family)}
        discovery_failed = False

        def repositories():
            nonlocal discovery_failed
            yield from seeds
            for alias in definition["architecture_aliases"]:
                try:
                    # Let the Hub iterator follow every result page. The local
                    # window bounds metadata inspection, rather than hiding the
                    # tail of each family behind a fixed discovery limit.
                    for item in api.list_models(filter=alias, sort="downloads", expand=["sha"]):
                        remaining()
                        repo = _repo(_field(item, "id"))
                        if repo is not None:
                            yield repo
                except Exception:
                    discovery_failed = True
                    return  # Retry this frontier instead of skipping a failed alias.

        candidates, seen = [], set()
        for repo in repositories():
            remaining()
            if repo in seen:
                continue
            seen.add(repo)
            if len(seen) > offset:
                candidates.append(repo)
                if len(candidates) > limit:
                    break
        omitted = len(candidates) > limit or discovery_failed
        candidates = candidates[:limit]
        if not candidates:
            return report(
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
                if type(metadata.size) is int and metadata.size > maximum:
                    raise _Inaccessible()
                if type(metadata.size) is not int or metadata.size <= 0:
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
            except _Inaccessible:
                raise
            except Exception as error:
                if _hub_source_unavailable(error):
                    raise _Inaccessible() from None
                raise _Unavailable() from None

        def inspect(repo):
            remaining()
            try:
                if repo in native_items:
                    return dict(native_items[repo])
                requested_revision = pins.get(repo)
                try:
                    info = api.model_info(repo_id=repo, revision=requested_revision, files_metadata=True, timeout=5)
                except Exception as error:
                    if _hub_source_unavailable(error):
                        return None
                    raise
                if not _application_source(info, repo):
                    return None
                revision = _field(info, "sha")
                if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
                    raise _Unavailable()
                if requested_revision is not None and revision != requested_revision:
                    raise _Unavailable()
                names, sizes = _weight_inventory(info)
                entry = get_model(repo)
                if entry is not None and entry.get("adapter") != family:
                    return None
                rule = (
                    entry["files"]["model_config"] if entry is not None else {"repo": repo, "filename": "config.json"}
                )
                config_repo = rule["repo"]
                config_revision = revision
                if config_repo != repo:
                    try:
                        config_info = api.model_info(repo_id=config_repo, timeout=5)
                    except Exception as error:
                        if _hub_source_unavailable(error):
                            return None
                        raise
                    config_revision = _field(config_info, "sha")
                    if not isinstance(config_revision, str) or _REVISION.fullmatch(config_revision) is None:
                        raise _Unavailable()
                config = _json(read(config_repo, rule["filename"], config_revision, MAX_CONFIG_BYTES))
                if not isinstance(config, dict) or not _generation_architecture(config, family):
                    return None
                verdict = preflight_model(config, mode="text", expected_architecture=family)
                if verdict["status"] != "conditional":
                    return None
                quantization = _quantization(config, info, entry, family=family)
                if quantization is None:
                    return None
                selected = _selected_weights(names, lambda filename, maximum: read(repo, filename, revision, maximum))
                if selected is None:
                    return None
                weight_bytes = _weight_bytes(selected, sizes)
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
                chat_template, chat_template_blocker = (None, None)
                if planning:

                    def read_source(filename, maximum):
                        return read(repo, filename, revision, maximum)

                    chat_template, chat_template_blocker = _source_chat_template(entry, repo, names, read_source)
                item = {
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
                    "model_group": _model_group(family, repo, repo.split("/")[-1], quantization, entry),
                }
                if planning:
                    item["planning"] = _planning_metadata(
                        family,
                        config,
                        info,
                        names,
                        sizes,
                        chat_template=chat_template,
                        chat_template_blocker=chat_template_blocker,
                    )
                    item["task_tags"] = item["planning"]["task_tags"]
                    item["license"] = item["planning"]["license"]
                return item
            except _Inaccessible:
                return None
            except _MalformedMetadata:
                return None
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
        if failures:
            return report(family, "error", "Models could not be loaded. Check your connection and try again.")
        return report(
            family,
            "ready",
            "Choose a model to download." if accepted else "No compatible models were found in this family.",
            accepted,
            next_offset=offset + len(candidates) if omitted else None,
        )
    except Exception:
        return report(family, "error", "Models could not be loaded. Check your connection and try again.")
