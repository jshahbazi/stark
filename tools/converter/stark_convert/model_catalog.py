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

from . import catalog_reasons, qwen38_external
from .model_support import (
    check_vision_processing_metadata,
    normalize_architecture,
    preflight_model,
    read_metadata_bytes,
    support_matrix,
)
from .registry import get_model, models
from .sources import huggingface_repository
from .streaming import ConversionError, UnsupportedConversion
from .vision_contract import REPO as EXACT_VISION_REPO
from .vision_contract import REVISION as EXACT_VISION_REVISION

MAX_ITEMS = 30
MAX_CANDIDATES = MAX_ITEMS
MAX_INVENTORY = 16384
# Listing positions one page may pass without a per-repository request.
MAX_SCREENED = 500
_LISTING_STEP = 100
# Listing positions searched for sources with a reference score, which are
# inspected before the rest of that window.
PRIORITY_WINDOW = 300
_WEIGHT_SUFFIXES = (".gguf", ".safetensors", ".bin", ".exl3")
MAX_CONFIG_BYTES = 1024 * 1024
MAX_TOKENIZER_CONFIG_BYTES = 2 * 1024 * 1024
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
_CODE_TASK_TAGS = {"code", "coding", "programming", "coder", "code-generation", "code_generation", "code-completion"}
_GENERATION_BLOCK_TAG = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
_MAX_CHAT_TEMPLATE_BYTES = 256 * 1024
_GEMMA3_IMAGE_TEMPLATE = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"
_SEARCH_TERMS = {
    "qwen3": ("Qwen3",),
    "qwen2": ("Qwen2",),
    "llama": ("Llama",),
    "mistral": ("Mistral",),
    "gemma2": ("gemma-2",),
    "gemma3_text": ("gemma-3",),
    "gemma3": ("gemma-3",),
    "gemma4_text": ("gemma-4",),
    "olmo3": ("OLMo-3",),
    "phi3": ("Phi-3",),
    "granite": ("Granite",),
    "smollm3": ("SmolLM3",),
    "qwen3_5": ("Qwen3.5", "Qwen3.6"),
    "qwen3_5_moe": ("Qwen3.5", "Qwen3.6"),
    "qwen4_exp": ("Qwen3.8-Flash-Next",),
    "lfm2": ("LFM2",),
    "llamba": ("LLaMBA",),
    "glm5_next": ("GLM-5.3-Flash",),
    "flint": ("flint",),
}


def _repository_search(api, definition, remaining, expected=0):
    """Interleave aliases and name searches so large tags cannot starve others.

    ``expected`` is how many listing positions the caller plans to walk. Each
    listing is requested at about that size and grows only if more are needed,
    because a listing reply carries every repository's file list.
    """
    queries = [
        *({"filter": alias} for alias in definition["architecture_aliases"]),
        *({"search": term} for term in _SEARCH_TERMS.get(definition["family_id"], ())),
    ]

    def results(query):
        size, sent = _LISTING_STEP * max(1, -(-expected // _LISTING_STEP)), 0
        while True:
            count = 0
            # Entries are passed on as they arrive, so a listing that fails
            # partway still contributes what it returned.
            for entry in api.list_models(
                **query, sort="downloads", expand=["sha", "pipeline_tag", "tags", "siblings"], limit=size
            ):
                count += 1
                if count > sent:
                    yield entry
            if count < size:
                return
            sent, size = count, size * 2

    streams = [iter(results(query)) for query in queries]
    while streams:
        active = []
        for stream in streams:
            remaining()
            try:
                item = next(stream)
            except StopIteration:
                continue
            active.append(stream)
            repo = _repo(_field(item, "id"))
            if repo is not None:
                # The listing's head revision lets a cached verdict answer
                # without any per-repository request.
                revision = _field(item, "sha")
                listed = isinstance(revision, str) and _REVISION.fullmatch(revision) is not None
                yield repo, revision if listed else None, item
        streams = active


def _listing_rejects(repo, listing):
    """Decide from the Hub listing alone that inspection would reject a source.

    Only outcomes that the per-repository checks reach from the same fields
    are settled here: another task, no model configuration, or no weights.
    A source whose only weights are GGUF files is also dropped. Inspection
    could only show it as unavailable for want of Safetensors weights, and
    some publishers give every GGUF repository the same placeholder
    configuration, which would list it under an unrelated family.
    """
    if isinstance(_field(listing, "tags"), list) and not _application_source(listing, repo):
        return True
    siblings = _field(listing, "siblings")
    if not isinstance(siblings, list) or not siblings:
        return False
    names = [_field(sibling, "rfilename") for sibling in siblings]
    if any(not isinstance(name, str) for name in names):
        return False
    if "config.json" not in names and get_model(repo) is None:
        return True
    weights = [
        name
        for name in names
        if name.endswith(_WEIGHT_SUFFIXES) and not PurePosixPath(name).name.startswith("adapter_model")
    ]
    return all(name.endswith(".gguf") for name in weights)


def _unsupported_quantization(config):
    """Describe declared packing for an unavailable row without granting support."""
    text = config.get("text_config", config)
    merged = {**text, **config} if isinstance(text, dict) else config
    packing = merged.get("quantization", merged.get("quantization_config"))
    if isinstance(packing, dict):
        bits = packing.get("bits")
        if type(bits) is int and 1 <= bits <= 16:
            return f"{bits}-bit"
    return "Unsupported format"


class _HeadersUnreadable(str):
    """The unverified-headers reason, marked as a header read failure.

    It equals ``qwen38_external.UNVERIFIED``; the caller replaces it with the
    access-approval reason for a gated source.
    """


def _verify_hub_layout(family, hub, repo, revision, config, shards, index, sizes, remaining, modes):
    from .checkpoint_metadata import CheckpointMetadata
    from .hub_headers import HeaderAccessError, HeaderBudgetError, checkpoint_headers
    from .planning import plan_from_headers

    unreadable = _HeadersUnreadable(qwen38_external.UNVERIFIED)
    try:
        tensors = checkpoint_headers(hub, repo, revision, shards, index, sizes, remaining)
    except (ConversionError, UnsupportedConversion) as error:
        return dict.fromkeys(modes, catalog_reasons.importer_reason(error))
    except HeaderBudgetError:
        return dict.fromkeys(modes, "A tensor header exceeds the catalog's 4 MiB verification limit.")
    except HeaderAccessError:
        return dict.fromkeys(modes, unreadable)
    except Exception:
        return dict.fromkeys(modes, unreadable)

    verdicts = {}
    for mode in modes:
        try:
            remaining()
            plan_from_headers(family, CheckpointMetadata(config, tensors), mode=mode)
            verdicts[mode] = None
        except (ConversionError, UnsupportedConversion) as error:
            verdicts[mode] = catalog_reasons.importer_reason(error)
        except (ValueError, TypeError, KeyError, OverflowError):
            verdicts[mode] = catalog_reasons.UNSUPPORTED_LAYOUT
        except Exception:
            verdicts[mode] = qwen38_external.UNVERIFIED
    return verdicts


class _Unavailable(Exception):
    pass


class _Inaccessible(Exception):
    """A source repo or metadata payload cannot be admitted to the catalogue."""


class _MalformedMetadata(Exception):
    """A bounded metadata document is invalid and cannot be retried usefully."""


def _read_revision_metadata(hub, repo, filename, revision, maximum, size=None):
    """Read pinned metadata, reusing a validated file in the HF disk cache when present.

    ``size`` is the file's size from the pinned repository inventory. When it is
    known, the bound is checked against it instead of a separate request.
    """
    try:
        # Only immutable full revisions can bypass the metadata HEAD request.
        # Hugging Face's cache lookup resolves precisely that snapshot; the
        # existing reader still verifies file type, size, stable identity and
        # that snapshot/blob symlinks remain within the authorized cache roots.
        cached_path = None
        if _REVISION.fullmatch(revision):
            lookup = getattr(hub, "try_to_load_from_cache", None)
            if lookup is None:
                try:
                    from huggingface_hub import try_to_load_from_cache as lookup
                except ImportError:
                    lookup = None
            if lookup is not None:
                try:
                    candidate = lookup(repo, filename, revision=revision)
                    if isinstance(candidate, (str, os.PathLike)):
                        cached_path = Path(candidate)
                except Exception:
                    # A broken cache index must not turn a cache miss into a
                    # permanent error; fall back to the ordinary pinned read.
                    cached_path = None
        if cached_path is not None:
            authority = cached_path
            for _part in PurePosixPath(filename).parts:
                authority = authority.parent
            if (
                authority.name != revision
                or authority.parent.name != "snapshots"
                or authority.parent.parent.name != "models--" + repo.replace("/", "--")
            ):
                cached_path = None
        if cached_path is not None:
            authority = cached_path
            for _part in PurePosixPath(filename).parts:
                authority = authority.parent
            return read_metadata_bytes(cached_path, root=authority, maximum=maximum)

        if type(size) is int:
            if size > maximum:
                raise _Inaccessible()
            if size <= 0:
                raise _Unavailable()
        else:
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
    except (_Inaccessible, _Unavailable):
        raise
    except Exception as error:
        if _hub_source_unavailable(error):
            raise _Inaccessible() from None
        raise _Unavailable() from None


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


def _dense_config_parameters(config, family):
    """Count logical parameters from complete supported decoder geometry.

    Quantized Hub tensor totals describe packed storage. These formulas instead
    follow the matrix, normalization and bias shapes checked by qwen3.py and
    adapters/dense.py. They describe the pinned config, not tensor admission;
    missing geometry and layouts outside these adapters remain unknown.
    """
    if family not in {"qwen3", "qwen2", "llama", "mistral"} or config.get("model_type") != family:
        return None
    bounds = {
        "hidden_size": 262144,
        "intermediate_size": 1048576,
        "num_hidden_layers": 512,
        "num_attention_heads": 512,
        "vocab_size": 1000000,
    }
    if any(type(config.get(key)) is not int or not 0 < config[key] <= high for key, high in bounds.items()):
        return None
    dim, intermediate, layers, heads, vocab = (config[key] for key in bounds)
    groups = config.get("num_key_value_heads", heads)
    if type(groups) is not int or not 0 < groups <= heads or heads % groups:
        return None
    head_dim = config.get("head_dim")
    if head_dim is None:
        if dim % heads:
            return None
        head_dim = dim // heads
    if type(head_dim) is not int or not 0 < head_dim <= 1024 or head_dim % 2:
        return None
    if family == "qwen2" and heads * head_dim != dim:
        return None
    tied = config.get("tie_word_embeddings", False)
    biased = config.get("attention_bias", family == "qwen2")
    if type(tied) is not bool or type(biased) is not bool:
        return None
    if family in {"llama", "mistral"} and biased or family == "qwen2" and not biased:
        return None
    if config.get("mlp_bias", False) is not False or config.get("use_qk_norm", True) is not True:
        return None
    if family == "qwen3" and config.get("qkv_bias", biased) is not biased:
        return None
    # Do not apply the dense formula to a custom expert, cross-attention or
    # pruned decoder even if its model_type reuses a dense family identifier.
    if any(
        config.get(key)
        for key in ("num_experts", "num_local_experts", "pruned_heads", "add_cross_attention", "is_encoder_decoder")
    ):
        return None
    query, kv = heads * head_dim, groups * head_dim
    attention = 2 * dim * (query + kv)
    mlp = 3 * dim * intermediate
    norms = 2 * dim + (2 * head_dim if family == "qwen3" else 0)
    biases = query + 2 * kv if biased else 0
    if family == "qwen3" and biased:
        biases += dim
    total = (1 if tied else 2) * vocab * dim + dim + layers * (attention + mlp + norms + biases)
    return total if 0 < total < 2**63 else None


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
        tokenizer_config = _json(read(tokenizer_file, MAX_TOKENIZER_CONFIG_BYTES))
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
    recurrent = None
    if family == "qwen3_5":
        from .guided_memory import qwen35_geometry

        recurrent = qwen35_geometry(text)
        if recurrent is not None:
            layers = recurrent["attention_layers"]
    if family not in _PLANNING_DENSE_FAMILIES and recurrent is None:
        blockers.append("runtime_layout_not_covered")
    if context is None:
        blockers.append("context_limit_unknown")
    if layers is None or heads is None or kv_heads is None or head_dim is None or kv_heads > heads:
        blockers.append("attention_geometry_incomplete")
    from .api_tools import tool_parser_for
    from .template_compat import normalize_chat_template, template_program_sha256

    native_template = normalize_chat_template(chat_template) if isinstance(chat_template, str) else None
    chat_template_available = template_program_sha256(native_template) is not None
    tool_parser = tool_parser_for({"prompt_template": native_template}) if chat_template_available else None
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
        **({"recurrent_memory": recurrent} if recurrent is not None else {}),
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
            quantization = parse_mlx_quantization(merged, allow_mixed=family in {"qwen3_5", "qwen3_5_moe"})
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
        if family in {"qwen3_5", "qwen3_5_moe"}:
            widths.update(override.bits for _, override in quantization.overrides if override is not None)
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
    if family == "gemma4_text":
        from .adapters.gemma4_vision import PROFILES

        return {repo: profile["revision"] for repo, profile in PROFILES.items()}
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
    if family == "gemma3":
        registered.extend(("google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"))
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
            or (family == "qwen3_5_moe" and name == "Qwen3_5MoeForConditionalGeneration")
            or (family == "qwen4_exp" and name == "Qwen4ExpForConditionalGeneration")
            or (family == "gemma3" and name == "Gemma3ForConditionalGeneration")
            or (
                family == "gemma4_text"
                and name
                in {
                    "Gemma4ForConditionalGeneration",
                    "Gemma4UnifiedForConditionalGeneration",
                    "Gemma4TextModel",
                    "Gemma4UnifiedTextModel",
                }
            )
        )
        for name in declared
    )


def _catalog_image_metadata(family, config, repo, revision, names, read):
    """Check reviewed image processing before advertising a download mode."""
    if preflight_model(config, mode="vision-language", expected_architecture=family)["status"] != "conditional":
        return False
    try:
        if family == "gemma3":
            from .gemma3_contract import normalize_processor

            normalize_processor(_json(read("preprocessor_config.json", 65536)))
            template, _reason = _source_chat_template(None, repo, names, read)
            return (
                isinstance(template, str) and hashlib.sha256(template.encode()).hexdigest() == _GEMMA3_IMAGE_TEMPLATE
            )
        if family == "gemma4_text":
            from .adapters.gemma4_vision import PROFILES

            profile = PROFILES.get(repo)
            if profile is None or profile["revision"] != revision:
                return False
            return all(
                hashlib.sha256(read(name, maximum)).hexdigest() == expected
                for name, maximum, expected in (
                    (
                        "processor_config.json",
                        65536,
                        profile.get(
                            "source_processor_sha256",
                            "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
                        ),
                    ),
                    (
                        "chat_template.jinja",
                        262144,
                        profile.get(
                            "template_sha256", "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
                        ),
                    ),
                )
            )
        check_vision_processing_metadata(names, read)
        return True
    except Exception:
        return False


def family_catalog(
    family_id,
    *,
    hub=None,
    limit=MAX_ITEMS,
    offline=False,
    offset=0,
    planning=False,
    deadline_seconds=80,
    cache_dir=None,
    on_item=None,
    physical_memory=None,
):
    """Return candidate downloads; actual installation must repeat pinned preflight.

    ``cache_dir`` reuses verdicts for unchanged revisions. ``on_item`` receives
    each accepted row as soon as it is checked, before the sorted report.
    Browsing rows carry a parent-model reference score when one is known, and
    sources with one are inspected first; ``physical_memory`` (this machine's
    by default) prefers models from a RAM list this machine can use.
    """
    from .catalog_cache import VerdictCache
    from .catalog_ranking import host_memory, reference_scores
    from .native_catalog import FAMILY_ID as NATIVE_FAMILY_ID
    from .native_catalog import FLINT_FAMILY_ID
    from .native_catalog import catalog_items as native_catalog_items
    from .native_catalog import unavailable_item as native_unavailable_item

    valid_offset = type(offset) is int and 0 <= offset <= 10_000_000

    def report(*args, **kwargs):
        return _report(*args, offset=offset if valid_offset else 0, **kwargs)

    if not valid_offset:
        return report(family_id or "", "error", "The model catalogue request is invalid.")
    if type(planning) is not bool or type(deadline_seconds) not in (int, float) or not 0 < deadline_seconds <= 600:
        return report(family_id or "", "error", "The model catalogue request is invalid.")

    native_family = family_id in {NATIVE_FAMILY_ID, FLINT_FAMILY_ID}
    family = family_id if native_family else normalize_architecture(family_id)
    definition = next((item for item in support_matrix()["families"] if item["family_id"] == family), None)
    if definition is None or not any(
        definition["modes"][mode]["status"] == "conditional" for mode in ("text", "vision-language")
    ):
        return report(family or "", "error", "This model family is not available in Stark.")
    if type(limit) is not int or not 1 <= limit <= MAX_ITEMS:
        return report(family, "error", "The model catalogue request is invalid.")
    if offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return report(family, "offline", "Connect to the internet to browse models.")
    if hub is None:
        import huggingface_hub

        hub = huggingface_hub
    deadline = time.monotonic() + deadline_seconds
    cache = VerdictCache(cache_dir, family, planning=planning) if cache_dir is not None else None
    # Guided setup ranks its own planned recipes; only browsing is ordered here.
    ranking = None if planning else reference_scores()
    if ranking is not None and physical_memory is None:
        physical_memory = host_memory()

    def scored(item, tags=()):
        """Attach the reference score a checked row may show."""
        if ranking is None:
            return item
        base_name = _without_quantization_suffix(item["repo"].split("/")[-1], item["quantization"])
        reference = ranking.reference(item["repo"], item["revision"], item["quantization"], tags, base_name)
        return item if reference is None else {**item, "reference_score": reference}

    def remaining():
        if time.monotonic() >= deadline:
            raise _Unavailable()

    try:
        try:
            api = hub.HfApi()
        except Exception:
            if not native_family:
                raise
            api = None
        pins = _catalog_pins(family)
        native_items = {item["repo"]: item for item in native_catalog_items(family)}
        seeds = list(native_items) if native_family else []
        seeds = list(dict.fromkeys([*seeds, *_catalog_seeds(family)]))
        discovery_failed = api is None

        def repositories():
            nonlocal discovery_failed
            for seed in seeds:
                yield seed, None, None
            if api is None:
                return
            expected = offset + limit + 1
            stream = _repository_search(
                api, definition, remaining, expected if ranking is None else max(expected, PRIORITY_WINDOW)
            )
            window, failed = [], False
            if ranking is not None:
                # The first listing positions are reordered so sources with a
                # reference score are inspected, and shown, before the rest.
                try:
                    for entry in stream:
                        window.append(entry)
                        if len(window) == PRIORITY_WINDOW:
                            break
                except Exception:
                    failed = True

                def placement(entry):
                    siblings = _field(entry[2], "siblings")
                    # A listed source without Safetensors weights cannot be
                    # downloaded, so its score earns it no earlier inspection.
                    importable = not isinstance(siblings, list) or any(
                        str(_field(sibling, "rfilename")).endswith(".safetensors") for sibling in siblings
                    )
                    rank = (
                        ranking.priority(entry[0], _field(entry[2], "tags"), physical_memory) if importable else None
                    )
                    # Scored sources from a list this machine can use, then
                    # other scored sources, then the rest in listing order.
                    return (2, 0) if rank is None else (0 if rank[0] else 1, -rank[1])

                yield from sorted(window, key=placement)
            if failed:
                discovery_failed = True
                return
            try:
                yield from stream
            except Exception:
                discovery_failed = True
                # Retain the frontier so a retry never skips a failed search.
                return

        candidates, seen, consumed, omitted = [], set(), offset, False
        for repo, listed_revision, listing in repositories():
            remaining()
            if repo.casefold() in seen:
                continue
            seen.add(repo.casefold())
            if len(seen) <= offset:
                continue
            if len(candidates) == limit or len(seen) - offset > limit + MAX_SCREENED:
                omitted = True
                break
            consumed = len(seen)
            # A source the listing already rules out uses its position without
            # any request of its own, so a page reaches further down the family.
            if listing is not None and not native_family and _listing_rejects(repo, listing):
                continue
            candidates.append((repo, listed_revision))
        omitted = omitted or discovery_failed
        if not candidates:
            if omitted and not discovery_failed:
                message = "No models on this page are available to download."
                return report(family, "ready", message, next_offset=consumed)
            return report(
                family,
                "error" if discovery_failed else "ready",
                "Models could not be loaded. Check your connection and try again."
                if discovery_failed
                else "No compatible models were found in this family.",
            )

        @lru_cache(maxsize=256)
        def read(repo, filename, revision, maximum, size=None):
            remaining()
            try:
                return _read_revision_metadata(hub, repo, filename, revision, maximum, size)
            except _Inaccessible:
                raise
            except Exception as error:
                if _hub_source_unavailable(error):
                    raise _Inaccessible() from None
                raise _Unavailable() from None

        def inspect(repo, listed_revision):
            remaining()
            if repo in native_items and repo not in pins:
                return scored(dict(native_items[repo]))
            known_revision = pins.get(repo) or listed_revision
            if cache is not None:
                # A registered source has no listed revision; its head is
                # trusted briefly after it was last confirmed.
                found, item = (
                    cache.load(repo, known_revision) if known_revision is not None else cache.load_recent(repo)
                )
                if found:
                    return item
            outcome = {"known": known_revision}
            item = examine(repo, outcome)
            revision = outcome.get("revision")
            if cache is None or revision is None:
                return item
            if outcome.get("reused"):
                cache.confirm(repo, revision)
            elif not (
                item is not None
                and item.get("unavailable_reason") in (qwen38_external.UNVERIFIED, catalog_reasons.GATED)
            ):
                # A verdict that could not be verified, including one waiting
                # on access approval, must be checked again.
                cache.store(repo, revision, item, settled=not outcome.get("volatile"))
            return item

        def examine(repo, outcome):
            try:
                requested_revision = pins.get(repo)
                try:
                    info = api.model_info(repo_id=repo, revision=requested_revision, files_metadata=True, timeout=5)
                except Exception as error:
                    if _hub_source_unavailable(error):
                        return None
                    raise
                revision = _field(info, "sha")
                if isinstance(revision, str) and _REVISION.fullmatch(revision) is not None:
                    outcome["revision"] = revision
                    # Sources without a listed revision (registered and pinned
                    # seeds) learn it here and can still skip every file read.
                    if cache is not None and revision != outcome["known"]:
                        found, item = cache.load(repo, revision)
                        if found:
                            outcome["reused"] = True
                            return item
                if not _application_source(info, repo):
                    return None
                if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
                    raise _Unavailable()
                if requested_revision is not None and revision != requested_revision:
                    raise _Unavailable()
                names, sizes = _weight_inventory(info)
                # The Hub reports "auto" or "manual" for a source whose files
                # need access approval, and False otherwise.
                gated = _field(info, "gated", False)
                gated = gated is True or isinstance(gated, str) and gated.casefold() in {"auto", "manual"}

                def read_pinned(filename, maximum):
                    # The pinned inventory already states each file's size.
                    size = sizes.get(filename)
                    return read(repo, filename, revision, maximum, size if type(size) is int else None)

                if native_family:
                    config = _json(read_pinned("config.json", MAX_CONFIG_BYTES)) if "config.json" in names else None
                    return scored(native_unavailable_item(family, repo, revision, config, names, sizes))
                # The original registry's Gemma3 wrappers predate the native
                # image adapter. Bind their complete, same-revision metadata
                # through the generic vision path rather than a text recipe.
                entry = None if family == "gemma3" else get_model(repo)
                if entry is not None and entry.get("adapter") != family:
                    return None
                rule = (
                    entry["files"]["model_config"] if entry is not None else {"repo": repo, "filename": "config.json"}
                )
                config_repo = rule["repo"]
                config_revision = revision
                if config_repo != repo:
                    # This verdict also depends on another repository's head.
                    outcome["volatile"] = True
                    try:
                        config_info = api.model_info(repo_id=config_repo, timeout=5)
                    except Exception as error:
                        if _hub_source_unavailable(error):
                            return None
                        raise
                    config_revision = _field(config_info, "sha")
                    if not isinstance(config_revision, str) or _REVISION.fullmatch(config_revision) is None:
                        raise _Unavailable()
                if config_repo == repo:
                    # The pinned inventory and the Hub's parsed architecture
                    # settle these rejections without requesting a file.
                    if rule["filename"] not in names:
                        return None
                    described = _field(info, "config")
                    declared = described.get("architectures") if isinstance(described, dict) else None
                    if (
                        requested_revision is None
                        and isinstance(declared, list)
                        and not _generation_architecture({"architectures": declared}, family)
                    ):
                        return None
                    config = _json(read_pinned(rule["filename"], MAX_CONFIG_BYTES))
                else:
                    config = _json(read(config_repo, rule["filename"], config_revision, MAX_CONFIG_BYTES))
                if not isinstance(config, dict) or not _generation_architecture(config, family):
                    return None
                text_config = config.get("text_config", config)
                bf16_qwen_moe = family == "qwen3_5_moe" and not any(
                    scope.get(key) is not None
                    for scope in (config, text_config)
                    if isinstance(scope, dict)
                    for key in ("quantization", "quantization_config")
                )
                base_mode = "vision-language" if family == "gemma3" or bf16_qwen_moe else "text"
                verdict = preflight_model(config, mode=base_mode, expected_architecture=family)
                if verdict["family_id"] != family:
                    return None
                unavailable_reason = None
                if verdict["status"] != "conditional":
                    unavailable_reason = verdict["message"]
                if repo in native_items:
                    # Retained image choices keep their exact native preparation
                    # identity after the pinned metadata check. A moved revision
                    # cannot silently replace the reviewed source or fall back
                    # to a text-only generic conversion.
                    item = dict(native_items[repo])
                    if unavailable_reason is not None:
                        item.update(downloadable=False, unavailable_reason=unavailable_reason)
                    return scored(item)
                external_qwen = qwen38_external.is_external_affine(config)
                quantization = (
                    qwen38_external.QUANTIZATION
                    if external_qwen
                    else _quantization(config, info, entry, family=family)
                )
                foreign = catalog_reasons.foreign_format(config) if quantization is None else None
                original = None
                if foreign is not None:
                    # A GPU-server quantization export is described by its
                    # format before any other check, and its tensor headers
                    # are never requested.
                    quantization, unavailable_reason = foreign
                    # The export's declared original lets the browser show it
                    # under that model's row instead of as a dead end.
                    original = catalog_reasons.declared_original(repo, _field(info, "tags", []))
                elif quantization is None:
                    unavailable_reason = unavailable_reason or (
                        "This checkpoint's declared weight format has no supported import path."
                    )
                    quantization = _unsupported_quantization(config)
                selected = _selected_weights(names, read_pinned)
                if selected is None:
                    selected = [
                        name
                        for name in names
                        if name.endswith(_WEIGHT_SUFFIXES) and not PurePosixPath(name).name.startswith("adapter_model")
                    ]
                    if not selected:
                        return None
                    unavailable_reason = unavailable_reason or (
                        "This checkpoint requires a supported Safetensors file or a complete shard index."
                    )
                if external_qwen and unavailable_reason is None:
                    indexed = qwen38_external.INDEX in names
                    if not qwen38_external.supported_shards(selected, indexed):
                        return None
                    try:
                        qwen38_external.external_files(repo, names)
                    except (ConversionError, UnsupportedConversion):
                        return None
                    index = (
                        _json(read_pinned(qwen38_external.INDEX, MAX_CONFIG_BYTES))["weight_map"] if indexed else None
                    )
                    try:
                        qwen38_external.verify_hub_layout(
                            hub, repo, revision, config, selected, index, remaining, sizes
                        )
                    except qwen38_external.LayoutError as error:
                        unavailable_reason = str(error)
                    except qwen38_external.HeaderAccessError:
                        unavailable_reason = catalog_reasons.GATED if gated else qwen38_external.UNVERIFIED
                    if unavailable_reason is not None:
                        quantization = "MLX-Serve"
                    selected = [*selected, qwen38_external.NGRAM_TABLE]
                weight_bytes = _weight_bytes(selected, sizes)
                modes = [base_mode]
                if unavailable_reason is None and family in {"qwen3_5", "qwen3_5_moe", "gemma3", "gemma4_text"}:
                    image_metadata = _catalog_image_metadata(family, config, repo, revision, names, read_pinned)
                    if base_mode == "vision-language" and (quantization != "BF16" or not image_metadata):
                        unavailable_reason = (
                            "Image import requires supported BF16 vision weights and complete processing metadata."
                        )
                    if base_mode != "vision-language" and image_metadata:
                        modes.append("vision-language")
                if unavailable_reason is None and not external_qwen:
                    index = (
                        _json(read_pinned("model.safetensors.index.json", MAX_CONFIG_BYTES))["weight_map"]
                        if "model.safetensors.index.json" in names
                        else None
                    )
                    layout = _verify_hub_layout(
                        family, hub, repo, revision, config, selected, index, sizes, remaining, modes
                    )
                    unavailable_reason = next((layout[mode] for mode in modes if layout[mode] is not None), None)
                    if gated and isinstance(unavailable_reason, _HeadersUnreadable):
                        # A gated source's header reads fail until the user
                        # has been granted access.
                        unavailable_reason = catalog_reasons.GATED
                    elif unavailable_reason is not None:
                        unavailable_reason = str(unavailable_reason)
                extra_weights = any(name.endswith(".safetensors") and name not in selected for name in names)
                parameters = _parameters(info, quantization != "BF16" or extra_weights)
                if parameters is None and quantization != "BF16" and not extra_weights:
                    parameters = _dense_config_parameters(config, family)
                chat_template, chat_template_blocker = (None, None)
                if planning:
                    chat_template, chat_template_blocker = _source_chat_template(entry, repo, names, read_pinned)
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
                    "default_artifact_mode": modes[-1]
                    if family == "qwen3_5" and quantization == "BF16"
                    else base_mode,
                    "downloadable": unavailable_reason is None,
                    **({"unavailable_reason": unavailable_reason} if unavailable_reason is not None else {}),
                    **({"original_repo": original} if original is not None else {}),
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
                return scored(item, _field(info, "tags", []))
            except _Inaccessible:
                # Gating and file availability can change without a new revision.
                outcome["volatile"] = True
                return None
            except _MalformedMetadata:
                return None
            except _Unavailable:
                raise
            except Exception:
                raise _Unavailable() from None

        accepted, failures = [], 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(inspect, repo, listed_revision) for repo, listed_revision in candidates]
            for future in as_completed(futures):
                try:
                    item = future.result()
                    if item is not None:
                        accepted.append(item)
                        if on_item is not None:
                            on_item(item)
                except Exception:
                    failures += 1
        accepted.sort(key=_size_order)
        if failures and (not planning or not accepted):
            return report(family, "error", "Models could not be loaded. Check your connection and try again.")
        partial_failure = discovery_failed or bool(failures)
        result = report(
            family,
            "ready",
            "Some model metadata could not be checked. Check again to retry."
            if failures
            else "Choose a model to download."
            if any(item["downloadable"] for item in accepted)
            else "No models on this page are available to download."
            if accepted
            else "No compatible models were found in this family.",
            accepted,
            # A partial guided result must retry this window instead of
            # advancing past sources whose metadata could not be checked.
            next_offset=offset if planning and partial_failure else consumed if omitted else None,
        )
        if planning:
            result["incomplete"] = omitted or partial_failure
            result["partial_failure"] = partial_failure
        return result
    except Exception:
        return report(family, "error", "Models could not be loaded. Check your connection and try again.")
