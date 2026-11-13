"""Fail-closed tokenizer admission for the pinned Edge0 evaluation artifact.

This compares public artifact metadata, never inference requests or responses.
No tokenizer is synthesized by reclassifying tokens or changing normalization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split

UPSTREAM_REPO = "Qwen/Qwen3.8-Flash-Next"
UPSTREAM_REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"
UPSTREAM_FILES = {
    "tokenizer.json": (12_809_320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    "tokenizer_config.json": (9_984, "207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e"),
}
_BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
_BPE_OPTIONS = {
    "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
    "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
}


class TokenizerEquivalenceError(ValueError):
    """Admission failed; report contains only public metadata aggregates/hashes."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("Edge0 tokenizer equivalence rejected: " + ", ".join(report["failures"]))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _upstream_bytes(name: str, *, offline: bool = False) -> bytes:
    size, digest = UPSTREAM_FILES[name]
    path = Path(hf_hub_download(
        repo_id=UPSTREAM_REPO, filename=name, revision=UPSTREAM_REVISION, local_files_only=offline,
    ))
    # Hub snapshots may link their content-addressed blob; verify response bytes.
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("Pinned upstream tokenizer metadata identity mismatch")
    data = path.read_bytes()
    if len(data) != size or _sha256(data) != digest:
        raise ValueError("Pinned upstream tokenizer metadata identity mismatch")
    return data


def _json(data: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate tokenizer metadata key")
            result[key] = value
        return result

    result = json.loads(data, object_pairs_hook=unique_pairs)
    if not isinstance(result, dict):
        raise ValueError("Tokenizer metadata must be an object")
    return result


def _read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, linked or oversized tokenizer metadata")
    return path.read_bytes()


def _vocab(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("Invalid tokenizer vocabulary")
    if any(not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in value.items()):
        raise ValueError("Invalid tokenizer token identity")
    if len(set(value.values())) != len(value):
        raise ValueError("Duplicate tokenizer token ID")
    return value


def _merges(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Invalid tokenizer merge array")
    result = []
    for item in value:
        if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            item = " ".join(item)
        if not isinstance(item, str) or not item or len(item.split(" ")) != 2:
            raise ValueError("Invalid tokenizer merge")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate tokenizer merge")
    return result


def _ranges(ids: list[int]) -> list[list[int]]:
    result = []
    for token_id in sorted(ids):
        if result and result[-1][1] + 1 == token_id:
            result[-1][1] = token_id
        else:
            result.append([token_id, token_id])
    return result


def _compare(directory: Path, gguf: dict, upstream: dict[str, bytes]) -> dict:
    files = {
        name: _read(directory / "tokenizer" / name, limit)
        for name, limit in (
            ("tokenizer.json", 64_000), ("vocab.json", 16_000_000),
            ("merges.txt", 16_000_000), ("token_type.json", 4_000_000),
        )
    }
    descriptor = _json(files["tokenizer.json"])
    shipped_vocab = _vocab(_json(files["vocab.json"]))
    shipped_merges = _merges(files["merges.txt"].decode("utf-8").splitlines())
    hf = _json(upstream["tokenizer.json"])
    config = _json(upstream["tokenizer_config.json"])
    model = hf.get("model", {})
    hf_vocab = dict(_vocab(model.get("vocab")))
    expected_types = {token_id: 1 for token_id in hf_vocab.values()}
    added = hf.get("added_tokens")
    if not isinstance(added, list):
        raise ValueError("Invalid upstream added tokens")
    unsupported_added = 0
    for token in added:
        if not isinstance(token, dict) or type(token.get("id")) is not int or token["id"] < 0:
            raise ValueError("Invalid upstream added token")
        content = token.get("content")
        if not isinstance(content, str) or not content or type(token.get("special")) is not bool:
            raise ValueError("Invalid upstream added token semantics")
        if content in hf_vocab and hf_vocab[content] != token["id"]:
            raise ValueError("Conflicting upstream token ID")
        hf_vocab[content] = token["id"]
        expected_types[token["id"]] = 3 if token["special"] else 4
        unsupported_added += any(
            token.get(flag) is not False for flag in ("lstrip", "rstrip", "normalized", "single_word")
        )
        if config.get("added_tokens_decoder", {}).get(str(token["id"])) != {
            key: value for key, value in token.items() if key != "id"
        }:
            raise ValueError("Upstream added-token metadata disagrees")
    _vocab(hf_vocab)
    tokens = gguf.get("tokenizer.ggml.tokens")
    types = gguf.get("tokenizer.ggml.token_type")
    if not isinstance(tokens, list) or not isinstance(types, list) or len(types) != len(tokens):
        raise ValueError("Invalid GGUF tokenizer arrays")
    if any(not isinstance(token, str) for token in tokens) or any(type(t) is not int for t in types):
        raise ValueError("Invalid GGUF tokenizer array value")
    shipped_types = json.loads(files["token_type.json"])
    if not isinstance(shipped_types, list) or any(type(t) is not int for t in shipped_types):
        raise ValueError("Invalid shipped token type array")
    failures = []
    if shipped_types != types:
        failures.append("shipped_gguf_token_types_mismatch")
    if set(shipped_vocab.values()) != set(range(len(shipped_vocab))):
        failures.append("shipped_token_ids_not_contiguous")
    if descriptor.get("vocab_size") != len(shipped_vocab) or descriptor.get("n_merges") != len(shipped_merges):
        failures.append("descriptor_counts_mismatch")
    if len(tokens) != len(shipped_vocab) or any(
        token_id >= len(tokens) or tokens[token_id] != token for token, token_id in shipped_vocab.items()
    ):
        failures.append("shipped_gguf_token_ids_mismatch")
    if shipped_merges != _merges(gguf.get("tokenizer.ggml.merges")):
        failures.append("shipped_gguf_merge_order_mismatch")
    if any(descriptor.get(key) != expected or gguf.get(f"tokenizer.ggml.{key}") != expected
           for key, expected in (("model", "gpt2"), ("pre", "qwen35"))):
        failures.append("unsupported_tokenizer_kind")
    if descriptor.get("add_bos_token") is not False or gguf.get("tokenizer.ggml.add_bos_token") is not False:
        failures.append("bos_insertion_mismatch")
    special_ids = descriptor.get("special_ids", {})
    if set(special_ids) != {f"tokenizer.ggml.{name}_token_id" for name in ("bos", "eos", "padding")}:
        failures.append("descriptor_special_id_keys_mismatch")
    if any(gguf.get(key) != value or type(value) is not int for key, value in special_ids.items()):
        failures.append("shipped_gguf_special_ids_mismatch")
    upstream_special_mismatches = []
    for gguf_name, hf_name in (("bos", "bos"), ("eos", "eos"), ("padding", "pad")):
        content = config.get(f"{hf_name}_token")
        upstream_id = hf_vocab.get(content) if isinstance(content, str) else None
        if special_ids.get(f"tokenizer.ggml.{gguf_name}_token_id") != upstream_id:
            upstream_special_mismatches.append(gguf_name)
    if upstream_special_mismatches:
        failures.append("upstream_special_ids_mismatch")
    mismatch_ids = [token_id for token_id, expected in expected_types.items()
                    if token_id >= len(types) or types[token_id] != expected]
    if mismatch_ids:
        failures.append("upstream_token_types_mismatch")
    absent_ids = [token_id for token, token_id in shipped_vocab.items() if token not in hf_vocab]
    if shipped_vocab != hf_vocab:
        failures.append("upstream_token_ids_mismatch")
    if shipped_merges != _merges(model.get("merges")):
        failures.append("upstream_merge_order_mismatch")
    if unsupported_added:
        failures.append("unsupported_added_token_semantics")
    if any(model.get(key) != value for key, value in _BPE_OPTIONS.items()):
        failures.append("unsupported_bpe_semantics")
    if hf.get("normalizer") is not None:
        # qwen35 in the pinned Strata llama.cpp BPE path has no NFC pass.
        failures.append("upstream_normalizer_mismatch")
    pattern = descriptor.get("pre_pattern")
    if not isinstance(pattern, str):
        raise ValueError("Missing Edge0 pre-tokenizer pattern")
    expected_pre = {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
        _BYTE_LEVEL,
    ]}
    if hf.get("pre_tokenizer") != expected_pre:
        failures.append("upstream_pretokenizer_mismatch")
    if config.get("pretokenize_regex") != pattern or config.get("add_prefix_space") is not False:
        failures.append("upstream_pretokenizer_config_mismatch")
    if hf.get("decoder") != _BYTE_LEVEL or hf.get("post_processor") != _BYTE_LEVEL:
        failures.append("unsupported_byte_level_semantics")
    if hf.get("truncation") is not None or hf.get("padding") is not None or config.get("add_bos_token") is not False:
        failures.append("unsupported_tokenizer_processing")
    # A bounded, public synthetic witness detects Unicode case-fold differences
    # between (?i:'s) and the ASCII-only [sS] expansion. No input text is retained.
    witnesses = []
    pre = hf.get("pre_tokenizer", {}).get("pretokenizers", [])
    if pre and isinstance(pre[0].get("pattern", {}).get("Regex"), str):
        upstream_split = Split(Regex(pre[0]["pattern"]["Regex"]), "isolated")
        shipped_split = Split(Regex(pattern), "isolated")
        witness = "\u0027\u017fa"
        upstream_pieces = upstream_split.pre_tokenize_str(witness)
        shipped_pieces = shipped_split.pre_tokenize_str(witness)
        witnesses.append({"sha256": _sha256(witness.encode()), "equivalent": upstream_pieces == shipped_pieces,
                          "upstream_piece_count": len(upstream_pieces), "shipped_piece_count": len(shipped_pieces)})
        if upstream_pieces != shipped_pieces and "upstream_pretokenizer_mismatch" not in failures:
            failures.append("upstream_pretokenizer_mismatch")
    return {
        "equivalent": not failures,
        "failures": failures,
        "upstream": {"repo_id": UPSTREAM_REPO, "revision": UPSTREAM_REVISION,
                     "files": {name: {"bytes": len(data), "sha256": _sha256(data)}
                               for name, data in upstream.items()}},
        "shipped_files": {f"tokenizer/{name}": {"bytes": len(data), "sha256": _sha256(data)}
                          for name, data in files.items()},
        "counts": {"shipped_vocab": len(shipped_vocab), "upstream_vocab_with_added": len(hf_vocab),
                   "merges": len(shipped_merges), "token_type_mismatches": len(mismatch_ids),
                   "shipped_ids_absent_upstream": len(absent_ids)},
        "token_type_mismatch_id_ranges": _ranges(mismatch_ids),
        "shipped_ids_absent_upstream_ranges": _ranges(absent_ids),
        "absent_upstream_all_gguf_unused": all(
            token_id < len(types) and types[token_id] == 5 for token_id in absent_ids
        ),
        "special_id_mismatch_fields": upstream_special_mismatches,
        "upstream_normalizer_type": (hf.get("normalizer") or {}).get("type"),
        "pretokenizer_witnesses": witnesses,
        "native_runtime_validated": False,
    }


def preflight_tokenizer(directory: Path, gguf_metadata: dict, *, offline: bool = False) -> dict:
    """Compare small metadata and bounded GGUF header without writing derived files."""
    upstream = {name: _upstream_bytes(name, offline=offline) for name in UPSTREAM_FILES}
    return _compare(Path(directory), gguf_metadata, upstream)


def prepare_tokenizer(directory: Path, gguf_metadata: dict, *, offline: bool = False) -> dict:
    """Copy pinned HF metadata only after exact equivalence; otherwise fail closed."""
    directory = Path(directory)
    upstream = {name: _upstream_bytes(name, offline=offline) for name in UPSTREAM_FILES}
    report = _compare(directory, gguf_metadata, upstream)
    if not report["equivalent"]:
        raise TokenizerEquivalenceError(report)
    # Publication of the enclosing staging directory is the caller's atomic step.
    for name in upstream:
        target = directory / name
        if target.is_symlink() or (target.exists() and _read(target, len(upstream[name])) != upstream[name]):
            raise ValueError("Derived tokenizer target identity mismatch")
    derived = {}
    for name, data in upstream.items():
        if not (directory / name).exists():
            (directory / name).write_bytes(data)
        derived[name] = {
            "sha256": _sha256(data), "bytes": len(data), "derivation": "byte-identical-after-tokenizer-equivalence",
            "source": {"repo_id": UPSTREAM_REPO, "revision": UPSTREAM_REVISION, "path": name, "sha256": _sha256(data)},
        }
    return {"derived_metadata": derived, "tokenizer_equivalence": report}
