"""Experimental, source-native Edge0 tokenizer qualification; no admission hook.

Independent representation of the pinned artifact metadata and the qwen35
semantics in llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d. No upstream
implementation is copied. A bounded oracle comparison is required separately.
Corpus strings, token sequences and decoded text are never written or reported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
import time
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split

from ._build_metadata import open_regular
from .edge0_source import EDGE0_FILES, EDGE0_REPO_ID, EDGE0_REVISION, source_path

CANDIDATE_REVISION = "edge0-native-tokenizer-v1"
UNICODE_CANDIDATE_REVISION = "edge0-native-tokenizer-v2-unicode"
CORPUS_REVISION = "edge0-native-tokenizer-corpus-v1"
TEXT_POLICY = "concatenate-native-pieces-then-utf8-replace"
_BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
UNICODE_FLAGS_SHA256 = "ac10d2cb7eea9e4686334ca6cf7e4d37fa586504ed793702ac81f49b55d7fdd4"
_UNICODE_MASKS = {"L": 4, "M": 16, "N": 2, "s": 256}


def _scalar(value: int) -> bool:
    return not 0xD800 <= value <= 0xDFFF


def derive_unicode_asset(flags: bytes, license_text: bytes) -> dict:
    """Derive public property intervals from the exact pinned native flag oracle."""
    if len(flags) != 0x110000 * 2 or _sha(flags) != UNICODE_FLAGS_SHA256:
        raise ValueError("Pinned native Unicode flag inventory mismatch")
    if _sha(license_text) != "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d":
        raise ValueError("Pinned Unicode source license identity mismatch")
    properties = {name: [] for name in _UNICODE_MASKS}
    for cp, (value,) in enumerate(struct.iter_unpack("<H", flags)):
        if not _scalar(cp):
            continue
        for name, mask in _UNICODE_MASKS.items():
            if value & mask:
                ranges = properties[name]
                if ranges and ranges[-1][1] + 1 == cp:
                    ranges[-1][1] = cp
                else:
                    ranges.append([cp, cp])
    return {
        "schema": 1, "license": "MIT", "license_text": license_text.decode(),
        "attribution": "Derived Unicode classification ranges; Copyright (c) 2023-2026 The ggml authors",
        "source": {"repo_id": "ggml-org/llama.cpp", "revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
                   "files": {
                       "src/unicode.h": "f1562388d5b9d2dac1152ad8eda56e0bcae25dd407cf15d9dd2e96fe0124f4e7",
                       "src/unicode.cpp": "b1f3a18646197e21f7300c6d3b45c9faf52a123d58a38d7d54f3f225018da45d",
                       "src/unicode-data.cpp": "95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d",
                   }},
        "derivation": {"function": "unicode_cpt_flags_from_cpt().as_uint()", "flags_sha256": _sha(flags),
                       "flags_encoding": "little-endian uint16", "property_masks": _UNICODE_MASKS,
                       "scalar_count": 0x110000 - 0x800, "surrogates_excluded": True},
        "properties": properties,
    }


def _range_text(ranges: list[list[int]]) -> str:
    previous = -1
    result = []
    for start, end in ranges:
        if not (type(start) is int and type(end) is int and previous < start <= end < 0x110000):
            raise ValueError("Invalid Unicode property interval")
        if start <= 0xDFFF and end >= 0xD800:
            raise ValueError("Surrogate in Unicode property interval")
        first = f"\\x{{{start:X}}}"
        result.append(first if start == end else first + f"-\\x{{{end:X}}}")
        previous = end
    return "".join(result)


def _complement_ranges(ranges: list[list[int]]) -> list[list[int]]:
    _range_text(ranges)
    result = []
    for low, high in ((0, 0xD7FF), (0xE000, 0x10FFFF)):
        cursor = low
        for start, end in ranges:
            if end < low or start > high:
                continue
            if cursor < start:
                result.append([cursor, start - 1])
            cursor = end + 1
        if cursor <= high:
            result.append([cursor, high])
    return result


def expanded_pattern(pattern: str, asset: dict) -> str:
    """Replace every source property with explicit ranges, respecting class scope."""
    content = {name: _range_text(ranges) for name, ranges in asset["properties"].items()}
    content["S"] = _range_text(_complement_ranges(asset["properties"]["s"]))
    replacements = {**{rf"\p{{{name}}}": content[name] for name in ("L", "M", "N")},
                    r"\s": content["s"], r"\S": content["S"]}
    output = []
    inside = False
    offset = 0
    while offset < len(pattern):
        replacement = next((key for key in replacements if pattern.startswith(key, offset)), None)
        if replacement is not None:
            output.append(replacements[replacement] if inside else "[" + replacements[replacement] + "]")
            offset += len(replacement)
            continue
        char = pattern[offset]
        if char == "\\":
            if offset + 1 == len(pattern) or pattern[offset + 1] in "pPsSdDwW":
                raise ValueError("Unsupported unexpanded regular expression property")
            output.append(pattern[offset:offset + 2])
            offset += 2
            continue
        if char == "[":
            if inside:
                raise ValueError("Nested source character class")
            inside = True
        elif char == "]":
            if not inside:
                raise ValueError("Unbalanced source character class")
            inside = False
        output.append(char)
        offset += 1
    if inside:
        raise ValueError("Unbalanced source character class")
    return "".join(output)


def verify_unicode_membership(asset: dict, flags: bytes) -> dict:
    """Check compiled HF regex membership for every valid scalar, in four classes."""
    if _sha(flags) != UNICODE_FLAGS_SHA256 or len(flags) != 0x110000 * 2:
        raise ValueError("Pinned Unicode property proof input mismatch")
    text = "".join(chr(cp) for cp in range(0x110000) if _scalar(cp))
    intervals = asset["properties"]
    mismatches = {}
    counts = {}
    for name, mask in _UNICODE_MASKS.items():
        pattern = "[" + _range_text(intervals[name]) + "]+"
        splitter = Split(Regex(pattern), "removed", invert=True)
        observed = bytearray(0x110000)
        for selected, _ in splitter.pre_tokenize_str(text):
            for char in selected:
                observed[ord(char)] = 1
        mismatch = 0
        for cp, (value,) in enumerate(struct.iter_unpack("<H", flags)):
            if _scalar(cp):
                mismatch += bool(observed[cp]) != bool(value & mask)
        mismatches[name] = mismatch
        counts[name] = sum(observed)
    return {"asset_sha256": _sha(_json_bytes(asset)), "flags_sha256": _sha(flags),
            "scalar_count": 0x110000 - 0x800, "membership_tests": 4 * (0x110000 - 0x800),
            "property_counts": counts, "property_mismatches": mismatches, "passed": not any(mismatches.values())}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sealed_metadata(directory: Path, gguf: dict) -> dict:
    """Verify immutable small inputs and their agreement with the GGUF arrays."""
    data = {}
    for name in ("tokenizer.json", "vocab.json", "merges.txt", "token_type.json"):
        rel = f"tokenizer/{name}"
        expected = EDGE0_FILES[rel]
        with open_regular(source_path(directory, rel)) as stream:
            value = stream.read(expected["size"] + 1)
        if len(value) != expected["size"] or _sha(value) != expected["sha256"]:
            raise ValueError("Native tokenizer source identity mismatch")
        data[name] = value
    descriptor = json.loads(data["tokenizer.json"])
    vocab = json.loads(data["vocab.json"])
    types = json.loads(data["token_type.json"])
    merges = data["merges.txt"].decode().splitlines()
    if (descriptor["model"], descriptor["pre"], descriptor["add_bos_token"]) != ("gpt2", "qwen35", False):
        raise ValueError("Unsupported native tokenizer semantics")
    if sorted(vocab.values()) != list(range(len(vocab))) or len(types) != len(vocab):
        raise ValueError("Invalid native tokenizer inventory")
    tokens = [None] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    for name, value in (("tokens", tokens), ("token_type", types), ("merges", merges)):
        if gguf.get(f"tokenizer.ggml.{name}") != value:
            raise ValueError("Native tokenizer GGUF arrays disagree")
    for key, value in descriptor["special_ids"].items():
        if gguf.get(key) != value:
            raise ValueError("Native tokenizer special IDs disagree")
    if descriptor["vocab_size"] != len(tokens) or descriptor["n_merges"] != len(merges):
        raise ValueError("Native tokenizer descriptor counts disagree")
    if set(types) - {1, 3, 4, 5}:
        raise ValueError("Unsupported native tokenizer token class")
    return {"descriptor": descriptor, "vocab": vocab, "types": types, "tokens": tokens, "merges": merges}


def candidate_program(metadata: dict, *, unicode_asset: dict | None = None) -> dict:
    """Construct an in-memory candidate. This does not qualify or publish it."""
    added = []
    unused = []
    for token_id, (token, kind) in enumerate(zip(metadata["tokens"], metadata["types"], strict=True)):
        if kind in (3, 4):
            if not token.isascii():
                raise ValueError("Non-ASCII added-token decoding requires a separate representation")
            added.append({"id": token_id, "content": token, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": kind == 3})
        elif kind == 5:
            unused.append(re.escape(token))
    decoder = _BYTE_LEVEL
    if unused:
        # UNUSED slots are in the output vocabulary, never literal added tokens.
        # Replace operates per token before ByteLevel concatenates its byte pieces.
        decoder = {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"Regex": "^(?:" + "|".join(unused) + ")$"}, "content": ""},
            _BYTE_LEVEL,
        ]}
    pattern = metadata["descriptor"]["pre_pattern"]
    if unicode_asset is not None:
        pattern = expanded_pattern(pattern, unicode_asset)
    return {
        "version": "1.0", "truncation": None, "padding": None, "normalizer": None,
        "added_tokens": added,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern},
             "behavior": "Isolated", "invert": False}, _BYTE_LEVEL,
        ]},
        "post_processor": _BYTE_LEVEL, "decoder": decoder,
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": metadata["vocab"], "merges": [merge.split(" ") for merge in metadata["merges"]]},
    }


@cache
def _byte_inverse() -> dict[str, int]:
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {chr(value): value for value in visible}
    for offset, value in enumerate(value for value in range(256) if value not in visible):
        mapping[chr(256 + offset)] = value
    return mapping


def raw_piece(metadata: dict, token_id: int, *, render_special: bool) -> bytes:
    """Independent inverse-byte mapping, before the worker UTF-8 text boundary."""
    kind = metadata["types"][token_id]
    token = metadata["tokens"][token_id]
    if kind == 5 or (kind == 3 and not render_special):
        return b""
    if kind in (3, 4):
        return token.encode()
    inverse = _byte_inverse()
    return bytes(inverse[char] for char in token)


def encode_corpus(metadata: dict) -> Iterator[str]:
    """Deterministic synthetic strings; generated only in memory."""
    yield from ("", "a", "b", "ab", "ba", "Hello.", "hello world", "  a   b\t\r\n", "\x00\x01\x7f",
                "e\u0301 é Å A\u030a", "한국어 中文 日本語 العربية русский", "'ſa x'ſa 'ſſ 's 'S 're 'RE",
                "1234567890", "{}[]()<>!?___", "\n\r\n\t\v\f ", "😀👩\u200d💻🇺🇸", "ababababababab")
    for value in range(256):
        yield chr(value)
    for value in range(0, 0x110000, 997):
        if not 0xD800 <= value <= 0xDFFF:
            char = chr(value)
            yield "a" + char + "'s " + char + " " + char * 2
    for token, kind in zip(metadata["tokens"], metadata["types"], strict=True):
        if kind in (3, 4, 5):
            yield token
            yield "a" + token + " b"
    yield "".join(token for token, kind in zip(metadata["tokens"], metadata["types"], strict=True) if kind in (3, 4))


def boundary_corpus(metadata: dict) -> Iterator[str]:
    yield from encode_corpus(metadata)
    yield "\u0027\u017fa"


def boundary_corpus_identity(metadata: dict) -> dict:
    digest = hashlib.sha256()
    count = 0
    for text in boundary_corpus(metadata):
        value = text.encode()
        digest.update(struct.pack("<I", len(value)) + value)
        count += 1
    return {"sha256": digest.hexdigest(), "cases": count,
            "additional_witness_sha256": _sha("\u0027\u017fa".encode())}


def decode_corpus(metadata: dict) -> Iterator[list[int]]:
    for token_id in range(len(metadata["tokens"])):
        yield [token_id]
    byte_ids = {raw_piece(metadata, i, render_special=True): i for i, kind in enumerate(metadata["types"])
                if kind == 1 and len(metadata["tokens"][i]) == 1}
    for value in (b"\xc3\xa9", b"\xf0\x9f\x98\x80", b"\xe2\x82\xac", b"\xff\xc0\x80\xe2\x82",
                  b"\xf0\x80\x80\x80", bytes(range(256))):
        yield [byte_ids[bytes([byte])] for byte in value]
    yield [i for i, kind in enumerate(metadata["types"]) if kind in (3, 4, 5)]


def corpus_identity(metadata: dict) -> dict:
    digest = hashlib.sha256()
    counts = {"encode": 0, "decode": 0}
    for text in encode_corpus(metadata):
        data = text.encode()
        digest.update(b"E" + struct.pack("<I", len(data)) + data)
        counts["encode"] += 1
    for ids in decode_corpus(metadata):
        digest.update(b"D" + struct.pack("<I", len(ids)))
        for token_id in ids:
            digest.update(struct.pack("<I", token_id))
        counts["decode"] += 1
    return {"revision": CORPUS_REVISION, "sha256": digest.hexdigest(), "cases": counts,
            "encode_modes": [{"add_special_tokens": False, "parse_special": True},
                             {"add_special_tokens": True, "parse_special": True}],
            "decode_skip_special_modes": [False, True], "text_policy": TEXT_POLICY,
            "raw_piece_modes": [False, True], "comparison_timeout_seconds": 120}


def negative_programs(program: dict, *, metadata: dict | None = None) -> Iterator[tuple[str, dict]]:
    names = ["nfc", "unicode_casefold_regex", "control_types", "vocab_ids", "merge_order"]
    if metadata is not None:
        names.append("implicit_unicode_properties")
    for name in names:
        mutant = copy.deepcopy(program)
        if name == "nfc":
            mutant["normalizer"] = {"type": "NFC"}
        elif name == "unicode_casefold_regex":
            node = mutant["pre_tokenizer"]["pretokenizers"][0]["pattern"]
            node["Regex"] = node["Regex"].replace(
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
            )
        elif name == "control_types":
            for token in mutant["added_tokens"]:
                if 248060 <= token["id"] <= 248065:
                    token["special"] = False
        elif name == "vocab_ids":
            vocab = mutant["model"]["vocab"]
            vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
        elif name == "merge_order":
            mutant["model"]["merges"].reverse()
        else:
            mutant["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = metadata["descriptor"]["pre_pattern"]
        yield name, mutant


def compare_candidate(
    metadata: dict, oracle, *, registered_corpus_sha256: str,
    worker_utf8_policy_verified: bool = False, timeout_seconds: float = 120,
    unicode_asset: dict | None = None, unicode_membership_proof: dict | None = None,
    pretokenizer_reference=None,
) -> dict:
    """Run a serialized, bounded comparison; return aggregates without artifacts.

    The caller must preregister corpus_identity() before invoking this function.
    UTF-8 replacement is the worker's text boundary, not native raw-piece parity.
    """
    corpus = corpus_identity(metadata)
    if corpus["sha256"] != registered_corpus_sha256 or not 0 < timeout_seconds <= 120:
        raise ValueError("Corpus registration or timeout mismatch")
    if unicode_asset is not None and (
        not unicode_membership_proof or unicode_membership_proof.get("passed") is not True
        or unicode_membership_proof.get("asset_sha256") != _sha(_json_bytes(unicode_asset))
        or unicode_membership_proof.get("flags_sha256") != UNICODE_FLAGS_SHA256
        or unicode_membership_proof.get("membership_tests") != 4 * (0x110000 - 0x800)
    ):
        raise ValueError("Unicode candidate requires an exhaustive source-bound membership proof")
    program = candidate_program(metadata, unicode_asset=unicode_asset)
    candidate = Tokenizer.from_str(_json_bytes(program).decode())
    deadline = time.monotonic() + timeout_seconds

    def check_deadline():
        if time.monotonic() > deadline:
            raise TimeoutError("Native tokenizer comparison deadline exceeded")

    mismatches = {"encode": 0, "pretokenizer_boundaries": 0,
                  "decode_valid_utf8": 0, "decode_utf8_replacement": 0,
                  "raw_piece": 0, "class_inventory": 0, "eog": 0}
    encode_references = []
    boundary_references = []
    split_pattern = program["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    candidate_splitter = Split(Regex(split_pattern), "isolated")
    encode_mismatch_cases = []
    for case, text in enumerate(encode_corpus(metadata)):
        check_deadline()
        reference = oracle.encode(text, parse_special=True)
        encode_references.append(reference)
        case_mismatch = False
        for add_special in (False, True):
            mismatch = candidate.encode(text, add_special_tokens=add_special).ids != reference
            mismatches["encode"] += mismatch
            case_mismatch |= mismatch
        if case_mismatch:
            encode_mismatch_cases.append(case)
    if pretokenizer_reference is not None:
        for text in boundary_corpus(metadata):
            check_deadline()
            expected_boundaries = pretokenizer_reference(text)
            observed_boundaries = [len(piece.encode()) for piece, _ in candidate_splitter.pre_tokenize_str(text)]
            if sum(expected_boundaries) != len(text.encode()) or any(value <= 0 for value in expected_boundaries):
                raise ValueError("Invalid native pre-tokenizer boundary inventory")
            boundary_references.append(expected_boundaries)
            mismatches["pretokenizer_boundaries"] += observed_boundaries != expected_boundaries
    raw = {False: [], True: []}
    expected_attrs = {1: 4, 3: 8, 4: 16, 5: 2}
    eog_expected = {metadata["vocab"][name] for name in (
        "<|endoftext|>", "<|im_end|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
    )}
    for token_id, kind in enumerate(metadata["types"]):
        check_deadline()
        mismatches["class_inventory"] += oracle.attr(token_id) != expected_attrs[kind]
        mismatches["eog"] += oracle.is_eog(token_id) != (token_id in eog_expected)
        for render in (False, True):
            piece = oracle.piece(token_id, special=render)
            raw[render].append(piece)
            mismatches["raw_piece"] += piece != raw_piece(metadata, token_id, render_special=render)
    decode_mode_counts = {"valid_utf8": 0, "utf8_replacement": 0}
    for ids in decode_corpus(metadata):
        check_deadline()
        for skip in (False, True):
            piece_bytes = b"".join(raw[not skip][token_id] for token_id in ids)
            try:
                reference = piece_bytes.decode("utf-8", errors="strict")
                kind = "valid_utf8"
            except UnicodeDecodeError:
                reference = piece_bytes.decode("utf-8", errors="replace")
                kind = "utf8_replacement"
            mismatches[f"decode_{kind}"] += candidate.decode(ids, skip_special_tokens=skip) != reference
            decode_mode_counts[kind] += 1
    negatives = {}
    # Counterexamples are counted independently in each mode; no text/IDs leave memory.
    for name, mutant in negative_programs(program, metadata=metadata if unicode_asset else None):
        check_deadline()
        changed = Tokenizer.from_str(_json_bytes(mutant).decode())
        changed_pattern = mutant["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        changed_splitter = Split(Regex(changed_pattern), "isolated")
        encoding_errors = 0
        for text, reference in zip(encode_corpus(metadata), encode_references, strict=True):
            check_deadline()
            encoding_errors += changed.encode(text, add_special_tokens=False).ids != reference
        decoding_errors = 0
        boundary_errors = 0
        if pretokenizer_reference is not None:
            for text, expected_boundaries in zip(boundary_corpus(metadata), boundary_references, strict=True):
                check_deadline()
                boundaries = [len(piece.encode()) for piece, _ in changed_splitter.pre_tokenize_str(text)]
                boundary_errors += boundaries != expected_boundaries
        # The type mutant is specifically exercised on all special classes.
        if name == "control_types":
            for token_id, kind in enumerate(metadata["types"]):
                if kind in (3, 4):
                    reference = raw[False][token_id].decode("utf-8", errors="replace")
                    decoding_errors += changed.decode([token_id], skip_special_tokens=True) != reference
        negatives[name] = {"encoding_mismatches": encoding_errors, "decoding_mismatches": decoding_errors,
                           "pretokenizer_boundary_mismatches": boundary_errors,
                           "detected": bool(encoding_errors or decoding_errors or boundary_errors)}
    special = oracle.special_ids()
    source_special = metadata["descriptor"]["special_ids"]
    expected_special = {
        "bos": source_special["tokenizer.ggml.bos_token_id"],
        "eos": source_special["tokenizer.ggml.eos_token_id"],
        "eot": metadata["vocab"]["<|im_end|>"], "sep": -1,
        "nl": metadata["vocab"]["Ċ"], "pad": source_special["tokenizer.ggml.padding_token_id"], "mask": -1,
        **{key: metadata["vocab"][token] for key, token in (
            ("fim_pre", "<|fim_prefix|>"), ("fim_suf", "<|fim_suffix|>"), ("fim_mid", "<|fim_middle|>"),
            ("fim_pad", "<|fim_pad|>"), ("fim_rep", "<|repo_name|>"), ("fim_sep", "<|file_sep|>"),
        )},
    }
    special_mismatch = special != expected_special
    return {
        "attempt": "20261112-EDGE0-03B.1" if unicode_asset else "20261112-EDGE0-03",
        "candidate_revision": UNICODE_CANDIDATE_REVISION if unicode_asset else CANDIDATE_REVISION,
        "candidate_sha256": _sha(_json_bytes(program)), "program_sha256": _sha(Path(__file__).read_bytes()),
        "source": {"repo_id": EDGE0_REPO_ID, "revision": EDGE0_REVISION},
        "corpus": corpus, "mismatches": mismatches, "special_ids_match": not special_mismatch,
        "special_ids": expected_special, "encode_mismatch_fixture_indices": encode_mismatch_cases,
        "decode_mode_counts": decode_mode_counts,
        "negative_controls": negatives, "eog_ids": sorted(eog_expected),
        "worker_utf8_policy_verified": worker_utf8_policy_verified,
        "pretokenizer_boundaries_validated": pretokenizer_reference is not None,
        "pretokenizer_boundary_corpus": boundary_corpus_identity(metadata) if pretokenizer_reference else None,
        "unicode_membership_proof": unicode_membership_proof,
        "upstream_frontend_text_policy_verified": False,
        "passed": (not any(mismatches.values()) and not special_mismatch
                   and worker_utf8_policy_verified and all(v["detected"] for v in negatives.values())
                   and (not unicode_asset or pretokenizer_reference is not None)),
        "native_runtime_validated": False,
    }
