"""Representational tests; independent native parity is an explicit local run."""

import copy
import hashlib
import json

import pytest
from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split

from stark_convert import edge0_native_tokenizer as native


@pytest.fixture
def metadata():
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    extra = {byte: 256 + index for index, byte in enumerate(b for b in range(256) if b not in visible)}
    tokens = [chr(byte if byte in visible else extra[byte]) for byte in range(256)]
    tokens += ["ab", "<|im_end|>", "<tool_call>", "[PAD259]"]
    vocab = {token: i for i, token in enumerate(tokens)}
    return {"tokens": tokens, "vocab": vocab, "types": [1] * 257 + [3, 4, 5], "merges": ["a b"],
            "descriptor": {"model": "gpt2", "pre": "qwen35", "add_bos_token": False,
                           "vocab_size": len(tokens), "n_merges": 1,
                           "pre_pattern": r"(?:'[sS])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|\s+",
                           "special_ids": {f"tokenizer.ggml.{k}_token_id": 257 for k in ("bos", "eos", "padding")}}}


def test_classes_decode_and_unused_is_not_literal_special(metadata):
    codec = Tokenizer.from_str(json.dumps(native.candidate_program(metadata)))
    assert codec.encode("ab", add_special_tokens=False).ids == [256]
    assert codec.encode("<|im_end|>", add_special_tokens=False).ids == [257]
    assert codec.encode("<tool_call>", add_special_tokens=False).ids == [258]
    assert codec.encode("[PAD259]", add_special_tokens=False).ids != [259]
    assert codec.decode([257, 258, 259], skip_special_tokens=False) == "<|im_end|><tool_call>"
    assert codec.decode([257, 258, 259], skip_special_tokens=True) == "<tool_call>"
    assert codec.decode([259], skip_special_tokens=False) == ""
    assert codec.decode([259], skip_special_tokens=True) == ""


def test_raw_piece_boundary_keeps_bytes_distinct_from_text(metadata):
    codec = Tokenizer.from_str(json.dumps(native.candidate_program(metadata)))
    assert native.raw_piece(metadata, 0, render_special=True) == b"\x00"
    assert native.raw_piece(metadata, 255, render_special=True) == b"\xff"
    assert codec.decode([255], skip_special_tokens=False) == "\ufffd"
    assert codec.decode([195, 169], skip_special_tokens=False) == "é"
    assert native.raw_piece(metadata, 257, render_special=False) == b""
    assert native.raw_piece(metadata, 258, render_special=False) == b"<tool_call>"
    assert native.raw_piece(metadata, 259, render_special=True) == b""


def test_no_nfc_is_inserted(metadata):
    program = native.candidate_program(metadata)
    assert program["normalizer"] is None
    codec = Tokenizer.from_str(json.dumps(program))
    altered = copy.deepcopy(program)
    altered["normalizer"] = {"type": "NFC"}
    changed = Tokenizer.from_str(json.dumps(altered))
    assert codec.encode("e\u0301").ids != changed.encode("e\u0301").ids


def test_non_ascii_added_tokens_need_new_policy(metadata):
    metadata["tokens"][258] = "é"
    with pytest.raises(ValueError, match="separate representation"):
        native.candidate_program(metadata)


def test_corpus_is_deterministic_and_source_bound(metadata):
    first = native.corpus_identity(metadata)
    assert first == native.corpus_identity(metadata)
    assert first["cases"]["decode"] == len(metadata["tokens"]) + 7
    metadata["tokens"][259] = "[PAD_NEW]"
    assert native.corpus_identity(metadata)["sha256"] != first["sha256"]


def test_boundary_corpus_includes_separately_registered_witness(metadata):
    cases = native.corpus_identity(metadata)["cases"]["encode"]
    assert native.boundary_corpus_identity(metadata)["cases"] == cases + 1
    assert native.boundary_corpus_identity(metadata)["additional_witness_sha256"] == (
        "826e174be5effb10384a99e534fbde86afab0af48f2e242f7cb9b6261cddc9c4"
    )


def test_sealed_inputs_are_checked_before_candidate(tmp_path, monkeypatch, metadata):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    data = {"tokenizer.json": json.dumps(metadata["descriptor"]).encode(),
            "vocab.json": json.dumps(metadata["vocab"]).encode(),
            "token_type.json": json.dumps(metadata["types"]).encode(), "merges.txt": b"a b\n"}
    identities = {}
    for name, value in data.items():
        (directory / name).write_bytes(value)
        identities[f"tokenizer/{name}"] = {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    monkeypatch.setattr(native, "EDGE0_FILES", identities)
    gguf = {f"tokenizer.ggml.{k}": metadata[v] for k, v in (
        ("tokens", "tokens"), ("token_type", "types"), ("merges", "merges"))}
    gguf.update(metadata["descriptor"]["special_ids"])
    assert native.sealed_metadata(tmp_path, gguf) == metadata
    gguf["tokenizer.ggml.token_type"] = [1] * len(metadata["tokens"])
    with pytest.raises(ValueError, match="GGUF arrays disagree"):
        native.sealed_metadata(tmp_path, gguf)
    (directory / "vocab.json").write_text("{}")
    with pytest.raises(ValueError, match="source identity mismatch"):
        native.sealed_metadata(tmp_path, gguf)


def test_comparison_requires_preregistered_corpus(metadata):
    with pytest.raises(ValueError, match="Corpus registration"):
        native.compare_candidate(metadata, None, registered_corpus_sha256="unregistered")


@pytest.fixture
def small_unicode():
    return {"properties": {"L": [[65, 90], [97, 122]], "M": [[0x301, 0x301]],
                           "N": [[48, 57]], "s": [[9, 13], [32, 32]]}}


@pytest.mark.parametrize("pattern,text,expected", [
    (r"[\p{L}\p{M}]+", "a\u03011!", ["a\u0301"]),
    (r"[^\s\p{L}\p{M}\p{N}]+", "a\u0301 1!", ["!"]),
    (r"\p{N}+", "a123!", ["123"]),
    (r"[\S]+", " a1!\t", ["a1!"]),
    (r"\s+(?!\S)", "   a", ["  "]),
    (r"\[\p{L}\]", "[a][1]", ["[a]"]),
])
def test_property_expansion_respects_scope_and_complement(small_unicode, pattern, text, expected):
    expanded = native.expanded_pattern(pattern, small_unicode)
    assert not any(value in expanded for value in (r"\p{L}", r"\p{M}", r"\p{N}", r"\s", r"\S"))
    splitter = Split(Regex(expanded), "removed", invert=True)
    assert [piece for piece, _ in splitter.pre_tokenize_str(text)] == expected


def test_escaped_backslash_is_not_a_property(small_unicode):
    assert native.expanded_pattern(r"\\p{L}", small_unicode) == r"\\p{L}"


@pytest.mark.parametrize("pattern", [r"\p{Z}", r"[\p{L}", r"[[\p{L}]]", r"\w"])
def test_unhandled_regex_semantics_are_rejected(small_unicode, pattern):
    with pytest.raises(ValueError):
        native.expanded_pattern(pattern, small_unicode)


def test_sixth_control_restores_original_properties(metadata, small_unicode):
    program = native.candidate_program(metadata, unicode_asset=small_unicode)
    mutants = dict(native.negative_programs(program, metadata=metadata))
    assert set(mutants) == {"nfc", "unicode_casefold_regex", "control_types", "vocab_ids", "merge_order",
                            "implicit_unicode_properties"}
    pattern = mutants["implicit_unicode_properties"]["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    assert pattern == metadata["descriptor"]["pre_pattern"]


def test_unicode_candidate_requires_exhaustive_bound_proof(metadata, small_unicode):
    with pytest.raises(ValueError, match="exhaustive source-bound membership proof"):
        native.compare_candidate(metadata, None, registered_corpus_sha256=native.corpus_identity(metadata)["sha256"],
                                 unicode_asset=small_unicode)


def test_casefold_control_needs_boundary_check_even_when_ids_match(metadata):
    prefix = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    metadata["descriptor"]["pre_pattern"] = prefix + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|\s+"
    program = native.candidate_program(metadata)
    mutant = dict(native.negative_programs(program))["unicode_casefold_regex"]
    witness = "'\u017fa"
    baseline_codec = Tokenizer.from_str(json.dumps(program))
    changed_codec = Tokenizer.from_str(json.dumps(mutant))
    assert baseline_codec.encode(witness).ids == changed_codec.encode(witness).ids
    pattern = program["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    changed_pattern = mutant["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    assert len(Split(Regex(pattern), "isolated").pre_tokenize_str(witness)) == 1
    assert len(Split(Regex(changed_pattern), "isolated").pre_tokenize_str(witness)) == 2
