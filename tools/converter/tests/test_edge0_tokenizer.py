"""Small independent metadata fixtures exercise the pre-download tokenizer gate."""

import copy
import hashlib
import json

import pytest

from stark_convert import edge0_tokenizer as tokenizer


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    pattern = r"[\p{L}\p{M}]+"
    added = {"id": 4, "content": "<|end|>", "lstrip": False, "rstrip": False,
             "normalized": False, "single_word": False, "special": True}
    model_vocab = {"a": 0, "b": 1, "ab": 2, "ba": 3}
    hf = {
        "version": "1.0", "truncation": None, "padding": None, "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            byte_level,
        ]},
        "post_processor": byte_level, "decoder": byte_level, "added_tokens": [added],
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": model_vocab, "merges": [["a", "b"], ["b", "a"]]},
    }
    config = {
        "added_tokens_decoder": {"4": {k: v for k, v in added.items() if k != "id"}},
        "bos_token": "<|end|>", "eos_token": "<|end|>", "pad_token": "<|end|>",
        "pretokenize_regex": pattern, "add_prefix_space": False, "add_bos_token": False,
    }
    vocab = model_vocab | {"<|end|>": 4}
    descriptor = {"model": "gpt2", "pre": "qwen35", "vocab_size": 5, "n_merges": 2,
                  "special_ids": {f"tokenizer.ggml.{name}_token_id": 4 for name in ("bos", "eos", "padding")},
                  "add_bos_token": False, "pre_pattern": pattern}
    gguf = {
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": list(vocab), "tokenizer.ggml.token_type": [1, 1, 1, 1, 3],
        "tokenizer.ggml.merges": ["a b", "b a"], "tokenizer.ggml.add_bos_token": False,
        **descriptor["special_ids"],
    }
    source = {"tokenizer.json": hf, "tokenizer_config.json": config}
    monkeypatch.setattr(tokenizer, "_upstream_bytes", lambda name, **kwargs: json.dumps(source[name]).encode())
    path = tmp_path / "tokenizer"
    path.mkdir()
    (path / "tokenizer.json").write_text(json.dumps(descriptor))
    (path / "vocab.json").write_text(json.dumps(vocab))
    (path / "merges.txt").write_text("a b\nb a\n")
    (path / "token_type.json").write_text(json.dumps(gguf["tokenizer.ggml.token_type"]))
    return tmp_path, gguf, source


def test_exact_equivalence_preserves_pinned_source_bytes(metadata):
    directory, gguf, source = metadata
    report = tokenizer.preflight_tokenizer(directory, gguf)
    assert report["equivalent"] is True
    assert not (directory / "tokenizer.json").exists()
    result = tokenizer.prepare_tokenizer(directory, gguf)
    for name, value in source.items():
        expected = json.dumps(value).encode()
        assert (directory / name).read_bytes() == expected
        assert result["derived_metadata"][name]["sha256"] == hashlib.sha256(expected).hexdigest()
    assert result["tokenizer_equivalence"]["native_runtime_validated"] is False
    assert tokenizer.prepare_tokenizer(directory, gguf) == result


def test_control_vs_user_defined_is_rejected_without_output(metadata):
    directory, gguf, source = metadata
    source["tokenizer.json"]["added_tokens"][0]["special"] = False
    source["tokenizer_config.json"]["added_tokens_decoder"]["4"]["special"] = False
    with pytest.raises(tokenizer.TokenizerEquivalenceError) as exc:
        tokenizer.prepare_tokenizer(directory, gguf)
    assert "upstream_token_types_mismatch" in exc.value.report["failures"]
    assert exc.value.report["token_type_mismatch_id_ranges"] == [[4, 4]]
    assert not (directory / "tokenizer.json").exists()
    assert not (directory / "tokenizer_config.json").exists()


def test_unused_padding_does_not_silently_extend_upstream_vocabulary(metadata):
    directory, gguf, _ = metadata
    path = directory / "tokenizer" / "vocab.json"
    vocab = json.loads(path.read_text()) | {"[PAD5]": 5}
    path.write_text(json.dumps(vocab))
    path = directory / "tokenizer" / "tokenizer.json"
    descriptor = json.loads(path.read_text())
    descriptor["vocab_size"] = 6
    path.write_text(json.dumps(descriptor))
    gguf["tokenizer.ggml.tokens"].append("[PAD5]")
    gguf["tokenizer.ggml.token_type"].append(5)
    (directory / "tokenizer" / "token_type.json").write_text(json.dumps(gguf["tokenizer.ggml.token_type"]))
    report = tokenizer.preflight_tokenizer(directory, gguf)
    assert report["equivalent"] is False
    assert report["shipped_ids_absent_upstream_ranges"] == [[5, 5]]
    assert report["absent_upstream_all_gguf_unused"] is True
    assert "upstream_token_ids_mismatch" in report["failures"]


@pytest.mark.parametrize("mutation,category", [
    (lambda g, s: g["tokenizer.ggml.merges"].reverse(), "shipped_gguf_merge_order_mismatch"),
    (lambda g, s: s["tokenizer.json"]["model"]["merges"].reverse(), "upstream_merge_order_mismatch"),
    (lambda g, s: s["tokenizer.json"].update(normalizer={"type": "NFC"}), "upstream_normalizer_mismatch"),
    (lambda g, s: s["tokenizer_config.json"].update(bos_token=None), "upstream_special_ids_mismatch"),
    (lambda g, s: s["tokenizer.json"]["model"].update(byte_fallback=True), "unsupported_bpe_semantics"),
    (lambda g, s: s["tokenizer.json"]["pre_tokenizer"]["pretokenizers"][1].update(use_regex=True),
     "upstream_pretokenizer_mismatch"),
    (lambda g, s: g.update({"tokenizer.ggml.bos_token_id": 0}), "shipped_gguf_special_ids_mismatch"),
    (lambda g, s: g["tokenizer.ggml.tokens"].reverse(), "shipped_gguf_token_ids_mismatch"),
])
def test_discrepancies_block_preparation(metadata, mutation, category):
    directory, gguf, source = metadata
    # Avoid accidental aliasing of byte-level nodes in this synthetic fixture.
    source["tokenizer.json"] = copy.deepcopy(source["tokenizer.json"])
    mutation(gguf, source)
    with pytest.raises(tokenizer.TokenizerEquivalenceError) as exc:
        tokenizer.prepare_tokenizer(directory, gguf)
    assert category in exc.value.report["failures"]
    assert not (directory / "tokenizer.json").exists()


def test_unicode_contraction_expansion_is_semantically_different(metadata):
    directory, gguf, source = metadata
    tail = r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    upstream = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)" + tail
    shipped = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])" + tail
    source["tokenizer.json"]["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = upstream
    source["tokenizer_config.json"]["pretokenize_regex"] = upstream
    path = directory / "tokenizer" / "tokenizer.json"
    descriptor = json.loads(path.read_text())
    descriptor["pre_pattern"] = shipped
    path.write_text(json.dumps(descriptor))
    report = tokenizer.preflight_tokenizer(directory, gguf)
    witness = report["pretokenizer_witnesses"][0]
    assert witness["equivalent"] is False
    assert (witness["upstream_piece_count"], witness["shipped_piece_count"]) == (2, 1)
    assert "upstream_pretokenizer_mismatch" in report["failures"]


def test_duplicate_token_ids_are_rejected(metadata):
    directory, gguf, _ = metadata
    (directory / "tokenizer" / "vocab.json").write_text('{"a": 0, "b": 0}')
    with pytest.raises(ValueError, match="Duplicate tokenizer token ID"):
        tokenizer.prepare_tokenizer(directory, gguf)


def test_shipped_types_are_independently_compared(metadata):
    directory, gguf, _ = metadata
    (directory / "tokenizer" / "token_type.json").write_text("[1,1,1,1,4]")
    report = tokenizer.preflight_tokenizer(directory, gguf)
    assert "shipped_gguf_token_types_mismatch" in report["failures"]


def test_derived_identity_is_checked_on_retry(metadata):
    directory, gguf, _ = metadata
    tokenizer.prepare_tokenizer(directory, gguf)
    (directory / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="Derived tokenizer target identity mismatch"):
        tokenizer.prepare_tokenizer(directory, gguf)


def test_pinned_upstream_hash_is_enforced(tmp_path, monkeypatch):
    path = tmp_path / "upstream.json"
    size, _ = tokenizer.UPSTREAM_FILES["tokenizer.json"]
    path.write_bytes(b"x" * size)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(tokenizer, "hf_hub_download", download)
    with pytest.raises(ValueError, match="identity mismatch"):
        tokenizer._upstream_bytes("tokenizer.json", offline=True)
    assert calls == [{"repo_id": tokenizer.UPSTREAM_REPO, "filename": "tokenizer.json",
                      "revision": tokenizer.UPSTREAM_REVISION, "local_files_only": True}]
