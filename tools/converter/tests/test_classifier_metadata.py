"""Classifier metadata parity using synthetic tokens and template expressions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, models, processors

from stark_convert.classifier_metadata import write_classifier_metadata
from stark_convert.metadata import UnsupportedMetadata
from stark_convert.profiles import GENERIC_PROFILE
from stark_convert.streaming import ConversionError

REPO = "trymirai/chat-moderation-router"
VOCAB = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4, "synthetic": 5}
TEMPLATE = "{{ cls_token or bos_token or '' }}{% for message in messages %}{{ message.content }}{% endfor %}"


def entry(repo=REPO, profile="modernbert-classifier"):
    return {
        "repo": repo,
        "model_kind": "ClassifierModelSpec",
        "config_class": "ModernBERTConfig",
        "metadata_profile": profile,
        "files": {
            role: {"repo": repo, "filename": filename}
            for role, filename in {
                "model_config": "config.json",
                "tokenizer": "tokenizer.json",
                "tokenizer_config": "tokenizer_config.json",
                "chat_template": "chat_template.jinja",
            }.items()
        },
        "metadata": {
            "roles": {"system": "system", "user": "user", "assistant": "assistant"},
            "parser_id": None,
            "reasoning_id": None,
            "end_of_thinking_id": None,
            "generation_config": None,
            "generation_params_overrides": None,
            "inline_sources": {},
        },
    }


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.save(str(root / "tokenizer.json"))
    token_config = {
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "unk_token": "[UNK]",
    }
    (root / "tokenizer_config.json").write_text(json.dumps(token_config))
    (root / "chat_template.jinja").write_text(TEMPLATE)
    config = {
        "model_type": "modernbert",
        "vocab_size": len(VOCAB),
        "id2label": {"0": "label_a", "1": "label_b"},
        "label2id": {"label_a": 0, "label_b": 1},
        "num_labels": 2,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "cls_token_id": 1,
        "sep_token_id": 2,
        "pad_token_id": 3,
    }
    (root / "config.json").write_text(json.dumps(config))
    return root, {**config, "_stark_model_repo": REPO}


def update(root, **values):
    config = json.loads((root / "tokenizer_config.json").read_text())
    (root / "tokenizer_config.json").write_text(json.dumps({**config, **values}))


def convert(source, tmp_path, *, config=None, model_entry=None, **kwargs):
    root, original = source
    return write_classifier_metadata(
        root,
        tmp_path / "output",
        original if config is None else config,
        model_entry=entry() if model_entry is None else model_entry,
        **kwargs,
    )


def test_classifier_codec_has_no_generation_policy_and_preserves_post_processor(source, tmp_path):
    root, _ = source
    result = convert(source, tmp_path)
    assert result == {
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "prompt_template": TEMPLATE,
            "output_parser_regex": None,
            "system_role_name": "system",
            "user_role_name": "user",
            "assistant_role_name": "assistant",
            "end_of_thinking_tag": None,
            "reasoning_config": None,
            "eos_token": "[SEP]",
            "bos_token": "[CLS]",
            "default_system_prompt": None,
        }
    }
    assert (tmp_path / "output/tokenizer.json").read_bytes() == (root / "tokenizer.json").read_bytes()


@pytest.mark.parametrize("profile", ["modernbert-classifier", GENERIC_PROFILE])
def test_bos_eos_are_optional_and_cls_sep_do_not_become_generation_stops(source, tmp_path, profile):
    root, config = source
    config = {key: value for key, value in config.items() if key not in {"bos_token_id", "eos_token_id"}}
    codec = convert(source, tmp_path, config=config, model_entry=entry(profile=profile))["token_codec_config"]
    assert codec["bos_token"] is None and codec["eos_token"] is None
    assert (tmp_path / "output/tokenizer.json").read_bytes() == (root / "tokenizer.json").read_bytes()


def test_explicit_bos_eos_and_identifier_lists_match_pinned_first_id_fallback(source, tmp_path):
    root, config = source
    update(root, bos_token="[CLS]", eos_token="[SEP]")
    result = convert(source, tmp_path, config={**config, "bos_token_id": [1, 3], "eos_token_id": [2, 3]})
    assert result["token_codec_config"]["bos_token"] == "[CLS]"
    assert result["token_codec_config"]["eos_token"] == "[SEP]"


def test_generation_sidecar_is_not_read(source, tmp_path):
    root, _ = source
    model_entry = entry()
    model_entry["files"]["generation_config"] = {"repo": REPO, "filename": "generation_config.json"}
    (root / "generation_config.json").write_text("invalid unused metadata")
    assert set(convert(source, tmp_path, model_entry=model_entry)) == {"token_codec_config"}


def test_added_special_and_regular_tokens_keep_pinned_registration_order(source, tmp_path):
    root, config = source
    added = {
        str(index): {
            "content": text,
            "special": special,
            "single_word": False,
            "normalized": False,
            "lstrip": False,
            "rstrip": False,
        }
        for index, text, special in [(7, "[REGULAR]", False), (6, "[SPECIAL]", True)]
    }
    update(root, added_tokens_decoder=added)
    convert(source, tmp_path, config={**config, "vocab_size": 8})
    expected = Tokenizer.from_file(str(root / "tokenizer.json"))
    expected.add_special_tokens([AddedToken("[SPECIAL]", normalized=False, special=True)])
    expected.add_tokens([AddedToken("[REGULAR]", normalized=False, special=False)])
    expected.save(str(tmp_path / "expected.json"))
    assert (tmp_path / "output/tokenizer.json").read_bytes() == (tmp_path / "expected.json").read_bytes()


def test_generation_tags_removed_and_source_sidecar_kept(source, tmp_path):
    root, _ = source
    (root / "chat_template.jinja").write_text("{%- generation -%}{{ messages }}{% endgeneration %}")
    assert convert(source, tmp_path)["token_codec_config"]["prompt_template"] == "{{ messages }}"


def test_classifier_system_prompt_and_roles_match_selected_profile(source, tmp_path):
    root, _ = source
    model_entry = entry()
    model_entry["metadata"]["roles"] = {"system": "sys", "user": "usr", "assistant": "asst"}
    model_entry["files"]["system_prompt"] = {"repo": REPO, "filename": "system_prompt.txt"}
    (root / "system_prompt.txt").write_text("synthetic-default")
    codec = convert(source, tmp_path, model_entry=model_entry)["token_codec_config"]
    assert codec["default_system_prompt"] == "synthetic-default"
    assert codec["system_role_name"] == "sys" and codec["user_role_name"] == "usr"
    assert codec["assistant_role_name"] == "asst"


@pytest.mark.parametrize("matching", [False, True])
def test_generic_template_inline_and_sidecar_must_agree(source, tmp_path, matching):
    root, _ = source
    update(root, chat_template=TEMPLATE if matching else "{{ mismatch }}")
    if matching:
        assert (
            convert(source, tmp_path, model_entry=entry(profile=GENERIC_PROFILE))["token_codec_config"][
                "prompt_template"
            ]
            == TEMPLATE
        )
    else:
        with pytest.raises(ConversionError, match="Conflicting generic classifier chat-template"):
            convert(source, tmp_path, model_entry=entry(profile=GENERIC_PROFILE))
        assert not (tmp_path / "output").exists()


def test_generic_self_contained_inline_template_and_unregistered_identity(source, tmp_path):
    root, config = source
    update(root, chat_template=TEMPLATE)
    model_entry = entry(repo="future/classifier", profile=GENERIC_PROFILE)
    del model_entry["files"]["chat_template"]
    codec = convert(
        source, tmp_path, config={**config, "_stark_model_repo": "future/classifier"}, model_entry=model_entry
    )
    assert codec["token_codec_config"]["prompt_template"] == TEMPLATE


def test_generic_anonymous_local_identity_and_sources(source, tmp_path):
    _, config = source
    model_entry = entry(repo=None, profile=GENERIC_PROFILE)
    codec = convert(source, tmp_path, config={**config, "_stark_model_repo": None}, model_entry=model_entry)
    assert codec["token_codec_config"]["bos_token"] == "[CLS]"
    assert codec["token_codec_config"]["eos_token"] == "[SEP]"


@pytest.mark.parametrize(
    "changes",
    [
        {"model_kind": "LanguageModelSpec"},
        {"config_class": "HFLlamaConfig"},
        {"metadata_profile": "modernbert"},
        {"repo": "unknown/classifier"},
    ],
)
def test_unverified_profile_rejected_before_output(source, tmp_path, changes):
    with pytest.raises(UnsupportedMetadata, match="Unverified classifier metadata profile"):
        convert(source, tmp_path, model_entry={**entry(), **changes})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "llama"},
        {"vocab_size": True},
        {"vocab_size": 5},
        {"bos_token_id": True},
        {"eos_token_id": -1},
        {"pad_token_id": 6},
        {"cls_token_id": [1, 9]},
        {"id2label": {"1": "label_a", "2": "label_b"}},
        {"id2label": {"00": "label_a", "1": "label_b"}},
        {"id2label": {"0": "label_a", "1": "label_a"}},
        {"id2label": {"0": "", "1": "label_b"}},
        {"id2label": {"9" * 5000: "label_a"}},
        {"label2id": {"label_a": True, "label_b": 1}},
        {"label2id": {"label_a": 1, "label_b": 0}},
        {"label2id": {"label_a": 0}},
        {"num_labels": 3},
        {"num_labels": True},
    ],
)
def test_invalid_model_metadata_is_sanitized_before_output(source, tmp_path, changes):
    _, config = source
    with pytest.raises(ConversionError):
        convert(source, tmp_path, config={**config, **changes})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"bos_token": "synthetic"},
        {"eos_token": "[ABSENT]"},
        {"cls_token": "[SEP]"},
        {"sep_token": {"content": "[SEP]"}},
        {"mask_token": 4},
        {"add_bos_token": 1},
        {"add_eos_token": "true"},
        {"added_tokens_decoder": []},
        {"added_tokens_decoder": {"6": {"content": "[SPECIAL]"}}},
        {"chat_template": []},
        {"chat_template": [{"name": "default", "template": "{{ messages }}"}]},
    ],
)
def test_invalid_tokenizer_metadata_is_sanitized_before_output(source, tmp_path, changes):
    root, _ = source
    update(root, **changes)
    with pytest.raises((ConversionError, UnsupportedMetadata)):
        convert(source, tmp_path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("identifier", ["00", "-1", "true", "9" * 5000, "7"])
def test_added_token_identifier_mismatch_rejected(source, tmp_path, identifier):
    root, config = source
    token = {
        "content": "[ADDED]",
        "single_word": False,
        "normalized": False,
        "special": True,
        "lstrip": False,
        "rstrip": False,
    }
    update(root, added_tokens_decoder={identifier: token})
    with pytest.raises(ConversionError):
        convert(source, tmp_path, config={**config, "vocab_size": 8})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("key", ["parser_id", "reasoning_id", "end_of_thinking_id", "generation_config"])
def test_unsupported_classifier_metadata_profile_rejected(source, tmp_path, key):
    model_entry = entry()
    model_entry["metadata"][key] = "unknown"
    with pytest.raises(UnsupportedMetadata):
        convert(source, tmp_path, model_entry=model_entry)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("filename", ["../tokenizer.json", "/tokenizer.json", "", "x\\tokenizer.json"])
def test_invalid_selected_source_filename_rejected(source, tmp_path, filename):
    model_entry = entry()
    model_entry["files"]["tokenizer"]["filename"] = filename
    with pytest.raises(ConversionError, match="Invalid classifier metadata sources"):
        convert(source, tmp_path, model_entry=model_entry)
    assert not (tmp_path / "output").exists()


def test_selected_external_files_require_resolver_and_do_not_use_local_lookalikes(source, tmp_path):
    root, _ = source
    model_entry = deepcopy(entry())
    model_entry["files"]["tokenizer"]["repo"] = "selected/external"
    with pytest.raises(ConversionError, match="External metadata source requires a resolved snapshot"):
        convert(source, tmp_path, model_entry=model_entry)
    external = tmp_path / "external"
    external.mkdir()
    (external / "tokenizer.json").write_bytes((root / "tokenizer.json").read_bytes())
    (root / "tokenizer.json").write_text("invalid unselected local lookalike")
    accesses = []

    def resolver(filename, *, repo):
        accesses.append((repo, filename))
        return (external if repo == "selected/external" else root) / filename

    convert(source, tmp_path, model_entry=model_entry, resolve_file=resolver)
    assert ("selected/external", "tokenizer.json") in accesses
    assert (REPO, "tokenizer.json") not in accesses


def test_generic_external_sidecars_rejected_even_with_resolver(source, tmp_path):
    model_entry = entry(profile=GENERIC_PROFILE)
    model_entry["files"]["tokenizer"]["repo"] = "external/model"
    with pytest.raises(ConversionError, match="Generic classifier metadata must be self-contained"):
        convert(source, tmp_path, model_entry=model_entry, resolve_file=lambda filename, repo: Path("unused"))
    assert not (tmp_path / "output").exists()


def test_resolver_failure_has_no_source_content_in_message(source, tmp_path):
    def resolver(*args, **kwargs):
        raise ValueError("synthetic-private-marker")

    with pytest.raises(ConversionError, match="^Cannot resolve metadata source file\\.$") as error:
        convert(source, tmp_path, resolve_file=resolver)
    assert "synthetic-private-marker" not in str(error.value)
    assert error.value.__suppress_context__
    assert not (tmp_path / "output").exists()


def test_invalid_tokenizer_has_sanitized_error(source, tmp_path):
    root, _ = source
    (root / "tokenizer.json").write_text('{"synthetic-private-marker": "invalid"}')
    with pytest.raises(ConversionError, match="^Classifier tokenizer normalization failed\\.$") as error:
        convert(source, tmp_path)
    assert "synthetic-private-marker" not in str(error.value)
    assert error.value.__suppress_context__
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["TemplateProcessing", "BertProcessing", "RobertaProcessing", "Sequence"])
def test_post_processor_literal_identifiers_must_match_vocabulary(source, tmp_path, kind):
    root, _ = source
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if kind in {"TemplateProcessing", "Sequence"}:
        processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 9), ("[SEP]", 2)])
        if kind == "Sequence":
            processor = processors.Sequence([processors.ByteLevel(), processor])
    elif kind == "BertProcessing":
        processor = processors.BertProcessing(("[SEP]", 2), ("[CLS]", 9))
    else:
        processor = processors.RobertaProcessing(("[SEP]", 2), ("[CLS]", 9))
    tokenizer.post_processor = processor
    tokenizer.save(str(root / "tokenizer.json"))
    with pytest.raises(ConversionError, match="Classifier post-processor tokens disagree with tokenizer"):
        convert(source, tmp_path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["BertProcessing", "RobertaProcessing", "Sequence", "ByteLevel"])
def test_known_post_processors_preserved_when_identifiers_match(source, tmp_path, kind):
    root, _ = source
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    processor = processors.BertProcessing(("[SEP]", 2), ("[CLS]", 1))
    if kind == "RobertaProcessing":
        processor = processors.RobertaProcessing(("[SEP]", 2), ("[CLS]", 1))
    elif kind == "ByteLevel":
        processor = processors.ByteLevel()
    elif kind == "Sequence":
        processor = processors.Sequence([processors.ByteLevel(), processor])
    tokenizer.post_processor = processor
    tokenizer.save(str(root / "tokenizer.json"))
    convert(source, tmp_path)
    assert (tmp_path / "output/tokenizer.json").read_bytes() == (root / "tokenizer.json").read_bytes()


@pytest.mark.parametrize("identifier,text", [(9, "[PAD]"), (3, "[UNKNOWN]"), (2, "[PAD]")])
def test_padding_literal_identifier_must_match_vocabulary(source, tmp_path, identifier, text):
    root, _ = source
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_padding(pad_id=identifier, pad_token=text)
    tokenizer.save(str(root / "tokenizer.json"))
    with pytest.raises(ConversionError, match="Classifier padding tokens disagree with tokenizer"):
        convert(source, tmp_path)
    assert not (tmp_path / "output").exists()


def test_valid_padding_literal_is_preserved(source, tmp_path):
    root, _ = source
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    tokenizer.save(str(root / "tokenizer.json"))
    convert(source, tmp_path)
    assert (tmp_path / "output/tokenizer.json").read_bytes() == (root / "tokenizer.json").read_bytes()
