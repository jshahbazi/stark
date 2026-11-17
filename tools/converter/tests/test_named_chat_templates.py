"""Hugging Face named chat-template lists render their `default` entry.

Model-free fixtures. Tool calling stays unavailable for such artifacts until a
reviewed parser admits them; plain string templates are unchanged.
"""

import hashlib
import json
from pathlib import Path

import pytest
from test_auto_dtype import FUTURE, source
from test_metadata import metadata_source  # noqa: F401

from stark_convert import cli
from stark_convert import model_catalog as catalog
from stark_convert.api_tools import UNREVIEWED_TOOL_TEMPLATES, codec_with_template_review, tool_parser_for
from stark_convert.chat import load_template
from stark_convert.metadata import UnsupportedMetadata, write_metadata
from stark_convert.model_inventory import model_tool_support, template_review_receipt
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError
from stark_convert.template_compat import NamedTemplateError, normalize_chat_template, select_named_chat_template

REVIEWED = normalize_chat_template((Path(__file__).parent / "fixtures/tool_templates/mistral_v03.jinja").read_text())
TOOL_USE = "{{ bos_token }}{% for message in messages %}[TOOL]{{ message.content }}{% endfor %}"


def named(default=REVIEWED, tool_use=TOOL_USE):
    return [{"name": "default", "template": default}, {"name": "tool_use", "template": tool_use}]


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_reviewed_fixture_would_otherwise_offer_tools():
    assert tool_parser_for({"prompt_template": REVIEWED}) == "mistral-json-v1"


def test_selection_uses_default_and_records_hashes_only():
    template, record = select_named_chat_template(named())
    assert template == REVIEWED
    assert record == {
        "form": "named_list",
        "selected": "default",
        "names": ["default", "tool_use"],
        "templates_sha256": {"default": sha256(REVIEWED), "tool_use": sha256(TOOL_USE)},
        "tool_templates": "unreviewed",
    }
    assert REVIEWED not in json.dumps(record) and TOOL_USE not in json.dumps(record)


@pytest.mark.parametrize(
    "value,reason",
    [
        ([{"name": "tool_use", "template": "x"}], "named_template_default_missing"),
        ([{"name": "default", "template": "x"}, {"name": "default", "template": "y"}], "named_template_duplicate"),
        ([{"name": "default", "template": 1}], "named_template_shape"),
        ([{"name": 1, "template": "x"}], "named_template_shape"),
        ([{"name": "default", "template": "x", "extra": "y"}], "named_template_shape"),
        ([{"name": "", "template": "x"}], "named_template_shape"),
        (["default"], "named_template_shape"),
        ([], "named_template_shape"),
        ({"default": "x"}, "named_template_shape"),
        ([{"name": f"t{index}", "template": "x"} for index in range(65)], "named_template_shape"),
    ],
)
def test_selection_fails_closed(value, reason):
    with pytest.raises(NamedTemplateError, match=reason):
        select_named_chat_template(value)


def rewrite_template(path, value):
    config = json.loads((path / "tokenizer_config.json").read_text())
    config["chat_template"] = value
    (path / "tokenizer_config.json").write_text(json.dumps(config))


def test_write_metadata_renders_default_and_reports_provenance(metadata_source, tmp_path):  # noqa: F811
    path, config = metadata_source
    rewrite_template(path, named())
    provenance = {}
    fragments = write_metadata(path, tmp_path / "named", config, provenance=provenance)
    assert fragments["token_codec_config"]["prompt_template"] == REVIEWED
    assert provenance["chat_template_selection"]["selected"] == "default"
    assert "chat_template_selection" not in json.dumps(fragments)
    rewrite_template(path, REVIEWED)
    plain = {}
    assert write_metadata(path, tmp_path / "plain", config, provenance=plain) == fragments
    assert plain == {}


@pytest.mark.parametrize(
    "value,error",
    [
        ([{"name": "tool_use", "template": "x"}], UnsupportedMetadata),
        ([{"name": "default", "template": "x"}, {"name": "default", "template": "x"}], UnsupportedMetadata),
        ([{"name": "default", "template": "x", "extra": 1}], UnsupportedMetadata),
        ({"default": "x"}, UnsupportedMetadata),
        ([{"name": "default", "template": 1}], ConversionError),
        ([1], ConversionError),
        (1, ConversionError),
    ],
)
def test_write_metadata_refuses_other_forms(metadata_source, tmp_path, value, error):  # noqa: F811
    path, config = metadata_source
    rewrite_template(path, value)
    with pytest.raises(error):
        write_metadata(path, tmp_path / "output", config)
    assert not (tmp_path / "output").exists()


def convert(tmp_path, path, name):
    snapshot = resolve_source(path, repo=FUTURE, metadata_profile="hf-generic")
    return cli.convert(snapshot, tmp_path / name, repo=FUTURE, chunk_bytes=4096, dtype="auto")


def test_named_list_converts_like_its_default_string_and_keeps_tools_unavailable(tmp_path):
    path = source(tmp_path, "mistral-f16")
    rewrite_template(path, named())
    receipt = convert(tmp_path, path, "named")
    selection = receipt["chat_template_selection"]
    assert selection["selected"] == "default" and selection["names"] == ["default", "tool_use"]
    assert receipt["dtype_selection"]["selected"] == "float16"
    named_config = json.loads((tmp_path / "named/config.json").read_text())
    assert named_config["token_codec_config"]["prompt_template"] == REVIEWED
    assert json.loads((tmp_path / "named/conversion.json").read_text())["chat_template_selection"] == selection

    rewrite_template(path, REVIEWED)
    plain = convert(tmp_path, path, "plain")
    assert "chat_template_selection" not in plain
    # The artifact is byte-identical to the plain `default` string conversion.
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        assert (tmp_path / "named" / name).read_bytes() == (tmp_path / "plain" / name).read_bytes()

    assert model_tool_support(tmp_path / "plain")["status"] == "available"
    support = model_tool_support(tmp_path / "named")
    assert support["status"] == "unavailable" and support["parser"] is None
    assert support["template_sha256"] == sha256(REVIEWED)
    codec, _template = load_template(tmp_path / "named")
    assert codec[UNREVIEWED_TOOL_TEMPLATES] is True and tool_parser_for(codec) is None
    codec, _template = load_template(tmp_path / "plain")
    assert UNREVIEWED_TOOL_TEMPLATES not in codec and tool_parser_for(codec) == "mistral-json-v1"


def test_unreadable_receipt_fails_closed_and_missing_receipt_adds_nothing(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    assert template_review_receipt(root) == {}
    (root / "conversion.json").write_text("not json")
    assert template_review_receipt(root) == {"chat_template_selection": "unreadable_receipt"}
    codec = {"type": "ChatCodecConfig", "prompt_template": REVIEWED}
    assert tool_parser_for(codec_with_template_review(codec, template_review_receipt(root))) is None
    assert codec_with_template_review(codec, {}) is codec


def catalog_template(value, names=("tokenizer_config.json",)):
    payload = json.dumps({"chat_template": value}).encode()
    return catalog._source_chat_template(None, "example/model", list(names), lambda _name, _limit: payload)


def test_catalogue_offers_named_default_without_tools():
    template, blocker, from_named_list = catalog_template(named())
    assert (template, blocker, from_named_list) == (REVIEWED, None, True)
    assert catalog_template(REVIEWED) == (REVIEWED, None, False)
    assert catalog_template([{"name": "tool_use", "template": "x"}])[:2] == (None, "chat_template_invalid")
    assert catalog_template({"default": "x"})[:2] == (None, "chat_template_missing")
    config = {"model_type": "mistral", "num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64}
    sizes = {"model.safetensors": 100}
    planning = catalog._planning_metadata(
        "mistral", config, {}, ["model.safetensors"], sizes, chat_template=REVIEWED, chat_template_named=True
    )
    assert planning["chat_template_available"] is True
    assert planning["supports_tools"] is False and planning["tool_parser"] is None
    plain = catalog._planning_metadata("mistral", config, {}, ["model.safetensors"], sizes, chat_template=REVIEWED)
    assert plain["supports_tools"] is True and plain["tool_parser"] == "mistral-json-v1"
