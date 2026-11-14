"""The shared config normalization pass ahead of every family planner."""

import copy
import importlib
import json
import math
from functools import partial

import pytest
from test_dense_adapters import dense_fixture
from test_gemma import gemma_fixture
from test_gemma4 import gemma4_fixture
from test_granite_smollm import fixture as granite_fixture
from test_lfm2 import lfm2_fixture
from test_olmo import olmo_fixture
from test_phi import phi_fixture
from test_qwen35 import qwen35_fixture
from test_streaming import qwen_fixture

from stark_convert import config_normalization
from stark_convert.checkpoint_metadata import CheckpointMetadata
from stark_convert.config_normalization import normalize_config
from stark_convert.planning import plan_from_headers, planner_for
from stark_convert.streaming import ConversionError, UnsupportedConversion, load_checkpoint

INERT = {
    "unsloth_fixed": True,
    "unsloth_version": "2024.9",
    "transformers.js_config": {"dtype": "q4"},
    "model_name": "example/base",
    "generation_config": {"temperature": 0.6, "top_p": 0.95},
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 20,
    "temperature": 0.6,
    "is_llama_config": True,
    "init_method": "mup",
}
MALFORMED = [
    ("unsloth_fixed", "yes"),
    ("unsloth_version", ""),
    ("transformers.js_config", "q4"),
    ("model_name", 7),
    ("generation_config", ["temperature"]),
    ("do_sample", 1),
    ("top_p", 1.5),
    ("top_p", math.nan),
    ("top_k", -1),
    ("top_k", True),
    ("temperature", -0.1),
    ("is_llama_config", None),
    ("init_method", {"name": "mup"}),
]


def base():
    return {"model_type": "llama", "hidden_size": 8, "rope_theta": 10000.0, "torch_dtype": "bfloat16"}


@pytest.mark.parametrize("key", sorted(INERT))
def test_inert_annotation_is_dropped(key):
    config, changes = normalize_config("llama", {**base(), key: INERT[key]})
    assert config == base()
    assert changes == (f"dropped {key}",)


@pytest.mark.parametrize("key,value", MALFORMED)
def test_malformed_annotation_is_refused(key, value):
    message = {"transformers.js_config": "browser-export", "unsloth_fixed": "training annotation"}.get(
        key, "malformed"
    )
    with pytest.raises(ConversionError, match=message):
        normalize_config("llama", {**base(), key: value})


def test_out_of_range_numbers_are_not_admitted():
    with pytest.raises(ConversionError):
        normalize_config("llama", {**base(), "temperature": 10**400})
    config = {**base(), "rope_parameters": {"rope_theta": 10**400}}
    del config["rope_theta"]
    assert normalize_config("smollm3", config) == (config, ())
    assert normalize_config("granite", {**base(), "attention_dropout": 10**400})[1] == ()


@pytest.mark.parametrize("family", ["qwen3", "llama", "lfm2"])
def test_planners_with_native_plain_rope_handling_judge_it_themselves(family):
    config = {"model_type": family, "rope_parameters": {"rope_theta": 10000.0}}
    assert normalize_config(family, config) == (config, ())


def test_unknown_keys_are_kept():
    config, changes = normalize_config("llama", {**base(), "stark_unknown_option": 1, "rope_interleaved_v2": False})
    assert config == {**base(), "stark_unknown_option": 1, "rope_interleaved_v2": False}
    assert changes == ()


def test_default_rope_flags_are_dropped_only_in_their_default_form():
    config, changes = normalize_config("llama", {**base(), "rope_interleaved": False, "rope_scaling": None})
    assert config == base()
    assert changes == ("dropped rope_interleaved=false", "dropped rope_scaling=null")
    kept = {**base(), "rope_interleaved": True, "rope_scaling": {"rope_type": "linear", "factor": 2.0}}
    assert normalize_config("llama", kept) == (kept, ())
    for value in (0, None, "false"):
        assert normalize_config("llama", {**base(), "rope_interleaved": value})[1] == ()


def test_dtype_is_renamed_only_without_a_competing_declaration():
    config = {**base(), "dtype": "bfloat16"}
    del config["torch_dtype"]
    assert normalize_config("llama", config) == (base(), ("renamed dtype to torch_dtype",))
    for conflict in ({**base(), "dtype": "float16"}, {**base(), "dtype": "bfloat16"}):
        assert normalize_config("llama", conflict) == (conflict, ())
    malformed = {"model_type": "llama", "dtype": ["bfloat16"]}
    assert normalize_config("llama", malformed) == (malformed, ())


def test_plain_rope_parameters_are_flattened():
    nested = {**base(), "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}}
    del nested["rope_theta"]
    assert normalize_config("smollm3", nested) == (
        {**base(), "rope_theta": 500000.0},
        ("flattened rope_parameters",),
    )
    theta_only = {**base(), "rope_parameters": {"rope_theta": 10000}}
    assert normalize_config("granite", theta_only) == ({**base(), "rope_theta": 10000}, ("flattened rope_parameters",))


@pytest.mark.parametrize(
    "parameters,extra",
    [
        # Two different declared bases, as published by some SmolLM3 exports.
        ({"rope_theta": 5000000.0, "rope_type": "default"}, {}),
        ({"rope_theta": 10000.0, "rope_type": "yarn", "factor": 2.0}, {}),
        ({"rope_theta": 10000.0, "rope_type": "linear"}, {}),
        ({"rope_type": "default"}, {}),
        ({"rope_theta": -1.0}, {}),
        ({"rope_theta": True}, {}),
        (
            {
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
            {},
        ),
        ({"rope_theta": 10000.0, "rope_type": "default"}, {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}),
        ({"rope_theta": 10000.0, "partial_rotary_factor": 0.5}, {}),
    ],
)
def test_other_rope_forms_are_left_for_the_planner(parameters, extra):
    config = {**base(), **extra, "rope_parameters": parameters}
    assert normalize_config("smollm3", config) == (config, ())


def test_rope_parameters_stay_where_the_planner_requires_them():
    config = {"model_type": "qwen3_5", "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}}
    assert normalize_config("qwen3_5", config) == (config, ())
    gemma3 = {"model_type": "gemma3_text", "rope_parameters": {"rope_theta": 10000.0}}
    assert normalize_config("gemma3_text", gemma3) == (gemma3, ())


def test_duplicate_gemma_sliding_window_size_is_dropped():
    config = {"model_type": "gemma2", "sliding_window": 4096, "sliding_window_size": 4096}
    assert normalize_config("gemma2", config) == (
        {"model_type": "gemma2", "sliding_window": 4096},
        ("dropped sliding_window_size",),
    )
    for kept in (
        {"model_type": "gemma2", "sliding_window": 4096, "sliding_window_size": 2048},
        {"model_type": "gemma2", "sliding_window_size": 4096},
        {"model_type": "gemma2", "sliding_window": True, "sliding_window_size": True},
        {"model_type": "gemma2", "sliding_window": 4096, "sliding_window_size": 4096.0},
    ):
        assert normalize_config("gemma2", kept) == (kept, ())
    other = {"model_type": "mistral", "sliding_window": 4096, "sliding_window_size": 4096}
    assert normalize_config("mistral", other) == (other, ())


def test_inactive_attention_dropout_is_zeroed():
    assert normalize_config("granite", {**base(), "attention_dropout": 0.1}) == (
        {**base(), "attention_dropout": 0.0},
        ("zeroed attention_dropout",),
    )
    for value in (0.0, 0, 1.5, -0.1, math.nan, math.inf, True, "0.1", None):
        config = {**base(), "attention_dropout": value}
        assert normalize_config("granite", config) == (config, ())


def test_attention_dropout_rule_can_be_removed_on_its_own(monkeypatch):
    monkeypatch.setattr(config_normalization, "ZERO_INACTIVE_ATTENTION_DROPOUT", False)
    config = {**base(), "attention_dropout": 0.1, "unsloth_fixed": True}
    assert normalize_config("granite", config) == ({**base(), "attention_dropout": 0.1}, ("dropped unsloth_fixed",))


def test_lfm2_aliases_follow_the_planner():
    config = {"model_type": "lfm2", "intermediate_size": 10752, "tie_word_embeddings": True}
    assert normalize_config("lfm2", config) == (
        {"model_type": "lfm2", "intermediate_size": 10752, "block_ff_dim": 10752},
        ("dropped tie_word_embeddings=true", "set block_ff_dim from intermediate_size"),
    )
    for kept in (
        {"model_type": "lfm2", "tie_word_embeddings": False},
        {"model_type": "lfm2", "tie_word_embeddings": True, "tie_embedding": False},
        {"model_type": "lfm2", "intermediate_size": 10752, "block_ff_dim": 8192},
        {"model_type": "lfm2", "intermediate_size": "10752"},
        {"model_type": "lfm2", "intermediate_size": 0},
    ):
        assert normalize_config("lfm2", kept) == (kept, ())
    other = {"model_type": "llama", "intermediate_size": 12, "tie_word_embeddings": True}
    assert normalize_config("llama", other) == (other, ())


def test_text_config_is_normalized_only_for_planners_that_read_it():
    text = {"model_type": "qwen3_5_text", "dtype": "bfloat16", "unsloth_fixed": True, "attention_dropout": 0.1}
    config, changes = normalize_config("qwen3_5", {"model_type": "qwen3_5", "text_config": text, "top_k": 20})
    assert config == {
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "torch_dtype": "bfloat16", "attention_dropout": 0.0},
    }
    assert changes == (
        "dropped top_k",
        "dropped text_config.unsloth_fixed",
        "renamed text_config.dtype to torch_dtype",
        "zeroed text_config.attention_dropout",
    )
    llama = {"model_type": "llama", "text_config": dict(text)}
    assert normalize_config("llama", llama) == (llama, ())


@pytest.mark.parametrize("family", ["gemma3", "qwen4_exp", "llamba", "modernbert", "unknown"])
def test_families_without_reviewed_rules_are_unchanged(family):
    config = {"model_type": family, "unsloth_fixed": True, "dtype": "bfloat16", "rope_scaling": None}
    result, changes = normalize_config(family, config)
    assert result is config
    assert changes == ()


def test_input_is_never_mutated():
    text = {"dtype": "bfloat16", "unsloth_fixed": True, "rope_scaling": None}
    config = {
        "model_type": "qwen3_5",
        "text_config": text,
        "dtype": "bfloat16",
        "generation_config": {"top_k": 20},
        "attention_dropout": 0.1,
    }
    snapshot = copy.deepcopy(config)
    result, changes = normalize_config("qwen3_5", config)
    assert changes
    assert config == snapshot
    assert config["text_config"] is text
    assert result["text_config"] is not text
    lfm2 = {
        "model_type": "lfm2",
        "intermediate_size": 12,
        "tie_word_embeddings": True,
        "rope_parameters": {"rope_theta": 1.0},
    }
    snapshot = copy.deepcopy(lfm2)
    assert normalize_config("lfm2", lfm2)[1]
    assert lfm2 == snapshot


def test_normalization_is_idempotent():
    config = {**base(), "rope_parameters": {"rope_theta": 10000.0}, "unsloth_fixed": True, "attention_dropout": 0.2}
    once, changes = normalize_config("smollm3", config)
    assert "flattened rope_parameters" in changes
    assert normalize_config("smollm3", once) == (once, ())


def _plain_rope(config):
    theta = config.pop("rope_theta")
    config["rope_parameters"] = {"rope_theta": theta, "rope_type": "default"}


def _annotate(config):
    config.update(copy.deepcopy(INERT))
    config["rope_interleaved"] = False
    config.setdefault("rope_scaling", None)
    if "torch_dtype" in config:
        config["dtype"] = config.pop("torch_dtype")


def _dropout(config):
    config["attention_dropout"] = 0.1


def _gemma(config):
    _annotate(config)
    _dropout(config)
    config["sliding_window_size"] = config["sliding_window"]


def _gemma2(config):
    _gemma(config)
    _plain_rope(config)


def _gemma4(config):
    _annotate(config["text_config"])
    _dropout(config["text_config"])


def _lfm2(config):
    _annotate(config)
    _plain_rope(config)
    config["intermediate_size"] = config.pop("block_ff_dim")
    config["tie_word_embeddings"] = config.pop("tie_embedding", True)


def _dense(config):
    _annotate(config)
    _dropout(config)
    _plain_rope(config)


def _annotations_and_dropout(config):
    _annotate(config)
    _dropout(config)


def _olmo(config):
    # The fixture's YaRN scaling is a real variant, so only annotations differ.
    _annotate(config)
    _dropout(config)


FAMILIES = [
    ("qwen3", partial(qwen_fixture, tied=True), "stark_convert.qwen3", _dense),
    ("qwen2", partial(dense_fixture, family="qwen2", tied=True), "stark_convert.adapters.dense", _dense),
    ("llama", partial(dense_fixture, family="llama", tied=True), "stark_convert.adapters.dense", _dense),
    ("mistral", partial(dense_fixture, family="mistral", tied=True), "stark_convert.adapters.dense", _dense),
    ("gemma2", partial(gemma_fixture, family="gemma2"), "stark_convert.adapters.gemma", _gemma2),
    ("gemma3_text", partial(gemma_fixture, family="gemma3_text"), "stark_convert.adapters.gemma", _gemma),
    ("gemma4_text", gemma4_fixture, "stark_convert.adapters.gemma4", _gemma4),
    ("olmo3", olmo_fixture, "stark_convert.adapters.olmo", _olmo),
    ("phi3", partial(phi_fixture, duplicate_head=True), "stark_convert.adapters.phi", _annotations_and_dropout),
    ("granite", granite_fixture, "stark_convert.adapters.granite", _dense),
    ("smollm3", partial(granite_fixture, case="smollm3"), "stark_convert.adapters.granite", _dense),
    ("qwen3_5", partial(qwen35_fixture, tied=True), "stark_convert.adapters.qwen35", _annotations_and_dropout),
    ("lfm2", lfm2_fixture, "stark_convert.adapters.lfm2", _lfm2),
]


def _rewrite(source, mutate):
    config = json.loads((source / "config.json").read_text())
    mutate(config)
    (source / "config.json").write_text(json.dumps(config))
    return config


@pytest.mark.parametrize("family,fixture,module,mutate", FAMILIES, ids=[case[0] for case in FAMILIES])
def test_rejected_publisher_config_now_plans_in_admission_and_conversion(
    tmp_path, monkeypatch, family, fixture, module, mutate
):
    source = tmp_path / "source"
    fixture(source)
    reference = planner_for(family)(source)
    config = _rewrite(source, mutate)
    tensors = load_checkpoint(source)

    # Without the shared pass the planner rejects this published form.
    with monkeypatch.context() as patch:
        patch.setattr(importlib.import_module(module), "normalize_config", lambda family, config: (config, ()))
        with pytest.raises((ConversionError, UnsupportedConversion)):
            planner_for(family)(source)
        with pytest.raises((ConversionError, UnsupportedConversion)):
            plan_from_headers(family, CheckpointMetadata(copy.deepcopy(config), tensors))

    # Conversion reads config.json from the source; admission receives the raw
    # downloaded config. Both normalize identically and plan the same model.
    raw = copy.deepcopy(config)
    converted = planner_for(family)(source)
    admitted = plan_from_headers(family, CheckpointMetadata(raw, tensors))
    assert raw == config
    assert converted == reference
    assert admitted.decoder_config == converted.decoder_config
    assert [operation.name for operation in admitted.operations] == [
        operation.name for operation in converted.operations
    ]

    # A genuinely unknown key is still refused on both paths.
    level = config["text_config"] if family == "gemma4_text" else config
    level["stark_unknown_option"] = 1
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion):
        planner_for(family)(source)
    with pytest.raises(UnsupportedConversion):
        plan_from_headers(family, CheckpointMetadata(config, tensors))
