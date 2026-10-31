"""Structural support reports and metadata-only rejection before weight fetches."""

import hashlib
import json

import pytest
from test_sources import PIN, PRIMARY, Hub, files
from test_vision_family_geometry import geometry

from stark_convert import cli, model_support, registry, sources
from stark_convert.model_support import preflight_model, preflight_source, support_matrix
from stark_convert.streaming import ConversionError, UnsupportedConversion

FUTURE = "example/future-model"


def test_matrix_is_shared_bounded_mutation_safe_and_separate_from_qualification():
    matrix = support_matrix()
    assert matrix["schema_version"] == 1 and matrix["modes"] == list(model_support.MODES)
    assert len(matrix["families"]) == 14
    assert len(json.dumps(matrix).encode()) < 65536
    assert matrix["qualification"] == {
        "status": "not_assessed",
        "scope": "checkpoint_and_build",
        "calibration": "not_assessed",
    }
    assert registry.CONFIG_ADAPTERS is model_support.CONFIG_ADAPTERS
    assert registry.architecture_for_config({"model_type": "qwen3_5_text"}) == "qwen3_5"
    matrix["families"][0]["modes"]["text"]["status"] = "qualified"
    assert support_matrix()["families"][0]["modes"]["text"]["status"] == "conditional"


@pytest.mark.parametrize("architecture", list(model_support.CONFIG_ADAPTERS.values()))
@pytest.mark.parametrize("mode", ["text", "decision"])
def test_each_family_has_explicit_app_modes_without_checkpoint_qualification(architecture, mode):
    report = preflight_model({"model_type": architecture}, mode=mode)
    assert report["status"] == ("unsupported" if architecture == "modernbert" else "conditional")
    assert report["qualification"]["status"] == "not_assessed"


def test_classifier_conversion_remains_distinct_from_app_modes():
    config = {"model_type": "modernbert"}
    assert preflight_model(config, model_kind="classifier")["status"] == "conditional"
    for mode in model_support.MODES:
        assert preflight_model(config, mode=mode)["status"] == "unsupported"


@pytest.mark.parametrize("source_dtype", ["float16", "float32", "bfloat16"])
def test_floating_source_precision_is_not_confused_with_native_output_dtype(source_dtype):
    config = {"model_type": "qwen3", "torch_dtype": source_dtype}
    assert preflight_model(config, mode="text", dtype="bfloat16")["status"] == "conditional"
    assert preflight_model(config, dtype="float32")["status"] == "conditional"
    assert preflight_model(config, mode="text", dtype="float32")["reasons"] == ["unsupported_native_dtype"]


@pytest.mark.parametrize("bits", [4, 8])
def test_mlx_text_representation_is_implemented_but_unqualified(bits):
    config = {"model_type": "qwen3", "quantization": {"bits": bits, "group_size": 64}}
    report = preflight_model(config, mode="text")
    assert report["status"] == "conditional" and report["qualification"]["status"] == "not_assessed"


@pytest.mark.parametrize(
    "config,reason",
    [
        ({"model_type": "qwen3_moe"}, "unsupported_architecture"),
        ({"model_type": "qwen3", "num_experts": 8}, "unsupported_moe"),
        ({"model_type": "qwen3", "auto_map": {"AutoModel": "private.Custom"}}, "custom_model_code"),
        (
            {"model_type": "llama", "quantization_config": {"quant_method": "gptq", "bits": 4}},
            "unsupported_weight_format",
        ),
        ({"model_type": "qwen2", "quantization": {"bits": 4, "group_size": 64}}, "unsupported_weight_format"),
    ],
)
def test_explicit_unsupported_metadata_has_a_fixed_reason(config, reason):
    assert preflight_model(config, mode="text")["reasons"] == [reason]


def test_known_phi_annotation_is_validated_without_executing_custom_code():
    config = {
        "model_type": "phi3",
        "auto_map": {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
            "AutoTokenizer": "Xenova/gpt-4o",
        },
    }
    assert preflight_model(config, mode="text")["status"] == "conditional"
    config["auto_map"]["AutoConfig"] = "other.CustomConfig"
    assert preflight_model(config, mode="text")["status"] == "unsupported"


def test_unknown_metadata_is_inconclusive_in_every_mode():
    for config in (None, {}):
        report = preflight_model(config, mode="vision-language")
        assert report["status"] == "inconclusive"
        assert set(report["modes"].values()) == {"inconclusive"}


def test_qwen_vision_requires_actual_supported_wrapper_and_geometry():
    config = geometry()
    assert preflight_model(config, mode="vision-language")["status"] == "conditional"
    config["vision_config"]["out_hidden_size"] += 1
    assert preflight_model(config, mode="vision-language")["reasons"] == ["unsupported_vision_geometry"]
    assert preflight_model({"model_type": "qwen3"}, mode="vision-language")["alternatives"] == ["text", "decision"]
    text_only = preflight_model({"model_type": "qwen3_5_text"}, mode="vision-language")
    assert text_only["reasons"] == ["vision_tower_missing"]


class ObservedHub(Hub):
    def get_hf_file_metadata(self, url):
        self.events.append(("metadata", url[0], url[1]))
        return super().get_hf_file_metadata(url)


def test_remote_preflight_pins_metadata_and_never_reads_or_heads_weights(tmp_path):
    root = files(tmp_path / "source")
    hub = ObservedHub({FUTURE: root})
    report = preflight_source(FUTURE, mode="text", hub=hub)
    assert report["status"] == "conditional"
    assert report["source"]["repo"] == FUTURE and report["source"]["revision"] == PIN
    assert report["source"]["metadata_sha256"] == {
        "config.json": hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    }
    assert all(event[2] == "config.json" for event in hub.events if event[0] in {"download", "metadata"})
    assert str(root) not in json.dumps(report)


def test_remote_oversized_config_is_rejected_before_the_downloader(tmp_path):
    root = files(tmp_path / "source")
    with (root / "config.json").open("wb") as stream:
        stream.truncate(1024 * 1024 + 1)
    hub = ObservedHub({FUTURE: root})
    report = preflight_source(FUTURE, hub=hub)
    assert report["status"] == "inconclusive" and report["reasons"] == ["metadata_size_limit"]
    assert not [event for event in hub.events if event[0] == "download"]


def test_malformed_bounded_metadata_is_distinct_from_oversized_metadata(tmp_path):
    root = files(tmp_path / "source")
    (root / "config.json").write_text("private malformed content")
    hub = ObservedHub({FUTURE: root})
    report = preflight_source(FUTURE, hub=hub)
    assert report["status"] == "inconclusive" and report["reasons"] != ["metadata_size_limit"]
    assert [event for event in hub.events if event[0] == "download"]
    assert "private malformed content" not in json.dumps(report)


def test_unreachable_metadata_has_no_exception_content_or_false_unsupported_claim():
    class Unavailable:
        def HfApi(self):
            raise RuntimeError("private credential or exception details")

    report = preflight_source(FUTURE, hub=Unavailable())
    assert report["status"] == "inconclusive"
    assert "private" not in json.dumps(report)


def test_offline_local_metadata_works_without_a_hub_and_remote_is_inconclusive(tmp_path):
    local = files(tmp_path / "local")
    assert preflight_source(local, offline=True, hub=object())["status"] == "conditional"
    report = preflight_source(FUTURE, offline=True, hub=object())
    assert report["reasons"] == ["offline_metadata_unavailable"]


@pytest.mark.parametrize(
    "config,error", [({"model_type": "unsupported_model"}, UnsupportedConversion), ({}, ConversionError)]
)
def test_resolver_rechecks_metadata_and_blocks_weights_for_unsupported_or_inconclusive(tmp_path, config, error):
    root = files(tmp_path / "source")
    (root / "config.json").write_text(json.dumps(config))
    hub = ObservedHub({PRIMARY: root})
    with pytest.raises(error):
        sources.resolve_source(PRIMARY, hub=hub)
    assert all(not event[2].endswith(".safetensors") for event in hub.events if event[0] == "download")


def test_resolver_checks_metadata_size_before_downloading_config(tmp_path):
    root = files(tmp_path / "source")
    with (root / "config.json").open("wb") as stream:
        stream.truncate(1024 * 1024 + 1)
    hub = ObservedHub({PRIMARY: root})
    with pytest.raises(ConversionError, match="size limit"):
        sources.resolve_source(PRIMARY, hub=hub)
    assert not [event for event in hub.events if event[0] == "download"]


@pytest.mark.parametrize("filename,maximum", [("preprocessor_config.json", 65536), ("chat_template.jinja", 262144)])
def test_remote_oversized_vision_sidecar_never_reaches_the_downloader(tmp_path, monkeypatch, filename, maximum):
    from stark_convert import vision_family

    root = files(tmp_path / "source")
    (root / "config.json").write_text(json.dumps(geometry()))
    (root / "preprocessor_config.json").write_text(json.dumps(vision_family._PROCESSOR))
    (root / "chat_template.jinja").write_text("synthetic reviewed template")
    monkeypatch.setattr(
        vision_family, "_TEMPLATE_SHA256", hashlib.sha256((root / "chat_template.jinja").read_bytes()).hexdigest()
    )
    with (root / filename).open("wb") as stream:
        stream.truncate(maximum + 1)
    hub = ObservedHub({FUTURE: root})
    report = preflight_source(FUTURE, mode="vision-language", hub=hub)
    assert report["reasons"] == ["image_metadata_size_limit"]
    assert filename not in [event[2] for event in hub.events if event[0] == "download"]
    assert not [event for event in hub.events if event[0] == "metadata" and event[2].endswith(".safetensors")]


def test_resolver_rechecks_small_vision_processing_before_any_weights(tmp_path):
    from stark_convert import vision_family

    root = files(tmp_path / "source")
    (root / "config.json").write_text(json.dumps(geometry()))
    (root / "preprocessor_config.json").write_text(json.dumps(vision_family._PROCESSOR))
    (root / "chat_template.jinja").write_text("unrecognized synthetic template")
    hub = ObservedHub({FUTURE: root})
    with pytest.raises(UnsupportedConversion, match="outside the implemented profile"):
        sources.resolve_source(FUTURE, artifact_mode="vision-language", hub=hub)
    assert all(not event[2].endswith(".safetensors") for event in hub.events if event[0] == "download")


@pytest.mark.parametrize("architecture", ["phi3", "gemma2", "gemma3_text"])
def test_known_untied_layout_exclusion_precedes_weights(tmp_path, architecture):
    root = files(tmp_path / "source", model_type=architecture)
    (root / "config.json").write_text(json.dumps({"model_type": architecture, "tie_word_embeddings": False}))
    hub = ObservedHub({FUTURE: root})
    with pytest.raises(UnsupportedConversion):
        sources.resolve_source(FUTURE, metadata_profile="hf-generic", hub=hub)
    assert all(not event[2].endswith(".safetensors") for event in hub.events if event[0] == "download")


def test_resolver_rejects_changed_explicit_immutable_revision_before_metadata(tmp_path):
    hub = ObservedHub({PRIMARY: files(tmp_path / "source")})
    with pytest.raises(ConversionError, match="different immutable revision"):
        sources.resolve_source(PRIMARY, revision="a" * 40, hub=hub)
    assert not [event for event in hub.events if event[0] == "download"]


def test_cli_family_listing_and_local_preflight_need_no_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stark-convert", "--list-families"])
    assert cli.main() == 0
    assert len(json.loads(capsys.readouterr().out)["families"]) == 14
    root = files(tmp_path / "source")
    monkeypatch.setattr(
        "sys.argv", ["stark-convert", str(root), "--preflight", "--offline", "--support-mode", "decision"]
    )
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "conditional" and report["mode"] == "decision"


def test_cli_preflight_vision_mode_does_not_require_an_artifact_output_flag(monkeypatch, capsys):
    expected = model_support._report(None, "vision-language", "inconclusive", "test", "Synthetic metadata check.")

    def inspect(_value, **kwargs):
        assert kwargs["mode"] == "vision-language"
        return expected

    monkeypatch.setattr(cli, "preflight_source", inspect)
    monkeypatch.setattr("sys.argv", ["stark-convert", FUTURE, "--preflight", "--support-mode", "vision-language"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "inconclusive"


def test_actual_conversion_mode_mismatch_stops_before_resolution(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "resolve_source", lambda *_args, **_kwargs: pytest.fail("resolved mismatched mode"))
    monkeypatch.setattr(
        "sys.argv",
        ["stark-convert", FUTURE, "--output", str(tmp_path / "output"), "--support-mode", "vision-language"],
    )
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().err)["error_category"] == "ValueError"


@pytest.mark.parametrize(
    "config,kwargs",
    [
        ({"model_type": "qwen3", "num_experts": 8}, {}),
        ({"model_type": "qwen3", "auto_map": {"AutoModel": "private.Custom"}}, {}),
        ({"model_type": "phi3", "tie_word_embeddings": False}, {}),
        ({"model_type": "qwen3", "quantization": {"quant_method": "gptq", "bits": 4}}, {}),
        ({"model_type": "qwen3"}, {"dtype": "float16"}),
        ({"model_type": "qwen3"}, {"container": ".gguf"}),
    ],
)
def test_source_wide_exclusions_do_not_offer_known_impossible_alternatives(config, kwargs):
    report = preflight_model(config, mode="text", **kwargs)
    assert report["status"] == "unsupported"
    assert set(report["modes"].values()) == {"unsupported"} and report["alternatives"] == []


def test_image_only_quantized_layout_preserves_valid_text_alternatives():
    config = geometry()
    config["quantization"] = {"bits": 4, "group_size": 64}
    report = preflight_model(config, mode="vision-language")
    assert report["reasons"] == ["unsupported_vision_weight_format"]
    assert report["alternatives"] == ["text", "decision"]


def test_missing_shared_inventory_does_not_offer_other_conditional_modes(tmp_path):
    root = files(tmp_path / "source")
    (root / "model.safetensors").unlink()
    report = preflight_source(root, mode="text", offline=True)
    assert report["status"] == "inconclusive"
    assert set(report["modes"].values()) == {"inconclusive"} and report["alternatives"] == []


def test_malformed_shared_tokenizer_metadata_leaves_no_conditional_alternatives(tmp_path, monkeypatch):
    from stark_convert import vision_family

    root = files(tmp_path / "source")
    (root / "config.json").write_text(json.dumps(geometry()))
    (root / "preprocessor_config.json").write_text(json.dumps(vision_family._PROCESSOR))
    (root / "chat_template.jinja").write_text("synthetic reviewed template")
    monkeypatch.setattr(
        vision_family, "_TEMPLATE_SHA256", hashlib.sha256((root / "chat_template.jinja").read_bytes()).hexdigest()
    )
    (root / "tokenizer_config.json").write_text("malformed synthetic metadata")
    report = preflight_source(root, mode="vision-language", offline=True)
    assert report["status"] == "inconclusive" and report["reasons"] == ["metadata_access_failed"]
    assert set(report["modes"].values()) == {"inconclusive"} and report["alternatives"] == []


@pytest.mark.parametrize(
    "config,kwargs",
    [
        ({"model_type": "qwen3", "num_experts": 8}, {}),
        ({"model_type": "qwen3", "auto_map": {"AutoModel": "private.Custom"}}, {}),
        ({"model_type": "phi3", "tie_word_embeddings": False}, {}),
        ({"model_type": "qwen3", "quantization": {"quant_method": "gptq", "bits": 4}}, {}),
        ({"model_type": "qwen3", "quantization_config": []}, {}),
        ({"model_type": "qwen3"}, {"dtype": "float16"}),
        ({"model_type": "qwen3"}, {"container": ".gguf"}),
    ],
)
def test_unimplemented_image_mode_does_not_mask_source_wide_exclusions(config, kwargs):
    report = preflight_model(config, mode="vision-language", **kwargs)
    assert report["status"] == "unsupported"
    assert report["reasons"] != ["unsupported_mode"]
    assert set(report["modes"].values()) == {"unsupported"} and report["alternatives"] == []


@pytest.mark.parametrize("mode", ["text", "vision-language"])
@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3", "text_config": []},
        {"model_type": "qwen3", "text_config": {}},
        {"model_type": "qwen3_5", "text_config": None},
        {"model_type": "qwen3_5", "text_config": []},
        {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_moe"}},
    ],
)
def test_malformed_or_unimplemented_nested_decoder_rules_out_all_modes(config, mode):
    report = preflight_model(config, mode=mode)
    assert report["status"] == "unsupported"
    assert set(report["modes"].values()) == {"unsupported"} and report["alternatives"] == []
