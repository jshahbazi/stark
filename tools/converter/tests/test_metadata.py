"""Metadata correctness and opt-in external reference artifact parity checks.

Fixtures contain synthetic vocabulary and template expressions only. No model
responses, evaluation data, or native validation artifacts are stored here.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer, models

from stark_convert.metadata import SUPPORTED_METADATA_REPOS, UnsupportedMetadata, write_metadata
from stark_convert.registry import get_model
from stark_convert.streaming import ConversionError


@pytest.fixture
def metadata_source(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    source = tmp_path / "source"
    source.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[ALT]": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer.save(str(source / "tokenizer.json"))
    added = {
        "content": "[ADDED]",
        "single_word": False,
        "normalized": False,
        "special": True,
        "lstrip": False,
        "rstrip": False,
    }
    tokenizer_config = {
        "chat_template": "{% generation %}{{ messages }}{% endgeneration %}",
        "added_tokens_decoder": {"4": added},
    }
    (source / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    generation = {
        "eos_token_id": [3, 2],
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.9,
        "min_p": 0.01,
        "repetition_penalty": 1.2,
    }
    (source / "generation_config.json").write_text(json.dumps(generation))
    config = {
        "eos_token_id": 2,
        "bos_token_id": 1,
        "torch_dtype": "bfloat16",
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 12,
        "max_position_embeddings": 64,
        "max_window_layers": 1,
        "model_type": "qwen3",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "vocab_size": 5,
        "head_dim": 4,
    }
    (source / "config.json").write_text(json.dumps(config))
    return source, {**config, "_stark_model_repo": "Qwen/Qwen3-0.6B"}


def _update_json(path: Path, changes: dict[str, Any]) -> None:
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))


def test_template_preserved_with_only_generation_tags_removed(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    template = "{% set x = 1 %}\n{%- generation -%}{{ messages }}{% endgeneration %}\n{{ x }}"
    _update_json(source / "tokenizer_config.json", {"chat_template": template})
    result = write_metadata(source, tmp_path / "output", config)
    assert result["token_codec_config"]["prompt_template"] == "{% set x = 1 %}\n{{ messages }}\n{{ x }}"
    assert result["token_codec_config"]["bos_token"] == "[BOS]"
    assert result["token_codec_config"]["eos_token"] == "[EOS]"
    tokenizer = Tokenizer.from_file(str(tmp_path / "output" / "tokenizer.json"))
    assert tokenizer.token_to_id("[ADDED]") == 4
    assert tokenizer.decode([4], skip_special_tokens=True) == ""


def test_stop_ids_merge_without_losing_sampling_defaults(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    result = write_metadata(source, tmp_path / "output", {**config, "eos_token_id": [3, 1, 3]})
    generation = result["generation_config"]
    assert generation["stop_token_ids"] == [1, 2, 3]
    assert generation["temperature"] == 0.6
    assert generation["top_k"] == 20
    (source / "generation_config.json").write_text("{}")
    defaults = write_metadata(source, tmp_path / "defaults", config)["generation_config"]
    assert defaults["stop_token_ids"] == [2]
    assert all(value is None for name, value in defaults.items() if name != "stop_token_ids")


@pytest.mark.parametrize("repo", [None, "unregistered/model", "Qwen/Qwen3-4B-MLX-4bit"])
def test_unverified_profile_leaves_no_output(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path, repo: str | None
) -> None:
    source, config = metadata_source
    output = tmp_path / "output"
    with pytest.raises(UnsupportedMetadata, match="Unverified model metadata profile"):
        write_metadata(source, output, {**config, "_stark_model_repo": repo})
    assert not output.exists()


@pytest.mark.parametrize(
    ("filename", "changes"),
    [
        ("tokenizer_config.json", {"chat_template": [{"name": "default", "template": "{{ messages }}"}]}),
        ("tokenizer_config.json", {"bos_token": {"content": "[BOS]"}}),
    ],
)
def test_unsupported_metadata_leaves_no_output(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path, filename: str, changes: dict[str, Any]
) -> None:
    source, config = metadata_source
    _update_json(source / filename, changes)
    output = tmp_path / "output"
    with pytest.raises(UnsupportedMetadata):
        write_metadata(source, output, config)
    assert not output.exists()


@pytest.mark.parametrize(
    ("filename", "changes"),
    [
        ("tokenizer_config.json", {"added_tokens_decoder": {"4": {"content": "[ADDED]"}}}),
        ("tokenizer_config.json", {"chat_template": None}),
        ("tokenizer_config.json", {"chat_template": []}),
        ("generation_config.json", {"top_k": "unsupported"}),
        ("generation_config.json", {"temperature": float("nan")}),
        ("generation_config.json", {"eos_token_id": [True]}),
    ],
)
def test_malformed_metadata_is_a_hard_error(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path, filename: str, changes: dict[str, Any]
) -> None:
    source, config = metadata_source
    _update_json(source / filename, changes)
    output = tmp_path / "output"
    with pytest.raises(ConversionError):
        write_metadata(source, output, config)
    assert not output.exists()


@pytest.mark.parametrize("contents", ["[]", '{"top_k": 1, "top_k": 2}', "not-json"])
def test_invalid_json_is_a_hard_error(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path, contents: str
) -> None:
    source, config = metadata_source
    (source / "generation_config.json").write_text(contents)
    output = tmp_path / "output"
    with pytest.raises(ConversionError, match="^Cannot read generation metadata\\.$"):
        write_metadata(source, output, config)
    assert not output.exists()


def test_invalid_tokenizer_error_is_sanitized(metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    source, config = metadata_source
    (source / "tokenizer.json").write_text('{"private-marker": "invalid-tokenizer"}')
    output = tmp_path / "output"
    with pytest.raises(ConversionError, match="^Tokenizer normalization failed\\.$"):
        write_metadata(source, output, config)
    assert not output.exists()


@pytest.mark.parametrize(
    ("repo", "model_type", "thinking", "reasoning"),
    [
        *[(f"Qwen/Qwen3-{size}", "qwen3", True, True) for size in ("0.6B", "1.7B", "4B", "8B", "14B", "32B")],
        ("Qwen/Qwen3-4B-Instruct-2507", "qwen3", False, False),
        ("Qwen/Qwen3-4B-Thinking-2507", "qwen3", True, False),
        *[
            (f"Qwen/Qwen2.5-Coder-{size}-Instruct", "qwen2", False, False)
            for size in ("0.5B", "1.5B", "3B", "7B", "14B", "32B")
        ],
        ("meta-llama/Llama-3.1-8B-Instruct", "llama", False, False),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama", False, False),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama", False, False),
        ("unsloth/Llama-3.2-1B-Instruct", "llama", False, False),
        ("mistralai/Codestral-22B-v0.1", "mistral", False, False),
        ("mistralai/Mistral-7B-Instruct-v0.2", "mistral", False, False),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", False, False),
    ],
)
def test_registered_profiles_preserve_explicit_defaults(
    metadata_source: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    repo: str,
    model_type: str,
    thinking: bool,
    reasoning: bool,
) -> None:
    source, config = metadata_source
    result = write_metadata(
        source, tmp_path / "output", {**config, "_stark_model_repo": repo, "model_type": model_type}
    )
    codec = result["token_codec_config"]
    assert (codec["output_parser_regex"] is not None) == thinking
    assert codec["end_of_thinking_tag"] == ("\n</think>" if thinking else None)
    assert codec["reasoning_config"] == (
        {
            "default_reasoning_effort": "medium",
            "field_name": "enable_thinking",
            "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
        }
        if reasoning
        else None
    )
    assert codec["default_system_prompt"] is None
    assert [codec[f"{role}_role_name"] for role in ("system", "user", "assistant")] == ["system", "user", "assistant"]


def test_profile_architecture_mismatch_is_a_hard_error(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    with pytest.raises(ConversionError, match="^Model architecture does not match metadata profile\\.$"):
        write_metadata(source, tmp_path / "output", {**config, "model_type": "llama"})
    assert not (tmp_path / "output").exists()


def test_devstral_uses_explicit_external_sources_and_json_field(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_bytes((source / "tokenizer.json").read_bytes())
    (base / "tokenizer_config.json").write_text(json.dumps({"bos_token": "[BOS]", "eos_token": "[ALT]"}))
    (base / "generation_config.json").write_text(json.dumps({"eos_token_id": 3, "temperature": 0.4}))
    instruct = tmp_path / "instruct"
    instruct.mkdir()
    (instruct / "chat_template.json").write_text(json.dumps({"chat_template": "{{ bos_token }}{{ messages }}"}))
    (source / "SYSTEM_PROMPT.txt").write_text("Synthetic fixture system text.")
    repo = "mistralai/Devstral-Small-2505"
    roots = {
        repo: source,
        "mistralai/Mistral-Small-3.1-24B-Base-2503": base,
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503": instruct,
    }
    calls = []

    def resolve_file(filename: str, *, repo: str) -> Path:
        calls.append((repo, filename))
        return roots[repo] / filename

    result = write_metadata(
        source,
        tmp_path / "output",
        {**config, "_stark_model_repo": repo, "model_type": "mistral"},
        resolve_file=resolve_file,
    )
    assert result["token_codec_config"]["prompt_template"] == "{{ bos_token }}{{ messages }}"
    assert result["token_codec_config"]["default_system_prompt"] == "Synthetic fixture system text."
    assert result["token_codec_config"]["eos_token"] == "[ALT]"
    assert result["generation_config"]["temperature"] == 0.4
    assert result["generation_config"]["stop_token_ids"] == [2, 3]
    assert ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "chat_template.json") in calls
    assert (repo, "SYSTEM_PROMPT.txt") in calls


def test_external_metadata_never_uses_local_lookalike_files(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    with pytest.raises(ConversionError, match="^External metadata source requires a resolved snapshot\\.$"):
        write_metadata(
            source,
            tmp_path / "output",
            {**config, "_stark_model_repo": "mistralai/Devstral-Small-2505", "model_type": "mistral"},
        )
    assert not (tmp_path / "output").exists()


def test_conflicting_template_sources_are_a_hard_error(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    entry = get_model(config["_stark_model_repo"])
    assert entry is not None
    entry = deepcopy(entry)
    entry["files"]["chat_template"] = {"repo": entry["repo"], "filename": "chat_template.jinja"}
    with pytest.raises(ConversionError, match="^Conflicting chat-template metadata\\.$"):
        write_metadata(source, tmp_path / "output", config, model_entry=entry)
    assert not (tmp_path / "output").exists()


def test_text_template_sidecar_is_used_when_tokenizer_template_is_absent(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    _update_json(source / "tokenizer_config.json", {"chat_template": None})
    (source / "chat_template.jinja").write_text("{% generation %}{{ messages }}{% endgeneration %}")
    entry = deepcopy(get_model(config["_stark_model_repo"]))
    entry["files"]["chat_template"] = {"repo": entry["repo"], "filename": "chat_template.jinja"}
    result = write_metadata(source, tmp_path / "output", config, model_entry=entry)
    assert result["token_codec_config"]["prompt_template"] == "{{ messages }}"


def test_native_generation_config_and_overrides_match_merge_semantics(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source
    entry = deepcopy(get_model(config["_stark_model_repo"]))
    del entry["files"]["generation_config"]
    entry["metadata"]["generation_config"] = {"stop_token_ids": [3], "temperature": 0.6, "banned_tokens": [1, 3]}
    entry["metadata"]["generation_params_overrides"] = {
        "stop_token_ids": [1, 3],
        "temperature": None,
        "top_k": 0,
        "banned_tokens": [2, 3],
        "presence_penalty": 1.5,
        "suffix_repetition_length": 1024,
    }
    result = write_metadata(source, tmp_path / "output", config, model_entry=entry)["generation_config"]
    assert result["stop_token_ids"] == [1, 2, 3]
    assert result["temperature"] == 0.6
    assert result["top_k"] == 0
    assert result["banned_tokens"] == [1, 2, 3]
    assert result["presence_penalty"] == 1.5
    assert result["suffix_repetition_length"] == 1024


def test_no_generation_source_preserves_eos_only(metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    source, config = metadata_source
    entry = deepcopy(get_model(config["_stark_model_repo"]))
    del entry["files"]["generation_config"]
    (source / "generation_config.json").write_text("not-json")
    result = write_metadata(source, tmp_path / "output", config, model_entry=entry)["generation_config"]
    assert result["stop_token_ids"] == [2]
    assert all(value is None for name, value in result.items() if name != "stop_token_ids")


def test_source_resolver_errors_never_expose_exception_content(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, config = metadata_source

    def resolve_file(filename: str, *, repo: str) -> Path:
        raise RuntimeError("private-fixture-marker")

    with pytest.raises(ConversionError, match="^Cannot resolve metadata source file\\.$") as error:
        write_metadata(source, tmp_path / "output", config, resolve_file=resolve_file)
    assert error.value.__suppress_context__
    assert "private-fixture-marker" not in str(error.value)
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(
    not os.environ.get("STARK_CONVERT_REFERENCE_ARTIFACTS"),
    reason="Set STARK_CONVERT_REFERENCE_ARTIFACTS to independently generated private reference artifacts.",
)
@pytest.mark.parametrize("repo", sorted(SUPPORTED_METADATA_REPOS))
def test_exact_external_reference_metadata_parity(tmp_path: Path, repo: str) -> None:
    reference = Path(os.environ["STARK_CONVERT_REFERENCE_ARTIFACTS"]) / repo.replace("/", "__")
    if not reference.is_dir():
        pytest.skip("This model profile has no supplied reference artifact.")
    provenance = json.loads((reference / "provenance.json").read_text())
    assert provenance["reference_revision"] == "831accb8252a6b6606c378048caeb44178f1a050"
    source = reference / "sources" / repo
    config = json.loads((source / "config.json").read_text())
    ours = tmp_path / "stream"
    fragments = write_metadata(
        source,
        ours,
        {**config, "_stark_model_repo": repo},
        resolve_file=lambda filename, *, repo: reference / "sources" / repo / filename,
    )
    oracle = reference / "expected"
    oracle_config = json.loads((oracle / "config.json").read_text())
    # Fixed failure output avoids exposing private template/token content.
    codec_matches = fragments["token_codec_config"] == oracle_config["token_codec_config"]
    generation_matches = fragments["generation_config"] == oracle_config["generation_config"]
    tokenizer_matches = (ours / "tokenizer.json").read_bytes() == (oracle / "tokenizer.json").read_bytes()
    assert codec_matches, "Codec metadata differs."
    assert generation_matches, "Generation metadata differs."
    assert tokenizer_matches, "Tokenizer differs."
