"""Metadata correctness and opt-in pinned Lalamo parity checks.

Fixtures contain synthetic vocabulary and template expressions only. No model
responses, evaluation data, or native validation artifacts are stored here.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer, models

from stark_convert.metadata import SUPPORTED_METADATA_REPOS, UnsupportedMetadata, write_metadata
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
    # Metadata-only Lalamo resolves a safetensors file and reads its header;
    # it does not inspect or load tensor payloads.
    header = b"{}"
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
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
def test_malformed_metadata_never_requests_fallback(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path, filename: str, changes: dict[str, Any]
) -> None:
    source, config = metadata_source
    _update_json(source / filename, changes)
    output = tmp_path / "output"
    with pytest.raises(ConversionError):
        write_metadata(source, output, config)
    assert not output.exists()


@pytest.mark.parametrize("contents", ["[]", '{"top_k": 1, "top_k": 2}', "not-json"])
def test_invalid_json_never_requests_fallback(
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


@pytest.mark.skipif(
    os.environ.get("STARK_CONVERT_LALAMO_INTEGRATION") != "1",
    reason="Set STARK_CONVERT_LALAMO_INTEGRATION=1 to use the separately locked Lalamo environment.",
)
@pytest.mark.parametrize("repo", sorted(SUPPORTED_METADATA_REPOS))
def test_exact_pinned_lalamo_metadata_parity(
    metadata_source: tuple[Path, dict[str, Any]], tmp_path: Path, repo: str
) -> None:
    source, config = metadata_source
    project = Path(__file__).resolve().parents[1]
    ours = tmp_path / "stream"
    oracle = tmp_path / "lalamo"
    fragments = write_metadata(source, ours, {**config, "_stark_model_repo": repo})
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(project / "lalamo"),
            "--locked",
            "python",
            str(project / "stark_convert" / "lalamo_runner.py"),
            "--metadata-only",
            "--source",
            str(source),
            "--repo",
            repo,
            "--output",
            str(oracle),
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, "Pinned Lalamo metadata conversion failed."
    oracle_config = json.loads((oracle / "config.json").read_text())
    # Keep the failure output aggregate: avoid exposing template/token content.
    codec_matches = fragments["token_codec_config"] == oracle_config["token_codec_config"]
    generation_matches = fragments["generation_config"] == oracle_config["generation_config"]
    tokenizer_matches = (ours / "tokenizer.json").read_bytes() == (oracle / "tokenizer.json").read_bytes()
    assert codec_matches, "Codec metadata differs."
    assert generation_matches, "Generation metadata differs."
    assert tokenizer_matches, "Tokenizer differs."
