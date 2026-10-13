"""Inventory denominator and conservative dispatch remain explicit."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stark_convert.registry import adapter_for_model, get_model, load_catalog, models, required_files
from stark_convert.streaming import UnsupportedConversion


def test_complete_pinned_builtin_inventory():
    catalog = load_catalog()
    assert catalog["entry_count"] == 110 == len(catalog["models"])
    assert catalog["extension_count"] == 4 == len(catalog["extensions"])
    assert catalog["reference"]["plugins"] is False
    assert catalog["reference"]["revision"] == "831accb8252a6b6606c378048caeb44178f1a050"
    assert len({entry["repo"] for entry in models()}) == 114
    assert sum(entry["model_kind"] == "LanguageModelSpec" for entry in catalog["models"]) == 107
    assert sum(entry["model_kind"] == "ClassifierModelSpec" for entry in models()) == 2
    assert sum(entry["model_kind"] == "TTSModelSpec" for entry in models()) == 1
    assert all(set(entry["status"]) == {"conversion", "runtime", "access", "qualification"} for entry in models())


def test_catalog_is_independent_of_reference_and_framework_imports():
    code = (
        "import sys; from stark_convert.registry import models; assert len(models()) == 114; "
        "assert not any(name.split('.')[0] in {'lalamo','jax','torch'} for name in sys.modules)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    assert result.returncode == 0


def test_profiles_have_no_embedded_prompt_or_vocabulary():
    for entry in models():
        assert "chat_template" not in entry["metadata"]
        assert "system_prompt" not in entry["metadata"]
        assert "vocab" not in entry["metadata"]
        assert "stop_token_ids" not in json.dumps(entry["metadata"])
    assert sum(entry["metadata_profile"] is not None for entry in models()) == 106
    assert sum(entry["adapter"] is not None for entry in load_catalog()["models"]) == 102


def test_specialized_profiles_select_explicit_dense_and_moe_contracts():
    assert {
        entry["repo"] for entry in models() if entry["adapter"] == "qwen3_5" and not entry["compressed_format"]
    } == {f"Qwen/Qwen3.5-{size}" for size in ("0.8B", "2B", "4B", "9B", "27B")} | {
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.8-27B",
    }
    assert {entry["repo"] for entry in models() if entry["adapter"] == "lfm2" and not entry["compressed_format"]} == {
        f"LiquidAI/LFM2-{size}" for size in ("350M", "700M", "1.2B", "2.6B")
    } | {"LiquidAI/LFM2.5-350M", "LiquidAI/LFM2.5-1.2B-Instruct", "LiquidAI/LFM2.5-1.2B-Thinking"}
    assert {entry["repo"] for entry in models() if entry["adapter"] == "qwen3_5_moe"} == {
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    for repo in ("openai/gpt-oss-20b", "google/gemma-4-E2B", "fishaudio/s1-mini"):
        entry = get_model(repo)
        assert entry is not None and entry["adapter"] is None
        assert entry["metadata_profile"] is None and entry["status"]["conversion"] == "planned"


def test_extensions_are_separate_from_builtin_inventory():
    assert all(entry["catalog_origin"] == "pinned_builtin" for entry in load_catalog()["models"])
    assert all(entry["catalog_origin"] == "stark_extension" for entry in load_catalog()["extensions"])
    assert adapter_for_model("unsloth/Llama-3.2-1B-Instruct", "llama") == "llama"
    assert adapter_for_model("mistralai/Mistral-7B-Instruct-v0.3", "mistral") == "mistral"


def test_mlx_profiles_bind_bit_width_without_promoting_other_packed_families():
    packed = [entry for entry in models() if entry.get("compressed_format") is not None]
    assert len(packed) == 48
    assert {
        adapter: sum(entry["adapter"] == adapter for entry in packed)
        for adapter in {"qwen3", "llama", "smollm3", "gemma3_text", "gemma3", "qwen3_5", "lfm2", "llamba"}
    } == {"qwen3": 12, "llama": 5, "smollm3": 2, "gemma3_text": 2, "gemma3": 4, "qwen3_5": 10, "lfm2": 12, "llamba": 1}
    assert {entry["compressed_format"]["bits"] for entry in packed} == {4, 8}
    for entry in packed:
        assert entry["adapter"] == entry["metadata_profile"]
        assert entry["compressed_format"]["scheme"] == "mlx_affine"
        assert (
            entry["repo"].endswith(f"-{entry['compressed_format']['bits']}bit")
            or entry["repo"] == "cartesia-ai/Llamba-1B-4bit-mlx"
        )
        assert entry["status"]["qualification"] == "pending"
    assert get_model("mlx-community/Qwen3.5-0.8B-MLX-4bit")["adapter"] == "qwen3_5"
    assert get_model("mlx-community/gemma-3-4b-it-4bit")["adapter"] == "gemma3"
    assert get_model("mlx-community/gemma-3-27b-it-8bit")["adapter"] == "gemma3"


def test_external_files_and_field_selection_are_retained():
    entry = get_model("mistralai/Devstral-Small-2505")
    rules = {rule["role"]: rule for rule in required_files(entry)}
    assert rules["tokenizer"]["repo"] == "mistralai/Mistral-Small-3.1-24B-Base-2503"
    assert rules["chat_template"]["field"] == "chat_template"
    assert rules["system_prompt"]["filename"] == "SYSTEM_PROMPT.txt"


def test_llamba_profiles_keep_external_config_and_exact_format_identity():
    assert {entry["repo"] for entry in models() if entry["adapter"] == "llamba"} == {
        "cartesia-ai/Llamba-1B",
        "cartesia-ai/Llamba-1B-4bit-mlx",
    }
    dense = get_model("cartesia-ai/Llamba-1B")
    packed = get_model("cartesia-ai/Llamba-1B-4bit-mlx")
    assert dense["compressed_format"] is None
    assert packed["compressed_format"] == {"scheme": "mlx_affine", "bits": 4}
    assert packed["files"]["model_config"] == {"repo": dense["repo"], "filename": "config.json"}
    assert {entry["files"]["tokenizer"]["repo"] for entry in (dense, packed)} == {"meta-llama/Llama-3.2-1B-Instruct"}


def test_dispatch_requires_an_implemented_specific_tuple():
    assert adapter_for_model("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2") == "qwen2"
    assert adapter_for_model("Qwen/Qwen3-0.6B-MLX-4bit", "qwen3") == "qwen3"
    assert adapter_for_model("Qwen/Qwen3-0.6B-MLX-8bit", "qwen3") == "qwen3"
    assert adapter_for_model("mlx-community/Qwen3.5-0.8B-MLX-4bit") == "qwen3_5"
    assert adapter_for_model("mlx-community/LFM2-350M-8bit") == "lfm2"
    with pytest.raises(UnsupportedConversion):
        adapter_for_model("google/gemma-4-E2B-it")
    with pytest.raises(UnsupportedConversion):
        adapter_for_model("Qwen/Qwen3-0.6B", "llama")
    assert get_model("unregistered/model") is None


def test_hybrid_mlx_promotions_use_exact_pinned_checkpoint_names():
    qwen = {
        f"mlx-community/Qwen3.5-{size}-MLX-{bits}bit" for size in ("0.8B", "2B", "4B", "9B") for bits in (4, 8)
    } | {f"mlx-community/Qwen3.5-27B-{bits}bit" for bits in (4, 8)}
    lfm = (
        {f"mlx-community/LFM2-{size}-{bits}bit" for size in ("350M", "700M", "1.2B", "2.6B") for bits in (4, 8)}
        | {f"LiquidAI/LFM2.5-1.2B-Instruct-MLX-{bits}bit" for bits in (4, 8)}
        | {f"mlx-community/LFM2.5-1.2B-Thinking-{bits}bit" for bits in (4, 8)}
    )
    for architecture, expected in (("qwen3_5", qwen), ("lfm2", lfm)):
        assert {
            entry["repo"] for entry in models() if entry["adapter"] == architecture and entry["compressed_format"]
        } == expected
    for entry in models():
        if entry["config_class"] in {"HFQwen35Config", "HFLFM2Config"} and entry["adapter"] is None:
            assert entry["compressed_format"] is None
            assert entry["status"]["conversion"] == "planned"


def test_catalog_callers_cannot_mutate_later_dispatch():
    entry = get_model("Qwen/Qwen3-0.6B")
    entry["adapter"] = None
    assert adapter_for_model("Qwen/Qwen3-0.6B") == "qwen3"
