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
    assert sum(entry["metadata_profile"] is not None for entry in models()) == 37
    assert sum(entry["adapter"] is not None for entry in load_catalog()["models"]) == 33


def test_extensions_are_separate_from_builtin_inventory():
    assert all(entry["catalog_origin"] == "pinned_builtin" for entry in load_catalog()["models"])
    assert all(entry["catalog_origin"] == "stark_extension" for entry in load_catalog()["extensions"])
    assert adapter_for_model("unsloth/Llama-3.2-1B-Instruct", "llama") == "llama"
    assert adapter_for_model("mistralai/Mistral-7B-Instruct-v0.3", "mistral") == "mistral"


def test_external_files_and_field_selection_are_retained():
    entry = get_model("mistralai/Devstral-Small-2505")
    rules = {rule["role"]: rule for rule in required_files(entry)}
    assert rules["tokenizer"]["repo"] == "mistralai/Mistral-Small-3.1-24B-Base-2503"
    assert rules["chat_template"]["field"] == "chat_template"
    assert rules["system_prompt"]["filename"] == "SYSTEM_PROMPT.txt"


def test_dispatch_requires_an_implemented_specific_tuple():
    assert adapter_for_model("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2") == "qwen2"
    with pytest.raises(UnsupportedConversion):
        adapter_for_model("Qwen/Qwen3-0.6B-MLX-4bit")
    with pytest.raises(UnsupportedConversion):
        adapter_for_model("google/gemma-4-E2B-it")
    with pytest.raises(UnsupportedConversion):
        adapter_for_model("Qwen/Qwen3-0.6B", "llama")
    assert get_model("unregistered/model") is None


def test_catalog_callers_cannot_mutate_later_dispatch():
    entry = get_model("Qwen/Qwen3-0.6B")
    entry["adapter"] = None
    assert adapter_for_model("Qwen/Qwen3-0.6B") == "qwen3"
