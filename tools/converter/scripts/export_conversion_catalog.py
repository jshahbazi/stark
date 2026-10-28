"""Export safe public inventory from an explicitly supplied pinned reference.

This maintenance command is optional and is never imported by the converter.
Use an external interpreter already containing the reference's dependencies.
Registry interpretation adapted from Lalamo, MIT, Copyright (c) 2025 Mirai
Tech Inc.; see ../LICENSE.lalamo. Literal prompts/templates/token data are
represented by hashes, never copied into this public catalog.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REFERENCE_REVISION = "831accb8252a6b6606c378048caeb44178f1a050"
QWEN38_DISTILL_REPO = "empero-ai/Qwen3.8-4B-Distill"
QWEN38_DISTILL_REVISION = "c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e"
OLMO3_REPO = "allenai/Olmo-3-7B-Instruct"
OLMO3_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
GEMMA4_REPO = "google/gemma-4-E2B-it"
GEMMA4_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def profile_extension(
    entries: list[dict],
    *,
    repo: str,
    revision: str,
    reference_repo: str,
    config_class: str,
    adapter: str,
    family: str,
    size: str,
    vendor: str,
    file_hashes: dict[str, str],
) -> dict:
    """Create an explicit Stark text-adapter profile for a pinned HF source."""
    entry = copy.deepcopy(next(entry for entry in entries if entry["repo"] == reference_repo))
    entry.update(
        repo=repo,
        catalog_origin="stark_extension",
        profile_reference=reference_repo,
        family=family,
        vendor=vendor,
        name=repo.split("/")[1],
        size=size,
        config_class=config_class,
        adapter=adapter,
        metadata_profile=adapter,
        profile_source={"repo": repo, "revision": revision, "files_sha256": file_hashes},
    )
    entry["files"] = {
        role: {"repo": repo, "filename": filename}
        for role, filename in {
            "model_config": "config.json",
            "tokenizer": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json",
            "generation_config": "generation_config.json",
            "chat_template": "chat_template.jinja",
        }.items()
    }
    if adapter == "gemma4_text":
        entry["files"]["processor_config"] = {"repo": repo, "filename": "processor_config.json"}
    entry["metadata"] = {
        "parser_id": None,
        "reasoning_id": None,
        "end_of_thinking_id": None,
        "generation_config": None,
        "generation_params_overrides": None,
        "grammar_start_tokens_count": 0,
        "grammar_start_tokens_sha256": _digest([]),
        "inline_sources": {},
        "roles": {"system": "system", "user": "user", "assistant": "assistant"},
    }
    entry["status"] = {
        "conversion": "implemented",
        "runtime": "not_assessed",
        "access": "not_assessed",
        "qualification": "pending",
    }
    return entry


def qwen38_distill_extension(entries: list[dict]) -> dict:
    """Bind the inspected dense hybrid and its own sidecars as a Stark profile.

    This identity is outside the pinned Lalamo inventory. Its public config and
    tensor headers match the existing Qwen3.5 text planner. Sampling defaults
    come only from this repository, rather than the reference model's overrides.
    """
    reference_repo = "Qwen/Qwen3.5-4B"
    entry = copy.deepcopy(next(entry for entry in entries if entry["repo"] == reference_repo))
    entry.update(
        repo=QWEN38_DISTILL_REPO,
        catalog_origin="stark_extension",
        profile_reference=reference_repo,
        vendor="Empero AI",
        family="Qwen3.8",
        name="Qwen3.8-4B-Distill",
        size="4B",
        profile_source={
            "repo": QWEN38_DISTILL_REPO,
            "revision": QWEN38_DISTILL_REVISION,
            "files_sha256": {
                "config.json": "0d721c3b6d378efee9b92741e8e4a05018882d4441c2fbf4f793be9ffff1376f",
                "generation_config.json": "f703e9fc5d6929b4e9e344a0cf4a188da6717da217fa24031067e35ab730ee53",
                "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
                "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            },
        },
    )
    for file in entry["files"].values():
        file["repo"] = QWEN38_DISTILL_REPO
    # The inline template is read from its own tokenizer_config.json. The
    # identical standalone template is fingerprinted above, not read twice.
    entry["metadata"]["inline_sources"] = {}
    return entry


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def export(reference: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_REVISION:
        raise ValueError("Reference revision does not match the pinned inventory")
    if subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain"], text=True).strip():
        raise ValueError("Reference checkout must be clean")
    sys.path.insert(0, str(reference))
    from lalamo.model_import.model_spec import JSONFieldSpec
    from lalamo.model_import.model_specs.output_parser_regexes import (
        GRANITE_THINKING_OUTPUT_PARSER_REGEX,
        OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
    )
    from lalamo.model_import.model_specs.reasoning_configs import (
        BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
        BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    )
    from lalamo.model_import.origins import FileSpec
    from lalamo.model_registry import ModelRegistry

    entries = []
    profiles = {
        "HFQwen3Config": "qwen3",
        "HFQwen2Config": "qwen2",
        "HFLlamaConfig": "llama",
        "HFMistralConfig": "mistral",
        "HFGemma3TextConfig": "gemma3_text",
        "HFPhi3Config": "phi3",
        "HFGraniteConfig": "granite",
        "HFSmolLM3Config": "smollm3",
        "HFQwen35Config": "qwen3_5",
        "HFLFM2Config": "lfm2",
        "HFLlambaConfig": "llamba",
        "ModernBERTConfig": "modernbert",
    }
    for spec in ModelRegistry.build(allow_third_party_plugins=False).models:
        repo = spec.origin.description
        class_name = spec.config_type.__name__
        container = getattr(spec.origin, "weight_format", None)
        container = container.value if container is not None else ".nemo"
        files = {}
        inline = {}
        for field in dataclasses.fields(spec.configs):
            value = getattr(spec.configs, field.name)
            if isinstance(value, JSONFieldSpec):
                files[field.name] = {
                    "repo": value.file_spec.repo or repo,
                    "filename": value.file_spec.filename,
                    "field": value.field_name,
                }
            elif isinstance(value, FileSpec):
                files[field.name] = {"repo": value.repo or repo, "filename": value.filename}
            elif value is not None:
                # Inline generation values may include token ID vectors.
                inline[field.name] = {
                    "sha256": _digest(dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value)
                }
        if type(spec).__name__ == "ClassifierModelSpec":
            # ConfigMap inherits this language default, but classifier import
            # never selects or reads a generation policy file.
            files.pop("generation_config", None)
        parser = getattr(spec, "output_parser_regex", None)
        parsers = {
            OPTIONAL_THINKING_OUTPUT_PARSER_REGEX: "optional_thinking",
            GRANITE_THINKING_OUTPUT_PARSER_REGEX: "granite_thinking",
        }
        parser_id = parsers.get(parser, "sha256:" + _digest(parser) if parser is not None else None)
        reasoning = getattr(spec, "reasoning_config", None)
        if reasoning == BOOLEAN_REASONING_DEFAULT_ON_CONFIG:
            reasoning_id = "boolean_default_on"
        elif reasoning == BOOLEAN_REASONING_DEFAULT_OFF_CONFIG:
            reasoning_id = "boolean_default_off"
        elif class_name == "HFGraniteConfig" and spec.family == "Granite" and spec.name.startswith("granite-3.3-"):
            # Bind the special field-name profile to the exact upstream value.
            expected = {
                "default_reasoning_effort": "no_reasoning",
                "field_name": "thinking",
                "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
            }
            if reasoning is None or dataclasses.asdict(reasoning) != expected:
                raise ValueError("Unrecognized Granite reasoning profile")
            reasoning_id = "granite_boolean_default_off"
        else:
            reasoning_id = "sha256:" + _digest(reasoning) if reasoning is not None else None
        thinking_end = getattr(spec, "end_of_thinking_tag", None)
        end_ids = {"\n</think>": "qwen_think_end", "\n\n</think>": "double_newline_think_end", "</think>": "think_end"}
        end_id = end_ids.get(thinking_end, "sha256:" + _digest(thinking_end) if thinking_end is not None else None)
        grammar = getattr(spec, "grammar_start_tokens", ())
        roles = {role: getattr(spec, role + "_role_name", role) for role in ("system", "user", "assistant")}
        dense = container == ".safetensors" and not any(word in spec.name.lower() for word in ("mlx", "4bit", "8bit"))
        # Only inspected affine MLX decoder layouts are independently planned.
        # Other packed formats/config classes keep their explicit planned state.
        mlx = (
            container == ".safetensors"
            and class_name
            in {
                "HFQwen3Config",
                "HFLlamaConfig",
                "HFSmolLM3Config",
                "HFGemma3TextConfig",
                "HFQwen35Config",
                "HFLFM2Config",
                "HFLlambaConfig",
            }
            and any(width in spec.name.lower() for width in ("4bit", "8bit"))
        )
        adapter = profiles.get(class_name) if dense or mlx else None
        # Promote inspected dense families; later generations keep their own
        # metadata/variant gates even when the reference shares a config class.
        # The shared upstream config class also handles MoE. These exact pinned
        # identities use the inspected dense decoder; 35B-A3B is qwen3_5_moe.
        qwen35_repos = {f"Qwen/Qwen3.5-{size}" for size in ("0.8B", "2B", "4B", "9B", "27B")}
        qwen35_repos.update(
            f"mlx-community/Qwen3.5-{size}-MLX-{bits}bit" for size in ("0.8B", "2B", "4B", "9B") for bits in (4, 8)
        )
        qwen35_repos.update(f"mlx-community/Qwen3.5-27B-{bits}bit" for bits in (4, 8))
        if class_name == "HFQwen35Config" and repo not in qwen35_repos:
            adapter = None
        if class_name == "HFLFM2Config" and spec.family != "LFM2":
            adapter = None
        if class_name == "HFLlambaConfig" and repo not in {"cartesia-ai/Llamba-1B", "cartesia-ai/Llamba-1B-4bit-mlx"}:
            adapter = None
        if class_name == "ModernBERTConfig" and repo not in {
            "trymirai/chat-moderation-router",
            "trymirai/health-related-router",
        }:
            adapter = None
        metadata = {
            "parser_id": parser_id,
            "reasoning_id": reasoning_id,
            "end_of_thinking_id": end_id,
            "roles": roles,
            "grammar_start_tokens_count": len(grammar),
            "grammar_start_tokens_sha256": _digest(grammar),
            "inline_sources": inline,
            "generation_config": None,
            "generation_params_overrides": None,
        }
        entries.append(
            {
                "catalog_origin": "pinned_builtin",
                "repo": repo,
                "origin": type(spec.origin).__name__,
                "model_kind": type(spec).__name__,
                "vendor": spec.vendor,
                "family": spec.family,
                "name": spec.name,
                "size": spec.size,
                "config_class": class_name,
                "container": container,
                "files": files,
                "metadata": metadata,
                "adapter": adapter,
                "metadata_profile": "modernbert-classifier" if adapter == "modernbert" else adapter,
                "compressed_format": {
                    "scheme": "mlx_affine",
                    "bits": 4 if "4bit" in spec.name.lower() else 8,
                }
                if mlx and adapter is not None
                else None,
                "status": {
                    "conversion": "implemented" if adapter else "planned",
                    "runtime": "not_assessed",
                    "access": "not_assessed",
                    "qualification": "pending",
                },
            }
        )
    # Gemma-4 E2B already exists in the pinned registry as a planned row. Bind
    # the new text adapter to that identity without duplicating the repository
    # or changing the source-bound builtin denominator.
    gemma4 = profile_extension(
        entries,
        repo=GEMMA4_REPO,
        revision=GEMMA4_REVISION,
        reference_repo="google/gemma-3-4b-it",
        config_class="HFGemma4TextConfig",
        adapter="gemma4_text",
        family="Gemma-4",
        vendor="Google",
        size="E2B",
        file_hashes={
            "config.json": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
            "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
            "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
            "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            "chat_template.jinja": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
            "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
        },
    )
    gemma4["catalog_origin"] = "pinned_builtin"
    entries[:] = [gemma4 if entry["repo"] == GEMMA4_REPO else entry for entry in entries]

    extensions = []
    for repository, reference_repo, family in (
        ("unsloth/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2"),
        ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Codestral-22B-v0.1", "Mistral-7B"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Codestral-22B-v0.1", "Mistral-7B"),
    ):
        entry = copy.deepcopy(next(entry for entry in entries if entry["repo"] == reference_repo))
        entry.update(
            repo=repository,
            catalog_origin="stark_extension",
            profile_reference=reference_repo,
            family=family,
            name=repository.split("/")[1],
            size="1B" if "Llama" in repository else "7B",
        )
        for file in entry["files"].values():
            file["repo"] = repository
        extensions.append(entry)
    # The pinned config/loader supports Gemma2 but the builtin registry has no
    # Gemma2 row. This explicit extension uses standard LanguageModelSpec files.
    gemma2 = copy.deepcopy(next(entry for entry in entries if entry["repo"] == "google/gemma-3-1b-it"))
    gemma2.update(
        repo="google/gemma-2-2b-it",
        catalog_origin="stark_extension",
        profile_reference="HFGemma2Config / standard LanguageModelSpec",
        family="Gemma-2",
        name="gemma-2-2b-it",
        size="2B",
        config_class="HFGemma2Config",
        adapter="gemma2",
        metadata_profile="gemma2",
    )
    for file in gemma2["files"].values():
        file["repo"] = gemma2["repo"]
    extensions.append(gemma2)
    extensions.append(qwen38_distill_extension(entries))
    extensions.append(
        profile_extension(
            entries,
            repo=OLMO3_REPO,
            revision=OLMO3_REVISION,
            reference_repo="Qwen/Qwen3-8B",
            config_class="HFOlmo3Config",
            adapter="olmo3",
            family="OLMo-3",
            vendor="AllenAI",
            size="7B",
            file_hashes={
                "config.json": "6282ae812f9e494bd212e9a3233d68dc4d5f1e2569642423d2b79a3e80373ef8",
                "generation_config.json": "745e78c16cc40cbb08827f7b7c9d5a888f7cb908236d4ca0443ff47f43a9d5af",
                "tokenizer_config.json": "f7a2753a4ff30657e452c8fef02b9a99a3b900ceb4c65a1e3fbea6a3bfab07a6",
                "tokenizer.json": "7738a25c46a6043a0e2e605138851baa450269ce9d800af6ca9bf6192e45510c",
                "chat_template.jinja": "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b",
            },
        )
    )
    return {
        "schema_version": 1,
        "reference": {
            "project": "trymirai/lalamo",
            "revision": REFERENCE_REVISION,
            "plugins": False,
            "license": "MIT",
            "copyright": "Copyright (c) 2025 Mirai Tech Inc.",
        },
        "exporter": "tools/converter/scripts/export_conversion_catalog.py",
        "entry_count": len(entries),
        "models": sorted(entries, key=lambda entry: entry["repo"]),
        "extension_count": len(extensions),
        "extensions": sorted(extensions, key=lambda entry: entry["repo"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export(args.reference_source.resolve())
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except Exception:
        print("Catalog export failed; check the pinned external reference environment.", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"entries": result["entry_count"], "catalog_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
