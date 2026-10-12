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
    from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
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
        parser = getattr(spec, "output_parser_regex", None)
        parser_id = (
            "optional_thinking"
            if parser == OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
            else ("sha256:" + _digest(parser) if parser is not None else None)
        )
        reasoning = getattr(spec, "reasoning_config", None)
        if reasoning == BOOLEAN_REASONING_DEFAULT_ON_CONFIG:
            reasoning_id = "boolean_default_on"
        elif reasoning == BOOLEAN_REASONING_DEFAULT_OFF_CONFIG:
            reasoning_id = "boolean_default_off"
        else:
            reasoning_id = "sha256:" + _digest(reasoning) if reasoning is not None else None
        thinking_end = getattr(spec, "end_of_thinking_tag", None)
        end_id = (
            "qwen_think_end"
            if thinking_end == "\n</think>"
            else ("sha256:" + _digest(thinking_end) if thinking_end is not None else None)
        )
        grammar = getattr(spec, "grammar_start_tokens", ())
        roles = {role: getattr(spec, role + "_role_name", role) for role in ("system", "user", "assistant")}
        dense = container == ".safetensors" and not any(word in spec.name.lower() for word in ("mlx", "4bit", "8bit"))
        milestone_family = spec.family in {"Qwen3", "Qwen2.5-Coder", "Llama-3.1", "Llama-3.2", "Codestral", "Devstral"}
        adapter = profiles.get(class_name) if dense and milestone_family else None
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
                "metadata_profile": adapter,
                "status": {
                    "conversion": "implemented" if adapter else "planned",
                    "runtime": "not_assessed",
                    "access": "not_assessed",
                    "qualification": "pending",
                },
            }
        )
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
