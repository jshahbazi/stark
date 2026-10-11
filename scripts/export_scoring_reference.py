"""Export a sealed scoring run to a self-contained private reference directory.

Only dataset, model content hashes and protocol expectations are copied. Native
timing and host path fields are omitted. The source configuration checksum is
retained as an opaque provenance binding; exported data never leaves private
mode-0600 files. Stdout contains only counts, checksums and authored error codes.

Usage: python3 scripts/export_scoring_reference.py --source-directory /private/sealed-run
       --output-directory /private/portable-reference --model /private/bf16-model
"""
from __future__ import annotations

import json
from pathlib import Path

from validate_scoring_rename import (Parser, ValidationFailure, file_hash, private_directory,
                                    private_json, read_private, require, verify_bindings)


def export(args):
    directory = args.source_directory
    require(directory.is_dir() and not directory.is_symlink()
            and directory.stat().st_mode & 0o077 == 0, "reference_privacy_invalid")
    config = read_private(directory / "configuration.json")
    config_hash = file_hash(directory / "configuration.json")
    source_files = {str(directory / name): file_hash(directory / name)
                    for name in ("configuration.json", "dataset.json", "pre-run-seal.json", "post-run-seal.json")}
    dataset = read_private(directory / "dataset.json")
    pre = read_private(directory / "pre-run-seal.json")
    post = read_private(directory / "post-run-seal.json")
    require(pre.get("configuration_sha256") == post.get("configuration_sha256") == config_hash
            and pre.get("dataset_sha256") == post.get("dataset_sha256") == file_hash(directory / "dataset.json")
            and post.get("ownership_settled") is True, "source_seal_invalid")
    require(config.get("schema_version") == 1 and config.get("smoke") is False
            and config.get("rounds") == 2 and config.get("warmups") == 3, "full_reference_required")
    records = {}
    keep_worker = ("candidate_token_ids", "logits", "log_probabilities", "probabilities", "decision",
                   "input_token_ids", "input_tokens", "truncated", "case_id", "condition")
    keep_response = ("candidate_token_ids", "logits", "log_probabilities", "candidate_probabilities", "temperature")
    keep_digest = ("vocab_logits_sha256", "vocab_size", "prompt_tokens_sha256", "input_tokens", "cache_hit", "truncated")
    for group in ("blocks", "digests"):
        for name, expected in post[group].items():
            require(Path(name).name == name and file_hash(directory / name) == expected, "source_seal_invalid")
            source_files[str(directory / name)] = expected
            source = read_private(directory / name)
            record = {key: source[key] for key in ("complete", "ownership_settled", "configuration_sha256")}
            record["readiness"] = {key: source["readiness"][key] for key in
                                   ("vocab_size", "upstream_revision", "scoring_patch_sha256")}
            if group == "blocks":
                record["rows"] = [{key: row[key] for key in keep_worker} for row in source["rows"]]
                record["boundaries"] = [{"length": row["length"], "kind": row["kind"], "tokens": row["tokens"],
                    "response": {key: row["response"][key] for key in keep_response}} for row in source["boundaries"]]
            else:
                record["outputs"] = [{key: value[key] for key in keep_digest} for value in source["outputs"]]
            records[name] = record
    source_model = Path(config["model"])
    model_files = {}
    for name, expected in config["files"].items():
        path = Path(name)
        if path.is_relative_to(source_model):
            relative = path.relative_to(source_model)
            require(file_hash(args.model / relative) == expected, "model_reference_mismatch")
            model_files[str(relative)] = expected
    require(model_files, "model_reference_missing")
    verify_bindings(source_files)
    fixture = {"schema_version": 1, "source_configuration_sha256": config_hash,
               "cases": dataset["cases"], "case_ids": config["case_ids"], "records": records,
               "model_files": model_files, "context_tokens": config["context_tokens"],
               "temperature": config["temperature"], "boundary_lengths": config["boundary_lengths"],
               "scoring_patch_sha256": records["round-1-baseline.json"]["readiness"]["scoring_patch_sha256"]}
    output = private_directory(args.output_directory)
    private_json(output / "REFERENCE.json", fixture)
    digest = file_hash(output / "REFERENCE.json")
    private_json(output / "MANIFEST.json", {"schema_version": 1, "reference_sha256": digest})
    print(json.dumps({"status": "exported", "cases": len(fixture["cases"]), "reference_sha256": digest}), flush=True)


def main():
    try:
        parser = Parser(description=__doc__)
        for flag in ("source-directory", "output-directory", "model"):
            parser.add_argument("--" + flag, type=Path, required=True)
        args = parser.parse_args()
        for key, value in vars(args).items():
            setattr(args, key, value.absolute())
        export(args)
    except (Exception, KeyboardInterrupt) as failure:
        code = str(failure) if isinstance(failure, ValidationFailure) else "reference_export_failed"
        print(json.dumps({"status": "failed", "error": code}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
