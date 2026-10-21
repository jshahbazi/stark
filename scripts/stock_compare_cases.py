"""Load the sealed JevBench population without exposing case payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def load_reference(path: Path | str):
    root = Path(path).resolve()
    config = _read(root / "configuration.json")
    dataset = _read(root / "dataset.json")
    pre = _read(root / "pre-run-seal.json")
    post = _read(root / "post-run-seal.json")
    if config.get("rounds") != 2:
        raise ValueError("reference_population_invalid")
    if post.get("ownership_settled") is not True:
        raise ValueError("reference_ownership_unsettled")
    config_hash = hashlib.sha256(_canonical(config)).hexdigest()
    if (
        pre.get("configuration_sha256") != config_hash
        or post.get("configuration_sha256") != config_hash
    ):
        raise ValueError("reference_configuration_hash_invalid")
    dataset_hash = _sha(root / "dataset.json")
    if (
        pre.get("dataset_sha256") != dataset_hash
        or post.get("dataset_sha256") != dataset_hash
    ):
        raise ValueError("reference_dataset_hash_invalid")
    cases = dataset.get("cases")
    if (
        dataset.get("schema_version") != 1
        or not isinstance(cases, list)
        or len(cases) != 243
    ):
        raise ValueError("dataset_schema_invalid")
    if len({case.get("id") for case in cases}) != 243:
        raise ValueError("dataset_ids_invalid")
    required_blocks = {
        f"round-{round_number}-{engine}.json"
        for round_number in (1, 2)
        for engine in ("baseline", "candidate")
    }
    required_digests = {"digest-baseline.json", "digest-candidate.json"}
    blocks = post.get("blocks", {})
    digests = post.get("digests", {})
    if set(blocks) != required_blocks or set(digests) != required_digests:
        raise ValueError("reference_records_invalid")
    for name, expected_hash in {**blocks, **digests}.items():
        if _sha(root / name) != expected_hash:
            raise ValueError("reference_record_hash_invalid")

    baseline = _read(root / "round-1-baseline.json")
    rows = {
        row["case_id"]: row
        for row in baseline["rows"]
        if row.get("condition") == "fresh"
    }
    if len(rows) != 243:
        raise ValueError("reference_case_rows_invalid")
    output = []
    for case in cases:
        question = case.get("question")
        if not isinstance(question, dict):
            raise ValueError("case_schema_invalid")
        # Token IDs are sealed in the reference rows, not in the public
        # dataset cases.  Use only the fresh baseline rows as the immutable
        # tokenization source; never regenerate prompts or tokens here.
        row = rows.get(case["id"])
        if row is None:
            raise ValueError("reference_case_row_missing")
        input_ids = row.get("input_token_ids")
        candidate_ids = row.get("candidate_token_ids")
        if not isinstance(input_ids, list) or not isinstance(candidate_ids, list):
            raise ValueError("case_tokens_missing")
        labels = case.get("labels")
        if not isinstance(labels, list):
            raise ValueError("case_labels_invalid")
        output.append(
            {
                "id": case["id"],
                "tier": case["tier"],
                "expected": case["expected"],
                "labels": labels,
                "input_token_ids": [int(value) for value in input_ids],
                "candidate_ids": [int(value) for value in candidate_ids],
            }
        )
    provenance = {
        "reference_path": str(root),
        "case_count": len(output),
        "public_case_count": sum(case["tier"] != "synthetic" for case in output),
        "synthetic_case_count": sum(case["tier"] == "synthetic" for case in output),
        "dataset_sha256": dataset_hash,
        "configuration_sha256": config_hash,
        "post_run_seal_sha256": _sha(root / "post-run-seal.json"),
    }
    return output, provenance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    loaded, provenance = load_reference(parser.parse_args().reference)
    print(json.dumps({"case_count": len(loaded), **provenance}, sort_keys=True))
