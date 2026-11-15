"""Bundle a reviewed coding aggregate with explicit repository identities.

This does not recompute scores. Supply the saved llmbench index used to resolve
model IDs; its identity-only provenance is separate from the frozen score inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path


def bundle(aggregate_path, index_path, output_path):
    aggregate_bytes, index_bytes = aggregate_path.read_bytes(), index_path.read_bytes()
    aggregate, index = json.loads(aggregate_bytes), json.loads(index_bytes)
    identities = {}
    for row in index["models"]:
        if row["id"] in identities:
            raise ValueError("Duplicate model identity")
        identities[row["id"]] = row.get("model_repo", "")
    repositories = {}
    for row in aggregate["models"]:
        url = identities[row["id"]]
        if not url.startswith("https://huggingface.co/"):
            raise ValueError("Missing model repository")
        repositories[row["id"]] = url.removeprefix("https://huggingface.co/")
    aggregate["model_repositories"] = repositories
    aggregate["compilation_license"] = "CC BY-SA 4.0"
    aggregate["attribution"] = (
        "Derived from llmbench.site's compilation (CC BY-SA 4.0), incorporating whatmodel.sh "
        "by Merlinsbeard Pty Ltd. Underlying sources retain their own licenses; see sources. "
        "Stark fits coding indexes and maps model IDs to repositories; no endorsement."
    )
    aggregate["method_url"] = "https://llmbench.site/method/"
    aggregate["aggregate_sha256"] = hashlib.sha256(aggregate_bytes).hexdigest()
    aggregate["identity_index"] = {
        "url": "https://llmbench.site/data/index.json",
        "generated_at": index["generated_at"],
        "sha256": hashlib.sha256(index_bytes).hexdigest(),
        "purpose": (
            "Repository identity mapping only; scores and source hashes remain the frozen November 14 aggregate"
        ),
    }
    output_path.write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle(args.aggregate, args.index, args.output)
