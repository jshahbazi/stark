"""Read frozen llmbench coding and non-coding scores for model choices."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

GIB = 1024**3
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SPEED_MINIMUM_SCORE = {"general": 35, "coding": 35}
SPEED_WEIGHT_BUDGET_DIVISOR = 4
MAX_ALTERNATIVE_SCORE_GAP = 10


def speed_policy(task, preference, physical_bytes):
    return {
        "applied": preference == "speed",
        "minimum_score": SPEED_MINIMUM_SCORE.get(task),
        "weight_budget_bytes": physical_bytes // SPEED_WEIGHT_BUDGET_DIVISOR
        if type(physical_bytes) is int and physical_bytes > 0
        else 0,
        "evidence_kind": "llmbench_summary_score",
    }


def _alternatives(ordered, records, model_id):
    if not ordered:
        return []
    minimum = records[model_id(ordered[0])]["score"] - MAX_ALTERNATIVE_SCORE_GAP
    selected, parents = [], set()
    for recipe in ordered:
        parent = model_id(recipe)
        if records[parent]["score"] >= minimum and parent not in parents:
            selected.append(recipe)
            parents.add(parent)
    return selected


class LeaderboardSnapshot:
    def __init__(self, raw):
        if raw.get("schema_version") != 2:
            raise ValueError("Unsupported llmbench snapshot")
        if not isinstance(raw.get("captured"), str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw["captured"]):
            raise ValueError("Invalid llmbench capture date")
        if raw.get("source_url") != "https://llmbench.site/" or raw.get("data_url") != "https://llmbench.site/data/":
            raise ValueError("Invalid llmbench source")
        generated_at = raw.get("generated_at")
        source_hashes = raw.get("source_files_sha256")
        if (
            not isinstance(generated_at, str)
            or generated_at[:10] != raw["captured"]
            or not isinstance(source_hashes, dict)
            or set(source_hashes) != {"llmbench-summary-scores.csv", "llmbench-models.csv"}
            or any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in source_hashes.values())
            or raw.get("summary_url") != "https://llmbench.site/downloads/llmbench-summary-scores.csv"
            or raw.get("method_url") != "https://llmbench.site/summary/"
        ):
            raise ValueError("Invalid llmbench provenance")
        models = raw.get("models")
        if not isinstance(models, list) or not 0 <= len(models) <= 512:
            raise ValueError("Invalid llmbench model indexes")
        self.captured = raw["captured"]
        self.source_url = raw["method_url"]
        self.download_url = raw["summary_url"]
        self.source_files_sha256 = source_hashes
        self.records = {"general": {}, "coding": {}}
        self.rows = {}
        identities = set()
        for model in models:
            identity, repo = model.get("id"), model.get("repo")
            if (
                not isinstance(identity, str)
                or re.fullmatch(r"[a-z0-9-]{1,200}", identity) is None
                or identity in identities
                or not isinstance(repo, str)
                or MODEL_ID.fullmatch(repo) is None
                or repo in self.rows
            ):
                raise ValueError("Invalid llmbench model index")
            identities.add(identity)
            task_rows = {}
            for task, key in (("general", "noncoding"), ("coding", "coding")):
                score = model.get(key)
                if score is not None:
                    if type(score) not in (int, float) or not math.isfinite(score):
                        raise ValueError("Invalid llmbench task index")
                    row = {"id": identity, "score": float(score)}
                    task_rows[task] = row
                    self.records[task][repo] = row
            if not task_rows:
                raise ValueError("Unscored llmbench model")
            self.rows[repo] = task_rows
        self.bucket = {
            task: {
                "task": task,
                "ram_gb": None,
                "url": self.download_url,
                "sha256": self.source_files_sha256["llmbench-summary-scores.csv"],
            }
            for task in ("general", "coding")
        }

    def for_host(self, task, physical_bytes):
        if task not in self.records:
            return None, {}
        return self.bucket[task], self.records[task]

    def rank(self, ordered, task, preference, physical_bytes, model_id, *, preserve_fallback=False):
        if task not in {"general", "coding"}:
            return ordered, None, {}, set()
        _records = {}
        used = set()
        _bucket, _records = self.for_host(task, physical_bytes)
        scored = [recipe for recipe in ordered if model_id(recipe) in _records]
        if preference == "quality":
            scored.sort(key=lambda recipe: -_records[model_id(recipe)]["score"])
            selected = _alternatives(scored, _records, model_id)
            used = {recipe.id for recipe in selected}
            if selected:
                ordered = selected
            elif task == "coding":
                ordered = []
            else:
                ordered = ordered[:1]
        elif preference == "speed":
            policy = speed_policy(task, preference, physical_bytes)
            scored = [
                recipe
                for recipe in scored
                if _records[model_id(recipe)]["score"] >= policy["minimum_score"]
                and recipe.weight_bytes <= policy["weight_budget_bytes"]
            ]
            scored.sort(
                key=lambda recipe: (
                    -_records[model_id(recipe)]["score"],
                    recipe.runtime_memory,
                    recipe.weight_bytes,
                )
            )
            ordered = _alternatives(scored, _records, model_id)
            used = {recipe.id for recipe in ordered}
        bucket = self.bucket[task] if task in self.bucket and used else None
        return ordered, bucket, self.records.get(task, {}), used

    def evidence(self, recipe, parent, task, bucket, records, used):
        row = records.get(parent)
        if bucket is None or row is None:
            return [], []
        evidence = {
            "benchmark": "llmbench_noncoding" if task == "general" else "llmbench_coding",
            "score": str(row["score"]),
            "model_id": parent,
            "candidate_repo": recipe.repo,
            "candidate_revision": recipe.revision,
            "evidence_kind": "upstream_reference",
            "used_for_ranking": recipe.id in used,
            "captured": self.captured,
            "source": {
                "url": self.download_url,
                "sha256": self.source_files_sha256["llmbench-summary-scores.csv"],
            },
            "source_url": self.source_url,
            "relationship": "parent_model_reference_only",
            "quantization_quality_evaluated": False,
        }
        details = [
            f"llmbench {'non-coding' if task == 'general' else 'coding'} score: {row['score']}; "
            f"cached {self.captured}. Source: {self.source_url}",
            "Relative llmbench summary score, not a pass percentage. The parent model was scored; this export, "
            "quantization and Stark tool workflow were not benchmarked.",
        ]
        return [evidence], details


@lru_cache(maxsize=1)
def load_snapshot():
    try:
        path = Path(__file__).with_name("llmbench_catalog_ranking.json")
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Oversized llmbench snapshot")
        return LeaderboardSnapshot(json.loads(path.read_bytes()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return LeaderboardSnapshot(
            {
                "schema_version": 2,
                "captured": "2026-11-17",
                "generated_at": "2026-11-17T00:00:00Z",
                "source_url": "https://llmbench.site/",
                "data_url": "https://llmbench.site/data/",
                "summary_url": "https://llmbench.site/downloads/llmbench-summary-scores.csv",
                "method_url": "https://llmbench.site/summary/",
                "source_files_sha256": {
                    "llmbench-summary-scores.csv": "0" * 64,
                    "llmbench-models.csv": "0" * 64,
                },
                "models": [],
            }
        )
