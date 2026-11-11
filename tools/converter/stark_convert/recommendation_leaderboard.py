"""Read the maintainer-curated whatmodel.sh snapshot without network access.

Parent-model scores are references. The site's memory plans, quantizations and
cross-machine speed figures never authorize a Stark artifact or measure it.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

GIB = 1024**3
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


class LeaderboardSnapshot:
    def __init__(self, raw):
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported leaderboard snapshot")
        self.captured = raw["captured"]
        if not isinstance(self.captured, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.captured):
            raise ValueError("Invalid capture date")
        repositories = raw["model_repositories"]
        if not isinstance(repositories, dict) or len(repositories) > 4096:
            raise ValueError("Invalid model identities")
        if any(not isinstance(repo, str) or not MODEL_ID.fullmatch(repo) for repo in repositories.values()):
            raise ValueError("Invalid model repository")
        self.buckets = {}
        buckets = raw["buckets"]
        if not isinstance(buckets, list) or len(buckets) > 64:
            raise ValueError("Invalid RAM buckets")
        for bucket in buckets:
            task, ram = bucket["task"], bucket["ram_gb"]
            if task not in {"general", "coding"} or type(ram) is not int or not 1 <= ram <= 4096:
                raise ValueError("Invalid task or RAM")
            if (task, ram) in self.buckets or not HEX64.fullmatch(bucket["sha256"]):
                raise ValueError("Invalid snapshot provenance")
            if not bucket["url"].startswith("https://whatmodel.sh/?"):
                raise ValueError("Invalid leaderboard source")
            rows = bucket["models"]
            if not isinstance(rows, list) or len(rows) > 1000:
                raise ValueError("Invalid leaderboard rows")
            records, seen = {}, set()
            for row in rows:
                identity, score = row["id"], row["score"]
                if (not isinstance(identity, str) or not re.fullmatch(r"[a-z0-9-]{1,200}", identity)
                        or identity in seen):
                    raise ValueError("Invalid model identity")
                seen.add(identity)
                if score is not None and not _number(score):
                    raise ValueError("Invalid leaderboard score")
                repo = repositories.get(identity)
                if repo is not None and score is not None:
                    # Multiple entries for one repository may describe different
                    # modes. Ambiguous parents receive no score automatically.
                    records[repo] = row if repo not in records else None
            self.buckets[task, ram] = (bucket, {repo: row for repo, row in records.items() if row is not None})

    def for_host(self, task, physical_bytes):
        if task not in {"general", "coding"} or type(physical_bytes) is not int:
            return None, {}
        sizes = [ram for kind, ram in self.buckets if kind == task and ram * GIB <= physical_bytes]
        return self.buckets[task, max(sizes)] if sizes else (None, {})

    def rank(self, ordered, task, preference, physical_bytes, model_id):
        bucket, records = self.for_host(task, physical_bytes)
        used = set()
        if preference == "quality":
            scored = [recipe for recipe in ordered if model_id(recipe) in records]
            # Stable sort retains resource preference and deterministic ties.
            scored.sort(key=lambda recipe: -records[model_id(recipe)]["score"])
            used = {recipe.id for recipe in scored}
            ordered = scored + [recipe for recipe in ordered if recipe.id not in used]
        return ordered, bucket, records, used

    def evidence(self, recipe, parent, task, bucket, records, used):
        row = records.get(parent)
        if bucket is None or row is None:
            return [], []
        evidence = {
            "benchmark": "whatmodel_" + task,
            "score": str(row["score"]),
            "model_id": parent,
            "candidate_repo": recipe.repo,
            "candidate_revision": recipe.revision,
            "evidence_kind": "upstream_reference",
            "used_for_ranking": recipe.id in used,
            "captured": self.captured,
            "ram_gb": bucket["ram_gb"],
            "source": {"url": bucket["url"], "sha256": bucket["sha256"]},
            "source_url": "https://whatmodel.sh/model/" + row["id"],
            "uncertainty": row.get("uncertainty"),
            "source_speed": row.get("speed"),
        }
        details = [
            f"whatmodel.sh {task} reference score: {row['score']}; cached {self.captured}, "
            f"Apple/local {bucket['ram_gb']} GB list. Source: {evidence['source_url']}",
            "Parent-model aggregate score; this export, quantization and Stark tool workflow were not benchmarked. "
            "Source memory estimates use an 8K context; Stark independently plans its 32K guide context. "
            "Source speed figures may use other machines and runtimes; they do not measure Stark speed.",
        ]
        return [evidence], details


@lru_cache(maxsize=1)
def load_snapshot():
    try:
        path = Path(__file__).with_name("recommendation_leaderboard.json")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Oversized leaderboard snapshot")
        return LeaderboardSnapshot(json.loads(path.read_bytes()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Optional reference corruption must not bypass resource checks or
        # cause a runtime website request. Existing evidence is the fallback.
        return LeaderboardSnapshot({"schema_version": 1, "captured": "2026-11-11",
                                    "model_repositories": {}, "buckets": []})
