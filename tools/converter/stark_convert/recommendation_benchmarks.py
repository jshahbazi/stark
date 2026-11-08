"""Offline, attributed upstream reference scores for the beginner guide.

These are not measurements of Stark artifacts. Compare one published cohort
at a time; leave candidates without a comparable result in their fallback slots.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

TASK_BENCHMARKS = {"code": "livecodebench", "files": "bfcl"}
BENCHMARKS = {"livecodebench", "bfcl", "ifeval"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
SCORE = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,6})?\Z")


def _text(value, limit=2_000):
    if not isinstance(value, str) or not 0 < len(value.encode()) <= limit or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid benchmark metadata")
    return value


def _source(raw):
    if not isinstance(raw, dict):
        raise ValueError("Invalid benchmark source")
    for key in ("url", "publisher", "revision", "sha256"):
        _text(raw[key], 1_000 if key == "url" else 200)
    if not raw["url"].startswith("https://") or not HEX64.fullmatch(raw["sha256"]):
        raise ValueError("Invalid benchmark provenance")


@dataclass(frozen=True)
class Cohort:
    id: str
    benchmark: str
    label: str
    metric: str
    settings: str
    source: dict
    authority: str
    results: dict


class BenchmarkRegistry:
    def __init__(self, raw):
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
            raise ValueError("Unsupported benchmark registry")
        cohorts = raw["cohorts"]
        bindings = raw.get("bindings", [])
        if not isinstance(cohorts, list) or len(cohorts) > 32 or not isinstance(bindings, list) or len(bindings) > 512:
            raise ValueError("Oversized benchmark registry")
        self.cohorts = []
        seen = set()
        for value in cohorts:
            identity = _text(value["id"], 128)
            if identity in seen or value["benchmark"] not in BENCHMARKS:
                raise ValueError("Invalid benchmark cohort")
            seen.add(identity)
            _source(value["source"])
            if value["authority"] not in {"benchmark_publisher", "model_developer"}:
                raise ValueError("Unknown benchmark authority")
            results = {}
            if not isinstance(value["results"], list) or len(value["results"]) > 512:
                raise ValueError("Oversized benchmark results")
            for row in value["results"]:
                model = _text(row["model_id"], 256)
                score = row["score"]
                if (
                    not MODEL_ID.fullmatch(model)
                    or model in results
                    or not isinstance(score, str)
                    or not SCORE.fullmatch(score)
                ):
                    raise ValueError("Invalid benchmark result")
                try:
                    number = Decimal(score)
                except InvalidOperation as error:
                    raise ValueError("Invalid benchmark score") from error
                if not number.is_finite() or not 0 <= number <= 100:
                    raise ValueError("Invalid benchmark score")
                revision = row.get("evaluated_revision")
                if revision is not None and not HEX40.fullmatch(_text(revision, 40)):
                    raise ValueError("Invalid evaluated revision")
                results[model] = (number, revision)
            self.cohorts.append(
                Cohort(
                    identity,
                    value["benchmark"],
                    _text(value["label"], 160),
                    _text(value["metric"], 160),
                    _text(value["settings"], 1_500),
                    dict(value["source"]),
                    value["authority"],
                    results,
                )
            )
        self.bindings = {}
        for value in bindings:
            repo, revision, parent = value["repo"], value["revision"], value["model_id"]
            if not MODEL_ID.fullmatch(repo) or not HEX40.fullmatch(revision) or not MODEL_ID.fullmatch(parent):
                raise ValueError("Invalid benchmark binding")
            _source(value["source"])
            key = (repo, revision)
            if value["relationship"] != "reviewed_quantized_export_reference_only":
                raise ValueError("Unreviewed benchmark binding")
            if key in self.bindings:
                raise ValueError("Duplicate benchmark binding")
            self.bindings[key] = parent

    def model_id(self, recipe):
        return self.bindings.get((recipe.repo, recipe.revision), recipe.repo)

    def rank(self, ordered, task, preference, fit_tier):
        """Reorder scored slots within each resource tier, keeping unknowns fixed."""
        benchmark = TASK_BENCHMARKS.get(task, "ifeval")
        tiers = {fit_tier(recipe) for recipe in ordered}
        matches = []
        for cohort in self.cohorts:
            if cohort.benchmark != benchmark:
                continue
            comparable = set()
            for tier in tiers:
                covered = {
                    self.model_id(recipe) for recipe in ordered if fit_tier(recipe) == tier
                } & cohort.results.keys()
                if len(covered) >= 2:
                    comparable.update(covered)
            if comparable:
                matches.append((cohort, len(comparable)))
        selected = min(
            matches,
            key=lambda pair: (-pair[1], pair[0].authority != "benchmark_publisher", pair[0].id),
            default=(None, 0),
        )[0]
        ranked, used = list(ordered), set()
        if selected is not None and preference not in {"speed", "answers"}:
            for tier in tiers:
                slots = [
                    i
                    for i, recipe in enumerate(ordered)
                    if fit_tier(recipe) == tier and self.model_id(recipe) in selected.results
                ]
                if len({self.model_id(ordered[i]) for i in slots}) < 2:
                    continue
                scored = sorted(
                    (ordered[i] for i in slots), key=lambda recipe: -selected.results[self.model_id(recipe)][0]
                )
                for index, recipe in zip(slots, scored, strict=True):
                    ranked[index] = recipe
                    used.add(recipe.id)
        return ranked, selected, used

    def evidence(self, recipe, task, selected, used):
        benchmark = TASK_BENCHMARKS.get(task, "ifeval")
        model = self.model_id(recipe)
        evidence = []
        details = []
        # Keep the optional display within the Swift report's 16 x 2,000-byte
        # contract, including settings and the two qualification statements.
        matching = [value for value in self.cohorts if value.benchmark == benchmark and model in value.results]
        for cohort in sorted(matching, key=lambda value: (value is not selected, value.id))[:7]:
            number, revision = cohort.results[model]
            compared = cohort is selected and recipe.id in used
            score = format(number, "f")
            evidence.append(
                {
                    "benchmark": benchmark,
                    "cohort": cohort.id,
                    "label": cohort.label,
                    "metric": cohort.metric,
                    "score": score,
                    "model_id": model,
                    "evaluated_revision": revision,
                    "settings": cohort.settings,
                    "source": cohort.source,
                    "authority": cohort.authority,
                    "evidence_kind": "upstream_reference",
                    "candidate_repo": recipe.repo,
                    "candidate_revision": recipe.revision,
                    "relationship": (
                        "reviewed_quantized_export_reference_only"
                        if (recipe.repo, recipe.revision) in self.bindings
                        else "model_repository_reference_only"
                    ),
                    "used_for_ranking": compared,
                }
            )
            details.append(
                f"{cohort.label}: {number:.2f}/100 ({cohort.metric}), published for {model}. "
                + ("Used to compare scored options. " if compared else "Reference only in this selection. ")
                + f"Source: {cohort.source['url']}"
            )
            details.append(cohort.settings)
        if evidence:
            details.append(
                "Published upstream results; this Stark export, quantization and tool workflow were not benchmarked."
            )
        else:
            label = {"livecodebench": "LiveCodeBench", "bfcl": "BFCL", "ifeval": "IFEval"}[benchmark]
            details.append(
                f"No comparable published {label} result is recorded for this source. Its task quality is unknown."
            )
        if benchmark == "ifeval":
            details.append(
                "IFEval measures instruction following; writing quality and factual accuracy need separate evaluation."
            )
        elif benchmark == "bfcl":
            details.append(
                "BFCL measures tool use under its own harness; "
                "it does not establish completion of file tasks in OpenWork."
            )
        return evidence, details


@lru_cache(maxsize=1)
def load_registry():
    try:
        path = Path(__file__).with_name("recommendation_benchmarks.json")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Oversized benchmark registry")
        return BenchmarkRegistry(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # A bad optional score snapshot must not disable resource-safe setup.
        return BenchmarkRegistry({"schema_version": 1, "cohorts": []})
