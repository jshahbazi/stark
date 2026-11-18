"""Reference scores that order a family catalogue.

Browsing uses the bundled llmbench coding score, while guided planning uses the
selected task's coding or non-coding score. A quantized export may show its
parent's score for ordering only; no export is benchmarked here and a score
never grants compatibility or a download.
"""

from __future__ import annotations

from functools import lru_cache

from .recommendation_benchmarks import load_registry
from .recommendation_leaderboard import load_snapshot

# How a catalogue row relates to the model the score was published for.
BENCHMARKED_MODEL = "benchmarked_model"
REVIEWED_EXPORT = "reviewed_export"
DECLARED_QUANTIZATION = "declared_quantization"
NAMED_QUANTIZATION = "named_quantization"
RELATIONSHIPS = (BENCHMARKED_MODEL, REVIEWED_EXPORT, DECLARED_QUANTIZATION, NAMED_QUANTIZATION)
_DECLARED = "base_model:quantized:"
_EXPORT_SUFFIXES = ("-mlx-4bit", "-mlx-8bit", "-4bit", "-8bit", "-4-bit", "-8-bit")


class CatalogReferenceScores:
    def __init__(self, snapshot, bindings):
        self.captured = snapshot.captured
        self.source_url = snapshot.source_url
        self.bindings = dict(bindings)
        self.rows = snapshot.rows
        self.names = {}
        for repo in self.rows:
            self.names.setdefault(repo.split("/", 1)[1].casefold(), set()).add(repo)

    def _named(self, repo, base_name):
        owner, name = repo.split("/", 1)
        if not isinstance(base_name, str) or not base_name or base_name.casefold() == name.casefold():
            return None
        matches = self.names.get(base_name.casefold(), set())
        same_owner = owner + "/" + base_name
        if same_owner in matches:
            return same_owner
        return next(iter(matches)) if len(matches) == 1 else None

    def _parent(self, repo, revision, tags, base_name, quantized):
        bound = self.bindings.get((repo, revision))
        if bound is not None:
            return bound, REVIEWED_EXPORT
        if repo in self.rows:
            return repo, BENCHMARKED_MODEL
        if not quantized:
            return None, None
        declared = {tag[len(_DECLARED) :] for tag in tags if isinstance(tag, str) and tag.startswith(_DECLARED)}
        if len(declared) == 1 and next(iter(declared)) in self.rows:
            return next(iter(declared)), DECLARED_QUANTIZATION
        named = self._named(repo, base_name)
        return (named, NAMED_QUANTIZATION) if named is not None else (None, None)

    def reference(self, repo, revision, quantization, tags, base_name):
        tags = tags if isinstance(tags, (list, tuple)) else ()
        parent, relationship = self._parent(repo, revision, tags, base_name, quantization != "BF16")
        rows = self.rows.get(parent)
        if not rows or "coding" not in rows:
            return None
        return {
            "model_id": parent,
            "relationship": relationship,
            "general": None,
            "coding": rows["coding"]["score"],
            "source_url": self.source_url,
            "captured": self.captured,
        }

    def priority(self, repo, tags, physical_memory=None, *, task=None, revision=None, exact_first=False):
        tags = tags if isinstance(tags, (list, tuple)) else ()
        name = repo.split("/", 1)[1]
        folded = name.casefold()
        base_name = next((name[: -len(suffix)] for suffix in _EXPORT_SUFFIXES if folded.endswith(suffix)), None)
        parent, relationship = self._parent(repo, revision, tags, base_name, True)
        selected_task = "coding" if task is None else task
        if selected_task not in {"general", "coding"} or parent not in self.rows:
            return None
        row = self.rows[parent].get(selected_task)
        if row is None:
            return None
        score = row["score"]
        if exact_first:
            return relationship in {BENCHMARKED_MODEL, REVIEWED_EXPORT}, True, score
        return True, score


@lru_cache(maxsize=1)
def catalog_reference_scores():
    try:
        bindings = load_registry().bindings
    except Exception:
        bindings = {}
    return CatalogReferenceScores(load_snapshot(), bindings)


def host_memory():
    """Physical memory in bytes, or ``None`` when it cannot be read."""
    try:
        import psutil

        total = psutil.virtual_memory().total
        return total if type(total) is int and total > 0 else None
    except Exception:
        return None
