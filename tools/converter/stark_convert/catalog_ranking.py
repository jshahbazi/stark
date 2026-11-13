"""Reference scores that order a family catalogue.

Scores come from the bundled whatmodel.sh snapshot and describe parent models.
A quantized export may show its parent's score for ordering only; no export is
benchmarked here and a score never grants compatibility or a download.
"""

from __future__ import annotations

from functools import lru_cache

from .recommendation_benchmarks import load_registry
from .recommendation_leaderboard import load_snapshot

TASKS = ("general", "coding")
# How a catalogue row relates to the model the score was published for.
BENCHMARKED_MODEL = "benchmarked_model"
REVIEWED_EXPORT = "reviewed_export"
DECLARED_QUANTIZATION = "declared_quantization"
NAMED_QUANTIZATION = "named_quantization"
RELATIONSHIPS = (BENCHMARKED_MODEL, REVIEWED_EXPORT, DECLARED_QUANTIZATION, NAMED_QUANTIZATION)
_DECLARED = "base_model:quantized:"
_EXPORT_SUFFIXES = ("-mlx-4bit", "-mlx-8bit", "-4bit", "-8bit", "-4-bit", "-8-bit")


class ReferenceScores:
    def __init__(self, snapshot, bindings):
        self.snapshot = snapshot
        self.captured = snapshot.captured
        self.bindings = dict(bindings)
        # A parent's score is the same in every RAM list that contains it, so
        # the largest list wins and smaller ones only add missing parents.
        self.rows = {}
        for task, _ram in sorted(snapshot.buckets, key=lambda key: -key[1]):
            for repo, row in snapshot.buckets[task, _ram][1].items():
                self.rows.setdefault(repo, {}).setdefault(task, row)
        self.names = {}
        for repo in self.rows:
            self.names.setdefault(repo.split("/", 1)[1].casefold(), set()).add(repo)

    def _named(self, repo, base_name):
        """Find the one scored model whose name an export's name extends."""
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
            # A fine-tune, merge or re-upload has its own weights and no score.
            return None, None
        declared = {tag[len(_DECLARED) :] for tag in tags if isinstance(tag, str) and tag.startswith(_DECLARED)}
        if len(declared) == 1 and next(iter(declared)) in self.rows:
            return next(iter(declared)), DECLARED_QUANTIZATION
        named = self._named(repo, base_name)
        return (named, NAMED_QUANTIZATION) if named is not None else (None, None)

    def reference(self, repo, revision, quantization, tags, base_name):
        """Return the reference score record for a checked catalogue row, or ``None``."""
        tags = tags if isinstance(tags, (list, tuple)) else ()
        parent, relationship = self._parent(repo, revision, tags, base_name, quantization != "BF16")
        rows = self.rows.get(parent)
        if not rows:
            return None
        row = rows.get("general") or rows["coding"]
        return {
            "model_id": parent,
            "relationship": relationship,
            **{task: rows[task]["score"] if task in rows else None for task in TASKS},
            "source_url": "https://whatmodel.sh/model/" + row["id"],
            "captured": self.captured,
        }

    def priority(self, repo, tags, physical_memory=None):
        """Order an unchecked listing entry: ``(fits a host-sized list, score)`` or ``None``.

        The export format is unknown before inspection, so common export
        suffixes stand in for it here. Only ordering depends on this guess; a
        displayed score is decided after inspection by :meth:`reference`.
        """
        tags = tags if isinstance(tags, (list, tuple)) else ()
        name = repo.split("/", 1)[1]
        folded = name.casefold()
        base_name = next((name[: -len(suffix)] for suffix in _EXPORT_SUFFIXES if folded.endswith(suffix)), None)
        parent, _relationship = self._parent(repo, None, tags, base_name, True)
        rows = self.rows.get(parent)
        if not rows:
            return None
        score = (rows.get("general") or rows["coding"])["score"]
        _bucket, hosted = self.snapshot.for_host("general", physical_memory)
        return parent in hosted, score


@lru_cache(maxsize=1)
def reference_scores():
    try:
        bindings = load_registry().bindings
    except Exception:
        # Reviewed bindings are optional evidence; their absence only removes scores.
        bindings = {}
    return ReferenceScores(load_snapshot(), bindings)


def host_memory():
    """Physical memory in bytes, or ``None`` when it cannot be read."""
    try:
        import psutil

        total = psutil.virtual_memory().total
        return total if type(total) is int and total > 0 else None
    except Exception:
        return None
