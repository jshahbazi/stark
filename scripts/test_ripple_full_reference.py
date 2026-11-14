"""Regression checks for real split GGUF metadata in the Ripple oracle."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from ripple_full_reference import _ordered_shards


def shard(index: int, *, architecture: str | None = None):
    metadata = {"split.no": index, "split.count": 2, "split.tensors.count": 3}
    if architecture is not None:
        metadata["general.architecture"] = architecture
    return SimpleNamespace(
        metadata=metadata, tensors={str(i): None for i in range(2 if index == 0 else 1)}
    )


class SplitGGUFTests(unittest.TestCase):
    def test_metadata_only_on_shard_zero_and_unordered_files(self):
        first, second = shard(0, architecture="ricursive"), shard(1)
        self.assertEqual(_ordered_shards([second, first]), [first, second])

    def test_rejects_missing_shards_duplicate_indices_and_tensor_count_drift(self):
        for sources in (
            [shard(0, architecture="ricursive")],
            [shard(0, architecture="ricursive"), shard(0, architecture="ricursive")],
            [shard(0, architecture="ricursive"), shard(1)],
        ):
            if len(sources) == 2 and sources[1].metadata["split.no"] == 1:
                sources[1].metadata["split.tensors.count"] = 4
            with self.subTest(inventory=len(sources)), self.assertRaises(ValueError):
                _ordered_shards(sources)

    def test_rejects_conflicting_or_missing_primary_architecture(self):
        for first, second in (
            (shard(0), shard(1, architecture="ricursive")),
            (shard(0, architecture="ricursive"), shard(1, architecture="llama")),
        ):
            with self.assertRaises(ValueError):
                _ordered_shards([first, second])

    def test_unsplit_fixture_remains_supported(self):
        source = SimpleNamespace(
            metadata={"general.architecture": "ricursive"}, tensors={"weight": None}
        )
        self.assertEqual(_ordered_shards([source]), [source])


if __name__ == "__main__":
    unittest.main()
