"""Model-free regression checks for the native hybrid worker harness."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("test_hybrid_workers.py")
SPEC = importlib.util.spec_from_file_location("hybrid_workers_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hybrid_workers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid_workers)


class HybridWorkerHarnessTests(unittest.TestCase):
    def test_raw_qwen_requires_uzu_conversion_before_other_fixture_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "fixture"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

            with self.assertRaises(hybrid_workers.CheckFailed) as raised:
                hybrid_workers.prepare_fixture(source, source / "prepared")

            self.assertEqual(str(raised.exception), "fixture_requires_uzu_conversion")


if __name__ == "__main__":
    unittest.main()
