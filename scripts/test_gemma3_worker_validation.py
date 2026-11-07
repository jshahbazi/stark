"""Preserve completed qualification evidence when a replay path is reused."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gemma3_image_worker_validation as validation


class WorkerValidationOutputTests(unittest.TestCase):
    def test_existing_completed_and_failure_receipts_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, fixture, output = (root / name for name in ("model", "fixture", "output"))
            for directory in (model, fixture, output):
                directory.mkdir()
            worker = root / "worker"
            worker.write_bytes(b"not an executable")
            (output / "receipt.json").write_bytes(b'{"passed":true}')
            (output / "failure.json").write_bytes(b'{"preserved":"earlier failure"}')
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            arguments = ["validation", "--worker", str(worker), "--model", str(model),
                         "--fixture", str(fixture), "--output", str(output)]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validation.main(), 1)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
