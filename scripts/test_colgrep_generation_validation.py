import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import validate_colgrep_generation as validation


class ValidationHelpersTest(unittest.TestCase):
    def test_calls_accepts_cdata_and_xml_unescapes_values(self):
        parsed = validation.calls(
            '<function name="terminal"><param name="command">'
            '<![CDATA[grep "a<b" src]]></param></function>'
            '<function name="colgrep"><param name="query">a &amp; b</param></function>'
        )
        self.assertEqual(parsed[0][1]["command"], 'grep "a<b" src')
        self.assertEqual(parsed[1][1]["query"], "a & b")

    def test_calls_rejects_malformed_unknown_and_missing_required_tools(self):
        cases = [
            '<function name="unknown"><param name="x">y</param></function>',
            '<function name="terminal"><param name="command">ls</param><param name="unexpected">x</param></function>',
            '<function name="terminal"></function>',
            '<function name="colgrep"><param name="k">5</param></function>',
            '<function name="finish"><param name="locations">[]</param></function>',
            '<function name="terminal"><param name="command">&bad;</param></function>',
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(validation.calls(text), [])

    def test_finish_requires_calculation_and_application_without_broad_range(self):
        self.assertTrue(
            validation.finish_valid([("finish", {"locations": "['src/retry.py:7-7', 'src/retry.py:16-16']"})])
        )
        self.assertFalse(validation.finish_valid([("finish", {"locations": "['src/retry.py:7-7']"})]))
        self.assertFalse(validation.finish_valid([("finish", {"locations": "['src/retry.py:1-16']"})]))

    def test_source_seal_checks_revision_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            payload = b"weights"
            (source / "model.safetensors").write_bytes(payload)
            metadata = source / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            digest = validation.digest_bytes(payload)
            (metadata / "model.safetensors.metadata").write_text(validation.REVISION + "\n" + digest + "\n")
            self.assertEqual(validation.seal_source(source)["model.safetensors"], digest)
            (metadata / "model.safetensors.metadata").write_text("0" * 40 + "\n" + digest + "\n")
            with self.assertRaisesRegex(RuntimeError, "source_revision_mismatch"):
                validation.seal_source(source)

    def test_malformed_ready_worker_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "worker.py"
            executable.write_text(
                "#!/usr/bin/env python3\nimport json\nprint(json.dumps({'ready': False}), flush=True)\n"
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            with self.assertRaisesRegex(RuntimeError, "worker_not_ready"):
                validation.Worker(executable, root / "model")
            self.assertEqual(
                os.system("ps -axo command= | grep -F 'worker.py' | grep -v grep >/dev/null"),
                256,
            )


if __name__ == "__main__":
    unittest.main()
