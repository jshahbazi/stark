import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import validate_colgrep_generation as validation


class ValidationHelpersTest(unittest.TestCase):
    def test_assistant_history_strips_reasoning_before_tool_calls(self):
        text = '<think>private reasoning</think>\nRead the file.\n<function name="terminal"></function>'
        self.assertEqual(validation.assistant_content(text), "Read the file.")
        self.assertEqual(validation.assistant_content('<function name="finish"></function>'), "")

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

    def test_stops_match_requires_the_reference_eos_and_eos_finish(self):
        self.assertTrue(validation.stops_match(1, {"finish_reason": "eos", "stop_token_id": 1}))
        self.assertFalse(validation.stops_match(None, {"finish_reason": "length", "stop_token_id": None}))
        self.assertFalse(validation.stops_match(1, {"finish_reason": "eos", "stop_token_id": 130073}))
        self.assertFalse(validation.stops_match(130073, {"finish_reason": "eos", "stop_token_id": 1}))

    def test_source_seal_rejects_missing_or_invalid_etag_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "model.safetensors").write_bytes(b"weights")
            metadata = source / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            path = metadata / "model.safetensors.metadata"
            for identity in [
                validation.REVISION + "\nnot-an-etag\n",
                validation.REVISION + "\n" + "0" * 39 + "\n",
                validation.REVISION + "\n" + "0" * 65 + "\n",
                validation.REVISION + "\n",
            ]:
                path.write_text(identity)
                with self.subTest(identity=identity):
                    with self.assertRaisesRegex(RuntimeError, "source_metadata_invalid"):
                        validation.seal_source(source)
            path.unlink()
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                validation.seal_source(source)

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
