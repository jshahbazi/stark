"""Model-free checks of the production Swift controller's decision lifecycle.

Run with `uv run --project tools/converter --locked python
scripts/test_menubar_controller.py` on an Apple silicon Mac with Xcode tools.
The temporary executable includes the actual controller source plus a same-file
test extension. Compiler/runtime output is suppressed; only counts are printed.
"""

from __future__ import annotations

import io
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "apps/menu-bar/Sources/StarkApp.swift"
BOUNDARY = "\nstruct StatusLine: View"
DRIVER = r'''

extension EngineController {
    static func exercise(_ scenario: String, backendError: String) -> Bool {
        let engine = EngineController()
        let pipe = Pipe()
        engine.input = pipe.fileHandleForWriting
        defer {
            try? pipe.fileHandleForWriting.close()
            try? pipe.fileHandleForReading.close()
        }
        engine.mode = .decision
        engine.loadedMode = .decision
        engine.selectedID = "fixture-model"
        engine.loadedID = "fixture-model"
        engine.state = "ready"
        let answers = ["First", "Second"]
        let event: [String: Any] = [
            "event": "decision", "answers": answers,
            "probabilities": [0.1, 0.9], "choice_index": 1
        ]
        switch scenario {
        case "unicode-validation-error":
            engine.decisionText = "synthetic context"
            engine.decisionQuestion = "synthetic question"
            engine.decisionAnswers = "straße, STRASSE"
            guard engine.decisionValidation == nil else { return false }
            engine.decide()
            guard engine.state == "deciding", engine.pendingDecisionAnswers != nil else { return false }
            engine.apply(["event": "error", "message": backendError])
            return engine.modelReady && engine.state == "ready" &&
                engine.pendingDecisionAnswers == nil && engine.decisionResult == nil &&
                engine.errorMessage == backendError
        case "unloaded-validation-error":
            engine.loadedID = ""
            engine.loadedMode = nil
            engine.state = "deciding"
            engine.pendingDecisionAnswers = answers
            engine.decisionResult = DecisionResult(answers: answers, probabilities: [0.1, 0.9], choiceIndex: 1)
            engine.apply(["event": "error", "message": "Use distinct answers."])
            return engine.state == "idle" && !engine.busy && !engine.modelReady &&
                engine.pendingDecisionAnswers == nil && engine.decisionResult == nil
        case "valid-decision":
            engine.state = "deciding"
            engine.pendingDecisionAnswers = answers
            engine.apply(event)
            guard engine.decisionResult?.choice == "Second", engine.pendingDecisionAnswers == nil,
                  engine.state == "deciding" else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            return engine.modelReady && engine.decisionResult?.answers == answers
        case "stop-clears-decision":
            engine.state = "deciding"
            engine.pendingDecisionAnswers = answers
            engine.decisionResult = DecisionResult(answers: answers, probabilities: [0.1, 0.9], choiceIndex: 1)
            engine.stop()
            guard engine.state == "stopping", engine.pendingDecisionAnswers == nil,
                  engine.decisionResult == nil else { return false }
            engine.apply(event)
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return engine.state == "idle" && engine.loadedID.isEmpty && engine.loadedMode == nil &&
                engine.pendingDecisionAnswers == nil && engine.decisionResult == nil
        case "ready-mode-guard":
            engine.loadedID = ""
            engine.loadedMode = nil
            engine.state = "loading"
            engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model"])
            guard engine.state == "loading", engine.loadedMode == nil else { return false }
            engine.apply(["event": "ready", "mode": "decision", "model_id": "different-fixture"])
            guard engine.state == "loading", engine.loadedMode == nil else { return false }
            engine.apply(["event": "ready", "mode": "decision", "model_id": "fixture-model"])
            guard engine.modelReady else { return false }
            engine.state = "stopping"
            engine.apply(["event": "ready", "mode": "decision", "model_id": "fixture-model"])
            return engine.state == "stopping" && !engine.modelReady
        case "wrong-mode-decision":
            engine.mode = .chat
            engine.loadedMode = .chat
            engine.state = "deciding"
            engine.pendingDecisionAnswers = answers
            engine.apply(event)
            return engine.decisionResult == nil
        case "invalid-decision":
            engine.state = "deciding"
            engine.pendingDecisionAnswers = answers
            engine.apply([
                "event": "decision", "answers": answers,
                "probabilities": [0.1, 0.9], "choice_index": 0
            ])
            return engine.state == "stopping" && engine.pendingDecisionAnswers == nil &&
                engine.decisionResult == nil && engine.errorMessage != nil
        default:
            return false
        }
    }
}

let scenario = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ""
let backendError = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : ""
exit(EngineController.exercise(scenario, backendError: backendError) ? 0 : 1)
'''


@unittest.skipUnless(sys.platform == "darwin" and platform.machine() == "arm64", "Apple silicon macOS required")
class ControllerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stark-controller-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        source = SOURCE.read_text()
        if source.count(BOUNDARY) != 1:
            raise RuntimeError("controller_source_boundary_invalid")
        test_source = directory / "main.swift"
        test_source.write_text(source.split(BOUNDARY, 1)[0] + DRIVER)
        cls.executable = directory / "controller-checks"
        try:
            compiled = subprocess.run(
                ["xcrun", "swiftc", "-target", "arm64-apple-macosx26.4", str(test_source), "-o", str(cls.executable)],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError("controller_compilation_unavailable") from None
        if compiled.returncode:
            raise RuntimeError("controller_compilation_failed")

    def exercise(self, scenario: str, backend_error: str = ""):
        try:
            result = subprocess.run(
                [str(self.executable), scenario, backend_error],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            self.fail("controller_check_unavailable")
        self.assertEqual(result.returncode, 0, "controller_check_failed")

    def test_unicode_backend_validation_error_settles_ready(self):
        from stark_convert.decision import DecisionInputError, validate_inputs

        try:
            validate_inputs("synthetic context", "synthetic question", ["straße", "STRASSE"])
        except DecisionInputError as error:
            backend_error = str(error)
        else:
            self.fail("unicode_validation_fixture_no_longer_rejected")
        self.exercise("unicode-validation-error", backend_error)

    def test_unloaded_decision_error_settles_idle(self):
        self.exercise("unloaded-validation-error")

    def test_valid_decision_is_retained_when_ready_status_arrives(self):
        self.exercise("valid-decision")

    def test_stop_clears_pending_and_late_decision(self):
        self.exercise("stop-clears-decision")

    def test_ready_requires_requested_model_and_mode(self):
        self.exercise("ready-mode-guard")

    def test_chat_mode_ignores_decision_event(self):
        self.exercise("wrong-mode-decision")

    def test_invalid_decision_is_stopped_and_discarded(self):
        self.exercise("invalid-decision")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControllerLifecycleTests)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(f"controller_checks: run={result.testsRun} failed={failed} skipped={len(result.skipped)}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
