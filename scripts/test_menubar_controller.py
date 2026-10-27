"""Model-free checks of production Swift decisions and image attachment ownership.

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
DRIVER = r"""

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
        let imageBytes = Data([137, 80, 78, 71, 13, 10, 26, 10])
        let imageReady: [String: Any] = [
            "event": "ready", "mode": "chat", "model_id": "fixture-model", "images": true,
            "image_attachment_transport": "private-file-v1", "max_image_bytes": 4 * 1024 * 1024
        ]
        if scenario.hasPrefix("image-") {
            engine.mode = .chat
            engine.loadedMode = .chat
            engine.apply(imageReady)
            engine.attachmentStaging = try! AttachmentStaging()
        }
        func command() -> [String: Any] {
            let bytes = pipe.fileHandleForReading.availableData
            return try! JSONSerialization.jsonObject(with: bytes) as! [String: Any]
        }
        func staged() -> [String] {
            try! FileManager.default.contentsOfDirectory(atPath: engine.attachmentStaging!.root.path)
        }
        func pump(_ complete: () -> Bool) -> Bool {
            let deadline = Date().addingTimeInterval(2)
            while !complete() && Date() < deadline {
                RunLoop.current.run(until: Date().addingTimeInterval(0.01))
            }
            return complete()
        }
        switch scenario {
        case "vision-install-opt-in":
            engine.install(source: "Qwen/Qwen3.5-0.8B", kind: "huggingface", artifactMode: "vision-language")
            let requested = command()
            return requested["command"] as? String == "install" &&
                requested["artifact_mode"] as? String == "vision-language"
        case "text-install-default":
            engine.install(source: "Qwen/Qwen3-0.6B", kind: "huggingface")
            return command()["artifact_mode"] == nil
        case "image-capability":
            guard engine.canAttachImage else { return false }
            engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model", "images": true])
            guard !engine.supportsImages, !engine.canAttachImage else { return false }
            engine.apply(imageReady)
            engine.selectedID = "different-fixture"
            return engine.supportsImages && !engine.canAttachImage
        case "image-bounds":
            for bytes in [Data(), Data([0, 1, 2]), imageBytes + Data(count: ImageAttachment.maximumBytes)] {
                do { try engine.attachImage(bytes: bytes); return false } catch {}
            }
            do { try engine.attachImage(bytes: imageBytes + Data(count: ImageAttachment.maximumBytes - imageBytes.count)) }
            catch { return false }
            return engine.draftImage?.bytes.count == ImageAttachment.maximumBytes
        case "image-selection-fence":
            let selection = engine.attachmentSelection
            engine.attachmentLoading = true
            try! engine.attachImage(bytes: imageBytes)
            guard engine.attachmentSelection != selection, !engine.attachmentLoading else { return false }
            let beforeGenerate = engine.attachmentSelection
            engine.draft = "synthetic question"
            engine.generate()
            _ = command()
            return engine.attachmentSelection != beforeGenerate
        case "image-provider-paste":
            let file = engine.attachmentStaging!.root.appendingPathComponent("fixture.png")
            try! imageBytes.write(to: file)
            defer { try? FileManager.default.removeItem(at: file) }
            let provider = NSItemProvider()
            provider.registerFileRepresentation(forTypeIdentifier: UTType.png.identifier, fileOptions: [], visibility: .all) { completion in
                completion(file, false, nil)
                return nil
            }
            guard engine.acceptImageProviders([provider]), engine.attachmentLoading, !engine.canSendMessage else { return false }
            return pump { !engine.attachmentLoading } && engine.draftImage?.bytes == imageBytes
        case "image-provider-drop":
            let file = engine.attachmentStaging!.root.appendingPathComponent("fixture.png")
            try! imageBytes.write(to: file)
            defer { try? FileManager.default.removeItem(at: file) }
            let provider = NSItemProvider()
            provider.registerDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier, visibility: .all) { completion in
                completion(file.dataRepresentation, nil)
                return nil
            }
            guard engine.acceptImageProviders([provider]) else { return false }
            return pump { !engine.attachmentLoading } && engine.draftImage?.bytes == imageBytes
        case "image-provider-cancel":
            let file = engine.attachmentStaging!.root.appendingPathComponent("fixture.png")
            try! imageBytes.write(to: file)
            defer { try? FileManager.default.removeItem(at: file) }
            var callback: ((URL?, Bool, Error?) -> Void)?
            let provider = NSItemProvider()
            provider.registerFileRepresentation(forTypeIdentifier: UTType.png.identifier, fileOptions: [], visibility: .all) { completion in
                callback = completion
                return nil
            }
            guard engine.acceptImageProviders([provider]), pump({ callback != nil }) else { return false }
            engine.removeImage()
            callback?(file, false, nil)
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
            return !engine.attachmentLoading && engine.draftImage == nil
        case "image-only-send":
            try! engine.attachImage(bytes: imageBytes)
            guard engine.canSendMessage, engine.draft.isEmpty else { return false }
            engine.generate()
            let request = command()
            return request["image"] is [String: Any] && request["text"] as? String == "" && engine.messages.count == 1
        case "image-file-boundaries":
            let root = engine.attachmentStaging!.root
            let file = root.appendingPathComponent("fixture")
            let link = root.appendingPathComponent("link")
            try! imageBytes.write(to: file)
            try! FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
            defer { try? FileManager.default.removeItem(at: file); try? FileManager.default.removeItem(at: link) }
            do { _ = try ImageAttachment.read(link); return false } catch {}
            do { _ = try ImageAttachment.read(root); return false } catch {}
            guard (try? ImageAttachment.read(file).bytes) == imageBytes else { return false }
            try! (imageBytes + Data(count: ImageAttachment.maximumBytes)).write(to: file)
            do { _ = try ImageAttachment.read(file); return false } catch { return true }
        case "image-stage-permissions":
            let store = engine.attachmentStaging!
            let metadata = try! store.stage(ImageAttachment(bytes: imageBytes))
            guard let id = metadata["id"] as? String, id.count == 32,
                  id.allSatisfy({ "0123456789abcdef".contains($0) }),
                  metadata["media_type"] as? String == "image/png",
                  metadata["byte_length"] as? Int == imageBytes.count else { return false }
            let rootMode = try! FileManager.default.attributesOfItem(atPath: store.root.path)[.posixPermissions] as! Int
            let fileMode = try! FileManager.default.attributesOfItem(atPath: store.root.appendingPathComponent(id).path)[.posixPermissions] as! Int
            store.clear()
            return rootMode == 0o700 && fileMode == 0o600 && staged().isEmpty
        case "image-stage-pinned-root":
            let store = engine.attachmentStaging!
            let moved = store.root.appendingPathExtension("moved")
            let foreign = store.root.appendingPathExtension("foreign")
            try! FileManager.default.moveItem(at: store.root, to: moved)
            try! FileManager.default.createDirectory(at: foreign, withIntermediateDirectories: false)
            try! FileManager.default.createSymbolicLink(at: store.root, withDestinationURL: foreign)
            defer {
                try? FileManager.default.removeItem(at: store.root)
                try? FileManager.default.removeItem(at: moved)
                try? FileManager.default.removeItem(at: foreign)
            }
            _ = try! store.stage(ImageAttachment(bytes: imageBytes))
            guard try! FileManager.default.contentsOfDirectory(atPath: foreign.path).isEmpty,
                  try! FileManager.default.contentsOfDirectory(atPath: moved.path).count == 1 else { return false }
            store.clear()
            return try! FileManager.default.contentsOfDirectory(atPath: moved.path).isEmpty
        case "image-generate-followup":
            try! engine.attachImage(bytes: imageBytes)
            engine.draft = "synthetic question"
            engine.generate()
            let initial = command()
            guard initial["image"] is [String: Any], initial.keys.sorted() == ["command", "image", "text"],
                  engine.messages.count == 1, engine.messages[0].image != nil,
                  engine.draftImage == nil, staged().count == 1 else { return false }
            engine.apply(["event": "image_progress", "phase": "preparing"])
            guard engine.imageProgress == "Preparing image…" else { return false }
            engine.apply(["event": "reply", "text": "synthetic answer", "finish_reason": "eos"])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.conversationImage != nil, !engine.canAttachImage, staged().isEmpty else { return false }
            engine.draft = "synthetic followup"
            engine.generate()
            let followup = command()
            return followup["image"] is [String: Any] && engine.messages.last?.image == nil && staged().count == 1
        case "image-error-retry":
            try! engine.attachImage(bytes: imageBytes)
            engine.draft = "synthetic question"
            engine.generate()
            _ = command()
            engine.apply(["event": "error", "message": "Image rejected"])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.draftImage != nil, engine.conversationImage == nil, engine.messages.isEmpty,
                  engine.draft == "synthetic question", engine.canAttachImage, staged().isEmpty else { return false }
            engine.generate()
            return command()["image"] is [String: Any] && engine.messages.count == 1
        case "image-reset-cleanup":
            try! engine.attachImage(bytes: imageBytes)
            engine.reset()
            guard command()["command"] as? String == "reset" else { return false }
            return engine.draftImage == nil && engine.conversationImage == nil && !engine.hasConversation && staged().isEmpty
        case "image-stop-cleanup":
            try! engine.attachImage(bytes: imageBytes)
            engine.draft = "synthetic question"
            engine.generate()
            _ = command()
            engine.stop()
            _ = command()
            engine.apply(["event": "reply", "text": "stale synthetic reply"])
            return engine.draftImage == nil && engine.conversationImage == nil && engine.messages.isEmpty &&
                !engine.supportsImages && staged().isEmpty && engine.imageProgress.isEmpty
        case "image-backend-error-cleanup":
            try! engine.attachImage(bytes: imageBytes)
            engine.draft = "synthetic question"
            engine.generate()
            _ = command()
            engine.apply(["event": "status", "state": "error", "message": "Runtime stopped"])
            return engine.draftImage == nil && engine.conversationImage == nil && engine.messages.isEmpty &&
                !engine.supportsImages && staged().isEmpty
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
"""


@unittest.skipUnless(
    sys.platform == "darwin" and platform.machine() == "arm64",
    "Apple silicon macOS required",
)
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
                [
                    "xcrun",
                    "swiftc",
                    "-target",
                    "arm64-apple-macosx26.4",
                    str(test_source),
                    "-o",
                    str(cls.executable),
                ],
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
            validate_inputs(
                "synthetic context", "synthetic question", ["straße", "STRASSE"]
            )
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

    def test_image_capability_requires_complete_negotiation(self):
        self.exercise("image-capability")

    def test_vision_install_is_explicit_opt_in(self):
        self.exercise("vision-install-opt-in")

    def test_existing_install_keeps_default_mode(self):
        self.exercise("text-install-default")

    def test_image_byte_and_signature_bounds(self):
        self.exercise("image-bounds")

    def test_image_selection_and_generate_fence_old_provider_callbacks(self):
        self.exercise("image-selection-fence")

    def test_png_paste_provider_uses_original_bytes(self):
        self.exercise("image-provider-paste")

    def test_file_url_drop_provider_uses_bounded_file_read(self):
        self.exercise("image-provider-drop")

    def test_cancelled_provider_cannot_restore_attachment(self):
        self.exercise("image-provider-cancel")

    def test_image_only_send_is_supported(self):
        self.exercise("image-only-send")

    def test_image_files_reject_symlinks_directories_and_oversize(self):
        self.exercise("image-file-boundaries")

    def test_image_stage_permissions_and_cleanup(self):
        self.exercise("image-stage-permissions")

    def test_image_stage_pins_directory_against_path_replacement(self):
        self.exercise("image-stage-pinned-root")

    def test_image_conversation_resupplies_bytes_using_small_metadata(self):
        self.exercise("image-generate-followup")

    def test_image_error_restores_attachment_and_retries(self):
        self.exercise("image-error-retry")

    def test_image_reset_releases_attachment(self):
        self.exercise("image-reset-cleanup")

    def test_image_stop_releases_attachment_and_ignores_late_reply(self):
        self.exercise("image-stop-cleanup")

    def test_image_backend_error_releases_attachment(self):
        self.exercise("image-backend-error-cleanup")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControllerLifecycleTests)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(
        f"controller_checks: run={result.testsRun} failed={failed} skipped={len(result.skipped)}"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
