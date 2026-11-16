"""Model-free Swift checks for Decider mode, image ownership and request fencing."""

from __future__ import annotations

import io
import platform
import sys
import unittest

import test_menubar_controller as controller

DECIDER_CASES = r'''
        case "decider-catalog-mode-and-family":
            var family = (familyRegistry()["families"] as! [[String:Any]])[0]
            family["family_id"] = "decider-flash-v1.3"; family["name"] = "Decider Flash v1.3"; family["model_kind"] = "decision"
            var modes = family["modes"] as! [String:[String:Any]]
            modes["text"]!["status"] = "unsupported"
            family["modes"] = modes
            var registry = familyRegistry(); registry["families"] = [family]
            guard let parsed = ModelFamilyRegistry(registry) else { return false }
            let inventory = ModelsInventory(models:[], loadedID:"", registry:parsed)
            guard inventory.families?.map(\.id) == ["decider-flash-v1.3"] else { return false }
            var item = catalogItem()
            item["artifact_modes"] = ["decision"]; item["default_artifact_mode"] = "decision"
            guard let catalog = FamilyModelCatalog(catalogReport("decider-flash-v1.3", items:[item])) else { return false }
            return catalog.items.first?.installationMode == .decision
        case "decider-ready-capabilities":
            var generic = imageReady
            generic["mode"] = "decision"
            engine.apply(generic)
            guard !engine.supportsImages, !engine.canAttachImage, !engine.decisionModel else { return false }
            prepareDecider()
            return engine.modelReady && engine.decisionModel && engine.decisionCalibrated && engine.supportsImages &&
                engine.canAttachImage && !engine.supportsTools && !engine.supportsMTP && !engine.canSendMessage
        case "decider-image-only-request":
            prepareDecider()
            try! engine.attachImage(bytes: imageBytes)
            guard engine.decisionValidation == nil, engine.decisionImage != nil, engine.draftImage == nil,
                  engine.conversationImage == nil else { return false }
            engine.decide()
            let request = command()
            guard request["command"] as? String == "decide", request["text"] as? String == "",
                  request["question"] as? String == "Choose a label", request["answers"] as? [String] == answers,
                  let image = request["image"] as? [String: Any], let id = image["id"] as? String,
                  let requestID = request["request_id"] as? String, UUID(uuidString: requestID) != nil,
                  image["byte_length"] as? Int == imageBytes.count, image["media_type"] as? String == "image/png",
                  image.keys.sorted() == ["byte_length", "id", "media_type"], staged().count == 1,
                  engine.messages.isEmpty, engine.state == "deciding" else { return false }
            let file = engine.attachmentStaging!.root.appendingPathComponent(id)
            guard (try! Data(contentsOf: file)) == imageBytes else { return false }
            engine.apply(["event":"decision", "request_id":requestID, "answers":answers,
                          "probabilities":[0.1, 0.9], "choice_index":1])
            engine.apply(["event":"status", "state":"ready", "message":"Ready"])
            guard engine.decisionResult?.choice == "Second", engine.decisionCalibrated,
                  engine.decisionImage != nil, staged().isEmpty, engine.messages.isEmpty else { return false }
            engine.clearDecision()
            return engine.decisionResult == nil && engine.decisionImage != nil && engine.modelReady
        case "decider-request-error-fence":
            prepareDecider()
            try! engine.attachImage(bytes: imageBytes)
            engine.decide()
            let request = command(), requestID = request["request_id"] as! String
            engine.apply(["event":"error", "request_id":UUID().uuidString, "message":"Stale failure"])
            guard engine.state == "deciding", engine.decisionImage != nil, staged().count == 1 else { return false }
            engine.apply(["event":"image_progress", "request_id":UUID().uuidString, "phase":"deciding"])
            guard engine.imageProgress == "Preparing image…" else { return false }
            engine.apply(["event":"image_progress", "request_id":requestID, "phase":"deciding"])
            guard engine.imageProgress == "Evaluating image…" else { return false }
            engine.apply(["event":"error", "request_id":requestID, "message":"Image rejected"])
            guard engine.modelReady, engine.decisionImage == nil, engine.pendingDecisionID == nil,
                  engine.pendingDecisionAnswers == nil, staged().isEmpty, engine.imageProgress.isEmpty else { return false }
            engine.apply(["event":"decision", "request_id":requestID, "answers":answers,
                          "probabilities":[0.1, 0.9], "choice_index":1])
            return engine.decisionResult == nil
        case "decider-attachment-lifecycle":
            for action in ["reset", "stop", "model", "error", "remove"] {
                prepareDecider()
                try! engine.attachImage(bytes: imageBytes)
                _ = try! engine.attachmentStaging!.stage(engine.decisionImage!)
                if action == "reset" { engine.reset(); _ = command() }
                if action == "stop" { engine.stop(); _ = command() }
                if action == "model" { engine.selectedID = "another-model" }
                if action == "error" { engine.apply(["event":"error", "message":"Rejected input"] ) }
                if action == "remove" {
                    let identifier = UUID().uuidString
                    engine.removalRequestID = identifier; engine.removingModelID = "fixture-model"
                    engine.removalAction = .unregister
                    engine.apply(["event":"model_removed", "request_id":identifier,
                                  "model_id":"fixture-model", "action":"unregister", "stopped_model_id":"fixture-model"])
                }
                guard engine.decisionImage == nil, staged().isEmpty, !engine.attachmentLoading,
                      engine.pendingDecisionAnswers == nil else { return false }
            }
            return true
        case "decider-stop-fences-reply":
            prepareDecider()
            try! engine.attachImage(bytes: imageBytes)
            engine.decide()
            let request = command()
            engine.stop(); _ = command()
            engine.apply(["event":"decision", "request_id":request["request_id"]!, "answers":answers,
                          "probabilities":[0.1, 0.9], "choice_index":1])
            return engine.state == "stopping" && engine.decisionResult == nil && engine.decisionImage == nil &&
                staged().isEmpty && engine.pendingDecisionID == nil && !engine.attachmentLoading
        case "decider-mode-routing":
            engine.models = [InstalledModel(id:"decider", name:"Decider Flash v1.3", path:"/private/model", apiID:"decider", artifactMode:.decision)]
            engine.mode = .chat; engine.loadedID = ""; engine.loadedMode = nil; engine.state = "idle"
            guard engine.runModel(id:"decider") else { return false }
            let run = command()
            guard run["mode"] as? String == "decision", engine.mode == .decision else { return false }
            engine.state = "ready"
            engine.changeMode(.chat)
            guard engine.mode == .decision, engine.errorMessage != nil, noCommandSent() else { return false }
            let item = ModelsInventory(models:engine.models, loadedID:"", registry:nil).models[0]
            return item.detailLabels.contains("Decision and images") && !item.detailLabels.contains("Tools unchecked")
        case "decider-unavailable-image-and-text-validation":
            prepareDecider()
            var ready = deciderReady
            ready["decision_images"] = false
            engine.apply(ready)
            guard !engine.canAttachImage, !engine.supportsImages, engine.decisionValidation != nil else { return false }
            engine.decisionText = "Some input"
            guard engine.decisionValidation == nil else { return false }
            engine.decide()
            return command()["image"] == nil && engine.state == "deciding"
'''

DECIDER_SETUP = r'''
        let deciderReady: [String: Any] = [
            "event":"ready", "mode":"decision", "model_id":"fixture-model", "decision_model":true,
            "decision_images":true, "images":true, "decision_calibration":"checkpoint_temperature",
            "image_attachment_transport":"private-file-v1", "max_image_bytes":4 * 1024 * 1024,
            "tools":true, "mtp":["supported":true],
            "image_limits":["max_source_pixels":16_777_216, "max_processed_pixels":1_638_400, "max_image_tokens":1600]
        ]
        func prepareDecider() {
            engine.models = [InstalledModel(id:"fixture-model", name:"Decider Flash", path:"/private/model", apiID:"decider", artifactMode:.decision)]
            engine.mode = .decision; engine.selectedID = "fixture-model"; engine.loadedID = "fixture-model"
            engine.loadedMode = .decision; engine.state = "ready"
            engine.apply(deciderReady)
            engine.attachmentStaging = try! AttachmentStaging()
            engine.decisionText = ""; engine.decisionQuestion = "Choose a label"; engine.decisionAnswers = "First, Second"
        }
'''


@unittest.skipUnless(sys.platform == "darwin" and platform.machine() == "arm64", "Apple silicon macOS required")
class DeciderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = controller.DRIVER
        controller.DRIVER = original.replace(
            "        switch scenario {", DECIDER_SETUP + "        switch scenario {\n" + DECIDER_CASES, 1
        )
        try:
            controller.ControllerLifecycleTests.setUpClass.__func__(cls)
        finally:
            controller.DRIVER = original

    def exercise(self, scenario):
        controller.ControllerLifecycleTests.exercise(self, scenario)

    def test_decider_capabilities_do_not_enable_generic_decision_images(self):
        self.exercise("decider-ready-capabilities")

    def test_decision_catalog_family_is_visible_and_keeps_its_mode(self):
        self.exercise("decider-catalog-mode-and-family")

    def test_image_only_decision_uses_private_transport_and_cleans_file(self):
        self.exercise("decider-image-only-request")

    def test_request_errors_and_image_progress_are_fenced(self):
        self.exercise("decider-request-error-fence")

    def test_reset_stop_replacement_removal_and_errors_release_images(self):
        self.exercise("decider-attachment-lifecycle")

    def test_cancelled_decision_cannot_restore_results_or_images(self):
        self.exercise("decider-stop-fences-reply")

    def test_decision_model_runs_in_decision_and_blocks_chat(self):
        self.exercise("decider-mode-routing")

    def test_image_capability_absence_keeps_text_required(self):
        self.exercise("decider-unavailable-image-and-text-validation")

    def test_chat_image_followup_ownership_is_unchanged(self):
        self.exercise("image-generate-followup")

    def test_chat_image_error_retry_is_unchanged(self):
        self.exercise("image-error-retry")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeciderControllerTests)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    print(f"decider_controller_checks: run={result.testsRun} failed={len(result.failures) + len(result.errors)} skipped={len(result.skipped)}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
