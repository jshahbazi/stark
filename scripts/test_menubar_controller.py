"""Model-free checks of production Swift decisions and image attachment ownership.

Run with `uv run --project tools/converter --locked python
scripts/test_menubar_controller.py` on an Apple silicon Mac with Xcode tools.
The temporary executable includes the actual controller source plus a same-file
test extension. Compiler/runtime output is suppressed; only counts are printed.
"""

from __future__ import annotations

import io
import json
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
        func supportReport(_ status: String = "conditional") -> [String: Any] {
            ["schema_version": 1, "status": status, "mode": "text", "message": "Fixture support check",
             "reasons": ["Fixture reason"], "warnings": ["Fixture warning"],
             "pending_checks": ["Fixture runtime check"],
             "modes": ["text": "conditional", "decision": "conditional", "vision-language": "unsupported"]]
        }
        func supportEvent(_ request: [String: Any], status: String = "conditional") -> [String: Any] {
            ["event": "model_support", "request_id": request["support_request_id"]!, "report": supportReport(status)]
        }
        func familyRegistry() -> [String: Any] {
            var modes: [String: Any] = [:]
            for mode in ["text", "decision", "vision-language"] {
                modes[mode] = ["status": mode == "vision-language" ? "unsupported" : "conditional",
                               "message": "Fixture mode support", "caveats": ["Fixture caveat"]]
            }
            return ["schema_version": 1, "modes": ["text", "decision", "vision-language"],
                    "families": [["family_id": "fixture", "name": "Fixture family", "architecture_aliases": ["FixtureModel"],
                                  "config_classes": ["FixtureConfig"], "model_kind": "causal-language-model",
                                  "conversion": ["status": "implemented", "containers": ["safetensors"],
                                                 "source_formats": ["BF16"], "output_dtypes": ["BF16"]], "modes": modes]]]
        }
        switch scenario {
        case "vision-install-opt-in":
            engine.install(source: "Qwen/Qwen3.5-0.8B", kind: "huggingface", artifactMode: "vision-language")
            let requested = command()
            return requested["command"] as? String == "install" &&
                requested["artifact_mode"] as? String == "vision-language" &&
                requested["support_mode"] as? String == "vision-language" &&
                UUID(uuidString: requested["support_request_id"] as? String ?? "") != nil &&
                engine.downloadProgress.state == "waiting"
        case "text-install-default":
            engine.install(source: "Qwen/Qwen3-0.6B", kind: "huggingface")
            let requested = command()
            return requested["artifact_mode"] == nil && requested["support_mode"] as? String == "decision"
        case "text-install-purpose":
            engine.mode = .chat
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            return requested["support_mode"] as? String == "text" &&
                requested["artifact_mode"] == nil && engine.downloadProgress.state == "waiting"
        case "support-registry":
            engine.apply(["event": "model_families", "report": familyRegistry()])
            guard let registry = engine.modelFamilies, registry.families.count == 1,
                  registry.families[0].name == "Fixture family",
                  registry.families[0].modes[.text]?.status.title == "Checks required",
                  registry.families[0].modes[.visionLanguage]?.status.title == "Unavailable" else { return false }
            var future = familyRegistry()
            future["schema_version"] = 2
            engine.apply(["event": "model_families", "report": future])
            engine.requestModelFamilies()
            return engine.modelFamilies?.families.first?.id == "fixture" && command()["command"] as? String == "support"
        case "support-live-contract":
            guard let payload = try? JSONSerialization.jsonObject(with: Data(backendError.utf8)) as? [String: Any],
                  let registry = payload["registry"] as? [String: Any], let report = payload["report"] as? [String: Any] else { return false }
            engine.apply(["event": "model_families", "report": registry])
            guard let families = engine.modelFamilies?.families, families.count == 13,
                  let classifier = families.first(where: { $0.modelKind == "classifier" }),
                  classifier.modes.values.allSatisfy({ $0.status == .unsupported }),
                  let qwen3 = families.first(where: { $0.id == "qwen3" }),
                  qwen3.sourceFormatLabels == ["Safetensors floating-point", "MLX 4-bit (affine)", "MLX 8-bit (affine)"],
                  qwen3.outputDtypeLabels == ["BF16", "F16", "F32"],
                  families.first(where: { $0.id == "qwen3_5" })?.modes[.visionLanguage]?.status == .conditional else { return false }
            engine.install(source: "fixture/source", kind: "huggingface", artifactMode: "vision-language")
            let requested = command()
            engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": report])
            return engine.installSupportReport?.status == .inconclusive &&
                engine.installSupportReport?.modes.values.allSatisfy({ $0 == .inconclusive }) == true
        case "support-registry-invalid":
            var raw = familyRegistry()
            raw["schema_version"] = true
            guard ModelFamilyRegistry(raw) == nil else { return false }
            raw = familyRegistry()
            raw["modes"] = ["text", "decision", "vision-language", "unrecognized"]
            guard ModelFamilyRegistry(raw) == nil else { return false }
            raw = familyRegistry()
            var families = raw["families"] as! [[String: Any]]
            var modes = families[0]["modes"] as! [String: [String: Any]]
            modes["text"]!["status"] = "qualified"
            families[0]["modes"] = modes
            raw["families"] = families
            engine.apply(["event": "model_families", "report": raw])
            return engine.modelFamilies == nil
        case "image-support-warning-keeps-capability":
            engine.imageVerification = "locally_tested"
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            var report = supportReport("unsupported")
            report["images"] = true
            report["image_verification"] = "reference_verified"
            engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": report])
            guard engine.state == "installing" && engine.loadedID == "fixture-model" &&
                engine.selectedID == "fixture-model" && engine.loadedMode == .chat && engine.supportsImages &&
                engine.imageVerification == "locally_tested" && engine.installSupportReport?.status == .unsupported &&
                engine.message == "Fixture support check" && engine.downloadProgress.state == "waiting" else { return false }
            engine.apply(["event": "error", "message": "Fixture unsupported model"])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            return engine.modelReady && engine.loadedID == "fixture-model" && engine.supportsImages &&
                engine.imageVerification == "locally_tested" && engine.installSupportReport?.status == .unsupported
        case "support-cannot-enable-images":
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            var report = supportReport()
            report["images"] = true
            report["image_verification"] = "reference_verified"
            engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": report])
            return !engine.supportsImages && engine.imageVerification == "unverified" &&
                engine.loadedID == "fixture-model" && engine.installSupportReport?.status == .conditional
        case "support-unknown-schema-status":
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            for (key, value) in [("schema_version", 2 as Any), ("schema_version", true as Any), ("status", "qualified" as Any)] {
                var report = supportReport()
                report[key] = value
                engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": report])
                guard engine.installSupportReport == nil else { return false }
            }
            var inconclusive = supportReport("inconclusive")
            inconclusive["modes"] = ["text": "inconclusive", "decision": "inconclusive", "vision-language": "inconclusive"]
            engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": inconclusive])
            return engine.installSupportReport?.status == .inconclusive &&
                engine.installSupportReport?.modes[.visionLanguage] == .inconclusive &&
                engine.installSupportReport?.details == ["Fixture warning", "Fixture runtime check"]
        case "support-stale-source":
            engine.install(source: "fixture/A", kind: "huggingface")
            let first = command()
            engine.apply(supportEvent(first))
            engine.state = "ready"
            engine.resetInstallProgress(kind: "local")
            guard engine.installSupportReport == nil else { return false }
            engine.install(source: "fixture/B", kind: "local")
            let second = command()
            engine.apply(supportEvent(first, status: "unsupported"))
            guard engine.installSupportReport == nil,
                  first["support_request_id"] as? String != second["support_request_id"] as? String else { return false }
            engine.apply(supportEvent(second))
            return engine.installSupportReport?.status == .conditional && engine.downloadProgress.state == "skipped"
        case "support-stale-mode":
            engine.install(source: "fixture/source", kind: "huggingface")
            let first = command()
            engine.apply(supportEvent(first))
            engine.state = "ready"
            engine.loadedID = ""
            engine.changeMode(.chat)
            guard engine.installSupportReport == nil else { return false }
            engine.install(source: "fixture/source", kind: "huggingface", artifactMode: "vision-language")
            let second = command()
            engine.apply(supportEvent(first, status: "unsupported"))
            guard engine.installSupportReport == nil, first["support_mode"] as? String == "decision",
                  second["support_mode"] as? String == "vision-language" else { return false }
            engine.apply(supportEvent(second))
            return engine.installSupportReport?.status == .conditional
        case "support-cancel":
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            engine.apply(supportEvent(requested))
            engine.stop()
            guard command()["command"] as? String == "stop", engine.installSupportReport == nil else { return false }
            engine.apply(supportEvent(requested))
            engine.state = "installing"
            engine.apply(supportEvent(requested))
            return engine.installSupportReport == nil && engine.downloadProgress.state == "cancelled"
        case "support-completion-fence":
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            engine.apply(supportEvent(requested))
            engine.apply(["event": "error", "message": "Fixture install failure"])
            engine.apply(supportEvent(requested, status: "unsupported"))
            guard engine.installSupportReport?.status == .conditional else { return false }
            engine.state = "ready"
            engine.resetInstallProgress(kind: "huggingface")
            return engine.installSupportReport == nil
        case "install-error-reset":
            let failure = "This model architecture has no implemented converter."
            engine.install(source: "fixture/unsupported", kind: "huggingface")
            let requested = command()
            var report = supportReport("unsupported")
            report["message"] = failure
            engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": report])
            engine.apply(["event": "error", "message": failure])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.errorMessage == failure, engine.installSupportReport != nil else { return false }
            engine.resetInstallProgress(kind: "huggingface")
            guard engine.errorMessage == nil, engine.installSupportReport == nil, engine.loadedID == "fixture-model",
                  engine.downloadProgress.state == "waiting" else { return false }
            engine.install(source: "fixture/metadata-unavailable", kind: "local")
            _ = command()
            engine.apply(["event": "error", "message": "Fixture support timeout without a report"])
            engine.apply(["event": "status", "state": "error", "message": "Operation failed"])
            engine.resetInstallProgress(kind: "local")
            return engine.errorMessage == nil && engine.downloadProgress.state == "skipped"
        case "install-error-runtime-retained":
            let failure = "Fixture failure shared by install and runtime"
            engine.install(source: "fixture/source", kind: "huggingface")
            _ = command()
            engine.apply(["event": "error", "message": failure])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            // Even identical text from a later runtime error has different ownership.
            engine.apply(["event": "error", "message": failure])
            engine.resetInstallProgress(kind: "local")
            guard engine.errorMessage == failure else { return false }
            engine.errorMessage = "Fixture attachment validation error"
            engine.resetInstallProgress(kind: "huggingface")
            return engine.errorMessage == "Fixture attachment validation error"
        case "install-error-active-guard":
            engine.install(source: "fixture/source", kind: "huggingface")
            _ = command()
            engine.apply(["event": "error", "message": "Fixture install failure"])
            engine.resetInstallProgress(kind: "local")
            guard engine.errorMessage == "Fixture install failure", engine.downloadProgress.state == "failed" else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            engine.resetInstallProgress(kind: "local")
            return engine.errorMessage == nil && engine.downloadProgress.state == "skipped"
        case "support-shutdown-fence":
            engine.apply(["event": "model_families", "report": familyRegistry()])
            engine.install(source: "fixture/source", kind: "huggingface")
            let requested = command()
            engine.apply(supportEvent(requested))
            engine.shutdown()
            guard command()["command"] as? String == "quit", engine.modelFamilies == nil,
                  engine.installSupportReport == nil else { return false }
            engine.apply(supportEvent(requested))
            return engine.installSupportReport == nil
        case "support-display-bounds":
            var report = supportReport()
            report["message"] = String(repeating: "a", count: 2_001)
            guard ModelSupportReport(report) == nil else { return false }
            report = supportReport()
            report["warnings"] = Array(repeating: "Fixture warning", count: 33)
            guard ModelSupportReport(report) == nil else { return false }
            report = supportReport()
            report["message"] = "Fixture\u{1b}control"
            guard ModelSupportReport(report) == nil else { return false }
            report = supportReport()
            report["pending_checks"] = ["tensor_headers", "tensor_operations", "native_loading", "local_execution", "single_token_labels"]
            return ModelSupportReport(report)?.details == ["Fixture warning", "Tensor layout and operations must pass validation.",
                "The checkpoint must run successfully in the native engine.", "Decision answer labels must each map to a unique token."]
        case "support-alternative-modes":
            var report = supportReport("unsupported")
            report["mode"] = "vision-language"
            guard let parsed = ModelSupportReport(report), parsed.alternativeModes == [.text, .decision] else { return false }
            report["mode"] = "text"
            guard ModelSupportReport(report)?.alternativeModes == [.decision] else { return false }
            report["mode"] = "unrecognized"
            return ModelSupportReport(report) == nil
        case "image-capability":
            guard engine.canAttachImage else { return false }
            engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model", "images": true])
            guard !engine.supportsImages, !engine.canAttachImage else { return false }
            engine.apply(imageReady)
            engine.selectedID = "different-fixture"
            return engine.supportsImages && !engine.canAttachImage
        case "image-local-evidence":
            var verified = imageReady
            verified["image_verification"] = "locally_tested"
            engine.apply(verified)
            guard engine.supportsImages, engine.imageVerification == "locally_tested",
                  engine.imageVerificationNote.contains("Reference parity is unverified"),
                  !engine.imageVerificationNote.contains("has reference-parity verification") else { return false }
            engine.pendingOptions = engine.options
            engine.state = "configuring"
            verified["event"] = "configured"
            engine.apply(verified)
            return engine.modelReady && engine.supportsImages && engine.imageVerification == "locally_tested"
        case "image-reference-evidence":
            var verified = imageReady
            verified["image_verification"] = "reference_verified"
            engine.apply(verified)
            guard engine.supportsImages, engine.imageVerification == "reference_verified",
                  engine.imageVerificationNote.contains("has reference-parity verification") else { return false }
            verified["images"] = false
            engine.apply(verified)
            return !engine.supportsImages && engine.imageVerification == "unverified" &&
                !engine.imageVerificationNote.contains("has reference-parity verification")
        case "image-unknown-evidence":
            var unsupported = imageReady
            unsupported["image_verification"] = "self-declared-reference-pass"
            engine.apply(unsupported)
            guard engine.imageVerification == "unverified",
                  !engine.imageVerificationNote.contains("has reference-parity verification") else { return false }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return !engine.supportsImages && engine.imageVerification == "unverified"
        case "image-install-check-clears":
            try! engine.attachImage(bytes: imageBytes)
            let image = engine.draftImage!
            engine.messages = [ChatMessage(text: "synthetic image turn", isUser: true, image: image)]
            engine.conversationImage = image
            _ = try! engine.attachmentStaging!.stage(image)
            engine.imageVerification = "reference_verified"
            engine.state = "installing"
            engine.apply(["event": "image_check", "state": "checking"])
            return engine.state == "installing" && engine.busy && engine.loadedID.isEmpty && engine.loadedMode == nil &&
                !engine.supportsImages && engine.imageVerification == "unverified" &&
                engine.messages.isEmpty && engine.draftImage == nil && engine.conversationImage == nil &&
                staged().isEmpty && engine.message == "Checking image support on this Mac…"
        case "image-install-check-state-guard":
            try! engine.attachImage(bytes: imageBytes)
            engine.apply(["event": "image_check", "state": "checking"])
            guard engine.modelReady, engine.loadedID == "fixture-model", engine.draftImage != nil else { return false }
            engine.state = "installing"
            engine.apply(["event": "image_check", "state": "self-declared-pass"])
            return engine.loadedID == "fixture-model" && engine.supportsImages && engine.draftImage != nil
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

    def test_chat_install_forwards_text_purpose_before_download(self):
        self.exercise("text-install-purpose")

    def test_family_registry_parses_mode_labels_and_ignores_future_schema(self):
        self.exercise("support-registry")

    def test_production_registry_and_inconclusive_report_match_swift_contract(self):
        from stark_convert.model_support import preflight_model, support_matrix

        self.exercise(
            "support-live-contract",
            json.dumps(
                {
                    "registry": support_matrix(),
                    "report": preflight_model(None, mode="vision-language"),
                }
            ),
        )

    def test_family_registry_rejects_unknown_modes_and_self_qualified_status(self):
        self.exercise("support-registry-invalid")

    def test_unsupported_preflight_preserves_loaded_model_and_image_evidence(self):
        self.exercise("image-support-warning-keeps-capability")

    def test_support_metadata_cannot_enable_images(self):
        self.exercise("support-cannot-enable-images")

    def test_support_report_ignores_future_schema_and_unknown_status(self):
        self.exercise("support-unknown-schema-status")

    def test_late_source_report_cannot_replace_new_install_report(self):
        self.exercise("support-stale-source")

    def test_mode_change_invalidates_previous_support_report(self):
        self.exercise("support-stale-mode")

    def test_cancelled_support_report_cannot_reappear(self):
        self.exercise("support-cancel")

    def test_failed_install_retains_current_report_and_fences_late_events(self):
        self.exercise("support-completion-fence")

    def test_source_or_install_reset_clears_only_previous_install_error(self):
        self.exercise("install-error-reset")

    def test_install_reset_preserves_later_runtime_or_attachment_errors(self):
        self.exercise("install-error-runtime-retained")

    def test_install_error_cannot_clear_before_active_install_finishes(self):
        self.exercise("install-error-active-guard")

    def test_shutdown_clears_registry_and_pending_support_events(self):
        self.exercise("support-shutdown-fence")

    def test_support_display_metadata_is_bounded(self):
        self.exercise("support-display-bounds")

    def test_unsupported_image_report_shows_other_conditional_modes(self):
        self.exercise("support-alternative-modes")

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

    def test_local_image_evidence_remains_distinct_after_configuration(self):
        self.exercise("image-local-evidence")

    def test_reference_image_evidence_requires_available_images(self):
        self.exercise("image-reference-evidence")

    def test_unknown_image_evidence_is_not_promoted(self):
        self.exercise("image-unknown-evidence")

    def test_install_image_checks_clear_previous_session_and_keep_busy(self):
        self.exercise("image-install-check-clears")

    def test_install_image_check_events_are_state_guarded(self):
        self.exercise("image-install-check-state-guard")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControllerLifecycleTests)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(
        f"controller_checks: run={result.testsRun} failed={failed} skipped={len(result.skipped)}"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
