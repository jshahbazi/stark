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

final class ControllerCredentialStorage: APICredentialStorage {
    var status: OSStatus = errSecSuccess
    var key = Data(String(repeating: "a", count: 64).utf8)
    var replacementStatus: OSStatus = errSecSuccess
    var readGate: DispatchSemaphore?
    var replaceGate: DispatchSemaphore?
    private let countLock = NSLock()
    private var reads = 0
    private var replaces = 0
    var readCount: Int { countLock.lock(); defer { countLock.unlock() }; return reads }
    var replaceCount: Int { countLock.lock(); defer { countLock.unlock() }; return replaces }
    func read() -> (OSStatus, Data?) {
        countLock.lock(); reads += 1; countLock.unlock()
        readGate?.wait()
        return (status, key)
    }
    func add(_ data: Data) -> OSStatus { key = data; return errSecSuccess }
    func replace(_ data: Data) -> OSStatus {
        countLock.lock(); replaces += 1; countLock.unlock()
        replaceGate?.wait()
        if replacementStatus == errSecSuccess { key = data }
        return replacementStatus
    }
}

extension EngineController {
    static func exercise(_ scenario: String, backendError: String) -> Bool {
        let previousSelectedModel = UserDefaults.standard.object(forKey: "selectedModel")
        let previousPlaygroundMode = UserDefaults.standard.object(forKey: "playgroundMode")
        defer {
            if let previousSelectedModel = previousSelectedModel {
                UserDefaults.standard.set(previousSelectedModel, forKey: "selectedModel")
            } else { UserDefaults.standard.removeObject(forKey: "selectedModel") }
            if let previousPlaygroundMode { UserDefaults.standard.set(previousPlaygroundMode, forKey: "playgroundMode") }
            else { UserDefaults.standard.removeObject(forKey: "playgroundMode") }
        }
        let credentialStorage = ControllerCredentialStorage()
        let apiSuite = "stark.controller-api-test." + UUID().uuidString
        let apiDefaults = UserDefaults(suiteName: apiSuite)!
        defer { apiDefaults.removePersistentDomain(forName: apiSuite) }
        let credentials = APICredentialStore(service: apiSuite, storage: credentialStorage) { String(repeating: "b", count: 64) }
        let engine = EngineController(apiCredentials: credentials, apiDefaults: apiDefaults,
                                      gpuMemoryBudgetBytes: 8_589_934_592)
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
        func commands() -> [[String: Any]] {
            pipe.fileHandleForReading.availableData.split(separator: 10).map {
                try! JSONSerialization.jsonObject(with: Data($0)) as! [String: Any]
            }
        }
        func noCommandSent() -> Bool {
            var descriptor = pollfd(fd: pipe.fileHandleForReading.fileDescriptor, events: Int16(POLLIN), revents: 0)
            return Darwin.poll(&descriptor, 1, 0) == 0
        }
        func staged() -> [String] {
            try! FileManager.default.contentsOfDirectory(atPath: engine.attachmentStaging!.root.path)
        }
        func pump(timeout: TimeInterval = 2, _ complete: () -> Bool) -> Bool {
            let deadline = Date().addingTimeInterval(timeout)
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
        func modelDisplay(_ groupID: String = "example/base", name: String = "Base") -> [String: Any] {
            ["schema_version": 1, "group_id": groupID, "group_name": name, "family_id": "qwen3",
             "quantization": "BF16", "weight_bytes": Int64(1_190_000_000), "source_revision": String(repeating: "a", count: 40)]
        }
        func variant(_ id: String, name: String, display: [String: Any]?) -> InstalledModel {
            InstalledModel(id: id, name: name, path: "/private/" + id, apiID: "private-" + id,
                           display: display.flatMap(InstalledModelDisplay.init))
        }
        func catalogItem(_ id: String = "entry") -> [String: Any] {
            ["id": id, "repo": "example/model", "revision": String(repeating: "a", count: 40), "name": "Model 2",
             "size_label": "0.6B", "parameter_count": Int64(600_000_000), "weight_bytes": Int64(1_200_000_000),
             "quantization": "BF16", "artifact_modes": ["text"],
             "default_artifact_mode": "text", "downloadable": true, "installed_ids": [String]()]
        }
        func groupedCatalogItem(_ id: String, repo: String, name: String, group: String, groupName: String,
                                publisher: String, size: String? = "0.6B", quantization: String = "BF16",
                                parameters: Int64? = nil) -> [String: Any] {
            var item = catalogItem(id)
            item["repo"] = repo; item["name"] = name; item["quantization"] = quantization
            item["size_label"] = size as Any? ?? NSNull()
            item["parameter_count"] = parameters as Any? ?? NSNull()
            item["model_group"] = ["id": String(repeating: group, count: 32), "name": groupName, "publisher": publisher]
            return item
        }
        func catalogReport(_ family: String, status: String = "ready", items: [[String: Any]]? = nil,
                           offset: Int64 = 0, nextOffset: Int64? = nil) -> [String: Any] {
            ["schema_version": 1, "family_id": family, "status": status, "message": "Fixture catalog message",
             "items": items ?? (status == "ready" ? [catalogItem()] : []), "has_more": nextOffset != nil,
             "offset": offset, "next_offset": nextOffset as Any? ?? NSNull()]
        }
        func catalogReply(_ request: [String: Any], status: String = "ready", items: [[String: Any]]? = nil,
                          nextOffset: Int64? = nil) -> [String: Any] {
            ["event": "model_catalog", "request_id": request["request_id"]!, "family_id": request["family_id"]!,
             "report": catalogReport(request["family_id"] as! String, status: status, items: items,
                                     offset: (request["offset"] as? NSNumber)?.int64Value ?? 0, nextOffset: nextOffset)]
        }
        func removal(_ action: String = "trash") -> [String: Any] {
            ["schema_version": 1, "action": action, "label": action == "trash" ? "Move to Trash" : "Remove from Stark",
             "message": action == "trash" ? "Its files will move to Trash." : "Its folder will remain on this Mac."]
        }
        func generationProgress(id: String? = nil, sampled: Int = 0, elapsed: Double = 0) -> [String: Any] {
            ["event": "generation_progress", "id": id ?? engine.pendingMessageID!.uuidString.lowercased(),
             "model": "fixture-model", "phase": sampled == 0 ? "prefill" : "decode",
             "sampled_tokens": sampled, "output_token_budget": 64, "elapsed_seconds": elapsed,
             "ttft_seconds": sampled == 0 ? NSNull() : 0.5,
             "ttft_lower_seconds": 0.3, "ttft_upper_seconds": 0.8,
             "remaining_lower_seconds": 2.0, "remaining_upper_seconds": 6.0, "calibrated": true]
        }
        func apiProgress(_ id: String, sampled: Int = 0, elapsed: Double = 0, reused: Int? = nil) -> [String: Any] {
            var event = generationProgress(id: id, sampled: sampled, elapsed: elapsed)
            event["source"] = "api"; event["input_tokens"] = 900
            if let reused { event["reused_input_tokens"] = reused }
            return event
        }
        func beginnerReport(_ installed: [String] = []) -> [String: Any] {
            let host: [String: Any] = ["physical_memory_bytes": Int64(16 * 1024 * 1024 * 1024),
                "available_memory_bytes": Int64(8 * 1024 * 1024 * 1024),
                "gpu_name": "Fixture GPU", "gpu_memory_budget_bytes": Int64(12 * 1024 * 1024 * 1024),
                "free_disk_bytes": Int64(50 * 1024 * 1024 * 1024)]
            let item: [String: Any] = ["id": "fixture-recipe", "name": "Starting model", "repo": "example/model",
                        "revision": String(repeating: "a", count: 40), "artifact_mode": "text",
                        "quantization": "BF16", "download_bytes": 1000, "installed_bytes_estimate": 1200,
                        "setup_disk_bytes_estimate": 4000, "runtime_memory_bytes_estimate": 8000,
                        "context_tokens": 4096, "installed_ids": installed, "license": "Apache-2.0",
                        "qualification": "Synthetic controller fixture", "recommendation_reason": "Fits the fixture GPU budget",
                        "caveats": ["Planning estimates"]]
            return ["schema_version": 1, "status": "ready", "message": "Checked starting recipes",
                    "host": host, "items": [item]]
        }
        func prepareBeginnerReport(_ installed: [String] = []) -> String {
            engine.requestBeginnerModels(task: "general")
            let id = command()["request_id"] as! String
            engine.apply(["event": "guided_models", "request_id": id, "report": beginnerReport(installed)])
            return id
        }
        let apiFirst = String(repeating: "a", count: 32), apiSecond = String(repeating: "b", count: 32)
        switch scenario {
        case "worker-pipe-small-guided-reply":
            let replies = Pipe()
            engine.observeBackendOutput(replies.fileHandleForReading)
            defer {
                replies.fileHandleForReading.readabilityHandler = nil
                try? replies.fileHandleForWriting.close()
                try? replies.fileHandleForReading.close()
            }
            engine.requestBeginnerModels(task: "coding", preference: "quality")
            let requestID = command()["request_id"] as! String
            guard engine.beginnerChoicesLoading else { return false }
            var bytes = try! JSONSerialization.data(withJSONObject: [
                "event": "guided_models", "request_id": requestID, "report": beginnerReport(["fixture-model"])
            ])
            bytes.append(10)
            guard bytes.count < 65_536 else { return false }
            replies.fileHandleForWriting.write(bytes)
            // The writer stays open throughout the wait, just like the desktop
            // backend. Filling a 64 KiB buffer or reaching EOF cannot unblock it.
            return pump({ !engine.beginnerChoicesLoading }) &&
                engine.beginnerReport?.items.first?.installedIDs == ["fixture-model"] &&
                replies.fileHandleForReading.readabilityHandler != nil
        case "worker-pipe-fragmented-and-coalesced-events":
            let replies = Pipe()
            engine.observeBackendOutput(replies.fileHandleForReading)
            defer {
                replies.fileHandleForReading.readabilityHandler = nil
                try? replies.fileHandleForWriting.close()
                try? replies.fileHandleForReading.close()
            }
            func frame(_ memory: Int) -> Data {
                var bytes = try! JSONSerialization.data(withJSONObject: [
                    "event": "gpu_memory", "model_id": "fixture-model", "mode": "decision",
                    "gpu_memory_bytes": memory
                ])
                bytes.append(10)
                return bytes
            }
            let first = frame(111), second = frame(222), third = frame(333)
            let firstSplit = first.count / 2, thirdSplit = third.count / 2
            replies.fileHandleForWriting.write(first.prefix(firstSplit))
            guard pump({ engine.reader.sync { engine.pending.count == firstSplit } }),
                  engine.gpuMemoryBytes == nil else { return false }
            replies.fileHandleForWriting.write(first.suffix(from: firstSplit) + second + third.prefix(thirdSplit))
            guard pump({ engine.gpuMemoryBytes == 222 && engine.reader.sync { engine.pending.count == thirdSplit } })
            else { return false }
            replies.fileHandleForWriting.write(third.suffix(from: thirdSplit))
            return pump({ engine.gpuMemoryBytes == 333 }) && engine.reader.sync { engine.pending.isEmpty }
        case "worker-pipe-eof-clears-handler":
            let replies = Pipe()
            engine.observeBackendOutput(replies.fileHandleForReading)
            defer {
                replies.fileHandleForReading.readabilityHandler = nil
                try? replies.fileHandleForReading.close()
            }
            guard replies.fileHandleForReading.readabilityHandler != nil else { return false }
            try? replies.fileHandleForWriting.close()
            return pump({ replies.fileHandleForReading.readabilityHandler == nil })
        case "resource-budget-memory-derived":
            for mib in [16, 64, 65, 256, 1024] {
                let input = mib * 1024 * 1024
                let wire = 6 * input + 2 * 1024 * 1024
                guard engine.applyResourceBudget(["resource_budget": [
                    "request_budget_mib": mib, "input_bytes": input, "max_request_bytes": wire
                ]]), engine.workerWireLimit() == wire else { return false }
            }
            let prior = engine.workerWireLimit()
            for invalid: [String: Any] in [
                ["request_budget_mib": Int.max, "input_bytes": Int.max, "max_request_bytes": Int.max],
                ["request_budget_mib": 0, "input_bytes": 0, "max_request_bytes": 2 * 1024 * 1024],
                ["request_budget_mib": 256, "input_bytes": 1, "max_request_bytes": 1]
            ] {
                guard !engine.applyResourceBudget(["resource_budget": invalid]),
                      engine.workerWireLimit() == prior else { return false }
            }
            return true
        case "backend-short-replies":
            let backend = Pipe()
            engine.observeBackendOutput(backend.fileHandleForReading)
            defer {
                backend.fileHandleForReading.readabilityHandler = nil
                try? backend.fileHandleForWriting.close()
                try? backend.fileHandleForReading.close()
            }
            engine.state = "starting"
            for message in ["Fixture backend ready", "Fixture next reply"] {
                let event = ["event": "status", "state": "idle", "message": message]
                var bytes = try! JSONSerialization.data(withJSONObject: event)
                bytes.append(10)
                backend.fileHandleForWriting.write(bytes)
                // The writer deliberately remains open, as in the real app.
                guard pump({ engine.state == "idle" && engine.message == message }) else { return false }
            }
            return true
        case "beginner-general-task-transport":
            engine.requestBeginnerModels(task: "general")
            let request = command()
            return engine.beginnerChoicesLoading && request["command"] as? String == "guided_models"
                && request["task"] as? String == "general"
        case "beginner-recommendation-fences":
            engine.requestBeginnerModels(task: "general")
            let request = command()
            let id = request["request_id"] as! String
            guard request["preference"] as? String == "quality",
                  let hardware = request["hardware"] as? [String: Any],
                  hardware["gpu_available"] is Bool, hardware["gpu_name"] is String,
                  hardware["gpu_memory_budget_bytes"] != nil, hardware["cpu_cores"] != nil else { return false }
            engine.apply(["event": "guided_models", "request_id": UUID().uuidString.lowercased(),
                          "report": beginnerReport()])
            guard engine.beginnerChoicesLoading, engine.beginnerReport == nil else { return false }
            var invalid = beginnerReport()
            var items = invalid["items"] as! [[String: Any]]
            items[0]["setup_disk_bytes_estimate"] = -1; invalid["items"] = items
            engine.apply(["event": "guided_models", "request_id": id, "report": invalid])
            guard !engine.beginnerChoicesLoading, engine.beginnerReport == nil,
                  engine.beginnerChoicesError != nil else { return false }
            _ = prepareBeginnerReport()
            guard engine.beginnerReport?.deviceSummary.contains("Fixture GPU") == true,
                  engine.beginnerReport?.items.first?.recommendationReason == "Fits the fixture GPU budget" else { return false }
            engine.requestBeginnerModels(task: "coding", excludingModelID: "fixture-model", preference: "speed")
            let retry = command()
            engine.apply(["event": "guided_models", "request_id": id, "report": beginnerReport()])
            let excluded = retry["excluded_model_ids"] as? [String]
            let preference = retry["preference"] as? String
            return engine.beginnerChoicesLoading && engine.beginnerReport == nil &&
                excluded == ["fixture-model"] && engine.loadedID == "fixture-model" && preference == "speed"
        case "beginner-installed-only-report":
            var report = beginnerReport(["fixture-model"])
            var items = report["items"] as! [[String: Any]]
            items[0]["download_bytes"] = 0; report["items"] = items
            guard BeginnerModelReport(report)?.items.first?.downloadBytes == 0 else { return false }
            items[0]["installed_ids"] = [String](); report["items"] = items
            guard BeginnerModelReport(report) == nil else { return false }
            items[0]["installed_ids"] = ["fixture-model"]
            items[0]["download_bytes"] = -1; report["items"] = items
            return BeginnerModelReport(report) == nil
        case "beginner-rejected-install-does-not-start":
            engine.models = [variant("fixture-model", name: "Current", display: nil)]
            _ = prepareBeginnerReport(["fixture-model"])
            guard !engine.installBeginnerModel(recipeID: "fixture-recipe"),
                  engine.beginnerInstallRequestID == nil, noCommandSent() else { return false }
            return !engine.installBeginnerModel(recipeID: "stale-recipe") &&
                engine.beginnerInstallRequestID == nil && engine.errorMessage != nil && noCommandSent()
        case "beginner-cancel-preserves-current":
            engine.models = [variant("fixture-model", name: "Current", display: nil)]
            engine.messages = [ChatMessage(text: "synthetic history", isUser: true)]
            engine.draft = "synthetic draft"
            let recommendation = prepareBeginnerReport()
            engine.installBeginnerModel(recipeID: "fixture-recipe")
            let install = command(); let id = install["support_request_id"] as! String
            guard install["command"] as? String == "install_guided",
                  install["recommendation_request_id"] as? String == recommendation,
                  engine.selectedID == "fixture-model", engine.loadedID == "fixture-model",
                  engine.loadedMode == .decision, engine.messages.count == 1 else { return false }
            let progress: [String: Any] = ["event": "progress", "phase": "download", "state": "running",
                                          "completed_bytes": Int64(50), "total_bytes": Int64(100)]
            engine.apply(progress.merging(["request_id": UUID().uuidString]) { _, new in new })
            guard engine.downloadProgress.completedBytes == 0 else { return false }
            engine.apply(progress.merging(["request_id": id]) { _, new in new })
            guard engine.downloadProgress.fraction == 0.5 else { return false }
            engine.cancelBeginnerInstallation()
            let cancel = command()
            guard cancel["command"] as? String == "cancel_install", cancel["request_id"] as? String == id,
                  engine.beginnerCancelPending, engine.loadedID == "fixture-model" else { return false }
            engine.apply(["event": "install_cancelled", "request_id": UUID().uuidString])
            guard engine.beginnerCancelPending else { return false }
            engine.apply(["event": "install_cancelled", "request_id": id])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            return engine.beginnerInstallState == "cancelled" && !engine.beginnerCancelPending &&
                engine.loadedID == "fixture-model" && engine.selectedID == "fixture-model" &&
                engine.modelReady && engine.messages.count == 1 && engine.draft == "synthetic draft" && noCommandSent()
        case "beginner-explicit-switch-fences":
            engine.models = [variant("fixture-model", name: "Current", display: nil)]
            engine.messages = [ChatMessage(text: "synthetic history", isUser: true)]
            let recommendation = prepareBeginnerReport()
            engine.installBeginnerModel(recipeID: "fixture-recipe")
            let install = command(); let id = install["support_request_id"] as! String
            engine.apply(["event": "installed", "support_request_id": UUID().uuidString,
                          "model": ["id": "new-model"]])
            guard engine.beginnerInstalledResultID == nil else { return false }
            engine.models.append(variant("new-model", name: "New", display: nil))
            engine.apply(["event": "installed", "support_request_id": id, "model": ["id": "new-model"]])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.selectedID == "fixture-model", engine.loadedID == "fixture-model",
                  engine.messages.count == 1, engine.beginnerInstalledResultID == "new-model" else { return false }
            guard engine.startBeginnerModel(recipeID: "fixture-recipe", modelID: "new-model") else { return false }
            let load = command(); let loadID = load["request_id"] as! String
            guard load["command"] as? String == "run_guided", load["recommendation_request_id"] as? String == recommendation,
                  load["model_id"] as? String == "new-model", engine.messages.count == 1 else { return false }
            var ready: [String: Any] = ["event": "ready", "request_id": UUID().uuidString.lowercased(),
                "mode": "chat", "model_id": "new-model", "max_context_tokens": 4096,
                "context_tokens": 4096, "max_output_tokens": 512, "tools": true]
            engine.apply(ready)
            guard engine.messages.count == 1, engine.loadedID == "fixture-model" else { return false }
            ready["request_id"] = loadID; ready["model_id"] = "unrelated-model"; engine.apply(ready)
            guard engine.messages.count == 1 else { return false }
            ready["model_id"] = "new-model"; engine.apply(ready)
            return engine.modelReady && engine.loadedID == "new-model" && engine.messages.isEmpty &&
                engine.apiContextTokens == 4096 && engine.models.count == 2 && noCommandSent()
        case "beginner-load-rejection-preserves-history":
            engine.mode = .chat; engine.loadedMode = .chat; engine.supportsTools = true; engine.toolParser = "fixture"
            engine.models = [variant("fixture-model", name: "Current", display: nil), variant("new-model", name: "New", display: nil)]
            engine.messages = [ChatMessage(text: "synthetic history", isUser: true)]
            _ = prepareBeginnerReport(["new-model"])
            guard engine.startBeginnerModel(recipeID: "fixture-recipe", modelID: "new-model") else { return false }
            let id = command()["request_id"] as! String
            engine.apply(["event": "error", "request_id": id, "message": "Source recipe changed"])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            return engine.modelReady && engine.selectedID == "fixture-model" && engine.loadedID == "fixture-model" &&
                engine.messages.count == 1 && engine.supportsTools && engine.toolParser == "fixture"
        case "beginner-invalid-ready-settles-start":
            engine.models = [variant("fixture-model", name: "Current", display: nil), variant("new-model", name: "New", display: nil)]
            _ = prepareBeginnerReport(["new-model"])
            guard engine.startBeginnerModel(recipeID: "fixture-recipe", modelID: "new-model") else { return false }
            let id = command()["request_id"] as! String
            engine.apply(["event": "ready", "request_id": id, "mode": "chat", "model_id": "new-model",
                          "resource_budget": ["schema_version": 999]])
            return engine.state == "error" && engine.errorMessage != nil &&
                engine.beginnerRunRequestID == nil && engine.beginnerRunBackup == nil && !engine.modelReady
        case "beginner-api-budget-keeps-native-capacity":
            guard BeginnerContextPolicy.requiresRestart(expected: 32768, local: 4096, api: 4096),
                  BeginnerContextPolicy.requiresRestart(expected: 32768, local: 32768, api: 4096),
                  BeginnerContextPolicy.requiresRestart(expected: 32768, local: 40960, api: 40960),
                  !BeginnerContextPolicy.requiresRestart(expected: 32768, local: 32768, api: 32768) else { return false }
            engine.mode = .chat; engine.loadedMode = .chat
            engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model",
                          "max_context_tokens": 262144, "context_tokens": 32768,
                          "api_context_tokens": 32768, "max_output_tokens": 262143,
                          "api_max_output_tokens": 32767, "tools": true])
            guard engine.contextLimit == 262144, engine.contextCapacityKnown,
                  engine.outputLimit == 262143, engine.apiMaxNewTokens == 32767,
                  engine.apiContextTokens == 32768, engine.options.contextTokens == 32768 else { return false }
            // A backend without the additive API field still needs a safe cap.
            engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model",
                          "max_context_tokens": 262144, "context_tokens": 32768,
                          "api_context_tokens": 32768, "max_output_tokens": 262143, "tools": true])
            guard engine.outputLimit == 262143, engine.apiMaxNewTokens == 32767 else { return false }
            engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model",
                          "max_context_tokens": Int(UInt32.max), "context_capacity_known": false,
                          "context_tokens": 32768, "api_context_tokens": 32768, "max_output_tokens": 4096])
            return !engine.contextCapacityKnown && engine.contextLimit == Int(UInt32.max) &&
                engine.apiContextTokens == 32768
        case "beginner-start-busy-rejection", "beginner-start-missing-rejection":
            engine.mode = .chat; engine.loadedMode = .chat
            engine.supportsTools = true; engine.toolParser = "fixture"
            engine.models = [variant("fixture-model", name: "Current", display: nil), variant("new-model", name: "New", display: nil)]
            engine.messages = [ChatMessage(text: "synthetic history", isUser: true)]
            engine.draft = "synthetic draft"
            _ = prepareBeginnerReport(["new-model"])
            let historyIDs = engine.messages.map(\.id)
            let context = engine.options.contextTokens
            let maximum = engine.options.maxNewTokens
            if scenario == "beginner-start-busy-rejection" {
                engine.state = "generating"; engine.message = "Synthetic active request"
            } else {
                engine.models.removeAll { $0.id == "new-model" }
            }
            let priorState = engine.state, priorMessage = engine.message
            guard !engine.startBeginnerModel(recipeID: "fixture-recipe", modelID: "new-model"),
                  noCommandSent(), engine.beginnerRunRequestID == nil, engine.beginnerRunBackup == nil,
                  engine.selectedID == "fixture-model", engine.loadedID == "fixture-model",
                  engine.mode == .chat, engine.loadedMode == .chat,
                  engine.state == priorState, engine.message == priorMessage,
                  engine.messages.map(\.id) == historyIDs, engine.draft == "synthetic draft",
                  engine.options.contextTokens == context, engine.options.maxNewTokens == maximum,
                  engine.supportsTools, engine.toolParser == "fixture",
                  let error = engine.errorMessage, error.count < 200 else { return false }
            return scenario == "beginner-start-busy-rejection"
                ? error.contains("current request") : error.contains("no longer installed")
        case "generation-progress-external-activity":
            engine.mode = .chat; engine.loadedMode = .chat
            engine.state = "generating"; engine.message = "Connected application request"
            guard !engine.hasPlaygroundGeneration, engine.hasAPIGeneration else { return false }
            engine.apply(generationProgress(id: UUID().uuidString))
            guard engine.generationProgress == nil, engine.message == "Connected application request" else { return false }
            engine.state = "ready"; engine.draft = "synthetic prompt"
            engine.generate(); _ = command()
            guard engine.hasPlaygroundGeneration, !engine.hasAPIGeneration else { return false }
            engine.apply(apiProgress(apiFirst))
            guard engine.generationProgress == nil else { return false }
            engine.apply(["event": "reply", "text": "synthetic answer"])
            return !engine.hasPlaygroundGeneration
        case "playground-request-error":
            engine.mode = .chat; engine.loadedMode = .chat; engine.draft = "synthetic prompt"
            engine.generate()
            let id = command()["request_id"] as! String
            engine.apply(["event": "error", "request_id": UUID().uuidString.lowercased(), "message": "Stale failure"])
            guard engine.pendingMessageID != nil, engine.errorMessage == nil, engine.messages.count == 1 else { return false }
            engine.apply(["event": "error", "request_id": id, "message": "Conversation is full."])
            guard engine.pendingMessageID == nil, engine.messages.isEmpty, engine.draft == "synthetic prompt",
                  engine.errorMessage == "Conversation is full.", engine.state == "generating" else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.modelReady else { return false }
            // A send rejected behind a connected application leaves that request's progress visible.
            engine.draft = "synthetic retry"
            engine.generate()
            let retry = command()["request_id"] as! String
            engine.apply(["event": "status", "state": "generating", "message": "Generating for an app…"])
            engine.apply(["event": "error", "request_id": retry, "message": "Wait for the current operation or choose Stop."])
            guard !engine.hasPlaygroundGeneration, engine.hasAPIGeneration, engine.draft == "synthetic retry" else { return false }
            engine.apply(apiProgress(apiFirst))
            return engine.generationProgress != nil
        case "generation-progress-api-requests":
            engine.mode = .decision; engine.loadedMode = .chat
            engine.apply(["event": "status", "state": "generating", "message": "Generating for an app…"])
            guard engine.hasAPIGeneration, !engine.hasPlaygroundGeneration, engine.generationProgress == nil else { return false }
            engine.apply(apiProgress(apiFirst))
            guard engine.generationProgress?.sampledTokens == 0, engine.generationProgress?.inputTokens == 900,
                  engine.generationProgress?.reusedInputTokens == nil,
                  engine.generationProgress?.promptDetail == "900" else { return false }
            var wrongModel = apiProgress(apiFirst, sampled: 1, elapsed: 0.5)
            wrongModel["model"] = "replacement-model"
            var unsourced = apiProgress(apiFirst, sampled: 1, elapsed: 0.5)
            unsourced.removeValue(forKey: "source")
            var otherSource = apiProgress(apiFirst, sampled: 1, elapsed: 0.5)
            otherSource["source"] = "client"
            var changedInput = apiProgress(apiFirst, sampled: 1, elapsed: 0.5)
            changedInput["input_tokens"] = 901
            for stale in [wrongModel, unsourced, otherSource, changedInput,
                          apiProgress("openwork.1.2.3", sampled: 1, elapsed: 0.5),
                          apiProgress(apiFirst.uppercased(), sampled: 1, elapsed: 0.5),
                          apiProgress(apiFirst, sampled: 1, elapsed: 0.5, reused: 512)] {
                engine.apply(stale)
            }
            guard engine.generationProgress?.sampledTokens == 0 else { return false }
            engine.apply(apiProgress(apiFirst, sampled: 4, elapsed: 0.9))
            guard engine.generationProgress?.sampledTokens == 4, engine.generationProgress?.firstTokenSeconds == 0.5,
                  engine.readoutIsLive, engine.readoutFirstToken == 0.5, engine.readoutPrompt == "900" else { return false }
            engine.apply(["event": "metrics", "input_tokens": 900, "output_tokens": 3, "input_tokens_per_second": 1800.0,
                          "output_tokens_per_second": 7.5, "prefill_seconds": 0.5])
            guard engine.firstTokenSeconds == 0.5, engine.outputSpeed == 7.5, engine.inputTokens == 900,
                  engine.reusedInputTokens == nil else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.generationProgress == nil, !engine.hasAPIGeneration, !engine.readoutIsLive,
                  engine.readoutPrompt == "900", engine.readoutOutputSpeed == 7.5 else { return false }
            engine.apply(apiProgress(apiFirst, sampled: 5, elapsed: 1.0))
            guard engine.generationProgress == nil else { return false }
            // The next serialized request starts fresh; its predecessor cannot reappear.
            engine.apply(["event": "status", "state": "generating", "message": "Generating for an app…"])
            engine.apply(apiProgress(apiFirst, sampled: 6, elapsed: 1.1))
            guard engine.generationProgress == nil else { return false }
            engine.apply(apiProgress(apiSecond, reused: 512))
            guard engine.generationProgress?.reusedInputTokens == 512,
                  engine.generationProgress?.promptDetail == "900 (512 reused)" else { return false }
            engine.apply(apiProgress(apiSecond, sampled: 1, elapsed: 0.5))
            guard engine.generationProgress?.sampledTokens == 0 else { return false }
            engine.apply(["event": "metrics", "input_tokens": 900, "reused_input_tokens": 512, "output_tokens": 1,
                          "prefill_seconds": 0.5])
            guard engine.reusedInputTokens == 512 else { return false }
            engine.apply(["event": "metrics", "input_tokens": 900, "reused_input_tokens": 900, "output_tokens": 1])
            guard engine.reusedInputTokens == nil else { return false }
            engine.apply(["event": "status", "state": "loading", "message": "Loading model…"])
            guard engine.generationProgress == nil else { return false }
            engine.apply(["event": "status", "state": "generating", "message": "Generating for an app…"])
            engine.loadedID = "replacement-model"
            engine.apply(apiProgress(String(repeating: "c", count: 32)))
            return engine.generationProgress == nil
        case "generation-readout":
            // Fixed rows show dashes, then the running request, then the finished one.
            engine.mode = .chat; engine.loadedMode = .chat
            guard !engine.readoutIsLive, engine.readoutInputSpeed == nil, engine.readoutOutputSpeed == nil,
                  engine.readoutFirstToken == nil, engine.readoutPrompt == nil,
                  engine.readoutTime(at: Date()) == nil else { return false }
            engine.apply(["event": "metrics", "input_tokens": 19_608, "reused_input_tokens": 16_384, "output_tokens": 512,
                          "input_tokens_per_second": 1446.7, "output_tokens_per_second": 70.4,
                          "prefill_seconds": 2.2, "generation_seconds": 9.5])
            guard engine.readoutInputSpeed == 1446.7, engine.readoutFirstToken == 2.2,
                  engine.readoutPrompt == "19,608 (16,384 reused)",
                  engine.readoutTime(at: Date()).map({ $0 == ("9.5 s", nil) }) == true else { return false }
            engine.apply(["event": "status", "state": "generating", "message": "Generating for an app…"])
            guard engine.readoutIsLive, engine.readoutInputSpeed == nil, engine.readoutFirstToken == nil,
                  engine.readoutPrompt == nil, engine.readoutTime(at: Date()) == nil else { return false }
            engine.apply(apiProgress(apiFirst, reused: 512))
            guard engine.readoutPrompt == "900 (512 reused)", engine.readoutInputSpeed == nil,
                  engine.readoutTime(at: Date())?.detail.hasSuffix(" of ~4.0 s") == true else { return false }
            engine.apply(apiProgress(apiFirst, sampled: 4, elapsed: 0.9, reused: 512))
            guard let input = engine.readoutInputSpeed, abs(input - 776) < 1e-9,
                  let output = engine.readoutOutputSpeed, abs(output - 7.5) < 1e-9,
                  engine.readoutFirstToken == 0.5 else { return false }
            engine.apply(["event": "metrics", "input_tokens": 900, "reused_input_tokens": 512, "output_tokens": 4,
                          "input_tokens_per_second": 776.0, "output_tokens_per_second": 7.4,
                          "prefill_seconds": 0.5, "generation_seconds": 1.0])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard !engine.readoutIsLive, engine.readoutOutputSpeed == 7.4, engine.readoutPrompt == "900 (512 reused)",
                  engine.readoutTime(at: Date()).map({ $0 == ("1.0 s", nil) }) == true else { return false }
            // A Playground message shows dashes until its own progress arrives.
            engine.draft = "synthetic prompt"
            engine.generate(); _ = command()
            return engine.readoutIsLive && engine.readoutOutputSpeed == nil && engine.readoutPrompt == nil
        case "generation-progress-request-and-lifecycle":
            engine.mode = .chat; engine.loadedMode = .chat; engine.draft = "synthetic prompt"
            engine.generate()
            let requested = command()
            let id = engine.pendingMessageID!.uuidString.lowercased()
            guard requested["request_id"] as? String == id else { return false }
            engine.apply(generationProgress())
            guard engine.generationProgress?.sampledTokens == 0 else { return false }
            var wrongModel = generationProgress(sampled: 1, elapsed: 0.5)
            wrongModel["model"] = "replacement-model"
            engine.apply(wrongModel)
            engine.apply(generationProgress(id: UUID().uuidString, sampled: 1, elapsed: 0.5))
            guard engine.generationProgress?.sampledTokens == 0 else { return false }
            engine.apply(generationProgress(sampled: 4, elapsed: 0.9))
            guard engine.generationProgress?.sampledTokens == 4,
                  engine.generationProgress?.firstTokenSeconds == 0.5 else { return false }
            engine.apply(generationProgress(sampled: 3, elapsed: 1.0))
            engine.apply(generationProgress(sampled: 5, elapsed: 0.8))
            var changedFirst = generationProgress(sampled: 5, elapsed: 1.0)
            changedFirst["ttft_seconds"] = 0.6
            engine.apply(changedFirst)
            guard engine.generationProgress?.sampledTokens == 4 else { return false }
            engine.apply(["event": "reply", "request_id": UUID().uuidString, "text": "stale synthetic answer"])
            guard engine.pendingMessageID != nil, engine.messages.count == 1 else { return false }
            engine.apply(["event": "reply", "request_id": id, "text": "synthetic answer", "prefill_seconds": 0.5])
            guard engine.generationProgress == nil, engine.pendingMessageID == nil,
                  engine.firstTokenSeconds == 0.5, engine.messages.count == 2 else { return false }
            engine.apply(generationProgress(id: id, sampled: 8, elapsed: 1.2))
            return engine.generationProgress == nil
        case "generation-progress-validation":
            let base = generationProgress(id: UUID().uuidString)
            guard GenerationTimingProgress(base) != nil else { return false }
            for key in ["sampled_tokens", "output_token_budget", "elapsed_seconds", "ttft_lower_seconds",
                        "ttft_upper_seconds", "remaining_lower_seconds", "remaining_upper_seconds"] {
                for invalid: Any in [true, -1, "1", Double.nan, Double.infinity] {
                    var event = base; event[key] = invalid
                    guard GenerationTimingProgress(event) == nil else { return false }
                }
            }
            for (key, invalid): (String, Any) in [
                ("sampled_tokens", 0.5), ("sampled_tokens", 65), ("output_token_budget", 0),
                ("phase", "decode"), ("ttft_seconds", 0.5), ("ttft_lower_seconds", 0.9),
                ("remaining_upper_seconds", 1.0), ("remaining_lower_seconds", NSNull()), ("calibrated", 1)
            ] {
                var event = base; event[key] = invalid
                guard GenerationTimingProgress(event) == nil else { return false }
            }
            let counted = apiProgress(apiFirst, reused: 512)
            guard GenerationTimingProgress(counted)?.reusedInputTokens == 512 else { return false }
            for (key, invalid): (String, Any) in [
                ("input_tokens", -1), ("input_tokens", 0.5), ("input_tokens", true), ("input_tokens", NSNull()),
                ("reused_input_tokens", 900), ("reused_input_tokens", -1), ("reused_input_tokens", false),
                ("reused_input_tokens", "512")
            ] {
                var event = counted; event[key] = invalid
                guard GenerationTimingProgress(event) == nil else { return false }
            }
            var orphanReuse = counted; orphanReuse.removeValue(forKey: "input_tokens")
            guard GenerationTimingProgress(orphanReuse) == nil else { return false }
            var decode = generationProgress(id: UUID().uuidString, sampled: 1, elapsed: 0.4)
            guard GenerationTimingProgress(decode) == nil else { return false }
            decode["elapsed_seconds"] = 0.5
            guard GenerationTimingProgress(decode) != nil else { return false }
            var learning = base
            for key in ["ttft_lower_seconds", "ttft_upper_seconds", "remaining_lower_seconds", "remaining_upper_seconds"] {
                learning[key] = NSNull()
            }
            learning["calibrated"] = false
            guard let progress = GenerationTimingProgress(learning) else { return false }
            return progress.firstTokenRange == nil && progress.remainingRange == nil && !progress.calibrated
        case "generation-progress-clear-and-fence":
            engine.mode = .chat; engine.loadedMode = .chat; engine.draft = "synthetic prompt"
            engine.generate(); _ = command()
            let oldID = engine.pendingMessageID!.uuidString
            engine.apply(generationProgress())
            engine.stop(); _ = command()
            engine.apply(generationProgress(id: oldID, sampled: 2, elapsed: 0.7))
            guard engine.generationProgress == nil else { return false }
            engine.loadedMode = .chat; engine.state = "ready"; engine.draft = "synthetic retry"
            engine.generate(); _ = command()
            engine.apply(generationProgress(id: oldID, sampled: 2, elapsed: 0.7))
            guard engine.generationProgress == nil else { return false }
            engine.apply(generationProgress())
            guard engine.generationProgress != nil else { return false }
            engine.apply(["event": "error", "message": "Synthetic failure"])
            guard engine.generationProgress == nil else { return false }
            engine.state = "ready"; engine.draft = "synthetic retry"
            engine.generate(); _ = command(); engine.apply(generationProgress())
            engine.state = "loading"
            return engine.generationProgress == nil
        case "generation-progress-formatting":
            let now = Date(timeIntervalSinceReferenceDate: 0)
            var initial = generationProgress(id: UUID().uuidString)
            func close(_ value: Double?, _ expected: Double) -> Bool { value.map { abs($0 - expected) < 1e-9 } ?? false }
            // Elapsed time runs against the middle of the remaining range: 0 + (2 + 6) / 2.
            guard let progress = GenerationTimingProgress(initial, receivedAt: now),
                  progress.time(at: now) == ("0.0 s of ~4.0 s", 0),
                  progress.time(at: now.addingTimeInterval(1)) == ("1.0 s of ~4.0 s", 0.25),
                  progress.time(at: now.addingTimeInterval(5)) == ("5.0 s", nil),
                  progress.promptDetail == nil, progress.inputSpeed == nil, progress.outputSpeed == nil else { return false }
            var learning = initial
            learning["calibrated"] = false
            guard let uncalibrated = GenerationTimingProgress(learning, receivedAt: now),
                  uncalibrated.time(at: now.addingTimeInterval(12.4)) == ("12 s", nil) else { return false }
            initial["sampled_tokens"] = 2; initial["phase"] = "decode"
            initial["elapsed_seconds"] = 0.7; initial["ttft_seconds"] = 0.5
            guard let decoded = GenerationTimingProgress(initial, receivedAt: now),
                  close(decoded.outputSpeed, 5), decoded.inputSpeed == nil,
                  decoded.time(at: now) == ("0.7 s of ~4.7 s", 0.7 / 4.7),
                  decoded.time(at: now.addingTimeInterval(20)) == ("21 s", nil) else { return false }
            initial["input_tokens"] = 900; initial["reused_input_tokens"] = 512
            guard let reused = GenerationTimingProgress(initial, receivedAt: now),
                  close(reused.inputSpeed, 776), reused.promptDetail == "900 (512 reused)" else { return false }
            return GenerationTimingProgress.duration(120) == "2.0 min" &&
                GenerationTimingProgress.duration(7200) == "2.0 hr"
        case "mtp-advanced-options":
            engine.mode = .chat
            let capability: [String: Any] = ["supported": true, "max_drafts": 3,
                "recommended_drafts": 2, "default_drafts": 0, "operations": ["generate"],
                "sampling": "exact-rejection-v1", "verification": "fixture-exact-v1"]
            let ready: [String: Any] = ["event": "ready", "mode": "chat", "model_id": "fixture-model",
                "max_context_tokens": 8192, "context_tokens": 8192, "max_output_tokens": 4096,
                "max_new_tokens": 512, "mtp": capability, "mtp_enabled": true]
            engine.apply(ready)
            guard engine.supportsMTP, engine.mtpEnabled, engine.defaultMTPEnabled else { return false }
            var gemmaReady = ready
            var gemmaCapability = capability
            gemmaCapability["verification"] = "gemma4-e2b-target-verify-v1"
            gemmaReady["mtp"] = gemmaCapability; gemmaReady["mtp_enabled"] = false
            engine.apply(gemmaReady)
            guard engine.supportsMTP, !engine.mtpEnabled, !engine.defaultMTPEnabled else { return false }
            engine.apply(ready)
            guard engine.defaultMTPEnabled else { return false }
            let originalModels = engine.models
            for (group, family, precision, enabled) in [
                ("repo:google/gemma-4-E2B-it", "gemma4_text", "BF16", false),
                ("repo:Qwen/Qwen3.5-0.8B", "qwen3_5", "BF16", false),
                ("repo:Qwen/Qwen3.5-0.8B", "qwen3_5", "4-bit", true),
                ("repo:Qwen/Qwen3.5-4B", "qwen3_5", "BF16", true),
            ] {
                guard let display = InstalledModelDisplay(["schema_version": 1, "group_id": group,
                    "group_name": "Fixture", "family_id": family, "quantization": precision]) else { return false }
                engine.models = [InstalledModel(id: "fixture-model", name: "Fixture", path: "/fixture",
                    apiID: "fixture", display: display)]
                guard engine.defaultMTPEnabled == enabled else { return false }
            }
            guard let unverifiedBF16 = InstalledModelDisplay(["schema_version": 1,
                "group_id": "installed:fixture-model", "group_name": "Fixture", "quantization": "BF16"]) else { return false }
            engine.models = [InstalledModel(id: "fixture-model", name: "Fixture", path: "/fixture",
                apiID: "Qwen/Qwen3.5-0.8B", display: unverifiedBF16, source: "huggingface:Qwen/Qwen3.5-0.8B")]
            guard !engine.defaultMTPEnabled else { return false }
            engine.models = [InstalledModel(id: "fixture-model", name: "Qwen/Qwen3.5-0.8B", path: "/fixture",
                apiID: "Qwen/Qwen3.5-0.8B", display: unverifiedBF16, source: "local:/fixture")]
            guard engine.defaultMTPEnabled else { return false }
            engine.models = originalModels
            let options = engine.options
            guard !engine.configure(options, mtpEnabled: false, modelID: "stale"), noCommandSent(),
                  engine.configure(options, mtpEnabled: false, modelID: "fixture-model") else { return false }
            let requested = command()
            guard requested["command"] as? String == "configure",
                  requested["model_id"] as? String == "fixture-model",
                  requested["mtp_enabled"] as? Bool == false, engine.mtpEnabled else { return false }
            engine.apply(["event": "error", "message": "Cannot save MTP options. Try again."])
            guard engine.modelReady, engine.mtpEnabled,
                  engine.configure(options, mtpEnabled: false, modelID: "fixture-model") else { return false }
            _ = command()
            var configured = ready
            configured["event"] = "configured"; configured["mtp_enabled"] = false
            configured["model_id"] = "stale"
            engine.apply(configured)
            guard engine.state == "configuring", engine.mtpEnabled else { return false }
            configured["model_id"] = "fixture-model"
            // An API operation can finish before its buffered statuses reach
            // the UI, while the backend accepts the configure command afterward.
            engine.messages = [ChatMessage(text: "synthetic previous turn", isUser: true)]
            engine.apply(["event": "status", "state": "generating", "message": "Generating for an app…"])
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            engine.apply(configured)
            guard engine.supportsMTP, !engine.mtpEnabled, engine.modelReady, engine.messages.isEmpty else { return false }
            guard engine.configure(options, mtpEnabled: true, modelID: "fixture-model") else { return false }
            _ = command()
            engine.stop()
            _ = command()
            configured["mtp_enabled"] = true
            engine.apply(configured)
            guard !engine.supportsMTP, !engine.mtpEnabled else { return false }
            engine.apply(["event": "status", "state": "idle"])
            var unsupported = ready
            unsupported.removeValue(forKey: "mtp")
            engine.apply(unsupported)
            guard !engine.supportsMTP, !engine.mtpEnabled,
                  !engine.configure(options, mtpEnabled: true), noCommandSent(),
                  engine.configure(options) else { return false }
            return command()["mtp_enabled"] == nil
        case "context-model-defaults":
            engine.mode = .chat
            for limit in [2, 512, 32_768, 262_144, 1_048_576] {
                let output = limit - 1
                engine.apply(["event": "ready", "mode": "chat", "model_id": "fixture-model",
                              "max_context_tokens": limit, "context_tokens": limit,
                              "max_output_tokens": output, "max_new_tokens": 0])
                guard engine.contextLimit == limit, engine.options.contextTokens == limit,
                      engine.apiContextTokens == limit else { return false }
                let defaults = ChatOptions.defaults(contextLimit: engine.contextLimit, outputLimit: engine.outputLimit)
                guard defaults.contextTokens == limit, defaults.maxNewTokens == 0,
                      defaults.useModelSystemPrompt, defaults.systemPrompt.isEmpty, !defaults.thinking,
                      engine.configure(defaults) else { return false }
                let requested = command()
                guard requested["context_tokens"] as? Int == limit,
                      requested["max_new_tokens"] as? Int == 0 else { return false }
                engine.apply(["event": "configured", "max_context_tokens": limit, "context_tokens": limit,
                              "max_output_tokens": output, "max_new_tokens": 0])
                guard engine.options.contextTokens == limit, engine.state == "ready" else { return false }
            }
            return true
        case "api-keychain-failure-isolated":
            credentialStorage.status = errSecInteractionNotAllowed
            engine.retryAPI()
            guard pump({ !engine.apiCredentialBusy }), noCommandSent(), engine.apiKey.isEmpty, engine.state == "ready",
                  engine.loadedID == "fixture-model", engine.apiMessage.contains("Unlock") else { return false }
            credentialStorage.status = errSecSuccess
            engine.retryAPI()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            let configured = command()
            return configured["command"] as? String == "configure_api" &&
                configured["port"] as? Int == 8000 && configured["api_key"] as? String == String(repeating: "a", count: 64)
        case "api-reuse-rotation-and-connection-events":
            engine.retryAPI()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            _ = command()
            let original = engine.apiKey
            engine.apply(["event": "connection", "base_url": "http://127.0.0.1:8000/v1", "api_key": "ignored-untrusted-key"])
            guard engine.apiKey == original else { return false }
            engine.retryAPI()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            _ = command()
            guard engine.apiKey == original else { return false }
            credentialStorage.replacementStatus = errSecNotAvailable
            engine.regenerateAPIKey()
            guard pump({ !engine.apiCredentialBusy }), noCommandSent(), engine.apiKey == original else { return false }
            credentialStorage.replacementStatus = errSecSuccess
            engine.regenerateAPIKey()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            let changed = command()
            return engine.apiKey != original && changed["api_key"] as? String == engine.apiKey && engine.state == "ready"
        case "api-port-settings-and-conflict":
            engine.configureAPIPort("invalid")
            guard noCommandSent(), engine.apiPort == 8000, engine.state == "ready" else { return false }
            engine.configureAPIPort("12345")
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            let configured = command()
            engine.apply(["event": "connection", "base_url": "http://127.0.0.1:8000/v1", "message": "Port 12345 is occupied."])
            let reopened = EngineController(apiCredentials: credentials, apiDefaults: apiDefaults)
            return configured["port"] as? Int == 12345 && engine.apiPort == 12345 && reopened.apiPort == 12345 &&
                engine.apiBaseURL == "http://127.0.0.1:8000/v1" && engine.state == "ready" && engine.loadedID == "fixture-model"
        case "api-blocked-read-keeps-main-loop-usable":
            let gate = DispatchSemaphore(value: 0)
            credentialStorage.readGate = gate
            defer { gate.signal() }
            engine.retryAPI()
            guard engine.apiCredentialBusy, noCommandSent(), engine.state == "ready" else { return false }
            for _ in 0..<20 {
                engine.retryAPI(); engine.regenerateAPIKey(); engine.configureAPIPort("12345")
            }
            guard engine.apiPort == 8000, engine.send(["command": "list"]), command()["command"] as? String == "list",
                  pump(timeout: 3, { engine.apiMessage.contains("Waiting for macOS Keychain access") }),
                  engine.apiCredentialBusy, credentialStorage.readCount == 1, engine.loadedID == "fixture-model" else { return false }
            gate.signal()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            return command()["command"] as? String == "configure_api" && credentialStorage.readCount == 1
        case "api-blocked-replacement-remains-pending":
            engine.retryAPI()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            _ = command()
            let original = engine.apiKey
            let gate = DispatchSemaphore(value: 0)
            credentialStorage.replaceGate = gate
            defer { gate.signal() }
            engine.regenerateAPIKey()
            guard pump({ credentialStorage.replaceCount == 1 }),
                  pump(timeout: 3, { engine.apiMessage.contains("replacement is still pending") }),
                  engine.apiCredentialBusy, engine.apiKey == original, noCommandSent(), engine.state == "ready" else { return false }
            engine.retryAPI(); engine.regenerateAPIKey()
            gate.signal()
            guard pump({ !engine.apiCredentialBusy }) else { return false }
            let changed = command()
            return engine.apiKey != original && changed["api_key"] as? String == engine.apiKey && noCommandSent()
        case "api-late-credential-result-is-fenced":
            let gate = DispatchSemaphore(value: 0)
            credentialStorage.readGate = gate
            defer { gate.signal() }
            engine.retryAPI()
            guard pump({ credentialStorage.readCount == 1 }) else { return false }
            let replacementPipe = Pipe()
            engine.input = replacementPipe.fileHandleForWriting
            gate.signal()
            guard pump({ !engine.apiCredentialBusy }), engine.apiKey.isEmpty, noCommandSent() else { return false }
            var descriptor = pollfd(fd: replacementPipe.fileHandleForReading.fileDescriptor, events: Int16(POLLIN), revents: 0)
            return Darwin.poll(&descriptor, 1, 0) == 0
        case "api-credential-result-after-shutdown-is-fenced":
            let gate = DispatchSemaphore(value: 0)
            credentialStorage.readGate = gate
            defer { gate.signal() }
            engine.retryAPI()
            guard pump({ credentialStorage.readCount == 1 }) else { return false }
            engine.shuttingDown = true
            gate.signal()
            return pump({ !engine.apiCredentialBusy }) && engine.apiKey.isEmpty && noCommandSent()
        case "gpu-memory-live-and-validation":
            func memory(_ bytes: Any, model: String = "fixture-model", mode: String = "decision") {
                engine.apply(["event": "gpu_memory", "model_id": model, "mode": mode, "gpu_memory_bytes": bytes])
            }
            memory(Int64(2_147_483_648))
            guard engine.gpuMemoryBytes == 2_147_483_648, engine.gpuMemoryText == "GPU memory 25%",
                  engine.gpuMemoryHelp.contains("2 GB"), engine.gpuMemoryHelp.contains("8 GB") else { return false }
            engine.state = "deciding"
            memory(Int64(1_073_741_824))
            guard engine.gpuMemoryBytes == 1_073_741_824 else { return false }
            engine.selectedID = "another-selection"
            memory(512)
            memory(999, model: "another-selection")
            guard engine.gpuMemoryBytes == 512 else { return false }
            for invalid: Any in [true, -1, 1.5, "42", NSNull(), NSNumber(value: UInt64.max)] {
                memory(invalid)
                guard engine.gpuMemoryBytes == nil else { return false }
            }
            memory(0)
            return engine.gpuMemoryBytes == 0
        case "gpu-memory-budget-indicator":
            let indicator = EngineController(apiCredentials: credentials, apiDefaults: apiDefaults, gpuMemoryBudgetBytes: 1000)
            for (bytes, color, percent): (Int64, Color, String) in [
                (0, .green, "0%"), (740, .green, "74%"), (749, .green, "74%"), (750, .orange, "75%"),
                (890, .orange, "89%"), (899, .orange, "89%"), (900, .red, "90%"),
                (999, .red, "99%"), (1000, .red, "100%"), (1200, .red, "120%")
            ] {
                indicator.gpuMemoryBytes = bytes
                guard indicator.gpuMemoryColor == color, indicator.gpuMemoryText == "GPU memory " + percent,
                      indicator.gpuMemoryFraction == Double(bytes) / 1000 else { return false }
            }
            for budget: UInt64? in [nil, 0] {
                let unavailable = EngineController(apiCredentials: credentials, apiDefaults: apiDefaults,
                                                    gpuMemoryBudgetBytes: budget)
                unavailable.gpuMemoryBytes = 2_147_483_648
                guard unavailable.gpuMemoryFraction == nil, unavailable.gpuMemoryColor == .secondary,
                      unavailable.gpuMemoryText.contains("2 GB"),
                      unavailable.gpuMemoryHelp.contains("budget is unavailable") else { return false }
            }
            indicator.gpuMemoryBytes = nil
            return indicator.gpuMemoryFraction == nil && indicator.gpuMemoryColor == .secondary &&
                indicator.gpuMemoryText == "GPU memory —"
        case "gpu-memory-stop-and-mode-fence":
            let sample: [String: Any] = ["event": "gpu_memory", "model_id": "fixture-model",
                                         "mode": "decision", "gpu_memory_bytes": 4096]
            engine.apply(sample)
            engine.stop()
            _ = command()
            engine.apply(sample)
            guard engine.gpuMemoryBytes == nil, engine.gpuMemoryText == "GPU memory —" else { return false }
            engine.apply(["event": "status", "state": "idle"])
            engine.changeMode(.chat)
            guard engine.run() else { return false }
            _ = command()
            engine.apply(sample)
            guard engine.gpuMemoryBytes == nil else { return false }
            engine.apply(["event": "gpu_memory", "model_id": "fixture-model", "mode": "chat", "gpu_memory_bytes": 8192])
            guard engine.gpuMemoryBytes == 8192 else { return false }
            engine.apply(["event": "status", "state": "error"])
            return engine.gpuMemoryBytes == nil
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
                  registry.families[0].modes[.text]?.status == .conditional,
                  registry.families[0].modes[.visionLanguage]?.status == .unsupported else { return false }
            var future = familyRegistry()
            future["schema_version"] = 2
            engine.apply(["event": "model_families", "report": future])
            engine.requestModelFamilies()
            return engine.modelFamilies?.families.first?.id == "fixture" && command()["command"] as? String == "support"
        case "support-live-contract":
            guard let payload = try? JSONSerialization.jsonObject(with: Data(backendError.utf8)) as? [String: Any],
                  let registry = payload["registry"] as? [String: Any], let report = payload["report"] as? [String: Any] else { return false }
            engine.apply(["event": "model_families", "report": registry])
            guard let families = engine.modelFamilies?.families,
                  let declared = registry["families"] as? [[String: Any]], families.count == declared.count,
                  let classifier = families.first(where: { $0.modelKind == "classifier" }),
                  classifier.modes.values.allSatisfy({ $0.status == .unsupported }),
                  let qwen3 = families.first(where: { $0.id == "qwen3" }),
                  qwen3.sourceFormats == ["safetensors-float", "mlx-affine-4bit", "mlx-affine-8bit"],
                  qwen3.outputDtypes == ["bfloat16", "float16", "float32"],
                  let glm = families.first(where: { $0.id == "glm5_next" }),
                  glm.containers == [".gguf"], glm.sourceFormats == ["UD-Q2_K_XL"],
                  glm.hasAppMode(.text), !glm.hasAppMode(.visionLanguage), !glm.hasAppMode(.decision),
                  families.first(where: { $0.id == "qwen3_5" })?.modes[.visionLanguage]?.status == .conditional else { return false }
            engine.install(source: "fixture/source", kind: "huggingface", artifactMode: "vision-language")
            let requested = command()
            engine.apply(["event": "model_support", "request_id": requested["support_request_id"]!, "report": report])
            return engine.installSupportReport?.status == .inconclusive &&
                engine.installSupportReport?.modes.values.allSatisfy({ $0 == .inconclusive }) == true
        case "models-downloaded-sort":
            let models = [
                InstalledModel(id: "third", name: "Model 10", path: "/private/third", apiID: "private-third"),
                InstalledModel(id: "second", name: "Model 2", path: "/private/second", apiID: "private-second"),
                InstalledModel(id: "first", name: "Model 1", path: "/private/first", apiID: "private-first")
            ]
            let inventory = ModelsInventory(models: models, loadedID: "", registry: nil)
            guard inventory.models.map(\.name) == ["Model 1", "Model 2", "Model 10"],
                  inventory.models.allSatisfy({ !$0.isRunning }) else { return false }
            let duplicates = [
                InstalledModel(id: "b", name: "Exact model name", path: "/private/b", apiID: "private-b"),
                InstalledModel(id: "a", name: "Exact model name", path: "/private/a", apiID: "private-a")
            ]
            return ModelsInventory(models: duplicates, loadedID: "b", registry: nil).models.map(\.id) == ["a", "b"]
        case "models-running-identity":
            engine.models = [
                InstalledModel(id: "loaded", name: "Same model name", path: "/private/loaded", apiID: "same-alias"),
                InstalledModel(id: "selected", name: "Same model name", path: "/private/selected", apiID: "same-alias")
            ]
            engine.selectedID = "selected"
            engine.loadedID = "loaded"
            let rows = ModelsInventory(models: engine.models, loadedID: engine.loadedID, registry: nil).models
            guard rows.filter(\.isRunning).map(\.id) == ["loaded"], engine.selectedID == "selected" else { return false }
            return ModelsInventory(models: engine.models, loadedID: "missing", registry: nil).models.allSatisfy({ !$0.isRunning })
        case "models-empty-state":
            let inventory = ModelsInventory(models: [], loadedID: "missing", registry: nil)
            guard inventory.models.isEmpty, inventory.families == nil,
                  ModelsSheetTab.initial == .installed,
                  ModelsSheetTab.allCases == [.installed, .download, .installModel],
                  ModelsSheetTab.installed.title(modelCount: 0) == "Installed (0)",
                  ModelsSheetTab.download.title(modelCount: 0) == "Model Catalog",
                  ModelsSheetTab.installModel.title(modelCount: 0) == "Add from Source",
                  ModelsSheetTab.installed.emptyTitle == "No models yet",
                  ModelsSheetTab.installed.emptyMessage == "Install a model to add it here." else { return false }
            var registry = familyRegistry()
            registry["families"] = [[String: Any]]()
            guard let parsed = ModelFamilyRegistry(registry) else { return false }
            return ModelsInventory(models: [], loadedID: "", registry: parsed).families?.isEmpty == true
        case "models-family-inventory":
            guard var raw = try? JSONSerialization.jsonObject(with: Data(backendError.utf8)) as? [String: Any],
                  let families = raw["families"] as? [[String: Any]] else { return false }
            raw["families"] = Array(families.reversed())
            guard let registry = ModelFamilyRegistry(raw),
                  let rows = ModelsInventory(models: [], loadedID: "", registry: registry).families else { return false }
            return rows.map(\.name) == ["Flint", "Gemma 2", "Gemma 3", "Gemma 3 images", "Gemma 4", "GLM-5.3 Flash", "Granite", "LFM2", "Llama", "LLaMBA",
                "Mistral", "OLMo 3", "Phi-3", "Qwen2", "Qwen3", "Qwen3.5", "Qwen3.5 MoE", "Qwen3.8 Flash Next", "SmolLM3"] &&
                !rows.contains(where: { $0.id == "modernbert" })
        case "models-display-event":
            engine.apply(["event": "models", "models": [
                ["id": "fixture-model", "name": "Exact saved model name", "path": "/private/model", "display": modelDisplay()]
            ]])
            guard let model = engine.models.first, model.name == "Exact saved model name",
                  model.display?.groupID == "example/base", model.display?.familyID == "qwen3",
                  model.display?.quantization == "BF16", model.display?.weightBytes == 1_190_000_000,
                  engine.loadedID == "fixture-model", !engine.supportsImages else { return false }
            let inventory = ModelsInventory(models: engine.models, loadedID: engine.loadedID, registry: nil)
            guard inventory.groups.count == 1, inventory.groups[0].name == "Base", inventory.groups[0].family == "Qwen3",
                  inventory.groups[0].models[0].isRunning, inventory.models[0].detailLabels.count == 4,
                  inventory.models[0].detailLabels.first == "BF16",
                  inventory.models[0].detailLabels.last == "Version aaaaaaa" else { return false }
            var unknown = modelDisplay()
            for key in ["family_id", "quantization", "weight_bytes", "source_revision"] { unknown[key] = NSNull() }
            guard let descriptor = InstalledModelDisplay(unknown) else { return false }
            return descriptor.familyID == nil && descriptor.quantization == nil && descriptor.weightBytes == nil && descriptor.sourceRevision == nil
        case "models-display-native-families":
            for (familyID, familyName, repository) in [
                ("olmo3", "OLMo 3", "allenai/Olmo-3-7B-Instruct"),
                ("gemma3", "Gemma 3 images", "google/gemma-3-4b-it"),
                ("qwen3_5_moe", "Qwen3.5 MoE", "mlx-community/Qwen3.5-35B-A3B-4bit"),
                ("gemma4_text", "Gemma 4", "google/gemma-4-E2B-it")
            ] {
                var raw = modelDisplay("repo:" + repository, name: repository.split(separator: "/").last.map(String.init) ?? repository)
                raw["family_id"] = familyID
                guard ModelFamily.displayName(for: familyID) == familyName,
                      let display = InstalledModelDisplay(raw),
                      display.repositoryURL?.absoluteString == "https://huggingface.co/" + repository,
                      display.familyID == familyID, display.quantization == "BF16",
                      display.weightBytes == 1_190_000_000,
                      display.sourceRevision == String(repeating: "a", count: 40) else { return false }
                let inventory = ModelsInventory(
                    models: [InstalledModel(id: familyID, name: repository, path: "/private/" + familyID,
                                            apiID: familyID, display: display)],
                    loadedID: "", registry: nil
                )
                guard inventory.groups.count == 1,
                      inventory.groups[0].family == familyName,
                      inventory.groups[0].name == repository.split(separator: "/").last.map(String.init) else { return false }
            }
            return true
        case "models-display-malformed":
            let mutations: [(String, Any)] = [
                ("schema_version", 2), ("schema_version", true), ("group_id", ""),
                ("group_id", String(repeating: "a", count: 513)), ("group_name", String(repeating: "a", count: 257)),
                ("group_name", "Two\nlines"), ("group_name", "Control\tname"), ("group_name", "Control\u{7f}name"),
                ("family_id", "unrecognized"), ("quantization", 4),
                ("weight_bytes", true), ("weight_bytes", -1), ("weight_bytes", 1.5),
                ("weight_bytes", Double.infinity), ("source_revision", String(repeating: "A", count: 40)),
                ("source_revision", "aaaaaaa")
            ]
            for (key, value) in mutations {
                var raw = modelDisplay()
                raw[key] = value
                engine.apply(["event": "models", "models": [
                    ["id": "fixture-model", "name": "Saved name 4-bit", "path": "/private/model", "display": raw]
                ]])
                guard let model = engine.models.first, model.display == nil else { return false }
                let inventory = ModelsInventory(models: engine.models, loadedID: "", registry: nil)
                guard inventory.groups.count == 1, inventory.groups[0].name == "Saved name 4-bit",
                      inventory.groups[0].family == nil, inventory.models[0].detailLabels == ["Tools unchecked"] else { return false }
            }
            return true
        case "models-tool-support-labels":
            let previousSupport = engine.supportsTools
            for (status, expected) in [("available", "Tools available"), ("unavailable", "Tools unavailable"),
                                       ("unchecked", "Tools unchecked"), ("unknown", "Tools unchecked")] {
                var raw = modelDisplay()
                raw["tool_support"] = ["status": status, "model_behavior": "unverified"]
                engine.apply(["event": "models", "models": [
                    ["id": "fixture-model", "name": "Exact saved name", "path": "/private/model", "display": raw]
                ]])
                let inventory = ModelsInventory(models: engine.models, loadedID: engine.loadedID, registry: nil)
                guard inventory.models.first?.toolSupport.title == expected,
                      engine.supportsTools == previousSupport else { return false }
            }
            return ModelToolSupport(nil) == .unchecked && ModelToolSupport(["status": true]) == .unchecked
        case "models-grouped-variants":
            var compact = modelDisplay()
            compact["quantization"] = "4-bit"; compact["weight_bytes"] = Int64(300_000_000)
            compact["source_revision"] = String(repeating: "b", count: 40)
            let models = [
                variant("bf16", name: "Saved 8-bit name", display: modelDisplay()),
                variant("compact", name: "Compact saved name", display: compact),
                variant("fine-tune", name: "Fine-tune saved name", display: modelDisplay("example/fine-tune", name: "Fine Tune"))
            ]
            let inventory = ModelsInventory(models: models, loadedID: "bf16", registry: nil)
            guard inventory.models.count == 3, inventory.groups.map(\.name) == ["Base", "Fine Tune"],
                  inventory.groups[0].models.map(\.id) == ["compact", "bf16"],
                  inventory.groups[0].models.filter(\.isRunning).map(\.id) == ["bf16"],
                  inventory.groups[0].models[1].name == "Saved 8-bit name" else { return false }
            return inventory.groups[0].models[1].quantization == "BF16" &&
                inventory.groups[0].models[0].detailLabels.last == "Version bbbbbbb"
        case "models-grouping-ambiguity":
            let first = variant("a", name: "First exact name", display: modelDisplay())
            let conflict = variant("b", name: "Second exact name", display: modelDisplay("example/base", name: "Conflicting base"))
            let inventory = ModelsInventory(models: [conflict, first], loadedID: "", registry: nil)
            guard inventory.groups.map(\.name) == ["First exact name", "Second exact name"],
                  inventory.groups.allSatisfy({ $0.models.count == 1 && $0.family == nil }) else { return false }
            let absent = [variant("a", name: "Same exact name", display: nil), variant("b", name: "Same exact name", display: nil)]
            return ModelsInventory(models: absent, loadedID: "", registry: nil).groups.count == 2
        case "models-variant-order":
            var compact = modelDisplay(); compact["quantization"] = "4-bit"
            var larger = modelDisplay(); larger["weight_bytes"] = Int64(8_410_000_000)
            var unknown = modelDisplay(); unknown["quantization"] = NSNull(); unknown["weight_bytes"] = Int64(100)
            let models = [
                variant("unknown", name: "Unknown", display: unknown), variant("larger", name: "Larger", display: larger),
                variant("bf16", name: "BF16", display: modelDisplay()), variant("compact", name: "Compact", display: compact),
                variant("other10", name: "Other exact10", display: modelDisplay("other10", name: "Model 10")),
                variant("other2", name: "Other exact2", display: modelDisplay("other2", name: "Model 2"))
            ]
            let inventory = ModelsInventory(models: models, loadedID: "", registry: nil)
            return inventory.groups.map(\.name) == ["Base", "Model 2", "Model 10"] &&
                inventory.groups[0].models.map(\.id) == ["compact", "bf16", "larger", "unknown"]
        case "saved-row-run-private-id":
            engine.models = [
                InstalledModel(id: "model-a", name: "Same name", path: "/private/model-a", apiID: "owner/Same"),
                InstalledModel(id: "model-b", name: "Same name", path: "/private/model-b", apiID: "owner/Same")
            ]
            engine.selectedID = "model-a"; engine.loadedID = "model-a"
            guard engine.runModel(id: "model-b") else { return false }
            let request = command()
            guard request["command"] as? String == "run", request["model_id"] as? String == "model-b",
                  request["mode"] as? String == "decision", engine.selectedID == "model-b",
                  UserDefaults.standard.string(forKey: "selectedModel") == "model-b",
                  engine.state == "loading", !engine.modelReady else { return false }
            engine.apply(["event": "ready", "mode": "decision", "model_id": "model-a"])
            guard !engine.modelReady else { return false }
            engine.apply(["event": "ready", "mode": "decision", "model_id": "model-b"])
            return engine.modelReady && engine.loadedID == "model-b"
        case "saved-row-run-guards":
            engine.models = [
                InstalledModel(id: "fixture-model", name: "Current", path: "/private/current", apiID: "current"),
                InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "other")
            ]
            engine.messages = [ChatMessage(text: "synthetic retained turn", isUser: true)]
            engine.state = "generating"
            guard !engine.runModel(id: "other"), engine.selectedID == "fixture-model", engine.state == "generating",
                  engine.messages.count == 1 else { return false }
            engine.state = "ready"
            guard !engine.runModel(id: "missing"), engine.selectedID == "fixture-model", engine.modelReady,
                  engine.messages.count == 1 else { return false }
            return noCommandSent()
        case "saved-row-run-already-loaded":
            engine.mode = .chat; engine.loadedMode = .chat
            engine.models = [
                InstalledModel(id: "fixture-model", name: "Current", path: "/private/current", apiID: "current"),
                InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "other")
            ]
            engine.selectedID = "other"
            engine.messages = [ChatMessage(text: "synthetic retained turn", isUser: true)]
            engine.supportsImages = true; engine.imageVerification = "locally_tested"
            guard !engine.modelReady, engine.runModel(id: "fixture-model") else { return false }
            return engine.modelReady && engine.loadedID == "fixture-model" && engine.messages.count == 1 &&
                engine.supportsImages && engine.imageVerification == "locally_tested" && noCommandSent()
        case "saved-row-run-mode-mismatch":
            engine.mode = .chat; engine.loadedMode = .decision
            engine.models = [InstalledModel(id: "fixture-model", name: "Current", path: "/private/current", apiID: "current")]
            engine.messages = [ChatMessage(text: "synthetic previous turn", isUser: true)]
            guard engine.runModel(id: "fixture-model") else { return false }
            let request = command()
            return request["model_id"] as? String == "fixture-model" && request["mode"] as? String == "chat" &&
                engine.state == "loading" && !engine.modelReady && engine.messages.isEmpty
        case "saved-row-run-unavailable-runtime":
            engine.models = [InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "other")]
            engine.input = nil
            return !engine.runModel(id: "other") && engine.state == "error" && !engine.modelReady && noCommandSent()
        case "saved-row-control-switch":
            engine.models = [
                InstalledModel(id: "fixture-model", name: "Same name", path: "/private/current", apiID: "owner/Same"),
                InstalledModel(id: "other", name: "Same name", path: "/private/other", apiID: "owner/Same")
            ]
            guard engine.modelControlID == "fixture-model", engine.canStopModel(id: "fixture-model"),
                  !engine.canStopModel(id: "other"), engine.runModel(id: "other") else { return false }
            let request = command()
            guard request["command"] as? String == "run", request["model_id"] as? String == "other",
                  engine.modelControlID == "other", engine.canStopModel(id: "other"),
                  !engine.canStopModel(id: "fixture-model"), engine.busy else { return false }
            engine.apply(["event": "ready", "mode": "decision", "model_id": "fixture-model"])
            guard engine.state == "loading", engine.modelControlID == "other" else { return false }
            engine.apply(["event": "ready", "mode": "decision", "model_id": "other"])
            guard engine.modelReady, engine.modelControlID == "other", engine.canStopModel(id: "other"),
                  !engine.canStopModel(id: "fixture-model") else { return false }
            engine.stopModel(id: "other")
            guard command()["command"] as? String == "stop", engine.state == "stopping",
                  !engine.canStopModel(id: "other") else { return false }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return engine.modelControlID.isEmpty && !engine.canStopModel(id: "other")
        case "saved-row-control-cancel-load":
            engine.models = [
                InstalledModel(id: "fixture-model", name: "Current", path: "/private/current", apiID: "current"),
                InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "other")
            ]
            guard engine.runModel(id: "other") else { return false }
            _ = command()
            engine.stopModel(id: "other")
            guard command()["command"] as? String == "stop", engine.state == "stopping",
                  engine.modelControlID == "other", !engine.canStopModel(id: "other") else { return false }
            engine.stopModel(id: "other")
            engine.apply(["event": "ready", "mode": "decision", "model_id": "other"])
            guard engine.state == "stopping", !engine.modelReady, engine.modelControlID == "other",
                  noCommandSent() else { return false }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return engine.modelControlID.isEmpty && engine.loadedID.isEmpty
        case "saved-row-control-identity-guards":
            let current = InstalledModel(id: "fixture-model", name: "Same name", path: "/private/current", apiID: "owner/Same")
            let other = InstalledModel(id: "other", name: "Same name", path: "/private/other", apiID: "owner/Same")
            engine.models = [current, other]
            engine.selectedID = "other"
            guard engine.modelControlID == current.id, engine.canStopModel(id: current.id),
                  !engine.canStopModel(id: other.id), !engine.canStopModel(id: "missing"),
                  !engine.canStopModel(id: "") else { return false }
            engine.stopModel(id: other.id); engine.stopModel(id: "missing"); engine.stopModel(id: "")
            guard engine.state == "ready", engine.loadedID == current.id, noCommandSent() else { return false }
            engine.models = [other]
            engine.stopModel(id: current.id)
            return !engine.canStopModel(id: current.id) && engine.state == "ready" &&
                engine.loadedID == current.id && noCommandSent()
        case "saved-row-control-stop-generation":
            engine.mode = .chat; engine.loadedMode = .chat
            engine.models = [InstalledModel(id: "fixture-model", name: "Current", path: "/private/current", apiID: "current")]
            engine.state = "generating"
            engine.messages = [ChatMessage(text: "synthetic retained turn", isUser: true)]
            guard engine.busy, engine.canStopModel(id: "fixture-model") else { return false }
            engine.stopModel(id: "fixture-model")
            return command()["command"] as? String == "stop" && engine.state == "stopping" &&
                engine.messages.isEmpty && !engine.canStopModel(id: "fixture-model")
        case "restart-controls-stop-and-start":
            let current = variant("fixture-model", name: "Current", display: nil)
            let selected = variant("other-model", name: "Selected", display: nil)
            engine.models = [current, selected]
            engine.mode = .chat; engine.loadedMode = .decision
            engine.loadedID = current.id; engine.selectedID = selected.id; engine.state = "ready"
            guard engine.modelControlID == current.id, engine.canStopCurrentModel,
                  !engine.canStartModel else { return false }
            engine.stopModel(id: engine.modelControlID)
            guard command()["command"] as? String == "stop", engine.state == "stopping",
                  engine.stoppedModel?.id == current.id, engine.stoppedModel?.mode == .decision,
                  engine.selectedID == selected.id, !engine.canStopCurrentModel else { return false }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            guard engine.state == "idle", engine.loadedID.isEmpty, engine.loadedMode == nil,
                  engine.startableModel?.id == current.id, engine.canStartModel else { return false }
            engine.startModel()
            let restart = command()
            return restart["command"] as? String == "run" && restart["model_id"] as? String == current.id &&
                restart["mode"] as? String == "decision" && engine.selectedID == current.id &&
                engine.mode == .decision && engine.state == "loading"
        case "restart-controls-explicit-target-after-failed-start":
            let stopped = variant("fixture-model", name: "Stopped A", display: nil)
            let selected = variant("other-model", name: "Selected B", display: nil)
            engine.models = [stopped, selected]
            engine.mode = .chat; engine.loadedMode = .chat
            engine.loadedID = stopped.id; engine.selectedID = stopped.id; engine.state = "ready"
            engine.stopModel(id: stopped.id)
            guard command()["command"] as? String == "stop", engine.stoppedModel?.id == stopped.id else {
                return false
            }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            guard engine.stoppedModel?.id == stopped.id, engine.state == "idle" else { return false }

            // A rejected busy run must preserve the valid restart target.
            engine.state = "generating"
            guard !engine.run(), engine.stoppedModel?.id == stopped.id, noCommandSent() else { return false }
            engine.state = "idle"

            guard engine.runModel(id: selected.id) else { return false }
            let firstAttempt = command()
            guard firstAttempt["command"] as? String == "run",
                  firstAttempt["model_id"] as? String == selected.id,
                  firstAttempt["mode"] as? String == "chat", engine.selectedID == selected.id else { return false }
            engine.apply(["event": "error", "message": "Synthetic startup failure"])
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            guard engine.state == "idle", engine.loadedID.isEmpty,
                  engine.stoppedModel?.id != stopped.id,
                  engine.startableModel?.id == selected.id, engine.canStartModel else { return false }
            engine.startModel()
            let retry = command()
            return retry["command"] as? String == "run" && retry["model_id"] as? String == selected.id &&
                retry["mode"] as? String == "chat" && engine.selectedID == selected.id && engine.state == "loading"
        case "restart-controls-running-busy-and-deleted-target":
            let current = variant("fixture-model", name: "Current", display: nil)
            engine.models = [current, variant("other-model", name: "Other", display: nil)]
            engine.mode = .chat; engine.loadedMode = .chat
            engine.loadedID = current.id; engine.selectedID = "other-model"; engine.state = "generating"
            guard engine.modelControlID == current.id, engine.canStopCurrentModel,
                  !engine.canStartModel else { return false }
            engine.stopModel(id: engine.modelControlID)
            guard command()["command"] as? String == "stop", engine.state == "stopping",
                  !engine.canStartModel, !engine.canStopCurrentModel else { return false }
            engine.startModel()
            guard noCommandSent(), engine.state == "stopping" else { return false }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            engine.models.removeAll { $0.id == current.id }
            engine.selectedID = "missing"
            guard !engine.canStartModel, engine.startableModel == nil else { return false }
            engine.startModel()
            guard noCommandSent(), engine.state == "idle", engine.loadedID.isEmpty else { return false }
            // A removed remembered model must not shadow a valid new selection
            // or apply the deleted model's old mode to that new selection.
            engine.selectedID = "other-model"; engine.mode = .decision
            guard engine.canStartModel, engine.startableModel?.id == "other-model" else { return false }
            engine.startModel()
            let fallback = command()
            return fallback["command"] as? String == "run" && fallback["model_id"] as? String == "other-model" &&
                fallback["mode"] as? String == "decision" && engine.state == "loading"
        case "saved-row-control-unrelated-operation":
            engine.models = [
                InstalledModel(id: "fixture-model", name: "Current", path: "/private/current", apiID: "current"),
                InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "other",
                               removal: InstalledModelRemoval(removal()))
            ]
            for state in ["installing", "removing"] {
                engine.state = state
                engine.stopModel(id: "fixture-model")
                guard !engine.canStopModel(id: "fixture-model"), engine.state == state,
                      noCommandSent() else { return false }
            }
            engine.state = "ready"
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            _ = command()
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            engine.stopModel(id: "fixture-model")
            guard engine.catalogInstallation != nil, !engine.canStopModel(id: "fixture-model"),
                  engine.state == "ready", noCommandSent() else { return false }
            engine.catalogInstallation = nil; engine.activeInstallRequestID = nil
            engine.removeModel(id: "other", expectedAction: .trash)
            _ = command()
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            engine.stopModel(id: "fixture-model")
            return engine.removingModelID == "other" && !engine.canStopModel(id: "fixture-model") &&
                engine.state == "ready" && engine.loadedID == "fixture-model" && noCommandSent()
        case "models-artifact-mode-label":
            let descriptor = InstalledModelDisplay(modelDisplay())
            let models = [
                InstalledModel(id: "images", name: "Same saved name", path: "/private/images", apiID: "images", display: descriptor, artifactMode: .visionLanguage),
                InstalledModel(id: "text", name: "Same saved name", path: "/private/text", apiID: "text", display: descriptor, artifactMode: .text)
            ]
            let inventory = ModelsInventory(models: models, loadedID: "text", registry: nil)
            guard inventory.groups.count == 1, inventory.groups[0].models.map(\.id) == ["text", "images"] else { return false }
            return !inventory.groups[0].models[0].detailLabels.contains("Text and images") &&
                inventory.groups[0].models[1].detailLabels.contains("Text and images") &&
                inventory.groups[0].models.filter(\.isRunning).map(\.id) == ["text"]
        case "models-readable-names":
            var lily = modelDisplay("repo:owner/Qwen3.8-Flash-Next-lily-q4-xl", name: "Qwen3.8-Flash-Next-lily-q4-xl")
            lily["quantization"] = "Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed)"
            var mlxServe = modelDisplay("repo:owner/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit", name: "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit")
            mlxServe["quantization"] = "MLX-Serve mixed 4/8-bit"
            var mlx = modelDisplay("repo:owner/Qwen3.8-Flash-Next-MLX-mixed-4_8bit", name: "Qwen3.8-Flash-Next-MLX-mixed-4_8bit")
            mlx["quantization"] = "MLX mixed 4/8-bit"
            var compact = modelDisplay("repo:owner/Qwen3-4B-4bit", name: "Qwen3-4B-4bit")
            compact["quantization"] = "4-bit"
            let saved = "qwen38-flash-next-lily-q4-xl-0f94842e"
            let dense = modelDisplay("repo:owner/Qwen3-0.6B", name: "Qwen3-0.6B")
            let fixtures = [variant("lily", name: saved, display: lily),
                            variant("mlx-serve", name: "qwen38-flash-next-mixed-4-8bit-788c07a8", display: mlxServe),
                            variant("mlx", name: "qwen38-flash-next-mixed-4-8bit-b2c422f3", display: mlx),
                            variant("mixed", name: "qwen38-flash-next-mixed-4-8bit-788c07a8", display: nil),
                            variant("small", name: "qwen3-4b-4bit", display: compact),
                            variant("dense", name: "qwen3-dense", display: dense),
                            variant("custom", name: "JevK5", display: nil)]
            let inventory = ModelsInventory(models: fixtures, loadedID: "lily", registry: nil)
            let rows = Dictionary(uniqueKeysWithValues: inventory.models.map { ($0.id, $0) })
            guard rows["lily"]?.displayName == "Qwen3.8 Flash Next", rows["lily"]?.name == saved,
                  rows["lily"]?.quantization == lily["quantization"] as? String,
                  rows["lily"]?.variantLabel == "Lily q4",
                  rows["mlx-serve"]?.displayName == "Qwen3.8 Flash Next", rows["mlx-serve"]?.variantLabel == "MLX-Serve",
                  rows["mlx"]?.displayName == "Qwen3.8 Flash Next", rows["mlx"]?.variantLabel == "MLX",
                  rows["mixed"]?.displayName == "Qwen3.8 Flash Next (Mixed 4/8-bit)",
                  rows["small"]?.displayName == "Qwen3 4B", rows["custom"]?.displayName == "JevK5",
                  rows["dense"]?.displayName == "Qwen3 Dense",
                  ["mixed", "small", "custom", "dense"].allSatisfy({ rows[$0]?.variantLabel == nil }),
                  inventory.runningGroup?.models.first?.variantLabel == "Lily q4" else { return false }
            for query in [saved, "0f94842e", "Qwen3.8 Flash Next", "Lily q4"] {
                guard ModelsInventory(models: fixtures, loadedID: "", registry: nil, query: query).models.contains(where: { $0.id == "lily" }) else { return false }
            }
            let serveResults = ModelsInventory(models: fixtures, loadedID: "", registry: nil, query: "MLX-Serve")
            guard serveResults.models.map(\.id) == ["mlx-serve"] else { return false }
            return ModelsInventory.readableName("My preferred model", display: InstalledModelDisplay(lily)) == "My preferred model"
                && ModelsInventory.readableName("my-favorite-model", display: InstalledModelDisplay(lily)) == "My Favorite Model"
                && ModelsInventory.readableName("experiment-deadbeef", display: nil) == "Experiment Deadbeef"
                && ModelsInventory.readableName("---", display: nil) == "---"
        case "models-visible-row-name-sort":
            let aliases = [
                variant("zeta", name: "Zeta", display: modelDisplay("owner/first", name: "Alpha")),
                variant("alpha", name: "Alpha", display: modelDisplay("owner/second", name: "Zeta"))
            ]
            let ascending = ModelsInventory(models: aliases, loadedID: "", registry: nil)
            let descending = ModelsInventory(models: aliases, loadedID: "", registry: nil, sort: .nameDescending)
            let sizeTie = ModelsInventory(models: aliases, loadedID: "", registry: nil, sort: .sizeAscending)
            guard ascending.groups.flatMap({ $0.models.map(\.id) }) == ["alpha", "zeta"],
                  descending.groups.flatMap({ $0.models.map(\.id) }) == ["zeta", "alpha"],
                  sizeTie.groups.flatMap({ $0.models.map(\.id) }) == ["alpha", "zeta"] else { return false }
            var compact = modelDisplay("owner/group", name: "Alpha"); compact["quantization"] = "4-bit"
            var beta = modelDisplay("owner/other", name: "Zulu"); beta["quantization"] = "4-bit"
            let grouped = [
                variant("zeta", name: "Zeta", display: compact),
                variant("full", name: "Other", display: modelDisplay("owner/group", name: "Alpha")),
                variant("beta", name: "Beta", display: beta)
            ]
            let all = ModelsInventory(models: grouped, loadedID: "", registry: nil)
            let filtered = ModelsInventory(models: grouped, loadedID: "", registry: nil, query: "4-bit")
            return all.groups.map(\.name) == ["Alpha", "Zulu"] &&
                filtered.groups.flatMap({ $0.models.map(\.id) }) == ["beta", "zeta"]
        case "models-running-version-extracted":
            var compact = modelDisplay("repo:example/base"); compact["quantization"] = "4-bit"
            let saved = InstalledModel(id: "active", name: "Active saved name", path: "/private/active", apiID: "same-public-name",
                                       display: InstalledModelDisplay(modelDisplay("repo:example/base")),
                                       artifactMode: .visionLanguage, removal: InstalledModelRemoval(removal()))
            let models = [saved, variant("compact", name: "Compact", display: compact),
                          variant("other", name: "Other", display: modelDisplay("repo:example/other", name: "Other"))]
            let inventory = ModelsInventory(models: models, loadedID: saved.id, registry: nil)
            guard let running = inventory.runningGroup, running.id == "source:repo:example/base",
                  running.name == "Base", running.family == "Qwen3", running.models.count == 1,
                  let row = running.models.first, row.id == saved.id, row.isRunning,
                  row.name == saved.name, row.quantization == "BF16", row.weightBytes == Int64(1_190_000_000),
                  row.sourceRevision == String(repeating: "a", count: 40), row.artifactMode == .visionLanguage,
                  row.removal?.action == .trash, row.repositoryURL?.absoluteString == "https://huggingface.co/example/base"
            else { return false }
            let visible = running.models + inventory.otherGroups.flatMap(\.models)
            return inventory.groups.count == 2 && inventory.models.count == 3 && visible.count == 3 &&
                Set(visible.map(\.id)) == Set(models.map(\.id)) &&
                !inventory.otherGroups.flatMap(\.models).contains(where: { $0.id == saved.id }) &&
                inventory.otherGroups.first(where: { $0.id == running.id })?.models.map(\.id) == ["compact"]
        case "models-running-extraction-resorts-remaining":
            var huge = modelDisplay("repo:example/base", name: "Alpha"); huge["weight_bytes"] = Int64(1_000)
            var compact = huge; compact["weight_bytes"] = Int64(100); compact["quantization"] = "4-bit"
            var beta = modelDisplay("repo:example/other", name: "Beta"); beta["weight_bytes"] = Int64(500)
            let models = [variant("active", name: "Active", display: huge), variant("zulu", name: "Zulu", display: compact),
                          variant("beta", name: "Beta", display: beta)]
            let sizeAscending = ModelsInventory(models: models, loadedID: "active", registry: nil, sort: .sizeAscending)
            let sizeDescending = ModelsInventory(models: models, loadedID: "active", registry: nil, sort: .sizeDescending)
            let nameAscending = ModelsInventory(models: models, loadedID: "active", registry: nil, sort: .nameAscending)
            let nameDescending = ModelsInventory(models: models, loadedID: "active", registry: nil, sort: .nameDescending)
            return sizeAscending.otherGroups.flatMap({ $0.models.map(\.id) }) == ["zulu", "beta"] &&
                sizeDescending.otherGroups.flatMap({ $0.models.map(\.id) }) == ["beta", "zulu"] &&
                nameAscending.otherGroups.flatMap({ $0.models.map(\.id) }) == ["beta", "zulu"] &&
                nameDescending.otherGroups.flatMap({ $0.models.map(\.id) }) == ["zulu", "beta"] &&
                [sizeAscending, sizeDescending, nameAscending, nameDescending].allSatisfy {
                    $0.runningGroup?.models.map(\.id) == ["active"]
                }
        case "models-running-extraction-search-and-idle":
            var compact = modelDisplay(); compact["quantization"] = "4-bit"
            let models = [variant("active", name: "Active", display: modelDisplay()),
                          variant("compact", name: "Compact", display: compact)]
            let filtered = ModelsInventory(models: models, loadedID: "active", registry: nil, query: "4-bit")
            let activeOnly = ModelsInventory(models: models, loadedID: "active", registry: nil, query: "Running")
            let idle = ModelsInventory(models: models, loadedID: "", registry: nil)
            let missing = ModelsInventory(models: models, loadedID: "missing", registry: nil)
            return filtered.runningGroup == nil && filtered.otherGroups.flatMap({ $0.models.map(\.id) }) == ["compact"] &&
                activeOnly.runningGroup?.models.map(\.id) == ["active"] && activeOnly.otherGroups.isEmpty &&
                idle.runningGroup == nil && idle.otherGroups.map(\.id) == idle.groups.map(\.id) &&
                idle.otherGroups.flatMap({ $0.models.map(\.id) }) == idle.groups.flatMap({ $0.models.map(\.id) }) &&
                missing.runningGroup == nil && missing.otherGroups.flatMap({ $0.models.map(\.id) }) ==
                    idle.otherGroups.flatMap({ $0.models.map(\.id) })
        case "running-model-version-details-identity":
            let full = InstalledModel(id: "full", name: "Same name", path: "/private/full", apiID: "owner/Same",
                                      display: InstalledModelDisplay(modelDisplay()))
            var compactDisplay = modelDisplay(); compactDisplay["quantization"] = "4-bit"
            compactDisplay["weight_bytes"] = Int64(400_000_000)
            let compact = InstalledModel(id: "compact", name: "Same name", path: "/private/compact", apiID: "owner/Same",
                                         display: InstalledModelDisplay(compactDisplay))
            engine.models = [full, compact]
            engine.selectedID = compact.id; engine.loadedID = full.id
            guard engine.activeModel?.id == full.id, engine.activeName == full.name,
                  engine.activeVersionDetails == "BF16 · " + ByteCountFormatter.string(fromByteCount: 1_190_000_000, countStyle: .file)
            else { return false }
            engine.selectedID = full.id; engine.loadedID = compact.id
            guard engine.activeModel?.id == compact.id,
                  engine.activeVersionDetails == "4-bit · " + ByteCountFormatter.string(fromByteCount: 400_000_000, countStyle: .file)
            else { return false }
            engine.loadedID = "missing"
            guard engine.activeModel == nil, engine.activeName == "No model running", engine.activeVersionDetails.isEmpty
            else { return false }
            engine.loadedID = compact.id
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return engine.selectedID == full.id && engine.activeModel == nil &&
                engine.activeName == "No model running" && engine.activeVersionDetails.isEmpty
        case "image-catalog-stale-family":
            engine.requestModelCatalog(familyID: "qwen3")
            let first = command()
            engine.requestModelCatalog(familyID: "qwen3_5")
            let secondBatch = commands()
            guard secondBatch.map({ $0["command"] as! String }) == ["catalog_cancel", "catalog"],
                  let second = secondBatch.last, UUID(uuidString: second["request_id"] as? String ?? "") != nil else { return false }
            engine.apply(catalogReply(first))
            guard engine.catalog == nil, engine.catalogLoading, engine.catalogFamilyID == "qwen3_5" else { return false }
            engine.apply(catalogReply(second))
            guard engine.catalog?.familyID == "qwen3_5", engine.modelReady, engine.supportsImages,
                  engine.loadedID == "fixture-model", engine.selectedID == "fixture-model" else { return false }
            engine.cancelModelCatalog()
            guard command()["command"] as? String == "catalog_cancel" else { return false }
            engine.apply(catalogReply(second))
            return engine.catalog == nil && !engine.catalogLoading && engine.modelReady && engine.supportsImages
        case "catalog-offline-retry":
            engine.errorMessage = "Fixture unrelated runtime warning"
            engine.requestModelCatalog(familyID: "qwen3")
            let first = command()
            engine.apply(catalogReply(first, status: "offline"))
            guard engine.catalog?.status == .offline, engine.catalogError == "Fixture catalog message",
                  engine.errorMessage == "Fixture unrelated runtime warning", engine.modelReady else { return false }
            engine.requestModelCatalog(familyID: "qwen3")
            let retry = commands().last!
            engine.apply(catalogReply(first))
            guard engine.catalogLoading, engine.catalog == nil else { return false }
            engine.apply(catalogReply(retry))
            return engine.catalog?.status == .ready && engine.catalogError == nil && engine.loadedID == "fixture-model"
        case "catalog-pagination-merge-and-fence":
            engine.requestModelCatalog(familyID: "qwen3")
            let first = command()
            let firstItems = (0..<30).map { catalogItem("entry-\($0)") }
            engine.apply(catalogReply(first, items: firstItems, nextOffset: 30))
            engine.requestMoreModelCatalog(loadRemaining: false)
            let second = command()
            guard second["request_id"] as? String == first["request_id"] as? String,
                  (second["offset"] as? NSNumber)?.int64Value == 30, engine.catalogLoading,
                  engine.catalog?.items.count == 30 else { return false }
            engine.requestMoreModelCatalog(loadRemaining: false)
            guard noCommandSent() else { return false }
            engine.apply(catalogReply(first, items: [catalogItem("stale")], nextOffset: 30))
            guard engine.catalogLoading, engine.catalog?.items.count == 30 else { return false }
            let secondItems = (29..<59).map { catalogItem("entry-\($0)") }
            engine.apply(catalogReply(second, items: secondItems))
            guard !engine.catalogLoading, engine.catalog?.items.count == 59,
                  engine.catalog?.items.first?.id == "entry-0", engine.catalog?.items.last?.id == "entry-58",
                  engine.catalog?.nextOffset == nil, engine.catalogError == nil else { return false }
            engine.apply(catalogReply(second, items: [catalogItem("duplicate-response")]))
            return engine.catalog?.items.count == 59 && engine.modelReady && engine.loadedID == "fixture-model"
        case "catalog-pagination-failure-retry-preserves-rows":
            engine.requestModelCatalog(familyID: "qwen3")
            let first = command()
            engine.apply(catalogReply(first, items: [catalogItem("initial")], nextOffset: 30))
            guard engine.catalog?.items.map(\.id) == ["initial"], engine.catalog?.nextOffset == 30,
                  !engine.catalogLoading, noCommandSent() else { return false }
            engine.requestMoreModelCatalog()
            let second = command()
            engine.apply(catalogReply(second, status: "error"))
            guard engine.catalog?.items.map(\.id) == ["initial"], engine.catalog?.status == .ready,
                  engine.catalog?.nextOffset == 30,
                  engine.catalogError != nil, !engine.catalogLoading else { return false }
            engine.requestMoreModelCatalog()
            let retry = command()
            guard (retry["offset"] as? NSNumber)?.int64Value == 30,
                  retry["request_id"] as? String == second["request_id"] as? String else { return false }
            engine.apply(catalogReply(retry, items: [catalogItem("later")], nextOffset: 60))
            let third = command()
            guard (third["offset"] as? NSNumber)?.int64Value == 60, engine.catalogLoading else { return false }
            engine.apply(catalogReply(third, status: "offline"))
            guard engine.catalog?.items.map(\.id) == ["initial", "later"], engine.catalog?.nextOffset == 60,
                  engine.catalogError != nil else { return false }
            engine.requestModelCatalog(familyID: "qwen3_5")
            let newFamily = commands().last!
            engine.apply(catalogReply(third, items: [catalogItem("stale-family")]))
            guard engine.catalog == nil, engine.catalogLoading else { return false }
            engine.apply(catalogReply(newFamily))
            return engine.catalog?.familyID == "qwen3_5" && engine.catalog?.items.map(\.id) == ["entry"]
        case "catalog-pagination-auto-drains-positive-empty-and-duplicate-pages":
            engine.requestModelCatalog(familyID: "qwen3")
            let first = command()
            engine.apply(catalogReply(first, items: [catalogItem("known")], nextOffset: 30))
            guard !engine.catalogLoading, noCommandSent() else { return false }
            engine.requestMoreModelCatalog()
            let second = command()
            guard (second["offset"] as? NSNumber)?.int64Value == 30, engine.catalogLoading else { return false }
            engine.requestMoreModelCatalog()
            guard noCommandSent() else { return false }
            engine.apply(catalogReply(second, items: [], nextOffset: 60))
            let third = command()
            guard engine.catalogLoading, engine.catalog?.items.map(\.id) == ["known"],
                  (third["offset"] as? NSNumber)?.int64Value == 60 else { return false }
            engine.apply(catalogReply(third, items: [catalogItem("known")], nextOffset: 90))
            let fourth = command()
            guard engine.catalogLoading, engine.catalog?.items.map(\.id) == ["known"],
                  (fourth["offset"] as? NSNumber)?.int64Value == 90 else { return false }
            engine.apply(catalogReply(fourth, items: [catalogItem("found")], nextOffset: 120))
            let fifth = command()
            guard engine.catalogLoading, engine.catalog?.items.map(\.id) == ["known", "found"],
                  (fifth["offset"] as? NSNumber)?.int64Value == 120 else { return false }
            engine.apply(catalogReply(fifth, items: [catalogItem("last")]))
            return !engine.catalogLoading && noCommandSent() && engine.catalog?.nextOffset == nil &&
                engine.catalog?.items.map(\.id) == ["known", "found", "last"]
        case "catalog-schema-bounds":
            var report = catalogReport("qwen3")
            report["schema_version"] = 2
            guard FamilyModelCatalog(report) == nil else { return false }
            report = catalogReport("qwen3"); report["items"] = (0..<31).map { catalogItem("entry-\($0)") }
            guard FamilyModelCatalog(report) == nil else { return false }
            report = catalogReport("qwen3", items: [catalogItem(), catalogItem()])
            guard FamilyModelCatalog(report) == nil else { return false }
            for value in [true as Any, -1 as Any, 0.5 as Any] {
                report = catalogReport("qwen3"); report["offset"] = value
                guard FamilyModelCatalog(report) == nil else { return false }
            }
            report = catalogReport("qwen3", offset: 30, nextOffset: 30)
            guard FamilyModelCatalog(report) == nil else { return false }
            report = catalogReport("qwen3", nextOffset: 30); report["has_more"] = false
            guard FamilyModelCatalog(report) == nil else { return false }
            report = catalogReport("qwen3"); report["has_more"] = true
            guard FamilyModelCatalog(report) == nil else { return false }
            for (key, value) in [("revision", "bad" as Any), ("weight_bytes", true as Any),
                                 ("parameter_count", -1 as Any), ("default_artifact_mode", "unsupported" as Any),
                                 ("downloadable", 1 as Any), ("artifact_modes", ["text", "unknown"] as Any)] {
                var item = catalogItem(); item[key] = value
                guard FamilyModelCatalog(catalogReport("qwen3", items: [item])) == nil else { return false }
            }
            engine.requestModelCatalog(familyID: "qwen3")
            let request = command()
            var future = catalogReply(request)
            var raw = future["report"] as! [String: Any]; raw["schema_version"] = 2; future["report"] = raw
            engine.apply(future)
            return engine.catalog == nil && engine.catalogLoading && !engine.supportsImages && engine.modelReady
        case "catalog-unavailable-row":
            var item = catalogItem("unsupported")
            item["downloadable"] = false
            guard FamilyModelCatalog(catalogReport("qwen3", items: [item])) == nil else { return false }
            let reason = "Expert weights use 3-bit packing with groups of 128."
            item["unavailable_reason"] = reason
            engine.requestModelCatalog(familyID: "qwen3")
            let request = command()
            engine.apply(catalogReply(request, items: [item]))
            guard let row = engine.catalog?.items.first, row.unavailableReason == reason,
                  !row.downloadable else { return false }
            engine.installCatalogItem(id: row.id)
            return noCommandSent()
        case "catalog-provider-order":
            var small4 = catalogItem("small4"); small4["parameter_count"] = NSNull(); small4["quantization"] = "4-bit"
            var small8 = catalogItem("small8"); small8["parameter_count"] = NSNull(); small8["quantization"] = "8-bit"
            var large = catalogItem("large"); large["parameter_count"] = Int64(1_700_000_000); large["size_label"] = "1.7B"
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: [catalogItem("small"), small4, small8, large])) else { return false }
            return catalog.items.map(\.id) == ["small", "small4", "small8", "large"]
        case "catalog-quantization-tabs":
            var four = catalogItem("four"); four["quantization"] = "4-bit"
            var duplicate = catalogItem("duplicate"); duplicate["quantization"] = "4-bit"
            var eight = catalogItem("eight"); eight["quantization"] = "8-bit"
            var sixteen = catalogItem("sixteen"); sixteen["quantization"] = "16-bit"
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: [sixteen, eight, catalogItem("full"), four, duplicate])) else { return false }
            let tabs = CatalogQuantizationFilter.tabs(for: catalog.items)
            guard tabs.map(\.title) == ["All models", "4-bit", "8-bit", "16-bit", "BF16"],
                  Set(tabs).count == tabs.count,
                  CatalogQuantizationFilter.tabs(for: []) == [.all],
                  CatalogQuantizationFilter.tabs(for: [catalog.items[0]]) == [.all, .quantization("16-bit")] else { return false }
            return CatalogQuantizationFilter.all.matches(catalog.items[0]) &&
                CatalogQuantizationFilter.quantization("4-bit").matches(catalog.items[3]) &&
                !CatalogQuantizationFilter.quantization("4-bit").matches(catalog.items[2])
        case "catalog-quantization-refresh-fallback":
            var four = catalogItem("four"); four["quantization"] = "4-bit"
            var eight = catalogItem("eight"); eight["quantization"] = "8-bit"
            guard let previous = FamilyModelCatalog(catalogReport("qwen3", items: [four, eight])),
                  let refreshed = FamilyModelCatalog(catalogReport("qwen3_5", items: [eight, catalogItem("full")])) else { return false }
            let selected = CatalogQuantizationFilter.quantization("4-bit")
            guard selected.resolved(for: previous.items) == selected,
                  selected.resolved(for: refreshed.items) == .all,
                  selected.resolved(for: []) == .all,
                  CatalogQuantizationFilter.all.resolved(for: refreshed.items) == .all else { return false }
            let fallback = CatalogBrowserInventory(items: refreshed.items, quantization: selected.resolved(for: refreshed.items))
            return Set(fallback.groups.flatMap { $0.items.map(\.id) }) == Set(["eight", "full"])
        case "catalog-quantization-search-sort-target":
            var downloaded = groupedCatalogItem("downloaded-four", repo: "owner/Model-0.6B-4bit", name: "Model-0.6B-4bit",
                                                group: "1", groupName: "Model", publisher: "owner", quantization: "4-bit")
            downloaded["installed_ids"] = ["fixture-model"]
            let raw = [
                groupedCatalogItem("full", repo: "owner/Model-2B", name: "Model-2B", group: "1", groupName: "Model", publisher: "owner", size: "2B"),
                groupedCatalogItem("eight", repo: "owner/Model-0.6B-8bit", name: "Model-0.6B-8bit", group: "1", groupName: "Model", publisher: "owner", quantization: "8-bit"),
                downloaded,
                groupedCatalogItem("available-four", repo: "owner/Model-1.7B-4bit", name: "Model-1.7B-4bit", group: "1", groupName: "Model", publisher: "owner", size: "1.7B", quantization: "4-bit")
            ]
            engine.models = [InstalledModel(id: "fixture-model", name: "Downloaded", path: "/private/fixture", apiID: "saved", artifactMode: .text)]
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse, items: raw))
            guard let catalog = engine.catalog else { return false }
            let filter = CatalogQuantizationFilter.quantization("4-bit")
            let visible = CatalogBrowserInventory(items: catalog.items, query: "MODEL", sort: .sizeDescending, quantization: filter)
            guard visible.groups.count == 1, visible.groups[0].items.map(\.id) == ["available-four", "downloaded-four"],
                  engine.catalogItemIsDownloaded(visible.groups[0].items[1], artifactMode: .text),
                  !engine.catalogItemIsDownloaded(visible.groups[0].items[0], artifactMode: .text),
                  CatalogBrowserInventory(items: catalog.items, query: "BF16", quantization: filter).groups.isEmpty,
                  CatalogBrowserInventory(items: catalog.items, quantization: .all).groups[0].items.count == 4 else { return false }
            engine.installCatalogItem(id: visible.groups[0].items[0].id)
            let request = command()
            return request["command"] as? String == "install_catalog" && request["entry_id"] as? String == "available-four" &&
                request["catalog_request_id"] as? String == browse["request_id"] as? String &&
                request["artifact_mode"] as? String == "text" && request["repo"] == nil && request["revision"] == nil &&
                engine.selectedID == "fixture-model" && engine.loadedID == "fixture-model"
        case "catalog-grouped-size-quantization-rows":
            let raw = [
                groupedCatalogItem("large", repo: "Qwen/Qwen3-1.7B", name: "Qwen3-1.7B", group: "1", groupName: "Qwen3", publisher: "Qwen", size: "1.7B"),
                groupedCatalogItem("compact", repo: "Qwen/Qwen3-0.6B-MLX-4bit", name: "Qwen3-0.6B-MLX-4bit", group: "1", groupName: "Qwen3", publisher: "Qwen", quantization: "4-bit"),
                groupedCatalogItem("small", repo: "Qwen/Qwen3-0.6B", name: "Qwen3-0.6B", group: "1", groupName: "Qwen3", publisher: "Qwen")
            ]
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: raw)) else { return false }
            let inventory = CatalogBrowserInventory(items: catalog.items)
            guard inventory.groups.count == 1, let group = inventory.groups.first,
                  group.name == "Qwen3", group.publisher == "Qwen", group.items.count == 3,
                  Set(group.items.map(\.id)) == Set(["large", "compact", "small"]),
                  group.items.last?.id == "large" else { return false }
            return group.items.first(where: { $0.id == "compact" })?.repo == "Qwen/Qwen3-0.6B-MLX-4bit" &&
                group.items.first(where: { $0.id == "small" })?.revision == String(repeating: "a", count: 40)
        case "catalog-group-publisher-instruction-distinction":
            let raw = [
                groupedCatalogItem("official", repo: "Qwen/Qwen3-0.6B", name: "Qwen3-0.6B", group: "1", groupName: "Qwen3", publisher: "Qwen"),
                groupedCatalogItem("other", repo: "Other/Qwen3-0.6B", name: "Qwen3-0.6B", group: "2", groupName: "Qwen3", publisher: "Other"),
                groupedCatalogItem("base", repo: "Qwen/Qwen3-0.6B-Base", name: "Qwen3-0.6B-Base", group: "3", groupName: "Qwen3 Base", publisher: "Qwen"),
                groupedCatalogItem("instruct", repo: "Qwen/Qwen3-4B-Instruct-2507", name: "Qwen3-4B-Instruct-2507", group: "4", groupName: "Qwen3 Instruct-2507", publisher: "Qwen", size: "4B")
            ]
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: raw)) else { return false }
            let groups = CatalogBrowserInventory(items: catalog.items).groups
            return groups.count == 4 && groups.allSatisfy({ $0.items.count == 1 }) &&
                Set(groups.map(\.publisher)) == Set(["Qwen", "Other"]) &&
                Set(groups.map(\.name)) == Set(["Qwen3", "Qwen3 Base", "Qwen3 Instruct-2507"])
        case "catalog-group-malformed-conflicting-fallback":
            var malformed = groupedCatalogItem("malformed", repo: "owner/Actual", name: "Actual", group: "1", groupName: "Untrusted", publisher: "owner")
            malformed["model_group"] = ["id": "not-hex", "name": "Untrusted", "publisher": "owner"]
            guard let parsed = FamilyModelCatalog.Item(malformed), parsed.modelGroup == nil else { return false }
            let fallback = CatalogBrowserInventory(items: [parsed])
            guard fallback.groups.first?.name == "Actual", fallback.groups.first?.publisher == "owner" else { return false }
            var first = groupedCatalogItem("one", repo: "one/Actual1", name: "Actual1", group: "a", groupName: "Spoofed", publisher: "one")
            let second = groupedCatalogItem("two", repo: "two/Actual2", name: "Actual2", group: "a", groupName: "Conflicting", publisher: "two")
            first["size_label"] = "1B"
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: [first, second])) else { return false }
            let conflicted = CatalogBrowserInventory(items: catalog.items)
            return Set(conflicted.groups.map(\.name)) == Set(["Actual1", "Actual2"]) &&
                conflicted.groups.allSatisfy({ $0.items.count == 1 }) &&
                CatalogBrowserInventory(items: catalog.items, query: "Spoofed").groups.isEmpty
        case "catalog-browser-live-search":
            var raw = [
                groupedCatalogItem("compact", repo: "owner/Cafe-0.6B-4bit", name: "Café-0.6B-4bit", group: "1", groupName: "Café Model", publisher: "owner", quantization: "4-bit"),
                groupedCatalogItem("full", repo: "owner/Cafe-0.6B", name: "Café-0.6B", group: "1", groupName: "Café Model", publisher: "owner")
            ]
            raw[1]["artifact_modes"] = ["text", "vision-language"]
            raw[1]["default_artifact_mode"] = "vision-language"
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: raw)) else { return false }
            let visible = CatalogBrowserInventory(items: catalog.items, query: " CAFE 4bit 0.6b ")
            guard visible.groups.count == 1, visible.groups[0].items.map(\.id) == ["compact"] else { return false }
            return CatalogBrowserInventory(items: catalog.items, query: "cafe 8-bit").groups.isEmpty &&
                CatalogBrowserInventory(items: catalog.items, query: "aAaAaAa").groups.first?.items.count == 2 &&
                CatalogBrowserInventory(items: catalog.items, query: "images").groups.first?.items.map(\.id) == ["full"] &&
                CatalogBrowserInventory(items: catalog.items, query: "").groups.first?.items.count == 2 &&
                ModelBrowserSearch.matches("  - .  ", fields: []) &&
                ModelBrowserSearch.matches("4 bit", fields: ["4-bit"]) &&
                !ModelBrowserSearch.matches("4 bit", fields: ["14B", "8-bit"]) &&
                !ModelBrowserSearch.matches("AB", fields: ["A", "B"])
        case "catalog-browser-name-and-size-sort":
            let raw = [
                groupedCatalogItem("unknown", repo: "owner/Unknown", name: "Unknown", group: "1", groupName: "Unknown", publisher: "owner", size: nil),
                groupedCatalogItem("large", repo: "owner/Ten", name: "Ten", group: "2", groupName: "Model 10", publisher: "owner", size: "10B"),
                groupedCatalogItem("small", repo: "owner/Two", name: "Two", group: "3", groupName: "Model 2", publisher: "owner", size: "2B")
            ]
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: raw)) else { return false }
            let asc = CatalogBrowserInventory(items: catalog.items, sort: .nameAscending).groups
            let desc = CatalogBrowserInventory(items: catalog.items, sort: .nameDescending).groups
            let small = CatalogBrowserInventory(items: catalog.items, sort: .sizeAscending).groups
            let large = CatalogBrowserInventory(items: catalog.items, sort: .sizeDescending).groups
            guard asc.map(\.name) == ["Model 2", "Model 10", "Unknown"],
                  desc.map(\.name) == ["Unknown", "Model 10", "Model 2"],
                  small.map({ $0.items[0].id }) == ["small", "large", "unknown"],
                  large.map({ $0.items[0].id }) == ["large", "small", "unknown"] else { return false }
            return ModelBrowserSort.nameDescending.precedes("Same", id: "a", size: nil, "Same", id: "b", size: nil)
        case "catalog-browser-row-size-and-numeric-fallback":
            let raw = [
                groupedCatalogItem("unknown", repo: "owner/Unknown", name: "Unknown", group: "1", groupName: "Model", publisher: "owner", size: nil),
                groupedCatalogItem("small", repo: "owner/Small", name: "Small", group: "1", groupName: "Model", publisher: "owner", size: "600M"),
                groupedCatalogItem("large", repo: "owner/Large", name: "Large", group: "1", groupName: "Model", publisher: "owner", size: "1.7B"),
                groupedCatalogItem("count", repo: "owner/Count", name: "Count", group: "1", groupName: "Model", publisher: "owner", size: nil, parameters: 800_000_000)
            ]
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: raw)) else { return false }
            let ascending = CatalogBrowserInventory(items: catalog.items, sort: .sizeAscending).groups[0]
            let descending = CatalogBrowserInventory(items: catalog.items, sort: .sizeDescending).groups[0]
            var overflow = groupedCatalogItem("overflow", repo: "owner/Overflow", name: "Overflow", group: "2", groupName: "Overflow", publisher: "owner", size: "1e308T")
            overflow["parameter_count"] = NSNull()
            return ascending.items.map(\.id) == ["small", "count", "large", "unknown"] &&
                descending.items.map(\.id) == ["large", "count", "small", "unknown"] &&
                ascending.size(for: .sizeAscending) == 600_000_000 && descending.size(for: .sizeDescending) == 1_700_000_000 &&
                FamilyModelCatalog.Item(overflow)?.numericSize == nil
        case "catalog-filtered-download-retains-entry-id":
            let raw = [
                groupedCatalogItem("full", repo: "owner/Model-BF16", name: "Model", group: "1", groupName: "Model", publisher: "owner"),
                groupedCatalogItem("compact", repo: "owner/Model-4bit", name: "Model-4bit", group: "1", groupName: "Model", publisher: "owner", quantization: "4-bit")
            ]
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse, items: raw))
            guard let catalog = engine.catalog,
                  let selected = CatalogBrowserInventory(items: catalog.items, query: "4bit", sort: .nameDescending).groups.first?.items.first,
                  selected.id == "compact", selected.repo == "owner/Model-4bit" else { return false }
            engine.installCatalogItem(id: selected.id)
            let request = command()
            return request["command"] as? String == "install_catalog" && request["entry_id"] as? String == "compact" &&
                request["catalog_request_id"] as? String == browse["request_id"] as? String &&
                request["artifact_mode"] as? String == "text" && request["repo"] == nil && request["revision"] == nil &&
                engine.selectedID == "fixture-model" && engine.loadedID == "fixture-model"
        case "models-browser-search-and-sum-sort":
            var small = modelDisplay(); small["quantization"] = "4-bit"; small["weight_bytes"] = Int64(100)
            var full = modelDisplay(); full["weight_bytes"] = Int64(400)
            var other = modelDisplay("owner/other", name: "Other"); other["weight_bytes"] = Int64(300)
            var large = modelDisplay("owner/large", name: "Large"); large["weight_bytes"] = Int64(900)
            var unknown = modelDisplay("owner/unknown", name: "Unknown"); unknown["weight_bytes"] = NSNull()
            let models = [variant("small", name: "Compact saved version", display: small), variant("full", name: "Full saved version", display: full),
                          variant("other", name: "Other", display: other), variant("large", name: "Large", display: large), variant("unknown", name: "Unknown", display: unknown)]
            let ascending = ModelsInventory(models: models, loadedID: "full", registry: nil, sort: .sizeAscending)
            let descending = ModelsInventory(models: models, loadedID: "full", registry: nil, sort: .sizeDescending)
            guard ascending.groups.map(\.name) == ["Other", "Base", "Large", "Unknown"],
                  descending.groups.map(\.name) == ["Large", "Base", "Other", "Unknown"] else { return false }
            let filtered = ModelsInventory(models: models, loadedID: "full", registry: nil, query: "qwen 4bit", sort: .nameDescending)
            return filtered.groups.count == 1 && filtered.groups[0].models.map(\.id) == ["small"] &&
                filtered.models.map(\.id) == ["small"] && models.count == 5 &&
                ModelsInventory(models: models, loadedID: "full", registry: nil, query: "compact").models.map(\.id) == ["small"] &&
                ModelsInventory(models: models, loadedID: "full", registry: nil, query: "missing").groups.isEmpty
        case "models-browser-ambiguous-filter-and-remove":
            let first = variant("one", name: "First exact", display: modelDisplay())
            let conflict = variant("two", name: "Second exact", display: modelDisplay("example/base", name: "Conflicting base"))
            let visible = ModelsInventory(models: [first, conflict], loadedID: "", registry: nil, query: "Second")
            guard visible.groups.count == 1, visible.groups[0].name == "Second exact", visible.groups[0].family == nil else { return false }
            let saved = InstalledModel(id: "private-target", name: "Compact", path: "/private/target", apiID: "public-alias",
                                       display: InstalledModelDisplay(modelDisplay()), removal: InstalledModelRemoval(removal("unregister")))
            let other = InstalledModel(id: "private-other", name: "Other", path: "/private/other", apiID: "public-alias")
            engine.models = [saved, other]
            let rows = ModelsInventory(models: engine.models, loadedID: engine.loadedID, registry: nil, query: "compact", sort: .nameDescending).models
            guard rows.map(\.id) == ["private-target"] else { return false }
            engine.removeModel(id: rows[0].id, expectedAction: .unregister)
            let request = command()
            return request["command"] as? String == "remove_model" && request["model_id"] as? String == "private-target" &&
                request["action"] as? String == "unregister" && engine.models.count == 2 && engine.loadedID == "fixture-model"
        case "model-browser-sort-preferences":
            let suite = "Stark.ModelBrowser.Tests." + UUID().uuidString
            guard let preferences = UserDefaults(suiteName: suite) else { return false }
            defer { preferences.removePersistentDomain(forName: suite) }
            guard ModelBrowserPreferences.load(.saved, defaults: preferences) == .nameAscending,
                  ModelBrowserPreferences.load(.families, defaults: preferences) == .nameAscending,
                  ModelBrowserPreferences.load(.catalog, defaults: preferences) == .sizeAscending else { return false }
            ModelBrowserPreferences.save(.sizeDescending, for: .saved, defaults: preferences)
            ModelBrowserPreferences.save(.nameDescending, for: .families, defaults: preferences)
            ModelBrowserPreferences.save(.nameAscending, for: .catalog, defaults: preferences)
            guard let reopened = UserDefaults(suiteName: suite),
                  ModelBrowserPreferences.load(.saved, defaults: reopened) == .sizeDescending,
                  ModelBrowserPreferences.load(.families, defaults: reopened) == .nameDescending,
                  ModelBrowserPreferences.load(.catalog, defaults: reopened) == .nameAscending else { return false }
            ModelBrowserPreferences.save(.sizeAscending, for: .families, defaults: reopened)
            guard ModelBrowserPreferences.load(.families, defaults: reopened) == .nameDescending else { return false }
            reopened.set("future-sort", forKey: ModelBrowserScope.saved.preferenceKey)
            reopened.set(ModelBrowserSort.sizeDescending.rawValue, forKey: ModelBrowserScope.families.preferenceKey)
            reopened.set(42, forKey: ModelBrowserScope.catalog.preferenceKey)
            return ModelBrowserPreferences.load(.saved, defaults: reopened) == .nameAscending &&
                ModelBrowserPreferences.load(.families, defaults: reopened) == .nameAscending &&
                ModelBrowserPreferences.load(.catalog, defaults: reopened) == .sizeAscending
        case "catalog-installed-mode":
            var item = catalogItem(); item["installed_ids"] = ["fixture-model", "not-in-library"]
            guard let parsed = FamilyModelCatalog.Item(item) else { return false }
            engine.apply(["event": "models", "models": [["id": "fixture-model", "name": "Exact saved name", "path": "/private/model"]]])
            guard engine.catalogItemIsDownloaded(parsed, artifactMode: .text),
                  !engine.catalogItemIsDownloaded(parsed, artifactMode: .visionLanguage) else { return false }
            engine.apply(["event": "models", "models": [["id": "fixture-model", "name": "Exact saved name", "path": "/private/model", "artifact_mode": "unknown"]]])
            return !engine.catalogItemIsDownloaded(parsed, artifactMode: .text) && engine.loadedID == "fixture-model"
        case "catalog-install-selection":
            var callbacks = 0; engine.onInstalled = { callbacks += 1 }
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            let install = command()
            guard install["command"] as? String == "install_catalog", install["catalog_request_id"] as? String == browse["request_id"] as? String,
                  install["artifact_mode"] as? String == "text", engine.busy, engine.downloadProgress.state == "waiting" else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.busy, engine.catalogInstallation != nil else { return false }
            engine.apply(["event": "models", "models": [
                ["id": "fixture-model", "name": "Existing", "path": "/private/existing"],
                ["id": "downloaded", "name": "New", "path": "/private/new", "artifact_mode": "text"]
            ]])
            engine.apply(["event": "installed", "support_request_id": install["support_request_id"]!, "model": ["id": "downloaded"]])
            let refresh = commands().last!
            guard refresh["command"] as? String == "catalog", engine.catalogInstallation == nil, callbacks == 0,
                  engine.selectedID == "fixture-model", engine.loadedID == "fixture-model", engine.modelReady else { return false }
            var item = catalogItem(); item["installed_ids"] = ["downloaded"]
            engine.apply(catalogReply(refresh, items: [item]))
            return engine.catalogItemIsDownloaded(engine.catalog!.items[0], artifactMode: .text)
        case "catalog-install-includes-images":
            var item = catalogItem()
            item["artifact_modes"] = ["text", "vision-language"]
            // Image capability wins even if an older catalog defaults to text.
            item["default_artifact_mode"] = "text"
            item["installed_ids"] = ["fixture-model"]
            engine.requestModelCatalog(familyID: "qwen3_5")
            let browse = command(); engine.apply(catalogReply(browse, items: [item]))
            guard let parsed = engine.catalog?.items.first,
                  parsed.installationMode == .visionLanguage,
                  !engine.catalogItemIsDownloaded(parsed, artifactMode: parsed.installationMode) else { return false }
            engine.installCatalogItem(id: "entry")
            let install = command()
            guard install["command"] as? String == "install_catalog",
                  install["artifact_mode"] as? String == "vision-language",
                  engine.catalogInstallation?.artifactMode == .visionLanguage else { return false }
            engine.apply(["event": "models", "models": [["id": "fixture-model", "name": "Model", "path": "/private/model",
                                                          "artifact_mode": "vision-language"]]])
            return engine.catalogItemIsDownloaded(parsed, artifactMode: parsed.installationMode)
        case "catalog-install-survives-browser-cancel":
            var callbacks = 0; engine.onInstalled = { callbacks += 1 }
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            let install = command()
            engine.apply(["event": "progress", "phase": "download", "state": "running",
                          "completed_bytes": Int64(10), "total_bytes": Int64(20)])
            engine.cancelModelCatalog()
            guard engine.catalog == nil, engine.catalogFamilyID == nil, !engine.catalogLoading,
                  engine.catalogInstallation != nil, engine.busy, engine.state == "installing",
                  engine.downloadProgress.state == "running", engine.downloadProgress.completedBytes == 10,
                  noCommandSent() else { return false }
            engine.apply(catalogReply(browse))
            guard engine.catalog == nil, engine.catalogInstallation != nil else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.busy else { return false }
            engine.apply(["event": "installed", "support_request_id": install["support_request_id"]!, "model": ["id": "downloaded"]])
            let cleanup = command()
            return cleanup["command"] as? String == "catalog_cancel" &&
                cleanup["request_id"] as? String == browse["request_id"] as? String &&
                engine.catalogInstallation == nil && engine.catalog == nil && callbacks == 0 &&
                engine.selectedID == "fixture-model" && engine.loadedID == "fixture-model" && engine.modelReady
        case "catalog-install-stale-error":
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            let install = command()
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            engine.apply(["event": "error", "request_id": "stale-request", "message": "Stale catalog error"])
            guard engine.catalogInstallation != nil, engine.errorMessage == nil, engine.busy else { return false }
            engine.apply(["event": "error", "request_id": install["support_request_id"]!, "message": "Fixture download failed"])
            guard engine.catalogInstallation == nil, engine.catalogInstallFailed,
                  engine.catalogInstallMessage == "Fixture download failed", engine.modelReady else { return false }
            engine.resetInstallProgress(kind: "huggingface")
            return engine.errorMessage == nil && engine.loadedID == "fixture-model"
        case "image-catalog-install-admission-error":
            engine.imageVerification = "locally_tested"
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            let install = command()
            engine.apply(["event": "error", "request_id": "unrelated", "message": "Unrelated rejection"])
            guard engine.state == "installing", engine.busy, engine.catalogInstallation != nil else { return false }
            engine.apply(["event": "error", "request_id": install["support_request_id"]!, "message": "Wait for the current operation."])
            return engine.state == "ready" && !engine.busy && engine.modelReady && engine.supportsImages &&
                engine.imageVerification == "locally_tested" &&
                engine.loadedID == "fixture-model" && engine.selectedID == "fixture-model" &&
                engine.catalogInstallation == nil && engine.catalogInstallFailed
        case "catalog-first-download-selection":
            engine.models = []; engine.selectedID = ""; engine.loadedID = ""; engine.loadedMode = nil; engine.state = "idle"
            var callbacks = 0; engine.onInstalled = { callbacks += 1 }
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            let install = command()
            engine.apply(["event": "status", "state": "idle", "message": "Model installed"])
            engine.apply(["event": "models", "models": [["id": "downloaded", "name": "First model", "path": "/private/first", "artifact_mode": "text"]]])
            guard engine.selectedID.isEmpty, engine.loadedID.isEmpty, engine.busy else { return false }
            engine.apply(["event": "installed", "support_request_id": install["support_request_id"]!, "model": ["id": "downloaded"]])
            _ = commands()
            return engine.models.count == 1 && engine.selectedID.isEmpty && engine.loadedID.isEmpty && !engine.busy && callbacks == 0
        case "catalog-stop-fence":
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry")
            let install = command()
            engine.stop(); _ = command()
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            engine.selectedID = "replacement"
            engine.apply(["event": "ready", "mode": "decision", "model_id": "replacement"])
            engine.apply(["event": "installed", "support_request_id": install["support_request_id"]!, "model": ["id": "stale-model"]])
            engine.apply(["event": "error", "request_id": install["support_request_id"]!, "message": "Stale failure"])
            engine.apply(catalogReply(browse))
            return engine.selectedID == "replacement" && engine.loadedID == "replacement" && engine.modelReady &&
                engine.catalog == nil && engine.errorMessage == nil
        case "model-removal-confirmation":
            let imported = InstalledModel(id: "imported", name: "Exact imported name", path: "/private/imported", apiID: "id",
                                          removal: InstalledModelRemoval(removal("unregister")))
            let owned = InstalledModel(id: "owned", name: "Exact owned name", path: "/private/owned", apiID: "id",
                                       removal: InstalledModelRemoval(removal()))
            guard let importedPrompt = ModelRemovalConfirmation(model: imported, details: ["BF16"], isRunning: false),
                  let ownedPrompt = ModelRemovalConfirmation(model: owned, details: ["4-bit"], isRunning: true) else { return false }
            guard importedPrompt.action.title == "Remove from Stark", importedPrompt.message.contains("folder will remain"),
                  ownedPrompt.action.title == "Move to Trash", ownedPrompt.message.contains("will be stopped"),
                  ownedPrompt.message.contains("4-bit") else { return false }
            engine.models = [imported]; engine.removeModel(id: imported.id, expectedAction: .unregister)
            let request = command()
            return request["action"] as? String == "unregister" && request["model_id"] as? String == imported.id
        case "image-model-removal-preserves-other":
            engine.models = [InstalledModel(id: "fixture-model", name: "Running", path: "/private/running", apiID: "id"),
                             InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "other",
                                            removal: InstalledModelRemoval(removal()))]
            engine.removeModel(id: "other", expectedAction: .trash)
            let request = command()
            guard request["action"] as? String == "trash", engine.loadedID == "fixture-model", engine.supportsImages,
                  engine.models.count == 2 else { return false }
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            guard engine.busy, !engine.modelReady else { return false }
            engine.apply(["event": "model_removed", "request_id": request["request_id"]!, "model_id": "wrong", "action": "trash"])
            guard engine.removingModelID == "other" else { return false }
            engine.apply(["event": "model_removed", "request_id": request["request_id"]!, "model_id": "other", "action": "trash", "stopped_model_id": NSNull()])
            return engine.modelReady && engine.loadedID == "fixture-model" && engine.supportsImages && engine.models.count == 2
        case "model-removal-action-fence":
            let saved = InstalledModel(id: "saved", name: "Saved", path: "/private/saved", apiID: "id", removal: InstalledModelRemoval(removal("unregister")))
            engine.models = [saved]
            engine.removeModel(id: "saved", expectedAction: .trash)
            guard engine.removingModelID == nil, engine.removalError != nil, engine.state == "ready" else { return false }
            var future = removal(); future["schema_version"] = 2
            guard InstalledModelRemoval(future) == nil else { return false }
            future = removal(); future["action"] = "delete"
            return InstalledModelRemoval(future) == nil && ModelRemovalConfirmation(model: InstalledModel(id: "missing", name: "Saved", path: "/private/missing", apiID: "id"), details: [], isRunning: false) == nil
        case "image-model-removal-error-fence":
            engine.models = [InstalledModel(id: "fixture-model", name: "Running", path: "/private/running", apiID: "id",
                                            removal: InstalledModelRemoval(removal("unregister")))]
            engine.removeModel(id: "fixture-model", expectedAction: .unregister)
            let request = command()
            engine.apply(["event": "status", "state": "ready", "message": "Ready"])
            engine.apply(["event": "model_remove_error", "request_id": "old", "model_id": "fixture-model", "message": "Stale error", "stopped_model_id": "fixture-model"])
            guard engine.busy, engine.loadedID == "fixture-model", engine.supportsImages else { return false }
            engine.apply(["event": "model_remove_error", "request_id": request["request_id"]!, "model_id": "fixture-model", "message": "Fixture rollback", "stopped_model_id": "fixture-model"])
            return !engine.busy && engine.loadedID.isEmpty && !engine.supportsImages && engine.models.count == 1 && engine.removalError == "Fixture rollback"
        case "image-model-removal-admission-error":
            engine.models = [InstalledModel(id: "other", name: "Other", path: "/private/other", apiID: "id",
                                            removal: InstalledModelRemoval(removal()))]
            engine.removeModel(id: "other", expectedAction: .trash)
            let request = command()
            engine.apply(["event": "error", "request_id": request["request_id"]!, "message": "Wait for the current operation."])
            return !engine.busy && engine.modelReady && engine.loadedID == "fixture-model" && engine.supportsImages &&
                engine.models.count == 1 && engine.removalError == "Wait for the current operation."
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
        case "image-limits-follow-model":
            var retained = imageReady
            retained["image_limits"] = ["max_source_pixels": 32_000_000,
                "max_processed_pixels": 16_777_216, "max_image_tokens": 16_384]
            engine.apply(retained)
            guard engine.imageLimits?.processedPixels == 16_777_216,
                  engine.imageLimits?.imageTokens == 16_384 else { return false }
            var gemma = imageReady
            gemma["image_limits"] = ["max_source_pixels": 32_000_000,
                "max_processed_pixels": 645_120, "max_image_tokens": 280]
            engine.apply(gemma)
            guard engine.imageLimits?.processedPixels == 645_120,
                  engine.imageLimits?.imageTokens == 280 else { return false }
            engine.apply(imageReady)
            guard engine.imageLimits == nil else { return false }
            engine.apply(retained)
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return !engine.supportsImages && engine.imageLimits == nil &&
                ModelImageLimits(["max_source_pixels": true,
                    "max_processed_pixels": 16_777_216, "max_image_tokens": 16_384]) == nil
        case "image-local-evidence":
            var verified = imageReady
            verified["image_verification"] = "locally_tested"
            engine.apply(verified)
            guard engine.supportsImages, engine.imageVerification == "locally_tested" else { return false }
            engine.pendingOptions = engine.options
            engine.pendingOptionsModelID = engine.loadedID
            engine.state = "configuring"
            verified["event"] = "configured"
            engine.apply(verified)
            return engine.modelReady && engine.supportsImages && engine.imageVerification == "locally_tested"
        case "image-reference-evidence":
            var verified = imageReady
            verified["image_verification"] = "reference_verified"
            engine.apply(verified)
            guard engine.supportsImages, engine.imageVerification == "reference_verified" else { return false }
            verified["images"] = false
            engine.apply(verified)
            return !engine.supportsImages && engine.imageVerification == "unverified"
        case "image-unknown-evidence":
            var unsupported = imageReady
            unsupported["image_verification"] = "self-declared-reference-pass"
            engine.apply(unsupported)
            guard engine.imageVerification == "unverified" else { return false }
            engine.apply(["event": "status", "state": "idle", "message": "Stopped"])
            return !engine.supportsImages && engine.imageVerification == "unverified"
        case "image-install-check-clears":
            try! engine.attachImage(bytes: imageBytes)
            engine.apply(["event": "gpu_memory", "model_id": "fixture-model", "mode": "chat", "gpu_memory_bytes": 4096])
            guard engine.gpuMemoryBytes == 4096 else { return false }
            let image = engine.draftImage!
            engine.messages = [ChatMessage(text: "synthetic image turn", isUser: true, image: image)]
            engine.conversationImage = image
            _ = try! engine.attachmentStaging!.stage(image)
            engine.imageVerification = "reference_verified"
            engine.state = "installing"
            engine.apply(["event": "image_check", "state": "checking"])
            return engine.state == "installing" && engine.busy && engine.loadedID.isEmpty && engine.loadedMode == nil &&
                engine.gpuMemoryBytes == nil && engine.gpuMemoryText == "GPU memory —" &&
                !engine.supportsImages && engine.imageVerification == "unverified" &&
                engine.messages.isEmpty && engine.draftImage == nil && engine.conversationImage == nil &&
                staged().isEmpty && engine.message == "Loading model…"
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
            guard initial["image"] is [String: Any], initial.keys.sorted() == ["command", "image", "request_id", "text"],
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
    def test_small_guided_reply_arrives_while_worker_pipe_stays_open(self):
        self.exercise("worker-pipe-small-guided-reply")

    def test_worker_pipe_delivers_fragmented_and_coalesced_events(self):
        self.exercise("worker-pipe-fragmented-and-coalesced-events")

    def test_worker_pipe_eof_removes_readability_handler(self):
        self.exercise("worker-pipe-eof-clears-handler")
    def test_small_backend_replies_do_not_wait_for_pipe_close(self):
        self.exercise("backend-short-replies")

    def test_connected_application_activity_does_not_show_playground_timing(self):
        self.exercise("generation-progress-external-activity")

    def test_api_progress_is_fenced_per_request_and_reports_prefix_reuse(self):
        self.exercise("generation-progress-api-requests")

    def test_playground_request_error_restores_draft_and_ignores_stale_ids(self):
        self.exercise("playground-request-error")

    def test_generation_progress_matches_active_request_and_retains_actual_first_token(
        self,
    ):
        self.exercise("generation-progress-request-and-lifecycle")

    def test_generation_progress_rejects_invalid_counts_times_and_partial_ranges(self):
        self.exercise("generation-progress-validation")

    def test_generation_progress_clears_on_failure_stop_and_model_replacement(self):
        self.exercise("generation-progress-clear-and-fence")

    def test_generation_timing_tracks_estimates_rates_and_overruns(self):
        self.exercise("generation-progress-formatting")

    def test_readout_keeps_fixed_rows_and_follows_the_running_request(self):
        self.exercise("generation-readout")

    def test_blocked_keychain_read_keeps_main_loop_usable_and_admission_bounded(self):
        self.exercise("api-blocked-read-keeps-main-loop-usable")

    def test_blocked_keychain_replacement_does_not_report_failure(self):
        self.exercise("api-blocked-replacement-remains-pending")

    def test_late_keychain_result_cannot_configure_replacement_backend(self):
        self.exercise("api-late-credential-result-is-fenced")

    def test_late_keychain_result_cannot_activate_after_shutdown(self):
        self.exercise("api-credential-result-after-shutdown-is-fenced")

    def test_api_credential_failure_keeps_model_usable(self):
        self.exercise("api-keychain-failure-isolated")

    def test_api_credential_reuse_and_failed_rotation(self):
        self.exercise("api-reuse-rotation-and-connection-events")

    def test_api_port_persistence_and_conflict_keeps_model_usable(self):
        self.exercise("api-port-settings-and-conflict")

    def test_gpu_memory_updates_during_requests_and_validates_measurements(self):
        self.exercise("gpu-memory-live-and-validation")

    def test_gpu_memory_clears_and_fences_stopped_or_replaced_workers(self):
        self.exercise("gpu-memory-stop-and-mode-fence")

    def test_gpu_memory_percentage_thresholds_and_unavailable_budget(self):
        self.exercise("gpu-memory-budget-indicator")

    def test_model_default_context_and_restore_defaults_use_published_maximum(self):
        self.exercise("context-model-defaults")

    def test_beginner_recommendations_reject_stale_or_invalid_reports(self):
        self.exercise("beginner-recommendation-fences")

    def test_beginner_zero_download_requires_an_installed_model(self):
        self.exercise("beginner-installed-only-report")

    def test_beginner_install_rejection_does_not_start_an_operation(self):
        self.exercise("beginner-rejected-install-does-not-start")
    def test_beginner_general_task_reaches_guided_model_backend(self):
        self.exercise("beginner-general-task-transport")

    def test_beginner_cancellation_preserves_the_current_worker_and_history(self):
        self.exercise("beginner-cancel-preserves-current")

    def test_beginner_switch_requires_an_exact_ready_acknowledgement(self):
        self.exercise("beginner-explicit-switch-fences")

    def test_beginner_rejected_start_preserves_the_current_chat(self):
        self.exercise("beginner-load-rejection-preserves-history")

    def test_beginner_invalid_ready_budget_finishes_pending_start(self):
        self.exercise("beginner-invalid-ready-settles-start")

    def test_beginner_api_budget_does_not_replace_native_model_capacity(self):
        self.exercise("beginner-api-budget-keeps-native-capacity")

    def test_beginner_local_start_rejection_preserves_the_current_operation(self):
        for scenario in [
            "beginner-start-busy-rejection",
            "beginner-start-missing-rejection",
        ]:
            with self.subTest(scenario=scenario):
                self.exercise(scenario)

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
                    str(ROOT / "apps/menu-bar/Sources/BeginnerModelReport.swift"),
                    str(ROOT / "apps/menu-bar/Sources/APICredentialStore.swift"),
                    str(ROOT / "apps/menu-bar/Sources/DesktopClientPeer.swift"),
                    str(ROOT / "apps/menu-bar/Sources/DesktopConnectionApproval.swift"),
                    str(ROOT / "apps/menu-bar/Sources/DesktopConnectionBroker.swift"),
                    str(ROOT / "apps/menu-bar/Sources/OpenWorkBeginnerHandoff.swift"),
                    str(
                        ROOT / "apps/menu-bar/Sources/OpenWorkConnectorInstaller.swift"
                    ),
                    str(
                        ROOT
                        / "apps/menu-bar/Sources/OpenWorkConnectionController.swift"
                    ),
                    str(
                        ROOT
                        / "apps/menu-bar/Sources/OpenCodeConnectionController.swift"
                    ),
                    str(ROOT / "apps/menu-bar/Sources/DesktopConnections.swift"),
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

    def test_image_limits_follow_loaded_model_and_clear_on_stop(self):
        self.exercise("image-limits-follow-model")

    def test_vision_install_is_explicit_opt_in(self):
        self.exercise("vision-install-opt-in")

    def test_existing_install_keeps_default_mode(self):
        self.exercise("text-install-default")

    def test_chat_install_forwards_text_purpose_before_download(self):
        self.exercise("text-install-purpose")

    def test_family_registry_parses_mode_statuses_and_ignores_future_schema(self):
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

    def test_models_inventory_sorts_naturally_and_preserves_exact_names(self):
        self.exercise("models-downloaded-sort")

    def test_models_running_badge_uses_only_loaded_identity(self):
        self.exercise("models-running-identity")

    def test_models_default_tab_and_empty_inventory_copy(self):
        self.exercise("models-empty-state")

    def test_available_families_are_friendly_sorted_and_exclude_conversion_only(self):
        from stark_convert.model_support import support_matrix

        self.exercise("models-family-inventory", json.dumps(support_matrix()))

    def test_model_display_metadata_cannot_change_runtime_capabilities(self):
        self.exercise("models-display-event")

    def test_installed_model_display_accepts_olmo3_and_gemma4_families(self):
        self.exercise("models-display-native-families")

    def test_tool_labels_distinguish_protocol_support_and_never_grant_live_capabilities(
        self,
    ):
        self.exercise("models-tool-support-labels")

    def test_malformed_display_keeps_exact_model_name_without_guessed_badges(self):
        self.exercise("models-display-malformed")

    def test_variants_group_by_explicit_source_identity_and_preserve_saved_names(self):
        self.exercise("models-grouped-variants")

    def test_ambiguous_or_missing_group_metadata_keeps_models_separate(self):
        self.exercise("models-grouping-ambiguity")

    def test_groups_and_variants_have_deterministic_natural_order(self):
        self.exercise("models-variant-order")

    def test_saved_name_sort_matches_visible_rows_after_group_filtering(self):
        self.exercise("models-visible-row-name-sort")

    def test_mtp_advanced_options_use_live_support_and_acknowledged_model_preference(
        self,
    ):
        self.exercise("mtp-advanced-options")

    def test_running_version_is_extracted_once_with_source_and_control_identity(self):
        self.exercise("models-running-version-extracted")

    def test_remaining_groups_resort_by_their_own_sizes_and_visible_names(self):
        self.exercise("models-running-extraction-resorts-remaining")

    def test_running_section_respects_search_and_is_absent_without_active_model(self):
        self.exercise("models-running-extraction-search-and-idle")

    def test_running_model_version_uses_loaded_private_identity_and_clears_on_stop(
        self,
    ):
        self.exercise("running-model-version-details-identity")

    def test_saved_row_run_uses_private_id_for_duplicate_names_and_loads_that_model(
        self,
    ):
        self.exercise("saved-row-run-private-id")

    def test_saved_row_run_rejects_busy_or_removed_rows_without_changing_selection(
        self,
    ):
        self.exercise("saved-row-run-guards")

    def test_running_the_loaded_row_preserves_its_conversation_and_image_capabilities(
        self,
    ):
        self.exercise("saved-row-run-already-loaded")

    def test_saved_row_run_reloads_when_the_loaded_mode_differs(self):
        self.exercise("saved-row-run-mode-mismatch")

    def test_saved_row_run_reports_send_failure_without_becoming_ready(self):
        self.exercise("saved-row-run-unavailable-runtime")

    def test_saved_row_controls_follow_switch_and_stop_lifecycle(self):
        self.exercise("saved-row-control-switch")

    def test_saved_row_stop_cancels_pending_load_and_rejects_late_ready(self):
        self.exercise("saved-row-control-cancel-load")

    def test_saved_row_stop_rejects_wrong_duplicate_name_or_removed_identity(self):
        self.exercise("saved-row-control-identity-guards")

    def test_saved_row_stop_can_cancel_active_generation(self):
        self.exercise("saved-row-control-stop-generation")

    def test_restart_controls_stop_exact_model_and_restart_in_stopped_mode(self):
        self.exercise("restart-controls-stop-and-start")

    def test_explicit_failed_start_retries_selected_model_without_clearing_on_busy_run(
        self,
    ):
        self.exercise("restart-controls-explicit-target-after-failed-start")

    def test_restart_controls_reject_running_busy_and_deleted_targets(self):
        self.exercise("restart-controls-running-busy-and-deleted-target")

    def test_saved_row_stop_cannot_cancel_unrelated_installation_or_removal(self):
        self.exercise("saved-row-control-unrelated-operation")

    def test_grouped_text_and_image_variants_have_distinct_stored_mode_label(self):
        self.exercise("models-artifact-mode-label")

    def test_readable_model_names_preserve_saved_identity_precision_and_search(self):
        self.exercise("models-readable-names")

    def test_catalog_family_changes_and_close_fence_late_results(self):
        self.exercise("image-catalog-stale-family")

    def test_catalog_offline_retry_preserves_runtime_state(self):
        self.exercise("catalog-offline-retry")

    def test_catalog_pagination_preserves_rows_merges_duplicates_and_fences_pages(self):
        self.exercise("catalog-pagination-merge-and-fence")

    def test_catalog_pagination_failure_retry_preserves_loaded_rows(self):
        self.exercise("catalog-pagination-failure-retry-preserves-rows")

    def test_catalog_pagination_single_click_drains_pages_and_ignores_duplicate_clicks(
        self,
    ):
        self.exercise(
            "catalog-pagination-auto-drains-positive-empty-and-duplicate-pages"
        )

    def test_catalog_schema_and_item_bounds_are_enforced(self):
        self.exercise("catalog-schema-bounds")

    def test_catalog_unavailable_row_retains_reason_and_cannot_install(self):
        self.exercise("catalog-unavailable-row")

    def test_catalog_preserves_provider_order_for_size_and_quantized_variants(self):
        self.exercise("catalog-provider-order")

    def test_catalog_quantization_tabs_are_dynamic_unique_and_naturally_ordered(self):
        self.exercise("catalog-quantization-tabs")

    def test_catalog_quantization_selection_falls_back_after_formats_change(self):
        self.exercise("catalog-quantization-refresh-fallback")

    def test_quantization_tabs_intersect_search_sort_and_keep_exact_download_targets(
        self,
    ):
        self.exercise("catalog-quantization-search-sort-target")

    def test_catalog_groups_sizes_and_quantizations_without_changing_entry_ids(self):
        self.exercise("catalog-grouped-size-quantization-rows")

    def test_catalog_groups_keep_publishers_base_and_instruct_models_distinct(self):
        self.exercise("catalog-group-publisher-instruction-distinction")

    def test_bad_or_conflicting_group_metadata_falls_back_to_original_model_names(self):
        self.exercise("catalog-group-malformed-conflicting-fallback")

    def test_catalog_search_matches_typed_terms_and_only_keeps_matching_versions(self):
        self.exercise("catalog-browser-live-search")

    def test_catalog_sort_uses_natural_names_and_keeps_unknown_sizes_last(self):
        self.exercise("catalog-browser-name-and-size-sort")

    def test_grouped_rows_sort_numeric_sizes_with_parameter_fallback_and_bounds(self):
        self.exercise("catalog-browser-row-size-and-numeric-fallback")

    def test_filtered_catalog_download_targets_the_original_cached_entry(self):
        self.exercise("catalog-filtered-download-retains-entry-id")

    def test_saved_search_filters_versions_and_size_sort_sums_visible_group_bytes(self):
        self.exercise("models-browser-search-and-sum-sort")

    def test_saved_filter_preserves_group_fallback_and_private_removal_target(self):
        self.exercise("models-browser-ambiguous-filter-and-remove")

    def test_sort_preferences_survive_reopening_and_are_independent_by_scope(self):
        self.exercise("model-browser-sort-preferences")

    def test_downloaded_catalog_status_matches_live_ids_and_artifact_mode(self):
        self.exercise("catalog-installed-mode")

    def test_catalog_install_retains_selection_and_browser_after_terminal_status(self):
        self.exercise("catalog-install-selection")

    def test_catalog_install_survives_leaving_browser_and_cleans_up_after_completion(
        self,
    ):
        self.exercise("catalog-install-survives-browser-cancel")

    def test_catalog_install_errors_require_current_request_and_clear_on_reset(self):
        self.exercise("catalog-install-stale-error")

    def test_catalog_busy_rejection_preserves_loaded_model_and_releases_pending(self):
        self.exercise("image-catalog-install-admission-error")

    def test_first_catalog_download_does_not_select_or_load_a_model(self):
        self.exercise("catalog-first-download-selection")

    def test_stopped_catalog_install_cannot_apply_stale_success_or_error(self):
        self.exercise("catalog-stop-fence")

    def test_removal_confirmation_distinguishes_imports_from_owned_files(self):
        self.exercise("model-removal-confirmation")

    def test_removal_preserves_other_running_model_and_waits_for_matching_result(self):
        self.exercise("image-model-removal-preserves-other")

    def test_changed_removal_action_and_unknown_schema_cannot_request_deletion(self):
        self.exercise("model-removal-action-fence")

    def test_removal_error_only_clears_explicitly_stopped_matching_worker(self):
        self.exercise("image-model-removal-error-fence")

    def test_busy_removal_rejection_releases_pending_without_touching_loaded_model(
        self,
    ):
        self.exercise("image-model-removal-admission-error")

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

    def test_resource_budget_follows_runtime_memory_and_rejects_overflow(self):
        self.exercise("resource-budget-memory-derived")

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

    def test_catalog_install_automatically_includes_available_images(self):
        self.exercise("catalog-install-includes-images")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControllerLifecycleTests)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(
        f"controller_checks: run={result.testsRun} failed={failed} skipped={len(result.skipped)}"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
