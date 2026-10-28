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
        let previousSelectedModel = UserDefaults.standard.object(forKey: "selectedModel")
        defer {
            if let previousSelectedModel = previousSelectedModel {
                UserDefaults.standard.set(previousSelectedModel, forKey: "selectedModel")
            } else { UserDefaults.standard.removeObject(forKey: "selectedModel") }
        }
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
             "quantization": "BF16", "artifact_modes": ["text", "vision-language"],
             "default_artifact_mode": "vision-language", "downloadable": true, "installed_ids": [String]()]
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
        func catalogReport(_ family: String, status: String = "ready", items: [[String: Any]]? = nil) -> [String: Any] {
            ["schema_version": 1, "family_id": family, "status": status, "message": "Fixture catalog message",
             "items": items ?? (status == "ready" ? [catalogItem()] : []), "has_more": false]
        }
        func catalogReply(_ request: [String: Any], status: String = "ready", items: [[String: Any]]? = nil) -> [String: Any] {
            ["event": "model_catalog", "request_id": request["request_id"]!, "family_id": request["family_id"]!,
             "report": catalogReport(request["family_id"] as! String, status: status, items: items)]
        }
        func removal(_ action: String = "trash") -> [String: Any] {
            ["schema_version": 1, "action": action, "label": action == "trash" ? "Move to Trash" : "Remove from Stark",
             "message": action == "trash" ? "Its files will move to Trash." : "Its folder will remain on this Mac."]
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
            guard let families = engine.modelFamilies?.families, families.count == 13,
                  let classifier = families.first(where: { $0.modelKind == "classifier" }),
                  classifier.modes.values.allSatisfy({ $0.status == .unsupported }),
                  let qwen3 = families.first(where: { $0.id == "qwen3" }),
                  qwen3.sourceFormats == ["safetensors-float", "mlx-affine-4bit", "mlx-affine-8bit"],
                  qwen3.outputDtypes == ["bfloat16", "float16", "float32"],
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
                  ModelsSheetTab.initial == .onThisMac,
                  ModelsSheetTab.onThisMac.title(modelCount: 0) == "On this Mac (0)",
                  ModelsSheetTab.onThisMac.emptyTitle == "No models yet",
                  ModelsSheetTab.onThisMac.emptyMessage == "Install a model to add it here." else { return false }
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
            return rows.map(\.name) == ["Gemma 2", "Gemma 3", "Granite", "LFM2", "Llama", "LLaMBA",
                "Mistral", "Phi-3", "Qwen2", "Qwen3", "Qwen3.5", "SmolLM3"] &&
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
                  inventory.groups[0].models[0].isRunning, inventory.models[0].detailLabels.count == 3,
                  inventory.models[0].detailLabels.first == "BF16",
                  inventory.models[0].detailLabels.last == "Version aaaaaaa" else { return false }
            var unknown = modelDisplay()
            for key in ["family_id", "quantization", "weight_bytes", "source_revision"] { unknown[key] = NSNull() }
            guard let descriptor = InstalledModelDisplay(unknown) else { return false }
            return descriptor.familyID == nil && descriptor.quantization == nil && descriptor.weightBytes == nil && descriptor.sourceRevision == nil
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
                      inventory.groups[0].family == nil, inventory.models[0].detailLabels.isEmpty else { return false }
            }
            return true
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
        case "catalog-schema-bounds":
            var report = catalogReport("qwen3")
            report["schema_version"] = 2
            guard FamilyModelCatalog(report) == nil else { return false }
            report = catalogReport("qwen3"); report["items"] = (0..<31).map { catalogItem("entry-\($0)") }
            guard FamilyModelCatalog(report) == nil else { return false }
            report = catalogReport("qwen3", items: [catalogItem(), catalogItem()])
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
            engine.installCatalogItem(id: visible.groups[0].items[0].id, artifactMode: .text)
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
            let raw = [
                groupedCatalogItem("compact", repo: "owner/Cafe-0.6B-4bit", name: "Café-0.6B-4bit", group: "1", groupName: "Café Model", publisher: "owner", quantization: "4-bit"),
                groupedCatalogItem("full", repo: "owner/Cafe-0.6B", name: "Café-0.6B", group: "1", groupName: "Café Model", publisher: "owner")
            ]
            guard let catalog = FamilyModelCatalog(catalogReport("qwen3", items: raw)) else { return false }
            let visible = CatalogBrowserInventory(items: catalog.items, query: " CAFE 4bit 0.6b ")
            guard visible.groups.count == 1, visible.groups[0].items.map(\.id) == ["compact"] else { return false }
            return CatalogBrowserInventory(items: catalog.items, query: "cafe 8-bit").groups.isEmpty &&
                CatalogBrowserInventory(items: catalog.items, query: "aAaAaAa").groups.first?.items.count == 2 &&
                CatalogBrowserInventory(items: catalog.items, query: "images").groups.first?.items.count == 2 &&
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
            engine.installCatalogItem(id: selected.id, artifactMode: .text)
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
            engine.installCatalogItem(id: "entry", artifactMode: .text)
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
        case "catalog-install-stale-error":
            engine.requestModelCatalog(familyID: "qwen3")
            let browse = command(); engine.apply(catalogReply(browse))
            engine.installCatalogItem(id: "entry", artifactMode: .text)
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
            engine.installCatalogItem(id: "entry", artifactMode: .text)
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
            engine.installCatalogItem(id: "entry", artifactMode: .text)
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
            engine.installCatalogItem(id: "entry", artifactMode: .text)
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

    def test_grouped_text_and_image_variants_have_distinct_stored_mode_label(self):
        self.exercise("models-artifact-mode-label")

    def test_catalog_family_changes_and_close_fence_late_results(self):
        self.exercise("image-catalog-stale-family")

    def test_catalog_offline_retry_preserves_runtime_state(self):
        self.exercise("catalog-offline-retry")

    def test_catalog_schema_and_item_bounds_are_enforced(self):
        self.exercise("catalog-schema-bounds")

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
