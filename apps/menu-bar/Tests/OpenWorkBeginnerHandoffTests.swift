import Foundation
import Darwin

@main struct OpenWorkBeginnerHandoffTests {
    @MainActor static func main() async {
        do {
            try checkDiscovery()
            for scenario in ["success", "missing-model", "wrong-provider", "create-failed", "bind-failed", "wrong-binding", "wrong-screen", "cancelled", "retry"] {
                try await run(scenario)
            }
        } catch { exit(1) }
    }

    enum Failed: Error { case check }
    @MainActor static func checkDiscovery() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("discovery.json")
        let valid: [String: Any] = ["version": 2, "identifier": "com.differentai.openwork",
            "baseUrl": "http://127.0.0.1:12345", "token": String(repeating: "a", count: 64)]
        for mutation in [["version": 1], ["baseUrl": "https://example.com"], ["baseUrl": "http://localhost:12345"],
                         ["baseUrl": "http://127.0.0.1:12345/path"], ["token": "short"], ["identifier": "other"]] as [[String: Any]] {
            let data = try JSONSerialization.data(withJSONObject: valid.merging(mutation) { _, new in new })
            try data.write(to: file)
            do { _ = try OpenWorkBeginnerHandoff(discoveryURL: file); throw Failed.check }
            catch is OpenWorkBeginnerHandoff.Failure { }
        }
        try JSONSerialization.data(withJSONObject: valid).write(to: file)
        _ = try OpenWorkBeginnerHandoff(discoveryURL: file)
        let link = directory.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        do { _ = try OpenWorkBeginnerHandoff(discoveryURL: link); throw Failed.check }
        catch is OpenWorkBeginnerHandoff.Failure { }
        guard chmod(file.path, 0o666) == 0 else { throw Failed.check }
        do { _ = try OpenWorkBeginnerHandoff(discoveryURL: file); throw Failed.check }
        catch is OpenWorkBeginnerHandoff.Failure { }
    }
    @MainActor static func run(_ scenario: String) async throws {
        var creates = 0
        var binds = 0
        var contexts = 0
        var current = true
        let handoff = OpenWorkBeginnerHandoff { path, body in
            if path == "/context" {
                contexts += 1
                return ["ok": true, "context": ["screen": ["workspaceId": "default", "sessionId": scenario == "wrong-screen" ? "old-task" : "new-task"]]]
            }
            if path == "/query" {
                if scenario == "cancelled" { current = false }
                return ["ok": true, "result": ["models": scenario == "missing-model" ? [] : [[
                    "providerId": scenario == "wrong-provider" ? "other" : "stark", "modelId": "running", "available": true,
                ]]]]
            }
            if body?["id"] as? String == "session.create_task" {
                creates += 1
                return ["ok": scenario != "create-failed", "result": "new-task"]
            }
            guard body?["id"] as? String == "session.set_model",
                  let args = body?["args"] as? [String: Any],
                  args["sessionId"] as? String == "new-task", args["workspaceId"] as? String == "default",
                  args["providerId"] as? String == "stark", args["modelId"] as? String == "running" else { throw Failed.check }
            binds += 1
            return ["ok": true, "result": ["ok": scenario != "bind-failed" && !(scenario == "retry" && binds == 1),
                "sessionId": "new-task", "workspaceId": "default",
                "model": ["providerId": "stark", "modelId": scenario == "wrong-binding" ? "different" : "running"]]]
        }
        var succeeded = false
        do { try await handoff.prepare(modelID: "running", isCurrent: { current }); succeeded = true }
        catch { }
        if scenario == "retry" {
            guard !succeeded else { throw Failed.check }
            try await handoff.prepare(modelID: "running")
            guard creates == 1, binds == 2 else { throw Failed.check }
        } else if scenario == "success" {
            guard succeeded, creates == 1, binds == 1, contexts == 3 else { throw Failed.check }
        } else {
            guard !succeeded else { throw Failed.check }
            if ["missing-model", "wrong-provider", "cancelled"].contains(scenario) {
                guard creates == 0, binds == 0 else { throw Failed.check }
            }
        }
    }
}
