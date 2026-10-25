import Foundation

@main
struct ConnectionInfoTests {
    static func main() throws {
        let info = ConnectionInfo(
            baseURL: "http://127.0.0.1:8000/v1", apiKey: "synthetic-key",
            modelID: "Local \"model\"\nname", runningModelID: "Qwen/Qwen3.8-27B",
            status: "ready", contextTokens: 8192, maxNewTokens: 512
        )
        let json = info.json!
        let parsed = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        precondition(parsed["model"] as? String == info.modelID)
        precondition(parsed["api_key"] as? String == "synthetic-key")
        precondition(parsed["running_model"] as? String == info.runningModelID)
        precondition(parsed["selected_model_is_running"] as? Bool == false)
        let limits = parsed["running_model_limits"] as! [String: Int]
        precondition(limits["context_tokens"] == 8192 && limits["max_output_tokens"] == 512)
        precondition(json.contains("\n") && json.contains("Qwen/Qwen3.8-27B"))

        let stopped = ConnectionInfo(
            baseURL: info.baseURL, apiKey: info.apiKey, modelID: "JevK5", runningModelID: "",
            status: "idle", contextTokens: 8192, maxNewTokens: 512
        )
        let stoppedJSON = try JSONSerialization.jsonObject(with: Data(stopped.json!.utf8)) as! [String: Any]
        precondition(stoppedJSON["running_model"] is NSNull)
        precondition(stoppedJSON["running_model_limits"] is NSNull)
        precondition(stoppedJSON["running_model_curl_example"] is NSNull)

        let quoted = ConnectionInfo(
            baseURL: info.baseURL, apiKey: info.apiKey, modelID: "Model's name", runningModelID: "Model's name",
            status: "ready", contextTokens: 8192, maxNewTokens: 512
        )
        let quotedJSON = try JSONSerialization.jsonObject(with: Data(quoted.json!.utf8)) as! [String: Any]
        precondition(quotedJSON["selected_model_is_running"] as? Bool == true)
        // Capture curl arguments without making a network request.
        let shell = Process()
        shell.executableURL = URL(fileURLWithPath: "/bin/sh")
        shell.arguments = ["-c", "curl() { for arg do printf '%s\\n' \"$arg\"; done; }; " + quoted.curl]
        let pipe = Pipe()
        shell.standardOutput = pipe
        try shell.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        shell.waitUntilExit()
        precondition(shell.terminationStatus == 0)
        let body = String(data: output, encoding: .utf8)!.split(separator: "\n").last!
        let request = try JSONSerialization.jsonObject(with: Data(body.utf8)) as! [String: Any]
        precondition(request["model"] as? String == quoted.runningModelID)
        precondition(request["max_tokens"] as? Int == 128)
        print("Connection JSON and curl checks passed")
    }
}
