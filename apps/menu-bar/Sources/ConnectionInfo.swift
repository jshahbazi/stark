import Foundation

struct ConnectionInfo {
    let baseURL: String
    let apiKey: String
    let modelID: String
    let runningModelID: String
    let selectedModelIsRunning: Bool
    let status: String
    let contextTokens: Int
    let maxNewTokens: Int
    var images = false

    static let usageNotes = [
        "Choose OpenAI-compatible or Custom OpenAI in your client.",
        "Choose Chat mode and use Run in Stark to serve the selected model before sending requests.",
        "Keep Stark open. The address is reachable by native applications on this Mac only; containers and other devices cannot connect.",
        "The API key changes when Stark reopens. Copy these settings again after restarting.",
        "Up to 8 requests can wait in order. Cancelled requests keep the model loaded."
    ]

    var json: String? {
        let value: [String: Any] = [
            "provider": "OpenAI-compatible",
            "base_url": baseURL,
            "api_key": apiKey,
            "model": modelID,
            "running_model": runningModelID.isEmpty ? NSNull() : runningModelID as Any,
            "selected_model_is_running": selectedModelIsRunning,
            "status": status,
            "running_model_limits": runningModelID.isEmpty ? NSNull() : [
                "context_tokens": contextTokens,
                "max_output_tokens": maxNewTokens
            ] as Any,
            "endpoints": [
                "chat_completions": baseURL + "/chat/completions",
                "models": baseURL + "/models",
                "capabilities": baseURL + "/capabilities"
            ],
            "capabilities": [
                "text_chat_completions": true, "streaming": true,
                "tools": false, "images": images && !runningModelID.isEmpty, "responses_api": false
            ],
            "sampling_parameters": ["temperature", "top_p", "top_k", "min_p", "seed", "stop"],
            "usage_notes": Self.usageNotes,
            "running_model_curl_example": runningModelID.isEmpty ? NSNull() : curl as Any
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    var curl: String {
        let request: [String: Any] = [
            "model": runningModelID,
            "messages": [["role": "user", "content": "Hello"]],
            "temperature": 0, "stream": true,
            "max_tokens": min(128, max(1, maxNewTokens))
        ]
        let data = (try? JSONSerialization.data(withJSONObject: request, options: [.sortedKeys])) ?? Data()
        let body = String(data: data, encoding: .utf8) ?? "{}"
        return "curl --no-buffer \(shellQuote(baseURL + "/chat/completions")) \\\n  -H \(shellQuote("Authorization: Bearer " + apiKey)) \\\n  -H 'Content-Type: application/json' \\\n  -d \(shellQuote(body))"
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
