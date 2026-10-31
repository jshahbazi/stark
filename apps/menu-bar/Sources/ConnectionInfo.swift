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
    var imageVerification = "unverified"
    var tools = false
    var toolParser = ""

    static let usageNotes = [
        "Choose OpenAI-compatible or Custom OpenAI in your client.",
        "Choose Chat mode and use Run in Stark to serve the selected model before sending requests.",
        "Keep Stark open. The address is reachable by native applications on this Mac only; containers and other devices cannot connect.",
        "The API key is saved in macOS Keychain. The configured port is reused across restarts.",
        "Regenerating the API key requires updating every connected app. Viewing or copying settings never changes the key.",
        "Tool support depends on the running model, its chat template and Stark's parser. The client executes tools and controls approvals.",
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
                "tools": tools && !runningModelID.isEmpty, "images": images && !runningModelID.isEmpty, "responses_api": false
            ],
            "tool_parser": tools && !runningModelID.isEmpty ? toolParser : NSNull(),
            "image_verification": images && !runningModelID.isEmpty && ["locally_tested", "reference_verified"].contains(imageVerification)
                ? imageVerification : "unverified",
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
        // This example never embeds the credential. A shell builtin reads it with echo
        // disabled and pipes it to curl's config input; it is not exported or an argv value.
        return "(set +x; unset stark_api_key; printf 'Stark API key: '; IFS= read -r -s stark_api_key; printf '\\n'; \\\n  printf 'header = \"Authorization: Bearer %s\"\\n' \"$stark_api_key\" | \\\n  curl --config - --no-buffer \(shellQuote(baseURL + "/chat/completions")) \\\n  -H 'Content-Type: application/json' \\\n  -d \(shellQuote(body)); unset stark_api_key)"
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
