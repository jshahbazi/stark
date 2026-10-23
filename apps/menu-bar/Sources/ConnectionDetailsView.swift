import AppKit
import SwiftUI

struct ConnectionDetailsView: View {
    @ObservedObject var engine: EngineController
    @State private var revealKey = false
    @State private var copied = ""

    private var canConnect: Bool { !engine.apiBaseURL.isEmpty && !engine.apiKey.isEmpty }
    private var curl: String {
        let request: [String: Any] = [
            "model": engine.loadedID,
            "messages": [["role": "user", "content": "Hello"]],
            "temperature": 0,
            "max_tokens": 128
        ]
        let data = (try? JSONSerialization.data(withJSONObject: request, options: [.sortedKeys])) ?? Data()
        let body = String(data: data, encoding: .utf8) ?? "{}"
        return "curl '\(engine.apiBaseURL)/chat/completions' \\\n  -H 'Authorization: Bearer \(engine.apiKey)' \\\n  -H 'Content-Type: application/json' \\\n  -d '\(body)'"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Connect to Stark").font(.title2.weight(.semibold))
            Text("In your application's provider settings, choose OpenAI-compatible or Custom OpenAI, then paste these values.")
                .font(.callout).foregroundStyle(.secondary)
            Text(engine.apiMessage).font(.callout)
            field("Base URL", value: engine.apiBaseURL, placeholder: "Unavailable")
            field("Model ID", value: engine.loadedID, placeholder: "Run a model first")
            VStack(alignment: .leading, spacing: 5) {
                Text("API key").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text(canConnect ? (revealKey ? engine.apiKey : String(repeating: "•", count: 24)) : "Unavailable")
                        .font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                    Spacer()
                    Button(revealKey ? "Hide" : "Show") { revealKey.toggle() }.disabled(!canConnect)
                    Button("Copy") { copy(engine.apiKey, label: "API key") }.disabled(!canConnect)
                        .accessibilityLabel("Copy API key")
                }
            }
            Divider()
            StatusLine(engine: engine)
            Text(engine.loadedID.isEmpty ? "Choose a model in Stark and select Run before connecting." : "Running model: \(engine.activeName)")
                .font(.callout)
            Text("Keep Stark open. These settings work on this Mac; containers and other devices cannot reach this address. The API key changes when Stark reopens.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Supports text Chat Completions and model discovery. Set temperature to 0 and the reply limit to 512 tokens or less. Streaming responses arrive after generation finishes. Tools, images and the Responses API are unavailable. Busy requests return an error for the client to retry.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Copy curl Example") { copy(curl, label: "curl example (includes API key)") }
                    .disabled(!canConnect || engine.loadedID.isEmpty)
                Spacer()
                Text(copied.isEmpty ? "" : "Copied \(copied)").font(.caption).foregroundStyle(.secondary)
            }
        }.padding(24).frame(width: 570)
    }

    private func field(_ title: String, value: String, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack {
                Text(value.isEmpty ? placeholder : value)
                    .font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                Spacer()
                Button("Copy") { copy(value, label: title) }.disabled(value.isEmpty)
                    .accessibilityLabel("Copy \(title)")
            }
        }
    }

    private func copy(_ value: String, label: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
        copied = label
    }
}
