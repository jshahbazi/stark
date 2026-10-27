import AppKit
import SwiftUI

struct ConnectionDetailsView: View {
    @ObservedObject var engine: EngineController
    @State private var revealKey = false
    @State private var copied = ""
    @State private var inspectedModelID = ""

    private var canConnect: Bool { !engine.apiBaseURL.isEmpty && !engine.apiKey.isEmpty }
    private var modelID: String { engine.loadedMode == .chat ? engine.loadedID : "" }
    private var inspectedLibraryID: String {
        if engine.models.contains(where: { $0.id == inspectedModelID }) { return inspectedModelID }
        if engine.models.contains(where: { $0.id == engine.loadedID }) { return engine.loadedID }
        if engine.models.contains(where: { $0.id == engine.selectedID }) { return engine.selectedID }
        return engine.models.first?.id ?? ""
    }
    private var displayedModelID: String {
        engine.models.first { $0.id == inspectedLibraryID }?.apiID ?? ""
    }
    private var connectionInfo: ConnectionInfo {
        ConnectionInfo(
            baseURL: engine.apiBaseURL, apiKey: engine.apiKey, modelID: displayedModelID,
            runningModelID: engine.models.first { $0.id == modelID }?.apiID ?? "",
            selectedModelIsRunning: !modelID.isEmpty && inspectedLibraryID == modelID,
            status: engine.state, contextTokens: engine.apiContextTokens, maxNewTokens: engine.apiMaxNewTokens,
            images: engine.supportsImages && engine.loadedMode == .chat
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Connect to Stark").font(.title2.weight(.semibold))
            Text("In your application's provider settings, choose OpenAI-compatible or Custom OpenAI, then paste these values.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(engine.apiMessage).font(.callout)
            field("Base URL", value: engine.apiBaseURL, placeholder: "Unavailable")
            VStack(alignment: .leading, spacing: 5) {
                Text("Model ID").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text(displayedModelID.isEmpty ? "Install a model first" : displayedModelID)
                        .font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                    Spacer()
                    Picker("Model", selection: Binding(
                        get: { inspectedLibraryID },
                        set: { inspectedModelID = $0 }
                    )) {
                        if engine.models.isEmpty { Text("No models installed").tag("") }
                        ForEach(engine.models) { Text($0.apiID).tag($0.id) }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 200)
                    .disabled(engine.models.isEmpty)
                    .accessibilityLabel("Model to view ID")
                    Button("Copy") { copy(displayedModelID, label: "Model ID") }
                        .disabled(displayedModelID.isEmpty)
                        .accessibilityLabel("Copy Model ID")
                }
                Text("Select a model to view its ID and copy its JSON settings. Choose Chat mode and use Run in Stark to serve that model. The curl example uses the running model.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("API key").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text(canConnect ? (revealKey ? engine.apiKey : String(repeating: "•", count: 24)) : "Unavailable")
                        .font(.system(.callout, design: .monospaced)).textSelection(.disabled)
                    Spacer()
                    Button(revealKey ? "Hide" : "Show") { revealKey.toggle() }.disabled(!canConnect)
                    Button("Copy") { copy(engine.apiKey, label: "API key", sensitive: true) }.disabled(!canConnect)
                        .accessibilityLabel("Copy API key")
                }
            }
            Divider()
            StatusLine(engine: engine)
            Text(modelID.isEmpty ? "Choose Chat mode and a model in Stark, then select Run before connecting." : "Running model: \(engine.activeName)")
                .font(.callout)
            if !modelID.isEmpty {
                Text("Context: \(engine.apiContextTokens) tokens · Reply limit: \(engine.apiMaxNewTokens) tokens")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Keep Stark open. These settings work on this Mac; containers and other devices cannot reach this address. The API key changes when Stark reopens.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Supports Chat Completions, live streaming, temperature, top-p, seed and stop sequences. Up to 8 requests can wait in order. Cancelled API requests keep the model loaded. Tools and the Responses API are unavailable. Model limits and capabilities are available through the API.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(engine.supportsImages && engine.loadedMode == .chat
                 ? "The running model accepts one PNG or JPEG up to 4 MiB through inline image data URLs."
                 : "Image input is unavailable for the running model and app build.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Copy All as JSON") {
                    if let json = connectionInfo.json { copy(json, label: "connection JSON (includes API key)", sensitive: true) }
                }
                .disabled(!canConnect || displayedModelID.isEmpty)
                .help("Copy connection settings, API key, model IDs, capabilities and usage notes as JSON.")
                Button("Copy curl Example") { copy(connectionInfo.curl, label: "curl example (includes API key)", sensitive: true) }
                    .disabled(!canConnect || modelID.isEmpty)
                Spacer()
            }
            Text(copied.isEmpty ? " " : "Copied \(copied)").font(.caption).foregroundStyle(.secondary)
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

    private func copy(_ value: String, label: String, sensitive: Bool = false) {
        copied = ConnectionClipboard.write(value, sensitive: sensitive) ? label : ""
    }
}
