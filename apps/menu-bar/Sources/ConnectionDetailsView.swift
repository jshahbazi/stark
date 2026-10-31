import AppKit
import SwiftUI

struct ConnectionDetailsView: View {
    @ObservedObject var engine: EngineController
    @State private var revealKey = false
    @State private var copied = ""
    @State private var inspectedModelID = ""
    @State private var portText = ""
    @State private var confirmRegeneration = false

    private var canConnect: Bool { !engine.apiBaseURL.isEmpty && !engine.apiKey.isEmpty }
    private var hasKey: Bool { !engine.apiKey.isEmpty }
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
            images: engine.supportsImages && engine.loadedMode == .chat,
            imageVerification: engine.imageVerification,
            tools: engine.supportsTools && engine.loadedMode == .chat,
            toolParser: engine.toolParser
        )
    }

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 16) {
            Text("Connect to Stark").font(.title2.weight(.semibold))
            Text("In your application's provider settings, choose OpenAI-compatible or Custom OpenAI, then paste these values.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(engine.apiMessage).font(.callout)
            field("Base URL", value: engine.apiBaseURL, placeholder: "Unavailable")
            HStack {
                Text("Saved port").font(.caption).foregroundStyle(.secondary)
                TextField("8000", text: $portText).frame(width: 70)
                    .accessibilityLabel("Local API listening port")
                Button("Apply Port") { engine.configureAPIPort(portText) }
                Button("Retry API") { engine.retryAPI() }
                Spacer()
            }
            Text("Stark reuses this port. If it is occupied, close the conflicting app or choose another port. Update connected apps after changing it.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
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
                    Text(hasKey ? (revealKey ? engine.apiKey : String(repeating: "•", count: 24)) : "Unavailable")
                        .font(.system(.callout, design: .monospaced)).textSelection(.disabled)
                    Spacer()
                    Button(revealKey ? "Hide" : "Show") { revealKey.toggle() }.disabled(!hasKey)
                    Button("Copy") { copy(engine.apiKey, label: "API key", sensitive: true) }.disabled(!hasKey)
                        .accessibilityLabel("Copy API key")
                }
                Button("Regenerate API Key…") { confirmRegeneration = true }
                    .confirmationDialog("Replace Stark's API key?", isPresented: $confirmRegeneration, titleVisibility: .visible) {
                        Button("Regenerate API Key", role: .destructive) {
                            revealKey = false
                            engine.regenerateAPIKey()
                        }
                        Button("Cancel", role: .cancel) {}
                    } message: {
                        Text("Every connected app will need the replacement key. A Keychain storage failure keeps the existing key working.")
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
            Text("Keep Stark open. These settings work on this Mac; containers and other devices cannot reach this address. The API key stays in macOS Keychain across restarts and upgrades. Copying or showing it never changes it.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Supports Chat Completions, live streaming, temperature, top-p, seed and stop sequences. Up to 8 requests can wait in order. Cancelled API requests keep the model loaded. The Responses API is unavailable. Model limits and capabilities are available through the API.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(engine.supportsTools && engine.loadedMode == .chat
                 ? "The running model supports function tool requests. Your client executes tools and controls approvals."
                 : "Tool use is unavailable for the running model and its chat template. Ordinary chat remains available.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(engine.supportsImages && engine.loadedMode == .chat
                 ? "The running model accepts one PNG or JPEG up to 4 MiB through inline image data URLs."
                 : "Image input is unavailable for the running model and app build.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if engine.supportsImages && engine.loadedMode == .chat {
                Text(engine.imageVerificationNote).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Copy All as JSON") {
                    if let json = connectionInfo.json { copy(json, label: "connection JSON (includes API key)", sensitive: true) }
                }
                .disabled(!canConnect || displayedModelID.isEmpty)
                .help("Copy connection settings, API key, model IDs, capabilities and usage notes as JSON.")
                Button("Copy curl Example") { copy(connectionInfo.curl, label: "curl example (prompts for API key)") }
                    .disabled(!canConnect || modelID.isEmpty)
                Spacer()
            }
            Text(copied.isEmpty ? " " : "Copied \(copied)").font(.caption).foregroundStyle(.secondary)
        }.padding(24).frame(width: 570)
        }.frame(width: 570, height: 790)
            .onAppear { portText = String(engine.apiPort) }
            .onChange(of: engine.apiPort) { _, port in portText = String(port) }
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
