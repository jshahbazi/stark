import AppKit
import SwiftUI

struct ConnectionDetailsView: View {
    @ObservedObject var engine: EngineController
    let manage: () -> Void
    let close: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var section = Section.apps
    @State private var revealKey = false
    @State private var copied = ""
    @State private var inspectedModelID = ""
    @State private var portText = ""
    @State private var confirmRegeneration = false

    private enum Section: String, CaseIterable {
        case apps = "Apps"
        case other = "Connect your own app"
    }

    private var canConnect: Bool { !engine.apiBaseURL.isEmpty && !engine.apiKey.isEmpty }
    private var hasKey: Bool { !engine.apiKey.isEmpty }
    private var modelID: String {
        // Unrelated library work can be busy while the same chat worker stays loaded.
        engine.loadedMode == .chat && !["idle", "error", "loading", "stopping"].contains(engine.state) ? engine.loadedID : ""
    }
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
            contextCapacityKnown: engine.contextCapacityKnown,
            images: engine.supportsImages && !modelID.isEmpty,
            imageVerification: engine.imageVerification,
            tools: engine.supportsTools && !modelID.isEmpty,
            toolParser: engine.toolParser
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Connections").font(.title2.weight(.semibold))
                        Text("Use your local model in the apps you love.").font(.callout).foregroundStyle(.secondary)
                    }
                    modelSummary
                    if engine.apiMessage != "Local API is available on this Mac." { apiStatus }
                    OpenCodeEngineApprovalView(connection: engine.desktopConnections.openCode)
                    Picker("Connection type", selection: $section) {
                        ForEach(Section.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented).labelsHidden().accessibilityLabel("Connection type")
                    .frame(maxWidth: .infinity)
                    if section == .apps { apps } else { manualSettings }
                    if !copied.isEmpty { Text("Copied \(copied)").font(.caption).foregroundStyle(.secondary) }
                }
                .padding(24).frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            HStack {
                Label("Only on this Mac · Keep Stark open", systemImage: "laptopcomputer")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Done", action: close).keyboardShortcut(.cancelAction)
            }.padding(.horizontal, 24).padding(.vertical, 14)
        }
        .frame(width: 570, height: 680)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear { portText = String(engine.apiPort) }
        .onChange(of: engine.apiPort) { _, port in portText = String(port) }
        .onChange(of: engine.apiKey) { _, _ in revealKey = false }
        .onChange(of: section) { _, _ in revealKey = false; copied = "" }
        .onDisappear { revealKey = false }
    }

    private var modelSummary: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: "cpu").font(.title3).foregroundStyle(.secondary)
                    .frame(width: 32, height: 32).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 3) {
                    if !modelID.isEmpty {
                        Text(engine.activeName)
                            .font(.headline).fixedSize(horizontal: false, vertical: true)
                        Text(engine.supportsTools ? "Chat mode · Tool use available" : "Chat mode · Text chat available")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        HStack(alignment: .center, spacing: 8) {
                            Image(systemName: "circle.fill")
                                .font(.system(size: 8)).foregroundStyle(.red).accessibilityHidden(true)
                            Text("Model is not running - will not show up in OpenWork")
                                .font(.body).fixedSize(horizontal: false, vertical: true)
                        }
                        .phaseAnimator(reduceMotion ? [false] : [false, true]) { content, dimmed in
                            content.opacity(dimmed ? 0.4 : 1)
                        } animation: { _ in
                            .easeInOut(duration: 1.2)
                        }
                    }
                }
                Spacer(minLength: 8)
                if !modelID.isEmpty {
                    Label(engine.state == "generating" ? "In use" : engine.busy ? "Busy" : "Running", systemImage: "circle.fill")
                        .font(.caption).foregroundStyle(engine.busy ? .orange : .green)
                }
            }
            if let error = engine.errorMessage {
                Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            Button("Manage Models", action: manage).buttonStyle(.link).font(.caption)
        }
        .padding(14).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    private var apiStatus: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(engine.apiMessage).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            if !canConnect { Button("Retry API") { engine.retryAPI() }.disabled(engine.apiCredentialBusy) }
        }
    }

    private var apps: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Connect an app").font(.headline)
                Text("Set up once. Stark remembers your connection.").font(.caption).foregroundStyle(.secondary)
            }
            VStack(spacing: 0) {
                OpenWorkConnectionView(connection: engine.desktopConnections.openWork)
                Divider()
                OpenCodeConnectionView(connection: engine.desktopConnections.openCode)
            }
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color(nsColor: .separatorColor), lineWidth: 0.5))
            Label("No API key to paste. Your key stays in Stark’s Keychain.", systemImage: "key")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            DisclosureGroup("Having trouble connecting?") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("If you added a provider named stark manually, remove it in that app’s settings, then reconnect here.")
                    Text("If Stark is missing in OpenCode, quit OpenCode, run opencode service stop in Terminal, reopen OpenCode, then allow its new engine here.")
                    Text("For the standalone OpenCode CLI, use the settings under Connect your own app. A terminal session sharing an approved Desktop engine can also use that engine’s Stark connection.")
                }
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.top, 8)
            }.font(.caption)
        }
    }

    private var manualSettings: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Connect any compatible app").font(.headline)
                Text("Choose OpenAI-compatible or Custom OpenAI in its provider settings.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            VStack(alignment: .leading, spacing: 14) {
                field("Base URL", value: engine.apiBaseURL, placeholder: "Unavailable")
                Divider()
                modelSettings
                Divider()
                VStack(alignment: .leading, spacing: 5) {
                    Text("API key").font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Text(hasKey ? (revealKey ? engine.apiKey : String(repeating: "•", count: 24)) : "Unavailable")
                            .font(.system(.callout, design: .monospaced)).textSelection(.disabled).privacySensitive()
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer()
                        Button(revealKey ? "Hide" : "Show") { revealKey.toggle() }.disabled(!hasKey)
                            .accessibilityLabel(revealKey ? "Hide API key" : "Show API key")
                        Button("Copy") { copy(engine.apiKey, label: "API key", sensitive: true) }.disabled(!hasKey)
                            .accessibilityLabel("Copy API key")
                    }
                }
            }
            .padding(14).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
            Label(hasKey ? "Saved securely. These settings stay the same when Stark restarts."
                         : "Stark stores your API key in macOS Keychain. See the API recovery details above.", systemImage: "key")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            DisclosureGroup("Advanced settings") { advancedSettings.padding(.top, 12) }.font(.caption)
        }
    }

    private var modelSettings: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Model ID").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if engine.models.count > 1 {
                    Picker("Model", selection: Binding(get: { inspectedLibraryID }, set: { inspectedModelID = $0 })) {
                        ForEach(engine.models) { Text($0.apiID).tag($0.id) }
                    }
                    .pickerStyle(.menu).labelsHidden().frame(maxWidth: 200).accessibilityLabel("Model to view ID")
                }
            }
            HStack {
                Text(displayedModelID.isEmpty ? "Install a model first" : displayedModelID)
                    .font(.system(.callout, design: .monospaced)).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                Spacer()
                Button("Copy") { copy(displayedModelID, label: "Model ID") }
                    .disabled(displayedModelID.isEmpty).accessibilityLabel("Copy Model ID")
            }
            Text(displayedModelID.isEmpty ? "Install a model in Manage Models to get its connection ID."
                 : !modelID.isEmpty && inspectedLibraryID == modelID
                 ? "This is the model currently served to your apps."
                 : "Preview only. This model is not running in Chat mode. Choose Chat mode and Run in Stark before using it.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var advancedSettings: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Saved port").foregroundStyle(.secondary)
                TextField("8000", text: $portText).frame(width: 70)
                    .accessibilityLabel("Local API listening port").disabled(engine.apiCredentialBusy)
                Button("Apply Port") { engine.configureAPIPort(portText) }.disabled(engine.apiCredentialBusy)
                Button("Retry API") { engine.retryAPI() }.disabled(engine.apiCredentialBusy)
                Spacer()
            }
            Text("Stark reuses this port. If it is occupied, close the conflicting app or choose another port. Reconnect apps after changing it.")
                .foregroundStyle(.secondary)
            HStack {
                Button("Copy All as JSON") {
                    if let json = connectionInfo.json { copy(json, label: "connection JSON (includes API key)", sensitive: true) }
                }.disabled(!canConnect || displayedModelID.isEmpty)
                Button("Copy curl Example") { copy(connectionInfo.curl, label: "curl example (prompts for API key)") }
                    .disabled(!canConnect || modelID.isEmpty)
            }
            Text("JSON includes your API key and the inspected model’s settings. The curl example uses the running chat model and prompts for the key.")
                .foregroundStyle(.secondary)
            Button("Regenerate API Key…") { confirmRegeneration = true }
                .disabled(engine.apiCredentialBusy)
                .confirmationDialog("Replace Stark's API key?", isPresented: $confirmRegeneration, titleVisibility: .visible) {
                    Button("Regenerate API Key", role: .destructive) { revealKey = false; engine.regenerateAPIKey() }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Apps using a copied API key will need the replacement. Managed OpenWork and OpenCode connections refresh automatically. A Keychain storage failure keeps the existing key working.")
                }
            Divider()
            Text("API capabilities").font(.headline)
            if !modelID.isEmpty {
                Text(engine.contextCapacityKnown
                     ? "Context: \(engine.apiContextTokens) tokens · Reply limit: \(engine.apiMaxNewTokens) tokens"
                     : "Context maximum: not declared · Reply limit: \(engine.apiMaxNewTokens) tokens").foregroundStyle(.secondary)
            }
            Text("Supports Chat Completions, live streaming, temperature, top-p, seed and stop sequences. Up to 8 requests can wait in order. Cancelled requests keep the model loaded. The Responses API is unavailable. Model limits and capabilities are available through the API.")
                .foregroundStyle(.secondary)
            Text(engine.supportsTools && !modelID.isEmpty
                 ? "The running model supports function tool requests. Your client executes tools and controls approvals."
                 : "Tool use is unavailable without a running chat model that supports it.").foregroundStyle(.secondary)
            Text(engine.supportsImages && !modelID.isEmpty
                 ? "The running model accepts one PNG or JPEG up to 4 MiB through inline image data URLs."
                 : "Image input is unavailable without a running chat model and app build that support it.").foregroundStyle(.secondary)
            if engine.supportsImages && !modelID.isEmpty { Text(engine.imageVerificationNote).foregroundStyle(.secondary) }
            Text("Containers and other devices cannot reach this address. The API key stays in macOS Keychain across restarts and upgrades. Copying or showing it never changes it.")
                .foregroundStyle(.secondary)
        }.font(.caption).fixedSize(horizontal: false, vertical: true)
    }

    private func field(_ title: String, value: String, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack {
                Text(value.isEmpty ? placeholder : value)
                    .font(.system(.callout, design: .monospaced)).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                Spacer()
                Button("Copy") { copy(value, label: title) }.disabled(value.isEmpty).accessibilityLabel("Copy \(title)")
            }
        }
    }

    private func copy(_ value: String, label: String, sensitive: Bool = false) {
        copied = ConnectionClipboard.write(value, sensitive: sensitive) ? label : ""
    }
}

private struct ManagedAppConnectionRow: View {
    let name: String
    let symbol: String
    let approved: Bool
    let busy: Bool
    let disconnectPending: Bool
    var needsApproval = false
    let message: String
    let detail: String
    let connect: () -> Void
    let disconnect: () -> Void
    @State private var expanded = false

    private var status: String {
        if disconnectPending { return "Disconnect incomplete" }
        if busy { return "Working…" }
        if needsApproval { return "Needs approval · New engine" }
        return approved ? "Access allowed" : "Not connected"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: symbol).font(.title3).frame(width: 32, height: 32).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 3) {
                    Text(name).font(.headline)
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if busy { ProgressView().controlSize(.small).accessibilityLabel("Updating \(name) connection") }
                if disconnectPending {
                    Button("Retry Disconnect", action: disconnect).disabled(busy).accessibilityLabel("Retry disconnecting \(name)")
                } else if approved {
                    Button("Manage", systemImage: expanded ? "chevron.up" : "chevron.down") { expanded.toggle() }
                        .disabled(busy).accessibilityLabel("Manage \(name) connection").accessibilityValue(expanded ? "Expanded" : "Collapsed")
                } else {
                    Button("Connect", action: connect).buttonStyle(.borderedProminent).disabled(busy).accessibilityLabel("Connect to \(name)")
                }
            }
            if !message.isEmpty {
                Text(message).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if approved && expanded && !disconnectPending {
                Text(detail).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 12) {
                    Button("Reconnect", action: connect).disabled(busy).accessibilityLabel("Reconnect \(name)")
                    Button("Disconnect", role: .destructive, action: disconnect).disabled(busy).accessibilityLabel("Disconnect \(name)")
                }
            }
        }.padding(16).onChange(of: approved) { _, allowed in expanded = allowed }
    }
}

private struct OpenWorkConnectionView: View {
    @ObservedObject var connection: OpenWorkConnectionController

    var body: some View {
        ManagedAppConnectionRow(
            name: "OpenWork", symbol: "macwindow", approved: connection.approved,
            busy: connection.busy, disconnectPending: connection.disconnectPending,
            message: connection.message == "OpenWork is not connected." ? "" : connection.message,
            detail: "Select Stark in OpenWork’s model picker. It uses the model currently running in Chat mode in Stark.",
            connect: connection.connect, disconnect: connection.disconnect
        )
    }
}

private struct OpenCodeConnectionView: View {
    @ObservedObject var connection: OpenCodeConnectionController

    var body: some View {
        ManagedAppConnectionRow(
            name: "OpenCode Desktop", symbol: "terminal", approved: connection.approved,
            busy: connection.busy, disconnectPending: connection.disconnectPending,
            needsApproval: connection.pendingEngine != nil,
            message: connection.message == "OpenCode is not connected." ? "" : connection.message,
            detail: "Select Stark in OpenCode’s model picker. Stark asks once for each new Desktop background engine. A terminal session sharing that approved engine can also use its Stark connection.",
            connect: connection.connect, disconnect: connection.disconnect
        )
    }
}

private struct OpenCodeEngineApprovalView: View {
    @ObservedObject var connection: OpenCodeConnectionController

    var body: some View {
        if let engine = connection.pendingEngine {
            VStack(alignment: .leading, spacing: 8) {
                Label("Allow OpenCode’s new engine?", systemImage: "exclamationmark.shield").font(.headline)
                Text("Background engine started \(Date(timeIntervalSince1970: TimeInterval(engine.seconds)).formatted(date: .abbreviated, time: .shortened)). Allow it if you opened or restarted OpenCode.")
                    .font(.caption).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Spacer()
                    Button("Deny") { connection.deny() }.disabled(connection.busy)
                    Button("Allow") { connection.allow() }.buttonStyle(.borderedProminent).disabled(connection.busy)
                }
            }.padding(14).background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}
