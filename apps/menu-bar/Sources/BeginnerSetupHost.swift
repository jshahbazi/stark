import AppKit
import SwiftUI

@MainActor
final class BeginnerOpenWorkStatus: ObservableObject {
    @Published private(set) var applicationURL: URL?
    @Published private(set) var running = false
    @Published private(set) var liveConnection = false
    @Published private(set) var scope = UUID().uuidString
    @Published var message: String?
    private var applicationChecked = false
    private var connectionBasis = ""
    private var verifiedClientBasis = ""
    private var wasApproved = false

    func refresh(engine: EngineController, force: Bool = false) {
        let connection = engine.desktopConnections.openWork
        if wasApproved && !connection.approved { scope = UUID().uuidString }
        wasApproved = connection.approved
        if let applicationURL, !FileManager.default.fileExists(atPath: applicationURL.path) { applicationChecked = false }
        if force || !applicationChecked {
            let url = DesktopConnectionTesting.application(.openWork)
                ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: DesktopClient.openWork.bundleID)
            applicationURL = url.flatMap { DesktopClientPeer.verifyApplication($0, client: .openWork) ? $0 : nil }
            applicationChecked = applicationURL != nil
        }
        let application = DesktopApplicationIdentity.running(.openWork, configurationRoot: DesktopConnectionTesting.profile)
        running = application != nil
        if let application {
            applicationURL = application.bundleURL; applicationChecked = true
            verifiedClientBasis = application.process.instanceID + "|" + (application.profile?.path ?? "")
        }
        liveConnection = connection.hasDiscoveredModel(apiID: engine.activeModel?.apiID ?? "",
                                                      contextTokens: engine.apiContextTokens) && application != nil
        let basis = (applicationURL?.path ?? "") + "|" + verifiedClientBasis + "|" + connection.discoveryScope + "|" +
            (engine.desktopConnections.brokerPath ?? "") + "|" + engine.apiBaseURL
        if !connectionBasis.isEmpty, connectionBasis != basis { scope = UUID().uuidString }
        connectionBasis = basis
    }
}

/// Adapts real engine/connector state into the guide without a second installer.
struct BeginnerSetupHost: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var openWork: OpenWorkConnectionController
    @ObservedObject var session: BeginnerSetupSession
    @ObservedObject var interface: StarkInterfacePreferences
    @ObservedObject var observation: BeginnerOpenWorkStatus
    let selectInterface: (StarkInterfaceMode) -> Void
    let openChat: () -> Void

    private let refresh = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Stark").font(.headline)
                Spacer()
                StarkInterfacePicker(preferences: interface, selected: selectInterface)
            }.padding(.horizontal, 20).padding(.vertical, 14)
            Divider()
            BeginnerSetupView(session: session, snapshot: snapshot, actions: actions)
        }
        .onAppear { refreshOpenWork() }
        .onReceive(refresh) { _ in refreshOpenWork() }
        .onChange(of: openWork.approved) { _, _ in refreshOpenWork() }
        .onChange(of: engine.apiBaseURL) { _, _ in refreshOpenWork() }
    }

    var snapshot: BeginnerSetupSnapshot {
        var result = BeginnerSetupSnapshot()
        result.choices = (engine.beginnerReport?.items ?? []).map { item in
            BeginnerModelChoice(id: item.id, name: item.name,
                summary: "A starting model checked for this setup. Review its answers and any actions in OpenWork.",
                downloadBytes: item.downloadBytes, installedBytes: item.installedBytes, setupBytes: item.setupBytes,
                contextTokens: item.contextTokens,
                installedModelID: item.installedIDs.first { id in engine.models.contains(where: { $0.id == id }) },
                source: item.source, revision: item.revision,
                qualification: ([item.qualification, "License: " + item.license,
                    "Estimated runtime memory: " + ByteCountFormatter.string(fromByteCount: item.runtimeBytes, countStyle: .memory)] + item.caveats).joined(separator: "\n"))
        }
        // Keep the exact operation choice available if another surface refreshes recommendations.
        if let item = engine.beginnerInstallChoice,
           !result.choices.contains(where: { $0.id == item.id }),
           engine.beginnerInstallState != "idle", (session.installingChoiceID == item.id || session.selectedChoiceID == item.id) {
            result.choices.append(BeginnerModelChoice(id: item.id, name: item.name, summary: "Your selected starting model.",
                downloadBytes: item.downloadBytes, installedBytes: item.installedBytes, setupBytes: item.setupBytes,
                contextTokens: item.contextTokens, source: item.source, revision: item.revision,
                qualification: item.qualification))
        }
        result.choicesLoading = engine.beginnerChoicesLoading
        result.choicesError = engine.beginnerChoicesError
        result.deviceSummary = engine.beginnerReport?.deviceSummary ?? ""
        result.installChoiceID = engine.beginnerInstallChoice?.id
        result.installedResultID = engine.beginnerInstalledResultID
        result.installPhase = installationPhase
        result.downloadFraction = engine.downloadProgress.totalBytes == nil ? nil : engine.downloadProgress.fraction
        result.preparationFraction = engine.conversionProgress.totalBytes == nil ? nil : engine.conversionProgress.fraction
        result.installMessage = engine.beginnerInstallState == "installed" ? "Download and preparation complete." : engine.message
        result.installError = engine.beginnerInstallError
        result.canInstall = !engine.busy && !engine.beginnerChoicesLoading
        result.canStartModel = !engine.busy
        result.runningModelID = engine.loadedID; result.runningModelName = engine.activeModel?.name ?? ""
        result.modelReady = engine.modelReady && engine.loadedMode == .chat
        result.modelStarting = engine.state == "loading"
        result.modelGenerating = engine.state == "generating"
        result.localAPIAvailable = !engine.apiBaseURL.isEmpty
        result.runningSupportsTools = engine.supportsTools
        result.runningContextKnown = engine.contextCapacityKnown
        result.engineBusy = engine.busy; result.engineMessage = engine.message; result.engineError = engine.errorMessage
        result.openWorkInstalled = observation.applicationURL != nil; result.openWorkRunning = observation.running
        result.openWorkApproved = openWork.approved; result.openWorkClientActive = observation.liveConnection
        result.openWorkBusy = openWork.busy
        result.openWorkMessage = engine.apiBaseURL.isEmpty ? engine.apiMessage : (observation.message ?? openWork.message)
        result.openWorkScope = observation.scope
        let choice = session.choice(in: result)
        if let choice, session.installedID(for: choice, in: result) == engine.loadedID {
            result.choiceRequiresRestart = BeginnerContextPolicy.requiresRestart(expected: choice.contextTokens,
                local: engine.options.contextTokens, api: engine.apiContextTokens)
        }
        return result
    }

    private var installationPhase: BeginnerInstallPhase {
        if engine.beginnerCancelPending { return .cancelling }
        switch engine.beginnerInstallState {
        case "installed": return .ready
        case "cancelled": return .cancelled
        case "failed": return .failed
        case "installing":
            if engine.conversionProgress.state == "running" { return .preparing }
            if engine.downloadProgress.state == "running" { return .downloading }
            return .checking
        default: return .idle
        }
    }

    private var actions: BeginnerSetupActions {
        BeginnerSetupActions(
            refreshChoices: { task, excluded in engine.requestBeginnerModels(task: task.rawValue, excludingModelID: excluded) },
            install: { engine.installBeginnerModel(recipeID: $0.id) },
            cancelInstall: engine.cancelBeginnerInstallation,
            startModel: { choice in
                guard let id = choice.installedModelID else { return false }
                return engine.startBeginnerModel(recipeID: choice.id, modelID: id)
            },
            openOpenWork: openOpenWork,
            downloadOpenWork: { NSWorkspace.shared.open(URL(string: "https://openworklabs.com/download")!) },
            connectOpenWork: { observation.message = nil; openWork.connect() },
            checkConnection: {
                observation.refresh(engine: engine, force: true)
                observation.message = observation.liveConnection
                    ? "OpenWork’s verified connector has requested this model’s settings. Confirm the workspace and model you selected there."
                    : "Open OpenWork with a local workspace, then select Stark and the running model. Quit and reopen OpenWork if its connection settings changed."
            },
            openChat: openChat,
            showAdvanced: { selectInterface(.advanced) },
            retryLocalAPI: { observation.message = nil; engine.retryAPI() })
    }

    private func refreshOpenWork() {
        observation.refresh(engine: engine)
    }

    private func openOpenWork() {
        observation.refresh(engine: engine, force: true)
        observation.message = nil
        openWork.openApplication()
    }
}

struct BeginnerChatView: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var interface: StarkInterfacePreferences
    let selectInterface: (StarkInterfaceMode) -> Void
    let setup: () -> Void
    let tryAnotherModel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Stark").font(.title3.weight(.semibold))
                Spacer()
                StarkInterfacePicker(preferences: interface, selected: selectInterface)
            }
            if engine.loadedMode == .chat, !engine.loadedID.isEmpty {
                RunningModelLabel(engine: engine).font(.callout.weight(.medium))
                if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red) }
                ChatContentView(engine: engine)
                HStack {
                    Button("Guided setup", action: setup).buttonStyle(.link)
                    Spacer()
                    Button("Try a different model", action: tryAnotherModel).disabled(!engine.modelReady)
                }.font(.caption)
            } else {
                Text("Start a model with guided setup, then come back for your first chat.").foregroundStyle(.secondary)
                Button("Continue setup", action: setup).buttonStyle(.borderedProminent)
                Spacer()
            }
        }.padding(20).frame(minWidth: 520, minHeight: 600)
    }
}

struct AdvancedModelsHost: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var navigation: ModelManagementState
    @ObservedObject var interface: StarkInterfacePreferences
    let selectInterface: (StarkInterfaceMode) -> Void
    let close: () -> Void
    let run: (String) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack { Spacer(); StarkInterfacePicker(preferences: interface, selected: selectInterface) }
                .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            ModelsView(engine: engine, navigation: navigation, onDone: close, onRun: run)
        }
    }
}

struct AdvancedPlaygroundHost: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var interface: StarkInterfacePreferences
    let selectInterface: (StarkInterfaceMode) -> Void
    let connection: () -> Void
    let manage: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack { Spacer(); StarkInterfacePicker(preferences: interface, selected: selectInterface) }
                .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            PlaygroundView(engine: engine, connection: connection, manage: manage)
        }
    }
}
