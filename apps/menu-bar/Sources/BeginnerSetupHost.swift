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
    private var refreshID = UUID()
    private var refreshInFlight = false
    private var pendingRefresh: (input: ScanInput, completions: [(Bool) -> Void])?

    private struct ScanInput: Sendable {
        var previousURL: URL?
        var shouldResolveApplication: Bool
        var profile: URL?
        var candidates: [DesktopApplicationIdentity.Candidate]
        var hasDiscoveredModel: Bool
        var discoveryScope: String
        var brokerPath: String
        var apiBaseURL: String
    }

    private struct ScanResult: Sendable {
        var applicationURL: URL?
        var applicationVerified: Bool
        var running: Bool
        var liveConnection: Bool
        var verifiedClientBasis: String
        var connectionBasis: String
    }

    func refresh(engine: EngineController, force: Bool = false, completion: ((Bool) -> Void)? = nil) {
        if refreshInFlight && !force { return }
        let connection = engine.desktopConnections.openWork
        if wasApproved && !connection.approved { scope = UUID().uuidString }
        wasApproved = connection.approved
        let previousURL = applicationURL
        let shouldResolveApplication = force || !applicationChecked || previousURL.map { !FileManager.default.fileExists(atPath: $0.path) } == true
        let apiID = engine.activeModel?.apiID ?? ""
        let contextTokens = engine.apiContextTokens
        let input = ScanInput(previousURL: previousURL, shouldResolveApplication: shouldResolveApplication,
            profile: DesktopConnectionTesting.profile,
            candidates: DesktopApplicationIdentity.candidates(.openWork),
            hasDiscoveredModel: connection.hasDiscoveredModel(apiID: apiID, contextTokens: contextTokens),
            discoveryScope: connection.discoveryScope, brokerPath: engine.desktopConnections.brokerPath ?? "",
            apiBaseURL: engine.apiBaseURL)

        if refreshInFlight {
            if var pending = pendingRefresh {
                pending.input = input
                if let completion { pending.completions.append(completion) }
                pendingRefresh = pending
            } else {
                pendingRefresh = (input, completion.map { [$0] } ?? [])
            }
            return
        }
        startScan(input, completions: completion.map { [$0] } ?? [])
    }

    private func startScan(_ input: ScanInput, completions: [(Bool) -> Void]) {
        let requestID = UUID()
        refreshID = requestID
        refreshInFlight = true

        Task.detached(priority: .utility) { [weak self] in
            let url: URL?
            if input.shouldResolveApplication {
                let candidate = DesktopConnectionTesting.application(.openWork)
                    ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: DesktopClient.openWork.bundleID)
                url = candidate.flatMap { DesktopClientPeer.verifyApplication($0, client: .openWork) ? $0 : nil }
            } else {
                url = input.previousURL
            }
            let application = DesktopApplicationIdentity.running(.openWork,
                configurationRoot: input.profile, candidates: input.candidates)
            let running = application != nil
            let verified = application.map { $0.process.instanceID + "|" + ($0.profile?.path ?? "") } ?? ""
            let result = ScanResult(
                applicationURL: application?.bundleURL ?? url,
                applicationVerified: url != nil || application != nil,
                running: running,
                liveConnection: input.hasDiscoveredModel && running,
                verifiedClientBasis: verified,
                connectionBasis: (application?.bundleURL.path ?? url?.path ?? "") + "|" + verified + "|" +
                    input.discoveryScope + "|" + input.brokerPath + "|" + input.apiBaseURL
            )
            await self?.apply(result, requestID: requestID, completions: completions)
        }
    }

    private func apply(_ result: ScanResult, requestID: UUID, completions: [(Bool) -> Void]) {
        guard refreshID == requestID else { return }
        if let pending = pendingRefresh {
            pendingRefresh = nil
            refreshInFlight = false
            startScan(pending.input, completions: completions + pending.completions)
            return
        }
        refreshInFlight = false
        applicationURL = result.applicationURL
        applicationChecked = result.applicationVerified
        running = result.running
        liveConnection = result.liveConnection
        verifiedClientBasis = result.verifiedClientBasis
        if !connectionBasis.isEmpty, connectionBasis != result.connectionBasis {
            scope = UUID().uuidString
        }
        connectionBasis = result.connectionBasis
        for completion in completions { completion(result.liveConnection) }
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
            ZStack {
                StarkInterfacePicker(preferences: interface, selected: selectInterface)
                HStack { Text("Stark").font(.headline); Spacer() }
            }.padding(.horizontal, 20).padding(.vertical, 14)
            Divider()
            BeginnerSetupView(session: session, snapshot: snapshot, actions: actions)
        }
        .onAppear { refreshOpenWork() }
        .onReceive(refresh) { _ in refreshOpenWork() }
        .onChange(of: openWork.approved) { _, _ in refreshOpenWork(force: true) }
        .onChange(of: engine.apiBaseURL) { _, _ in refreshOpenWork(force: true) }
        .onChange(of: engine.activeModel?.apiID) { _, _ in refreshOpenWork(force: true) }
        .onChange(of: engine.apiContextTokens) { _, _ in refreshOpenWork(force: true) }
    }

    var snapshot: BeginnerSetupSnapshot {
        var result = BeginnerSetupSnapshot()
        result.choices = (engine.beginnerReport?.items ?? []).map { item in
            BeginnerModelChoice(id: item.id, name: item.name,
                summary: item.recommendationReason,
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
        result.recommendationMessage = engine.beginnerReport?.message ?? ""
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
        result.openWorkApproved = openWork.approved
        // Query against this snapshot's model and budget. The application scan
        // is periodic; its cached connection result may belong to a prior model.
        result.openWorkClientActive = observation.running && openWork.hasDiscoveredModel(
            apiID: engine.activeModel?.apiID ?? "", contextTokens: engine.apiContextTokens)
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
            refreshChoices: { task, excluded, preference in
                engine.requestBeginnerModels(task: task.rawValue, excludingModelID: excluded, preference: preference)
            },
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
                observation.message = "Checking OpenWork’s connection…"
                observation.refresh(engine: engine, force: true) { connected in
                    observation.message = connected
                        ? "OpenWork’s verified connector has requested this model’s settings. Confirm the workspace and model you selected there."
                        : "Open OpenWork with a local workspace, then select Stark and the running model. Quit and reopen OpenWork if its connection settings changed."
                }
            },
            openChat: openChat,
            showAdvanced: { selectInterface(.advanced) },
            retryLocalAPI: { observation.message = nil; engine.retryAPI() })
    }

    private func refreshOpenWork(force: Bool = false) {
        observation.refresh(engine: engine, force: force)
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
            ZStack {
                StarkInterfacePicker(preferences: interface, selected: selectInterface)
                HStack { Text("Stark").font(.title3.weight(.semibold)); Spacer() }
                    .allowsHitTesting(false)
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
            HStack { Spacer(); StarkInterfacePicker(preferences: interface, selected: selectInterface); Spacer() }
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
            HStack { Spacer(); StarkInterfacePicker(preferences: interface, selected: selectInterface); Spacer() }
                .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            PlaygroundView(engine: engine, connection: connection, manage: manage)
        }
    }
}
