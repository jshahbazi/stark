import Combine
import Foundation

enum BeginnerClient: String, CaseIterable, Identifiable {
    case openWork, openCode

    var id: String { rawValue }
    var name: String { self == .openWork ? "OpenWork" : "OpenCode" }
    var detail: String {
        self == .openWork ? "A workspace for everyday tasks and files" : "A workspace for coding and projects"
    }
}

enum BeginnerTask: String, CaseIterable, Identifiable {
    case general, coding

    var id: String { rawValue }
    var title: String { self == .general ? "General Work" : "Coding" }
    var detail: String {
        self == .general ? "Writing, questions, and files with OpenWork" : "Code and projects with OpenCode"
    }
    var symbol: String { self == .general ? "folder" : "chevron.left.forwardslash.chevron.right" }
    var client: BeginnerClient { self == .general ? .openWork : .openCode }
}

enum BeginnerModelPriority: String, CaseIterable, Identifiable {
    case speed, quality

    var id: String { rawValue }
    var title: String { self == .speed ? "Speed" : "Quality" }
    var detail: String {
        self == .speed ? "Balance model quality with a speed budget for this Mac" : "Favor answer quality within this Mac’s limits"
    }
}

enum BeginnerSetupStage: String, CaseIterable {
    case task, priority, recommendation, installation, openWork, connection, openOpenWork, ready, retry

    var step: Int {
        switch self {
        case .task, .retry: return 0
        case .priority, .recommendation: return 1
        case .installation: return 2
        case .openWork: return 3
        case .connection: return 4
        case .openOpenWork: return 5
        case .ready: return 6
        }
    }
}

enum BeginnerInstallPhase: String {
    case idle, checking, downloading, preparing, ready, cancelling, cancelled, failed

    var inProgress: Bool { [.checking, .downloading, .preparing, .cancelling].contains(self) }
}

struct BeginnerModelChoice: Identifiable, Equatable {
    var id: String
    var name: String
    var summary: String
    var downloadBytes: Int64? = nil
    var installedBytes: Int64? = nil
    var setupBytes: Int64? = nil
    var contextTokens: Int = 0
    var installedModelID: String? = nil
    var source = ""
    var revision = ""
    var qualification = ""
    var benchmarkSourceURL: URL? = nil
    var availabilityMessage: String? = nil
    var variantName = ""

    var repositoryURL: URL? {
        guard source.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$",
                           options: .regularExpression) != nil else { return nil }
        return URL(string: "https://huggingface.co/" + source)
    }

    func installed(as modelID: String) -> BeginnerModelChoice {
        var result = self
        result.installedModelID = modelID
        result.availabilityMessage = nil
        return result
    }

    func resolvingInstalledCopy(as modelID: String?) -> BeginnerModelChoice {
        var result = self
        result.installedModelID = modelID
        if downloadBytes == 0 && modelID == nil {
            result.availabilityMessage = "This installed model is no longer available. Check models again to refresh your choices."
        }
        return result
    }
}

struct BeginnerSetupSnapshot: Equatable {
    // The connection fields describe only the selected guide's client.
    var client: BeginnerClient = .openWork
    var choices: [BeginnerModelChoice] = []
    var choicesLoading = false
    var choicesError: String? = nil
    var memoryAdvice: String? = nil
    var deviceSummary = ""
    var installChoiceID: String? = nil
    var installedResultID: String? = nil
    var installPhase: BeginnerInstallPhase = .idle
    var downloadFraction: Double? = nil
    var preparationFraction: Double? = nil
    var installMessage = ""
    var installError: String? = nil
    var canInstall = false
    var canStartModel = false
    var choiceRequiresRestart = false
    var runningModelID = ""
    var runningModelName = ""
    var runningModelVariantName = ""
    var runningModelRepositoryURL: URL? = nil
    var modelReady = false
    var modelStarting = false
    var modelGenerating = false
    var runningSupportsTools = false
    var runningContextKnown = false
    var localAPIAvailable = true
    var engineBusy = false
    var engineMessage = ""
    var engineError: String? = nil
    var openWorkInstalled = false
    var openWorkRunning = false
    var openWorkApproved = false
    var openCodeApprovalPending = false
    var openWorkClientActive = false
    var openWorkPreparedModelID: String? = nil
    var openWorkPreparing = false
    var openWorkBusy = false
    var openWorkMessage = ""
    // An adapter-owned identity for the connection/profile. Revocation or a
    // different client must change this value to invalidate the automatic handoff.
    var openWorkScope = ""

    var canUseOpenWork: Bool {
        modelReady && !runningModelID.isEmpty && runningSupportsTools && runningContextKnown && localAPIAvailable
    }
}

struct BeginnerSetupActions {
    var refreshChoices: (BeginnerTask, String?, String) -> Void = { _, _, _ in }
    var install: (BeginnerModelChoice) -> Bool = { _ in false }
    var cancelInstall: () -> Void = {}
    var startModel: (BeginnerModelChoice) -> Bool = { _ in false }
    var openOpenWork: () -> Void = {}
    var downloadOpenWork: () -> Void = {}
    var connectOpenWork: () -> Void = {}
    var allowOpenCodeEngine: () -> Void = {}
    var denyOpenCodeEngine: () -> Void = {}
    var checkConnection: () -> Void = {}
    var openChat: () -> Void = {}
    var showAdvanced: () -> Void = {}
    var retryLocalAPI: () -> Void = {}
}

@MainActor
final class BeginnerSetupSession: ObservableObject {
    @Published var stage: BeginnerSetupStage = .task
    var client: BeginnerClient { task?.client ?? .openWork }
    @Published private(set) var confirmedOpenCodeScope: String?
    @Published private(set) var task: BeginnerTask?
    @Published var modelPriority: BeginnerModelPriority = .quality
    @Published var selectedChoiceID: String?
    @Published private(set) var choosingAnotherModel = false
    @Published private(set) var excludedModelID: String?
    @Published private(set) var startRequestedModelID: String?
    @Published private(set) var installingChoiceID: String?

    private var previousTask: BeginnerTask?
    private var previousPriority: BeginnerModelPriority = .quality
    private var previousStage: BeginnerSetupStage = .ready
    private var observedModelID: String?
    private var connectionScope: String?
    private var recommendationTask: BeginnerTask?
    private var recommendationPriority: BeginnerModelPriority?

    private func confirmationBasis(_ snapshot: BeginnerSetupSnapshot) -> String {
        [snapshot.openWorkScope, snapshot.runningModelID].joined(separator: "|")
    }

    func confirmOpenCodeModel(snapshot: BeginnerSetupSnapshot) {
        guard client == .openCode, snapshot.client == client, snapshot.canUseOpenWork,
              snapshot.openWorkRunning, snapshot.openWorkApproved, snapshot.openWorkClientActive else { return }
        confirmedOpenCodeScope = confirmationBasis(snapshot)
        finish(snapshot: snapshot)
    }

    private func modelSelected(snapshot: BeginnerSetupSnapshot) -> Bool {
        guard snapshot.client == client else { return false }
        return client == .openWork ? snapshot.openWorkPreparedModelID == snapshot.runningModelID
            : confirmedOpenCodeScope == confirmationBasis(snapshot)
    }

    var recommendationsMatchSelection: Bool {
        recommendationTask == nil || (recommendationTask == task && recommendationPriority == modelPriority)
    }

    func selectTask(_ value: BeginnerTask) {
        guard task != value else { return }
        if client != value.client {
            confirmedOpenCodeScope = nil
            connectionScope = nil
        }
        task = value
        selectedChoiceID = nil
    }

    func selectModelPriority(_ value: BeginnerModelPriority) {
        modelPriority = value
        selectedChoiceID = nil
    }

    func requestRecommendations() -> String {
        recommendationTask = task
        recommendationPriority = modelPriority
        selectedChoiceID = nil
        stage = .recommendation
        return modelPriority.rawValue
    }

    func choices(in snapshot: BeginnerSetupSnapshot) -> [BeginnerModelChoice] {
        guard recommendationsMatchSelection else { return [] }
        return snapshot.choices.filter { choice in
            !choosingAnotherModel || excludedModelID == nil || choice.installedModelID != excludedModelID
        }
    }

    func choice(in snapshot: BeginnerSetupSnapshot) -> BeginnerModelChoice? {
        let choices = choices(in: snapshot)
        return choices.first { $0.id == selectedChoiceID } ?? choices.first
    }

    func installedID(for choice: BeginnerModelChoice, in snapshot: BeginnerSetupSnapshot) -> String? {
        if let identifier = choice.installedModelID, !identifier.isEmpty { return identifier }
        guard snapshot.installChoiceID == choice.id, snapshot.installPhase == .ready,
              let identifier = snapshot.installedResultID, !identifier.isEmpty else { return nil }
        return identifier
    }

    func beginAnotherModel(snapshot: BeginnerSetupSnapshot) {
        guard snapshot.modelReady, !snapshot.runningModelID.isEmpty else { return }
        if choosingAnotherModel {
            stage = snapshot.installPhase.inProgress ? .installation : .retry
            return
        }
        previousTask = task
        previousPriority = modelPriority
        previousStage = stage
        task = task ?? (snapshot.client == .openCode ? .coding : .general)
        choosingAnotherModel = true
        excludedModelID = snapshot.runningModelID
        selectedChoiceID = nil
        startRequestedModelID = nil
        stage = .retry
    }

    func keepCurrentModel() {
        choosingAnotherModel = false
        task = previousTask ?? task
        modelPriority = previousPriority
        selectedChoiceID = nil
        excludedModelID = nil
        startRequestedModelID = nil
        stage = previousStage == .retry ? .ready : previousStage
    }

    func requestStart(modelID: String) {
        startRequestedModelID = modelID
        stage = .installation
    }

    /// Reopen selection after a failed switch without claiming that a stopped
    /// worker was restored. The next request may reuse a checked installed copy.
    func chooseAgain() {
        choosingAnotherModel = false
        excludedModelID = nil
        selectedChoiceID = nil
        startRequestedModelID = nil
        installingChoiceID = nil
        task = task ?? .general
        stage = .task
    }

    func requestInstall(choiceID: String) {
        selectedChoiceID = choiceID
        installingChoiceID = choiceID
        stage = .installation
    }

    func canFinish(snapshot: BeginnerSetupSnapshot) -> Bool {
        task != nil && snapshot.canUseOpenWork && snapshot.openWorkRunning && snapshot.openWorkApproved &&
            snapshot.openWorkClientActive && modelSelected(snapshot: snapshot)
    }

    // A request temporarily makes the worker unavailable for a model switch;
    // it does not undo a completed setup for the same verified connection.
    func keepsCompletedSetupWhileGenerating(snapshot: BeginnerSetupSnapshot) -> Bool {
        task != nil && snapshot.modelGenerating && !snapshot.runningModelID.isEmpty &&
            snapshot.runningSupportsTools && snapshot.runningContextKnown && snapshot.localAPIAvailable &&
            snapshot.openWorkRunning && snapshot.openWorkApproved && snapshot.openWorkClientActive &&
            modelSelected(snapshot: snapshot)
    }

    func finish(snapshot: BeginnerSetupSnapshot) {
        guard canFinish(snapshot: snapshot) else { return }
        choosingAnotherModel = false
        excludedModelID = nil
        startRequestedModelID = nil
        stage = .ready
    }

    func canNavigate(to target: BeginnerSetupStage, snapshot: BeginnerSetupSnapshot) -> Bool {
        guard !snapshot.installPhase.inProgress, installingChoiceID == nil,
              startRequestedModelID == nil, !snapshot.modelStarting else { return false }
        if target == .ready {
            return canFinish(snapshot: snapshot) || keepsCompletedSetupWhileGenerating(snapshot: snapshot)
        }
        if target == .task { return true }
        if target == .priority { return task != nil }
        if target == .recommendation || target == .retry { return task != nil }
        if target == .installation {
            guard recommendationsMatchSelection else { return false }
            return snapshot.installChoiceID != nil || snapshot.installedResultID != nil
                || selectedChoiceID != nil || canFinish(snapshot: snapshot)
        }
        if target == .openWork || target == .connection || target == .openOpenWork {
            guard task != nil, snapshot.client == client else { return false }
            if target != .openWork && !snapshot.openWorkInstalled { return false }
            if target == .openOpenWork && (!snapshot.openWorkApproved || snapshot.openWorkBusy) { return false }
            return (snapshot.modelReady || keepsCompletedSetupWhileGenerating(snapshot: snapshot))
                && !snapshot.runningModelID.isEmpty
        }
        return false
    }

    func navigate(to target: BeginnerSetupStage, snapshot: BeginnerSetupSnapshot) {
        guard canNavigate(to: target, snapshot: snapshot) else { return }
        stage = target
        synchronize(with: snapshot)
    }

    func continueAfterModel(snapshot: BeginnerSetupSnapshot) {
        guard task != nil, snapshot.client == client, snapshot.modelReady, !snapshot.runningModelID.isEmpty else { return }
        startRequestedModelID = nil
        choosingAnotherModel = false
        excludedModelID = nil
        stage = snapshot.openWorkInstalled ? .connection : .openWork
        synchronize(with: snapshot)
    }

    func synchronize(with snapshot: BeginnerSetupSnapshot) {
        if client == .openCode,
           !snapshot.openWorkRunning || !snapshot.openWorkApproved || !snapshot.openWorkClientActive {
            confirmedOpenCodeScope = nil
        }
        if let scope = connectionScope, scope != snapshot.openWorkScope {
            confirmedOpenCodeScope = nil
            if stage == .ready { stage = .openWork }
        }
        connectionScope = snapshot.openWorkScope
        if let previous = observedModelID, previous != snapshot.runningModelID {
            confirmedOpenCodeScope = nil
            if stage == .ready { stage = .connection }
        }
        observedModelID = snapshot.runningModelID
        if stage == .ready, !canFinish(snapshot: snapshot), !keepsCompletedSetupWhileGenerating(snapshot: snapshot) {
            stage = snapshot.openWorkInstalled ? .connection : .openWork
        }
        // Authorization and discovery do not prove that the automatic model
        // binding was saved and displayed in OpenWork.
        if stage == .openWork, snapshot.openWorkInstalled { stage = .connection }
        if stage == .connection, snapshot.openWorkInstalled, snapshot.openWorkApproved,
           !snapshot.openWorkBusy, snapshot.canUseOpenWork { stage = .openOpenWork }
        if stage == .connection || stage == .openOpenWork {
            if !snapshot.openWorkInstalled { stage = .openWork }
            else if stage == .openOpenWork,
                    !snapshot.openWorkApproved || (!snapshot.canUseOpenWork && !keepsCompletedSetupWhileGenerating(snapshot: snapshot)) {
                stage = .connection
            }
        }
        if stage == .openOpenWork, canFinish(snapshot: snapshot) { finish(snapshot: snapshot) }
        if let pending = installingChoiceID, pending == snapshot.installChoiceID,
           [.ready, .cancelled, .failed].contains(snapshot.installPhase) {
            installingChoiceID = nil
        }
        if startRequestedModelID != nil, !snapshot.modelStarting, snapshot.engineError != nil {
            startRequestedModelID = nil
        }
        if let target = startRequestedModelID, snapshot.modelReady, snapshot.runningModelID == target {
            continueAfterModel(snapshot: snapshot)
        }
    }
}
