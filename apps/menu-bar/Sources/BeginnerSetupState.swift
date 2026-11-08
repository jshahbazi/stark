import Combine
import Foundation

enum BeginnerTask: String, CaseIterable, Identifiable {
    case everyday, text, files, code, unsure

    var id: String { rawValue }
    var title: String {
        switch self {
        case .everyday: return "Writing and everyday help"
        case .text: return "Explain text"
        case .files: return "Work with files"
        case .code: return "Help with code"
        case .unsure: return "I’m not sure"
        }
    }
    var detail: String {
        switch self {
        case .everyday: return "Emails, ideas, and questions"
        case .text: return "Summaries, notes, and explanations"
        case .files: return "Read or edit a folder in OpenWork"
        case .code: return "Understand an error or a small project"
        case .unsure: return "Choose a starting point for me"
        }
    }
    var symbol: String {
        switch self {
        case .everyday: return "pencil.line"
        case .text: return "doc.text"
        case .files: return "folder"
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .unsure: return "questionmark.circle"
        }
    }
}

enum BeginnerRetryPreference: String, CaseIterable, Identifiable {
    case speed, answers, task, unsure

    var id: String { rawValue }
    var title: String {
        switch self {
        case .speed: return "Quicker responses"
        case .answers: return "Different answers"
        case .task: return "A different task"
        case .unsure: return "Help me choose again"
        }
    }
    var detail: String {
        switch self {
        case .speed: return "Look at another available starting model"
        case .answers: return "Compare a different model on my task"
        case .task: return "Change what I need help with"
        case .unsure: return "Review the other starting options"
        }
    }
}

enum BeginnerModelPriority: String, CaseIterable, Identifiable {
    case balanced, speed, answers

    var id: String { rawValue }
    var title: String {
        switch self {
        case .balanced: return "Balanced"
        case .speed: return "Lighter model"
        case .answers: return "More model capacity"
        }
    }
}

enum BeginnerSetupStage: String, CaseIterable {
    case task, recommendation, installation, openWork, connection, ready, retry

    var step: Int {
        switch self {
        case .task, .recommendation, .retry: return 0
        case .installation: return 1
        case .openWork: return 2
        case .connection: return 3
        case .ready: return 4
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

    func installed(as modelID: String) -> BeginnerModelChoice {
        var result = self
        result.installedModelID = modelID
        return result
    }
}

struct BeginnerSetupSnapshot: Equatable {
    var choices: [BeginnerModelChoice] = []
    var choicesLoading = false
    var choicesError: String? = nil
    var deviceSummary = ""
    var recommendationMessage = ""
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
    var openWorkClientActive = false
    var openWorkBusy = false
    var openWorkMessage = ""
    // An adapter-owned identity for the connection/profile. Revocation or a
    // different client must change this value to invalidate local confirmations.
    var openWorkScope = ""

    var canUseOpenWork: Bool {
        modelReady && !runningModelID.isEmpty && runningSupportsTools && runningContextKnown && localAPIAvailable
    }
}

struct BeginnerSetupActions {
    var refreshChoices: (BeginnerTask, String?, String) -> Void = { _, _, _ in }
    var install: (BeginnerModelChoice) -> Void = { _ in }
    var cancelInstall: () -> Void = {}
    var startModel: (BeginnerModelChoice) -> Bool = { _ in false }
    var openOpenWork: () -> Void = {}
    var downloadOpenWork: () -> Void = {}
    var connectOpenWork: () -> Void = {}
    var checkConnection: () -> Void = {}
    var openChat: () -> Void = {}
    var showAdvanced: () -> Void = {}
    var retryLocalAPI: () -> Void = {}
}

@MainActor
final class BeginnerSetupSession: ObservableObject {
    @Published var stage: BeginnerSetupStage = .task
    @Published var task: BeginnerTask?
    @Published var modelPriority: BeginnerModelPriority = .balanced
    @Published var selectedChoiceID: String?
    @Published var retryPreference: BeginnerRetryPreference?
    @Published private(set) var choosingAnotherModel = false
    @Published private(set) var excludedModelID: String?
    @Published private(set) var workspaceConfirmed = false
    @Published private(set) var confirmedOpenWorkModelID: String?
    @Published private(set) var startRequestedModelID: String?
    @Published private(set) var installingChoiceID: String?

    private var previousTask: BeginnerTask?
    private var previousStage: BeginnerSetupStage = .ready
    private var observedModelID: String?
    private var connectionScope: String?

    func choices(in snapshot: BeginnerSetupSnapshot) -> [BeginnerModelChoice] {
        snapshot.choices.filter { choice in
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
        previousStage = stage
        task = task ?? .unsure
        choosingAnotherModel = true
        excludedModelID = snapshot.runningModelID
        selectedChoiceID = nil
        retryPreference = nil
        startRequestedModelID = nil
        stage = .retry
    }

    func keepCurrentModel() {
        choosingAnotherModel = false
        task = previousTask ?? task
        selectedChoiceID = nil
        excludedModelID = nil
        retryPreference = nil
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
        retryPreference = nil
        selectedChoiceID = nil
        startRequestedModelID = nil
        installingChoiceID = nil
        task = task ?? .unsure
        stage = .task
    }

    func requestInstall(choiceID: String) {
        selectedChoiceID = choiceID
        installingChoiceID = choiceID
        stage = .installation
    }

    func confirmWorkspace() { workspaceConfirmed = true }

    func confirmModelSelection(snapshot: BeginnerSetupSnapshot) {
        guard snapshot.canUseOpenWork, snapshot.openWorkApproved,
              snapshot.openWorkRunning else { return }
        confirmedOpenWorkModelID = snapshot.runningModelID
    }

    func canFinish(snapshot: BeginnerSetupSnapshot) -> Bool {
        snapshot.canUseOpenWork && snapshot.openWorkRunning && snapshot.openWorkApproved &&
            snapshot.openWorkClientActive && workspaceConfirmed &&
            confirmedOpenWorkModelID == snapshot.runningModelID
    }

    // A request temporarily makes the worker unavailable for a model switch;
    // it does not undo a completed setup for the same verified connection.
    func keepsCompletedSetupWhileGenerating(snapshot: BeginnerSetupSnapshot) -> Bool {
        snapshot.modelGenerating && !snapshot.runningModelID.isEmpty &&
            snapshot.runningSupportsTools && snapshot.runningContextKnown && snapshot.localAPIAvailable &&
            snapshot.openWorkRunning && snapshot.openWorkApproved && snapshot.openWorkClientActive &&
            workspaceConfirmed && confirmedOpenWorkModelID == snapshot.runningModelID
    }

    func finish(snapshot: BeginnerSetupSnapshot) {
        guard canFinish(snapshot: snapshot) else { return }
        choosingAnotherModel = false
        excludedModelID = nil
        retryPreference = nil
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
        if target == .recommendation || target == .retry { return task != nil }
        if target == .installation {
            return snapshot.installChoiceID != nil || snapshot.installedResultID != nil
                || selectedChoiceID != nil || canFinish(snapshot: snapshot)
        }
        if target == .openWork || target == .connection {
            return (snapshot.modelReady || keepsCompletedSetupWhileGenerating(snapshot: snapshot))
                && !snapshot.runningModelID.isEmpty
        }
        return false
    }

    func navigate(to target: BeginnerSetupStage, snapshot: BeginnerSetupSnapshot) {
        guard canNavigate(to: target, snapshot: snapshot) else { return }
        stage = target
    }

    func continueAfterModel(snapshot: BeginnerSetupSnapshot) {
        guard snapshot.modelReady, !snapshot.runningModelID.isEmpty else { return }
        startRequestedModelID = nil
        choosingAnotherModel = false
        excludedModelID = nil
        retryPreference = nil
        stage = snapshot.openWorkInstalled && workspaceConfirmed ? .connection : .openWork
    }

    func synchronize(with snapshot: BeginnerSetupSnapshot) {
        if let scope = connectionScope, scope != snapshot.openWorkScope {
            workspaceConfirmed = false
            confirmedOpenWorkModelID = nil
            if stage == .ready { stage = .openWork }
        }
        connectionScope = snapshot.openWorkScope
        if let previous = observedModelID, previous != snapshot.runningModelID {
            confirmedOpenWorkModelID = nil
            if stage == .ready { stage = .connection }
        }
        observedModelID = snapshot.runningModelID
        if !snapshot.openWorkApproved { confirmedOpenWorkModelID = nil }
        if stage == .ready, !canFinish(snapshot: snapshot), !keepsCompletedSetupWhileGenerating(snapshot: snapshot) {
            stage = snapshot.openWorkInstalled && workspaceConfirmed ? .connection : .openWork
        }
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
