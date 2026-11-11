import SwiftUI

struct BeginnerSetupView: View {
    @ObservedObject var session: BeginnerSetupSession
    var snapshot: BeginnerSetupSnapshot
    var actions: BeginnerSetupActions
    @State private var showAlternatives = false

    private let steps = ["Choose a model", "Set up model", "Get OpenWork", "Connect", "Open OpenWork", "Ready"]
    private var choice: BeginnerModelChoice? { session.choice(in: snapshot) }
    private var ownInstallation: Bool {
        guard let identifier = snapshot.installChoiceID else { return false }
        return choice?.id == identifier || session.installingChoiceID == identifier
    }
    private var installInProgress: Bool { ownInstallation && snapshot.installPhase.inProgress }

    var body: some View {
        HStack(spacing: 0) {
            sidebar.frame(width: 180)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    content
                    if let error = snapshot.engineError, session.stage != .installation {
                        notice(error, symbol: "exclamationmark.triangle", color: .red)
                    }
                }
                .padding(28)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(minWidth: 700, minHeight: 600)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear { session.synchronize(with: snapshot) }
        .onChange(of: snapshot) { _, value in session.synchronize(with: value) }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Set up your assistant").font(.headline)
            VStack(alignment: .leading, spacing: 5) {
                ForEach(Array(steps.enumerated()), id: \.offset) { index, title in
                    let target = navigationTarget(for: index)
                    let enabled = session.canNavigate(to: target, snapshot: snapshot)
                    Button {
                        session.navigate(to: target, snapshot: snapshot)
                    } label: {
                        HStack(spacing: 9) {
                            Text(String(index + 1))
                                .font(.caption.monospacedDigit())
                                .frame(width: 22, height: 22)
                                .background(session.stage.step == index ? Color.primary : Color.secondary.opacity(0.1), in: Circle())
                                .foregroundStyle(session.stage.step == index ? Color(nsColor: .windowBackgroundColor) : .secondary)
                            Text(title).font(.callout)
                            Spacer(minLength: 0)
                        }
                        .padding(.vertical, 9).padding(.horizontal, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(session.stage.step == index ? Color.primary.opacity(0.06) : .clear, in: RoundedRectangle(cornerRadius: 7))
                        .foregroundStyle(session.stage.step == index ? .primary : .secondary)
                    }
                    .buttonStyle(.plain)
                    .disabled(!enabled)
                    .help(navigationHelp(for: target))
                    .accessibilityAddTraits(session.stage.step == index ? .isSelected : [])
                }
            }
            Spacer(minLength: 20)
            VStack(alignment: .leading, spacing: 6) {
                Label(snapshot.modelGenerating ? "Model working" : snapshot.modelReady ? "Model running" : snapshot.modelStarting ? "Starting model" : !snapshot.runningModelID.isEmpty ? "Model loaded" : "No model running",
                      systemImage: snapshot.modelReady || snapshot.modelGenerating ? "checkmark.circle" : "circle")
                if !snapshot.runningModelName.isEmpty { Text(snapshot.runningModelName) }
                if !snapshot.runningModelVariantName.isEmpty, snapshot.runningModelVariantName != snapshot.runningModelName {
                    Text(snapshot.runningModelVariantName).font(.caption2)
                }
            }.font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxHeight: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
    }

    @ViewBuilder private var content: some View {
        switch session.stage {
        case .task: taskStep
        case .recommendation: recommendationStep
        case .installation: installationStep
        case .openWork: openWorkStep
        case .connection: connectionStep
        case .openOpenWork: openOpenWorkStep
        case .ready:
            if session.canFinish(snapshot: snapshot) || session.keepsCompletedSetupWhileGenerating(snapshot: snapshot) { readyStep }
            else { connectionStep }
        case .retry: retryStep
        }
    }

    private var taskStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            heading("What would you like help with?", detail: "Pick a starting point. Stark will choose the download settings for you.")
            ForEach(BeginnerTask.allCases) { task in
                selection(title: task.title, detail: task.detail, symbol: task.symbol, selected: session.task == task) {
                    session.selectTask(task)
                }
            }
            Picker("Model priority", selection: Binding(get: { session.modelPriority }, set: session.selectModelPriority)) {
                ForEach(BeginnerModelPriority.allCases) { priority in
                    Text(priority.title).tag(priority)
                }
            }.pickerStyle(.menu)
            Text("Stark estimates memory needs and model capacity. Speed and answer quality depend on the model and your task.")
                .font(.caption).foregroundStyle(.secondary)
            deviceSummary
            HStack {
                Button(session.choosingAnotherModel ? "See another model" : "Find a model") { requestChoices() }
                    .buttonStyle(.borderedProminent).disabled(session.task == nil || snapshot.choicesLoading)
                if session.choosingAnotherModel { keepCurrentButton }
            }
            if session.choosingAnotherModel {
                Text("Your current model stays installed while you choose. Your OpenWork setup is kept.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var recommendationStep: some View {
        if snapshot.choicesLoading {
            heading("Checking models for this Mac…", detail: "Reading memory and GPU limits, then checking supported model metadata. This can take about a minute.")
            ProgressView().controlSize(.small)
        } else if let choice {
            let installed = session.installedID(for: choice, in: snapshot)
            heading(choice.availabilityMessage != nil ? "Check this model again." : session.choosingAnotherModel ? "Try another starting model." : installed == nil ? "Start with this model." : "Use the model already on your Mac.",
                    detail: choice.availabilityMessage != nil ? "The saved model inventory changed." : installed == nil ? "You can change models later." : "No new download is needed.")
            modelCard(choice)
            deviceSummary
            if let unavailable = choice.availabilityMessage {
                notice(unavailable, symbol: "exclamationmark.triangle", color: .orange)
            }
            if snapshot.modelReady, snapshot.runningModelID != installed {
                notice("\(snapshot.runningModelName) stays installed while you prepare this model. You’ll choose when to switch.", symbol: "checkmark.circle")
            }
            HStack {
                if choice.availabilityMessage != nil {
                    Button("Check models again") { requestChoices() }.buttonStyle(.borderedProminent)
                } else if installed != nil {
                    Button(startLabel(choice)) { start(choice) }.buttonStyle(.borderedProminent)
                        .disabled(!snapshot.canStartModel || choice.availabilityMessage != nil)
                } else {
                    Button(downloadLabel(choice)) {
                        install(choice)
                    }.buttonStyle(.borderedProminent).disabled(!snapshot.canInstall || choice.availabilityMessage != nil)
                }
                if session.choosingAnotherModel { keepCurrentButton }
                else { Button("Change task") { session.stage = .task } }
            }
            if session.choices(in: snapshot).count > 1 {
                DisclosureGroup("See another option", isExpanded: $showAlternatives) {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(session.choices(in: snapshot).filter { $0.id != choice.id }) { alternative in
                            Button {
                                session.selectedChoiceID = alternative.id
                                showAlternatives = false
                            } label: {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(alternative.name)
                                    if !alternative.variantName.isEmpty, alternative.variantName != alternative.name {
                                        Text(alternative.variantName).font(.caption).foregroundStyle(.secondary)
                                    }
                                    Text(alternative.summary).font(.caption).foregroundStyle(.secondary)
                                }.frame(maxWidth: .infinity, alignment: .leading)
                            }.buttonStyle(.bordered).disabled(installInProgress)
                        }
                    }.padding(.top, 8)
                }.font(.caption)
            }
            Text("Download once. The model runs on this Mac. Downloads contact the model host; OpenWork tools may contact other services.")
                .font(.caption).foregroundStyle(.secondary)
        } else {
            heading("No suitable starting model is available.", detail: snapshot.choicesError ?? "Stark could not establish a memory plan for another supported model on this Mac. Your current setup is kept.")
            deviceSummary
            HStack {
                Button("Check again") { requestChoices() }.buttonStyle(.borderedProminent)
                Button("Browse models in Advanced", action: actions.showAdvanced)
                if session.choosingAnotherModel { keepCurrentButton }
                else { Button("Change task") { session.stage = .task } }
            }
        }
    }

    @ViewBuilder private var installationStep: some View {
        if let choice {
            let installed = session.installedID(for: choice, in: snapshot)
            let starting = session.startRequestedModelID != nil
            heading(starting ? "Starting your model…" : installed != nil ? "Your model is installed." : "Set up your model.",
                    detail: starting ? "Waiting for the selected model to be ready before moving on."
                        : installed != nil ? "Start it when you’re ready." : "Downloading and preparing are separate steps.")
            modelIdentity(choice)
            if let unavailable = choice.availabilityMessage {
                notice(unavailable, symbol: "exclamationmark.triangle", color: .orange)
            }
            if ownInstallation {
                if snapshot.installPhase == .downloading { progress(snapshot.downloadFraction, label: "Downloading model") }
                if snapshot.installPhase == .preparing { progress(snapshot.preparationFraction, label: "Preparing for your Mac") }
                if snapshot.installPhase == .checking || snapshot.installPhase == .cancelling { ProgressView().controlSize(.small) }
                if !snapshot.installMessage.isEmpty { Text(snapshot.installMessage).font(.callout).foregroundStyle(.secondary) }
                if let error = snapshot.installError { notice(error, symbol: "exclamationmark.triangle", color: .red) }
            }
            if let error = snapshot.engineError { notice(error, symbol: "exclamationmark.triangle", color: .red) }
            if starting { ProgressView().controlSize(.small) }
            if snapshot.modelReady, snapshot.runningModelID != installed {
                Text("\(snapshot.runningModelName) is still running. It stays installed when you switch.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                if choice.availabilityMessage != nil {
                    Button("Check models again") { requestChoices() }.buttonStyle(.borderedProminent)
                } else if installed != nil {
                    Button(starting ? "Starting…" : startLabel(choice)) { start(choice) }
                        .buttonStyle(.borderedProminent).disabled(starting || !snapshot.canStartModel)
                } else if installInProgress {
                    Button(snapshot.installPhase == .cancelling ? "Canceling…" : "Cancel new setup", action: actions.cancelInstall)
                        .disabled(snapshot.installPhase == .cancelling)
                } else {
                    Button("Try setup again") { install(choice) }
                        .buttonStyle(.borderedProminent).disabled(!snapshot.canInstall || choice.availabilityMessage != nil)
                }
                if session.choosingAnotherModel {
                    if snapshot.modelReady {
                        keepCurrentButton.disabled(starting || installInProgress)
                    } else if !starting && !installInProgress {
                        Button("Choose another model") { session.chooseAgain() }
                    }
                } else if !starting && !installInProgress {
                    Button("Back") { session.stage = .recommendation }
                }
            }
        } else {
            heading("Choose a starting model.", detail: "Recheck the recommendation before starting a download.")
            Button("Find a model") { requestChoices() }.buttonStyle(.borderedProminent)
        }
    }

    private var openWorkStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            heading("Get OpenWork.",
                    detail: "OpenWork gives your local model a workspace and tools. Stark supplies the model.")
            modelIdentity("OpenWork", detail: "Install the official macOS app")
            Text("Download OpenWork and move it to Applications. Stark will take you to the connection step when it detects the installed app.")
                .foregroundStyle(.secondary)
            Button("Download OpenWork", action: actions.downloadOpenWork).buttonStyle(.borderedProminent)
            Button("Check installation", action: actions.checkConnection)
            if !snapshot.openWorkMessage.isEmpty { Text(snapshot.openWorkMessage).font(.caption).foregroundStyle(.secondary) }
            Button("Chat in Stark instead", action: actions.openChat).buttonStyle(.link)
        }
    }

    private var reconnectOpenWorkButton: some View {
        Button(snapshot.openWorkBusy ? "Reconnecting…" : "Reconnect", action: actions.connectOpenWork)
            .disabled(snapshot.openWorkBusy || snapshot.openWorkPreparing || !snapshot.localAPIAvailable || !snapshot.openWorkInstalled)
            .help("Refresh the connection and reopen OpenWork automatically to apply its settings.")
    }

    @ViewBuilder private var connectionStep: some View {
        heading("Connect OpenWork to Stark.", detail: "Use the model running on this Mac. Your saved connection can be reused when you change models.")
        modelIdentity(snapshot.runningModelName.isEmpty ? "No model running" : snapshot.runningModelName,
                      variant: snapshot.runningModelVariantName)
        modelRepositoryLink(snapshot.runningModelRepositoryURL)
        if !snapshot.localAPIAvailable {
            notice("Stark’s local connection is unavailable.", symbol: "exclamationmark.triangle", color: .orange)
            if !snapshot.openWorkMessage.isEmpty { Text(snapshot.openWorkMessage).font(.callout).foregroundStyle(.secondary) }
            HStack {
                Button("Retry local connection", action: actions.retryLocalAPI).buttonStyle(.borderedProminent)
                Button("Advanced settings", action: actions.showAdvanced)
            }
            Button("Chat in Stark instead", action: actions.openChat).buttonStyle(.link)
        } else if !snapshot.canUseOpenWork {
            notice(!snapshot.modelReady ? "Start a Chat-mode model before connecting."
                : !snapshot.runningContextKnown ? "This model has no declared conversation limit. Choose another checked model to use OpenWork."
                : "This model is available for chat, but Stark has not enabled tool use for it. Choose another model for OpenWork.",
                   symbol: "exclamationmark.triangle", color: .orange)
            HStack {
                Button(snapshot.modelReady ? "Choose another model" : "Choose a starting model") {
                    if snapshot.modelReady { session.beginAnotherModel(snapshot: snapshot) }
                    else { session.chooseAgain() }
                }.disabled(snapshot.engineBusy)
                Button("Chat in Stark instead", action: actions.openChat)
            }
        } else {
            checkRow("Connection allowed", detail: "Allow OpenWork to send requests to this model.", checked: snapshot.openWorkApproved)
            if !snapshot.openWorkMessage.isEmpty { Text(snapshot.openWorkMessage).font(.callout).foregroundStyle(.secondary) }
            Button(snapshot.openWorkBusy ? "Connecting…" : "Allow and connect", action: actions.connectOpenWork)
                .buttonStyle(.borderedProminent).disabled(snapshot.openWorkBusy || !snapshot.openWorkInstalled)
            Text("Stark uses OpenWork’s existing workspace and opens a new task with the running model selected. OpenWork may reopen to apply connection settings. You can revoke access in Stark’s Connections window.")
                .font(.caption).foregroundStyle(.secondary)
            Button("Chat in Stark instead", action: actions.openChat).buttonStyle(.link)
        }
    }

    private var openOpenWorkStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            heading("Getting OpenWork ready…", detail: "Stark uses OpenWork’s current workspace and selects your running model in a new task.")
            modelIdentity("OpenWork", detail: "Installed on this Mac")
            VStack(alignment: .leading, spacing: 14) {
                checkRow("Connection allowed", detail: "OpenWork is authorized to use Stark.", checked: snapshot.openWorkApproved)
                checkRow("OpenWork checked this model", detail: "Waiting for its verified connector to refresh.", checked: snapshot.openWorkClientActive)
                checkRow("Task configured", detail: "Stark and \(snapshot.runningModelName) selected automatically.", checked: snapshot.openWorkPreparedModelID == snapshot.runningModelID)
            }.padding(16).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
            if !snapshot.openWorkMessage.isEmpty { Text(snapshot.openWorkMessage).font(.callout).foregroundStyle(.secondary) }
            if snapshot.openWorkPreparing { ProgressView("Selecting your model…") }
            HStack {
                Button("Open OpenWork", action: actions.openOpenWork).buttonStyle(.borderedProminent)
                reconnectOpenWorkButton
                Button("Try again", action: actions.checkConnection).disabled(snapshot.openWorkBusy || snapshot.openWorkPreparing)
            }
            Text("Review file and tool permissions in OpenWork before starting a task.").font(.caption).foregroundStyle(.secondary)
            Button("Chat in Stark instead", action: actions.openChat).buttonStyle(.link)
        }
    }

    private var readyStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            Image(systemName: "checkmark.circle.fill").font(.largeTitle).foregroundStyle(.green).accessibilityHidden(true)
            heading("OpenWork is ready.", detail: "Your local model is selected in a new OpenWork task, using its existing workspace.")
            VStack(alignment: .leading, spacing: 15) {
                checkRow(snapshot.runningModelName,
                         detail: [snapshot.runningModelVariantName, "Running on this Mac."].filter { !$0.isEmpty }.joined(separator: "\n"),
                         checked: snapshot.modelReady || snapshot.modelGenerating)
                modelRepositoryLink(snapshot.runningModelRepositoryURL)
                checkRow("OpenWork checked this model", detail: "Recent connector check detected by Stark.", checked: snapshot.openWorkClientActive)
                checkRow("Task configured automatically", detail: "Stark and your running model are selected.", checked: snapshot.openWorkPreparedModelID == snapshot.runningModelID)
            }.padding(16).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
            HStack {
                Button("Open OpenWork", action: actions.openOpenWork).buttonStyle(.borderedProminent)
                reconnectOpenWorkButton
                Button("Chat in Stark instead", action: actions.openChat)
            }
            if !snapshot.openWorkMessage.isEmpty { Text(snapshot.openWorkMessage).font(.callout).foregroundStyle(.secondary) }
            Text("Start with one small task. Models can make mistakes; review file and tool actions in OpenWork.")
                .font(.caption).foregroundStyle(.secondary)
            Divider()
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Not the right fit?").font(.headline)
                    Text("Get help choosing another model.").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Try a different model") { session.beginAnotherModel(snapshot: snapshot) }
                    .disabled(!snapshot.modelReady)
            }
        }
    }

    private var retryStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            heading("What would you like to change?", detail: "We’ll walk through model choice again. Your OpenWork connection and workspace choice are kept.")
            ForEach(BeginnerRetryPreference.allCases) { preference in
                selection(title: preference.title, detail: preference.detail, symbol: "arrow.triangle.2.circlepath", selected: session.retryPreference == preference) {
                    session.retryPreference = preference
                }
            }
            notice("Currently using \(snapshot.runningModelName). It stays installed while you choose and prepare another model.", symbol: "checkmark.circle")
            HStack {
                Button("Continue") {
                    if session.retryPreference == .task { session.stage = .task }
                    else { requestChoices() }
                }.buttonStyle(.borderedProminent).disabled(session.retryPreference == nil)
                keepCurrentButton
            }
        }
    }

    private var keepCurrentButton: some View {
        Button("Keep current model") {
            session.keepCurrentModel()
            session.synchronize(with: snapshot)
        }.disabled(installInProgress || session.startRequestedModelID != nil)
    }

    @ViewBuilder private var deviceSummary: some View {
        if !snapshot.deviceSummary.isEmpty {
            Label(snapshot.deviceSummary, systemImage: "laptopcomputer").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func requestChoices() {
        guard let task = session.task else { return }
        let preference = session.requestRecommendations()
        actions.refreshChoices(task, session.excludedModelID, preference)
    }

    private func navigationTarget(for index: Int) -> BeginnerSetupStage {
        switch index {
        case 0: return .task
        case 1: return .installation
        case 2: return .openWork
        case 3: return .connection
        case 4: return .openOpenWork
        default: return .ready
        }
    }

    private func navigationHelp(for target: BeginnerSetupStage) -> String {
        if session.installingChoiceID != nil || snapshot.installPhase.inProgress || snapshot.modelStarting {
            return "Wait for the current model setup or startup to finish."
        }
        switch target {
        case .task: return "Choose or change the kind of help you need."
        case .recommendation, .retry:
            return session.task == nil ? "Choose a task first." : "Review model choices for your selected task."
        case .installation:
            return snapshot.installChoiceID != nil || snapshot.installedResultID != nil || session.selectedChoiceID != nil
                ? "Review the selected model setup." : "Choose a model first."
        case .openOpenWork:
            return snapshot.openWorkApproved ? "Prepare an OpenWork task with your running model automatically." : "Connect OpenWork first."
        case .openWork, .connection:
            return snapshot.modelReady || session.keepsCompletedSetupWhileGenerating(snapshot: snapshot)
                ? "Continue OpenWork setup." : "Start a model first."
        case .ready:
            return session.canFinish(snapshot: snapshot) || session.keepsCompletedSetupWhileGenerating(snapshot: snapshot)
                ? "Review completed setup." : "Wait for automatic OpenWork setup to finish."
        }
    }

    private func start(_ choice: BeginnerModelChoice) {
        guard let identifier = session.installedID(for: choice, in: snapshot) else { return }
        session.selectedChoiceID = choice.id
        if snapshot.modelReady && snapshot.runningModelID == identifier && !snapshot.choiceRequiresRestart {
            session.continueAfterModel(snapshot: snapshot)
        } else if actions.startModel(choice.installed(as: identifier)) {
            session.requestStart(modelID: identifier)
        }
    }

    private func install(_ choice: BeginnerModelChoice) {
        if choice.availabilityMessage != nil { requestChoices(); return }
        if actions.install(choice) { session.requestInstall(choiceID: choice.id) }
    }

    private func startLabel(_ choice: BeginnerModelChoice) -> String {
        let identifier = session.installedID(for: choice, in: snapshot)
        if snapshot.modelReady && snapshot.runningModelID == identifier {
            return snapshot.choiceRequiresRestart ? "Apply setup settings and restart" : "Continue to OpenWork"
        }
        if snapshot.modelReady { return "Switch to \(choice.name)" }
        return "Start \(choice.name)"
    }

    private func downloadLabel(_ choice: BeginnerModelChoice) -> String {
        if let bytes = choice.downloadBytes { return "Download and set up · \(Self.bytes(bytes))" }
        return "Download and set up"
    }

    private func heading(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.title2.weight(.semibold)).fixedSize(horizontal: false, vertical: true)
            Text(detail).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func selection(title: String, detail: String, symbol: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: symbol).frame(width: 22).foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.callout.weight(.medium))
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                if selected { Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.accentColor) }
            }.padding(13).frame(maxWidth: .infinity, alignment: .leading)
                .background(selected ? Color.accentColor.opacity(0.07) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 9))
                .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(selected ? Color.accentColor : Color(nsColor: .separatorColor), lineWidth: selected ? 1 : 0.5))
        }.buttonStyle(.plain).accessibilityAddTraits(selected ? .isSelected : [])
    }

    private func modelIdentity(_ name: String, variant: String = "", detail: String = "") -> some View {
        HStack(spacing: 12) {
            Image(systemName: "shippingbox").font(.title2).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                Text(name).font(.headline).fixedSize(horizontal: false, vertical: true)
                if !variant.isEmpty, variant != name {
                    Text(variant).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
                }
                if !detail.isEmpty { Text(detail).font(.caption).foregroundStyle(.secondary) }
            }
        }
    }

    private func modelIdentity(_ choice: BeginnerModelChoice, detail: String = "") -> some View {
        VStack(alignment: .leading, spacing: 8) {
            modelIdentity(choice.name, variant: choice.variantName, detail: detail)
            modelRepositoryLink(choice.repositoryURL)
        }
    }

    @ViewBuilder private func modelRepositoryLink(_ url: URL?) -> some View {
        if let url {
            Link("View on Hugging Face", destination: url).font(.caption).help(url.absoluteString)
        }
    }

    private func modelCard(_ choice: BeginnerModelChoice) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            modelIdentity(choice, detail: choice.summary)
            if session.installedID(for: choice, in: snapshot) != nil {
                Label("Already on your Mac", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.secondary)
            } else if let bytes = choice.downloadBytes, bytes > 0 {
                Text("\(Self.bytes(bytes)) download").font(.callout)
            }
            if let bytes = choice.installedBytes { Text("Estimated installed storage: \(Self.bytes(bytes))").font(.caption).foregroundStyle(.secondary) }
            if session.installedID(for: choice, in: snapshot) == nil, let bytes = choice.setupBytes {
                Text("Estimated free space needed during setup: \(Self.bytes(bytes))").font(.caption).foregroundStyle(.secondary)
            }
            DisclosureGroup("Model and download details") {
                VStack(alignment: .leading, spacing: 8) {
                    if !choice.source.isEmpty { Text("Source: \(choice.source)").textSelection(.enabled) }
                    if !choice.revision.isEmpty { Text("Revision: \(choice.revision)").textSelection(.enabled) }
                    if choice.contextTokens > 0 { Text("Starting conversation limit: \(choice.contextTokens) tokens") }
                    if !choice.qualification.isEmpty { Text(choice.qualification) }
                }.font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.top, 8)
            }.font(.caption)
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    private func notice(_ text: String, symbol: String, color: Color = .secondary) -> some View {
        Label(text, systemImage: symbol).font(.caption).foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func checkRow(_ title: String, detail: String, checked: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: checked ? "checkmark.circle" : "circle").foregroundStyle(checked ? .green : .secondary)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.callout)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }.accessibilityElement(children: .combine)
    }

    @ViewBuilder private func progress(_ fraction: Double?, label: String) -> some View {
        if let fraction, fraction.isFinite {
            ProgressView(label, value: min(1, max(0, fraction))).accessibilityLabel(label)
        } else { ProgressView(label) }
    }

    private static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: max(0, value), countStyle: .file)
    }
}
