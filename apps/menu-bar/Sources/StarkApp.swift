import AppKit
import SwiftUI

struct InstalledModel: Identifiable, Equatable {
    let id: String
    let name: String
    let path: String
    let apiID: String
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
}

enum PlaygroundMode: String, CaseIterable, Identifiable {
    case chat, decision
    var id: String { rawValue }
    var title: String { self == .chat ? "Chat" : "Decision" }
}

struct DecisionResult {
    let answers: [String]
    let probabilities: [Double]
    let choiceIndex: Int
    var choice: String { answers[choiceIndex] }
}

struct ChatOptions {
    var useModelSystemPrompt = true
    var systemPrompt = ""
    var contextTokens = 8192
    var maxNewTokens = 512
    var thinking = false
}

struct InstallProgress {
    var state = "waiting"
    var completedBytes: Int64 = 0
    var totalBytes: Int64?

    var fraction: Double {
        if state == "complete" || state == "skipped" { return 1 }
        guard let total = totalBytes, total > 0 else { return 0 }
        return min(1, max(0, Double(completedBytes) / Double(total)))
    }

    func detail(for phase: String) -> String {
        switch state {
        case "complete": return "Complete"
        case "skipped": return "Not needed"
        case "cancelled": return "Cancelled"
        case "failed": return "Failed"
        case "waiting": return "Waiting"
        default:
            guard let total = totalBytes, total > 0 else { return "Preparing…" }
            if completedBytes >= total { return phase == "download" ? "Verifying…" : "Finalizing…" }
            return String(format: "%.0f%%", fraction * 100)
        }
    }
}

final class EngineController: ObservableObject {
    @Published var models: [InstalledModel] = []
    @Published var selectedID = UserDefaults.standard.string(forKey: "selectedModel") ?? ""
    @Published var loadedID = ""
    @Published var state = "idle"
    @Published var message = "Starting Stark…"
    @Published var errorMessage: String?
    @Published var apiBaseURL = ""
    @Published var apiKey = ""
    @Published var apiMessage = "Starting the local API…"
    @Published var apiContextTokens = 0
    @Published var apiMaxNewTokens = 0
    @Published var messages: [ChatMessage] = []
    @Published var draft = ""
    @Published private(set) var mode = PlaygroundMode(rawValue: UserDefaults.standard.string(forKey: "playgroundMode") ?? "") ?? .chat
    @Published private(set) var loadedMode: PlaygroundMode?
    @Published var decisionText = ""
    @Published var decisionQuestion = ""
    @Published var decisionAnswers = "Yes, No"
    @Published private(set) var decisionResult: DecisionResult?
    @Published var finishReason = ""
    @Published var inputSpeed: Double?
    @Published var outputSpeed: Double?
    @Published var inputTokens = 0
    @Published var outputTokens = 0
    @Published var downloadProgress = InstallProgress()
    @Published var conversionProgress = InstallProgress()
    @Published var options = ChatOptions()
    @Published var contextLimit = 8192
    @Published var outputLimit = 512
    var onInstalled: (() -> Void)?
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var pending = Data()
    private let reader = DispatchQueue(label: "stark.desktop.reader")
    private var shuttingDown = false
    private var pendingMessageID: UUID?
    private var pendingOptions: ChatOptions?
    private var pendingDecisionAnswers: [String]?

    var busy: Bool { ["loading", "generating", "deciding", "installing", "stopping", "configuring"].contains(state) }
    var modelReady: Bool { state == "ready" && !loadedID.isEmpty && loadedID == selectedID && loadedMode == mode }
    var activeName: String { models.first { $0.id == loadedID }?.name ?? "No model running" }
    var statusColor: Color {
        if state == "error" { return .red }
        if state == "ready" { return .green }
        return busy ? .orange : .secondary
    }

    func start() {
        guard process == nil, let resources = Bundle.main.resourceURL else { return }
        let runtime = resources.appendingPathComponent("runtime")
        let python = runtime.appendingPathComponent("libexec/python/bin/python3.13")
        let data = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Stark", isDirectory: true)
        let child = Process()
        child.executableURL = python
        child.arguments = ["-I", "-B", "-m", "stark_convert.desktop", "--worker",
                           runtime.appendingPathComponent("libexec/stark-generate").path,
                           "--score-worker", runtime.appendingPathComponent("libexec/stark-score").path,
                           "--data-dir", data.path]
        let incoming = Pipe()
        let outgoing = Pipe()
        child.standardInput = incoming
        child.standardOutput = outgoing
        child.standardError = FileHandle.nullDevice
        // Keep native diagnostics and Python history out of inherited capture paths.
        child.environment = ProcessInfo.processInfo.environment.filter {
            !$0.key.hasPrefix("STARK_CAPTURE_") && !$0.key.hasPrefix("UZU_CAPTURE_") &&
            !["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"].contains($0.key)
        }
        child.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self = self, !self.shuttingDown else { return }
                if self.state == "installing" { self.endInstallProgress("failed") }
                self.state = "error"
                self.message = "Stark stopped. Quit and reopen the app to restart."
                self.loadedID = ""
                self.apiBaseURL = ""
                self.apiKey = ""
                self.apiMessage = "Stark stopped. Reopen the app to reconnect."
                self.loadedMode = nil
                self.pendingOptions = nil
                self.input = nil
                self.discardPendingMessage()
                self.clearConversation()
                self.clearDecision()
            }
        }
        output = outgoing.fileHandleForReading
        output?.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData
            guard !bytes.isEmpty else { handle.readabilityHandler = nil; return }
            self?.reader.async { [weak self] in self?.receive(bytes) }
        }
        do {
            try child.run()
            process = child
            input = incoming.fileHandleForWriting
            send(["command": "list"])
        } catch {
            output?.readabilityHandler = nil
            state = "error"
            message = "Cannot start the bundled runtime. Reinstall Stark."
        }
    }

    private func receive(_ bytes: Data) {
        pending.append(bytes)
        guard pending.count <= 20 * 1024 * 1024 else {
            pending.removeAll()
            DispatchQueue.main.async { self.stop(); self.message = "Worker reply exceeded the size limit." }
            return
        }
        while let boundary = pending.firstIndex(of: 10) {
            let line = pending.prefix(upTo: boundary)
            pending.removeSubrange(...boundary)
            guard let value = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
            DispatchQueue.main.async { [weak self] in self?.apply(value) }
        }
    }

    private func apply(_ event: [String: Any]) {
        switch event["event"] as? String {
        case "connection":
            apiBaseURL = event["base_url"] as? String ?? ""
            apiKey = event["api_key"] as? String ?? ""
            apiMessage = event["message"] as? String ?? "Local API is available on this Mac."
        case "models":
            models = (event["models"] as? [[String: Any]] ?? []).compactMap {
                guard let id = $0["id"] as? String, let name = $0["name"] as? String,
                      let path = $0["path"] as? String else { return nil }
                return InstalledModel(id: id, name: name, path: path, apiID: $0["api_id"] as? String ?? name)
            }
            if !models.contains(where: { $0.id == selectedID }) { selectedID = models.first?.id ?? "" }
        case "status":
            state = event["state"] as? String ?? "idle"
            message = event["message"] as? String ?? ""
            if state == "idle" || state == "error" {
                loadedID = ""
                loadedMode = nil
                discardPendingMessage()
                clearConversation()
                clearDecision()
            }
        case "ready":
            guard let readyMode = PlaygroundMode(rawValue: event["mode"] as? String ?? "chat"), readyMode == mode,
                  event["model_id"] as? String == selectedID, state != "stopping" else { return }
            loadedID = event["model_id"] as? String ?? selectedID
            apiContextTokens = event["max_context_tokens"] as? Int ?? 0
            apiMaxNewTokens = event["max_output_tokens"] as? Int ?? 0
            loadedMode = readyMode
            applyLimits(event)
            state = "ready"
            message = "Ready"
        case "configured":
            guard let configured = pendingOptions else { return }
            options = configured
            pendingOptions = nil
            applyLimits(event)
            if state == "configuring" {
                clearConversation()
                state = "ready"
                message = "Ready"
            }
        case "installed":
            if let model = event["model"] as? [String: Any], let id = model["id"] as? String {
                selectedID = id
                UserDefaults.standard.set(id, forKey: "selectedModel")
            }
            onInstalled?()
        case "reply":
            guard mode == .chat, loadedMode == .chat, pendingMessageID != nil,
                  let text = event["text"] as? String else { return }
            messages.append(ChatMessage(text: text, isUser: false))
            pendingMessageID = nil
            finishReason = event["finish_reason"] as? String ?? ""
            inputTokens = event["input_tokens"] as? Int ?? 0
            outputTokens = event["output_tokens"] as? Int ?? 0
            inputSpeed = rate(event["input_tokens_per_second"])
            outputSpeed = rate(event["output_tokens_per_second"])
        case "decision":
            guard mode == .decision, loadedMode == .decision, state == "deciding",
                  let expected = pendingDecisionAnswers else { return }
            guard let answers = event["answers"] as? [String], answers == expected,
                  let probabilities = event["probabilities"] as? [Double], probabilities.count == answers.count,
                  let index = event["choice_index"] as? Int, answers.indices.contains(index),
                  probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                  abs(probabilities.reduce(0, +) - 1) <= 0.0001,
                  probabilities[index] == probabilities.max() else {
                clearDecision()
                errorMessage = "Stark returned an invalid decision. Run the model again."
                stopAfterInvalidDecision()
                return
            }
            decisionResult = DecisionResult(answers: answers, probabilities: probabilities, choiceIndex: index)
            pendingDecisionAnswers = nil
        case "error":
            errorMessage = event["message"] as? String ?? "Unable to complete the operation."
            if state == "installing" { endInstallProgress("failed") }
            discardPendingMessage()
            pendingOptions = nil
            clearDecision()
            if state == "configuring" || state == "deciding" {
                state = loadedID.isEmpty ? "idle" : "ready"
                message = loadedID.isEmpty ? "Stopped" : "Ready"
            }
        case "progress":
            guard let phase = event["phase"] as? String, ["download", "conversion"].contains(phase),
                  let progressState = event["state"] as? String,
                  ["waiting", "running", "complete", "skipped", "cancelled", "failed"].contains(progressState),
                  let completed = event["completed_bytes"] as? Int64, completed >= 0 else { return }
            let total = event["total_bytes"] as? Int64
            guard total == nil || (total! >= 0 && completed <= total!) else { return }
            let progress = InstallProgress(state: progressState, completedBytes: completed, totalBytes: total)
            if phase == "download" { downloadProgress = progress } else { conversionProgress = progress }
        default: break
        }
    }

    private func rate(_ value: Any?) -> Double? {
        guard let number = value as? Double, number.isFinite, number >= 0 else { return nil }
        return number
    }

    private func applyLimits(_ event: [String: Any]) {
        contextLimit = event["max_context_tokens"] as? Int ?? contextLimit
        outputLimit = event["max_output_tokens"] as? Int ?? outputLimit
        options.contextTokens = event["context_tokens"] as? Int ?? options.contextTokens
        options.maxNewTokens = event["max_new_tokens"] as? Int ?? options.maxNewTokens
    }

    @discardableResult
    func configure(_ updated: ChatOptions) -> Bool {
        guard modelReady, mode == .chat else { return false }
        errorMessage = nil
        let payload: [String: Any] = [
            "command": "configure", "context_tokens": updated.contextTokens,
            "max_new_tokens": updated.maxNewTokens, "thinking": updated.thinking,
            "system_prompt": updated.useModelSystemPrompt ? NSNull() : updated.systemPrompt as Any
        ]
        guard send(payload) else { return false }
        pendingOptions = updated
        state = "configuring"
        message = "Applying advanced options…"
        return true
    }

    @discardableResult
    func send(_ value: [String: Any]) -> Bool {
        guard let input = input, var bytes = try? JSONSerialization.data(withJSONObject: value) else {
            state = "error"; message = "Cannot reach the runtime. Quit and reopen Stark."
            return false
        }
        guard bytes.count < 1024 * 1024 else {
            errorMessage = "The prompt is too large. Send a shorter message."
            state = loadedID.isEmpty ? "idle" : "ready"
            message = loadedID.isEmpty ? "Stopped" : "Ready"
            return false
        }
        bytes.append(10)
        do { try input.write(contentsOf: bytes); return true }
        catch { state = "error"; message = "Cannot reach the runtime. Quit and reopen Stark."; return false }
    }

    func run() {
        guard !selectedID.isEmpty, !busy else { return }
        UserDefaults.standard.set(selectedID, forKey: "selectedModel")
        errorMessage = nil
        clearConversation()
        clearDecision()
        loadedMode = nil
        state = "loading"; message = "Loading model…"
        send(["command": "run", "model_id": selectedID, "mode": mode.rawValue])
    }

    func changeMode(_ updated: PlaygroundMode) {
        guard !busy, updated != mode else { return }
        let reload = !loadedID.isEmpty
        mode = updated
        UserDefaults.standard.set(updated.rawValue, forKey: "playgroundMode")
        errorMessage = nil
        clearConversation()
        clearDecision()
        if reload { run() }
    }

    func install(source: String, kind: String) {
        errorMessage = nil
        resetInstallProgress(kind: kind)
        if kind == "huggingface" { downloadProgress.state = "running" }
        state = "installing"; message = kind == "local" ? "Installing local model…" : "Downloading and converting model…"
        if !send(["command": "install", "source": source, "kind": kind]) { endInstallProgress("failed") }
    }

    func generate() {
        let text = draft
        guard modelReady, mode == .chat,
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        errorMessage = nil
        state = "generating"; message = "Generating…"; finishReason = ""
        if send(["command": "generate", "text": text]) {
            let entry = ChatMessage(text: text, isUser: true)
            messages.append(entry)
            pendingMessageID = entry.id
            draft = ""
        }
    }

    var parsedDecisionAnswers: [String] {
        decisionAnswers.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    var decisionValidation: String? {
        let answers = parsedDecisionAnswers
        if decisionText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter text to evaluate." }
        if decisionQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a question." }
        if !(2...16).contains(answers.count) { return "Enter between 2 and 16 answers." }
        if answers.contains(where: { $0.isEmpty }) { return "Each answer must have a label." }
        if Set(answers.map { $0.lowercased() }).count != answers.count { return "Use distinct answers." }
        return nil
    }

    func decide() {
        guard modelReady, mode == .decision else { return }
        if let validation = decisionValidation { errorMessage = validation; return }
        let answers = parsedDecisionAnswers
        errorMessage = nil
        clearDecision()
        state = "deciding"; message = "Deciding…"
        if send(["command": "decide", "text": decisionText, "question": decisionQuestion, "answers": answers]) {
            pendingDecisionAnswers = answers
        }
    }

    func clearDecision() {
        decisionResult = nil
        pendingDecisionAnswers = nil
    }

    private func stopAfterInvalidDecision() {
        state = "stopping"; message = "Stopping…"
        send(["command": "stop"])
    }

    func reset() {
        guard !busy else { return }
        errorMessage = nil
        if send(["command": "reset"]) { clearConversation(); draft = "" }
    }

    func stop() {
        errorMessage = nil
        if state == "installing" { endInstallProgress("cancelled") }
        discardPendingMessage()
        clearConversation()
        clearDecision()
        state = "stopping"; message = "Stopping…"
        send(["command": "stop"])
    }

    func resetInstallProgress(kind: String) {
        guard state != "installing" else { return }
        downloadProgress = InstallProgress(state: kind == "local" ? "skipped" : "waiting")
        conversionProgress = InstallProgress()
    }

    private func endInstallProgress(_ state: String) {
        if ["waiting", "running"].contains(downloadProgress.state) { downloadProgress.state = state }
        if ["waiting", "running"].contains(conversionProgress.state) { conversionProgress.state = state }
    }

    private func discardPendingMessage() {
        guard let id = pendingMessageID else { return }
        if let entry = messages.first(where: { $0.id == id }) { draft = entry.text }
        messages.removeAll { $0.id == id }
        pendingMessageID = nil
    }

    private func clearConversation() {
        messages.removeAll(); pendingMessageID = nil
        finishReason = ""; inputSpeed = nil; outputSpeed = nil
        inputTokens = 0; outputTokens = 0
    }

    func shutdown() {
        shuttingDown = true
        output?.readabilityHandler = nil
        send(["command": "quit"])
        try? input?.close()
        if let child = process, child.isRunning {
            let deadline = Date().addingTimeInterval(5)
            while child.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.02) }
            if child.isRunning { child.terminate() }
        }
        process = nil
    }
}

struct StatusLine: View {
    @ObservedObject var engine: EngineController
    var body: some View {
        HStack(spacing: 7) {
            Circle().fill(engine.statusColor).frame(width: 7, height: 7)
            Text(engine.message).font(.callout).lineLimit(3).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            if engine.busy { ProgressView().controlSize(.small) }
        }.accessibilityElement(children: .combine)
    }
}

struct SpeedView: View {
    @ObservedObject var engine: EngineController
    var body: some View {
        HStack(spacing: 12) {
            metric("Input", engine.inputSpeed)
            Divider().frame(height: 40)
            metric("Output", engine.outputSpeed)
        }
        .padding(12)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .help("Last request. Input uses time to first token, including setup. Output measures subsequent sampled tokens. A dash means no measurement is available.")
    }
    private func metric(_ title: String, _ value: Double?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value.map { String(format: "%.1f", $0) } ?? "—").font(.title3.monospacedDigit().weight(.medium))
                Text("tok/s").font(.caption).foregroundStyle(.secondary)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct MenuView: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var updater: AppUpdateController
    let install: () -> Void
    let run: () -> Void
    let console: () -> Void
    let connection: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                if let art = Bundle.main.url(forResource: "Stark", withExtension: "jpg"), let image = NSImage(contentsOf: art) {
                    Image(nsImage: image).resizable().scaledToFill().frame(width: 38, height: 38).clipShape(RoundedRectangle(cornerRadius: 8))
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Stark").font(.headline)
                    Text(engine.activeName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
            }
            StatusLine(engine: engine)
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            Picker("Model", selection: $engine.selectedID) {
                if engine.models.isEmpty { Text("Install a model to begin").tag("") }
                ForEach(engine.models) { Text($0.name).tag($0.id) }
            }.disabled(engine.busy)
            if engine.mode == .chat {
                SpeedView(engine: engine)
                Text("Last request").font(.caption2).foregroundStyle(.secondary)
                    .padding(.top, -10)
            }
            HStack {
                Button(engine.state == "installing" ? "Installation Progress…" : "Install Model…", action: install)
                    .disabled(engine.busy && engine.state != "installing")
                Spacer()
                if engine.busy || !engine.loadedID.isEmpty {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                }
                Button("Run", action: run).buttonStyle(.borderedProminent)
                    .disabled(engine.selectedID.isEmpty || engine.busy)
            }
            Divider()
            Button("Connection Details…", action: connection)
                .frame(maxWidth: .infinity, alignment: .leading)
            AppUpdateView(updater: updater)
            HStack {
                Button("Open Playground", action: console).buttonStyle(.link)
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }.buttonStyle(.link)
            }.font(.caption)
        }.padding(18).frame(width: 350)
    }
}

struct InstallStageView: View {
    let title: String
    let phase: String
    let progress: InstallProgress
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.callout.weight(.medium))
                Spacer()
                Text(progress.detail(for: phase)).font(.caption.monospacedDigit())
                    .foregroundStyle(progress.state == "failed" ? Color.red : .secondary)
            }
            Group {
                if progress.state == "running" && progress.totalBytes == nil {
                    ProgressView()
                } else {
                    ProgressView(value: progress.fraction)
                }
            }
            .progressViewStyle(.linear)
            .tint(progress.state == "failed" ? .red : .accentColor)
            .accessibilityLabel("\(title) progress")
            .accessibilityValue(progress.detail(for: phase))
            if let total = progress.totalBytes, total > 0 {
                Text("\(bytes(progress.completedBytes)) of \(bytes(total))")
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
    }
    private func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }
}

struct InstallView: View {
    @ObservedObject var engine: EngineController
    @State private var kind = "local"
    @State private var source = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Install a model").font(.title2.weight(.semibold))
            Picker("Source", selection: $kind) {
                Text("Local Folder").tag("local")
                Text("Hugging Face").tag("huggingface")
            }.pickerStyle(.segmented).onChange(of: kind) { source = ""; engine.resetInstallProgress(kind: kind) }.disabled(engine.busy)
            if kind == "local" {
                Text("Choose a converted Stark model or a supported Safetensors checkpoint folder. Converted models stay in the selected folder; keep that folder available.")
                    .font(.callout).foregroundStyle(.secondary)
                HStack {
                    Text(source.isEmpty ? "No folder selected" : source).lineLimit(2).truncationMode(.middle)
                        .font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Choose Folder…") {
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.allowsMultipleSelection = false
                        panel.prompt = "Choose Model"
                        if panel.runModal() == .OK { source = panel.url?.path ?? "" }
                    }.disabled(engine.busy)
                }
            } else {
                TextField("Repository name or full Hugging Face URL", text: $source).textFieldStyle(.roundedBorder).disabled(engine.busy)
                Text("Downloads and converts a supported model. Gated repositories use your existing Hugging Face sign-in.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Text("Models stay on this Mac. Conversion may take several minutes.")
                .font(.caption).foregroundStyle(.secondary)
            VStack(spacing: 14) {
                InstallStageView(title: "Download", phase: "download", progress: engine.downloadProgress)
                InstallStageView(title: "Conversion", phase: "conversion", progress: engine.conversionProgress)
            }
            .padding(14)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
            if engine.state == "installing" || engine.state == "error" { StatusLine(engine: engine) }
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            Spacer(minLength: 0)
            HStack {
                if engine.state == "installing" { Button("Cancel Installation", action: engine.stop) }
                Spacer()
                Button("Install") { engine.install(source: source.trimmingCharacters(in: .whitespacesAndNewlines), kind: kind) }
                    .buttonStyle(.borderedProminent).disabled(source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || engine.busy)
                    .keyboardShortcut(.defaultAction)
            }
        }.padding(24).frame(width: 470, height: 500)
            .onAppear { engine.resetInstallProgress(kind: kind) }
            .onChange(of: source) { if !engine.busy { engine.resetInstallProgress(kind: kind) } }
    }
}

struct ChatBubble: View {
    let message: ChatMessage
    var body: some View {
        HStack {
            if message.isUser { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 6) {
                Text(message.isUser ? "You" : "Stark").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(message.text).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(message.isUser ? Color.accentColor.opacity(0.12) : Color.primary.opacity(0.045),
                        in: RoundedRectangle(cornerRadius: 12))
            if !message.isUser { Spacer(minLength: 40) }
        }.frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
    }
}

struct AdvancedOptionsView: View {
    @ObservedObject var engine: EngineController
    @Environment(\.dismiss) private var dismiss
    @State private var useModelSystemPrompt: Bool
    @State private var systemPrompt: String
    @State private var context: String
    @State private var output: String
    @State private var thinking: Bool

    init(engine: EngineController) {
        self.engine = engine
        _useModelSystemPrompt = State(initialValue: engine.options.useModelSystemPrompt)
        _systemPrompt = State(initialValue: engine.options.systemPrompt)
        _context = State(initialValue: String(engine.options.contextTokens))
        _output = State(initialValue: String(engine.options.maxNewTokens))
        _thinking = State(initialValue: engine.options.thinking)
    }

    private var validation: String? {
        guard let count = Int(context), (2...engine.contextLimit).contains(count) else {
            return "Context size must be between 2 and \(engine.contextLimit) tokens."
        }
        guard let maximum = Int(output), (1...engine.outputLimit).contains(maximum) else {
            return "Output limit must be between 1 and \(engine.outputLimit) tokens."
        }
        return maximum < count ? nil : "Output limit must be smaller than context size."
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Advanced Options").font(.title2.weight(.semibold))
            Text("Applying changes starts a new chat with the model still loaded.")
                .font(.callout).foregroundStyle(.secondary)
            Toggle("Use model's default system prompt", isOn: $useModelSystemPrompt)
            if !useModelSystemPrompt {
                Text("System prompt").font(.headline)
                TextEditor(text: $systemPrompt).font(.body).frame(height: 100)
                    .padding(5).overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                    .accessibilityLabel("System prompt")
                Text("Leave empty to omit the system message.").font(.caption).foregroundStyle(.secondary)
            }
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                GridRow {
                    Text("Context size (tokens)")
                    TextField("Context size", text: $context).textFieldStyle(.roundedBorder)
                        .accessibilityLabel("Context size in tokens")
                }
                GridRow {
                    Text("Output limit (tokens)")
                    TextField("Output limit", text: $output).textFieldStyle(.roundedBorder)
                        .accessibilityLabel("Output limit in tokens")
                }
            }
            Text("Context includes the conversation and room for the full reply budget.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Thinking mode", isOn: $thinking)
            Text("Enables reasoning when the model's chat template supports it.")
                .font(.caption).foregroundStyle(.secondary)
            if let validation = validation { Text(validation).font(.caption).foregroundStyle(.red) }
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red) }
            Text("Settings stay in memory for this app session. Switching models may reduce token limits.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Restore Defaults") {
                    useModelSystemPrompt = true; systemPrompt = ""; thinking = false
                    let count = min(8192, engine.contextLimit)
                    context = String(count)
                    output = String(min(512, engine.outputLimit, count - 1))
                }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Apply and New Chat") {
                    guard let count = Int(context), let maximum = Int(output) else { return }
                    var options = ChatOptions()
                    options.useModelSystemPrompt = useModelSystemPrompt
                    options.systemPrompt = systemPrompt
                    options.contextTokens = count; options.maxNewTokens = maximum; options.thinking = thinking
                    if engine.configure(options) { dismiss() }
                }.buttonStyle(.borderedProminent).disabled(validation != nil || engine.busy || engine.state != "ready")
            }
        }.padding(24).frame(width: 510)
    }
}

struct PlaygroundView: View {
    @ObservedObject var engine: EngineController
    let connection: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Stark").font(.title3.weight(.semibold))
                Spacer()
                Button("Connection Details…", action: connection)
                Picker("Mode", selection: Binding(get: { engine.mode }, set: engine.changeMode)) {
                    ForEach(PlaygroundMode.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.segmented).frame(width: 180).disabled(engine.busy)
                    .accessibilityLabel("Chat or Decision mode")
            }
            StatusLine(engine: engine)
            HStack {
                Picker("Model", selection: $engine.selectedID) {
                    if engine.models.isEmpty { Text("Install a model to begin").tag("") }
                    ForEach(engine.models) { Text($0.name).tag($0.id) }
                }.disabled(engine.busy)
                if engine.loadedID != engine.selectedID || engine.loadedID.isEmpty || engine.loadedMode != engine.mode {
                    Button("Run", action: engine.run).disabled(engine.selectedID.isEmpty || engine.busy)
                }
                if !engine.loadedID.isEmpty || engine.busy {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                }
            }
            if !engine.loadedID.isEmpty && engine.loadedID != engine.selectedID {
                Text("Select Run to load the chosen model.").font(.caption).foregroundStyle(.secondary)
            }
            if let error = engine.errorMessage {
                Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            if engine.mode == .chat { ChatContentView(engine: engine) }
            else { DecisionView(engine: engine) }
        }.padding(20).frame(minWidth: 500, minHeight: 600)
            .onChange(of: engine.selectedID) { engine.clearDecision() }
    }
}

struct DecisionView: View {
    @ObservedObject var engine: EngineController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Text to evaluate").font(.callout)
                    TextEditor(text: $engine.decisionText).font(.body).frame(minHeight: 100, maxHeight: 150)
                        .padding(5).overlay(RoundedRectangle(cornerRadius: 7).stroke(.quaternary))
                        .accessibilityLabel("Text to evaluate")
                }
                VStack(alignment: .leading, spacing: 7) {
                    Text("Question").font(.callout)
                    TextField("Which answer fits this text?", text: $engine.decisionQuestion)
                        .textFieldStyle(.roundedBorder).accessibilityLabel("Decision question")
                }
                VStack(alignment: .leading, spacing: 7) {
                    Text("Answers").font(.callout)
                    TextField("Billing, Support, Sales", text: $engine.decisionAnswers)
                        .textFieldStyle(.roundedBorder).accessibilityLabel("Decision answers")
                    Text("Separate 2–16 answers with commas.").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Text("Inputs and results stay in memory").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Run decision", action: engine.decide)
                        .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                        .disabled(!engine.modelReady || engine.decisionValidation != nil)
                }
                if let result = engine.decisionResult {
                    Divider()
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Result").font(.caption).foregroundStyle(.secondary)
                        Text(result.choice).font(.title3.weight(.medium)).textSelection(.enabled)
                        ForEach(result.answers.indices, id: \.self) { index in
                            HStack(alignment: .firstTextBaseline) {
                                Text(result.answers[index]).fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: 12)
                                Text(result.probabilities[index], format: .percent.precision(.fractionLength(1)))
                                    .monospacedDigit()
                            }.font(.callout)
                        }
                        Text("Shares across these answers; calibration is unverified.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.accessibilityElement(children: .contain)
                }
            }.padding(.vertical, 2)
        }.disabled(engine.busy)
            .onChange(of: engine.decisionText) { engine.clearDecision() }
            .onChange(of: engine.decisionQuestion) { engine.clearDecision() }
            .onChange(of: engine.decisionAnswers) { engine.clearDecision() }
    }
}

struct ChatContentView: View {
    @ObservedObject var engine: EngineController
    @FocusState private var composerFocused: Bool
    @State private var advancedOptions = false
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Button("Advanced Options…") { advancedOptions = true }
                    .disabled(!engine.modelReady)
                    .help("Run a model to adjust its chat options.")
                Spacer()
                Button("New Chat") { engine.reset(); composerFocused = true }
                    .disabled(engine.busy || engine.messages.isEmpty || engine.loadedID != engine.selectedID)
            }
            SpeedView(engine: engine)
            ScrollViewReader { scroll in
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        if engine.messages.isEmpty {
                            VStack(spacing: 10) {
                                Image(systemName: "bubble.left.and.bubble.right").font(.system(size: 28)).foregroundStyle(.secondary)
                                Text("Try a conversation").font(.headline)
                                Text("Choose a model, select Run, and send a message.")
                                    .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                            }.frame(maxWidth: .infinity).padding(.vertical, 36)
                        }
                        ForEach(engine.messages) { ChatBubble(message: $0) }
                        if engine.state == "generating" {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Stark is thinking…").font(.callout).foregroundStyle(.secondary)
                            }.padding(.horizontal, 4)
                        }
                        Color.clear.frame(height: 1).id("latest")
                    }.padding(14)
                }
                .onChange(of: engine.messages.count) { withAnimation(.easeOut(duration: 0.15)) { scroll.scrollTo("latest", anchor: .bottom) } }
                .onChange(of: engine.state) { if engine.state == "generating" { scroll.scrollTo("latest", anchor: .bottom) } }
            }.frame(maxHeight: .infinity).background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 12))
            if ["length", "context"].contains(engine.finishReason) {
                Text("Reply reached the token limit. Clear the conversation to start fresh.").font(.caption).foregroundStyle(.orange)
            }
            ZStack(alignment: .topLeading) {
                if engine.draft.isEmpty { Text("Message the model…").font(.body).foregroundStyle(.tertiary).padding(.horizontal, 9).padding(.top, 4).allowsHitTesting(false) }
                TextEditor(text: $engine.draft).font(.body).scrollContentBackground(.hidden).frame(height: 70).padding(4)
                    .focused($composerFocused).disabled(engine.busy).accessibilityLabel("Message")
            }.overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            HStack {
                Text("Conversation stays in memory").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Send") { engine.generate(); composerFocused = true }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                    .disabled(!engine.modelReady || engine.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
            .sheet(isPresented: $advancedOptions) { AdvancedOptionsView(engine: engine) }
            .onAppear { composerFocused = true }
            .onChange(of: engine.state) { if engine.state == "ready" { composerFocused = true } }
    }
}

private final class MenuPanel: NSPanel {
    var onEscape: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    override func cancelOperation(_ sender: Any?) { onEscape?() }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let engine = EngineController()
    private let updater = AppUpdateController()
    private var status: NSStatusItem!
    private let menuPanel = MenuPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
    private var localMenuMonitor: Any?
    private var globalMenuMonitor: Any?
    private var installWindow: NSWindow?
    private var promptWindow: NSWindow?
    private var connectionWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.accessory)
        let menu = NSMenu()
        let applicationItem = NSMenuItem()
        let applicationMenu = NSMenu()
        let showMenu = NSMenuItem(title: "Show Stark Menu", action: #selector(toggle), keyEquivalent: "")
        showMenu.target = self
        applicationMenu.addItem(showMenu)
        let installMenu = NSMenuItem(title: "Install Model…", action: #selector(showInstall), keyEquivalent: "i")
        installMenu.target = self
        applicationMenu.addItem(installMenu)
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(withTitle: "Quit Stark", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        applicationItem.submenu = applicationMenu
        menu.addItem(applicationItem)
        let editItem = NSMenuItem()
        let edit = NSMenu(title: "Edit")
        for (title, action, key) in [("Undo", "undo:", "z"), ("Cut", "cut:", "x"),
                                      ("Copy", "copy:", "c"), ("Paste", "paste:", "v"),
                                      ("Select All", "selectAll:", "a")] {
            edit.addItem(withTitle: title, action: Selector(action), keyEquivalent: key)
        }
        editItem.submenu = edit; menu.addItem(editItem)
        NSApplication.shared.mainMenu = menu
        status = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let url = Bundle.main.url(forResource: "MenuBarTemplate", withExtension: "png"), let icon = NSImage(contentsOf: url) {
            let ratio = icon.size.height / icon.size.width
            icon.size = NSSize(width: 24, height: 24 * ratio); icon.isTemplate = true
            status.button?.image = icon
        } else { status.button?.title = "Stark" }
        status.button?.toolTip = "Stark"
        status.button?.setAccessibilityLabel("Stark menu bar")
        status.button?.target = self; status.button?.action = #selector(toggle)
        status.button?.sendAction(on: [.leftMouseUp, .rightMouseUp])
        menuPanel.level = .popUpMenu
        menuPanel.isOpaque = false
        menuPanel.backgroundColor = .clear
        menuPanel.hasShadow = true
        menuPanel.isReleasedWhenClosed = false
        menuPanel.collectionBehavior = [.transient, .moveToActiveSpace, .fullScreenAuxiliary]
        menuPanel.onEscape = { [weak self] in self?.dismissMenu() }
        menuPanel.contentViewController = NSHostingController(rootView: MenuView(
            engine: engine, updater: updater, install: { [weak self] in self?.showInstall() },
            run: { [weak self] in self?.run() }, console: { [weak self] in self?.showPrompt() },
            connection: { [weak self] in self?.showConnection() })
            .onChange(of: updater.message) { [weak self] _, _ in
                DispatchQueue.main.async { self?.resizeMenuForUpdate() }
            }
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13))
            .clipShape(RoundedRectangle(cornerRadius: 13)))
        engine.onInstalled = { [weak self] in self?.installWindow?.orderOut(nil); self?.showPrompt() }
        engine.start()
        updater.startAutomaticChecks()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
            guard let self = self else { return }
            if self.engine.models.isEmpty { self.showInstall() } else { self.toggle() }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { if engine.models.isEmpty { showInstall() } else { showPrompt() } }
        return true
    }

    @objc private func toggle() {
        if menuPanel.isVisible { dismissMenu(); return }
        updater.checkIfNeeded()
        guard let button = status.button, let statusWindow = button.window,
              let screen = statusWindow.screen else { return }
        let buttonFrame = statusWindow.convertToScreen(button.convert(button.bounds, to: nil))
        let safe = screen.visibleFrame.insetBy(dx: 8, dy: 8)
        let size = menuPanel.contentView?.fittingSize ?? NSSize(width: 350, height: 350)
        let width = min(size.width, safe.width)
        let height = min(size.height, safe.height)
        // Match MeeCP's visible status-item highlight edge (two points outside its window).
        let x = min(max(statusWindow.frame.minX - 2, safe.minX), safe.maxX - width)
        let top = min(max(buttonFrame.minY - 5, safe.minY + height), safe.maxY)
        menuPanel.setFrame(NSRect(x: x, y: top - height, width: width, height: height), display: true)
        menuPanel.makeKeyAndOrderFront(nil)
        updateMenuHighlight()
        localMenuMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown]) { [weak self] event in
            guard let self else { return event }
            if event.type == .keyDown {
                if event.keyCode == 53 { self.dismissMenu(); return nil }
            } else if event.window !== self.menuPanel && event.window !== self.status.button?.window {
                self.dismissMenu()
            }
            return event
        }
        globalMenuMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            self?.dismissMenu()
        }
    }

    private func updateMenuHighlight() {
        status.button?.highlight(menuPanel.isVisible)
        // AppKit clears the button's tracking highlight after dispatching its action.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.status.button?.highlight(self.menuPanel.isVisible)
        }
    }

    private func resizeMenuForUpdate() {
        guard menuPanel.isVisible, let screen = menuPanel.screen,
              let size = menuPanel.contentView?.fittingSize else { return }
        let safe = screen.visibleFrame.insetBy(dx: 8, dy: 8)
        let height = min(size.height, safe.height)
        let frame = menuPanel.frame
        menuPanel.setFrame(NSRect(x: frame.minX, y: max(safe.minY, frame.maxY - height),
                                 width: frame.width, height: height), display: true)
    }

    private func dismissMenu() {
        menuPanel.orderOut(nil)
        if let monitor = localMenuMonitor { NSEvent.removeMonitor(monitor); localMenuMonitor = nil }
        if let monitor = globalMenuMonitor { NSEvent.removeMonitor(monitor); globalMenuMonitor = nil }
        updateMenuHighlight()
    }

    @objc private func showInstall() {
        dismissMenu()
        if installWindow == nil { installWindow = window(title: "Stark — Install Model", view: InstallView(engine: engine), resizable: false) }
        present(installWindow)
    }

    private func showPrompt() {
        dismissMenu()
        if promptWindow == nil { promptWindow = window(title: "Stark — Playground", view: PlaygroundView(engine: engine, connection: { [weak self] in self?.showConnection() }), resizable: true) }
        present(promptWindow)
    }

    private func showConnection() {
        dismissMenu()
        if connectionWindow == nil {
            connectionWindow = window(title: "Stark — Connection Details", view: ConnectionDetailsView(engine: engine), resizable: false)
        }
        present(connectionWindow)
    }

    private func run() {
        if !engine.modelReady { engine.run() }
        showPrompt()
    }

    private func window<V: View>(title: String, view: V, resizable: Bool) -> NSWindow {
        let result = NSWindow(contentViewController: NSHostingController(rootView: view))
        result.title = title
        result.styleMask = resizable ? [.titled, .closable, .miniaturizable, .resizable] : [.titled, .closable]
        result.isReleasedWhenClosed = false
        result.center()
        return result
    }

    private func present(_ window: NSWindow?) {
        NSApplication.shared.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationWillTerminate(_ notification: Notification) {
        updater.stopAutomaticChecks()
        engine.shutdown()
    }
}

@main
struct StarkApplication {
    @MainActor
    static func main() {
        let application = NSApplication.shared
        let delegate = AppDelegate()
        application.delegate = delegate
        application.run()
        withExtendedLifetime(delegate) {}
    }
}
