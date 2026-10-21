import AppKit
import SwiftUI

struct InstalledModel: Identifiable, Equatable {
    let id: String
    let name: String
    let path: String
}

final class EngineController: ObservableObject {
    @Published var models: [InstalledModel] = []
    @Published var selectedID = UserDefaults.standard.string(forKey: "selectedModel") ?? ""
    @Published var loadedID = ""
    @Published var state = "idle"
    @Published var message = "Starting Stark…"
    @Published var errorMessage: String?
    @Published var response = ""
    @Published var finishReason = ""
    @Published var inputSpeed: Double?
    @Published var outputSpeed: Double?
    @Published var inputTokens = 0
    @Published var outputTokens = 0
    var onInstalled: (() -> Void)?
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var pending = Data()
    private let reader = DispatchQueue(label: "stark.desktop.reader")
    private var shuttingDown = false

    var busy: Bool { ["loading", "generating", "installing", "stopping"].contains(state) }
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
                self.state = "error"
                self.message = "Stark stopped. Quit and reopen the app to restart."
                self.loadedID = ""
                self.input = nil
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
        case "models":
            models = (event["models"] as? [[String: Any]] ?? []).compactMap {
                guard let id = $0["id"] as? String, let name = $0["name"] as? String,
                      let path = $0["path"] as? String else { return nil }
                return InstalledModel(id: id, name: name, path: path)
            }
            if !models.contains(where: { $0.id == selectedID }) { selectedID = models.first?.id ?? "" }
        case "status":
            state = event["state"] as? String ?? "idle"
            message = event["message"] as? String ?? ""
            if state == "idle" || state == "error" { loadedID = "" }
        case "ready":
            loadedID = event["model_id"] as? String ?? selectedID
            state = "ready"
            message = "Ready"
        case "installed":
            if let model = event["model"] as? [String: Any], let id = model["id"] as? String {
                selectedID = id
                UserDefaults.standard.set(id, forKey: "selectedModel")
            }
            onInstalled?()
        case "reply":
            response = event["text"] as? String ?? ""
            finishReason = event["finish_reason"] as? String ?? ""
            inputTokens = event["input_tokens"] as? Int ?? 0
            outputTokens = event["output_tokens"] as? Int ?? 0
            inputSpeed = rate(event["input_tokens_per_second"])
            outputSpeed = rate(event["output_tokens_per_second"])
        case "error":
            errorMessage = event["message"] as? String ?? "Unable to complete the operation."
        default: break
        }
    }

    private func rate(_ value: Any?) -> Double? {
        guard let number = value as? Double, number.isFinite, number >= 0 else { return nil }
        return number
    }

    func send(_ value: [String: Any]) {
        guard let input = input, var bytes = try? JSONSerialization.data(withJSONObject: value) else { return }
        guard bytes.count < 1024 * 1024 else {
            errorMessage = "The prompt is too large. Send a shorter message."
            state = loadedID.isEmpty ? "idle" : "ready"
            message = loadedID.isEmpty ? "Stopped" : "Ready"
            return
        }
        bytes.append(10)
        do { try input.write(contentsOf: bytes) }
        catch { state = "error"; message = "Cannot reach the runtime. Quit and reopen Stark." }
    }

    func run() {
        guard !selectedID.isEmpty, !busy else { return }
        UserDefaults.standard.set(selectedID, forKey: "selectedModel")
        errorMessage = nil
        response = ""; finishReason = ""; inputSpeed = nil; outputSpeed = nil
        state = "loading"; message = "Loading model…"
        send(["command": "run", "model_id": selectedID])
    }

    func install(source: String, kind: String) {
        errorMessage = nil
        state = "installing"; message = kind == "local" ? "Installing local model…" : "Downloading and converting model…"
        send(["command": "install", "source": source, "kind": kind])
    }

    func generate(_ text: String) {
        guard state == "ready", !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        errorMessage = nil
        state = "generating"; message = "Generating…"; response = ""; finishReason = ""
        send(["command": "generate", "text": text])
    }

    func reset() {
        guard !busy else { return }
        errorMessage = nil
        response = ""; finishReason = ""; inputSpeed = nil; outputSpeed = nil
        send(["command": "reset"])
    }

    func stop() {
        errorMessage = nil
        state = "stopping"; message = "Stopping…"
        send(["command": "stop"])
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
    let install: () -> Void
    let run: () -> Void
    let console: () -> Void
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
            SpeedView(engine: engine)
            Text("Last request").font(.caption2).foregroundStyle(.secondary)
                .padding(.top, -10)
            HStack {
                Button("Install Model…", action: install).disabled(engine.busy)
                Spacer()
                if engine.busy || !engine.loadedID.isEmpty {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                }
                Button("Run", action: run).buttonStyle(.borderedProminent)
                    .disabled(engine.selectedID.isEmpty || engine.busy)
            }
            Divider()
            HStack {
                Button("Open Prompt Window", action: console).buttonStyle(.link)
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }.buttonStyle(.link)
            }.font(.caption)
        }.padding(18).frame(width: 350)
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
            }.pickerStyle(.segmented).onChange(of: kind) { source = "" }.disabled(engine.busy)
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
                TextField("Repository, e.g. Qwen/Qwen3-0.6B", text: $source).textFieldStyle(.roundedBorder).disabled(engine.busy)
                Text("Downloads and converts a supported model. Gated repositories use your existing Hugging Face sign-in.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Text("Models stay on this Mac. Conversion may take several minutes.")
                .font(.caption).foregroundStyle(.secondary)
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
        }.padding(24).frame(width: 470, height: 320)
    }
}

struct PromptView: View {
    @ObservedObject var engine: EngineController
    @State private var text = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(engine.activeName).font(.title3.weight(.semibold))
                    StatusLine(engine: engine)
                }
                Spacer()
                if !engine.loadedID.isEmpty || engine.busy {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                } else {
                    Button("Run", action: engine.run).disabled(engine.selectedID.isEmpty)
                }
                Button("Clear", action: engine.reset).disabled(engine.busy)
            }
            SpeedView(engine: engine)
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            ScrollView {
                Text(engine.response.isEmpty ? "Run a model, then send a prompt. Replies stay in memory for this session." : engine.response)
                    .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .topLeading)
                    .foregroundStyle(engine.response.isEmpty ? .secondary : .primary).padding(12)
            }.frame(maxHeight: .infinity).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
            if ["length", "context"].contains(engine.finishReason) {
                Text("Reply reached the token limit. Clear the conversation to start fresh.").font(.caption).foregroundStyle(.orange)
            }
            TextEditor(text: $text).font(.body).frame(height: 95).padding(4)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                .disabled(engine.state != "ready")
                .accessibilityLabel("Prompt")
            HStack {
                Text("Conversation stays in memory").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if engine.state == "generating" { Button("Stop", action: engine.stop) }
                Button("Send") { let prompt = text; text = ""; engine.generate(prompt) }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                    .disabled(engine.state != "ready" || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.padding(20).frame(minWidth: 540, minHeight: 570)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let engine = EngineController()
    private var status: NSStatusItem!
    private let popover = NSPopover()
    private var installWindow: NSWindow?
    private var promptWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.accessory)
        let menu = NSMenu()
        let applicationItem = NSMenuItem()
        let applicationMenu = NSMenu()
        let showMenu = NSMenuItem(title: "Show Stark Menu", action: #selector(toggle), keyEquivalent: "")
        showMenu.target = self
        applicationMenu.addItem(showMenu)
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
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(rootView: MenuView(
            engine: engine, install: { [weak self] in self?.showInstall() },
            run: { [weak self] in self?.run() }, console: { [weak self] in self?.showPrompt() }))
        engine.onInstalled = { [weak self] in self?.installWindow?.orderOut(nil); self?.showPrompt() }
        engine.start()
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
        guard let button = status.button else { return }
        if popover.isShown { popover.performClose(nil) }
        else { popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY) }
    }

    private func showInstall() {
        popover.performClose(nil)
        if installWindow == nil { installWindow = window(title: "Stark — Install Model", view: InstallView(engine: engine), resizable: false) }
        present(installWindow)
    }

    private func showPrompt() {
        popover.performClose(nil)
        if promptWindow == nil { promptWindow = window(title: "Stark", view: PromptView(engine: engine), resizable: true) }
        present(promptWindow)
    }

    private func run() {
        if engine.loadedID != engine.selectedID || engine.state != "ready" { engine.run() }
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
    func applicationWillTerminate(_ notification: Notification) { engine.shutdown() }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.run()
