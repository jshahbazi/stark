import AppKit
import SwiftUI

struct InstalledModel: Identifiable, Equatable {
    let id: String
    let name: String
    let path: String
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
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
    @Published var messages: [ChatMessage] = []
    @Published var draft = ""
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
    private var pendingMessageID: UUID?

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
                self.apiBaseURL = ""
                self.apiKey = ""
                self.apiMessage = "Stark stopped. Reopen the app to reconnect."
                self.input = nil
                self.discardPendingMessage()
                self.clearConversation()
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
                return InstalledModel(id: id, name: name, path: path)
            }
            if !models.contains(where: { $0.id == selectedID }) { selectedID = models.first?.id ?? "" }
        case "status":
            state = event["state"] as? String ?? "idle"
            message = event["message"] as? String ?? ""
            if state == "idle" || state == "error" {
                loadedID = ""
                discardPendingMessage()
                clearConversation()
            }
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
            guard pendingMessageID != nil, let text = event["text"] as? String else { return }
            messages.append(ChatMessage(text: text, isUser: false))
            pendingMessageID = nil
            finishReason = event["finish_reason"] as? String ?? ""
            inputTokens = event["input_tokens"] as? Int ?? 0
            outputTokens = event["output_tokens"] as? Int ?? 0
            inputSpeed = rate(event["input_tokens_per_second"])
            outputSpeed = rate(event["output_tokens_per_second"])
        case "error":
            errorMessage = event["message"] as? String ?? "Unable to complete the operation."
            discardPendingMessage()
        default: break
        }
    }

    private func rate(_ value: Any?) -> Double? {
        guard let number = value as? Double, number.isFinite, number >= 0 else { return nil }
        return number
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
        state = "loading"; message = "Loading model…"
        send(["command": "run", "model_id": selectedID])
    }

    func install(source: String, kind: String) {
        errorMessage = nil
        state = "installing"; message = kind == "local" ? "Installing local model…" : "Downloading and converting model…"
        send(["command": "install", "source": source, "kind": kind])
    }

    func generate() {
        let text = draft
        guard state == "ready", loadedID == selectedID,
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

    func reset() {
        guard !busy else { return }
        errorMessage = nil
        if send(["command": "reset"]) { clearConversation(); draft = "" }
    }

    func stop() {
        errorMessage = nil
        discardPendingMessage()
        clearConversation()
        state = "stopping"; message = "Stopping…"
        send(["command": "stop"])
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
            Button("Connection Details…", action: connection)
                .frame(maxWidth: .infinity, alignment: .leading)
            HStack {
                Button("Open Chat", action: console).buttonStyle(.link)
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

struct ChatView: View {
    @ObservedObject var engine: EngineController
    let connection: () -> Void
    @FocusState private var composerFocused: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Stark Chat").font(.title3.weight(.semibold))
                    StatusLine(engine: engine)
                }
                Spacer()
                Button("Connection Details…", action: connection)
                Button("New Chat") { engine.reset(); composerFocused = true }
                    .disabled(engine.busy || engine.messages.isEmpty || engine.loadedID != engine.selectedID)
            }
            HStack {
                Picker("Model", selection: $engine.selectedID) {
                    if engine.models.isEmpty { Text("Install a model to begin").tag("") }
                    ForEach(engine.models) { Text($0.name).tag($0.id) }
                }.disabled(engine.busy)
                if engine.loadedID != engine.selectedID || engine.loadedID.isEmpty {
                    Button("Run", action: engine.run).disabled(engine.selectedID.isEmpty || engine.busy)
                }
                if !engine.loadedID.isEmpty || engine.busy {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                }
            }
            if !engine.loadedID.isEmpty && engine.loadedID != engine.selectedID {
                Text("Run switches models and starts a new chat.").font(.caption).foregroundStyle(.secondary)
            }
            SpeedView(engine: engine)
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
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
                if engine.draft.isEmpty { Text("Message the model…").foregroundStyle(.tertiary).padding(.horizontal, 9).padding(.top, 12).allowsHitTesting(false) }
                TextEditor(text: $engine.draft).font(.body).scrollContentBackground(.hidden).frame(height: 70).padding(4)
                    .focused($composerFocused).disabled(engine.busy).accessibilityLabel("Message")
            }.overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            HStack {
                Text("Conversation stays in memory").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Send") { engine.generate(); composerFocused = true }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                    .disabled(engine.state != "ready" || engine.loadedID != engine.selectedID || engine.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.padding(20).frame(minWidth: 500, minHeight: 600)
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

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let engine = EngineController()
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
            engine: engine, install: { [weak self] in self?.showInstall() },
            run: { [weak self] in self?.run() }, console: { [weak self] in self?.showPrompt() },
            connection: { [weak self] in self?.showConnection() })
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13))
            .clipShape(RoundedRectangle(cornerRadius: 13)))
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
        if menuPanel.isVisible { dismissMenu(); return }
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

    private func dismissMenu() {
        menuPanel.orderOut(nil)
        if let monitor = localMenuMonitor { NSEvent.removeMonitor(monitor); localMenuMonitor = nil }
        if let monitor = globalMenuMonitor { NSEvent.removeMonitor(monitor); globalMenuMonitor = nil }
        updateMenuHighlight()
    }

    private func showInstall() {
        dismissMenu()
        if installWindow == nil { installWindow = window(title: "Stark — Install Model", view: InstallView(engine: engine), resizable: false) }
        present(installWindow)
    }

    private func showPrompt() {
        dismissMenu()
        if promptWindow == nil { promptWindow = window(title: "Stark — Chat", view: ChatView(engine: engine, connection: { [weak self] in self?.showConnection() }), resizable: true) }
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

@main
struct StarkApplication {
    static func main() {
        let application = NSApplication.shared
        let delegate = AppDelegate()
        application.delegate = delegate
        application.run()
        withExtendedLifetime(delegate) {}
    }
}
