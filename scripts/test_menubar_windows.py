"""Native checks of explicit window presentation, without starting a backend.

The executable includes the production delegate and views. It creates only
hidden windows, uses injected credential storage that fails on every access,
and never calls the application-launch callback or presents test UI. App hub
fixtures exercise both sections and expanded advanced settings across model
lifecycle states, using only synthetic models and a non-production key.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY = "\n@main\nstruct StarkApplication"
DRIVER = r"""

final class ForbiddenWindowCredentialStorage: APICredentialStorage {
    func read() -> (OSStatus, Data?) { fatalError("unexpected_credential_access") }
    func add(_ data: Data) -> OSStatus { fatalError("unexpected_credential_access") }
    func replace(_ data: Data) -> OSStatus { fatalError("unexpected_credential_access") }
}

final class ConnectionModeWindowProbe: NSWindow {
    var hiddenForBeginner = false
    override func orderOut(_ sender: Any?) { hiddenForBeginner = true; super.orderOut(sender) }
}

struct CatalogPickerBoundsProbe: NSViewRepresentable {
    let view: NSView
    func makeNSView(context: Context) -> NSView { view }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

@MainActor
enum SetupLayoutProbes {
    static let action = NSView()
    static let form = NSView()
}

extension AppDelegate {
    @MainActor static func exerciseBeginnerInitialLayout() -> Bool {
        _ = NSApplication.shared
        let suite = "stark.setup-initial-layout-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let engine = EngineController(apiCredentials: APICredentialStore(service: suite,
            storage: ForbiddenWindowCredentialStorage()), apiDefaults: defaults)
        let delegate = AppDelegate(engine: engine)
        delegate.beginnerSession.selectTask(.coding)
        let window = delegate.makeBeginnerWindow()
        defer { window.close(); engine.desktopConnections.shutdown() }
        guard let content = window.contentView, let screen = window.screen ?? NSScreen.main else { return false }
        for _ in 0..<3 {
            content.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        }
        guard abs(content.bounds.width - 700) < 1,
              window.frame.height <= screen.visibleFrame.height + 1,
              !window.isVisible else { return false }
        let titleHeight = window.frame.height - content.bounds.height
        if screen.visibleFrame.height >= 800 + titleHeight {
            let form = SetupLayoutProbes.form
            let action = SetupLayoutProbes.action
            guard form.window === window, action.window === window else { return false }
            let bounds = form.convert(form.bounds, to: content)
            let actionBounds = action.convert(action.bounds, to: content)
            // The whole task form, including its heading and final copy, fits at once.
            guard abs(content.bounds.height - 800) < 1,
                  content.bounds.contains(bounds), content.bounds.contains(actionBounds),
                  !bounds.intersects(actionBounds),
                  form.visibleRect.height >= form.bounds.height - 1 else { return false }
        }
        return true
    }

    @MainActor static func exerciseBeginnerSetupFooterLayouts() -> Bool {
        _ = NSApplication.shared
        let suite = "stark.setup-footer-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
        let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults)
        let delegate = AppDelegate(engine: engine)
        defer { engine.desktopConnections.shutdown() }
        for anotherModel in [false, true] {
            engine.configureConnectionFixture(state: anotherModel ? "ready" : "idle",
                loadedMode: anotherModel ? .chat : nil, apiAvailable: false, hasModels: anotherModel)
            let session = BeginnerSetupSession()
            session.selectTask(.coding)
            if anotherModel {
                var snapshot = BeginnerSetupSnapshot()
                snapshot.modelReady = true; snapshot.runningModelID = "window-fixture"
                session.beginAnotherModel(snapshot: snapshot)
                session.stage = .task
            }
            for stage in [BeginnerSetupStage.task, .priority, .retry] {
                session.stage = stage
                let view = BeginnerSetupHost(engine: engine, openWork: engine.desktopConnections.openWork, openCode: engine.desktopConnections.openCode,
                    session: session, interface: StarkInterfacePreferences(defaults: defaults),
                    observation: BeginnerOpenWorkStatus(), openCodeObservation: BeginnerOpenWorkStatus(client: .openCode), selectInterface: { _ in }, openChat: {})
                let window = delegate.window(title: "Setup footer fixture", view: view, resizable: true)
                // Resize the same retained window down, up, then back to the reported size.
                for size in [NSSize(width: 700, height: 650), NSSize(width: 700, height: 600),
                             NSSize(width: 900, height: 850), NSSize(width: 700, height: 650)] {
                    window.setContentSize(size)
                    guard let content = window.contentView else { return false }
                    for _ in 0..<3 {
                        content.layoutSubtreeIfNeeded()
                        RunLoop.current.run(until: Date().addingTimeInterval(0.02))
                    }
                    let probe = SetupLayoutProbes.action
                    guard probe.window === window else { return false }
                    let bounds = probe.convert(probe.bounds, to: content)
                    let bottomGap = content.isFlipped ? content.bounds.maxY - bounds.maxY : bounds.minY - content.bounds.minY
                    guard abs(bottomGap - 28) < 1, bounds.height >= 20,
                          bounds.minX >= 208, abs(content.bounds.maxX - bounds.maxX - 28) < 1,
                          abs(content.bounds.width - size.width) < 1,
                          abs(content.bounds.height - size.height) < 1,
                          content.bounds.contains(bounds), !window.isVisible else { return false }
                }
                window.close()
            }
        }
        return true
    }

    @MainActor static func exerciseBeginnerChatFooterLayouts() -> Bool {
        _ = NSApplication.shared
        let suite = "stark.chat-footer-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
        let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults)
        engine.configureConnectionFixture(state: "ready", loadedMode: .chat, apiAvailable: true, hasModels: true)
        let delegate = AppDelegate(engine: engine)
        for height in [600, 650, 850] {
            for error in [nil, "A longer model error that wraps across multiple lines. " + String(repeating: "Details. ", count: 30)] as [String?] {
                engine.errorMessage = error
                let view = BeginnerChatView(engine: engine, interface: StarkInterfacePreferences(defaults: defaults),
                    selectInterface: { _ in }, setup: {}, tryAnotherModel: {})
                let window = delegate.window(title: "Chat footer fixture", view: view, resizable: true)
                window.setContentSize(NSSize(width: 520, height: height))
                guard let content = window.contentView else { return false }
                for _ in 0..<3 {
                    content.layoutSubtreeIfNeeded()
                    RunLoop.current.run(until: Date().addingTimeInterval(0.02))
                }
                let probe = view.windowFixtureFooterProbe
                guard probe.window === window else { return false }
                let bounds = probe.convert(probe.bounds, to: content)
                let bottomGap = content.isFlipped ? content.bounds.maxY - bounds.maxY : bounds.minY - content.bounds.minY
                guard abs(bottomGap - 20) < 1, bounds.height > 10,
                      bounds.minX >= 19, bounds.maxX <= content.bounds.maxX - 19,
                      content.bounds.contains(bounds), !window.isVisible else { return false }
                window.close()
            }
        }
        engine.desktopConnections.shutdown()
        return true
    }

}

@MainActor
func exerciseCatalogPickerLayouts() -> Bool {
    _ = NSApplication.shared
    let compact: [CatalogQuantizationFilter] = [.all, .quantization("4-bit"), .quantization("8-bit"), .quantization("BF16")]
    let long: [CatalogQuantizationFilter] = [.all,
        .quantization("Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed)"),
        .quantization("MLX mixed 4/8-bit"), .quantization("MLX-Serve mixed 4/8-bit")]
    for choices in [[.all], compact, long] {
        for selection in choices {
            for width: CGFloat in [320, 592] {
                let probe = NSView()
                let picker = CatalogQuantizationPicker(choices: choices, selection: .constant(selection))
                    .overlay(CatalogPickerBoundsProbe(view: probe))
                    .frame(width: width, height: 40)
                let host = NSHostingView(rootView: picker)
                let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: 40),
                                      styleMask: [.titled], backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                window.contentView = host
                defer { window.close() }
                host.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date().addingTimeInterval(0.01))
                host.layoutSubtreeIfNeeded()
                let bounds = probe.convert(probe.bounds, to: host)
                guard bounds.width > 0, bounds.minX >= -1, bounds.maxX <= width + 1,
                      bounds.minY >= -1, bounds.maxY <= host.bounds.height + 1 else { return false }
            }
        }
    }
    return true
}

extension ModelsView {
    static func catalogWindowFixture(engine: EngineController, navigation: ModelManagementState,
                                     selection: CatalogQuantizationFilter) -> Self {
        var result = Self(engine: engine, navigation: navigation, onDone: {}, onRun: { _ in })
        result._selectedFamily = State(initialValue: ModelsInventory.Family(id: "qwen3_8", name: "Qwen3.8 Flash Next"))
        result._catalogQuantization = State(initialValue: selection)
        return result
    }
}

@MainActor
func exerciseCatalogFamilyLayout() -> Bool {
    _ = NSApplication.shared
    let suite = "stark.catalog-window-test." + UUID().uuidString
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
    let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults)
    let labels = ["Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed)", "MLX mixed 4/8-bit", "MLX-Serve mixed 4/8-bit"]
    var items: [[String: Any]] = labels.enumerated().map { index, label in
        ["id": "fixture-\(index)", "repo": "fixture/Qwen3.8-Flash-Next-\(index)",
         "revision": String(repeating: "a", count: 40), "name": "Qwen3.8-Flash-Next-\(index)",
         "quantization": label, "artifact_modes": ["text", "vision-language"], "default_artifact_mode": "text",
         "weight_bytes": Int64(106_910_000_000), "downloadable": true, "installed_ids": [String]()]
    }
    items[2]["downloadable"] = false
    items[2]["unavailable_reason"] = "Expert weights use 3-bit packing with groups of 128. Stark’s loader requires 4-bit packing with groups of 64 for these weights."
    let report: [String: Any] = ["schema_version": 1, "family_id": "qwen3_8", "status": "ready",
        "message": "", "items": items, "has_more": true, "offset": 0, "next_offset": 3]
    let navigation = ModelManagementState()
    navigation.tab = .download
    for selection in [.all] + labels.map(CatalogQuantizationFilter.quantization) {
        engine.configureCatalogWindowFixture(report)
        guard engine.catalog?.items.count == 3 else { return false }
        let content = ModelsView.catalogWindowFixture(engine: engine, navigation: navigation, selection: selection)
        let host = NSHostingView(rootView: content)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 640, height: 580),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        host.layoutSubtreeIfNeeded()
        let probe = content.windowFixtureBoundsProbe
        let bounds = probe.convert(probe.bounds, to: host)
        guard bounds.width > 0, bounds.minX >= 23, bounds.maxX <= 617,
              bounds.minY >= 23, bounds.maxY <= 557 else { return false }
    }
    return true
}

@MainActor
func exerciseRunningModelNameLayout() -> Bool {
    _ = NSApplication.shared
    let suite = "stark.running-model-name-window-test." + UUID().uuidString
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
    let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults)
    let name = "Qwen3.8 Flash Next"
    let quantization = "Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed)"
    let display = InstalledModelDisplay([
        "schema_version": 1, "group_id": "repo:owner/Qwen3.8-Flash-Next-lily-q4-xl",
        "group_name": "Qwen3.8-Flash-Next-lily-q4-xl", "family_id": "qwen4_exp",
        "quantization": quantization, "weight_bytes": Int64(106_910_000_000),
        "source_revision": String(repeating: "a", count: 40)
    ])!
    let model = InstalledModel(id: "fixture-qwen38", name: "qwen38-flash-next-lily-q4-xl-0f94842e",
        path: "/private/fixture-qwen38", apiID: "owner/Qwen3.8-Flash-Next-lily-q4-xl", display: display)
    engine.models = [model]
    engine.loadedID = model.id
    engine.state = "ready"
    let host = NSHostingView(rootView: RunningModelLabel(engine: engine).font(.caption))
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 350, height: 720),
                          styleMask: [.titled], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    defer { window.close(); engine.desktopConnections.shutdown() }
    host.layoutSubtreeIfNeeded()
    RunLoop.current.run(until: Date().addingTimeInterval(0.03))
    host.layoutSubtreeIfNeeded()

    let naturalWidth = host.fittingSize.width
    // The 350-point menu leaves 256 points after padding, artwork and spacing.
    return model.displayName == name && naturalWidth > 0 && naturalWidth <= 256
}

extension EngineController {
    func configureCatalogWindowFixture(_ report: [String: Any]) {
        catalog = FamilyModelCatalog(report)
    }

    func exerciseBackgroundWindowUpdates() {
        loadedID = "window-fixture"
        loadedMode = .chat
        state = "ready"
        apply(["event": "gpu_memory", "model_id": "window-fixture", "mode": "chat",
               "gpu_memory_bytes": 1_073_741_824])
        state = "generating"
        apply(["event": "gpu_memory", "model_id": "window-fixture", "mode": "chat",
               "gpu_memory_bytes": 2_147_483_648])
        state = "ready"
        message = "Ready"
    }

    func configureConnectionFixture(state nextState: String, loadedMode nextMode: PlaygroundMode?,
                                    apiAvailable: Bool, hasModels: Bool) {
        models = hasModels ? [
            InstalledModel(id: "window-fixture", name: "Window fixture chat model",
                           path: "/window-fixture", apiID: "fixture/chat-model"),
            InstalledModel(id: "window-fixture-other", name: "Another saved model",
                           path: "/window-fixture-other",
                           apiID: "fixture/a-long-saved-model-identifier-for-layout-coverage")
        ] : []
        selectedID = hasModels ? "window-fixture" : ""
        loadedID = nextMode == nil ? "" : "window-fixture"
        mode = nextMode ?? .chat
        loadedMode = nextMode
        state = nextState
        message = nextState.capitalized
        apiBaseURL = apiAvailable ? "http://127.0.0.1:8000/v1" : ""
        apiKey = apiAvailable ? String(repeating: "a", count: 64) : ""
        apiMessage = apiAvailable ? "Local API is available on this Mac."
            : "The saved API port is occupied. Close the conflicting app or choose another port."
        supportsTools = nextMode == .chat && !["idle", "error", "loading", "stopping"].contains(nextState)
        apiContextTokens = 32_768
        apiMaxNewTokens = 512
    }

    func switchConnectionFixtureToOtherModel() {
        selectedID = "window-fixture-other"
        loadedID = selectedID
    }

    func configureAdvancedOptionsFixture(supported: Bool) {
        configureConnectionFixture(state: "ready", loadedMode: .chat, apiAvailable: false, hasModels: true)
        contextLimit = 8192; contextCapacityKnown = true; outputLimit = 4096
        options = ChatOptions.defaults(contextLimit: contextLimit, outputLimit: outputLimit)
        supportsMTP = supported; mtpEnabled = supported
    }
}

extension AppDelegate {
    static func exerciseParentNavigation() -> Bool {
        _ = NSApplication.shared
        let suite = "stark.parent-window-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
        let delegate = AppDelegate(engine: EngineController(apiCredentials: credentials, apiDefaults: defaults))
        let setup = delegate.window(title: "Setup fixture", view: EmptyView(), resizable: true)
        let chat = delegate.window(title: "Chat fixture", view: EmptyView(), resizable: true)
        let models = delegate.window(title: "Models fixture", view: EmptyView(), resizable: false)
        defer { setup.close(); chat.close(); models.close() }
        guard setup.delegate === delegate, chat.delegate === delegate, models.delegate === delegate else { return false }
        delegate.rememberWindowParent(setup, from: delegate.menuPanel)
        delegate.rememberWindowParent(chat, from: setup)
        delegate.rememberWindowParent(models, from: chat)
        guard delegate.removeWindowParent(of: models) === chat else { return false }
        // Reopening an existing ancestor from its child must not create a cycle.
        delegate.rememberWindowParent(setup, from: chat)
        guard delegate.removeWindowParent(of: setup) === chat,
              delegate.removeWindowParent(of: chat) === delegate.menuPanel else { return false }
        delegate.rememberWindowParent(setup, from: delegate.menuPanel)
        delegate.rememberWindowParent(chat, from: setup)
        delegate.rememberWindowParent(models, from: chat)
        // Closing a parent first redirects its surviving child to the ancestor.
        guard delegate.removeWindowParent(of: chat) === setup,
              delegate.removeWindowParent(of: models) === setup,
              delegate.removeWindowParent(of: setup) === delegate.menuPanel,
              delegate.removeWindowParent(of: setup) == nil,
              delegate.windowParents.isEmpty else { return false }
        return !setup.isVisible && !chat.isVisible && !models.isVisible && !delegate.menuPanel.isVisible
    }

    static func exerciseAdvancedOptionsLayouts() -> Bool {
        let suite = "stark.advanced-window-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
        let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults,
                                      gpuMemoryBudgetBytes: 8_589_934_592)
        let delegate = AppDelegate(engine: engine)
        for supported in [false, true] {
            engine.configureAdvancedOptionsFixture(supported: supported)
            let window = delegate.window(title: "Advanced Options", view: AdvancedOptionsView(engine: engine), resizable: false)
            guard let content = window.contentView else { return false }
            content.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
            content.layoutSubtreeIfNeeded()
            let size = content.fittingSize
            guard !window.isVisible, size.width >= 500, size.width <= 540,
                  size.height > 350, size.height < 800 else { return false }
            window.close()
        }
        return true
    }

    static func exerciseWindowLifecycle() -> Bool {
        let application = NSApplication.shared
        let suite = "stark.window-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let storage = ForbiddenWindowCredentialStorage()
        let credentials = APICredentialStore(service: suite, storage: storage)
        let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults,
                                      gpuMemoryBudgetBytes: 8_589_934_592)
        let delegate = AppDelegate(engine: engine, interface: StarkInterfacePreferences(defaults: defaults))
        // Reopen may arrive before the status item exists. It must stay quiet.
        guard !delegate.applicationShouldHandleReopen(application, hasVisibleWindows: false),
              delegate.promptWindow == nil, delegate.connectionWindow == nil,
              !delegate.menuPanel.isVisible else { return false }

        let prompt = delegate.window(title: "Window fixture", view: EmptyView(), resizable: true)
        let connection = delegate.window(title: "Connection fixture", view: EmptyView(), resizable: false)
        delegate.promptWindow = prompt
        delegate.connectionWindow = connection
        guard !prompt.isVisible, !connection.isVisible,
              !prompt.isRestorable, !connection.isRestorable,
              prompt.styleMask.contains(.resizable), !connection.styleMask.contains(.resizable) else { return false }
        // Returning false suppresses AppKit's default reopen of existing windows.
        for visible in [false, true] {
            guard !delegate.applicationShouldHandleReopen(application, hasVisibleWindows: visible),
                  !prompt.isVisible, !connection.isVisible else { return false }
        }
        prompt.close()
        connection.close()
        let modeConnection = ConnectionModeWindowProbe(contentRect: .zero, styleMask: [.titled], backing: .buffered, defer: false)
        delegate.connectionWindow = modeConnection
        delegate.interface.select(.beginner)
        delegate.interfaceDidChange()
        guard modeConnection.hiddenForBeginner, !modeConnection.isVisible,
              delegate.beginnerWindow == nil, delegate.beginnerChatWindow == nil,
              !delegate.menuPanel.isVisible else { return false }
        delegate.interface.select(.advanced)
        delegate.interfaceDidChange()
        guard !modeConnection.isVisible, delegate.modelsWindow == nil else { return false }
        modeConnection.close()
        for _ in 0..<20 { engine.exerciseBackgroundWindowUpdates() }
        RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        return !prompt.isVisible && !connection.isVisible && !delegate.menuPanel.isVisible
            && engine.gpuMemoryBytes == 2_147_483_648
    }

    static func exerciseConnectionLayouts() -> Bool {
        _ = NSApplication.shared
        let suite = "stark.connection-layout-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
        let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults,
                                      gpuMemoryBudgetBytes: 8_589_934_592)
        let delegate = AppDelegate(engine: engine)
        let scenarios: [(String, PlaygroundMode?, Bool, Bool, Bool)] = [
            ("idle", nil, false, false, false),
            ("idle", nil, true, true, false),
            ("loading", .chat, true, true, false),
            ("ready", .chat, true, true, true),
            ("generating", .chat, true, true, true),
            ("installing", .chat, true, true, true),
            ("configuring", .chat, true, true, true),
            ("removing", .chat, true, true, true),
            ("stopping", .chat, true, true, false),
            ("error", .chat, false, true, false),
            ("ready", .decision, true, true, false)
        ]
        for (state, mode, apiAvailable, hasModels, servesChat) in scenarios {
            engine.configureConnectionFixture(state: state, loadedMode: mode,
                                              apiAvailable: apiAvailable, hasModels: hasModels)
            for (otherApps, advanced, revealKey) in [(false, false, false), (true, false, false), (true, true, true)] {
                let view = ConnectionDetailsView.windowFixture(engine: engine, otherApps: otherApps,
                                                               advanced: advanced, revealKey: revealKey)
                guard view.windowFixtureServesChat == servesChat,
                      view.windowFixtureExportMatches(servesChat: servesChat, selectedMatches: servesChat),
                      view.windowFixtureInspectedID == (hasModels ? "fixture/chat-model" : "") else { return false }
                let window = delegate.window(title: "Stark — Connections", view: view, resizable: false)
                guard let content = window.contentView else { return false }
                content.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date().addingTimeInterval(0.01))
                content.layoutSubtreeIfNeeded()
                let fitting = content.fittingSize
                let size = content.frame.size
                guard !window.isVisible, !window.isRestorable,
                      !window.styleMask.contains(.resizable),
                      fitting.width.isFinite, fitting.height.isFinite,
                      abs(fitting.width - 570) < 1, abs(fitting.height - 680) < 1,
                      abs(size.width - 570) < 1, abs(size.height - 680) < 1 else { return false }
                window.close()
            }
            // Inspecting a different saved model must not change the running model.
            if hasModels {
                let inspected = ConnectionDetailsView.windowFixture(engine: engine, otherApps: true,
                    advanced: false, revealKey: false, inspectedModel: "window-fixture-other")
                guard inspected.windowFixtureInspectedID == "fixture/a-long-saved-model-identifier-for-layout-coverage",
                      inspected.windowFixtureServesChat == servesChat,
                      inspected.windowFixtureExportMatches(servesChat: servesChat, selectedMatches: false),
                      engine.loadedID == (mode == nil ? "" : "window-fixture") else { return false }
                let window = delegate.window(title: "Stark — Connections", view: inspected, resizable: false)
                window.contentView?.layoutSubtreeIfNeeded()
                guard !window.isVisible, window.contentView?.fittingSize.width == 570,
                      window.contentView?.fittingSize.height == 680 else { return false }
                window.close()
            }
        }
        engine.desktopConnections.shutdown()
        return !delegate.menuPanel.isVisible
    }

    static func exerciseCurrentModelDiscovery() -> Bool {
        let suite = "stark.beginner-discovery-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let credentials = APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage())
        let engine = EngineController(apiCredentials: credentials, apiDefaults: defaults)
        engine.configureConnectionFixture(state: "ready", loadedMode: .chat, apiAvailable: true, hasModels: true)
        let observation = BeginnerOpenWorkStatus.staleWindowFixture()
        let session = BeginnerSetupSession()
        let host = BeginnerSetupHost(engine: engine, openWork: engine.desktopConnections.openWork, openCode: engine.desktopConnections.openCode,
            session: session, interface: StarkInterfacePreferences(defaults: defaults),
            observation: observation, openCodeObservation: BeginnerOpenWorkStatus(client: .openCode), selectInterface: { _ in }, openChat: {})
        // A periodic observation from a previous model is still true while the
        // engine changes. It cannot replace a verified current-model handshake.
        engine.switchConnectionFixtureToOtherModel()
        let snapshot = host.snapshot
        session.selectTask(.coding)
        let codeSnapshot = host.snapshot
        guard codeSnapshot.client == .openCode, !codeSnapshot.openWorkRunning,
              !codeSnapshot.openWorkClientActive, codeSnapshot.openWorkPreparedModelID == nil else { return false }
        session.selectTask(.general)
        guard host.snapshot.openWorkRunning else { return false }
        engine.desktopConnections.shutdown()
        return observation.running && observation.liveConnection && snapshot.openWorkRunning
            && snapshot.runningModelID == "window-fixture-other" && !snapshot.openWorkClientActive
    }

    static func exerciseBeginnerInstalledModelIdentity() -> Bool {
        let suite = "stark.beginner-identity-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let engine = EngineController(apiCredentials: APICredentialStore(service: suite, storage: ForbiddenWindowCredentialStorage()),
                                      apiDefaults: defaults)
        engine.configureConnectionFixture(state: "ready", loadedMode: .chat, apiAvailable: true, hasModels: true)
        let host = BeginnerSetupHost(engine: engine, openWork: engine.desktopConnections.openWork, openCode: engine.desktopConnections.openCode,
            session: BeginnerSetupSession(), interface: StarkInterfacePreferences(defaults: defaults),
            observation: BeginnerOpenWorkStatus(), openCodeObservation: BeginnerOpenWorkStatus(client: .openCode), selectInterface: { _ in }, openChat: {})
        defer { engine.desktopConnections.shutdown() }
        // Reopening setup has no temporary recommendation report to supply a name.
        guard engine.beginnerReport == nil else { return false }
        for (family, group, expected) in [("qwen4_exp", "Qwen3.8-Flash-Next-lily-q4-xl", "Qwen3.8 Flash Next"),
                                          ("glm5_next", "GLM-5.3-Flash-GGUF", "GLM 5.3 Flash"),
                                          ("qwen3", "Qwen3-4B", "Qwen3 4B")] {
            let display = InstalledModelDisplay(["schema_version": 1, "group_id": "repo:fixture/Model",
                "group_name": group, "family_id": family, "quantization": "Fixture version"])
            guard let display else { return false }
            engine.models = [InstalledModel(id: "window-fixture", name: group + "-aaaaaaaa",
                path: "/window-fixture", apiID: "fixture/chat-model", display: display)]
            let snapshot = host.snapshot
            guard snapshot.runningModelName == expected, snapshot.runningModelVariantName == "Fixture version",
                  snapshot.runningModelID == "window-fixture", engine.activeModel?.apiID == "fixture/chat-model",
                  snapshot.runningModelRepositoryURL?.absoluteString == "https://huggingface.co/fixture/Model" else { return false }
        }
        engine.configureConnectionFixture(state: "ready", loadedMode: .chat, apiAvailable: true, hasModels: true)
        return host.snapshot.runningModelName == "Window fixture chat model" && host.snapshot.runningModelVariantName.isEmpty
    }

    static func exerciseApplicationScanThreading() -> Bool {
        _ = DesktopApplicationIdentity.candidates(.openWork)
        let finished = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async {
            _ = DesktopApplicationIdentity.running(.openWork, configurationRoot: nil, candidates: [])
            finished.signal()
        }
        return finished.wait(timeout: .now() + 5) == .success
    }
}

@main
struct WindowChecks {
    @MainActor static func main() {
        switch CommandLine.arguments.dropFirst().first {
        case "parents": exit(AppDelegate.exerciseParentNavigation() ? 0 : 1)
        case "setup-initial": exit(AppDelegate.exerciseBeginnerInitialLayout() ? 0 : 1)
        case "setup-footer": exit(AppDelegate.exerciseBeginnerSetupFooterLayouts() ? 0 : 1)
        case "chat-footer": exit(AppDelegate.exerciseBeginnerChatFooterLayouts() ? 0 : 1)
        case "lifecycle": exit(AppDelegate.exerciseWindowLifecycle() ? 0 : 1)
        case "connections": exit(AppDelegate.exerciseConnectionLayouts() ? 0 : 1)
        case "beginner-discovery": exit(AppDelegate.exerciseCurrentModelDiscovery() ? 0 : 1)
        case "beginner-identity": exit(AppDelegate.exerciseBeginnerInstalledModelIdentity() ? 0 : 1)
        case "beginner-queued-scan": exit(BeginnerOpenWorkStatus.queuedScanFixture() ? 0 : 1)
        case "application-scan-threading": exit(AppDelegate.exerciseApplicationScanThreading() ? 0 : 1)
        case "advanced-options": exit(AppDelegate.exerciseAdvancedOptionsLayouts() ? 0 : 1)
        case "catalog-picker": exit(exerciseCatalogPickerLayouts() ? 0 : 1)
        case "catalog-family": exit(exerciseCatalogFamilyLayout() ? 0 : 1)
        case "running-model-name": exit(exerciseRunningModelNameLayout() ? 0 : 1)
        default: exit(2)
        }
    }
}
"""

CONNECTION_FIXTURE = r"""

extension ConnectionDetailsView {
    static func windowFixture(engine: EngineController, otherApps: Bool, advanced: Bool,
                              revealKey: Bool, inspectedModel: String = "") -> Self {
        var result = Self(engine: engine, manage: {}, close: {})
        result._section = State(initialValue: otherApps ? .other : .apps)
        result._windowFixtureAdvancedExpanded = State(initialValue: advanced)
        result._revealKey = State(initialValue: revealKey)
        result._inspectedModelID = State(initialValue: inspectedModel)
        return result
    }

    var windowFixtureServesChat: Bool { !modelID.isEmpty }
    var windowFixtureInspectedID: String { displayedModelID }

    func windowFixtureExportMatches(servesChat: Bool, selectedMatches: Bool) -> Bool {
        guard let data = connectionInfo.json?.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              json["selected_model_is_running"] as? Bool == selectedMatches else { return false }
        return servesChat ? json["running_model"] as? String == "fixture/chat-model"
            : json["running_model"] is NSNull
    }
}
"""

BEGINNER_CONNECTION_FIXTURE = r"""

@MainActor
final class PendingScanCallbackFixture {
    var currentCount = 0
    var pendingCount = 0
    var staleCount = 0
    var currentValue: Bool?
    var pendingValue: Bool?
}

extension BeginnerOpenWorkStatus {
    static func staleWindowFixture() -> BeginnerOpenWorkStatus {
        let result = BeginnerOpenWorkStatus()
        result.running = true
        result.liveConnection = true
        return result
    }

    static func queuedScanFixture() -> Bool {
        let status = BeginnerOpenWorkStatus()
        let callbacks = PendingScanCallbackFixture()
        let activeID = UUID()
        status.refreshID = activeID
        status.refreshInFlight = true
        let input = ScanInput(previousURL: nil, shouldResolveApplication: false, profile: nil,
            candidates: [], hasDiscoveredModel: false, discoveryScope: "pending-scope",
            brokerPath: "pending-broker", apiBaseURL: "pending-api")
        status.pendingRefresh = (input, [{ value in
            callbacks.pendingCount += 1
            callbacks.pendingValue = value
        }])
        let initialScope = status.scope
        let stale = ScanResult(applicationURL: URL(fileURLWithPath: "/stale"), applicationVerified: true,
            running: true, liveConnection: true, verifiedClientBasis: "stale",
            connectionBasis: "stale")
        status.apply(stale, requestID: UUID(), completions: [{ _ in callbacks.staleCount += 1 }])
        guard status.applicationURL == nil, !status.running, callbacks.staleCount == 0 else { return false }

        status.apply(stale, requestID: activeID, completions: [{ value in
            callbacks.currentCount += 1
            callbacks.currentValue = value
        }])
        guard status.applicationURL == nil, !status.running,
              status.refreshInFlight, status.refreshID != activeID else { return false }
        status.apply(stale, requestID: activeID, completions: [{ _ in callbacks.staleCount += 1 }])
        guard status.applicationURL == nil, !status.running, callbacks.staleCount == 0 else { return false }

        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline && (status.refreshInFlight || callbacks.currentCount != 1 || callbacks.pendingCount != 1) {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        return !status.refreshInFlight && status.applicationURL == nil && !status.running
            && status.scope == initialScope && callbacks.currentCount == 1 && callbacks.pendingCount == 1
            && callbacks.staleCount == 0 && callbacks.currentValue == false && callbacks.pendingValue == false
    }
}
"""


@unittest.skipUnless(platform.system() == "Darwin", "macOS AppKit checks")
class WindowLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stark-window-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        sources = ROOT / "apps/menu-bar/Sources"
        source = (sources / "StarkApp.swift").read_text()
        if source.count(BOUNDARY) != 1:
            raise RuntimeError("window_source_boundary_invalid")
        main = directory / "WindowChecks.swift"
        # Preserve the selected family in this disposable copy and measure the
        # inner stack, whose minimum width originally pushed controls offscreen.
        family_reset = ".onAppear { selectedFamily = nil; catalogQuantization = .all }"
        models_frame = "}.padding(24).frame(width: 640, height: 580)"
        models_declaration = "struct ModelsView: View {"
        if any(source.count(boundary) != 1 for boundary in [family_reset, models_frame, models_declaration]):
            raise RuntimeError("catalog_fixture_source_boundary_invalid")
        source = source.replace(family_reset, ".onAppear { }").replace(
            models_declaration, models_declaration + "\n    let windowFixtureBoundsProbe = NSView()"
        ).replace(models_frame, "}.background(CatalogPickerBoundsProbe(view: windowFixtureBoundsProbe)).padding(24).frame(width: 640, height: 580)")
        main.write_text(source.split(BOUNDARY, 1)[0] + DRIVER)

        # Bind the existing disclosure only in this disposable source copy, so
        # tests can open Advanced settings without test APIs in the app or UI input.
        connection_source = (sources / "ConnectionDetailsView.swift").read_text()
        declaration = "    @State private var section = Section.apps\n"
        disclosure = 'DisclosureGroup("Advanced settings") {'
        if (
            connection_source.count(declaration) != 1
            or connection_source.count(disclosure) != 1
        ):
            raise RuntimeError("connection_fixture_source_boundary_invalid")
        connection_source = connection_source.replace(
            declaration,
            declaration
            + "    @State private var windowFixtureAdvancedExpanded = false\n",
        ).replace(
            disclosure,
            'DisclosureGroup("Advanced settings", isExpanded: $windowFixtureAdvancedExpanded) {',
        )
        connection = directory / "ConnectionDetailsView.swift"
        connection.write_text(connection_source + CONNECTION_FIXTURE)
        beginner_host = directory / "BeginnerSetupHost.swift"
        beginner_source = (sources / "BeginnerSetupHost.swift").read_text()
        chat_declaration = "struct BeginnerChatView: View {"
        footer_end = 'Button("Try a different model", action: tryAnotherModel).disabled(!engine.modelReady)\n                }.font(.caption)'
        if beginner_source.count(chat_declaration) != 1 or beginner_source.count(footer_end) != 1:
            raise RuntimeError("chat_footer_fixture_source_boundary_invalid")
        beginner_source = beginner_source.replace(
            chat_declaration, chat_declaration + "\n    let windowFixtureFooterProbe = NSView()"
        ).replace(footer_end, footer_end + ".background(CatalogPickerBoundsProbe(view: windowFixtureFooterProbe))")
        beginner_host.write_text(
            beginner_source
            + BEGINNER_CONNECTION_FIXTURE
        )
        setup_view = directory / "BeginnerSetupView.swift"
        setup_source = (sources / "BeginnerSetupView.swift").read_text()
        for boundary in ["    private var priorityStep: some View {", "    // Keep the next step visible"]:
            form_end = "        }\n    }\n\n" + boundary
            if setup_source.count(form_end) != 1:
                raise RuntimeError("setup_form_fixture_source_boundary_invalid")
            setup_source = setup_source.replace(form_end,
                "        }.background(CatalogPickerBoundsProbe(view: SetupLayoutProbes.form))\n    }\n\n" + boundary)
        action_end = "        }\n    }\n\n    @ViewBuilder private var recommendationStep"
        if setup_source.count(action_end) != 1:
            raise RuntimeError("setup_footer_fixture_source_boundary_invalid")
        setup_view.write_text(setup_source.replace(action_end,
            "        }.background(CatalogPickerBoundsProbe(view: SetupLayoutProbes.action))\n    }\n\n    @ViewBuilder private var recommendationStep"))
        cls.executable = directory / "window-checks"
        compiled = subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-parse-as-library",
                "-target",
                "arm64-apple-macosx26.4",
                str(main),
                str(connection),
                str(beginner_host),
                str(setup_view),
                *(
                    str(path)
                    for path in sorted(sources.glob("*.swift"))
                    if path.name
                    not in {
                        "StarkApp.swift",
                        "ConnectionDetailsView.swift",
                        "BeginnerSetupHost.swift",
                        "BeginnerSetupView.swift",
                    }
                ),
                "-o",
                str(cls.executable),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        if compiled.returncode:
            raise RuntimeError("window_check_compilation_failed")

    def run_check(self, scenario: str):
        result = subprocess.run(
            [str(self.executable), scenario],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"window_{scenario}_check_failed")

    def test_reopen_and_background_updates_do_not_present_retained_windows(self):
        self.run_check("lifecycle")

    def test_beginner_setup_action_stays_visible_in_short_windows(self):
        self.run_check("setup-footer")

    def test_beginner_initial_window_shows_entire_task_form(self):
        self.run_check("setup-initial")

    def test_beginner_chat_footer_keeps_bottom_padding_at_short_and_tall_sizes(self):
        self.run_check("chat-footer")

    def test_nested_window_returns_and_reopened_ancestors_do_not_form_cycles(self):
        self.run_check("parents")

    def test_connection_sections_fit_across_model_lifecycle_states(self):
        self.run_check("connections")

    def test_beginner_snapshot_rejects_cached_discovery_after_model_switch(self):
        self.run_check("beginner-discovery")

    def test_reopened_beginner_guide_uses_installed_name_and_version(self):
        self.run_check("beginner-identity")

    def test_forced_scan_queues_callbacks_and_discards_stale_result(self):
        self.run_check("beginner-queued-scan")

    def test_application_enumeration_and_verification_use_separate_queues(self):
        self.run_check("application-scan-threading")

    def test_advanced_options_fit_with_and_without_mtp_support(self):
        self.run_check("advanced-options")

    def test_catalog_filters_fit_with_long_labels_and_each_selected_value(self):
        self.run_check("catalog-picker")

    def test_qwen_family_content_stays_inside_manage_models_padding(self):
        self.run_check("catalog-family")

    def test_menu_running_model_name_fits_without_variant_metadata(self):
        self.run_check("running-model-name")


if __name__ == "__main__":
    unittest.main()
