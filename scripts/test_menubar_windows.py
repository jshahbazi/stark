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

extension EngineController {
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
        let delegate = AppDelegate(engine: engine)
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
        let host = BeginnerSetupHost(engine: engine, openWork: engine.desktopConnections.openWork,
            session: BeginnerSetupSession(), interface: StarkInterfacePreferences(defaults: defaults),
            observation: observation, selectInterface: { _ in }, openChat: {})
        // A periodic observation from a previous model is still true while the
        // engine changes. It cannot replace a verified current-model handshake.
        engine.switchConnectionFixtureToOtherModel()
        let snapshot = host.snapshot
        engine.desktopConnections.shutdown()
        return observation.running && observation.liveConnection && snapshot.openWorkRunning
            && snapshot.runningModelID == "window-fixture-other" && !snapshot.openWorkClientActive
    }
}

@main
struct WindowChecks {
    @MainActor static func main() {
        switch CommandLine.arguments.dropFirst().first {
        case "lifecycle": exit(AppDelegate.exerciseWindowLifecycle() ? 0 : 1)
        case "connections": exit(AppDelegate.exerciseConnectionLayouts() ? 0 : 1)
        case "beginner-discovery": exit(AppDelegate.exerciseCurrentModelDiscovery() ? 0 : 1)
        case "advanced-options": exit(AppDelegate.exerciseAdvancedOptionsLayouts() ? 0 : 1)
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

extension BeginnerOpenWorkStatus {
    static func staleWindowFixture() -> BeginnerOpenWorkStatus {
        let result = BeginnerOpenWorkStatus()
        result.running = true
        result.liveConnection = true
        return result
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
        beginner_host.write_text(
            (sources / "BeginnerSetupHost.swift").read_text()
            + BEGINNER_CONNECTION_FIXTURE
        )
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
                *(
                    str(path)
                    for path in sorted(sources.glob("*.swift"))
                    if path.name
                    not in {
                        "StarkApp.swift",
                        "ConnectionDetailsView.swift",
                        "BeginnerSetupHost.swift",
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

    def test_connection_sections_fit_across_model_lifecycle_states(self):
        self.run_check("connections")

    def test_beginner_snapshot_rejects_cached_discovery_after_model_switch(self):
        self.run_check("beginner-discovery")

    def test_advanced_options_fit_with_and_without_mtp_support(self):
        self.run_check("advanced-options")


if __name__ == "__main__":
    unittest.main()
