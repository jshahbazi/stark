"""Native checks of explicit window presentation, without starting a backend.

The executable includes the production delegate and views. It creates only
hidden windows, uses injected credential storage that fails on every access,
and never calls the application-launch callback or presents test UI.
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
}

extension AppDelegate {
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
}

@main
struct WindowChecks {
    @MainActor static func main() {
        exit(AppDelegate.exerciseWindowLifecycle() ? 0 : 1)
    }
}
"""


@unittest.skipUnless(platform.system() == "Darwin", "macOS AppKit checks")
class WindowLifecycleTests(unittest.TestCase):
    def test_reopen_and_background_updates_do_not_present_retained_windows(self):
        with tempfile.TemporaryDirectory(prefix="stark-window-test-") as temporary:
            directory = Path(temporary)
            sources = ROOT / "apps/menu-bar/Sources"
            source = (sources / "StarkApp.swift").read_text()
            if source.count(BOUNDARY) != 1:
                raise RuntimeError("window_source_boundary_invalid")
            main = directory / "WindowChecks.swift"
            main.write_text(source.split(BOUNDARY, 1)[0] + DRIVER)
            executable = directory / "window-checks"
            compiled = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-parse-as-library",
                    "-target",
                    "arm64-apple-macosx26.4",
                    str(main),
                    *(
                        str(path)
                        for path in sorted(sources.glob("*.swift"))
                        if path.name != "StarkApp.swift"
                    ),
                    "-o",
                    str(executable),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, "window_check_compilation_failed")
            result = subprocess.run(
                [str(executable)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            self.assertEqual(result.returncode, 0, "window_lifecycle_check_failed")


if __name__ == "__main__":
    unittest.main()
