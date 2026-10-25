import AppKit

enum InstallerError: Error, LocalizedError {
    case source, prepare, install

    var errorDescription: String? {
        switch self {
        case .source: return "The installer does not contain a compatible Stark release. Download it again."
        case .prepare: return "Stark could not prepare the installation. Your current app was kept."
        case .install: return "Stark could not complete the installation. Try again."
        }
    }
}

struct InstallerResult: Decodable {
    let path: String
    let version: String
}

struct InstallerFailure: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum InstallerFiles {
    static func createWorkspace() throws -> URL {
        do { return try AppUpdateFiles.createWorkspace(prefix: "stark-install-") }
        catch { throw InstallerError.prepare }
    }

    static func helperResult(status: Int32, output: Data, workspace: URL) throws -> InstallerResult {
        if status != 0 {
            let recorded = try? Data(contentsOf: workspace.appendingPathComponent("result.json"))
            // Workspace validation can fail before result.json may be written.
            // The verified helper also returns these fixed errors over stdout.
            for data in [recorded, output] {
                if let data, data.count <= 16_384,
                   let failure = try? JSONDecoder().decode([String: String].self, from: data),
                   let message = failure["error"] { throw InstallerFailure(message: message) }
            }
            throw InstallerError.install
        }
        guard output.count <= 16_384,
              let installed = try? JSONDecoder().decode(InstallerResult.self, from: output) else {
            throw InstallerError.install
        }
        return installed
    }

    static func install(bundle: URL) throws -> InstallerResult {
        let manager = FileManager.default
        // Keep the payload inside the signed installer. Gatekeeper may
        // translocate this app, making disk-image siblings inaccessible.
        let source = bundle.appendingPathComponent("Contents/Helpers/Stark.app")
        let metadata = source.appendingPathComponent("Contents/Info.plist")
        guard source == source.resolvingSymlinksInPath(),
              let data = try? Data(contentsOf: metadata),
              let info = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
              let version = info["CFBundleShortVersionString"] as? String,
              ReleaseVersion(version) != nil else { throw InstallerError.source }
        // Verify before executing any interpreter or helper from the payload.
        try AppUpdateFiles.verifySignature(source, app: true)
        try AppUpdateFiles.verifyMetadata(source, version: version)
        let workspace = try createWorkspace()
        defer { try? manager.removeItem(at: workspace) }
        let candidate = workspace.appendingPathComponent("Stark.app")
        try manager.copyItem(at: source, to: candidate)
        try AppUpdateFiles.verifySignature(candidate, app: true)
        try AppUpdateFiles.verifyMetadata(candidate, version: version)
        for name in ["update_app.py", "install_app.py"] {
            try manager.copyItem(at: bundle.appendingPathComponent("Contents/Resources/" + name),
                                 to: workspace.appendingPathComponent(name))
        }
        let process = Process()
        process.executableURL = candidate.appendingPathComponent("Contents/Resources/runtime/libexec/python/bin/python3.13")
        process.arguments = ["-I", "-B", workspace.appendingPathComponent("update_app.py").path,
                             "--candidate", candidate.path, "--workspace", workspace.path, "--install-from-dmg"]
        process.environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": manager.homeDirectoryForCurrentUser.path,
                               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "en_US.UTF-8"]
        process.standardInput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        let output = Pipe()
        process.standardOutput = output
        try process.run()
        let result = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return try helperResult(status: process.terminationStatus, output: result, workspace: workspace)
    }
}

@MainActor
final class InstallerDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    private var installing = true

    func applicationDidFinishLaunching(_ notification: Notification) {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 420, height: 140),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.title = "Install Stark"
        let title = NSTextField(labelWithString: "Installing Stark…")
        title.font = .boldSystemFont(ofSize: 17)
        let detail = NSTextField(wrappingLabelWithString: "Stark will restart when installation is complete.")
        let spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.startAnimation(nil)
        let stack = NSStackView(views: [title, detail, spinner])
        stack.orientation = .vertical
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        window.contentView?.addSubview(stack)
        if let content = window.contentView {
            NSLayoutConstraint.activate([
                stack.centerXAnchor.constraint(equalTo: content.centerXAnchor),
                stack.centerYAnchor.constraint(equalTo: content.centerYAnchor),
                stack.widthAnchor.constraint(equalToConstant: 370)
            ])
        }
        window.center()
        window.makeKeyAndOrderFront(nil)
        self.window = window
        NSApplication.shared.activate(ignoringOtherApps: true)
        let bundle = Bundle.main.bundleURL
        Task {
            do {
                let result = try await Task.detached { try InstallerFiles.install(bundle: bundle) }.value
                finish(title: "Stark \(result.version) is installed and launched.",
                       detail: "Installed at \(result.path)", failed: false)
            } catch {
                let message = (error as? LocalizedError)?.errorDescription ?? InstallerError.prepare.localizedDescription
                finish(title: "Stark installation failed", detail: message, failed: true)
            }
        }
    }

    private func finish(title: String, detail: String, failed: Bool) {
        installing = false
        window?.orderOut(nil)
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = detail
        alert.alertStyle = failed ? .warning : .informational
        alert.addButton(withTitle: "OK")
        NSApplication.shared.activate(ignoringOtherApps: true)
        alert.runModal()
        NSApplication.shared.terminate(nil)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        installing ? .terminateCancel : .terminateNow
    }
}

#if !STARK_INSTALLER_LIBRARY
@main
struct StarkInstallerApplication {
    @MainActor
    static func main() {
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        let delegate = InstallerDelegate()
        application.delegate = delegate
        application.run()
        withExtendedLifetime(delegate) {}
    }
}
#endif
