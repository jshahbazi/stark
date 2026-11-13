import AppKit
import Combine
import Foundation
import Darwin

/// Manual Pi setup uses a separate public profile. The master key is passed to
/// the setup helper only in its environment, then explicitly copied for the
/// user's hidden Terminal prompt. No managed capability or approval is issued.
final class PiConnectionController: ObservableObject {
    @Published private(set) var busy = false
    @Published private(set) var installed = false
    @Published private(set) var message = "Checking Pi installation…"
    @Published private(set) var profile: String?
    @Published private(set) var configuredModel = ""
    @Published private(set) var configuredBaseURL = ""

    struct Tools {
        let node: URL
        let pi: URL
    }
    typealias Completion = ([String: Any]) -> Void
    typealias Runner = (URL, [String], [String: String], @escaping Completion) -> Void
    private let resources: URL
    private let root: URL
    private let defaults: UserDefaults
    private let runner: Runner
    private var tools: Tools?
    private var refreshed = false
    private static let profilePreference = "stark.pi.manualProfile.v1"

    init(resources: URL = Bundle.main.resourceURL?.appendingPathComponent("PiClient") ?? URL(fileURLWithPath: "/nonexistent"),
         root: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Stark/Pi", isDirectory: true),
         defaults: UserDefaults = .standard, tools: Tools? = nil,
         runner: @escaping Runner = PiConnectionController.runHelper) {
        self.resources = resources; self.root = root; self.defaults = defaults
        self.tools = tools; self.runner = runner
    }

    func matches(baseURL: String, modelID: String) -> Bool {
        profile != nil && configuredBaseURL == baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) &&
            !modelID.isEmpty && configuredModel == modelID
    }

    func refresh(force: Bool = false) {
        guard !busy, force || !refreshed else { return }
        refreshed = true
        guard FileManager.default.fileExists(atPath: resources.appendingPathComponent("src/desktop-setup.mjs").path) else {
            message = "Pi setup is unavailable in this build."; return
        }
        tools = tools ?? Self.findTools()
        guard tools != nil else { installed = false; message = "Install Pi 1.5.2 and Node.js 22.19 or newer, then check again."; return }
        var arguments = ["--action", "inspect", "--pi", tools!.pi.path]
        if let saved = defaults.string(forKey: Self.profilePreference), ownedProfilePath(saved) {
            arguments += ["--profile", saved]
        }
        perform(arguments) { [weak self] result in
            guard let self else { return }
            if result["ok"] as? Bool == true {
                self.installed = true
                self.acceptProfile(result)
                self.message = self.profile == nil ? "Pi 1.5.2 found. Start a model with tool support to set it up." : "Setup saved for \(self.configuredModel)."
            } else {
                self.installed = false
                self.message = Self.failure(result)
                // A missing/changed old profile must not prevent repairing setup.
                if arguments.contains("--profile") {
                    self.profile = nil
                    self.defaults.removeObject(forKey: Self.profilePreference)
                    self.refreshed = false
                    self.refresh()
                }
            }
        }
    }

    func setup(baseURL: String, modelID: String, key: String) {
        guard !busy, installed, tools != nil, !baseURL.isEmpty, !modelID.isEmpty, !key.isEmpty else { return }
        perform(["--action", "setup", "--pi", tools!.pi.path, "--root", root.path,
                 "--base-url", baseURL, "--model", modelID], key: key) { [weak self] result in
            guard let self else { return }
            guard result["ok"] as? Bool == true, let path = result["profile"] as? String,
                  self.ownedProfilePath(path) else { self.message = Self.failure(result); return }
            self.acceptProfile(result)
            self.defaults.set(path, forKey: Self.profilePreference)
            self.message = "Setup saved. Choose a folder to open Pi. Paste the copied key at its hidden prompt."
        }
    }

    func open(baseURL: String, modelID: String, key: String) {
        guard !busy, installed, matches(baseURL: baseURL, modelID: modelID), !key.isEmpty else { return }
        let picker = NSOpenPanel()
        picker.title = "Choose a folder for Pi"
        picker.message = "Pi can read and edit files and run commands in this folder."
        picker.prompt = "Open Pi"
        picker.canChooseDirectories = true; picker.canChooseFiles = false
        picker.allowsMultipleSelection = false
        guard picker.runModal() == .OK, let workspace = picker.url else { return }
        prepareLaunch(workspace: workspace, key: key)
    }

    /// Kept separate from the folder picker so tests can exercise the real helper
    /// boundary with a disposable profile without presenting or launching Terminal.
    func prepareLaunch(workspace: URL, key: String,
                       openTerminal: @escaping (URL, @escaping (Bool) -> Void) -> Void = PiConnectionController.openTerminal,
                       copyKey: @escaping (String) -> Bool = PiConnectionController.copyKey) {
        guard !busy, let profile, !key.isEmpty else { return }
        perform(["--action", "prepare-launch", "--profile", profile, "--workspace", workspace.path]) { [weak self] result in
            guard let self else { return }
            guard result["ok"] as? Bool == true, let command = result["command"] as? String,
                  URL(fileURLWithPath: command).deletingLastPathComponent().path == profile,
                  command.hasSuffix(".command"), copyKey(key) else {
                self.message = Self.failure(result); return
            }
            openTerminal(URL(fileURLWithPath: command)) { [weak self] opened in
                self?.message = opened ? "Paste the copied API key in Terminal. It is hidden and is not saved in the setup files."
                    : "Could not open Terminal. Check Pi setup and try again."
            }
        }
    }

    private func acceptProfile(_ result: [String: Any]) {
        guard let path = result["profile"] as? String, ownedProfilePath(path),
              let model = result["modelId"] as? String, let base = result["baseUrl"] as? String else { return }
        profile = path; configuredModel = model; configuredBaseURL = base
    }

    private func ownedProfilePath(_ path: String) -> Bool {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        return url.deletingLastPathComponent().path == root.standardizedFileURL.path && url.lastPathComponent.hasPrefix("profile-")
    }

    private func perform(_ arguments: [String], key: String? = nil, completion: @escaping Completion) {
        guard let tools else { return }
        busy = true
        var environment = ["HOME": NSHomeDirectory(), "PATH": [tools.node.deletingLastPathComponent().path,
            NSHomeDirectory() + "/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"].joined(separator: ":")]
        if let key { environment["STARK_API_KEY"] = key }
        runner(tools.node, [resources.appendingPathComponent("src/desktop-setup.mjs").path] + arguments, environment) { [weak self] result in
            self?.busy = false
            completion(result)
        }
    }

    private static func findTools() -> Tools? {
        let home = NSHomeDirectory()
        let nodes = [home + "/.local/share/pi-node/current/bin/node", "/opt/homebrew/bin/node", "/usr/local/bin/node"]
        let clients = [home + "/.local/bin/pi", home + "/.pi/agent/bin/pi", "/opt/homebrew/bin/pi", "/usr/local/bin/pi"]
        guard let node = nodes.first(where: { FileManager.default.isExecutableFile(atPath: $0) }),
              let pi = clients.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else { return nil }
        return Tools(node: URL(fileURLWithPath: node), pi: URL(fileURLWithPath: pi))
    }

    private static func failure(_ result: [String: Any]) -> String {
        switch result["code"] as? String {
        case "pi_version_unsupported": return "This setup supports Pi 1.5.2. Install that version, then check again."
        case "node_version_unsupported": return "Pi setup needs Node.js 22.19 or newer."
        case "pi_not_found": return "Pi could not start. Check its installation and try again."
        case "authentication_failed": return "Stark's API key changed. Retry setup with the current key."
        case "tools_unsupported": return "Start a model with tool support, then update Pi setup."
        case "unknown_or_invalid_limits": return "The running model has no verified context limits. Choose another model."
        case "unsafe_setup_location": return "Pi's setup folder has unsafe permissions or a redirected path. Check the folder and try again."
        case "model_unavailable_or_changed", "setup_stale": return "The running model changed. Update Pi setup."
        case "setup_changed", "invalid_setup": return "The saved Pi setup changed. Set it up again."
        case "helper_timeout": return "Pi setup took too long. Check Stark and Pi, then try again."
        default: return "Pi setup could not finish. Check that the model and local API are running, then try again."
        }
    }

    private static func copyKey(_ key: String) -> Bool {
        guard ConnectionClipboard.write(key, sensitive: true) else { return false }
        let change = NSPasteboard.general.changeCount
        DispatchQueue.main.asyncAfter(deadline: .now() + 120) {
            if NSPasteboard.general.changeCount == change { NSPasteboard.general.clearContents() }
        }
        return true
    }

    private static func openTerminal(_ command: URL, completion: @escaping (Bool) -> Void) {
        let terminal = URL(fileURLWithPath: "/System/Applications/Utilities/Terminal.app")
        NSWorkspace.shared.open([command], withApplicationAt: terminal, configuration: NSWorkspace.OpenConfiguration()) { app, _ in
            DispatchQueue.main.async { completion(app != nil) }
        }
    }

    private static func runHelper(_ executable: URL, _ arguments: [String], _ environment: [String: String],
                                  completion: @escaping Completion) {
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process(), output = Pipe()
            process.executableURL = executable; process.arguments = arguments; process.environment = environment
            process.standardInput = FileHandle.nullDevice; process.standardOutput = output
            process.standardError = FileHandle.nullDevice
            do { try process.run() } catch {
                DispatchQueue.main.async { completion(["ok": false, "code": "helper_failed"]) }; return
            }
            let timeout = DispatchWorkItem {
                if process.isRunning {
                    process.terminate()
                    DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
                        if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
                    }
                }
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 25, execute: timeout)
            var data = Data()
            while let next = try? output.fileHandleForReading.read(upToCount: 4096), !next.isEmpty {
                data.append(next)
                if data.count > 16 * 1024 { process.terminate(); break }
            }
            process.waitUntilExit(); timeout.cancel()
            let result = data.count <= 16 * 1024 ? (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] : nil
            DispatchQueue.main.async { completion(result ?? ["ok": false, "code": "helper_failed"]) }
        }
    }
}
