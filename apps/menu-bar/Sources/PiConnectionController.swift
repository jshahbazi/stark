import AppKit
import Combine
import Foundation
import Darwin

/// Manual Pi setup uses a separate public profile. The master key is passed to
/// the setup helper only in its environment; launches read it from the saved
/// Keychain command in Pi's normal configuration. No managed capability or
/// approval is issued.
enum PiTerminal: String, CaseIterable, Identifiable {
    case terminal, iTerm

    var id: String { rawValue }
    var name: String { self == .terminal ? "Terminal" : "iTerm2" }
    var bundleID: String { self == .terminal ? "com.apple.Terminal" : "com.googlecode.iterm2" }
    var applicationURL: URL? { NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) }
}

final class PiConnectionController: ObservableObject {
    @Published private(set) var busy = false
    @Published private(set) var installed = false
    @Published private(set) var message = "Checking Pi installation…"
    @Published private(set) var profile: String?
    @Published private(set) var configuredModel = ""
    @Published private(set) var configuredBaseURL = ""
    @Published private(set) var workingFolder: URL?
    @Published private(set) var terminal: PiTerminal = .terminal

    struct Tools {
        let node: URL
        let pi: URL
    }
    typealias Completion = ([String: Any]) -> Void
    typealias Runner = (URL, [String], [String: String], @escaping Completion) -> Void
    typealias FolderChooser = (NSWindow, URL?, @escaping (URL?) -> Void) -> Void
    typealias KeychainVerifier = (String, @escaping (Bool) -> Void) -> Void
    typealias TerminalOpener = (URL, URL, @escaping (Bool) -> Void) -> Void
    private let resources: URL
    private let root: URL
    private let defaults: UserDefaults
    private let runner: Runner
    private let finder: () -> Tools?
    private let keychainVerifier: KeychainVerifier
    private var tools: Tools?
    private var refreshed = false
    private static let profilePreference = "stark.pi.manualProfile.v1"
    private static let workingFolderPreference = "stark.pi.workingFolder.v1"
    private static let terminalPreference = "stark.pi.terminal.v1"

    init(resources: URL = Bundle.main.resourceURL?.appendingPathComponent("PiClient") ?? URL(fileURLWithPath: "/nonexistent"),
         root: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Stark/Pi", isDirectory: true),
         defaults: UserDefaults = .standard, tools: Tools? = nil,
         finder: @escaping () -> Tools? = { PiConnectionController.findTools() },
         runner: @escaping Runner = PiConnectionController.runHelper,
         keychainVerifier: @escaping KeychainVerifier = PiConnectionController.verifyKeychain) {
        self.resources = resources; self.root = root; self.defaults = defaults
        self.tools = tools; self.finder = finder; self.runner = runner
        self.keychainVerifier = keychainVerifier
        if let path = defaults.string(forKey: Self.workingFolderPreference), path.hasPrefix("/"),
           !path.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) {
            workingFolder = URL(fileURLWithPath: path, isDirectory: true)
        }
        if let saved = defaults.string(forKey: Self.terminalPreference),
           let selected = PiTerminal(rawValue: saved), selected.applicationURL != nil {
            terminal = selected
        }
    }

    var workingFolderLabel: String { workingFolder?.lastPathComponent ?? "Stark workspace (automatic)" }
    var availableTerminals: [PiTerminal] { PiTerminal.allCases.filter { $0.applicationURL != nil } }

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
        if force || tools == nil { tools = finder() }
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
                self.message = self.profile == nil ? "Pi 1.5.2 found." : "Setup saved for \(self.configuredModel)."
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

    func setup(baseURL: String, modelID: String, key: String, keychainService: String? = nil) {
        guard !busy, installed, tools != nil, !baseURL.isEmpty, !modelID.isEmpty, !key.isEmpty else { return }
        var arguments = ["--action", "setup", "--pi", tools!.pi.path, "--root", root.path,
                         "--base-url", baseURL, "--model", modelID]
        if let keychainService {
            arguments += ["--default-agent-directory", NSHomeDirectory() + "/.pi/agent", "--keychain-service", keychainService]
        }
        perform(arguments, key: key) { [weak self] result in
            guard let self else { return }
            guard result["ok"] as? Bool == true, let path = result["profile"] as? String,
                  self.ownedProfilePath(path) else { self.message = Self.failure(result); return }
            guard keychainService == nil || result["commandLineConfigured"] as? Bool == true else {
                self.message = "Pi's command-line setup could not be saved. Try setup again."; return
            }
            self.acceptProfile(result)
            self.defaults.set(path, forKey: Self.profilePreference)
            self.message = keychainService == nil ? "Ready. Open Pi to use the running model."
                : "Ready. Typing pi in Terminal will use Stark."
        }
    }

    func open(baseURL: String, modelID: String, key: String, keychainService: String,
              openTerminal: @escaping TerminalOpener = PiConnectionController.openTerminal) {
        guard !busy, installed, matches(baseURL: baseURL, modelID: modelID), !key.isEmpty else { return }
        let application = terminal.applicationURL ?? PiTerminal.terminal.applicationURL
        guard let application else {
            message = "Terminal is unavailable. Install Terminal or iTerm2, then try again."
            return
        }
        busy = true
        message = "Waiting for macOS to allow Pi’s Keychain access…"
        keychainVerifier(keychainService) { [weak self] allowed in
            guard let self else { return }
            self.busy = false
            guard allowed else {
                self.message = "Allow Stark’s API key in Keychain Access, then choose Open Pi again."
                return
            }
            self.prepareLaunch(workspace: self.workingFolder) { command, completion in
                openTerminal(command, application, completion)
            }
        }
    }

    func chooseWorkingFolder(in window: NSWindow?, chooser: @escaping FolderChooser = PiConnectionController.chooseFolder) {
        guard !busy else { return }
        guard let window else { message = "Open Connections to change Pi's working folder."; return }
        busy = true
        chooser(window, workingFolder) { [weak self] chosen in
            guard let self else { return }
            self.busy = false
            guard let chosen, chosen.isFileURL else { return }
            self.workingFolder = chosen
            self.defaults.set(chosen.path, forKey: Self.workingFolderPreference)
            self.message = "Working folder updated for new Stark Pi sessions."
        }
    }

    func useDefaultWorkspace() {
        guard !busy else { return }
        workingFolder = nil
        defaults.removeObject(forKey: Self.workingFolderPreference)
        message = "New Stark Pi sessions will use the automatic workspace."
    }

    func chooseTerminal(_ terminal: PiTerminal) {
        guard terminal.applicationURL != nil else { return }
        self.terminal = terminal
        defaults.set(terminal.rawValue, forKey: Self.terminalPreference)
    }

    private static func chooseFolder(in window: NSWindow, current: URL?, completion: @escaping (URL?) -> Void) {
        let picker = NSOpenPanel()
        picker.title = "Choose a working folder for Stark Pi"
        picker.message = "Select a project folder for Pi to read and edit. This applies to new Stark Pi sessions."
        picker.prompt = "Use Folder"
        picker.canChooseDirectories = true; picker.canChooseFiles = false
        picker.allowsMultipleSelection = false; picker.canCreateDirectories = true
        picker.directoryURL = current ?? FileManager.default.homeDirectoryForCurrentUser
        picker.beginSheetModal(for: window) { response in completion(response == .OK ? picker.url : nil) }
    }

    /// Omitting workspace asks the helper to create/reuse Stark's private default.
    func prepareLaunch(workspace: URL? = nil,
                       openTerminal: @escaping (URL, @escaping (Bool) -> Void) -> Void) {
        guard !busy, let profile else { return }
        var arguments = ["--action", "prepare-launch", "--profile", profile]
        if let workspace { arguments += ["--workspace", workspace.path] }
        perform(arguments) { [weak self] result in
            guard let self else { return }
            guard result["ok"] as? Bool == true, let command = result["command"] as? String,
                  let workspace = result["workspace"] as? String,
                  URL(fileURLWithPath: command).deletingLastPathComponent().path == profile,
                  command.hasSuffix(".command"),
                  let launch = self.keychainLaunchCommand(command: URL(fileURLWithPath: command),
                                                          workspace: URL(fileURLWithPath: workspace)) else {
                self.message = Self.failure(result); return
            }
            openTerminal(launch) { [weak self] opened in
                self?.message = opened ? "Opening Pi in the selected terminal."
                    : "Could not open the selected terminal. Check Pi setup and try again."
            }
        }
    }

    private func keychainLaunchCommand(command: URL, workspace: URL) -> URL? {
        guard let profile, ownedProfilePath(profile), let tools,
              workspace.isFileURL, FileManager.default.fileExists(atPath: workspace.path),
              command.deletingLastPathComponent().path == profile, command.pathExtension == "command",
              !configuredModel.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
            message = "Pi's launch setup is unavailable. Update setup, then try again."
            return nil
        }
        let home = NSHomeDirectory()
        let agent = home + "/.pi/agent"
        let path = [tools.node.deletingLastPathComponent().path, tools.pi.deletingLastPathComponent().path,
                    home + "/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
                    "/usr/sbin", "/sbin"].joined(separator: ":")
        let extensionPath = URL(fileURLWithPath: profile).appendingPathComponent("extensions/stark-stream-guard.mjs").path
        let script = """
        #!/bin/zsh -f
        unsetopt XTRACE VERBOSE
        cd -- \(Self.shellQuote(workspace.path)) || exit 1
        exec /usr/bin/env -i HOME=\(Self.shellQuote(home)) TMPDIR=\(Self.shellQuote(FileManager.default.temporaryDirectory.path)) LANG='en_US.UTF-8' SHELL='/bin/zsh' TERM='xterm-256color' PATH=\(Self.shellQuote(path)) PI_TELEMETRY='0' PI_CODING_AGENT_DIR=\(Self.shellQuote(agent)) \(Self.shellQuote(tools.pi.path)) --provider 'stark-local' --model \(Self.shellQuote(configuredModel)) --thinking 'off' --offline --no-session --no-extensions --extension \(Self.shellQuote(extensionPath)) --no-mcp --no-skills --no-prompt-templates --no-themes --no-context-files --no-approve --tools 'read,edit,write,bash'
        """
        do {
            try Data(script.utf8).write(to: command, options: .atomic)
            guard chmod(command.path, 0o700) == 0 else {
                message = "Pi's Terminal launcher could not be saved. Update setup and try again."
                return nil
            }
            return command
        } catch {
            message = "Pi's Terminal launcher could not be saved. Update setup and try again."
            return nil
        }
    }

    private static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
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

    static func findTools(home: String = NSHomeDirectory()) -> Tools? {
        // Desktop apps do not inherit nvm's interactive-shell PATH. Inspect its
        // public installed-version names without sourcing shell startup files.
        let nvm = URL(fileURLWithPath: home + "/.nvm/versions/node", isDirectory: true)
        let versions = ((try? FileManager.default.contentsOfDirectory(atPath: nvm.path)) ?? [])
            .filter { $0.range(of: #"^v[0-9]+\.[0-9]+\.[0-9]+$"#, options: .regularExpression) != nil }
            .sorted { $0.compare($1, options: .numeric) == .orderedDescending }
            .prefix(64)
        let nvmNodes = versions.compactMap { version -> String? in
            let parts = version.dropFirst().split(separator: ".").compactMap { Int($0) }
            guard parts.count == 3, parts[0] > 22 || (parts[0] == 22 && parts[1] >= 19) else { return nil }
            return nvm.appendingPathComponent(version).appendingPathComponent("bin/node").path
        }
        let nodes = [home + "/.local/share/pi-node/current/bin/node"] + nvmNodes + ["/opt/homebrew/bin/node", "/usr/local/bin/node"]
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
        case "model_busy": return "Stark is using the model. Wait for the current operation, then retry setup."
        case "runtime_unavailable": return "Stark could not prepare the model for Pi. Run the model again or choose another model."
        case "discovery_timeout": return "Stark did not finish model discovery in time. It may still be preparing the model; wait for Ready, then retry setup."
        case "tools_unsupported": return "Start a model with tool support, then update Pi setup."
        case "unknown_or_invalid_limits": return "The running model has no verified context limits. Choose another model."
        case "unsafe_setup_location": return "Pi's setup folder has unsafe permissions or a redirected path. Check the folder and try again."
        case "model_unavailable_or_changed": return "The running model changed. Update Pi setup."
        case "setup_stale": return "Pi setup is out of date. Update setup."
        case "workspace_unavailable": return "The working folder is unavailable. Change the working folder and try again."
        case "configuration_busy": return "Pi is updating its settings. Close Pi, then try setup again."
        case "configuration_changed": return "Pi's settings changed during setup. Close Pi, then try again."
        case "provider_conflict": return "Pi already has a different stark-local provider. Preserve or remove that entry in Pi's models.json before setting up Stark."
        case "credential_conflict": return "Pi has a saved Stark key. In Pi, use /logout for stark-local, then retry setup to use Keychain."
        case "invalid_default_configuration", "unsafe_default_profile": return "Pi's saved configuration could not be safely updated. Check its models.json and settings.json, then try again."
        case "configuration_recovery_required": return "Pi's settings changed during recovery. Your backup is in Pi's .stark-backup folder; restore it before retrying setup."
        case "setup_changed", "invalid_setup": return "The saved Pi setup changed. Set it up again."
        case "helper_timeout": return "Pi setup took too long. Check Stark and Pi, then try again."
        default: return "Pi setup could not finish. Check that the model and local API are running, then try again."
        }
    }

    private static func openTerminal(_ command: URL, at terminal: URL,
                                     completion: @escaping (Bool) -> Void) {
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        NSWorkspace.shared.open([command], withApplicationAt: terminal, configuration: configuration) { app, _ in
            DispatchQueue.main.async { completion(app != nil) }
        }
    }

    private static func verifyKeychain(_ service: String, completion: @escaping (Bool) -> Void) {
        guard service.range(of: #"^com\.starkengine\.menubar\.local-api(?:\.[A-Za-z0-9_.-]+)?$"#,
                            options: .regularExpression) != nil else {
            DispatchQueue.main.async { completion(false) }
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/security")
            process.arguments = ["find-generic-password", "-s", service, "-a", "bearer-key-v1", "-w"]
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            do {
                try process.run()
                process.waitUntilExit()
                let allowed = process.terminationStatus == 0
                DispatchQueue.main.async { completion(allowed) }
            } catch {
                DispatchQueue.main.async { completion(false) }
            }
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
