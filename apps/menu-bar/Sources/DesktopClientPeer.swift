import AppKit
import Darwin
import Foundation
import Security

/// Managed desktop clients. Each has a fixed Developer ID identity and a
/// backend client-ID prefix that scopes its grants and revocation.
enum DesktopClient: CaseIterable {
    case openWork, openCode

    var displayName: String { self == .openWork ? "OpenWork" : "OpenCode" }
    var bundleID: String { self == .openWork ? "com.differentai.openwork" : "ai.opencode.desktop" }
    var teamID: String { self == .openWork ? "F5DJWB4CCV" : "5NZ4Q7NXJ4" }
    /// Signing identifier of the engine process that connects to Stark's socket.
    var engineIdentifier: String { self == .openWork ? "opencode2" : "opencode-cli" }
    var clientPrefix: String { self == .openWork ? "openwork." : "opencode." }
}

enum DesktopConnectionError: Error {
    case unavailable, unapproved, identity, unsafePath, busy, protocolError, backend

    func message(_ client: DesktopClient) -> String {
        let name = client.displayName
        switch self {
        case .unavailable:
            return client == .openWork ? "Open OpenWork and a local workspace, then choose Connect to OpenWork."
                : "Open OpenCode, then choose Connect to OpenCode."
        case .unapproved:
            return client == .openWork ? "Connect OpenWork in Stark's Connection Details."
                : "Allow this OpenCode engine in Stark's Connection Details."
        case .identity: return "Stark could not verify the running \(name) app. Use the official macOS app and reopen it without debugger or Node options."
        case .unsafePath: return "Stark could not safely update the \(name) connection files. Check their permissions and try again."
        case .busy: return "The \(name) connection is busy. Try again shortly."
        case .protocolError: return "Update Stark and its \(name) connector, then reconnect."
        case .backend: return "Stark's local API is unavailable. Choose Retry API, then reconnect \(name)."
        }
    }
}

/// Public kernel process metadata is used only for identity checks. Argument and
/// environment buffers never leave this module, enter a log, or enter a file.
struct DesktopProcessIdentity: Equatable {
    let pid: pid_t
    let parent: pid_t
    let seconds: UInt64
    let microseconds: UInt64

    static func read(_ pid: pid_t) -> DesktopProcessIdentity? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        guard pid > 1, proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size,
              info.pbi_uid == getuid() else { return nil }
        return Self(pid: pid, parent: pid_t(info.pbi_ppid), seconds: info.pbi_start_tvsec,
                    microseconds: info.pbi_start_tvusec)
    }

    /// Process ID plus start time: one process instance on this Mac.
    var instanceID: String { "\(pid).\(seconds).\(microseconds)" }
    func clientID(_ client: DesktopClient) -> String { client.clientPrefix + instanceID }
    var isCurrent: Bool { Self.read(pid) == self }

    static func isRunning(instanceID: String) -> Bool {
        let parts = instanceID.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3, let pid = pid_t(parts[0]) else { return false }
        return read(pid)?.instanceID == instanceID
    }
}

/// Public launch selectors of a verified process. Only paths are kept.
struct DesktopLaunchEnvironment: Equatable {
    var configDirectory = ""
    var xdgConfigHome = ""
    var home = ""
    /// Started as `serve --service`, the way OpenCode Desktop runs its engine.
    var service = false
}

struct DesktopApplicationIdentity {
    let client: DesktopClient
    let process: DesktopProcessIdentity
    let bundleURL: URL
    /// Isolated validation profile that a test bundle supplied, if any.
    var profile: URL?

    /// Called on the main queue. A command-line invocation of the signed
    /// interpreter is not an approved, regular LaunchServices application.
    static func running(_ client: DesktopClient, configurationRoot: URL? = nil) -> DesktopApplicationIdentity? {
        for app in NSWorkspace.shared.runningApplications {
            guard app.bundleIdentifier == client.bundleID, app.activationPolicy == .regular,
                  app.isFinishedLaunching, !app.isTerminated, let url = app.bundleURL,
                  let identity = DesktopProcessIdentity.read(app.processIdentifier),
                  DesktopClientPeer.verify(pid: identity.pid, identifier: client.bundleID, client: client),
                  let configPath = DesktopClientPeer.launchEnvironment(identity.pid, client: client)?.configDirectory,
                  identity.isCurrent else { continue }
            let candidate = Self(client: client, process: identity, bundleURL: url.resolvingSymlinksInPath(), profile: configurationRoot)
            if let configurationRoot {
                // Electron may remove the parent's OPENCODE_CONFIG_DIR before
                // launching its engine. Bind isolated tests to the verified
                // descendant's actual native configuration, never a different
                // regular client profile.
                let direct = DesktopClientPeer.isWithin(configPath, root: configurationRoot)
                let engine = client == .openWork ? DesktopClientPeer.engineConfigDirectory(application: candidate) : nil
                guard direct || engine.map({ DesktopClientPeer.isWithin($0.path, root: configurationRoot) }) == true else { continue }
            }
            return candidate
        }
        return nil
    }
}

/// A detached OpenCode Desktop engine and the configuration directory it reads.
struct OpenCodeEngine {
    let process: DesktopProcessIdentity
    let configDirectory: URL
}

enum DesktopClientPeer {
    static func isWithin(_ path: String, root: URL) -> Bool {
        guard !path.isEmpty, let boundary = realpath(root.path, nil) else { return false }
        defer { free(boundary) }
        guard let candidate = realpath(path, nil) else { return false }
        defer { free(candidate) }
        return String(cString: candidate).hasPrefix(String(cString: boundary) + "/")
    }

    private static func requirement(_ identifier: String, client: DesktopClient) -> SecRequirement? {
        let source = "anchor apple generic and identifier \"\(identifier)\" and certificate leaf[subject.OU] = \"\(client.teamID)\" and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
        var result: SecRequirement?
        guard SecRequirementCreateWithString(source as CFString, [], &result) == errSecSuccess else { return nil }
        return result
    }

    static func verify(pid: pid_t, identifier: String, client: DesktopClient) -> Bool {
        guard let before = DesktopProcessIdentity.read(pid), let required = requirement(identifier, client: client) else { return false }
        var code: SecCode?
        guard SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributePid as String: pid] as CFDictionary, [], &code) == errSecSuccess,
              let code, SecCodeCheckValidity(code, SecCSFlags(rawValue: kSecCSStrictValidate), required) == errSecSuccess else { return false }
        return before.isCurrent
    }

    /// Checks an application bundle on disk before Stark opens it.
    static func verifyApplication(_ url: URL, client: DesktopClient) -> Bool {
        var code: SecStaticCode?
        var requirement: SecRequirement?
        let source = "anchor apple generic and identifier \"\(client.bundleID)\" and certificate leaf[subject.OU] = \"\(client.teamID)\""
        return SecStaticCodeCreateWithPath(url as CFURL, [], &code) == errSecSuccess
            && SecRequirementCreateWithString(source as CFString, [], &requirement) == errSecSuccess
            && code.map { SecStaticCodeCheckValidity($0, SecCSFlags(rawValue: kSecCSStrictValidate | kSecCSCheckNestedCode), requirement) == errSecSuccess } == true
    }

    static func executable(_ pid: pid_t) -> String? {
        var bytes = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        guard proc_pidpath(pid, &bytes, UInt32(bytes.count)) > 0 else { return nil }
        return String(cString: bytes)
    }

    static func launchConfiguration(_ pid: pid_t) -> String? {
        launchEnvironment(pid, client: .openWork)?.configDirectory
    }

    /// Extract only public config-directory selectors. Reject supported
    /// interpreter/debugger injection modes as defense in depth; this does not
    /// claim isolation from code/plugins inside an authorized client runtime.
    /// OpenCode's Bun engine additionally refuses loader and inspector variables.
    static func launchEnvironment(_ pid: pid_t, client: DesktopClient) -> DesktopLaunchEnvironment? {
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var count = 0
        guard sysctl(&mib, UInt32(mib.count), nil, &count, nil, 0) == 0,
              count > MemoryLayout<Int32>.size, count <= 1024 * 1024 else { return nil }
        var bytes = [UInt8](repeating: 0, count: count)
        defer { _ = bytes.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) } }
        guard sysctl(&mib, UInt32(mib.count), &bytes, &count, nil, 0) == 0 else { return nil }
        let argc = bytes.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }
        guard (1...256).contains(argc) else { return nil }
        var index = MemoryLayout<Int32>.size
        func next() -> ArraySlice<UInt8>? {
            guard index < count, let end = bytes[index..<count].firstIndex(of: 0) else { return nil }
            let part = bytes[index..<end]; index = end + 1; return part
        }
        guard next() != nil else { return nil } // executable path
        while index < count && bytes[index] == 0 { index += 1 }
        var result = DesktopLaunchEnvironment()
        var command: [String] = []
        for position in 0..<Int(argc) {
            guard let argument = next() else { return nil }
            let value = String(decoding: argument, as: UTF8.self)
            if value == "-e" || value == "--eval" || value == "-p" || value == "--print"
                || value.hasPrefix("--inspect") || value.hasPrefix("--remote-debugging")
                || value.hasPrefix("--require") || value.hasPrefix("--import") || value.hasPrefix("--preload") {
                return nil
            }
            if (1...2).contains(position) { command.append(value) }
        }
        result.service = command == ["serve", "--service"]
        func path(_ value: ArraySlice<UInt8>) -> String? {
            let text = String(decoding: value, as: UTF8.self)
            return text.utf8.count <= 4096 && text.hasPrefix("/") && !text.contains("\n") ? text : nil
        }
        while index < count, let entry = next(), !entry.isEmpty {
            guard let equals = entry.firstIndex(of: 61) else { continue }
            let name = String(decoding: entry[..<equals], as: UTF8.self)
            let value = entry[entry.index(after: equals)...]
            if ["ELECTRON_RUN_AS_NODE", "NODE_OPTIONS", "BUN_OPTIONS"].contains(name), !value.isEmpty { return nil }
            if name == "OPENWORK_ELECTRON_REMOTE_DEBUG_PORT", !value.isEmpty, value != [48][...] { return nil }
            if name == "OPENCODE_CONFIG_DIR" {
                guard let directory = path(value) else { return nil }
                result.configDirectory = directory
            }
            guard client == .openCode else { continue }
            // BUN_BE_BUN turns the compiled engine into the plain Bun interpreter.
            if (name.hasPrefix("DYLD_") || name.hasPrefix("BUN_INSPECT") || name == "BUN_BE_BUN"
                || name == "OPENCODE_DESKTOP_REMOTE_DEBUGGING_PORT"), !value.isEmpty { return nil }
            if name == "XDG_CONFIG_HOME", !value.isEmpty {
                guard let directory = path(value) else { return nil }
                result.xdgConfigHome = directory
            }
            if name == "HOME", !value.isEmpty {
                guard let directory = path(value) else { return nil }
                result.home = directory
            }
        }
        return result
    }

    static func authenticate(socket: Int32, application: DesktopApplicationIdentity) -> DesktopProcessIdentity? {
        let client = application.client
        var audit = audit_token_t()
        var size = socklen_t(MemoryLayout.size(ofValue: audit))
        guard getsockopt(socket, SOL_LOCAL, LOCAL_PEERTOKEN, &audit, &size) == 0,
              size == MemoryLayout.size(ofValue: audit), audit.val.1 == getuid(),
              let peer = DesktopProcessIdentity.read(pid_t(audit.val.5)),
              let required = requirement(client.engineIdentifier, client: client) else { return nil }
        var code: SecCode?
        let token = Data(bytes: &audit, count: MemoryLayout.size(ofValue: audit))
        guard SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributeAudit as String: token] as CFDictionary, [], &code) == errSecSuccess,
              let code, SecCodeCheckValidity(code, SecCSFlags(rawValue: kSecCSStrictValidate), required) == errSecSuccess,
              let path = executable(peer.pid) else { return nil }
        if client == .openCode {
            // OpenCode Desktop runs a detached copy of its engine whose parent is
            // launchd, so a running verified app replaces the ancestry check, and
            // the engine must read the configuration that app hands its engine.
            guard application.process.isCurrent,
                  verify(pid: application.process.pid, identifier: client.bundleID, client: client),
                  let expected = openCodeConfigDirectory(application: application),
                  openCodeEngine(peer, path: path, profile: application.profile, configDirectory: expected) != nil
            else { return nil }
            return peer
        }
        guard path == application.bundleURL.appendingPathComponent("Contents/Resources/sidecars/opencode2").path,
              launchConfiguration(peer.pid) != nil else { return nil }
        var chain = [peer]
        for _ in 0..<8 {
            guard let child = chain.last else { return nil }
            if child == application.process {
                guard chain.allSatisfy(\.isCurrent),
                      verify(pid: application.process.pid, identifier: client.bundleID, client: client),
                      launchConfiguration(application.process.pid) != nil else { return nil }
                return peer
            }
            guard let parent = DesktopProcessIdentity.read(child.parent), !chain.contains(parent) else { return nil }
            chain.append(parent)
        }
        return nil
    }

    private static func processes() -> [pid_t] {
        let bytes = proc_listallpids(nil, 0)
        guard bytes > 0, bytes < 100_000 else { return [] }
        var pids = [pid_t](repeating: 0, count: Int(bytes) + 128)
        let actual = proc_listallpids(&pids, Int32(pids.count * MemoryLayout<pid_t>.size))
        return actual > 0 ? Array(pids.prefix(Int(actual))) : []
    }

    static func engineConfigDirectory(application: DesktopApplicationIdentity) -> URL? {
        for pid in processes() {
            guard let identity = DesktopProcessIdentity.read(pid),
                  executable(pid) == application.bundleURL.appendingPathComponent("Contents/Resources/sidecars/opencode2").path,
                  verify(pid: pid, identifier: "opencode2", client: .openWork),
                  let path = launchConfiguration(pid), !path.isEmpty else { continue }
            var chain = [identity]
            for _ in 0..<8 {
                guard let child = chain.last else { break }
                if child == application.process, chain.allSatisfy(\.isCurrent) { return URL(fileURLWithPath: path, isDirectory: true) }
                guard let parent = DesktopProcessIdentity.read(child.parent), !chain.contains(parent) else { break }
                chain.append(parent)
            }
        }
        return nil
    }

    /// Account home from the user database rather than a caller's environment.
    static func userHome() -> URL? {
        var entry = passwd()
        var result: UnsafeMutablePointer<passwd>?
        var buffer = [CChar](repeating: 0, count: 16 * 1024)
        guard getpwuid_r(getuid(), &entry, &buffer, buffer.count, &result) == 0, result != nil,
              let directory = entry.pw_dir else { return nil }
        let path = String(cString: directory)
        return path.hasPrefix("/") ? URL(fileURLWithPath: path, isDirectory: true) : nil
    }

    /// OpenCode Desktop copies its signed engine to `<userData>/cli/<version>-<arch>/`.
    /// A validation profile uses the app's own isolated test root instead.
    static func openCodeEngineRoot(profile: URL?) -> URL? {
        if let profile { return profile.appendingPathComponent("opencode-desktop/desktop/cli", isDirectory: true) }
        return userHome()?.appendingPathComponent("Library/Application Support/ai.opencode.desktop/cli", isDirectory: true)
    }

    static func isStagedOpenCodeEngine(_ path: String, root: URL) -> Bool {
        guard let boundary = realpath(root.path, nil) else { return false }
        defer { free(boundary) }
        guard let resolved = realpath(path, nil) else { return false }
        defer { free(resolved) }
        let prefix = String(cString: boundary) + "/"
        let candidate = String(cString: resolved)
        guard candidate.hasPrefix(prefix) else { return false }
        let parts = candidate.dropFirst(prefix.count).split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts[1] == "opencode-cli", (1...64).contains(parts[0].utf8.count),
              parts[0] != ".", parts[0] != ".." else { return false }
        return parts[0].utf8.allSatisfy { byte in
            (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte) || [45, 46, 95].contains(byte)
        }
    }

    /// OpenCode reads `OPENCODE_CONFIG_DIR`, else `$XDG_CONFIG_HOME/opencode`, else `~/.config/opencode`.
    static func openCodeConfigDirectory(_ environment: DesktopLaunchEnvironment) -> URL? {
        if !environment.configDirectory.isEmpty { return URL(fileURLWithPath: environment.configDirectory, isDirectory: true) }
        if !environment.xdgConfigHome.isEmpty {
            return URL(fileURLWithPath: environment.xdgConfigHome, isDirectory: true).appendingPathComponent("opencode", isDirectory: true)
        }
        let home = environment.home.isEmpty ? userHome() : URL(fileURLWithPath: environment.home, isDirectory: true)
        return home?.appendingPathComponent(".config/opencode", isDirectory: true)
    }

    /// The engine inherits the Desktop app's environment, so the app's own
    /// selectors name the configuration its background engine reads.
    static func openCodeConfigDirectory(application: DesktopApplicationIdentity) -> URL? {
        launchEnvironment(application.process.pid, client: .openCode).flatMap(openCodeConfigDirectory)
    }

    /// A signed `opencode-cli` from Anomaly's team, running from the Desktop
    /// staging folder as `serve --service`, without loader, inspector or
    /// interpreter injection, and reading the running Desktop app's configuration.
    private static func openCodeEngine(_ identity: DesktopProcessIdentity, path: String, profile: URL?,
                                       configDirectory expected: URL) -> OpenCodeEngine? {
        guard let root = openCodeEngineRoot(profile: profile), isStagedOpenCodeEngine(path, root: root),
              let environment = launchEnvironment(identity.pid, client: .openCode), environment.service,
              let directory = openCodeConfigDirectory(environment),
              directory.standardizedFileURL.path == expected.standardizedFileURL.path, identity.isCurrent else { return nil }
        if let profile, !isWithin(directory.path, root: profile) { return nil }
        return OpenCodeEngine(process: identity, configDirectory: directory)
    }

    /// Verified engines for the Desktop app's configuration, newest first.
    /// Called off the main queue.
    static func openCodeEngines(profile: URL?, configDirectory: URL) -> [OpenCodeEngine] {
        let client = DesktopClient.openCode
        return processes().compactMap { pid -> OpenCodeEngine? in
            guard let identity = DesktopProcessIdentity.read(pid), let path = executable(pid),
                  let root = openCodeEngineRoot(profile: profile), isStagedOpenCodeEngine(path, root: root),
                  verify(pid: pid, identifier: client.engineIdentifier, client: client) else { return nil }
            return openCodeEngine(identity, path: path, profile: profile, configDirectory: configDirectory)
        }.sorted { ($0.process.seconds, $0.process.microseconds) > ($1.process.seconds, $1.process.microseconds) }
    }
}

/// Validation bundles can isolate an official, unchanged client release's HOME
/// and data paths. No test option widens signing, peer checks or authorization.
enum DesktopConnectionTesting {
    static var profile: URL? {
        guard Bundle.main.bundleIdentifier?.hasPrefix("com.starkengine.menubar.validation.") == true,
              let path = Bundle.main.object(forInfoDictionaryKey: "StarkOpenWorkTestProfile") as? String,
              path.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func application(_ client: DesktopClient) -> URL? {
        let key = client == .openWork ? "StarkOpenWorkTestApplication" : "StarkOpenCodeTestApplication"
        guard profile != nil, let path = Bundle.main.object(forInfoDictionaryKey: key) as? String else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}
