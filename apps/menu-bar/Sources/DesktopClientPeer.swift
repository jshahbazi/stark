import AppKit
import Darwin
import Foundation
import Security

enum DesktopConnectionError: Error {
    case unavailable, unapproved, identity, unsafePath, busy, protocolError, backend

    var message: String {
        switch self {
        case .unavailable: return "Open OpenWork and a local workspace, then choose Connect to OpenWork."
        case .unapproved: return "Connect OpenWork in Stark's Connection Details."
        case .identity: return "Stark could not verify the running OpenWork app. Use the official macOS app and reopen it without debugger or Node options."
        case .unsafePath: return "Stark could not safely update the OpenWork connection files. Check their permissions and try again."
        case .busy: return "The OpenWork connection is busy. Try again shortly."
        case .protocolError: return "Update Stark and its OpenWork connector, then reconnect."
        case .backend: return "Stark's local API is unavailable. Choose Retry API, then reconnect OpenWork."
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

    var clientID: String { "openwork.\(pid).\(seconds).\(microseconds)" }
    var isCurrent: Bool { Self.read(pid) == self }
}

struct OpenWorkApplicationIdentity {
    static let bundleID = "com.differentai.openwork"
    static let teamID = "F5DJWB4CCV"
    let process: DesktopProcessIdentity
    let bundleURL: URL

    /// Called on the main queue. A command-line invocation of the signed
    /// interpreter is not an approved, regular LaunchServices application.
    static func running(configurationRoot: URL? = nil) -> OpenWorkApplicationIdentity? {
        for app in NSWorkspace.shared.runningApplications {
            guard app.bundleIdentifier == bundleID, app.activationPolicy == .regular,
                  app.isFinishedLaunching, !app.isTerminated, let url = app.bundleURL,
                  let identity = DesktopProcessIdentity.read(app.processIdentifier),
                  DesktopClientPeer.verify(pid: identity.pid, identifier: bundleID),
                  let configPath = DesktopClientPeer.launchConfiguration(identity.pid),
                  identity.isCurrent else { continue }
            let candidate = Self(process: identity, bundleURL: url.resolvingSymlinksInPath())
            if let configurationRoot {
                // Electron may remove the parent's OPENCODE_CONFIG_DIR before
                // launching its engine. Bind isolated tests to the verified
                // descendant's actual native configuration, never a different
                // regular OpenWork profile.
                let direct = DesktopClientPeer.isWithin(configPath, root: configurationRoot)
                let engine = DesktopClientPeer.engineConfigDirectory(application: candidate)
                guard direct || engine.map({ DesktopClientPeer.isWithin($0.path, root: configurationRoot) }) == true else { continue }
            }
            return candidate
        }
        return nil
    }
}

enum DesktopClientPeer {
    static func isWithin(_ path: String, root: URL) -> Bool {
        guard !path.isEmpty, let boundary = realpath(root.path, nil) else { return false }
        defer { free(boundary) }
        guard let candidate = realpath(path, nil) else { return false }
        defer { free(candidate) }
        return String(cString: candidate).hasPrefix(String(cString: boundary) + "/")
    }
    private static func requirement(_ identifier: String) -> SecRequirement? {
        let source = "anchor apple generic and identifier \"\(identifier)\" and certificate leaf[subject.OU] = \"\(OpenWorkApplicationIdentity.teamID)\" and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
        var result: SecRequirement?
        guard SecRequirementCreateWithString(source as CFString, [], &result) == errSecSuccess else { return nil }
        return result
    }

    static func verify(pid: pid_t, identifier: String) -> Bool {
        guard let before = DesktopProcessIdentity.read(pid), let required = requirement(identifier) else { return false }
        var code: SecCode?
        guard SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributePid as String: pid] as CFDictionary, [], &code) == errSecSuccess,
              let code, SecCodeCheckValidity(code, SecCSFlags(rawValue: kSecCSStrictValidate), required) == errSecSuccess else { return false }
        return before.isCurrent
    }

    static func executable(_ pid: pid_t) -> String? {
        var bytes = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        guard proc_pidpath(pid, &bytes, UInt32(bytes.count)) > 0 else { return nil }
        return String(cString: bytes)
    }

    /// Extract only a public config-directory selector. Reject supported
    /// interpreter/debugger injection modes as defense in depth; this does not
    /// claim isolation from code/plugins inside an authorized OpenWork runtime.
    static func launchConfiguration(_ pid: pid_t) -> String? {
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
        for _ in 0..<argc {
            guard let argument = next() else { return nil }
            let value = String(decoding: argument, as: UTF8.self)
            if value == "-e" || value == "--eval" || value == "-p" || value == "--print"
                || value.hasPrefix("--inspect") || value.hasPrefix("--remote-debugging")
                || value.hasPrefix("--require") || value.hasPrefix("--import") || value.hasPrefix("--preload") {
                return nil
            }
        }
        var directory = ""
        while index < count, let entry = next(), !entry.isEmpty {
            guard let equals = entry.firstIndex(of: 61) else { continue }
            let name = String(decoding: entry[..<equals], as: UTF8.self)
            let value = entry[entry.index(after: equals)...]
            if ["ELECTRON_RUN_AS_NODE", "NODE_OPTIONS", "BUN_OPTIONS"].contains(name), !value.isEmpty { return nil }
            if name == "OPENWORK_ELECTRON_REMOTE_DEBUG_PORT", !value.isEmpty, value != [48][...] { return nil }
            if name == "OPENCODE_CONFIG_DIR" {
                directory = String(decoding: value, as: UTF8.self)
                guard directory.utf8.count <= 4096, directory.hasPrefix("/"), !directory.contains("\n") else { return nil }
            }
        }
        return directory
    }

    static func authenticate(socket: Int32, application: OpenWorkApplicationIdentity) -> DesktopProcessIdentity? {
        var audit = audit_token_t()
        var size = socklen_t(MemoryLayout.size(ofValue: audit))
        guard getsockopt(socket, SOL_LOCAL, LOCAL_PEERTOKEN, &audit, &size) == 0,
              size == MemoryLayout.size(ofValue: audit), audit.val.1 == getuid(),
              let peer = DesktopProcessIdentity.read(pid_t(audit.val.5)),
              let required = requirement("opencode2") else { return nil }
        var code: SecCode?
        let token = Data(bytes: &audit, count: MemoryLayout.size(ofValue: audit))
        guard SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributeAudit as String: token] as CFDictionary, [], &code) == errSecSuccess,
              let code, SecCodeCheckValidity(code, SecCSFlags(rawValue: kSecCSStrictValidate), required) == errSecSuccess,
              executable(peer.pid) == application.bundleURL.appendingPathComponent("Contents/Resources/sidecars/opencode2").path,
              launchConfiguration(peer.pid) != nil else { return nil }
        var chain = [peer]
        for _ in 0..<8 {
            guard let child = chain.last else { return nil }
            if child == application.process {
                guard chain.allSatisfy(\.isCurrent),
                      verify(pid: application.process.pid, identifier: OpenWorkApplicationIdentity.bundleID),
                      launchConfiguration(application.process.pid) != nil else { return nil }
                return peer
            }
            guard let parent = DesktopProcessIdentity.read(child.parent), !chain.contains(parent) else { return nil }
            chain.append(parent)
        }
        return nil
    }

    static func engineConfigDirectory(application: OpenWorkApplicationIdentity) -> URL? {
        let bytes = proc_listallpids(nil, 0)
        guard bytes > 0, bytes < 100_000 else { return nil }
        var pids = [pid_t](repeating: 0, count: Int(bytes) + 128)
        let actual = proc_listallpids(&pids, Int32(pids.count * MemoryLayout<pid_t>.size))
        guard actual > 0 else { return nil }
        for pid in pids.prefix(Int(actual)) {
            guard let identity = DesktopProcessIdentity.read(pid),
                  executable(pid) == application.bundleURL.appendingPathComponent("Contents/Resources/sidecars/opencode2").path,
                  verify(pid: pid, identifier: "opencode2"), let path = launchConfiguration(pid), !path.isEmpty else { continue }
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
}
