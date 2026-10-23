import AppKit
import CryptoKit
import SwiftUI

struct ReleaseVersion: Comparable, Equatable {
    let components: [Int]

    init?(_ value: String) {
        let parts = value.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3, parts.allSatisfy({ part in
            !part.isEmpty && part.count <= 9 && part.utf8.allSatisfy({ (48...57).contains($0) }) &&
                (part == "0" || part.first != "0")
        }) else { return nil }
        components = parts.compactMap { Int($0) }
    }

    static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.components.lexicographicallyPrecedes(rhs.components)
    }
}

enum AppUpdateError: Error, LocalizedError {
    case release, download, checksum, signature, metadata, location, helper, command

    var errorDescription: String? {
        switch self {
        case .release: return "No compatible app update was found in the latest stable release."
        case .download: return "Could not download the update. Check your connection and try again."
        case .checksum: return "The update checksum did not match. Try downloading it again."
        case .signature: return "The update could not be verified as a notarized Stark release."
        case .metadata: return "The downloaded app does not match the selected release or this Mac."
        case .location: return "Move Stark to a writable Applications folder before updating."
        case .helper: return "The update could not be prepared. Your current app was kept."
        case .command: return "macOS could not prepare the update. Your current app was kept."
        }
    }
}

struct AppRelease: Decodable {
    struct Asset: Decodable {
        let name: String
        let browser_download_url: String
        let digest: String?
        let size: Int64
    }
    let tag_name: String
    let draft: Bool
    let prerelease: Bool
    let assets: [Asset]

    func update(after current: String) throws -> AppUpdateDownload? {
        guard !draft, !prerelease, tag_name.first == "v",
              let version = ReleaseVersion(String(tag_name.dropFirst())),
              let installed = ReleaseVersion(current) else { throw AppUpdateError.release }
        guard version > installed else { return nil }
        let number = String(tag_name.dropFirst())
        let filename = "stark-\(number)-aarch64-apple-darwin.dmg"
        let expectedURL = "https://github.com/jshahbazi/stark/releases/download/\(tag_name)/\(filename)"
        guard let asset = assets.first(where: { $0.name == filename }),
              asset.browser_download_url == expectedURL,
              let url = URL(string: expectedURL),
              let digest = asset.digest, digest.hasPrefix("sha256:"),
              digest.count == 71, digest.dropFirst(7).utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              asset.size > 0, asset.size <= 1_073_741_824 else { throw AppUpdateError.release }
        return AppUpdateDownload(version: number, url: url, sha256: String(digest.dropFirst(7)), size: asset.size)
    }
}

struct AppUpdateDownload {
    let version: String
    let url: URL
    let sha256: String
    let size: Int64
}

enum AppUpdateFiles {
    static let team = "S5BT2PRN5B"
    static let appID = "com.starkengine.menubar"

    static func command(_ executable: String, _ arguments: [String], timeout: TimeInterval = 90) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.1) }
        if process.isRunning {
            process.terminate()
            Thread.sleep(forTimeInterval: 0.2)
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            throw AppUpdateError.command
        }
        guard process.terminationStatus == 0 else { throw AppUpdateError.command }
    }

    static func verifySignature(_ url: URL, app: Bool) throws {
        let identifier = app ? " and identifier \"\(appID)\"" : ""
        let requirement = "=anchor apple generic\(identifier) and " +
            "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and " +
            "certificate leaf[subject.OU] = \"\(team)\""
        do {
            try command("/usr/bin/codesign", ["--verify", "--deep", "--strict", "-R", requirement, url.path])
            if app { try command("/usr/sbin/spctl", ["--assess", "--type", "execute", url.path]) }
        } catch { throw AppUpdateError.signature }
    }

    static func verifyMetadata(_ app: URL, version: String) throws {
        let infoURL = app.appendingPathComponent("Contents/Info.plist")
        guard let data = try? Data(contentsOf: infoURL),
              let info = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
              info["CFBundleIdentifier"] as? String == appID,
              info["CFBundleExecutable"] as? String == "Stark",
              info["CFBundleShortVersionString"] as? String == version,
              let minimum = info["LSMinimumSystemVersion"] as? String else { throw AppUpdateError.metadata }
        let minimumParts = minimum.split(separator: ".", omittingEmptySubsequences: false)
        let parts = minimumParts.compactMap { Int($0) }
        guard (1...3).contains(parts.count), parts.count == minimumParts.count,
              parts.allSatisfy({ $0 >= 0 }),
              ProcessInfo.processInfo.isOperatingSystemAtLeast(OperatingSystemVersion(
                majorVersion: parts[0], minorVersion: parts.count > 1 ? parts[1] : 0,
                patchVersion: parts.count > 2 ? parts[2] : 0)) else { throw AppUpdateError.metadata }
        let executable = app.appendingPathComponent("Contents/MacOS/Stark")
        let handle = try FileHandle(forReadingFrom: executable)
        defer { try? handle.close() }
        guard let header = try handle.read(upToCount: 4096), supportsArm64(header) else { throw AppUpdateError.metadata }
    }

    // Read Mach-O headers directly so updating never requires Xcode/lipo.
    static func supportsArm64(_ header: Data) -> Bool {
        let bytes = [UInt8](header)
        guard bytes.count >= 8 else { return false }
        func word(_ offset: Int, little: Bool) -> UInt32 {
            let four = Array(bytes[offset..<offset + 4])
            return (little ? Array(four.reversed()) : four).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        }
        let magic = Array(bytes.prefix(4))
        if magic == [0xcf, 0xfa, 0xed, 0xfe] { return word(4, little: true) == 0x0100000c }
        if magic == [0xfe, 0xed, 0xfa, 0xcf] { return word(4, little: false) == 0x0100000c }
        let fat32 = magic == [0xca, 0xfe, 0xba, 0xbe] || magic == [0xbe, 0xba, 0xfe, 0xca]
        let fat64 = magic == [0xca, 0xfe, 0xba, 0xbf] || magic == [0xbf, 0xba, 0xfe, 0xca]
        guard fat32 || fat64 else { return false }
        let little = magic.first == 0xbe || magic.first == 0xbf
        let count = Int(word(4, little: little)), stride = fat64 ? 32 : 20
        guard count > 0, count <= 64, bytes.count >= 8 + count * stride else { return false }
        return (0..<count).contains { word(8 + $0 * stride, little: little) == 0x0100000c }
    }

    static func prepare(_ download: URL, release: AppUpdateDownload, workspace: URL) throws -> URL {
        let manager = FileManager.default
        let image = workspace.appendingPathComponent("release.dmg")
        try manager.moveItem(at: download, to: image)
        let handle = try FileHandle(forReadingFrom: image)
        defer { try? handle.close() }
        var hash = SHA256()
        var size: Int64 = 0
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            hash.update(data: data); size += Int64(data.count)
        }
        guard size == release.size,
              hash.finalize().map({ String(format: "%02x", $0) }).joined() == release.sha256 else { throw AppUpdateError.checksum }
        try verifySignature(image, app: false)
        do {
            try command("/usr/sbin/spctl", ["--assess", "--type", "open", "--context", "context:primary-signature", image.path])
        } catch { throw AppUpdateError.signature }
        try command("/usr/bin/hdiutil", ["verify", image.path])
        let mount = workspace.appendingPathComponent("mount")
        try manager.createDirectory(at: mount, withIntermediateDirectories: false)
        try command("/usr/bin/hdiutil", ["attach", "-readonly", "-nobrowse", "-noautoopen", "-mountpoint", mount.path, image.path])
        defer { try? command("/usr/bin/hdiutil", ["detach", mount.path]) }
        let source = mount.appendingPathComponent("Stark.app")
        let candidate = workspace.appendingPathComponent("Stark.app")
        try verifySignature(source, app: true)
        try verifyMetadata(source, version: release.version)
        try manager.copyItem(at: source, to: candidate)
        try verifySignature(candidate, app: true)
        try manager.removeItem(at: image)
        return candidate
    }
}

@MainActor
final class AppUpdateController: ObservableObject {
    @Published private(set) var message = ""
    @Published private(set) var busy = false
    @Published private(set) var ready = false
    @Published private(set) var available: AppUpdateDownload?
    private var workspace: URL?
    private var candidate: URL?
    private var task: Task<Void, Never>?
    private var helper: Process?
    private let installed = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
    private let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 900
        return URLSession(configuration: configuration)
    }()

    var actionTitle: String {
        if ready { return "Install and Relaunch" }
        if let available { return "Upgrade to \(available.version)…" }
        return "Check for Updates…"
    }

    var versionLabel: String { "Version \(installed)" }

    init() {
        if let path = UserDefaults.standard.string(forKey: "appUpdateWorkspace") {
            let previous = URL(fileURLWithPath: path)
            if let data = try? Data(contentsOf: previous.appendingPathComponent("result.json")),
               let result = try? JSONDecoder().decode([String: String].self, from: data),
               let error = result["error"] {
                message = error
            }
            // Failed workspaces may retain a recovery backup; never delete them
            // automatically. Successful helpers remove their private workspace.
            UserDefaults.standard.removeObject(forKey: "appUpdateWorkspace")
        }
    }

    func performAction() {
        guard !busy else { return }
        if ready { install(); return }
        if let available { download(available); return }
        check()
    }

    func check() {
        guard !busy else { return }
        busy = true; message = "Checking for updates…"
        task = Task {
            do {
                var request = URLRequest(url: URL(string: "https://api.github.com/repos/jshahbazi/stark/releases/latest")!)
                request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
                request.setValue("Stark-App-Updater", forHTTPHeaderField: "User-Agent")
                let (data, response) = try await session.data(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200, data.count <= 4 * 1024 * 1024 else { throw AppUpdateError.download }
                available = try JSONDecoder().decode(AppRelease.self, from: data).update(after: installed)
                message = available == nil ? "Stark is up to date." : "A new version is available."
            } catch { message = (error as? AppUpdateError)?.localizedDescription ?? AppUpdateError.download.localizedDescription }
            busy = false
        }
    }

    private func download(_ release: AppUpdateDownload) {
        busy = true; message = "Downloading Stark \(release.version)…"
        task = Task {
            var directory: URL?
            do {
                let target = Bundle.main.bundleURL
                guard target.pathExtension == "app", target.lastPathComponent == "Stark.app",
                      target == target.resolvingSymlinksInPath(),
                      FileManager.default.isWritableFile(atPath: target.deletingLastPathComponent().path) else { throw AppUpdateError.location }
                let root = FileManager.default.temporaryDirectory.appendingPathComponent("stark-update-" + UUID().uuidString)
                directory = root
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
                let (file, response) = try await session.download(from: release.url)
                guard (response as? HTTPURLResponse)?.statusCode == 200,
                      response.url?.scheme == "https" else { throw AppUpdateError.download }
                message = "Verifying and preparing Stark \(release.version)…"
                let prepared = try await Task.detached {
                    try AppUpdateFiles.prepare(file, release: release, workspace: root)
                }.value
                workspace = root; candidate = prepared; ready = true
                message = "Ready to install. Relaunching stops active requests and clears the current conversation."
            } catch {
                if let directory { try? FileManager.default.removeItem(at: directory) }
                message = (error as? AppUpdateError)?.localizedDescription ?? AppUpdateError.download.localizedDescription
            }
            busy = false
        }
    }

    private func install() {
        guard let workspace, let candidate, let release = available,
              let resources = Bundle.main.resourceURL else { return }
        busy = true; message = "Preparing to relaunch…"
        task = Task {
            do {
                for name in ["update_app.py", "install_app.py"] {
                    let output = workspace.appendingPathComponent(name)
                    try? FileManager.default.removeItem(at: output)
                    try FileManager.default.copyItem(at: resources.appendingPathComponent(name), to: output)
                }
                for name in ["ready", "result.json"] { try? FileManager.default.removeItem(at: workspace.appendingPathComponent(name)) }
                let process = Process()
                process.executableURL = candidate.appendingPathComponent("Contents/Resources/runtime/libexec/python/bin/python3.13")
                process.arguments = ["-I", "-B", workspace.appendingPathComponent("update_app.py").path,
                                     "--candidate", candidate.path, "--target", Bundle.main.bundleURL.path,
                                     "--version", release.version, "--pid", String(ProcessInfo.processInfo.processIdentifier),
                                     "--workspace", workspace.path]
                process.standardInput = FileHandle.nullDevice
                process.standardOutput = FileHandle.nullDevice
                process.standardError = FileHandle.nullDevice
                process.environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
                                       "PYTHONDONTWRITEBYTECODE": "1", "LANG": "en_US.UTF-8"]
                try process.run()
                helper = process
                UserDefaults.standard.set(workspace.path, forKey: "appUpdateWorkspace")
                // Staging can copy a large runtime. Keep the main thread and
                // the current app usable until the helper explicitly reports ready.
                while process.isRunning {
                    if FileManager.default.fileExists(atPath: workspace.appendingPathComponent("ready").path) {
                        NSApplication.shared.terminate(nil)
                        return
                    }
                    try await Task.sleep(nanoseconds: 200_000_000)
                }
                if let data = try? Data(contentsOf: workspace.appendingPathComponent("result.json")),
                   let result = try? JSONDecoder().decode([String: String].self, from: data), let error = result["error"] {
                    message = error
                } else { message = AppUpdateError.helper.localizedDescription }
            } catch { message = AppUpdateError.helper.localizedDescription }
            busy = false
        }
    }
}

struct AppUpdateView: View {
    @ObservedObject var updater: AppUpdateController
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(updater.versionLabel).font(.caption).foregroundStyle(.secondary)
                Spacer()
                if updater.busy { ProgressView().controlSize(.small) }
                Button(updater.actionTitle, action: updater.performAction).disabled(updater.busy)
            }
            if !updater.message.isEmpty {
                Text(updater.message).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
