import Darwin
import Foundation

enum OpenWorkConnectorInstallError: Error {
    case unsafePath, unsafeFile, invalidConfiguration, conflict, concurrentChange, storage, durability

    var message: String {
        switch self {
        case .unsafePath, .unsafeFile:
            return "OpenWork's settings location is unsafe. Check its ownership and permissions, then reconnect."
        case .invalidConfiguration:
            return "OpenWork's settings are invalid or too large. Fix them in OpenWork, then reconnect."
        case .conflict:
            return "A different Stark connector is already configured. Remove that connector before reconnecting."
        case .concurrentChange:
            return "OpenWork's settings changed during connection. Close its settings editor and try again."
        case .storage:
            return "Stark could not save the OpenWork connection. Check the settings folder and try again."
        case .durability:
            return "The connection settings were written, but saving them could not be confirmed. Reopen OpenWork and check the connection."
        }
    }
}

/// Edits only public plugin metadata. No credentials or client auth files enter this type.
struct OpenWorkConnectorInstaller {
    static let ownershipMarker = "com.starkengine.openwork/v1"
    static let maximumConfigurationBytes = 256 * 1024

    let configDirectory: URL
    let connectorDirectory: URL
    let socketPath: String
    let apiBaseURL: String
    /// A transaction boundary used by focused filesystem tests; production callers leave it nil.
    var beforeCommit: (() throws -> Void)?

    init(configDirectory: URL, connectorDirectory: URL, socketPath: String, apiBaseURL: String,
         beforeCommit: (() throws -> Void)? = nil) {
        self.configDirectory = configDirectory
        self.connectorDirectory = connectorDirectory
        self.socketPath = socketPath
        self.apiBaseURL = apiBaseURL
        self.beforeCommit = beforeCommit
    }

    @discardableResult func install() throws -> Bool { try update(removing: false) }
    @discardableResult func remove() throws -> Bool { try update(removing: true) }

    private var package: String {
        connectorDirectory.standardizedFileURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private func update(removing: Bool) throws -> Bool {
        guard configDirectory.isFileURL, connectorDirectory.isFileURL,
              [nil, ""].contains(configDirectory.host), [nil, ""].contains(connectorDirectory.host),
              connectorDirectory.query == nil, connectorDirectory.fragment == nil,
              Self.validAPIBaseURL(apiBaseURL),
              socketPath.hasPrefix("/"), !socketPath.utf8.contains(0),
              !socketPath.contains("\n"), !socketPath.contains("\r"), socketPath.utf8.count < 104 else {
            throw OpenWorkConnectorInstallError.unsafePath
        }
        let directory = try Self.openDirectory(configDirectory)
        defer { close(directory) }
        try Self.validateConnectorDirectory(connectorDirectory)
        var directoryInfo = stat()
        guard fstat(directory, &directoryInfo) == 0 else { throw OpenWorkConnectorInstallError.storage }

        let lock = openat(directory, ".stark-openwork-connector.lock", O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard lock >= 0 else { throw OpenWorkConnectorInstallError.unsafeFile }
        defer { close(lock) }
        var lockInfo = stat()
        guard fstat(lock, &lockInfo) == 0, Self.safeRegular(lockInfo), lockInfo.st_mode & 0o077 == 0,
              flock(lock, LOCK_EX | LOCK_NB) == 0 else { throw OpenWorkConnectorInstallError.concurrentChange }
        defer { flock(lock, LOCK_UN) }

        let original = try Self.readConfiguration(directory)
        if let original, flock(original.descriptor, LOCK_EX | LOCK_NB) != 0 {
            close(original.descriptor)
            throw OpenWorkConnectorInstallError.concurrentChange
        }
        defer { if let original { flock(original.descriptor, LOCK_UN); close(original.descriptor) } }
        let source = original?.data ?? Data("{}\n".utf8)
        var parser = ConnectorJSONCParser(source)
        let root = try parser.parse()
        guard case .object(let properties, let rootCommas) = root.value else {
            throw OpenWorkConnectorInstallError.invalidConfiguration
        }
        let pluginProperty = properties.first { $0.key == "plugins" }
        var edits: [(Range<Int>, Data)] = []
        let entry: [String: Any] = ["package": package, "options": [
            "socketPath": socketPath, "starkConnector": Self.ownershipMarker, "expectedBaseURL": apiBaseURL,
        ]]
        let encoded = try JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys, .withoutEscapingSlashes])

        if let pluginProperty {
            guard case .array(let items, let commas) = pluginProperty.node.value else {
                throw OpenWorkConnectorInstallError.invalidConfiguration
            }
            var owned: [Int] = []
            for (index, item) in items.enumerated() {
                let marker = item.property("options")?.property("starkConnector")?.string
                let samePackage = item.property("package")?.string.map(Self.normalizePackage) == Self.normalizePackage(package)
                let oldSource = item.property("source")?.string.map(Self.normalizePackage) == Self.normalizePackage(package)
                let stringPackage = item.string.map(Self.normalizePackage) == Self.normalizePackage(package)
                if marker?.hasPrefix("com.starkengine.openwork/") == true || samePackage || oldSource || stringPackage {
                    guard marker == Self.ownershipMarker,
                          let previousPackage = item.property("package")?.string,
                          Self.localPackageURL(previousPackage) != nil,
                          samePackage || Self.bundledStarkConnector(previousPackage),
                          item.exactKeys(["package", "options"]),
                          let options = item.property("options"), options.exactKeys(["socketPath", "starkConnector", "expectedBaseURL"]),
                          let previousSocket = options.property("socketPath")?.string,
                          previousSocket == socketPath || Self.starkBrokerSocket(previousSocket),
                          let expectedBase = options.property("expectedBaseURL")?.string,
                          Self.validAPIBaseURL(expectedBase) else {
                        throw OpenWorkConnectorInstallError.conflict
                    }
                    owned.append(index)
                }
            }
            guard owned.count <= 1 else { throw OpenWorkConnectorInstallError.conflict }
            if let index = owned.first {
                if !removing {
                    let node = items[index]
                    let options = node.property("options")!
                    // These are our exact, public v1 fields. Refresh stale app
                    // locations/namespaces without touching unrelated JSONC or
                    // requiring an old app/socket to still exist after upgrade.
                    for (field, replacement) in [(node.property("package")!, package),
                                                  (options.property("socketPath")!, socketPath),
                                                  (options.property("expectedBaseURL")!, apiBaseURL)] {
                        if field.string == replacement { continue }
                        let value = try JSONSerialization.data(withJSONObject: replacement, options: [.fragmentsAllowed, .withoutEscapingSlashes])
                        edits.append((field.range, value))
                    }
                    if edits.isEmpty { return false }
                } else {
                    edits.append((items[index].range, Data()))
                    if index < commas.count { edits.append((commas[index]..<(commas[index] + 1), Data())) }
                    else if index > 0 { edits.append((commas[index - 1]..<(commas[index - 1] + 1), Data())) }
                }
            } else {
                if removing { return false }
                let comma = !items.isEmpty && commas.count < items.count ? "," : ""
                let insertion = Data((comma + "\n    ").utf8) + encoded + Data("\n  ".utf8)
                let end = pluginProperty.node.range.upperBound - 1
                edits.append((end..<end, insertion))
            }
        } else {
            if removing { return false }
            let comma = !properties.isEmpty && rootCommas.count < properties.count ? "," : ""
            let insertion = Data((comma + "\n  \"plugins\": [\n    ").utf8) + encoded + Data("\n  ]\n".utf8)
            let end = root.range.upperBound - 1
            edits.append((end..<end, insertion))
        }

        var replacement = source
        for (range, value) in edits.sorted(by: { $0.0.lowerBound > $1.0.lowerBound }) {
            replacement.replaceSubrange(range, with: value)
        }
        guard replacement.count <= Self.maximumConfigurationBytes else {
            throw OpenWorkConnectorInstallError.invalidConfiguration
        }
        // Validate the actual bytes after range edits, including retained comments and commas.
        var checked = ConnectorJSONCParser(replacement)
        _ = try checked.parse()
        try commit(replacement, original: original, directory: directory,
                   directoryInfo: directoryInfo, lock: lock, lockInfo: lockInfo)
        return true
    }

    private func commit(_ data: Data, original: ConnectorConfigurationSnapshot?, directory: Int32,
                        directoryInfo: stat, lock: Int32, lockInfo: stat) throws {
        let temporary = ".stark-openwork-\(UUID().uuidString).tmp"
        let staged = openat(directory, temporary, O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard staged >= 0 else { throw OpenWorkConnectorInstallError.storage }
        defer { close(staged); unlinkat(directory, temporary, 0) }
        try data.withUnsafeBytes { bytes in
            var written = 0
            while written < bytes.count {
                let count = Darwin.write(staged, bytes.baseAddress!.advanced(by: written), bytes.count - written)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw OpenWorkConnectorInstallError.storage }
                written += count
            }
        }
        var stagedInfo = stat()
        guard fsync(staged) == 0, fsync(directory) == 0, fstat(staged, &stagedInfo) == 0,
              Self.safeRegular(stagedInfo), stagedInfo.st_mode & 0o077 == 0 else { throw OpenWorkConnectorInstallError.storage }
        try beforeCommit?()

        let currentDirectory = try Self.openDirectory(configDirectory)
        defer { close(currentDirectory) }
        var currentDirectoryInfo = stat(), currentLockInfo = stat(), currentStagedInfo = stat()
        guard fstat(currentDirectory, &currentDirectoryInfo) == 0,
              Self.sameIdentity(directoryInfo, currentDirectoryInfo),
              fstatat(directory, ".stark-openwork-connector.lock", &currentLockInfo, AT_SYMLINK_NOFOLLOW) == 0,
              Self.sameIdentity(lockInfo, currentLockInfo), Self.safeRegular(currentLockInfo), currentLockInfo.st_mode & 0o077 == 0,
              fstatat(directory, temporary, &currentStagedInfo, AT_SYMLINK_NOFOLLOW) == 0,
              Self.sameMetadata(stagedInfo, currentStagedInfo) else { throw OpenWorkConnectorInstallError.concurrentChange }
        let current = try Self.readConfiguration(directory)
        defer { if let current { close(current.descriptor) } }
        if let original {
            guard let current, Self.sameMetadata(original.info, current.info), current.data == original.data else {
                throw OpenWorkConnectorInstallError.concurrentChange
            }
        } else if current != nil { throw OpenWorkConnectorInstallError.concurrentChange }

        // Advisory locks coordinate installers/editors that use flock. The final identity,
        // metadata and byte comparison also refuses changes observed from other editors.
        let result = original == nil
            ? renameatx_np(directory, temporary, directory, "opencode.jsonc", UInt32(RENAME_EXCL))
            : renameat(directory, temporary, directory, "opencode.jsonc")
        guard result == 0 else {
            if errno == EEXIST { throw OpenWorkConnectorInstallError.concurrentChange }
            throw OpenWorkConnectorInstallError.storage
        }
        guard fsync(directory) == 0 else { throw OpenWorkConnectorInstallError.durability }
    }

    private static func openDirectory(_ url: URL, allowRootOwner: Bool = false) throws -> Int32 {
        let path = url.path
        guard path.hasPrefix("/"), !path.utf8.contains(0), url.query == nil, url.fragment == nil else {
            throw OpenWorkConnectorInstallError.unsafePath
        }
        var descriptor = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard descriptor >= 0 else { throw OpenWorkConnectorInstallError.storage }
        for component in path.split(separator: "/", omittingEmptySubsequences: true) {
            guard component != ".", component != ".." else { close(descriptor); throw OpenWorkConnectorInstallError.unsafePath }
            let next = String(component).withCString { openat(descriptor, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
            close(descriptor)
            guard next >= 0 else { throw OpenWorkConnectorInstallError.unsafePath }
            descriptor = next
        }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_uid == geteuid() || (allowRootOwner && info.st_uid == 0),
              info.st_mode & 0o022 == 0 else {
            close(descriptor); throw OpenWorkConnectorInstallError.unsafePath
        }
        return descriptor
    }

    private static func validateConnectorDirectory(_ url: URL) throws {
        let descriptor = try openDirectory(url, allowRootOwner: true)
        defer { close(descriptor) }
        for name in ["package.json", "server.js"] {
            let file = openat(descriptor, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
            guard file >= 0 else { throw OpenWorkConnectorInstallError.unsafePath }
            defer { close(file) }
            var info = stat()
            guard fstat(file, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
                  info.st_uid == geteuid() || info.st_uid == 0, info.st_mode & 0o022 == 0,
                  info.st_nlink == 1, info.st_size > 0, info.st_size <= maximumConfigurationBytes else {
                throw OpenWorkConnectorInstallError.unsafePath
            }
        }
    }

    private static func normalizePackage(_ value: String) -> String {
        guard let url = URL(string: value), url.isFileURL, url.host == nil || url.host == "",
              url.query == nil, url.fragment == nil else { return value }
        return url.standardizedFileURL.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    /// The v1 marker is reserved for Stark's bundled connector. Recognize its
    /// former resource location even after that app has been moved or removed.
    /// Never read or execute the old package while repairing public metadata.
    private static func bundledStarkConnector(_ value: String) -> Bool {
        guard let url = localPackageURL(value) else { return false }
        return Array(url.standardizedFileURL.pathComponents.suffix(4)) == ["Stark.app", "Contents", "Resources", "OpenWorkConnector"]
    }

    private static func localPackageURL(_ value: String) -> URL? {
        guard value.utf8.count <= 4096, let url = URL(string: value), url.isFileURL,
              [nil, ""].contains(url.host), url.user == nil, url.password == nil, url.port == nil,
              url.query == nil, url.fragment == nil, url.path.hasPrefix("/"),
              !url.path.utf8.contains(0) else { return nil }
        return url
    }

    private static func starkBrokerSocket(_ value: String) -> Bool {
        for prefix in ["/private/tmp/stark-\(geteuid())-", "/tmp/stark-\(geteuid())-"] {
            guard value.hasPrefix(prefix), value.hasSuffix("/openwork.sock") else { continue }
            let hash = value.dropFirst(prefix.count).dropLast("/openwork.sock".count)
            return hash.utf8.count == 16 && hash.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
        }
        return false
    }

    private static func validAPIBaseURL(_ value: String) -> Bool {
        guard value.utf8.count <= 64, let parts = URLComponents(string: value), parts.scheme == "http", parts.host == "127.0.0.1",
              let port = parts.port, (1...65535).contains(port), parts.percentEncodedPath == "/v1",
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil else { return false }
        return value == "http://127.0.0.1:\(port)/v1"
    }

    private static func safeRegular(_ info: stat) -> Bool {
        info.st_mode & S_IFMT == S_IFREG && info.st_uid == geteuid() && info.st_nlink == 1
            && info.st_mode & 0o022 == 0 && info.st_mode & 0o600 == 0o600
    }

    private static func sameIdentity(_ one: stat, _ two: stat) -> Bool {
        one.st_dev == two.st_dev && one.st_ino == two.st_ino
    }

    private static func sameMetadata(_ one: stat, _ two: stat) -> Bool {
        sameIdentity(one, two) && one.st_uid == two.st_uid && one.st_mode == two.st_mode && one.st_nlink == two.st_nlink
            && one.st_size == two.st_size && one.st_mtimespec.tv_sec == two.st_mtimespec.tv_sec
            && one.st_mtimespec.tv_nsec == two.st_mtimespec.tv_nsec && one.st_ctimespec.tv_sec == two.st_ctimespec.tv_sec
            && one.st_ctimespec.tv_nsec == two.st_ctimespec.tv_nsec
    }

    private static func readConfiguration(_ directory: Int32) throws -> ConnectorConfigurationSnapshot? {
        let descriptor = openat(directory, "opencode.jsonc", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if descriptor < 0 {
            if errno == ENOENT { return nil }
            throw OpenWorkConnectorInstallError.unsafeFile
        }
        var keep = false
        defer { if !keep { close(descriptor) } }
        var info = stat()
        guard fstat(descriptor, &info) == 0, safeRegular(info) else { throw OpenWorkConnectorInstallError.unsafeFile }
        guard info.st_size >= 0, info.st_size <= maximumConfigurationBytes else {
            throw OpenWorkConnectorInstallError.invalidConfiguration
        }
        var data = Data(), buffer = [UInt8](repeating: 0, count: 8192)
        while true {
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            if count < 0 && errno == EINTR { continue }
            guard count >= 0 else { throw OpenWorkConnectorInstallError.storage }
            if count == 0 { break }
            guard data.count + count <= maximumConfigurationBytes else { throw OpenWorkConnectorInstallError.invalidConfiguration }
            data.append(contentsOf: buffer.prefix(count))
        }
        var after = stat()
        guard fstat(descriptor, &after) == 0, sameMetadata(info, after), data.count == info.st_size else {
            throw OpenWorkConnectorInstallError.concurrentChange
        }
        keep = true
        return ConnectorConfigurationSnapshot(descriptor: descriptor, info: info, data: data)
    }
}

private struct ConnectorConfigurationSnapshot {
    let descriptor: Int32
    let info: stat
    let data: Data
}

private struct ConnectorJSONCNode {
    indirect enum Value {
        case object([ConnectorJSONCProperty], [Int]), array([ConnectorJSONCNode], [Int]), string(String), scalar
    }
    let range: Range<Int>
    let value: Value
    var string: String? { if case .string(let value) = value { return value }; return nil }
    func property(_ name: String) -> ConnectorJSONCNode? {
        if case .object(let properties, _) = value { return properties.first { $0.key == name }?.node }
        return nil
    }
    func exactKeys(_ keys: Set<String>) -> Bool {
        if case .object(let properties, _) = value { return Set(properties.map(\.key)) == keys }
        return false
    }
}

private struct ConnectorJSONCProperty { let key: String; let node: ConnectorJSONCNode }

/// Bounded JSONC grammar with decoded duplicate-key checks and original byte ranges.
private struct ConnectorJSONCParser {
    private let bytes: [UInt8]
    private var offset = 0
    private var items = 0
    init(_ data: Data) { bytes = Array(data) }

    mutating func parse() throws -> ConnectorJSONCNode {
        guard bytes.count <= OpenWorkConnectorInstaller.maximumConfigurationBytes,
              String(bytes: bytes, encoding: .utf8) != nil else { throw failure() }
        let value = try node(depth: 0)
        try trivia()
        guard offset == bytes.count else { throw failure() }
        return value
    }

    private func failure() -> OpenWorkConnectorInstallError { .invalidConfiguration }
    private mutating func trivia() throws {
        while offset < bytes.count {
            if [9, 10, 13, 32].contains(bytes[offset]) { offset += 1; continue }
            if offset + 1 < bytes.count, bytes[offset] == 47, bytes[offset + 1] == 47 {
                offset += 2
                while offset < bytes.count, ![10, 13].contains(bytes[offset]) { offset += 1 }
                continue
            }
            if offset + 1 < bytes.count, bytes[offset] == 47, bytes[offset + 1] == 42 {
                offset += 2
                while offset + 1 < bytes.count, !(bytes[offset] == 42 && bytes[offset + 1] == 47) { offset += 1 }
                guard offset + 1 < bytes.count else { throw failure() }
                offset += 2
                continue
            }
            break
        }
    }

    private mutating func node(depth: Int) throws -> ConnectorJSONCNode {
        guard depth <= 64, items < 16_384 else { throw failure() }
        items += 1
        try trivia()
        guard offset < bytes.count else { throw failure() }
        let start = offset
        if bytes[offset] == 34 {
            let text = try quoted()
            return ConnectorJSONCNode(range: start..<offset, value: .string(text))
        }
        if bytes[offset] == 123 {
            offset += 1
            var properties: [ConnectorJSONCProperty] = [], commas: [Int] = [], keys: Set<String> = []
            try trivia()
            while offset < bytes.count, bytes[offset] != 125 {
                let key = try quoted()
                guard keys.insert(key).inserted else { throw failure() }
                try trivia()
                guard offset < bytes.count, bytes[offset] == 58 else { throw failure() }
                offset += 1
                properties.append(ConnectorJSONCProperty(key: key, node: try node(depth: depth + 1)))
                try trivia()
                guard offset < bytes.count else { throw failure() }
                if bytes[offset] != 44 { break }
                commas.append(offset); offset += 1
                try trivia()
            }
            guard offset < bytes.count, bytes[offset] == 125 else { throw failure() }
            offset += 1
            return ConnectorJSONCNode(range: start..<offset, value: .object(properties, commas))
        }
        if bytes[offset] == 91 {
            offset += 1
            var values: [ConnectorJSONCNode] = [], commas: [Int] = []
            try trivia()
            while offset < bytes.count, bytes[offset] != 93 {
                values.append(try node(depth: depth + 1))
                try trivia()
                guard offset < bytes.count else { throw failure() }
                if bytes[offset] != 44 { break }
                commas.append(offset); offset += 1
                try trivia()
            }
            guard offset < bytes.count, bytes[offset] == 93 else { throw failure() }
            offset += 1
            return ConnectorJSONCNode(range: start..<offset, value: .array(values, commas))
        }
        while offset < bytes.count, ![9, 10, 13, 32, 44, 47, 93, 125].contains(bytes[offset]) { offset += 1 }
        guard offset > start else { throw failure() }
        let data = Data(bytes[start..<offset])
        guard let value = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed),
              value is NSNull || value is NSNumber else { throw failure() }
        if let number = value as? NSNumber, !number.doubleValue.isFinite { throw failure() }
        return ConnectorJSONCNode(range: start..<offset, value: .scalar)
    }

    private mutating func quoted() throws -> String {
        guard offset < bytes.count, bytes[offset] == 34 else { throw failure() }
        let start = offset
        offset += 1
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            if byte == 34 {
                guard let value = try? JSONSerialization.jsonObject(with: Data(bytes[start..<offset]), options: .fragmentsAllowed),
                      let text = value as? String else { throw failure() }
                return text
            }
            guard byte >= 32 else { throw failure() }
            if byte == 92 {
                guard offset < bytes.count else { throw failure() }
                offset += 1
            }
        }
        throw failure()
    }
}
