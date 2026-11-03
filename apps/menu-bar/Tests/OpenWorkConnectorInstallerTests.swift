import Darwin
import Foundation

private enum CheckError: Error { case failed, injected }
private func check(_ condition: @autoclosure () throws -> Bool, line: UInt = #line) throws {
    if try !condition() {
        fputs("installer_check_failed_\(line)\n", stderr)
        throw CheckError.failed
    }
}

private final class Fixture {
    let root: URL
    let config: URL
    let connector: URL
    let socket = "/tmp/stark-installer-synthetic/openwork.sock"
    let apiBaseURL = "http://127.0.0.1:18133/v1"
    var file: URL { config.appendingPathComponent("opencode.jsonc") }
    var installer: OpenWorkConnectorInstaller {
        OpenWorkConnectorInstaller(configDirectory: config, connectorDirectory: connector, socketPath: socket, apiBaseURL: apiBaseURL)
    }
    init() throws {
        guard let temporary = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw CheckError.failed }
        let base = URL(fileURLWithPath: String(cString: temporary), isDirectory: true)
        free(temporary)
        root = base
            .appendingPathComponent("stark-public-config-\(UUID().uuidString)", isDirectory: true)
        config = root.appendingPathComponent("config", isDirectory: true)
        connector = root.appendingPathComponent("Stark Connector", isDirectory: true)
        for folder in [root, config, connector] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false,
                                                    attributes: [.posixPermissions: 0o700])
        }
        try Data("{\"type\":\"module\"}".utf8).write(to: connector.appendingPathComponent("package.json"))
        try Data("export default {};\n".utf8).write(to: connector.appendingPathComponent("server.js"))
    }
    deinit { try? FileManager.default.removeItem(at: root) }
    func write(_ source: String) throws { try write(Data(source.utf8)) }
    func write(_ data: Data) throws {
        try data.write(to: file)
        guard chmod(file.path, 0o600) == 0 else { throw CheckError.failed }
    }
    func text() throws -> String { try String(contentsOf: file, encoding: .utf8) }
    func noTemporaryFiles() throws -> Bool {
        try !FileManager.default.contentsOfDirectory(atPath: config.path).contains { $0.hasSuffix(".tmp") }
    }
    func ownedEntry(socket: String? = nil, baseURL: String? = nil, package: String? = nil, extra: Bool = false) throws -> String {
        var options: [String: Any] = ["socketPath": socket ?? self.socket,
                                       "starkConnector": OpenWorkConnectorInstaller.ownershipMarker,
                                       "expectedBaseURL": baseURL ?? apiBaseURL]
        if extra { options["userSetting"] = true }
        let value: [String: Any] = ["package": package ?? connector.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")),
                                    "options": options]
        return String(data: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]),
                      encoding: .utf8)!
    }
}

private func rejected(_ expected: OpenWorkConnectorInstallError? = nil, _ action: () throws -> Void) throws {
    do { try action() }
    catch let error as OpenWorkConnectorInstallError {
        if let expected { try check(String(describing: error) == String(describing: expected)) }
        return
    }
    throw CheckError.failed
}

@main
struct OpenWorkConnectorInstallerTests {
    static func main() {
        do {
            guard CommandLine.arguments.count == 2 else { exit(2) }
            try run(CommandLine.arguments[1])
        } catch let error as OpenWorkConnectorInstallError {
            fputs("installer_failure_\(String(describing: error))\n", stderr)
            exit(1)
        } catch { exit(1) }
    }

    private static func run(_ scenario: String) throws {
        let fixture = try Fixture()
        switch scenario {
        case "create":
            try check(try fixture.installer.install())
            let before = try Data(contentsOf: fixture.file)
            let parsed = try JSONSerialization.jsonObject(with: before) as! [String: Any]
            let plugins = parsed["plugins"] as! [[String: Any]]
            try check(plugins.count == 1)
            try check((plugins[0]["options"] as! [String: String])["socketPath"] == fixture.socket)
            try check((plugins[0]["options"] as! [String: String])["expectedBaseURL"] == fixture.apiBaseURL)
            try check(!(try fixture.installer.install()))
            try check(try Data(contentsOf: fixture.file) == before)
            var info = stat()
            try check(lstat(fixture.file.path, &info) == 0 && info.st_mode & 0o777 == 0o600)
            try check(try fixture.installer.remove())
            try check(!(try fixture.installer.remove()))
            try check(!(try fixture.text()).contains(OpenWorkConnectorInstaller.ownershipMarker))
            try check(try fixture.noTemporaryFiles())
        case "comments":
            let source = """
            // User's opening comment
            {
              "other": { "escaped": "a \\\"quote\\\" // } plugins", "unicode": "日本語", },
              "plugins": [
                {"package":"file:///other-one", "options":{"values":[true,1,null,],},},
                // Comment for the next user's plugin
                "another-plugin", /* keep this trailing comment */
              ],
              "last": 1e2, // Last user setting
            }
            // User's closing comment
            """
            try fixture.write(source)
            try check(try fixture.installer.install())
            let installed = try fixture.text()
            for fragment in ["// User's opening comment", "// User's closing comment", "\"unicode\": \"日本語\"",
                             "{\"package\":\"file:///other-one\", \"options\":{\"values\":[true,1,null,],},},",
                             "// Comment for the next user's plugin", "\"another-plugin\", /* keep this trailing comment */",
                             "\"last\": 1e2, // Last user setting"] {
                try check(installed.contains(fragment))
            }
            try check(!(try fixture.installer.install()))
            try check(try fixture.installer.remove())
            try check(!(try fixture.installer.remove()))
            try check((try fixture.text()).contains("// Comment for the next user's plugin"))
        case "empty-array-comment":
            try fixture.write("{\"plugins\":[// keep\n],\"other\":true}")
            try check(try fixture.installer.install())
            try check((try fixture.text()).contains("// keep\n"))
            try check(try fixture.installer.remove())
            try check(!(try fixture.installer.remove()))
        case "malformed":
            for source in ["", "[]", "{", "{\"x\":1 \"y\":2}", "{\"x\":NaN}", "{\"x\":1e1000}",
                           "{\"plugins\":[],,}", "{\"x\":/* incomplete}", "{\"x\":\"bad\\q\"}",
                           "{\"x\":01}", "{\"x\":+1}", "{\"x\":falsex}", "{} false"] {
                try fixture.write(source)
                try rejected(.invalidConfiguration) { _ = try fixture.installer.install() }
                try check(try fixture.text() == source)
                try check(try fixture.noTemporaryFiles())
            }
        case "duplicate-keys":
            for source in ["{\"plugins\":[],\"plugins\":[]}", "{\"plugins\":[],\"\\u0070lugins\":[]}",
                           "{\"unrelated\":{\"x\":1,\"x\":2}}"] {
                try fixture.write(source)
                try rejected(.invalidConfiguration) { _ = try fixture.installer.install() }
                try check(try fixture.text() == source)
            }
        case "bounds":
            let cases = [Data(repeating: 32, count: OpenWorkConnectorInstaller.maximumConfigurationBytes + 1),
                         Data(("{\"x\":" + String(repeating: "[", count: 65) + "0" + String(repeating: "]", count: 65) + "}").utf8),
                         Data(("{\"x\":[" + Array(repeating: "0", count: 16_384).joined(separator: ",") + "]}").utf8),
                         Data([123, 34, 120, 34, 58, 34, 255, 34, 125])]
            for value in cases {
                try fixture.write(value)
                try rejected(.invalidConfiguration) { _ = try fixture.installer.install() }
                try check(try Data(contentsOf: fixture.file) == value)
            }
        case "symlink-file":
            let victim = fixture.root.appendingPathComponent("victim")
            try Data("{}".utf8).write(to: victim)
            try FileManager.default.createSymbolicLink(at: fixture.file, withDestinationURL: victim)
            try rejected(.unsafeFile) { _ = try fixture.installer.install() }
            try check(try String(contentsOf: victim, encoding: .utf8) == "{}")
            try check(try FileManager.default.destinationOfSymbolicLink(atPath: fixture.file.path) == victim.path)
        case "symlink-directory":
            let alias = fixture.root.appendingPathComponent("config-alias")
            try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.config)
            let other = OpenWorkConnectorInstaller(configDirectory: alias, connectorDirectory: fixture.connector,
                                                    socketPath: fixture.socket, apiBaseURL: fixture.apiBaseURL)
            try rejected(.unsafePath) { _ = try other.install() }
            try check(!FileManager.default.fileExists(atPath: fixture.file.path))
        case "symlink-ancestor":
            let alias = fixture.root.appendingPathComponent("parent-alias")
            try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.root)
            let other = OpenWorkConnectorInstaller(configDirectory: alias.appendingPathComponent("config"),
                                                    connectorDirectory: fixture.connector, socketPath: fixture.socket, apiBaseURL: fixture.apiBaseURL)
            try rejected(.unsafePath) { _ = try other.install() }
        case "hardlink-file":
            try fixture.write("{}")
            let other = fixture.root.appendingPathComponent("linked")
            try FileManager.default.linkItem(at: fixture.file, to: other)
            try rejected(.unsafeFile) { _ = try fixture.installer.install() }
            try check(try fixture.text() == "{}")
        case "permissions":
            try fixture.write("{}")
            try check(chmod(fixture.file.path, 0o660) == 0)
            try rejected(.unsafeFile) { _ = try fixture.installer.install() }
            try check(chmod(fixture.file.path, 0o600) == 0)
            try check(chmod(fixture.config.path, 0o770) == 0)
            try rejected(.unsafePath) { _ = try fixture.installer.install() }
        case "read-only":
            try fixture.write("{}")
            try check(chmod(fixture.file.path, 0o400) == 0)
            try rejected(.unsafeFile) { _ = try fixture.installer.install() }
            try check(try fixture.text() == "{}")
        case "connector-safety":
            try FileManager.default.removeItem(at: fixture.connector.appendingPathComponent("server.js"))
            try rejected(.unsafePath) { _ = try fixture.installer.install() }
            try check(!FileManager.default.fileExists(atPath: fixture.file.path))
            let alias = fixture.root.appendingPathComponent("connector-alias")
            try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.connector)
            let other = OpenWorkConnectorInstaller(configDirectory: fixture.config, connectorDirectory: alias,
                                                    socketPath: fixture.socket, apiBaseURL: fixture.apiBaseURL)
            try rejected(.unsafePath) { _ = try other.install() }
        case "conflict":
            let package = fixture.connector.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            let variants = [try fixture.ownedEntry(socket: "/tmp/another-broker.sock"), try fixture.ownedEntry(extra: true),
                            try fixture.ownedEntry(baseURL: "https://example.invalid/v1"),
                            "{\"package\":\"\(package)\"}", "{\"source\":\"\(package)\"}",
                            "{\"package\":\"file:///other\",\"options\":{\"starkConnector\":\"\(OpenWorkConnectorInstaller.ownershipMarker)\"}}"]
            for entry in variants {
                let source = "{\"plugins\":[\(entry)]}"
                try fixture.write(source)
                try rejected(.conflict) { _ = try fixture.installer.install() }
                try rejected(.conflict) { _ = try fixture.installer.remove() }
                try check(try fixture.text() == source)
            }
        case "duplicate-owned":
            let entry = try fixture.ownedEntry()
            try fixture.write("{\"plugins\":[\(entry),\(entry)]}")
            try rejected(.conflict) { _ = try fixture.installer.install() }
        case "remove-ranges":
            let entry = try fixture.ownedEntry()
            for content in [entry, "\(entry),/* a */\"other\"", "\"one\",/* b */\(entry),/* c */\"two\"",
                            "\"one\",\(entry)", "\"one\",\(entry),/* trailing */"] {
                try fixture.write("{\"plugins\":[\(content)],\"unrelated\":true}")
                try check(try fixture.installer.remove())
                let result = try fixture.text()
                try check(!result.contains(OpenWorkConnectorInstaller.ownershipMarker))
                try check(result.contains("\"unrelated\":true"))
                for marker in ["/* a */", "/* b */", "/* c */", "/* trailing */", "\"one\"", "\"other\"", "\"two\""] {
                    if content.contains(marker) { try check(result.contains(marker)) }
                }
                try check(!(try fixture.installer.remove()))
            }
        case "concurrent-edit":
            try fixture.write("{\"original\":true}")
            var installer = fixture.installer
            installer.beforeCommit = { try fixture.write("{\"editor\":true}") }
            try rejected(.concurrentChange) { _ = try installer.install() }
            try check(try fixture.text() == "{\"editor\":true}")
            try check(try fixture.noTemporaryFiles())
        case "concurrent-create":
            var installer = fixture.installer
            installer.beforeCommit = { try fixture.write("{\"editor\":true}") }
            try rejected(.concurrentChange) { _ = try installer.install() }
            try check(try fixture.text() == "{\"editor\":true}")
        case "concurrent-symlink":
            try fixture.write("{}")
            let victim = fixture.root.appendingPathComponent("victim")
            try Data("{\"victim\":true}".utf8).write(to: victim)
            var installer = fixture.installer
            installer.beforeCommit = {
                try FileManager.default.removeItem(at: fixture.file)
                try FileManager.default.createSymbolicLink(at: fixture.file, withDestinationURL: victim)
            }
            try rejected(.unsafeFile) { _ = try installer.install() }
            try check(try String(contentsOf: victim, encoding: .utf8) == "{\"victim\":true}")
        case "concurrent-directory":
            try fixture.write("{}")
            let moved = fixture.root.appendingPathComponent("moved")
            var installer = fixture.installer
            installer.beforeCommit = {
                try FileManager.default.moveItem(at: fixture.config, to: moved)
                try FileManager.default.createDirectory(at: fixture.config, withIntermediateDirectories: false,
                                                        attributes: [.posixPermissions: 0o700])
                try fixture.write("{\"editor\":true}")
            }
            try rejected(.concurrentChange) { _ = try installer.install() }
            try check(try fixture.text() == "{\"editor\":true}")
            try check(try String(contentsOf: moved.appendingPathComponent("opencode.jsonc"), encoding: .utf8) == "{}")
        case "abort":
            try fixture.write("{\"preserve\":true}")
            var installer = fixture.installer
            installer.beforeCommit = { throw CheckError.injected }
            do { _ = try installer.install(); throw CheckError.failed }
            catch CheckError.injected { }
            try check(try fixture.text() == "{\"preserve\":true}")
            try check(try fixture.noTemporaryFiles())
        case "installer-lock":
            var installer = fixture.installer
            installer.beforeCommit = { try rejected(.concurrentChange) { _ = try fixture.installer.install() } }
            try check(try installer.install())
            try check(!(try fixture.installer.install()))
        case "invalid-socket":
            for socket in ["relative", "/tmp/invalid\npath", "/tmp/invalid\0path", "/" + String(repeating: "x", count: 104)] {
                let installer = OpenWorkConnectorInstaller(configDirectory: fixture.config,
                                                            connectorDirectory: fixture.connector, socketPath: socket, apiBaseURL: fixture.apiBaseURL)
                try rejected(.unsafePath) { _ = try installer.install() }
            }
        case "port-reconnect":
            let oldBase = "http://127.0.0.1:8000/v1"
            let entry = try fixture.ownedEntry(baseURL: oldBase)
            let source = "{\"plugins\":[\(entry)],\"unrelated\":\"\(oldBase)\"}"
            try fixture.write(source)
            try check(try fixture.installer.install())
            let result = try fixture.text()
            try check(result.contains("\"expectedBaseURL\":\"\(fixture.apiBaseURL)\""))
            try check(result.hasSuffix(",\"unrelated\":\"\(oldBase)\"}"))
            try check(!(try fixture.installer.install()))
            let otherPort = OpenWorkConnectorInstaller(configDirectory: fixture.config, connectorDirectory: fixture.connector,
                                                        socketPath: fixture.socket, apiBaseURL: "http://127.0.0.1:18134/v1")
            try check(try otherPort.remove())
            try check(!(try fixture.installer.remove()))
        case "shared-client-entry":
            // OpenWork and OpenCode can read the same global configuration and
            // install the same entry: one client's install is the other's no-op.
            try fixture.write("{\"plugins\":[\"other-plugin\"]}")
            try check(try fixture.installer.install())
            let first = try Data(contentsOf: fixture.file)
            let second = OpenWorkConnectorInstaller(configDirectory: fixture.config, connectorDirectory: fixture.connector,
                                                     socketPath: fixture.socket, apiBaseURL: fixture.apiBaseURL)
            try check(!(try second.install()))
            try check(try Data(contentsOf: fixture.file) == first)
            let moved = OpenWorkConnectorInstaller(configDirectory: fixture.config, connectorDirectory: fixture.connector,
                                                    socketPath: fixture.socket, apiBaseURL: "http://127.0.0.1:18134/v1")
            try check(try moved.install())
            let text = try fixture.text()
            try check(text.components(separatedBy: OpenWorkConnectorInstaller.ownershipMarker).count == 2)
            try check(text.contains("\"other-plugin\"") && text.contains("http://127.0.0.1:18134/v1"))
            try check(!(try moved.install()) && !(try moved.install()))
            try check(OpenWorkConnectorInstallError.invalidConfiguration.message == OpenWorkConnectorInstallError.invalidConfiguration.message("OpenWork"))
            try check(OpenWorkConnectorInstallError.invalidConfiguration.message("OpenCode")
                      == "OpenCode's settings are invalid or too large. Fix them in OpenCode, then reconnect.")
        case "namespace-reconnect":
            let oldSocket = "/private/tmp/stark-\(geteuid())-0123456789abcdef/openwork.sock"
            let entry = try fixture.ownedEntry(socket: oldSocket)
            try fixture.write("{\"plugins\":[\(entry)],\"unrelated\":\"\(oldSocket)\"}")
            try check(try fixture.installer.install())
            try check((try fixture.text()).contains("\"socketPath\":\"\(fixture.socket)\""))
            try check((try fixture.text()).hasSuffix(",\"unrelated\":\"\(oldSocket)\"}"))
            try check(!(try fixture.installer.install()))
        case "app-location-migration":
            // No old app directory exists: upgrades can already have removed it.
            let oldPackage = "file:///Users/Example/Old%20Applications/Stark.app/Contents/Resources/OpenWorkConnector"
            let entry = try fixture.ownedEntry(package: oldPackage)
            try fixture.write("{\"plugins\":[\(entry)],\"unrelated\":\"\(oldPackage)\"}")
            try check(try fixture.installer.install())
            let parsed = try JSONSerialization.jsonObject(with: Data(fixture.text().utf8)) as! [String: Any]
            let plugins = parsed["plugins"] as! [[String: Any]]
            let installed = URL(string: plugins[0]["package"] as! String)!
            try check(installed.standardizedFileURL.path == fixture.connector.standardizedFileURL.path)
            try check((try fixture.text()).hasSuffix(",\"unrelated\":\"\(oldPackage)\"}"))
            try check(!(try fixture.installer.install()))
        case "combined-migration":
            let oldPackage = "file:///removed/Stark.app/Contents/Resources/OpenWorkConnector"
            let oldSocket = "/tmp/stark-\(geteuid())-fedcba9876543210/openwork.sock"
            let entry = try fixture.ownedEntry(socket: oldSocket, baseURL: "http://127.0.0.1:8000/v1", package: oldPackage)
                .replacingOccurrences(of: "\"socketPath\":", with: "\"socketPath\" /* keep socket comment */ :")
            let unrelated = "\"unrelated\":{\"package\":\"\(oldPackage)\",\"socketPath\":\"\(oldSocket)\"}"
            try fixture.write("// keep opening\n{\"plugins\":[\"other\",/* keep plugin comment */\(entry),],\(unrelated)}// keep end\n")
            try check(try fixture.installer.install())
            let result = try fixture.text()
            for text in ["// keep opening", "/* keep socket comment */", "/* keep plugin comment */", "// keep end", unrelated, "\"other\""] {
                try check(result.contains(text))
            }
            try check(!(try fixture.installer.install()))
            try check(try fixture.noTemporaryFiles())
        case "migration-conflicts":
            let validSocket = "/private/tmp/stark-\(geteuid())-0123456789abcdef/openwork.sock"
            let validPackage = "file:///removed/Stark.app/Contents/Resources/OpenWorkConnector"
            for package in ["https://example.invalid/Stark.app/Contents/Resources/OpenWorkConnector",
                            "file://example.invalid/old/Stark.app/Contents/Resources/OpenWorkConnector",
                            "file:///old/Other.app/Contents/Resources/OpenWorkConnector",
                            "file:///old/Stark.app/Contents/Resources/other",
                            "file:///old/Stark.app/Contents/Resources/OpenWorkConnector?x=1",
                            "file:///old/Stark.app/Contents/Resources/OpenWorkConnector#fragment",
                            "file:///old/Stark.app/Contents/Resources/OpenWorkConnector/server.js"] {
                let entry = try fixture.ownedEntry(socket: validSocket, package: package)
                let source = "{\"plugins\":[\(entry)]}"
                try fixture.write(source)
                try rejected(.conflict) { _ = try fixture.installer.install() }
                try check(try fixture.text() == source)
            }
            for socket in ["/private/tmp/stark-\(geteuid() + 1)-0123456789abcdef/openwork.sock",
                           "/private/tmp/stark-\(geteuid())-0123456789ABCDEF/openwork.sock",
                           "/private/tmp/stark-\(geteuid())-0123456789abcde/openwork.sock",
                           "/private/tmp/stark-\(geteuid())-0123456789abcdef/other.sock"] {
                let entry = try fixture.ownedEntry(socket: socket, package: validPackage)
                let source = "{\"plugins\":[\(entry)]}"
                try fixture.write(source)
                try rejected(.conflict) { _ = try fixture.installer.install() }
                try check(try fixture.text() == source)
            }
            let future = try fixture.ownedEntry(socket: validSocket, package: validPackage)
                .replacingOccurrences(of: OpenWorkConnectorInstaller.ownershipMarker, with: "com.starkengine.openwork/v2")
            try fixture.write("{\"plugins\":[\(future)]}")
            try rejected(.conflict) { _ = try fixture.installer.install() }
            let old = try fixture.ownedEntry(socket: validSocket, package: validPackage)
            let current = try fixture.ownedEntry()
            try fixture.write("{\"plugins\":[\(old),\(current)]}")
            try rejected(.conflict) { _ = try fixture.installer.install() }
        case "remove-stale":
            let old = try fixture.ownedEntry(socket: "/private/tmp/stark-\(geteuid())-0123456789abcdef/openwork.sock",
                                             package: "file:///removed/Stark.app/Contents/Resources/OpenWorkConnector")
            try fixture.write("{\"plugins\":[\"other\",\(old)],\"unrelated\":true}")
            try check(try fixture.installer.remove())
            try check((try fixture.text()).contains("\"other\""))
            try check((try fixture.text()).contains("\"unrelated\":true"))
            try check(!(try fixture.installer.remove()))
        case "invalid-base-url":
            for base in ["http://127.0.0.1/v1", "https://127.0.0.1:8000/v1", "http://localhost:8000/v1",
                         "http://127.0.0.1:0/v1", "http://127.0.0.1:65536/v1", "http://127.0.0.1:8000/v1/",
                         "http://user@127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1?x=1", "http://127.0.0.1:8000/v1#x",
                         "http://example.invalid:8000/v1", "http://127.0.0.1:8000/v1/chat/completions",
                         "http://127.0.0.1:08000/v1", "HTTP://127.0.0.1:8000/v1",
                         "http://127.0.0.1:99999999999999999999999999999999999999999999999999999/v1"] {
                let installer = OpenWorkConnectorInstaller(configDirectory: fixture.config, connectorDirectory: fixture.connector,
                                                            socketPath: fixture.socket, apiBaseURL: base)
                try rejected(.unsafePath) { _ = try installer.install() }
            }
        default: throw CheckError.failed
        }
    }
}
