import AppKit
import Darwin
import Foundation
import Security

private final class MockApprovalStorage: DesktopConnectionApprovalStorage {
    static let allowed = Data("openwork:F5DJWB4CCV:com.differentai.openwork:v1:approved".utf8)
    static let denied = Data("openwork:F5DJWB4CCV:com.differentai.openwork:v1:revoked".utf8)
    var value: Data?
    var readStatus: OSStatus?
    var addStatus = errSecSuccess
    var replaceStatus = errSecSuccess
    private let readLock = NSLock()
    private var readCount = 0
    var reads: Int { readLock.lock(); defer { readLock.unlock() }; return readCount }
    var adds = 0
    var replacements = 0
    init(approved: Bool? = nil) {
        value = approved.map { $0 ? Self.allowed : Self.denied }
    }
    func read() -> (OSStatus, Data?) {
        readLock.lock(); readCount += 1; readLock.unlock()
        return (readStatus ?? (value == nil ? errSecItemNotFound : errSecSuccess), value)
    }
    func add(_ next: Data) -> OSStatus {
        adds += 1
        if addStatus == errSecSuccess { value = next }
        return addStatus
    }
    func replace(_ next: Data) -> OSStatus {
        replacements += 1
        if replaceStatus == errSecSuccess { value = next; readStatus = nil }
        return replaceStatus
    }
}

private final class ResultBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value?
    func set(_ value: Value) { lock.lock(); stored = value; lock.unlock() }
    var value: Value? { lock.lock(); defer { lock.unlock() }; return stored }
}

private struct CheckFailure: Error {}

@main
struct DesktopConnectionsTests {
    private static let firstSession = String(repeating: "1", count: 32)
    private static let nextSession = String(repeating: "2", count: 32)

    static func main() {
        if let index = CommandLine.arguments.firstIndex(of: "child-listen"), index + 1 < CommandLine.arguments.count {
            // Synthetic loopback server: report its port in a file, then hold accepted connections.
            guard let (server, port) = listener() else { exit(1) }
            let report = CommandLine.arguments[index + 1]
            guard FileManager.default.createFile(atPath: report + ".tmp", contents: Data(String(port).utf8)),
                  rename(report + ".tmp", report) == 0 else { exit(1) }
            Thread.detachNewThread {
                var held: [Int32] = []
                while true {
                    let connection = accept(server, nil, nil)
                    if connection < 0 { break }
                    held.append(connection)
                }
            }
            Thread.sleep(forTimeInterval: 10)
            exit(0)
        }
        if CommandLine.arguments.count >= 2, CommandLine.arguments.dropFirst().contains("child-wait") {
            // Synthetic child for launch-environment checks; it never reads its input.
            Thread.sleep(forTimeInterval: 10)
            exit(0)
        }
        do {
            guard CommandLine.arguments.count == 2 else { throw CheckFailure() }
            switch CommandLine.arguments[1] {
            case "approval": try approval()
            case "approval-errors": try approvalErrors()
            case "approval-ui": try approvalUI()
            case "controller-ack": try controllerAck()
            case "controller-refresh": try controllerRefresh()
            case "controller-model-discovery": try controllerModelDiscovery()
            case "controller-session": try controllerSession()
            case "controller-revoke": try controllerRevoke(storageFails: false)
            case "controller-store-failure":
                try controllerRevoke(storageFails: true)
                try controllerRevoke(storageFails: true, revokeFails: true)
            case "controller-revoke-timeout": try controllerRevokeTimeout()
            case "controller-send-failure": try controllerSendFailure()
            case "controller-shutdown": try controllerShutdown()
            case "broker-auth": try brokerAuth()
            case "broker-paths": try brokerPaths()
            case "broker-framing": try brokerFraming()
            case "broker-deadline": try brokerDeadline()
            case "broker-write": try brokerWrite()
            case "broker-cleanup": try brokerCleanup()
            case "opencode-approval": try openCodeApproval()
            case "opencode-pending": try openCodePending()
            case "opencode-deny": try openCodeDeny()
            case "opencode-disabled": try openCodeDisabled()
            case "opencode-revoke": try openCodeRevoke()
            case "opencode-prune": try openCodePrune()
            case "opencode-engine-path": try openCodeEnginePath()
            case "opencode-config-directory": try openCodeConfigDirectory()
            case "launch-environment": try launchEnvironment()
            case "connection-binding": try connectionBinding()
            case "coordinator": try coordinator()
            default: throw CheckFailure()
            }
            exit(0)
        } catch { exit(1) }
    }

    private static func check(_ condition: @autoclosure () throws -> Bool) throws {
        guard try condition() else { throw CheckFailure() }
    }

    private static func wait(_ predicate: () -> Bool, timeout: TimeInterval = 4) throws {
        let end = ProcessInfo.processInfo.systemUptime + timeout
        while !predicate() {
            guard ProcessInfo.processInfo.systemUptime < end else { throw CheckFailure() }
            _ = RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
        }
    }

    private static func pump(_ duration: TimeInterval = 0.05) {
        let end = ProcessInfo.processInfo.systemUptime + duration
        while ProcessInfo.processInfo.systemUptime < end {
            _ = RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.005))
        }
    }

    private static func namespace() -> String {
        "com.starkengine.menubar.validation.desktop-connections." + UUID().uuidString
    }

    private static func broker(_ service: String, applications: @escaping () -> [DesktopApplicationIdentity] = { [] },
                               handler: @escaping DesktopConnectionBroker.Handler = { _, _, _, reply in reply(.failure(.backend)) }) -> DesktopConnectionBroker {
        DesktopConnectionBroker(service: service, applications: applications, handler: handler)
    }

    private static func cleanup(_ path: String) {
        // Every caller supplies a path derived from its fresh test namespace.
        _ = unlink(path)
        _ = rmdir(URL(fileURLWithPath: path).deletingLastPathComponent().path)
    }

    private static func approval() throws {
        let mock = MockApprovalStorage()
        let store = DesktopConnectionApprovalStore(service: namespace(), storage: mock)
        try check(try !store.load())
        try check(mock.adds == 0 && mock.replacements == 0)
        try store.save(approved: true)
        try check(try store.load())
        try check(mock.adds == 1 && mock.replacements == 0)
        try store.save(approved: false)
        try check(try !store.load())
        try check(mock.value == MockApprovalStorage.denied && mock.replacements == 1)
        mock.value = Data("synthetic invalid approval".utf8)
        do { _ = try store.load(); throw CheckFailure() }
        catch APICredentialError.corrupt {}
        try store.save(approved: true)
        try check(try store.load())
        mock.readStatus = errSecDecode
        try store.save(approved: true)
        try check(try store.load())
    }

    private static func approvalErrors() throws {
        let cases: [(OSStatus, APICredentialError)] = [
            (errSecInteractionNotAllowed, .locked), (errSecAuthFailed, .denied),
            (errSecUserCanceled, .denied), (errSecMissingEntitlement, .denied),
            (errSecDecode, .corrupt), (errSecNotAvailable, .storage)
        ]
        for (status, expected) in cases {
            let mock = MockApprovalStorage(approved: true)
            mock.readStatus = status
            let store = DesktopConnectionApprovalStore(service: namespace(), storage: mock)
            do { _ = try store.load(); throw CheckFailure() }
            catch let error as APICredentialError { try check(error == expected) }
            if status != errSecDecode {
                do { try store.save(approved: false); throw CheckFailure() }
                catch let error as APICredentialError { try check(error == expected) }
                try check(mock.adds == 0 && mock.replacements == 0)
            }
            try check(mock.value == MockApprovalStorage.allowed)
        }
        for status in [errSecInteractionNotAllowed, errSecAuthFailed, errSecNotAvailable] {
            let missing = MockApprovalStorage()
            missing.addStatus = status
            do { try DesktopConnectionApprovalStore(service: namespace(), storage: missing).save(approved: true); throw CheckFailure() }
            catch is APICredentialError {}
            try check(missing.value == nil)
            let saved = MockApprovalStorage(approved: true)
            saved.replaceStatus = status
            do { try DesktopConnectionApprovalStore(service: namespace(), storage: saved).save(approved: false); throw CheckFailure() }
            catch is APICredentialError {}
            try check(saved.value == MockApprovalStorage.allowed)
        }
    }

    private static func withController(_ mock: MockApprovalStorage = MockApprovalStorage(approved: true),
                                       sendWorks: Bool = true,
                                       models: [[String: Any]] = [],
                                       _ body: (OpenWorkConnectionController, ResultBox<[[String: Any]]>, () -> TimeInterval, (TimeInterval) -> Void) throws -> Void) throws {
        let service = namespace()
        let packets = ResultBox<[[String: Any]]>()
        packets.set([])
        var now: TimeInterval = 1000
        let value = OpenWorkConnectionController(service: service,
            storage: DesktopConnectionApprovalStore(service: service, storage: mock), clock: { now },
            send: { packet in packets.set((packets.value ?? []) + [packet]); return sendWorks },
            snapshot: { ["api_base_url": "http://127.0.0.1:8000/v1", "models": models] },
            brokerPath: { "/private/tmp/stark-synthetic-broker/openwork.sock" })
        defer { value.shutdown(); pump() }
        value.receiveConnection(["api_session": firstSession, "base_url": "http://127.0.0.1:8000/v1"])
        try wait { mock.reads > 0 && value.message != "OpenWork is not connected." || value.approved }
        try body(value, packets, { now }, { now = $0 })
    }

    private static func peer() throws -> DesktopProcessIdentity {
        guard let value = DesktopProcessIdentity.read(getpid()) else { throw CheckFailure() }
        return value
    }

    private static func acknowledge(_ controller: OpenWorkConnectionController, _ packet: [String: Any],
                                     ok: Bool = true, session: String = firstSession) {
        controller.receiveAcknowledgement(["event": "api_client", "request_id": packet["request_id"] as? String ?? "",
                                          "ok": ok, "api_session": session])
    }

    private static func token(_ result: Result<[String: Any], DesktopConnectionError>?) -> String? {
        guard case .success(let value) = result else { return nil }
        return value["api_key"] as? String
    }

    private static func approvalUI() throws {
        let mock = MockApprovalStorage()
        mock.value = Data("synthetic invalid approval".utf8)
        try withController(mock) { controller, packets, _, _ in
            try check(!controller.approved)
            try check(controller.message.contains("approval is invalid"))
            try check(controller.message.contains("API key has not changed"))
            try check(!controller.message.contains("Regenerate API Key"))
            var response: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: try peer()) { response = $0 }
            try check(token(response) == nil && (packets.value ?? []).isEmpty)
            try check(mock.adds == 0 && mock.replacements == 0)
        }
    }

    private static func controllerAck() throws {
        try withController { controller, packets, _, _ in
            var responses: [Result<[String: Any], DesktopConnectionError>] = []
            let current = try peer()
            controller.handle("connection", peer: current) { responses.append($0) }
            controller.handle("connection", peer: current) { responses.append($0) }
            try check(responses.isEmpty && packets.value?.count == 1)
            let packet = packets.value![0]
            try check(packet["command"] as? String == "grant_api_client" && packet["ttl_seconds"] as? Int == 120)
            try check(packet["client_id"] as? String == current.clientID(.openWork))
            try check(current.clientID(.openWork) == "openwork.\(current.pid).\(current.seconds).\(current.microseconds)")
            try check(APICredentialStore.valid(packet["api_key"] as? String ?? ""))
            acknowledge(controller, packet, session: nextSession)
            try check(responses.count == 2 && responses.allSatisfy { token($0) == nil })
            acknowledge(controller, packet)
            try check(responses.count == 2)
            controller.handle("connection", peer: current) { responses.append($0) }
            let next = packets.value!.last!
            try check(responses.count == 2)
            acknowledge(controller, next)
            try check(responses.count == 3 && token(responses.last) == next["api_key"] as? String)
            guard case .success(let details) = responses.last else { throw CheckFailure() }
            try check(details["expires_in_ms"] == nil)
        }
        try withController { controller, packets, _, _ in
            var responses: [Result<[String: Any], DesktopConnectionError>] = []
            controller.handle("connection", peer: try peer()) { responses.append($0) }
            let pending = packets.value!.last!
            try check(responses.isEmpty)
            try wait({ responses.count == 1 }, timeout: 3)
            try check(token(responses.last) == nil && !controller.clientActive)
            acknowledge(controller, pending)
            try check(responses.count == 1 && token(responses.last) == nil)
        }
    }

    private static func controllerRefresh() throws {
        try withController { controller, packets, clock, advance in
            let current = try peer()
            var responses: [Result<[String: Any], DesktopConnectionError>] = []
            controller.handle("connection", peer: current) { responses.append($0) }
            acknowledge(controller, packets.value!.last!)
            let existing = token(responses.last)
            try check(existing != nil)
            advance(clock() + 29)
            controller.handle("connection", peer: current) { responses.append($0) }
            try check(packets.value!.count == 1 && token(responses.last) == existing)
            advance(clock() + 2)
            controller.handle("connection", peer: current) { responses.append($0) }
            try check(packets.value!.count == 2 && responses.count == 2)
            try check(packets.value!.last!["api_key"] as? String == existing)
            acknowledge(controller, packets.value!.last!)
            try check(responses.count == 3 && token(responses.last) == existing)
        }
    }

    private static func controllerModelDiscovery() throws {
        let model: [String: Any] = ["id": "fixture/model", "context_length": 4096, "tool_calling": true]
        try withController(models: [model]) { controller, packets, clock, advance in
            let current = try peer()
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            var response: Result<[String: Any], DesktopConnectionError>?
            controller.handle("status", peer: current) { response = $0 }
            try check(token(response) == nil && packets.value?.isEmpty == true && !controller.clientActive)
            try check(controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            try check(!controller.hasDiscoveredModel(apiID: "other/model", contextTokens: 4096))
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 8192))
            advance(clock() + 29)
            try check(controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            advance(clock() + 1)
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            controller.handle("status", peer: current) { response = $0 }
            try check(controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            controller.receiveConnection(["api_session": nextSession, "base_url": "http://127.0.0.1:8000/v1"])
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            let invalid = DesktopProcessIdentity(pid: 0, parent: 0, seconds: 0, microseconds: 0)
            controller.handle("status", peer: invalid) { response = $0 }
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            controller.handle("status", peer: current) { response = $0 }
            try check(controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            let previousScope = controller.discoveryScope
            controller.disconnect()
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
            try check(controller.discoveryScope != previousScope)
        }
        try withController(models: [["id": "fixture/model", "context_length": 4096, "tool_calling": false]]) { controller, _, _, _ in
            controller.handle("status", peer: try peer()) { _ in }
            try check(!controller.hasDiscoveredModel(apiID: "fixture/model", contextTokens: 4096))
        }
    }

    private static func controllerSession() throws {
        try withController { controller, packets, _, _ in
            var responses: [Result<[String: Any], DesktopConnectionError>] = []
            let current = try peer()
            controller.handle("connection", peer: current) { responses.append($0) }
            let old = packets.value!.last!
            controller.receiveConnection(["api_session": nextSession, "base_url": "http://127.0.0.1:8000/v1"])
            try check(responses.count == 1 && token(responses.last) == nil)
            acknowledge(controller, old)
            try check(responses.count == 1)
            controller.handle("connection", peer: current) { responses.append($0) }
            let next = packets.value!.last!
            try check(next["api_session"] as? String == nextSession && next["api_key"] as? String != old["api_key"] as? String)
            acknowledge(controller, next, session: nextSession)
            try check(token(responses.last) != nil)
            let invalid = DesktopProcessIdentity(pid: 0, parent: 0, seconds: 0, microseconds: 0)
            var rejected: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: invalid) { rejected = $0 }
            try check(token(rejected) == nil && packets.value!.count == 2)
        }
    }

    private static func controllerRevoke(storageFails: Bool, revokeFails: Bool = false) throws {
        let mock = MockApprovalStorage(approved: true)
        if storageFails { mock.replaceStatus = errSecInteractionNotAllowed }
        try withController(mock) { controller, packets, _, _ in
            var response: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: try peer()) { response = $0 }
            acknowledge(controller, packets.value!.last!)
            try check(token(response) != nil && controller.clientActive)
            controller.disconnect()
            let revoke = packets.value!.last!
            try check(!controller.approved && !controller.clientActive)
            try check(revoke["command"] as? String == "revoke_api_clients")
            try check(revoke["api_key"] == nil && revoke["client_id"] == nil)
            try check(revoke["client_prefix"] as? String == "openwork." && revoke.count == 4)
            acknowledge(controller, revoke, ok: !revokeFails)
            try wait { !controller.busy }
            try check(!controller.approved)
            if storageFails {
                try check(mock.value == MockApprovalStorage.allowed)
                if revokeFails {
                    try check(controller.disconnectPending && controller.message.contains("disconnection is pending"))
                    try check(controller.message.contains("Quit Stark") && controller.message.contains("Keychain"))
                    try check(!controller.message.contains("disabled for this Stark session"))
                    mock.replaceStatus = errSecSuccess
                    controller.disconnect()
                    acknowledge(controller, packets.value!.last!)
                    try wait { !controller.busy }
                    try check(!controller.disconnectPending && controller.message == "OpenWork disconnected.")
                    try check(mock.value == MockApprovalStorage.denied)
                } else {
                    try check(controller.message.contains("this Stark session") && !controller.message.contains("OpenWork disconnected."))
                }
            } else {
                try check(mock.value == MockApprovalStorage.denied)
                try check(controller.message == "OpenWork disconnected.")
            }
            var denied: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: try peer()) { denied = $0 }
            try check(token(denied) == nil)
        }
    }

    private static func controllerRevokeTimeout() throws {
        try withController { controller, packets, _, _ in
            controller.disconnect()
            let timedOut = packets.value!.last!
            try wait({ !controller.busy }, timeout: 3)
            try check(controller.disconnectPending && !controller.approved)
            try check(!controller.message.contains("OpenWork disconnected."))
            let count = packets.value!.count
            controller.connect()
            try check(controller.disconnectPending && !controller.approved && packets.value!.count == count)
            acknowledge(controller, timedOut)
            try check(controller.disconnectPending)
            controller.disconnect()
            let retry = packets.value!.last!
            try check(retry["request_id"] as? String != timedOut["request_id"] as? String)
            acknowledge(controller, retry)
            try wait { !controller.busy }
            try check(!controller.disconnectPending && !controller.approved)
            try check(controller.message == "OpenWork disconnected.")
        }
    }

    private static func controllerSendFailure() throws {
        try withController(sendWorks: false) { controller, packets, _, _ in
            var response: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: try peer()) { response = $0 }
            try check(packets.value!.count == 1 && token(response) == nil && !controller.clientActive)
        }
    }

    private static func controllerShutdown() throws {
        try withController { controller, packets, _, _ in
            var responses: [Result<[String: Any], DesktopConnectionError>] = []
            controller.handle("connection", peer: try peer()) { responses.append($0) }
            let pending = packets.value!.last!
            controller.shutdown()
            try check(responses.count == 1 && token(responses.last) == nil)
            acknowledge(controller, pending)
            controller.handle("connection", peer: try peer()) { responses.append($0) }
            try check(responses.count == 2 && responses.allSatisfy { token($0) == nil })
        }
    }

    private static func connect(_ path: String) throws -> Int32 {
        let descriptor = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw CheckFailure() }
        var timeout = timeval(tv_sec: 2, tv_usec: 0)
        var one: Int32 = 1
        _ = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
        _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one)))
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        let bytes = Array(path.utf8) + [0]
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: bytes) }
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard connected == 0 else { Darwin.close(descriptor); throw CheckFailure() }
        return descriptor
    }

    private static func socketReply(_ path: String) throws -> [String: Any] {
        let descriptor = try connect(path)
        defer { Darwin.close(descriptor) }
        let data = Data("{\"version\":1,\"operation\":\"connection\"}\n".utf8)
        _ = data.withUnsafeBytes { Darwin.send(descriptor, $0.baseAddress, $0.count, 0) }
        var output = Data()
        var bytes = [UInt8](repeating: 0, count: 1024)
        while output.count < 65536 {
            let count = recv(descriptor, &bytes, bytes.count, 0)
            if count <= 0 { break }
            output.append(contentsOf: bytes.prefix(count))
            if output.last == 10 { break }
        }
        guard let value = try JSONSerialization.jsonObject(with: output) as? [String: Any] else { throw CheckFailure() }
        return value
    }

    private static func brokerAuth() throws {
        let own = try peer()
        let openWork = DesktopApplicationIdentity(client: .openWork, process: own, bundleURL: URL(fileURLWithPath: "/private/synthetic/OpenWork.app"))
        let openCode = DesktopApplicationIdentity(client: .openCode, process: own, bundleURL: URL(fileURLWithPath: "/private/synthetic/OpenCode.app"))
        for applications in [[], [openWork], [openCode], [openWork, openCode]] {
            var calls = 0
            let value = broker(namespace(), applications: { applications }, handler: { _, _, _, reply in calls += 1; reply(.failure(.backend)) })
            defer { value.stop(); pump(); cleanup(value.path) }
            try value.start()
            let box = ResultBox<Result<[String: Any], Error>>()
            DispatchQueue.global().async { box.set(Result { try socketReply(value.path) }) }
            try wait { box.value != nil }
            let response = try box.value!.get()
            try check(response["api_key"] == nil && response["status"] as? String == "unapproved")
            try check(calls == 0)
        }
    }

    private static func expectStartFailure(_ value: DesktopConnectionBroker) throws {
        do { try value.start(); value.stop(); throw CheckFailure() }
        catch is DesktopConnectionError {}
    }

    private static func brokerPaths() throws {
        for kind in ["regular", "symlink", "unsafe-directory", "symlink-directory"] {
            let value = broker(namespace())
            let parent = URL(fileURLWithPath: value.path).deletingLastPathComponent().path
            let target = "/private/tmp/stark-connection-target-" + UUID().uuidString
            defer { value.stop(); pump(); cleanup(value.path); _ = unlink(parent); _ = rmdir(parent); _ = unlink(target); _ = rmdir(target) }
            if kind == "symlink-directory" {
                try check(mkdir(target, 0o700) == 0 && symlink(target, parent) == 0)
            } else {
                try check(mkdir(parent, kind == "unsafe-directory" ? 0o755 : 0o700) == 0)
                if kind == "regular" { try Data("synthetic marker".utf8).write(to: URL(fileURLWithPath: value.path)) }
                if kind == "symlink" {
                    try Data("synthetic marker".utf8).write(to: URL(fileURLWithPath: target))
                    try check(symlink(target, value.path) == 0)
                }
            }
            try expectStartFailure(value)
            if kind == "regular" { try check(try Data(contentsOf: URL(fileURLWithPath: value.path)) == Data("synthetic marker".utf8)) }
            if kind == "symlink" { try check(try Data(contentsOf: URL(fileURLWithPath: target)) == Data("synthetic marker".utf8)) }
        }
        let service = namespace()
        let value = broker(service)
        defer { value.stop(); pump(); cleanup(value.path) }
        try value.start()
        var file = stat(), directory = stat()
        try check(lstat(value.path, &file) == 0 && file.st_mode & S_IFMT == S_IFSOCK && file.st_mode & 0o777 == 0o600)
        try check(lstat(URL(fileURLWithPath: value.path).deletingLastPathComponent().path, &directory) == 0 && directory.st_mode & 0o777 == 0o700)
        try expectStartFailure(broker(service))
        try check(unlink(value.path) == 0)
        let replacement = Data("synthetic replacement".utf8)
        try replacement.write(to: URL(fileURLWithPath: value.path))
        value.stop()
        try check(try Data(contentsOf: URL(fileURLWithPath: value.path)) == replacement)
    }

    private static func pair() throws -> [Int32] {
        var descriptors = [Int32](repeating: -1, count: 2)
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else { throw CheckFailure() }
        for descriptor in descriptors {
            var timeout = timeval(tv_sec: 2, tv_usec: 0)
            var one: Int32 = 1
            _ = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
            _ = setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
            _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one)))
        }
        return descriptors
    }

    private static func parse(_ frame: Data) throws -> String? {
        let descriptors = try pair()
        defer { descriptors.forEach { Darwin.close($0) } }
        _ = frame.withUnsafeBytes { Darwin.send(descriptors[1], $0.baseAddress, $0.count, 0) }
        _ = Darwin.shutdown(descriptors[1], SHUT_WR)
        return broker(namespace()).readOperation(descriptors[0])
    }

    private static func brokerFraming() throws {
        try check(try parse(Data("{\"version\":1,\"operation\":\"status\"}\n".utf8)) == "status")
        try check(try parse(Data("{\"version\":1,\"operation\":\"connection\"}\n".utf8)) == "connection")
        let valid = "{\"version\":1,\"operation\":\"status\"}"
        try check(try parse(Data((valid + String(repeating: " ", count: 255 - valid.utf8.count) + "\n").utf8)) == "status")
        for frame in [
            "{}\n", "[]\n", "null\n", "{", valid,
            "{\"version\":true,\"operation\":\"status\"}\n",
            "{\"version\":2,\"operation\":\"status\"}\n",
            "{\"version\":1,\"operation\":\"unknown\"}\n",
            "{\"version\":1,\"operation\":\"status\",\"extra\":1}\n",
            "{\"version\":1,\"operation\":\"status\",\"operation\":\"connection\"}\n",
            valid + "\n" + valid + "\n",
            valid + String(repeating: " ", count: 256 - valid.utf8.count) + "\n"
        ] { try check(try parse(Data(frame.utf8)) == nil) }
    }

    private static func brokerDeadline() throws {
        let descriptors = try pair()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            defer { Darwin.close(descriptors[1]); done.signal() }
            let end = ProcessInfo.processInfo.systemUptime + 4
            var byte: UInt8 = 32
            while ProcessInfo.processInfo.systemUptime < end {
                if Darwin.send(descriptors[1], &byte, 1, 0) != 1 { break }
                Thread.sleep(forTimeInterval: 0.04)
            }
        }
        let began = ProcessInfo.processInfo.systemUptime
        let result = broker(namespace()).readOperation(descriptors[0])
        Darwin.close(descriptors[0])
        let elapsed = ProcessInfo.processInfo.systemUptime - began
        try check(done.wait(timeout: .now() + 2) == .success)
        try check(result == nil && elapsed < 2.9)
    }

    private static func fdCount() -> Int {
        (0..<1024).reduce(0) { $0 + (fcntl(Int32($1), F_GETFD) >= 0 ? 1 : 0) }
    }

    private static func brokerWrite() throws {
        let descriptors = try pair()
        var bufferSize: Int32 = 4096
        _ = setsockopt(descriptors[0], SOL_SOCKET, SO_SNDBUF, &bufferSize, socklen_t(MemoryLayout.size(ofValue: bufferSize)))
        _ = fcntl(descriptors[0], F_SETFL, O_NONBLOCK)
        var filler = [UInt8](repeating: 42, count: 4096)
        var filled = 0
        while filled < 1024 * 1024 {
            let sent = Darwin.send(descriptors[0], &filler, filler.count, 0)
            if sent <= 0 { break }
            filled += sent
        }
        try check(filled > 0 && filled < 1024 * 1024)
        _ = fcntl(descriptors[0], F_SETFL, 0)
        let done = DispatchSemaphore(value: 0)
        let received = ResultBox<Int>()
        DispatchQueue.global().async {
            defer { Darwin.close(descriptors[1]); done.signal() }
            var bytes = [UInt8](repeating: 0, count: 128)
            var count = 0
            let end = ProcessInfo.processInfo.systemUptime + 4
            while ProcessInfo.processInfo.systemUptime < end {
                let next = recv(descriptors[1], &bytes, bytes.count, MSG_DONTWAIT)
                if next == 0 { break }
                if next > 0 { count += next }
                Thread.sleep(forTimeInterval: 0.04)
            }
            received.set(count)
        }
        let value = broker(namespace())
        let began = ProcessInfo.processInfo.systemUptime
        value.write(["version": 1, "payload": String(repeating: "x", count: 60000)], descriptors[0])
        let elapsed = ProcessInfo.processInfo.systemUptime - began
        _ = Darwin.shutdown(descriptors[0], SHUT_RDWR)
        Darwin.close(descriptors[0])
        try check(done.wait(timeout: .now() + 2) == .success)
        try check(elapsed < 2.9 && (received.value ?? 0) > 0)

        let oversized = try pair()
        defer { oversized.forEach { Darwin.close($0) } }
        value.write(["payload": String(repeating: "x", count: 70000)], oversized[0])
        var byte: UInt8 = 0
        try check(recv(oversized[1], &byte, 1, MSG_DONTWAIT) < 0)
    }

    private static func brokerCleanup() throws {
        let value = broker(namespace())
        defer { value.stop(); pump(); cleanup(value.path) }
        try value.start()
        let warm = try connect(value.path)
        value.stop(); Darwin.close(warm); pump(0.1)
        let before = fdCount()
        for _ in 0..<20 {
            try value.start()
            let socket = try connect(value.path)
            value.stop()
            Darwin.close(socket)
            pump(0.03)
            try check(access(value.path, F_OK) != 0)
        }
        pump(0.2)
        try check(fdCount() <= before)
        try value.start()
        let box = ResultBox<Result<[String: Any], Error>>()
        DispatchQueue.global().async { box.set(Result { try socketReply(value.path) }) }
        try wait { box.value != nil }
        let response = try box.value!.get()
        try check(response["api_key"] == nil)
    }

    // MARK: OpenCode

    private static func approvals(_ mock: MockApprovalStorage) throws -> OpenCodeEngineApprovals {
        try OpenCodeEngineApprovalStore(service: namespace(), storage: mock).load()
    }

    private static func openCodeStorage(enabled: Bool, engines: [String] = []) -> MockApprovalStorage {
        let mock = MockApprovalStorage()
        mock.value = OpenCodeEngineApprovalStore.encode(OpenCodeEngineApprovals(enabled: enabled, engines: engines))
        return mock
    }

    private static func withOpenCode(_ mock: MockApprovalStorage,
                                     _ body: (OpenCodeConnectionController, ResultBox<[[String: Any]]>) throws -> Void) throws {
        let service = namespace()
        let packets = ResultBox<[[String: Any]]>()
        packets.set([])
        let value = OpenCodeConnectionController(service: service,
            storage: OpenCodeEngineApprovalStore(service: service, storage: mock), clock: { 1000 },
            send: { packet in packets.set((packets.value ?? []) + [packet]); return true },
            snapshot: { ["api_base_url": "http://127.0.0.1:8000/v1", "models": []] },
            brokerPath: { "/private/tmp/stark-synthetic-broker/openwork.sock" })
        defer { value.shutdown(); pump() }
        value.receiveConnection(["api_session": firstSession, "base_url": "http://127.0.0.1:8000/v1"])
        try wait { mock.reads > 0 }
        pump(0.1)
        try body(value, packets)
    }

    private static func acknowledge(_ controller: OpenCodeConnectionController, _ packet: [String: Any], ok: Bool = true) {
        controller.receiveAcknowledgement(["event": "api_client", "request_id": packet["request_id"] as? String ?? "",
                                          "ok": ok, "api_session": firstSession])
    }

    private static func notConnected(_ result: Result<[String: Any], DesktopConnectionError>?) -> Bool {
        guard case .success(let value) = result else { return false }
        return value.count == 2 && value["status"] as? String == "not_connected" && value["api_key"] == nil
    }

    private static func openCodeApproval() throws {
        let mock = MockApprovalStorage()
        let store = OpenCodeEngineApprovalStore(service: namespace(), storage: mock)
        try check(try store.load() == OpenCodeEngineApprovals())
        let engines = ["123.1700000000.42", "99999.1700000001.999999"]
        try store.save(OpenCodeEngineApprovals(enabled: true, engines: engines))
        try check(mock.adds == 1 && String(data: mock.value!, encoding: .utf8)
                  == "opencode:5NZ4Q7NXJ4:ai.opencode.desktop:v1:enabled:123.1700000000.42,99999.1700000001.999999")
        try check(try store.load() == OpenCodeEngineApprovals(enabled: true, engines: engines))
        try store.save(OpenCodeEngineApprovals(enabled: true))
        try check(try store.load() == OpenCodeEngineApprovals(enabled: true))
        try store.save(OpenCodeEngineApprovals())
        try check(String(data: mock.value!, encoding: .utf8) == "opencode:5NZ4Q7NXJ4:ai.opencode.desktop:v1:revoked")
        try check(try store.load() == OpenCodeEngineApprovals() && mock.replacements == 2)
        for invalid in [OpenCodeEngineApprovals(enabled: false, engines: ["1.2.3"]),
                        OpenCodeEngineApprovals(enabled: true, engines: (0..<9).map { "\($0 + 2).1.1" }),
                        OpenCodeEngineApprovals(enabled: true, engines: ["1.2"])] {
            do { try store.save(invalid); throw CheckFailure() }
            catch APICredentialError.storage {}
        }
        let prefix = "opencode:5NZ4Q7NXJ4:ai.opencode.desktop:v1:"
        for corrupt in ["synthetic invalid approval", prefix, prefix + "approved", prefix + "enabled:1.2",
                        prefix + "enabled:1.2.3,1.2.3", prefix + "enabled:1.2.3,", prefix + "enabled:a.2.3",
                        prefix + "enabled:1.2.1234567", prefix + "enabled:" + (0..<9).map { "\($0 + 2).1.1" }.joined(separator: ","),
                        "openwork:F5DJWB4CCV:com.differentai.openwork:v1:approved",
                        "opencode:F5DJWB4CCV:ai.opencode.desktop:v1:enabled:1.2.3"] {
            mock.value = Data(corrupt.utf8)
            do { _ = try store.load(); throw CheckFailure() }
            catch APICredentialError.corrupt {}
        }
        try store.save(OpenCodeEngineApprovals(enabled: true, engines: ["5.6.7"]))
        try check(try store.load().engines == ["5.6.7"])
        mock.readStatus = errSecDecode
        try store.save(OpenCodeEngineApprovals(enabled: true, engines: ["5.6.8"]))
        try check(try store.load().engines == ["5.6.8"])
        for (status, expected) in [(errSecInteractionNotAllowed, APICredentialError.locked), (errSecAuthFailed, .denied)] {
            let locked = openCodeStorage(enabled: true, engines: ["5.6.7"])
            locked.readStatus = status
            let lockedStore = OpenCodeEngineApprovalStore(service: namespace(), storage: locked)
            do { _ = try lockedStore.load(); throw CheckFailure() }
            catch let error as APICredentialError { try check(error == expected) }
            do { try lockedStore.save(OpenCodeEngineApprovals()); throw CheckFailure() }
            catch let error as APICredentialError { try check(error == expected) }
            try check(locked.adds == 0 && locked.replacements == 0)
        }
    }

    private static func openCodePending() throws {
        let mock = openCodeStorage(enabled: true, engines: ["1.1.1"])
        try withOpenCode(mock) { controller, packets in
            let current = try peer()
            try check(controller.approved && controller.pendingEngine == nil)
            var status: Result<[String: Any], DesktopConnectionError>?
            controller.handle("status", peer: current) { status = $0 }
            try check(notConnected(status) && controller.pendingEngine == current)
            try check(controller.message.contains("asking to use Stark"))
            // A newer engine replaces the pending one; an older one does not.
            let newer = try child()
            guard let restarted = DesktopProcessIdentity.read(newer) else { reap(newer); throw CheckFailure() }
            controller.handle("status", peer: restarted) { _ in }
            try check(controller.pendingEngine == restarted)
            controller.handle("status", peer: current) { _ in }
            try check(controller.pendingEngine == restarted)
            reap(newer)
            controller.handle("status", peer: current) { _ in }
            try check(controller.pendingEngine == current)
            var connection: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: current) { connection = $0 }
            guard case .failure(.unapproved) = connection else { throw CheckFailure() }
            try check((packets.value ?? []).isEmpty && token(connection) == nil)
            controller.allow()
            try wait { !controller.busy }
            try check(controller.pendingEngine == nil && controller.message.contains("allowed"))
            // The stale engine is dropped; the allowed one is persisted.
            try check(try approvals(mock) == OpenCodeEngineApprovals(enabled: true, engines: [current.instanceID]))
            var granted: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: current) { granted = $0 }
            let packet = packets.value!.last!
            try check(packet["command"] as? String == "grant_api_client" && packet["ttl_seconds"] as? Int == 120)
            try check(packet["client_id"] as? String == "opencode." + current.instanceID)
            acknowledge(controller, packet)
            try check(token(granted) == packet["api_key"] as? String && controller.clientActive)
            guard case .success(let details) = granted else { throw CheckFailure() }
            try check(details["status"] as? String == "connected" && details["api_session"] as? String == firstSession)
            var refreshed: Result<[String: Any], DesktopConnectionError>?
            controller.handle("status", peer: current) { refreshed = $0 }
            guard case .success(let value) = refreshed else { throw CheckFailure() }
            try check(value["status"] as? String == "connected" && value["api_key"] == nil)
        }
    }

    private static func openCodeDeny() throws {
        let mock = openCodeStorage(enabled: true)
        try withOpenCode(mock) { controller, packets in
            let current = try peer()
            controller.handle("status", peer: current) { _ in }
            try check(controller.pendingEngine == current)
            controller.deny()
            try check(controller.pendingEngine == nil && controller.message.contains("denied"))
            var status: Result<[String: Any], DesktopConnectionError>?
            controller.handle("status", peer: current) { status = $0 }
            try check(notConnected(status) && controller.pendingEngine == nil)
            controller.allow()
            try check(!controller.busy && (packets.value ?? []).isEmpty)
            try check(try approvals(mock) == OpenCodeEngineApprovals(enabled: true))
            let stopped = DesktopProcessIdentity(pid: 0, parent: 0, seconds: 0, microseconds: 0)
            controller.handle("status", peer: stopped) { _ in }
            try check(controller.pendingEngine == nil)
        }
    }

    private static func openCodeDisabled() throws {
        for mock in [MockApprovalStorage(), openCodeStorage(enabled: false)] {
            try withOpenCode(mock) { controller, packets in
                let current = try peer()
                try check(!controller.approved && controller.message == "OpenCode is not connected.")
                var status: Result<[String: Any], DesktopConnectionError>?
                controller.handle("status", peer: current) { status = $0 }
                try check(notConnected(status) && controller.pendingEngine == nil)
                var connection: Result<[String: Any], DesktopConnectionError>?
                controller.handle("connection", peer: current) { connection = $0 }
                guard case .failure(.unapproved) = connection else { throw CheckFailure() }
                try check((packets.value ?? []).isEmpty && mock.adds == 0 && mock.replacements == 0)
            }
        }
        let corrupt = MockApprovalStorage()
        corrupt.value = Data("synthetic invalid approval".utf8)
        try withOpenCode(corrupt) { controller, _ in
            try check(!controller.approved && controller.message.contains("approval is invalid"))
            try check(controller.message.contains("API key has not changed"))
        }
        // Connect never saves over approvals that are still being read.
        let saved = openCodeStorage(enabled: true, engines: ["1.1.1"])
        let service = namespace()
        let early = OpenCodeConnectionController(service: service,
            storage: OpenCodeEngineApprovalStore(service: service, storage: saved), send: { _ in true },
            snapshot: { ["api_base_url": "http://127.0.0.1:8000/v1", "models": []] },
            brokerPath: { "/private/tmp/stark-synthetic-broker/openwork.sock" })
        defer { early.shutdown(); pump() }
        early.connect()
        try check(early.message.contains("Start Stark's local API"))
        early.receiveConnection(["api_session": firstSession, "base_url": "http://127.0.0.1:8000/v1"])
        early.connect()
        try check(!early.busy && early.message.contains("still reading"))
        try wait { early.approved }
        try check(try approvals(saved).engines == ["1.1.1"] && saved.adds == 0 && saved.replacements == 0)
        // A locked read is not an empty list: Connect reads the approvals again before saving.
        let locked = openCodeStorage(enabled: true, engines: ["1.1.1"])
        locked.readStatus = errSecInteractionNotAllowed
        let lockedService = namespace()
        let unlocked = OpenCodeConnectionController(service: lockedService,
            storage: OpenCodeEngineApprovalStore(service: lockedService, storage: locked), send: { _ in true },
            snapshot: { ["api_base_url": "http://127.0.0.1:8000/v1", "models": []] },
            brokerPath: { "/private/tmp/stark-synthetic-broker/openwork.sock" })
        defer { unlocked.shutdown(); pump() }
        unlocked.receiveConnection(["api_session": firstSession, "base_url": "http://127.0.0.1:8000/v1"])
        try wait { unlocked.message.contains("Unlock your login Keychain") }
        locked.readStatus = nil
        let reads = locked.reads
        unlocked.connect()
        try check(!unlocked.busy && unlocked.message.contains("still reading"))
        try wait { unlocked.approved }
        try check(locked.reads > reads)
        try check(try approvals(locked).engines == ["1.1.1"] && locked.adds == 0 && locked.replacements == 0)
    }

    private static func openCodeRevoke() throws {
        let current = try peer()
        let mock = openCodeStorage(enabled: true, engines: [current.instanceID])
        try withOpenCode(mock) { controller, packets in
            var response: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: current) { response = $0 }
            acknowledge(controller, packets.value!.last!)
            try check(token(response) != nil && controller.clientActive)
            var waiting: Result<[String: Any], DesktopConnectionError>?
            controller.handle("connection", peer: current) { waiting = $0 }
            try check(waiting != nil)
            controller.disconnect()
            let revoke = packets.value!.last!
            try check(revoke["command"] as? String == "revoke_api_clients" && revoke["client_prefix"] as? String == "opencode.")
            try check(revoke["client_id"] == nil && revoke["api_key"] == nil && revoke.count == 4)
            try check(!controller.approved && !controller.clientActive && controller.disconnectPending)
            acknowledge(controller, revoke)
            try wait { !controller.busy }
            try check(!controller.disconnectPending && controller.message == "OpenCode disconnected.")
            try check(String(data: mock.value!, encoding: .utf8) == "opencode:5NZ4Q7NXJ4:ai.opencode.desktop:v1:revoked")
            var status: Result<[String: Any], DesktopConnectionError>?
            controller.handle("status", peer: current) { status = $0 }
            try check(notConnected(status) && controller.pendingEngine == nil)
        }
        let pending = openCodeStorage(enabled: true, engines: [current.instanceID])
        try withOpenCode(pending) { controller, packets in
            controller.disconnect()
            try wait({ !controller.busy }, timeout: 3)
            try check(controller.disconnectPending && controller.message.contains("disconnection is pending"))
            controller.disconnect()
            acknowledge(controller, packets.value!.last!)
            try wait { !controller.busy }
            try check(!controller.disconnectPending && controller.message == "OpenCode disconnected.")
        }
    }

    private static func spawn(_ arguments: [String], environment: [String]) throws -> pid_t {
        var pid: pid_t = 0
        let argv = arguments.map { strdup($0) } + [nil]
        let envp = environment.map { strdup($0) } + [nil]
        defer { (argv + envp).forEach { free($0) } }
        guard posix_spawn(&pid, arguments[0], nil, nil, argv, envp) == 0 else { throw CheckFailure() }
        return pid
    }

    private static func reap(_ pid: pid_t) {
        var status: Int32 = 0
        _ = kill(pid, SIGKILL)
        _ = waitpid(pid, &status, 0)
    }

    private static func child(_ arguments: [String] = [], environment: [String] = [], leading: [String] = []) throws -> pid_t {
        try spawn([CommandLine.arguments[0]] + leading + ["child-wait"] + arguments, environment: ["PATH=/usr/bin:/bin"] + environment)
    }

    private static func openCodePrune() throws {
        var children: [pid_t] = []
        defer { children.forEach(reap) }
        for _ in 0..<9 { children.append(try child()) }
        let ids = try children.map { pid -> String in
            guard let identity = DesktopProcessIdentity.read(pid) else { throw CheckFailure() }
            return identity.instanceID
        }
        try check(Set(ids).count == 9 && ids.allSatisfy { DesktopProcessIdentity.isRunning(instanceID: $0) })
        try check(OpenCodeEngineApprovalStore.approving(ids[8], after: Array(ids[0..<8])) == Array(ids[1...8]))
        try check(OpenCodeEngineApprovalStore.approving(ids[8], after: ["1.1.1", ids[0], ids[8]]) == [ids[0], ids[8]])
        try check(OpenCodeEngineApprovalStore.approving(ids[0], after: []) == [ids[0]])
        try check(!DesktopProcessIdentity.isRunning(instanceID: "1.1.1") && !DesktopProcessIdentity.isRunning(instanceID: "x"))
        reap(children[0])
        try check(!DesktopProcessIdentity.isRunning(instanceID: ids[0]))
        try check(OpenCodeEngineApprovalStore.approving(ids[2], after: [ids[0], ids[1]]) == [ids[1], ids[2]])
    }

    private static func openCodeEnginePath() throws {
        guard let temporary = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw CheckFailure() }
        let base = URL(fileURLWithPath: String(cString: temporary), isDirectory: true)
            .appendingPathComponent("stark-engine-path-\(UUID().uuidString)", isDirectory: true)
        free(temporary)
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appendingPathComponent("cli", isDirectory: true)
        let outside = base.appendingPathComponent("outside", isDirectory: true)
        func file(_ relative: String, under folder: URL = root) throws -> String {
            let url = folder.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("synthetic".utf8).write(to: url)
            return url.path
        }
        try check(DesktopClientPeer.isStagedOpenCodeEngine(try file("2.2.5-arm64/opencode-cli"), root: root))
        try check(DesktopClientPeer.isStagedOpenCodeEngine(try file(String(repeating: "a", count: 64) + "/opencode-cli"), root: root))
        try check(DesktopClientPeer.isStagedOpenCodeEngine(try file("v_1.x-y/opencode-cli"), root: root))
        for relative in ["opencode-cli", "2.2.5-arm64/nested/opencode-cli", "2.2.5-arm64/opencode", "bad stage/opencode-cli",
                         String(repeating: "a", count: 65) + "/opencode-cli", "2.2.5+arm64/opencode-cli"] {
            try check(!DesktopClientPeer.isStagedOpenCodeEngine(try file(relative), root: root))
        }
        try check(!DesktopClientPeer.isStagedOpenCodeEngine(root.appendingPathComponent("missing/opencode-cli").path, root: root))
        try check(!DesktopClientPeer.isStagedOpenCodeEngine(try file("2.2.5-arm64/opencode-cli", under: outside), root: root))
        try check(symlink(outside.appendingPathComponent("2.2.5-arm64").path, root.appendingPathComponent("linked").path) == 0)
        try check(!DesktopClientPeer.isStagedOpenCodeEngine(root.appendingPathComponent("linked/opencode-cli").path, root: root))
        try check(!DesktopClientPeer.isStagedOpenCodeEngine(try file("2.2.5-arm64/opencode-cli"), root: base.appendingPathComponent("absent")))
        guard let home = DesktopClientPeer.userHome() else { throw CheckFailure() }
        try check(DesktopClientPeer.openCodeEngineRoot(profile: nil)?.path
                  == home.appendingPathComponent("Library/Application Support/ai.opencode.desktop/cli").path)
        try check(DesktopClientPeer.openCodeEngineRoot(profile: base)?.path == base.appendingPathComponent("opencode-desktop/desktop/cli").path)
    }

    private static func openCodeConfigDirectory() throws {
        guard let home = DesktopClientPeer.userHome() else { throw CheckFailure() }
        let cases: [(DesktopLaunchEnvironment, String)] = [
            (DesktopLaunchEnvironment(configDirectory: "/synthetic/config", xdgConfigHome: "/synthetic/xdg", home: "/synthetic/home"), "/synthetic/config"),
            (DesktopLaunchEnvironment(xdgConfigHome: "/synthetic/xdg", home: "/synthetic/home"), "/synthetic/xdg/opencode"),
            (DesktopLaunchEnvironment(home: "/synthetic/home"), "/synthetic/home/.config/opencode"),
            (DesktopLaunchEnvironment(), home.appendingPathComponent(".config/opencode").path),
        ]
        for (environment, expected) in cases {
            try check(DesktopClientPeer.openCodeConfigDirectory(environment)?.path == expected)
        }
    }

    private static func launchEnvironment() throws {
        typealias Views = (DesktopLaunchEnvironment?, DesktopLaunchEnvironment?, DesktopLaunchEnvironment?)
        // OpenWork, the OpenCode Desktop app, and OpenCode's engine.
        func read(arguments: [String] = [], _ environment: [String], leading: [String] = []) throws -> Views {
            let pid = try child(arguments, environment: environment, leading: leading)
            defer { reap(pid) }
            return (DesktopClientPeer.launchEnvironment(pid, client: .openWork), DesktopClientPeer.launchEnvironment(pid, client: .openCode),
                    DesktopClientPeer.launchEnvironment(pid, client: .openCode, engine: true))
        }
        let paths = ["OPENCODE_CONFIG_DIR=/synthetic/config", "XDG_CONFIG_HOME=/synthetic/xdg", "HOME=/synthetic/home"]
        let (openWork, openCode, engine) = try read(paths)
        try check(openWork == DesktopLaunchEnvironment(configDirectory: "/synthetic/config"))
        try check(openCode == DesktopLaunchEnvironment(configDirectory: "/synthetic/config", xdgConfigHome: "/synthetic/xdg", home: "/synthetic/home"))
        try check(engine == openCode)
        let (_, unset, _) = try read(["XDG_CONFIG_HOME=", "OPENCODE_DESKTOP_REMOTE_DEBUGGING_PORT=", "DYLD_FRAMEWORK_PATH=", "BUN_INSPECT="])
        try check(unset == DesktopLaunchEnvironment())
        // OpenCode refuses loader and inspector variables; OpenWork's checks are unchanged.
        for variable in ["DYLD_FRAMEWORK_PATH=/synthetic/missing", "DYLD_LIBRARY_PATH=/synthetic/missing", "BUN_INSPECT=1", "BUN_BE_BUN=1",
                         "BUN_INSPECT_CONNECT_TO=ws://127.0.0.1:9", "BUN_INSPECT_PRELOAD=/synthetic/preload.js",
                         "OPENCODE_DESKTOP_REMOTE_DEBUGGING_PORT=9222", "XDG_CONFIG_HOME=relative", "HOME=relative"] {
            let (work, code, engine) = try read([variable])
            try check(work != nil && code == nil && engine == nil)
        }
        for variable in ["BUN_OPTIONS=--preload /synthetic/x.js", "ELECTRON_RUN_AS_NODE=1", "OPENCODE_CONFIG_DIR=relative", "OPENCODE_CONFIG_DIR="] {
            let (work, code, engine) = try read([variable])
            try check(work == nil && code == nil && engine == nil)
        }
        // Desktop copies login-shell variables into its compiled engine, which ignores
        // NODE_OPTIONS. The Electron app and OpenWork still refuse it.
        let (work, app, shellEngine) = try read(["NODE_OPTIONS=--max-old-space-size=4096"])
        try check(work == nil && app == nil && shellEngine == DesktopLaunchEnvironment())
        for argument in ["--inspect", "--inspect-brk=9229", "--preload", "-e", "--eval", "--require=x", "--remote-debugging-port=9222"] {
            let (work, code, engine) = try read(arguments: [argument], [])
            try check(work == nil && code == nil && engine == nil)
        }
        let (served, code, _) = try read(arguments: ["serve", "--service"], [])
        try check(served != nil && code == DesktopLaunchEnvironment())
        // Only `serve --service` as the first two arguments marks Desktop's background engine.
        let (_, service, _) = try read(paths, leading: ["serve", "--service"])
        try check(service == DesktopLaunchEnvironment(configDirectory: "/synthetic/config", xdgConfigHome: "/synthetic/xdg",
                                                      home: "/synthetic/home", service: true))
        for leading in [["serve"], ["--service", "serve"], ["run", "--service"], ["serve", "--stdio"]] {
            let (_, other, _) = try read([], leading: leading)
            try check(other == DesktopLaunchEnvironment())
        }
    }

    /// A loopback listener on an ephemeral port.
    private static func listener() -> (Int32, UInt16)? {
        let server = socket(AF_INET, SOCK_STREAM, 0)
        guard server >= 0 else { return nil }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(server, $0, length) == 0 && listen(server, 8) == 0 && getsockname(server, $0, &length) == 0
            }
        }
        guard bound else { Darwin.close(server); return nil }
        return (server, UInt16(bigEndian: address.sin_port))
    }

    private static func connect(port: UInt16) throws -> Int32 {
        let client = socket(AF_INET, SOCK_STREAM, 0)
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = port.bigEndian
        let connected = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(client, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
        guard client >= 0, connected else { throw CheckFailure() }
        return client
    }

    /// The engine binding: Desktop must hold a loopback connection that the
    /// engine process itself accepted.
    private static func connectionBinding() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("stark-binding-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        var servers: [(pid_t, UInt16)] = []
        defer { servers.filter { $0.0 > 0 }.forEach { reap($0.0) } }
        for name in ["first", "second"] {
            let report = folder.appendingPathComponent(name).path
            let pid = try spawn([CommandLine.arguments[0], "child-listen", report], environment: ["PATH=/usr/bin:/bin"])
            servers.append((pid, 0))
            try wait({ FileManager.default.fileExists(atPath: report) }, timeout: 5)
            guard let text = try? String(contentsOfFile: report, encoding: .utf8), let port = UInt16(text) else { throw CheckFailure() }
            servers[servers.count - 1].1 = port
        }
        let (first, firstPort) = servers[0]
        let (second, secondPort) = servers[1]
        let me = getpid()
        try check(!DesktopClientPeer.isConnected(me, to: first) && !DesktopClientPeer.isConnected(me, to: second))
        let toFirst = try connect(port: firstPort)
        try wait({ DesktopClientPeer.isConnected(me, to: first) }, timeout: 3)
        // A connection to one listener never binds another process, in either direction.
        try check(!DesktopClientPeer.isConnected(me, to: second) && !DesktopClientPeer.isConnected(first, to: me))
        try check(!DesktopClientPeer.isConnected(second, to: first) && !DesktopClientPeer.isConnected(first, to: first))
        let toSecond = try connect(port: secondPort)
        try wait({ DesktopClientPeer.isConnected(me, to: second) }, timeout: 3)
        Darwin.close(toFirst)
        try check(!DesktopClientPeer.isConnected(me, to: first) && DesktopClientPeer.isConnected(me, to: second))
        Darwin.close(toSecond)
        try check(!DesktopClientPeer.isConnected(me, to: second))
        // An exited server leaves nothing to match.
        reap(first)
        servers[0].0 = 0
        try check(!DesktopClientPeer.isConnected(me, to: first))
    }

    private static func coordinator() throws {
        let service = namespace()
        let current = try peer()
        let openWorkMock = MockApprovalStorage(approved: true)
        let openCodeMock = openCodeStorage(enabled: true, engines: [current.instanceID])
        let packets = ResultBox<[[String: Any]]>()
        packets.set([])
        let connections = DesktopConnections(service: service,
            openWorkStorage: DesktopConnectionApprovalStore(service: service, storage: openWorkMock),
            openCodeStorage: OpenCodeEngineApprovalStore(service: service, storage: openCodeMock),
            send: { packet in packets.set((packets.value ?? []) + [packet]); return true },
            snapshot: { ["api_base_url": "http://127.0.0.1:8000/v1", "models": []] })
        let expectedPath = broker(service).path
        defer { connections.shutdown(); pump(); cleanup(expectedPath) }
        try check(connections.brokerPath == nil)
        connections.receiveConnection(["api_session": firstSession, "base_url": "http://127.0.0.1:8000/v1"])
        try check(connections.brokerPath == expectedPath)
        var info = stat()
        try check(lstat(expectedPath, &info) == 0 && info.st_mode & S_IFMT == S_IFSOCK && info.st_mode & 0o777 == 0o600)
        try wait { connections.openWork.approved && connections.openCode.approved }
        var work: Result<[String: Any], DesktopConnectionError>?
        var code: Result<[String: Any], DesktopConnectionError>?
        connections.openWork.handle("connection", peer: current) { work = $0 }
        connections.openCode.handle("connection", peer: current) { code = $0 }
        let grants = packets.value!
        try check(grants.count == 2 && grants.map { $0["client_id"] as? String ?? "" }
                  == ["openwork." + current.instanceID, "opencode." + current.instanceID])
        let ack: ([String: Any]) -> [String: Any] = { ["event": "api_client", "request_id": $0["request_id"] as? String ?? "",
                                                       "ok": true, "api_session": firstSession] }
        connections.receiveAcknowledgement(ack(grants[1]))
        try check(work == nil && token(code) == grants[1]["api_key"] as? String)
        connections.receiveAcknowledgement(ack(grants[0]))
        try check(token(work) == grants[0]["api_key"] as? String)
        // An unverified peer on the shared socket gets no capability from either client.
        let box = ResultBox<Result<[String: Any], Error>>()
        DispatchQueue.global().async { box.set(Result { try socketReply(expectedPath) }) }
        try wait { box.value != nil }
        let response = try box.value!.get()
        try check(response["api_key"] == nil && response["status"] as? String == "unapproved")
        connections.openCode.disconnect()
        let revoke = packets.value!.last!
        try check(revoke["client_prefix"] as? String == "opencode.")
        connections.receiveAcknowledgement(ack(revoke))
        try wait { !connections.openCode.busy }
        try check(connections.openCode.message == "OpenCode disconnected.")
        try check(connections.openWork.approved && connections.openWork.clientActive)
        try check(openWorkMock.value == MockApprovalStorage.allowed)
        connections.receiveConnection(["api_session": nextSession, "base_url": "http://127.0.0.1:8000/v1"])
        try check(connections.brokerPath == expectedPath)
        connections.shutdown()
        pump()
        try check(connections.brokerPath == nil && access(expectedPath, F_OK) != 0)
        var late: Result<[String: Any], DesktopConnectionError>?
        connections.openWork.handle("connection", peer: current) { late = $0 }
        try check(token(late) == nil)
    }
}
