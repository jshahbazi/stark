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
        do {
            guard CommandLine.arguments.count == 2 else { throw CheckFailure() }
            switch CommandLine.arguments[1] {
            case "approval": try approval()
            case "approval-errors": try approvalErrors()
            case "approval-ui": try approvalUI()
            case "controller-ack": try controllerAck()
            case "controller-refresh": try controllerRefresh()
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

    private static func broker(_ service: String, application: @escaping () -> OpenWorkApplicationIdentity? = { nil },
                               handler: @escaping DesktopConnectionBroker.Handler = { _, _, reply in reply(.failure(.backend)) }) -> DesktopConnectionBroker {
        DesktopConnectionBroker(service: service, application: application, handler: handler)
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
                                       _ body: (OpenWorkConnectionController, ResultBox<[[String: Any]]>, () -> TimeInterval, (TimeInterval) -> Void) throws -> Void) throws {
        let service = namespace()
        let path = broker(service).path
        let packets = ResultBox<[[String: Any]]>()
        packets.set([])
        var now: TimeInterval = 1000
        let value = OpenWorkConnectionController(service: service,
            storage: DesktopConnectionApprovalStore(service: service, storage: mock), clock: { now },
            send: { packet in packets.set((packets.value ?? []) + [packet]); return sendWorks },
            snapshot: { ["api_base_url": "http://127.0.0.1:8000/v1", "models": []] })
        defer { value.shutdown(); pump(); cleanup(path) }
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
            try check(packet["client_id"] as? String == current.clientID)
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
        for application: OpenWorkApplicationIdentity? in [nil, OpenWorkApplicationIdentity(process: own, bundleURL: URL(fileURLWithPath: "/private/synthetic/OpenWork.app"))] {
            var calls = 0
            let value = broker(namespace(), application: { application }, handler: { _, _, reply in calls += 1; reply(.failure(.backend)) })
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
}
