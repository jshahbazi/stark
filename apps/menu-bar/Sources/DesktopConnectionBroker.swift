import CryptoKit
import Darwin
import Foundation

/// Small, owner-private transport. It never reads or returns the persistent API
/// credential. The host supplies only backend-acknowledged, expiring capabilities.
final class DesktopConnectionBroker {
    typealias Reply = (Result<[String: Any], DesktopConnectionError>) -> Void
    typealias Handler = (String, DesktopProcessIdentity, @escaping Reply) -> Void
    let path: String
    private let queue = DispatchQueue(label: "stark.desktop.connection.listener")
    private let workers = DispatchQueue(label: "stark.desktop.connection.requests", attributes: .concurrent)
    private let slots = DispatchSemaphore(value: 8)
    private let lock = NSLock()
    private var source: DispatchSourceRead?
    private var descriptors = Set<Int32>()
    private var stopped = true
    private var socketIdentity: (dev_t, ino_t)?
    private let application: () -> OpenWorkApplicationIdentity?
    private let handler: Handler

    init(service: String, application: @escaping () -> OpenWorkApplicationIdentity?, handler: @escaping Handler) {
        let namespace = SHA256.hash(data: Data(service.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
        path = "/private/tmp/stark-\(getuid())-\(namespace)/openwork.sock"
        self.application = application
        self.handler = handler
    }

    deinit { stop() }

    func start() throws {
        lock.lock(); let active = !stopped; lock.unlock()
        if active { return }
        let directory = URL(fileURLWithPath: path).deletingLastPathComponent().path
        if mkdir(directory, 0o700) != 0 && errno != EEXIST { throw DesktopConnectionError.unsafePath }
        var parent = stat()
        guard lstat(directory, &parent) == 0, parent.st_mode & S_IFMT == S_IFDIR,
              parent.st_uid == getuid(), parent.st_mode & 0o777 == 0o700 else { throw DesktopConnectionError.unsafePath }
        var old = stat()
        if lstat(path, &old) == 0 {
            guard old.st_mode & S_IFMT == S_IFSOCK, old.st_uid == getuid() else { throw DesktopConnectionError.unsafePath }
            let probe = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
            guard probe >= 0 else { throw DesktopConnectionError.unavailable }
            _ = fcntl(probe, F_SETFL, O_NONBLOCK)
            let result = withAddress { Darwin.connect(probe, $0, $1) }
            let failure = errno
            Darwin.close(probe)
            guard result != 0, failure == ECONNREFUSED, Self.same(path, old.st_dev, old.st_ino), unlink(path) == 0 else {
                throw DesktopConnectionError.busy
            }
        } else if errno != ENOENT { throw DesktopConnectionError.unsafePath }
        let listener = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard listener >= 0 else { throw DesktopConnectionError.unavailable }
        _ = fcntl(listener, F_SETFD, FD_CLOEXEC)
        _ = fcntl(listener, F_SETFL, O_NONBLOCK)
        guard withAddress({ Darwin.bind(listener, $0, $1) }) == 0 else {
            Darwin.close(listener); throw DesktopConnectionError.unsafePath
        }
        var created = stat()
        guard chmod(path, 0o600) == 0, lstat(path, &created) == 0, Darwin.listen(listener, 8) == 0 else {
            Darwin.close(listener); throw DesktopConnectionError.unavailable
        }
        let next = DispatchSource.makeReadSource(fileDescriptor: listener, queue: queue)
        next.setCancelHandler { Darwin.close(listener) }
        next.setEventHandler { [weak self] in self?.accept(listener) }
        lock.lock()
        stopped = false; socketIdentity = (created.st_dev, created.st_ino); source = next
        lock.unlock()
        next.resume()
    }

    func stop() {
        lock.lock()
        stopped = true
        let previous = source; source = nil
        // Workers retain ownership of close; holding the lock during shutdown
        // prevents descriptor reuse from targeting an unrelated descriptor.
        for descriptor in descriptors { _ = Darwin.shutdown(descriptor, SHUT_RDWR) }
        let identity = socketIdentity; socketIdentity = nil
        lock.unlock()
        previous?.cancel()
        if let identity, Self.same(path, identity.0, identity.1) { _ = unlink(path) }
    }

    private static func same(_ path: String, _ device: dev_t, _ inode: ino_t) -> Bool {
        var info = stat()
        return lstat(path, &info) == 0 && info.st_mode & S_IFMT == S_IFSOCK
            && info.st_uid == getuid() && info.st_dev == device && info.st_ino == inode
    }

    private func withAddress<T>(_ body: (UnsafePointer<sockaddr>, socklen_t) -> T) -> T {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        let bytes = Array(path.utf8) + [0]
        precondition(bytes.count <= MemoryLayout.size(ofValue: address.sun_path))
        withUnsafeMutableBytes(of: &address.sun_path) { destination in destination.copyBytes(from: bytes) }
        return withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { body($0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
    }

    private func accept(_ listener: Int32) {
        while true {
            let descriptor = Darwin.accept(listener, nil, nil)
            if descriptor < 0 { return }
            guard slots.wait(timeout: .now()) == .success else { Darwin.close(descriptor); continue }
            lock.lock()
            if stopped { lock.unlock(); Darwin.close(descriptor); slots.signal(); return }
            descriptors.insert(descriptor)
            lock.unlock()
            _ = fcntl(descriptor, F_SETFD, FD_CLOEXEC)
            _ = fcntl(descriptor, F_SETFL, 0)
            var timeout = timeval(tv_sec: 2, tv_usec: 0)
            var one: Int32 = 1
            _ = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
            _ = setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
            _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one)))
            workers.async { self.serve(descriptor) }
        }
    }

    private final class ResponseBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Result<[String: Any], DesktopConnectionError>?
        func set(_ next: Result<[String: Any], DesktopConnectionError>) { lock.lock(); value = next; lock.unlock() }
        func get() -> Result<[String: Any], DesktopConnectionError>? { lock.lock(); defer { lock.unlock() }; return value }
    }

    private func serve(_ descriptor: Int32) {
        defer {
            lock.lock(); descriptors.remove(descriptor); Darwin.close(descriptor); lock.unlock()
            slots.signal()
        }
        var approved: OpenWorkApplicationIdentity?
        DispatchQueue.main.sync { approved = application() }
        guard let approved, let peer = DesktopClientPeer.authenticate(socket: descriptor, application: approved) else {
            write(["version": 1, "status": "unapproved", "error": "OpenWork identity could not be verified."], descriptor)
            return
        }
        guard let operation = readOperation(descriptor) else {
            write(["version": 1, "status": "unavailable", "error": DesktopConnectionError.protocolError.message], descriptor)
            return
        }
        let result = ResponseBox()
        let signal = DispatchSemaphore(value: 0)
        DispatchQueue.main.async { [weak self] in
            guard let self else { result.set(.failure(.unavailable)); signal.signal(); return }
            self.handler(operation, peer) { response in result.set(response); signal.signal() }
        }
        guard signal.wait(timeout: .now() + 6) == .success, peer.isCurrent,
              approved.process.isCurrent, let response = result.get() else {
            write(["version": 1, "status": "unavailable", "error": DesktopConnectionError.backend.message], descriptor)
            return
        }
        switch response {
        case .success(let value): write(value, descriptor)
        case .failure(let error): write(["version": 1, "status": "unavailable", "error": error.message], descriptor)
        }
    }

    func readOperation(_ descriptor: Int32) -> String? {
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 257)
        let deadline = ProcessInfo.processInfo.systemUptime + 2
        while data.count <= 256 {
            guard Self.setRemainingTimeout(descriptor, option: SO_RCVTIMEO, deadline: deadline) else { return nil }
            let count = recv(descriptor, &buffer, min(buffer.count, 257 - data.count), 0)
            guard count > 0, ProcessInfo.processInfo.systemUptime < deadline else { return nil }
            data.append(contentsOf: buffer.prefix(count))
            guard data.count <= 256 else { return nil }
            if let newline = data.firstIndex(of: 10) {
                guard newline == data.count - 1 else { return nil }
                data.removeLast()
                // The v1 request has two fixed ASCII keys and one of two
                // literal operations. Matching its wire grammar also rejects
                // duplicate keys rather than letting JSONSerialization repair
                // an ambiguous dictionary. JSON whitespace outside strings is
                // permitted; string contents are never changed.
                var compact = Data()
                var quoted = false
                var escaped = false
                for byte in data {
                    if quoted {
                        compact.append(byte)
                        if escaped { escaped = false }
                        else if byte == 92 { escaped = true }
                        else if byte == 34 { quoted = false }
                    } else if byte == 34 {
                        quoted = true; compact.append(byte)
                    } else if ![9, 10, 13, 32].contains(byte) { compact.append(byte) }
                }
                for operation in ["status", "connection"] {
                    for wire in ["{\"version\":1,\"operation\":\"\(operation)\"}", "{\"operation\":\"\(operation)\",\"version\":1}"] {
                        if compact == Data(wire.utf8) { return operation }
                    }
                }
                return nil
            }
        }
        return nil
    }

    func write(_ object: [String: Any], _ descriptor: Int32) {
        guard var data = try? JSONSerialization.data(withJSONObject: object), data.count <= 64 * 1024 else { return }
        data.append(10)
        let deadline = ProcessInfo.processInfo.systemUptime + 2
        let flags = fcntl(descriptor, F_GETFL)
        guard flags >= 0, fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0 else { return }
        var one: Int32 = 1
        guard setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one))) == 0 else { return }
        data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                let remaining = deadline - ProcessInfo.processInfo.systemUptime
                guard remaining > 0 else { return }
                var state = pollfd(fd: descriptor, events: Int16(POLLOUT), revents: 0)
                let ready = poll(&state, 1, Int32(max(1, ceil(remaining * 1000))))
                if ready < 0, errno == EINTR { continue }
                guard ready > 0, ProcessInfo.processInfo.systemUptime < deadline,
                      state.revents & Int16(POLLERR | POLLHUP | POLLNVAL) == 0 else { return }
                let count = Darwin.send(descriptor, bytes.baseAddress!.advanced(by: offset), bytes.count - offset, 0)
                if count < 0, [EINTR, EAGAIN, EWOULDBLOCK].contains(errno) { continue }
                if count <= 0 { return }
                offset += count
            }
        }
    }

    private static func setRemainingTimeout(_ descriptor: Int32, option: Int32, deadline: TimeInterval) -> Bool {
        let remaining = deadline - ProcessInfo.processInfo.systemUptime
        guard remaining > 0 else { return false }
        let micros = max(1, Int(remaining * 1_000_000))
        var timeout = timeval(tv_sec: micros / 1_000_000, tv_usec: Int32(micros % 1_000_000))
        return setsockopt(descriptor, SOL_SOCKET, option, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout))) == 0
    }
}
