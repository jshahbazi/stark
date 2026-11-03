import Foundation

/// Owns Stark's single private broker socket. OpenWork and OpenCode can read
/// the same global OpenCode configuration, so they share one socket path and
/// one connector entry; the broker routes each verified peer to its client.
final class DesktopConnections {
    let openWork: OpenWorkConnectionController
    let openCode: OpenCodeConnectionController
    private let service: String
    private let socket: BrokerSocket
    private var broker: DesktopConnectionBroker?
    private var closing = false

    private final class BrokerSocket { var path: String? }

    init(service: String, openWorkStorage: DesktopConnectionApprovalStore? = nil,
         openCodeStorage: OpenCodeEngineApprovalStore? = nil,
         send: @escaping ([String: Any]) -> Bool, snapshot: @escaping () -> [String: Any]) {
        let socket = BrokerSocket()
        self.service = service
        self.socket = socket
        openWork = OpenWorkConnectionController(service: service, storage: openWorkStorage, send: send,
                                                snapshot: snapshot, brokerPath: { socket.path })
        openCode = OpenCodeConnectionController(service: service, storage: openCodeStorage, send: send,
                                                snapshot: snapshot, brokerPath: { socket.path })
    }

    var brokerPath: String? { socket.path }

    func receiveConnection(_ event: [String: Any]) {
        guard !closing else { return }
        if broker == nil, let session = event["api_session"] as? String, session.count == 32, event["base_url"] is String {
            let profile = DesktopConnectionTesting.profile
            let next = DesktopConnectionBroker(service: service, applications: {
                DesktopClient.allCases.compactMap { DesktopApplicationIdentity.running($0, configurationRoot: profile) }
            }, handler: { [weak self] client, method, peer, reply in
                guard let self else { reply(.failure(.unavailable)); return }
                switch client {
                case .openWork: self.openWork.handle(method, peer: peer, reply: reply)
                case .openCode: self.openCode.handle(method, peer: peer, reply: reply)
                }
            })
            do { try next.start(); broker = next; socket.path = next.path } catch {}
        }
        openWork.receiveConnection(event)
        openCode.receiveConnection(event)
    }

    func receiveAcknowledgement(_ event: [String: Any]) {
        // Request IDs are unique per controller; each ignores the other's.
        openWork.receiveAcknowledgement(event)
        openCode.receiveAcknowledgement(event)
    }

    func shutdown() {
        closing = true
        openWork.shutdown()
        openCode.shutdown()
        socket.path = nil
        broker?.stop(); broker = nil
    }
}

/// Issues backend-acknowledged, renewable capabilities to verified peers of one
/// client and revokes them by that client's ID prefix, leaving other clients'
/// grants in place.
final class DesktopClientCapabilities {
    typealias Completion = (Result<String, DesktopConnectionError>) -> Void
    private struct Capability {
        let peer: DesktopProcessIdentity
        let token: String
        let session: String
        let refreshed: TimeInterval
    }

    let client: DesktopClient
    private let limit: Int
    private let send: ([String: Any]) -> Bool
    private let clock: () -> TimeInterval
    private(set) var apiSession = ""
    private var closing = false
    private var capabilities: [String: Capability] = [:]
    private var waiting: [String: [Completion]] = [:]
    private var acknowledgements: [String: (Bool) -> Void] = [:]

    init(client: DesktopClient, limit: Int, clock: @escaping () -> TimeInterval, send: @escaping ([String: Any]) -> Bool) {
        self.client = client
        self.limit = limit
        self.clock = clock
        self.send = send
    }

    func update(session: String) {
        guard session != apiSession else { return }
        failPending()
        capabilities.removeAll()
        apiSession = session
    }

    func receiveAcknowledgement(_ event: [String: Any]) {
        guard let id = event["request_id"] as? String, let completion = acknowledgements.removeValue(forKey: id) else { return }
        completion(!closing && event["ok"] as? Bool == true && event["api_session"] as? String == apiSession)
    }

    /// Reuses a capability refreshed in the last 30 seconds; otherwise grants or
    /// renews the same token. `approved` is checked again on acknowledgement.
    func issue(to peer: DesktopProcessIdentity, approved: @escaping () -> Bool, completion: @escaping Completion) {
        capabilities = capabilities.filter { $0.value.session == apiSession && $0.value.peer.isCurrent }
        let id = peer.clientID(client)
        if let current = capabilities[id], clock() - current.refreshed < 30 { completion(.success(current.token)); return }
        if waiting[id] != nil { waiting[id]?.append(completion); return }
        guard capabilities.count < limit || capabilities[id] != nil else { completion(.failure(.busy)); return }
        let token: String
        do { token = try capabilities[id]?.token ?? APICredentialStore.randomKey() }
        catch { completion(.failure(.backend)); return }
        let expected = apiSession
        waiting[id] = [completion]
        control(["command": "grant_api_client", "api_session": expected,
                 "client_id": id, "api_key": token, "ttl_seconds": 120]) { [weak self] accepted in
            guard let self else { return }
            let callbacks = self.waiting.removeValue(forKey: id) ?? []
            guard accepted, approved(), expected == self.apiSession, peer.isCurrent else {
                for callback in callbacks { callback(.failure(.backend)) }
                return
            }
            self.capabilities[id] = Capability(peer: peer, token: token, session: expected, refreshed: self.clock())
#if STARK_CONNECTION_VALIDATION
            ConnectionPrivacyAudit.remember(token)
#endif
            for callback in callbacks { callback(.success(token)) }
        }
    }

    /// Forgets this client's capabilities and fails waiting peers.
    func clear(_ error: DesktopConnectionError) {
        capabilities.removeAll()
        let callbacks = waiting.values.flatMap { $0 }
        waiting.removeAll()
        for callback in callbacks { callback(.failure(error)) }
    }

    /// Revokes every grant with this client's prefix in the current API session.
    func revoke(completion: @escaping (Bool) -> Void) {
        control(["command": "revoke_api_clients", "api_session": apiSession, "client_prefix": client.clientPrefix],
                completion: completion)
    }

    func shutdown() {
        closing = true
        failPending()
        capabilities.removeAll()
    }

    private func control(_ value: [String: Any], completion: @escaping (Bool) -> Void) {
        guard !closing, acknowledgements.count < 16 else { completion(false); return }
        let id = UUID().uuidString
        var payload = value
        payload["request_id"] = id
        acknowledgements[id] = completion
        guard send(payload) else { acknowledgements.removeValue(forKey: id)?(false); return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            self?.acknowledgements.removeValue(forKey: id)?(false)
        }
    }

    private func failPending() {
        let callbacks = Array(acknowledgements.values)
        acknowledgements.removeAll()
        for callback in callbacks { callback(false) }
        for callbacks in waiting.values { for callback in callbacks { callback(.failure(.backend)) } }
        waiting.removeAll()
    }
}
