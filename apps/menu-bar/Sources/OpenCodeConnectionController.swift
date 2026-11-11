import AppKit
import Darwin
import Foundation

/// OpenCode Desktop runs one detached engine that outlives the app window and
/// can be shared with the terminal. Stark therefore approves each engine
/// instance once: Connect approves the engine it verified while installing,
/// and any later engine waits for Allow or Deny in Connections.
final class OpenCodeConnectionController: ObservableObject {
    @Published private(set) var approved = false
    @Published private(set) var busy = false
    @Published private(set) var message = "OpenCode is not connected."
    @Published private(set) var clientActive = false
    @Published private(set) var disconnectPending = false
    /// A verified engine that asked for Stark and is not yet allowed or denied.
    @Published private(set) var pendingEngine: DesktopProcessIdentity?

    @Published private(set) var discoveryScope = UUID().uuidString
    private let observationClock: () -> TimeInterval
    private var observedPeer: DesktopProcessIdentity?
    private var observedSession = ""
    private var observedAt: TimeInterval?
    private var observedModels: [String: Int] = [:]

    /// A verified status handshake advertises model metadata without issuing
    /// an API capability. The connector polls every five seconds, even idle.
    func hasDiscoveredModel(apiID: String, contextTokens: Int) -> Bool {
        guard approved, !closing, !disconnectPending, !apiSession.isEmpty,
              pendingEngine == nil, observedSession == apiSession,
              let observedPeer, observedPeer.isCurrent, isApproved(observedPeer),
              let observedAt, observationClock() >= observedAt,
              observationClock() - observedAt < 30, contextTokens > 0 else { return false }
        return observedModels[apiID] == contextTokens
    }

    private func observeStatus(_ value: [String: Any], peer: DesktopProcessIdentity) {
        if observedPeer != nil && (observedPeer != peer || observedSession != apiSession) {
            discoveryScope = UUID().uuidString
        }
        observedPeer = peer; observedSession = apiSession; observedAt = observationClock()
        observedModels.removeAll()
        for model in value["models"] as? [[String: Any]] ?? [] {
            if let id = model["id"] as? String, let context = model["context_length"] as? Int,
               context > 0, model["tool_calling"] as? Bool == true {
                observedModels[id] = context
            }
        }
    }

    private let storage: OpenCodeEngineApprovalStore
    private let capabilities: DesktopClientCapabilities
    private let snapshot: () -> [String: Any]
    private let brokerPath: () -> String?
    private let queue = DispatchQueue(label: "stark.opencode.setup", qos: .userInitiated)
    private var apiSession: String { capabilities.apiSession }
    private var engines: [String] = []
    private var denied: Set<String> = []
    private var closing = false
    private var approvalLoaded = false
    private var approvalLoading = false
    private var lastBaseURL = ""
    private var operation = UUID()

    init(service: String, storage: OpenCodeEngineApprovalStore? = nil,
         clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
         send: @escaping ([String: Any]) -> Bool, snapshot: @escaping () -> [String: Any],
         brokerPath: @escaping () -> String?) {
        self.observationClock = clock
        self.storage = storage ?? OpenCodeEngineApprovalStore(service: service)
        self.capabilities = DesktopClientCapabilities(client: .openCode, limit: OpenCodeEngineApprovalStore.limit,
                                                      clock: clock, send: send)
        self.snapshot = snapshot
        self.brokerPath = brokerPath
    }

    func receiveConnection(_ event: [String: Any]) {
        guard !closing, let session = event["api_session"] as? String, session.count == 32,
              event["base_url"] as? String != nil else { return }
        if apiSession != session { discoveryScope = UUID().uuidString }
        capabilities.update(session: session)
        let baseURL = event["base_url"] as? String ?? ""
        if !lastBaseURL.isEmpty, lastBaseURL != baseURL, approved {
            clientActive = false
            message = "Stark's port changed. Choose Manage, then Reconnect to update OpenCode."
        }
        lastBaseURL = baseURL
        if brokerPath() == nil { message = "Stark could not start its private OpenCode connection. Reopen Stark and try again." }
        if !approvalLoaded { loadApproval() }
    }

    func receiveAcknowledgement(_ event: [String: Any]) {
        capabilities.receiveAcknowledgement(event)
    }

    private func loadApproval() {
        approvalLoaded = true
        approvalLoading = true
        let expected = operation
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            guard let self, self.operation == expected, self.approvalLoading else { return }
            self.message = "Waiting for Keychain access. Unlock your login Keychain or allow Stark's OpenCode Connection item."
        }
        let storage = storage
        queue.async { [weak self] in
            let result = Result { try storage.load() }
            DispatchQueue.main.async {
                guard let self, !self.closing else { return }
                self.approvalLoading = false
                guard self.operation == expected else { return }
                switch result {
                case .success(let value):
                    self.approved = value.enabled
                    self.engines = value.engines
                    self.message = value.enabled ? "OpenCode is authorized. Select Stark in OpenCode." : "OpenCode is not connected."
                case .failure(let error):
                    // Only a corrupt item may be replaced; otherwise Connect reads the list again first.
                    if case .corrupt? = error as? APICredentialError {} else { self.approvalLoaded = false }
                    self.message = Self.storageMessage(error)
                }
                if self.brokerPath() == nil {
                    self.message = "Stark could not start its private OpenCode connection. Reopen Stark and try again."
                }
            }
        }
    }

    private func isApproved(_ peer: DesktopProcessIdentity) -> Bool {
        approved && engines.contains(peer.instanceID)
    }

    private func status() -> [String: Any] {
        var value = snapshot()
        value["version"] = 1
        value["api_session"] = apiSession
        value["status"] = apiSession.isEmpty ? "unavailable" : "connected"
        return value
    }

    func handle(_ method: String, peer: DesktopProcessIdentity, reply: @escaping DesktopConnectionBroker.Reply) {
        guard !closing else { reply(.failure(.unavailable)); return }
        guard isApproved(peer) else {
            notice(peer)
            if method == "status" { reply(.success(["version": 1, "status": "not_connected"])) }
            else { reply(.failure(.unapproved)) }
            return
        }
        guard !apiSession.isEmpty, snapshot()["api_base_url"] as? String != "" else { reply(.failure(.backend)); return }
        if method == "status" {
            guard peer.isCurrent else { reply(.failure(.identity)); return }
            let value = status()
            observeStatus(value, peer: peer)
            reply(.success(value))
            return
        }
        guard method == "connection", peer.isCurrent else { reply(.failure(.identity)); return }
        capabilities.issue(to: peer, approved: { [weak self] in self?.isApproved(peer) == true }) { [weak self] result in
            guard let self else { reply(.failure(.unavailable)); return }
            switch result {
            case .success(let token):
                self.clientActive = true
                self.message = "Connected to OpenCode."
                var value = self.status()
                self.observeStatus(value, peer: peer)
                value["api_key"] = token
                reply(.success(value))
            case .failure(let error): reply(.failure(error))
            }
        }
    }

    /// Records an unapproved, verified engine once OpenCode is connected. A
    /// newer engine replaces an older pending one, and a denied engine is not
    /// asked about again while Stark runs.
    private func notice(_ peer: DesktopProcessIdentity) {
        guard approved, !busy, !disconnectPending, peer.isCurrent, !denied.contains(peer.instanceID) else { return }
        if let pending = pendingEngine, pending == peer
            || (pending.isCurrent && (peer.seconds, peer.microseconds) <= (pending.seconds, pending.microseconds)) { return }
        pendingEngine = peer
        discoveryScope = UUID().uuidString
        observedModels.removeAll()
        message = "A new OpenCode engine is asking to use Stark. Allow it if you opened or restarted OpenCode."
    }

    func allow() {
        guard let engine = pendingEngine, approved, !busy, !closing, !disconnectPending else { return }
        guard engine.isCurrent else {
            pendingEngine = nil
            message = "That OpenCode engine has stopped. Reopen OpenCode to connect it."
            return
        }
        busy = true
        operation = UUID()
        let expected = operation
        let next = OpenCodeEngineApprovals(enabled: true,
                                           engines: OpenCodeEngineApprovalStore.approving(engine.instanceID, after: engines))
        let storage = storage
        queue.async { [weak self] in
            let result = Result { try storage.save(next) }
            DispatchQueue.main.async {
                guard let self, self.operation == expected, !self.closing else { return }
                self.busy = false
                switch result {
                case .success:
                    self.engines = next.engines
                    if self.pendingEngine == engine { self.pendingEngine = nil }
                    self.message = "OpenCode engine allowed. Select Stark in OpenCode."
                case .failure(let error): self.message = Self.storageMessage(error)
                }
            }
        }
    }

    func deny() {
        guard let engine = pendingEngine, !closing else { return }
        pendingEngine = nil
        if denied.count >= 16 { denied.removeAll() }
        denied.insert(engine.instanceID)
        message = "OpenCode engine denied. Choose Manage, then Reconnect to allow the current engine."
    }

    func connect() {
        guard !busy, !closing, !disconnectPending, brokerPath() != nil, !apiSession.isEmpty,
              let base = snapshot()["api_base_url"] as? String, !base.isEmpty else {
            message = "Start Stark's local API with Retry API, then connect OpenCode."
            return
        }
        // Connect saves approvals on top of the loaded ones; never on top of an unread list.
        guard approvalLoaded, !approvalLoading else {
            if !approvalLoading { loadApproval() }
            message = "Stark is still reading its OpenCode approval. Try again shortly."
            return
        }
        guard connectorDirectory != nil else { message = "This Stark build is missing its OpenCode connector. Reinstall Stark."; return }
        busy = true
        operation = UUID()
        message = "Waiting for OpenCode."
        let identifier = operation
        let profile = DesktopConnectionTesting.profile
        if DesktopApplicationIdentity.running(.openCode, configurationRoot: profile) != nil {
            install(baseURL: base, operation: identifier, attempt: 0)
            return
        }
        guard let url = DesktopConnectionTesting.application(.openCode)
                ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: DesktopClient.openCode.bundleID),
              DesktopClientPeer.verifyApplication(url, client: .openCode) else {
            busy = false; message = "Install the official OpenCode macOS app, then connect again."
            return
        }
        let configuration = NSWorkspace.OpenConfiguration()
        if let profile {
            configuration.createsNewApplicationInstance = true
            do { configuration.environment = try Self.testEnvironment(profile) }
            catch { busy = false; message = "The isolated connection test profile is unavailable."; return }
        }
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { [weak self] _, error in
            DispatchQueue.main.async {
                guard let self, self.operation == identifier, !self.closing else { return }
                guard error == nil else { self.busy = false; self.message = "Open OpenCode, then connect again."; return }
                self.waitForApplication(baseURL: base, operation: identifier, attempt: 0)
            }
        }
    }

    private func waitForApplication(baseURL: String, operation: UUID, attempt: Int) {
        guard self.operation == operation, !closing else { return }
        if DesktopApplicationIdentity.running(.openCode, configurationRoot: DesktopConnectionTesting.profile) != nil {
            install(baseURL: baseURL, operation: operation, attempt: 0)
        } else if attempt < 40 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.waitForApplication(baseURL: baseURL, operation: operation, attempt: attempt + 1)
            }
        } else { busy = false; message = "Open OpenCode, then connect again." }
    }

    private func install(baseURL: String, operation: UUID, attempt: Int) {
        guard self.operation == operation, !closing, let connectorDirectory, let path = brokerPath() else { return }
        guard let application = DesktopApplicationIdentity.running(.openCode, configurationRoot: DesktopConnectionTesting.profile) else {
            busy = false; message = "Open OpenCode, then connect again."
            return
        }
        let storage = storage
        let current = engines
        queue.async { [weak self] in
            let engine = DesktopClientPeer.openCodeEngines(application: application).first
            let directory = engine.flatMap { Self.canonicalDirectory($0.configDirectory) }
            let result: Result<(Bool, OpenCodeEngineApprovals), Error>?
            if let engine, let directory {
                result = Result {
                    let changed = try OpenWorkConnectorInstaller(configDirectory: directory, connectorDirectory: connectorDirectory,
                                                                socketPath: path, apiBaseURL: baseURL).install()
                    let next = OpenCodeEngineApprovals(
                        enabled: true, engines: OpenCodeEngineApprovalStore.approving(engine.process.instanceID, after: current))
                    try storage.save(next)
                    return (changed, next)
                }
            } else { result = nil }
            DispatchQueue.main.async {
                guard let self, self.operation == operation, !self.closing else { return }
                if let result, let engine {
                    self.busy = false
                    switch result {
                    case .success(let (changed, next)):
                        // Keychain already holds this approval; keep memory in step with it.
                        self.engines = next.engines
                        self.approved = true
                        self.denied.remove(engine.process.instanceID)
                        if self.pendingEngine == engine.process { self.pendingEngine = nil }
                        guard self.snapshot()["api_base_url"] as? String == baseURL else {
                            self.message = "Stark's port changed during connection. Choose Manage, then Reconnect to update OpenCode."
                            return
                        }
                        self.message = changed
                            ? "Stark was added to OpenCode. Select Stark in OpenCode's model picker."
                            : "OpenCode is authorized. Select Stark in OpenCode."
                        if let models = self.snapshot()["models"] as? [[String: Any]],
                           models.contains(where: { $0["context_capacity_known"] as? Bool == false }) {
                            self.message = "This model has no declared context limit. Run a model with a known context limit to use it in OpenCode."
                        }
                        self.watch(engine.process, reported: self.message, operation: operation)
                    case .failure(let error):
                        self.message = (error as? OpenWorkConnectorInstallError)?.message(DesktopClient.openCode.displayName)
                            ?? Self.storageMessage(error)
                    }
                } else if attempt < 40 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                        self?.install(baseURL: baseURL, operation: operation, attempt: attempt + 1)
                    }
                } else {
                    self.busy = false
                    self.message = "OpenCode's background engine is not running. Wait for OpenCode to finish opening, then connect again."
                }
            }
        }
    }

    /// OpenCode can lose the engine Connect just approved, for example when its
    /// service port is already taken. Replace the success message if it does.
    private func watch(_ engine: DesktopProcessIdentity, reported: String, operation: UUID, checks: Int = 6) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self, self.operation == operation, !self.closing, self.message == reported else { return }
            guard !engine.isCurrent else {
                if checks > 1 { self.watch(engine, reported: reported, operation: operation, checks: checks - 1) }
                return
            }
            self.message = "OpenCode's background engine stopped after connecting. Reopen OpenCode, then choose Allow when Stark asks about the new engine."
        }
    }

    func disconnect() {
        guard !busy, !closing else { return }
        discoveryScope = UUID().uuidString
        observedModels.removeAll()
        observedPeer = nil
        observedAt = nil
        busy = true
        approved = false
        disconnectPending = true
        clientActive = false
        engines = []
        pendingEngine = nil
        denied.removeAll()
        capabilities.clear(.unapproved)
        operation = UUID()
        let expected = operation
        let expectedSession = apiSession
        let finish: (Bool) -> Void = { [weak self] revoked in
            guard let self, self.operation == expected else { return }
            let storage = self.storage
            self.queue.async { [weak self] in
                let result = Result { try storage.save(OpenCodeEngineApprovals()) }
                DispatchQueue.main.async {
                    guard let self, self.operation == expected, !self.closing else { return }
                    self.busy = false
                    switch result {
                    case .success:
                        if revoked || self.apiSession != expectedSession {
                            self.disconnectPending = false
                            self.message = "OpenCode disconnected."
                        } else {
                            self.message = "OpenCode disconnection is pending. Choose Retry Disconnect, or quit Stark to stop remaining requests."
                        }
                    case .failure:
                        self.message = revoked || self.apiSession != expectedSession
                            ? "OpenCode is disabled for this Stark session. Unlock Keychain and choose Disconnect again to remember this after a restart."
                            : "OpenCode disconnection is pending. Quit Stark to stop remaining requests. Unlock Keychain and choose Disconnect again to remember this after a restart."
                    }
                }
            }
        }
        // Revoke only OpenCode's grants; OpenWork keeps its connection.
        if apiSession.isEmpty { finish(true) }
        else { capabilities.revoke(completion: finish) }
    }

    func shutdown() {
        closing = true
        discoveryScope = UUID().uuidString
        operation = UUID()
        capabilities.shutdown()
    }

    private var connectorDirectory: URL? {
        guard let root = Bundle.main.resourceURL else { return nil }
        let directory = root.appendingPathComponent("OpenWorkConnector", isDirectory: true)
        return FileManager.default.fileExists(atPath: directory.appendingPathComponent("package.json").path) ? directory : nil
    }

    private static func storageMessage(_ error: Error) -> String {
        switch error as? APICredentialError {
        case .locked, .denied: return "Unlock your login Keychain and allow Stark's OpenCode Connection item, then connect again."
        case .corrupt: return "Stark's saved OpenCode approval is invalid. Choose Connect to replace that approval. Your API key has not changed."
        default: return "Stark could not save or read its OpenCode approval in Keychain. Check Keychain Access, then connect again."
        }
    }

    private static func canonicalDirectory(_ url: URL) -> URL? {
        guard let resolved = realpath(url.path, nil) else { return nil }
        defer { free(resolved) }
        return URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
    }

    /// OpenCode Desktop's own `OPENCODE_DESKTOP_TEST_ROOT` isolates its data,
    /// XDG folders, staged engine and background-service registration.
    private static func testEnvironment(_ root: URL) throws -> [String: String] {
        let base = root.appendingPathComponent("opencode-desktop", isDirectory: true)
        let locations = ["OPENCODE_DESKTOP_TEST_ROOT": "", "HOME": "home", "USERPROFILE": "home",
                         "OPENCODE_CONFIG_DIR": "config/opencode"]
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        var values: [String: String] = [:]
        for (key, relative) in locations {
            let url = relative.isEmpty ? base : base.appendingPathComponent(relative, isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            values[key] = url.path
        }
        return values
    }
}
