import AppKit
import Darwin
import Foundation
import Security

final class OpenWorkConnectionController: ObservableObject {
    @Published private(set) var approved = false
    @Published private(set) var busy = false
    @Published private(set) var message = "OpenWork is not connected."
    @Published private(set) var clientActive = false
    @Published private(set) var disconnectPending = false
    @Published private(set) var discoveryScope = UUID().uuidString

    /// A verified status handshake advertises model metadata without issuing
    /// an API capability. The connector polls every five seconds, even idle.
    func hasDiscoveredModel(apiID: String, contextTokens: Int) -> Bool {
        guard approved, !closing, !disconnectPending, !apiSession.isEmpty,
              observedSession == apiSession, observedPeer?.isCurrent == true,
              let observedAt, observationClock() >= observedAt,
              observationClock() - observedAt < 30, contextTokens > 0 else { return false }
        return observedModels[apiID] == contextTokens
    }

    private let storage: DesktopConnectionApprovalStore
    private let capabilities: DesktopClientCapabilities
    private let snapshot: () -> [String: Any]
    private let brokerPath: () -> String?
    private let queue = DispatchQueue(label: "stark.openwork.setup", qos: .userInitiated)
    private var apiSession: String { capabilities.apiSession }
    private var closing = false
    private var approvalLoaded = false
    private var approvalLoading = false
    private var lastBaseURL = ""
    private var operation = UUID()
    private var installedDirectory: URL?
    private let observationClock: () -> TimeInterval
    private var observedPeer: DesktopProcessIdentity?
    private var observedSession = ""
    private var observedAt: TimeInterval?
    private var observedModels: [String: Int] = [:]

    init(service: String, storage: DesktopConnectionApprovalStore? = nil,
         clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
         send: @escaping ([String: Any]) -> Bool, snapshot: @escaping () -> [String: Any],
         brokerPath: @escaping () -> String?) {
        self.storage = storage ?? DesktopConnectionApprovalStore(service: service)
        self.capabilities = DesktopClientCapabilities(client: .openWork, limit: 16, clock: clock, send: send)
        self.snapshot = snapshot
        self.brokerPath = brokerPath
        self.observationClock = clock
    }

    func receiveConnection(_ event: [String: Any]) {
        guard !closing, let session = event["api_session"] as? String, session.count == 32,
              event["base_url"] as? String != nil else { return }
        capabilities.update(session: session)
        let baseURL = event["base_url"] as? String ?? ""
        if !lastBaseURL.isEmpty, lastBaseURL != baseURL, approved {
            clientActive = false
            message = "Stark's port changed. Choose Manage, then Reconnect, and reopen OpenWork."
        }
        lastBaseURL = baseURL
        if brokerPath() == nil { message = "Stark could not start its private OpenWork connection. Reopen Stark and try again." }
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
            self.message = "Waiting for Keychain access. Unlock your login Keychain or allow Stark's OpenWork Connection item."
        }
        let storage = storage
        queue.async { [weak self] in
            let result = Result { try storage.load() }
            DispatchQueue.main.async {
                guard let self, !self.closing else { return }
                self.approvalLoading = false
                guard self.operation == expected else { return }
                switch result {
                case .success(let allowed):
                    self.approved = allowed
                    self.message = allowed ? "OpenWork is authorized. Select Stark in OpenWork." : "OpenWork is not connected."
                case .failure(let error): self.message = Self.storageMessage(error)
                }
                if self.brokerPath() == nil {
                    self.message = "Stark could not start its private OpenWork connection. Reopen Stark and try again."
                }
            }
        }
    }

    private func status() -> [String: Any] {
        var value = snapshot()
        value["version"] = 1
        value["api_session"] = apiSession
        value["status"] = approved ? (apiSession.isEmpty ? "unavailable" : "connected") : "not_connected"
        return value
    }

    func handle(_ method: String, peer: DesktopProcessIdentity, reply: @escaping DesktopConnectionBroker.Reply) {
        guard !closing else { reply(.failure(.unavailable)); return }
        guard approved else { reply(.success(["version": 1, "status": "not_connected"])); return }
        guard !apiSession.isEmpty, snapshot()["api_base_url"] as? String != "" else { reply(.failure(.backend)); return }
        if method == "status" {
            guard peer.isCurrent else { reply(.failure(.identity)); return }
            let value = status()
            observeStatus(value, peer: peer)
            reply(.success(value))
            return
        }
        guard method == "connection", peer.isCurrent else { reply(.failure(.identity)); return }
        capabilities.issue(to: peer, approved: { [weak self] in self?.approved == true }) { [weak self] result in
            guard let self else { reply(.failure(.unavailable)); return }
            switch result {
            case .success(let token):
                self.clientActive = true
                self.message = "Connected to OpenWork."
                let value = self.connection(token)
                self.observeStatus(value, peer: peer)
                reply(.success(value))
            case .failure(let error): reply(.failure(error))
            }
        }
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

    private func connection(_ token: String) -> [String: Any] {
        var value = status()
        value["api_key"] = token
        return value
    }

    func openApplication() {
        guard !closing else { return }
        if let application = DesktopApplicationIdentity.running(.openWork, configurationRoot: testProfileDirectory),
           let running = NSRunningApplication(processIdentifier: application.process.pid) {
            running.activate()
            return
        }
        guard let url = DesktopConnectionTesting.application(.openWork)
                ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: DesktopClient.openWork.bundleID),
              DesktopClientPeer.verifyApplication(url, client: .openWork) else {
            message = "Install the official OpenWork macOS app, then choose Check installation."
            return
        }
        let configuration: NSWorkspace.OpenConfiguration
        do { configuration = try Self.launchConfiguration(profile: testProfileDirectory) }
        catch { message = "The isolated connection test profile is unavailable."; return }
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { [weak self] _, error in
            DispatchQueue.main.async {
                guard let self, !self.closing else { return }
                if error != nil { self.message = "OpenWork could not open. Open it from Applications, then return here." }
            }
        }
    }

    static func launchConfiguration(profile: URL?) throws -> NSWorkspace.OpenConfiguration {
        let configuration = NSWorkspace.OpenConfiguration()
        if let profile {
            configuration.createsNewApplicationInstance = true
            configuration.environment = try testEnvironment(profile)
        }
        return configuration
    }

    func connect() {
        guard !busy, !closing, !disconnectPending, brokerPath() != nil, !apiSession.isEmpty,
              let base = snapshot()["api_base_url"] as? String, !base.isEmpty else {
            message = "Start Stark's local API with Retry API, then connect OpenWork."
            return
        }
        guard connectorDirectory != nil else { message = "This Stark build is missing its OpenWork connector. Reinstall Stark."; return }
        busy = true
        operation = UUID()
        message = "Waiting for OpenWork."
        let identifier = operation
        if let application = DesktopApplicationIdentity.running(.openWork, configurationRoot: testProfileDirectory) {
            install(in: application, baseURL: base, operation: identifier, attempt: 0)
            return
        }
        guard let url = DesktopConnectionTesting.application(.openWork)
                ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: DesktopClient.openWork.bundleID),
              DesktopClientPeer.verifyApplication(url, client: .openWork) else {
            busy = false; message = "Install the official OpenWork macOS app, open a local workspace, then connect again."
            return
        }
        let configuration: NSWorkspace.OpenConfiguration
        do { configuration = try Self.launchConfiguration(profile: testProfileDirectory) }
        catch { busy = false; message = "The isolated connection test profile is unavailable."; return }
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { [weak self] _, error in
            DispatchQueue.main.async {
                guard let self, self.operation == identifier, !self.closing else { return }
                guard error == nil else { self.busy = false; self.message = "Open OpenWork, then connect again."; return }
                self.waitForApplication(baseURL: base, operation: identifier, attempt: 0)
            }
        }
    }

    private func waitForApplication(baseURL: String, operation: UUID, attempt: Int) {
        guard self.operation == operation, !closing else { return }
        if let application = DesktopApplicationIdentity.running(.openWork, configurationRoot: testProfileDirectory) {
            install(in: application, baseURL: baseURL, operation: operation, attempt: 0)
        } else if attempt < 40 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.waitForApplication(baseURL: baseURL, operation: operation, attempt: attempt + 1)
            }
        } else { busy = false; message = "Open a local workspace in OpenWork, then connect again." }
    }

    private func install(in application: DesktopApplicationIdentity, baseURL: String, operation: UUID, attempt: Int) {
        guard self.operation == operation, !closing, let connectorDirectory, let path = brokerPath() else { return }
        let storage = storage
        let testRoot = testProfileDirectory
        queue.async { [weak self] in
            let directory = DesktopClientPeer.engineConfigDirectory(application: application).flatMap(Self.canonicalDirectory).flatMap { value -> URL? in
                if let testRoot, !DesktopClientPeer.isWithin(value.path, root: testRoot) { return nil }
                return value
            }
            let result: Result<Bool, Error>?
            if let directory {
                result = Result {
                    let changed = try OpenWorkConnectorInstaller(configDirectory: directory, connectorDirectory: connectorDirectory,
                                                                socketPath: path, apiBaseURL: baseURL).install()
                    try storage.save(approved: true)
                    return changed
                }
            } else { result = nil }
            DispatchQueue.main.async {
                guard let self, self.operation == operation, !self.closing else { return }
                if let result {
                    self.busy = false
                    guard self.snapshot()["api_base_url"] as? String == baseURL else {
                        self.message = "Stark's port changed during connection. Connect again, or choose Manage then Reconnect if OpenWork already has access."
                        return
                    }
                    switch result {
                    case .success:
                        self.installedDirectory = directory
                        self.approved = true
                        // Matching disk settings do not prove the running engine
                        // has reloaded its connector or discarded stale auth state.
                        self.message = "Connection settings refreshed. Quit and reopen OpenWork to finish connecting, then select Stark and the running model. Closing its window is not enough."
                        if let models = self.snapshot()["models"] as? [[String: Any]],
                           models.contains(where: { $0["context_capacity_known"] as? Bool == false }) {
                            self.message = "This model has no declared context limit. Run a model with a known context limit to use it in OpenWork."
                        }
                    case .failure(let error):
                        self.message = (error as? OpenWorkConnectorInstallError)?.message(DesktopClient.openWork.displayName)
                            ?? Self.storageMessage(error)
                    }
                } else if attempt < 40 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                        self?.install(in: application, baseURL: baseURL, operation: operation, attempt: attempt + 1)
                    }
                } else {
                    self.busy = false
                    self.message = "Open a local workspace in OpenWork, then choose Connect again."
                }
            }
        }
    }

    func disconnect() {
        guard !busy, !closing else { return }
        discoveryScope = UUID().uuidString
        busy = true
        approved = false
        disconnectPending = true
        clientActive = false
        capabilities.clear(.unapproved)
        operation = UUID()
        let expected = operation
        let expectedSession = apiSession
        let finish: (Bool) -> Void = { [weak self] revoked in
            guard let self, self.operation == expected else { return }
            let storage = self.storage
            self.queue.async { [weak self] in
                let result = Result { try storage.save(approved: false) }
                DispatchQueue.main.async {
                    guard let self, self.operation == expected, !self.closing else { return }
                    self.busy = false
                    switch result {
                    case .success:
                        if revoked || self.apiSession != expectedSession {
                            self.disconnectPending = false
                            self.message = "OpenWork disconnected."
                        } else {
                            self.message = "OpenWork disconnection is pending. Choose Retry Disconnect, or quit Stark to stop remaining requests."
                        }
                    case .failure:
                        self.message = revoked || self.apiSession != expectedSession
                            ? "OpenWork is disabled for this Stark session. Unlock Keychain and choose Disconnect again to remember this after a restart."
                            : "OpenWork disconnection is pending. Quit Stark to stop remaining requests. Unlock Keychain and choose Disconnect again to remember this after a restart."
                    }
                }
            }
        }
        // Revoke only OpenWork's grants; OpenCode keeps its connection.
        if apiSession.isEmpty { finish(true) }
        else { capabilities.revoke(completion: finish) }
    }

    func shutdown() {
        closing = true
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
        case .locked, .denied: return "Unlock your login Keychain and allow Stark's OpenWork Connection item, then connect again."
        case .corrupt: return "Stark's saved OpenWork approval is invalid. Choose Connect to replace that approval. Your API key has not changed."
        default: return "Stark could not save or read its OpenWork approval in Keychain. Check Keychain Access, then connect again."
        }
    }

    private static func canonicalDirectory(_ url: URL) -> URL? {
        guard let resolved = realpath(url.path, nil) else { return nil }
        defer { free(resolved) }
        return URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
    }

    // Test bundles can isolate the official, unchanged OpenWork release's HOME
    // and data paths. No test option changes signing, peer checks or authorization.
    private var testProfileDirectory: URL? { DesktopConnectionTesting.profile }
    private static func testEnvironment(_ root: URL) throws -> [String: String] {
        let locations = ["HOME": "home", "USERPROFILE": "home", "XDG_CONFIG_HOME": "xdg/config", "XDG_DATA_HOME": "xdg/data",
                         "XDG_CACHE_HOME": "xdg/cache", "XDG_STATE_HOME": "xdg/state", "OPENWORK_ELECTRON_USERDATA": "electron/user-data",
                         "OPENWORK_DESKTOP_BOOTSTRAP_PATH": "openwork/config/desktop-bootstrap.json", "OPENWORK_SERVER_CONFIG": "openwork/config/server.json",
                         "OPENWORK_ENV_STORE": "openwork/config/env.json", "OPENWORK_TOKEN_STORE": "openwork/config/tokens.json",
                         "OPENWORK_RUNTIME_DB": "openwork/config/runtime.sqlite", "OPENWORK_DATA_DIR": "openwork/data",
                         "OPENCODE_CONFIG_DIR": "opencode/config", "OPENCODE_DB": "opencode/data/opencode.db"]
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        var values: [String: String] = [:]
        for (key, relative) in locations {
            let url = root.appendingPathComponent(relative)
            let directory = relative.contains(".") ? url.deletingLastPathComponent() : url
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            values[key] = url.path
        }
        return values
    }
}
