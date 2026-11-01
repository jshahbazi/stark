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

    private let service: String
    private let storage: DesktopConnectionApprovalStore
    private let send: ([String: Any]) -> Bool
    private let snapshot: () -> [String: Any]
    private let clock: () -> TimeInterval
    private let queue = DispatchQueue(label: "stark.openwork.setup", qos: .userInitiated)
    private var apiSession = ""
    private var closing = false
    private var approvalLoaded = false
    private var approvalLoading = false
    private var lastBaseURL = ""
    private var operation = UUID()
    private var broker: DesktopConnectionBroker?
    private var installedDirectory: URL?
    private var capabilities: [String: Capability] = [:]
    private var waiting: [String: [DesktopConnectionBroker.Reply]] = [:]
    private var acknowledgements: [String: (Bool) -> Void] = [:]

    private struct Capability {
        let peer: DesktopProcessIdentity
        let token: String
        let session: String
        let refreshed: TimeInterval
    }

    init(service: String, storage: DesktopConnectionApprovalStore? = nil,
         clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
         send: @escaping ([String: Any]) -> Bool, snapshot: @escaping () -> [String: Any]) {
        self.service = service
        self.storage = storage ?? DesktopConnectionApprovalStore(service: service)
        self.send = send
        self.snapshot = snapshot
        self.clock = clock
    }

    func receiveConnection(_ event: [String: Any]) {
        guard !closing, let session = event["api_session"] as? String, session.count == 32,
              event["base_url"] as? String != nil else { return }
        if session != apiSession {
            failPending()
            capabilities.removeAll()
            apiSession = session
        }
        let baseURL = event["base_url"] as? String ?? ""
        if !lastBaseURL.isEmpty, lastBaseURL != baseURL, approved {
            clientActive = false
            message = "Stark's port changed. Choose Reconnect, then reopen OpenWork."
        }
        lastBaseURL = baseURL
        if broker == nil {
            let next = DesktopConnectionBroker(service: service, application: { [weak self] in
                OpenWorkApplicationIdentity.running(configurationRoot: self?.testProfileDirectory)
            }, handler: { [weak self] method, peer, reply in
                guard let self else { reply(.failure(.unavailable)); return }
                self.handle(method, peer: peer, reply: reply)
            })
            do { try next.start(); broker = next }
            catch { message = "Stark could not start its private OpenWork connection. Reopen Stark and try again." }
        }
        if !approvalLoaded { loadApproval() }
    }

    func receiveAcknowledgement(_ event: [String: Any]) {
        guard let id = event["request_id"] as? String, let completion = acknowledgements.removeValue(forKey: id) else { return }
        completion(!closing && event["ok"] as? Bool == true && event["api_session"] as? String == apiSession)
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
        if method == "status" { reply(.success(status())); return }
        guard method == "connection", peer.isCurrent else { reply(.failure(.identity)); return }
        capabilities = capabilities.filter { $0.value.session == apiSession && $0.value.peer.isCurrent }
        let client = peer.clientID
        let now = clock()
        if let current = capabilities[client], now - current.refreshed < 30 {
            reply(.success(connection(current.token)))
            clientActive = true
            message = "Connected to OpenWork."
            return
        }
        if waiting[client] != nil { waiting[client]?.append(reply); return }
        guard capabilities.count < 16 || capabilities[client] != nil else { reply(.failure(.busy)); return }
        let token: String
        do { token = try capabilities[client]?.token ?? APICredentialStore.randomKey() }
        catch { reply(.failure(.backend)); return }
        let expected = apiSession
        waiting[client] = [reply]
        control(["command": "grant_api_client", "api_session": expected,
                 "client_id": client, "api_key": token, "ttl_seconds": 120]) { [weak self] accepted in
            guard let self else { return }
            let callbacks = self.waiting.removeValue(forKey: client) ?? []
            guard accepted, self.approved, expected == self.apiSession, peer.isCurrent else {
                for callback in callbacks { callback(.failure(.backend)) }
                return
            }
            self.capabilities[client] = Capability(peer: peer, token: token, session: expected,
                                                 refreshed: self.clock())
#if STARK_CONNECTION_VALIDATION
            ConnectionPrivacyAudit.remember(token)
#endif
            self.clientActive = true
            self.message = "Connected to OpenWork."
            for callback in callbacks { callback(.success(self.connection(token))) }
        }
    }

    private func connection(_ token: String) -> [String: Any] {
        var value = status()
        value["api_key"] = token
        return value
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

    func connect() {
        guard !busy, !closing, !disconnectPending, broker != nil, !apiSession.isEmpty,
              let base = snapshot()["api_base_url"] as? String, !base.isEmpty else {
            message = "Start Stark's local API with Retry API, then connect OpenWork."
            return
        }
        guard connectorDirectory != nil else { message = "This Stark build is missing its OpenWork connector. Reinstall Stark."; return }
        busy = true
        operation = UUID()
        message = "Waiting for OpenWork."
        let identifier = operation
        if let application = OpenWorkApplicationIdentity.running(configurationRoot: testProfileDirectory) {
            install(in: application, baseURL: base, operation: identifier, attempt: 0)
            return
        }
        guard let url = testApplicationURL ?? NSWorkspace.shared.urlForApplication(withBundleIdentifier: OpenWorkApplicationIdentity.bundleID),
              Self.verifyApplication(url) else {
            busy = false; message = "Install the official OpenWork macOS app, open a local workspace, then connect again."
            return
        }
        let configuration = NSWorkspace.OpenConfiguration()
        if let profile = testProfileDirectory {
            configuration.createsNewApplicationInstance = true
            do { configuration.environment = try Self.testEnvironment(profile) }
            catch { busy = false; message = "The isolated connection test profile is unavailable."; return }
        }
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
        if let application = OpenWorkApplicationIdentity.running(configurationRoot: testProfileDirectory) {
            install(in: application, baseURL: baseURL, operation: operation, attempt: 0)
        } else if attempt < 40 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.waitForApplication(baseURL: baseURL, operation: operation, attempt: attempt + 1)
            }
        } else { busy = false; message = "Open a local workspace in OpenWork, then connect again." }
    }

    private func install(in application: OpenWorkApplicationIdentity, baseURL: String, operation: UUID, attempt: Int) {
        guard self.operation == operation, !closing, let connectorDirectory, let broker else { return }
        let path = broker.path
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
                        self.message = "Stark's port changed during connection. Choose Connect or Reconnect again."
                        return
                    }
                    switch result {
                    case .success(let changed):
                        self.installedDirectory = directory
                        self.approved = true
                        self.message = changed
                            ? "Reopen OpenWork to finish connecting, then select Stark."
                            : "OpenWork is authorized. Select Stark in OpenWork."
                        if let models = self.snapshot()["models"] as? [[String: Any]],
                           models.contains(where: { $0["context_capacity_known"] as? Bool == false }) {
                            self.message = "This model has no declared context limit. Run a model with a known context limit to use it in OpenWork."
                        }
                    case .failure(let error):
                        self.message = (error as? OpenWorkConnectorInstallError)?.message ?? Self.storageMessage(error)
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
        busy = true
        approved = false
        disconnectPending = true
        clientActive = false
        capabilities.removeAll()
        for callbacks in waiting.values { for callback in callbacks { callback(.failure(.unapproved)) } }
        waiting.removeAll()
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
        if apiSession.isEmpty { finish(true) }
        else { control(["command": "revoke_api_clients", "api_session": apiSession], completion: finish) }
    }

    func shutdown() {
        closing = true
        operation = UUID()
        failPending()
        capabilities.removeAll()
        broker?.stop(); broker = nil
    }

    private func failPending() {
        let callbacks = Array(acknowledgements.values)
        acknowledgements.removeAll()
        for callback in callbacks { callback(false) }
        for callbacks in waiting.values { for callback in callbacks { callback(.failure(.backend)) } }
        waiting.removeAll()
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

    private static func verifyApplication(_ url: URL) -> Bool {
        var code: SecStaticCode?
        var requirement: SecRequirement?
        let source = "anchor apple generic and identifier \"com.differentai.openwork\" and certificate leaf[subject.OU] = \"F5DJWB4CCV\""
        return SecStaticCodeCreateWithPath(url as CFURL, [], &code) == errSecSuccess
            && SecRequirementCreateWithString(source as CFString, [], &requirement) == errSecSuccess
            && code.map { SecStaticCodeCheckValidity($0, SecCSFlags(rawValue: kSecCSStrictValidate | kSecCSCheckNestedCode), requirement) == errSecSuccess } == true
    }

    // Test bundles can isolate the official, unchanged OpenWork release's HOME
    // and data paths. No test option changes signing, peer checks or authorization.
    private var testProfileDirectory: URL? {
        guard Bundle.main.bundleIdentifier?.hasPrefix("com.starkengine.menubar.validation.") == true,
              let path = Bundle.main.object(forInfoDictionaryKey: "StarkOpenWorkTestProfile") as? String,
              path.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
    private var testApplicationURL: URL? {
        guard testProfileDirectory != nil,
              let path = Bundle.main.object(forInfoDictionaryKey: "StarkOpenWorkTestApplication") as? String else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
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
