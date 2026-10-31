import Foundation
import LocalAuthentication
import Security

enum APICredentialError: Error {
    case locked, denied, corrupt, storage, random

    var message: String {
        switch self {
        case .locked:
            return "Unlock your login Keychain and allow Stark to access its Local API item in Keychain Access, then choose Retry API. Your API key has not changed."
        case .denied:
            return "Allow Stark to access its local API credential in Keychain Access, then choose Retry API. Your API key has not changed."
        case .corrupt:
            return "The stored API credential is invalid. Use Regenerate API Key to replace it; connected apps will need the replacement key."
        case .storage:
            return "Stark cannot read or save its API credential in Keychain. Check your login Keychain, then choose Retry API. Your API key has not changed."
        case .random:
            return "Stark could not securely generate an API key. Choose Retry API. Your stored API key has not changed."
        }
    }
}

/// Keychain operations are injected so failure tests never access a user's credential.
protocol APICredentialStorage {
    func read() -> (OSStatus, Data?)
    func add(_ data: Data) -> OSStatus
    func replace(_ data: Data) -> OSStatus
}

struct KeychainAPICredentialStorage: APICredentialStorage {
    let service: String
    static let account = "bearer-key-v1"

    private var query: [String: Any] {
        let context = LAContext()
        context.interactionNotAllowed = true
        return [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: Self.account,
         kSecAttrSynchronizable as String: false,
         // A background launch must fail visibly rather than opening an unexpected prompt.
         kSecUseAuthenticationContext as String: context]
    }

    func read() -> (OSStatus, Data?) {
        var value = query
        value[kSecReturnData as String] = true
        value[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(value as CFDictionary, &result)
        return (status, result as? Data)
    }

    func add(_ data: Data) -> OSStatus {
        var value = query
        value[kSecValueData as String] = data
        value[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        value[kSecAttrLabel as String] = "Stark Local API"
        return SecItemAdd(value as CFDictionary, nil)
    }

    func replace(_ data: Data) -> OSStatus {
        SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
    }
}

final class APICredentialStore {
    static let installedService = "com.starkengine.menubar.local-api"
    static let developmentService = installedService + ".development"

    /// Developer ID builds retain their namespace across versions and install locations.
    /// Ad-hoc packages and command-line development harnesses cannot touch that credential.
    static var applicationService: String {
        let identifier = Bundle.main.bundleIdentifier
        guard identifier == "com.starkengine.menubar" else { return namespace(bundleIdentifier: identifier, signingTeam: nil) }
        var code: SecCode?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code else { return developmentService }
        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode else { return developmentService }
        var information: CFDictionary?
        guard SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &information) == errSecSuccess,
              let values = information as? [String: Any],
              let team = values[kSecCodeInfoTeamIdentifier as String] as? String, !team.isEmpty else {
            return developmentService
        }
        return namespace(bundleIdentifier: identifier, signingTeam: team)
    }

    static func namespace(bundleIdentifier: String?, signingTeam: String?) -> String {
        guard let bundleIdentifier else { return developmentService }
        if bundleIdentifier == "com.starkengine.menubar" {
            return signingTeam?.isEmpty == false ? installedService : developmentService
        }
        let component = bundleIdentifier.utf8.map { byte -> String in
            (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte) || [45, 46, 95].contains(byte)
                ? String(UnicodeScalar(byte)) : "_"
        }.joined()
        return developmentService + "." + component
    }

    let service: String
    private let storage: APICredentialStorage
    private let generate: () throws -> String

    init(service: String = APICredentialStore.applicationService,
         storage: APICredentialStorage? = nil,
         generate: @escaping () throws -> String = APICredentialStore.randomKey) {
        self.service = service
        self.storage = storage ?? KeychainAPICredentialStorage(service: service)
        self.generate = generate
    }

    func loadOrCreate() throws -> String {
        let (status, data) = storage.read()
        if status == errSecSuccess { return try decode(data) }
        guard status == errSecItemNotFound else { throw error(status) }
        let key = try generate()
        guard Self.valid(key) else { throw APICredentialError.random }
        let added = storage.add(Data(key.utf8))
        if added == errSecSuccess { return key }
        if added == errSecDuplicateItem {
            // Another app instance won creation. Use only the value that was persisted.
            let (winnerStatus, winner) = storage.read()
            guard winnerStatus == errSecSuccess else { throw error(winnerStatus) }
            return try decode(winner)
        }
        throw error(added)
    }

    func regenerate() throws -> String {
        let (status, _) = storage.read()
        // Explicit recovery can replace a corrupt entry, but must not bypass denied/locked storage.
        guard status == errSecSuccess || status == errSecItemNotFound || status == errSecDecode else { throw error(status) }
        let key = try generate()
        guard Self.valid(key) else { throw APICredentialError.random }
        let saved = status == errSecItemNotFound ? storage.add(Data(key.utf8)) : storage.replace(Data(key.utf8))
        guard saved == errSecSuccess else { throw error(saved) }
        return key
    }

    private func decode(_ data: Data?) throws -> String {
        guard let data, let key = String(data: data, encoding: .utf8), Self.valid(key) else {
            throw APICredentialError.corrupt
        }
        return key
    }

    private func error(_ status: OSStatus) -> APICredentialError {
        switch status {
        case errSecInteractionNotAllowed: return .locked
        case errSecAuthFailed, errSecUserCanceled, errSecMissingEntitlement: return .denied
        case errSecDecode: return .corrupt
        default: return .storage
        }
    }

    static func valid(_ key: String) -> Bool {
        key.utf8.count == 64 && key.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    static func randomKey() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw APICredentialError.random
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
}

struct APIConnectionPreferences {
    let defaults: UserDefaults
    let key: String

    init(service: String, defaults: UserDefaults = .standard) {
        self.defaults = defaults
        key = service + ".port"
    }

    var port: Int {
        // Keep a malformed saved value visible as a configuration error, rather than changing ports.
        defaults.object(forKey: key) == nil ? 8000 : defaults.integer(forKey: key)
    }

    func save(port: Int) throws {
        guard (1...65535).contains(port) else { throw APIConnectionPreferenceError.invalidPort }
        defaults.set(port, forKey: key)
    }
}

enum APIConnectionPreferenceError: Error { case invalidPort }
