import Foundation
import LocalAuthentication
import Security

protocol DesktopConnectionApprovalStorage {
    func read() -> (OSStatus, Data?)
    func add(_ value: Data) -> OSStatus
    func replace(_ value: Data) -> OSStatus
}

struct KeychainDesktopConnectionApprovalStorage: DesktopConnectionApprovalStorage {
    let service: String
    var account = "openwork-approval-v1"
    var label = "Stark OpenWork Connection"
    private var query: [String: Any] {
        let context = LAContext()
        context.interactionNotAllowed = true
        return [kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service + ".desktop-connections",
                kSecAttrAccount as String: account,
                kSecAttrSynchronizable as String: false,
                kSecUseAuthenticationContext as String: context]
    }
    func read() -> (OSStatus, Data?) {
        var request = query
        request[kSecReturnData as String] = true
        request[kSecMatchLimit as String] = kSecMatchLimitOne
        var value: CFTypeRef?
        let result = SecItemCopyMatching(request as CFDictionary, &value)
        return (result, value as? Data)
    }
    func add(_ value: Data) -> OSStatus {
        var request = query
        request[kSecValueData as String] = value
        request[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        request[kSecAttrLabel as String] = label
        return SecItemAdd(request as CFDictionary, nil)
    }
    func replace(_ value: Data) -> OSStatus {
        SecItemUpdate(query as CFDictionary, [kSecValueData as String: value] as CFDictionary)
    }
}

/// Approval is separate from the persistent bearer. Failure never enables an
/// unapproved client and never creates, rotates or exports the API credential.
final class DesktopConnectionApprovalStore {
    private let storage: DesktopConnectionApprovalStorage
    private static let approved = Data("openwork:F5DJWB4CCV:com.differentai.openwork:v1:approved".utf8)
    private static let revoked = Data("openwork:F5DJWB4CCV:com.differentai.openwork:v1:revoked".utf8)
    init(service: String, storage: DesktopConnectionApprovalStorage? = nil) {
        self.storage = storage ?? KeychainDesktopConnectionApprovalStorage(service: service)
    }
    func load() throws -> Bool {
        let (status, value) = storage.read()
        if status == errSecItemNotFound { return false }
        guard status == errSecSuccess else { throw failure(status) }
        if value == Self.approved { return true }
        if value == Self.revoked { return false }
        throw APICredentialError.corrupt
    }
    func save(approved: Bool) throws {
        let (status, _) = storage.read()
        // Connect/Disconnect are explicit approval recovery actions. A corrupt
        // item may be replaced; locked or denied storage is never bypassed.
        guard status == errSecSuccess || status == errSecItemNotFound || status == errSecDecode else { throw failure(status) }
        let value = approved ? Self.approved : Self.revoked
        let result = status == errSecItemNotFound ? storage.add(value) : storage.replace(value)
        guard result == errSecSuccess else { throw failure(result) }
    }
    private func failure(_ status: OSStatus) -> APICredentialError { Self.failure(status) }
    static func failure(_ status: OSStatus) -> APICredentialError {
        switch status {
        case errSecInteractionNotAllowed: return .locked
        case errSecAuthFailed, errSecUserCanceled, errSecMissingEntitlement: return .denied
        case errSecDecode: return .corrupt
        default: return .storage
        }
    }
}

/// OpenCode approval: whether the user connected OpenCode, and which engine
/// instances (process ID plus start time) they approved, oldest first.
struct OpenCodeEngineApprovals: Equatable {
    var enabled = false
    var engines: [String] = []
}

/// Stored in its own item. It holds public process identities only, never a
/// credential, and is separate from the OpenWork approval.
final class OpenCodeEngineApprovalStore {
    static let limit = 8
    private static let prefix = "opencode:5NZ4Q7NXJ4:ai.opencode.desktop:v1:"
    private let storage: DesktopConnectionApprovalStorage
    init(service: String, storage: DesktopConnectionApprovalStorage? = nil) {
        self.storage = storage ?? KeychainDesktopConnectionApprovalStorage(
            service: service, account: "opencode-approval-v1", label: "Stark OpenCode Connection")
    }

    static func validEngine(_ value: Substring) -> Bool {
        let parts = value.split(separator: ".", omittingEmptySubsequences: false)
        return parts.count == 3 && zip(parts, [10, 20, 6]).allSatisfy { part, width in
            (1...width).contains(part.utf8.count) && part.utf8.allSatisfy { (48...57).contains($0) }
        }
    }

    static func encode(_ value: OpenCodeEngineApprovals) -> Data {
        Data((prefix + (value.enabled ? "enabled:" + value.engines.joined(separator: ",") : "revoked")).utf8)
    }

    func load() throws -> OpenCodeEngineApprovals {
        let (status, value) = storage.read()
        if status == errSecItemNotFound { return OpenCodeEngineApprovals() }
        guard status == errSecSuccess else { throw DesktopConnectionApprovalStore.failure(status) }
        guard let value, value.count <= 1024, let text = String(data: value, encoding: .utf8),
              text.hasPrefix(Self.prefix) else { throw APICredentialError.corrupt }
        let body = text.dropFirst(Self.prefix.count)
        if body == "revoked" { return OpenCodeEngineApprovals() }
        guard body.hasPrefix("enabled:") else { throw APICredentialError.corrupt }
        let list = body.dropFirst("enabled:".count)
        let engines = list.isEmpty ? [] : list.split(separator: ",", omittingEmptySubsequences: false)
        guard engines.count <= Self.limit, engines.allSatisfy(Self.validEngine),
              Set(engines).count == engines.count else { throw APICredentialError.corrupt }
        return OpenCodeEngineApprovals(enabled: true, engines: engines.map(String.init))
    }

    func save(_ value: OpenCodeEngineApprovals) throws {
        guard value.engines.count <= Self.limit, value.engines.allSatisfy({ Self.validEngine($0[...]) }),
              value.enabled || value.engines.isEmpty else { throw APICredentialError.storage }
        let (status, _) = storage.read()
        // Connect, Allow and Disconnect are explicit recovery actions. A corrupt
        // item may be replaced; locked or denied storage is never bypassed.
        guard status == errSecSuccess || status == errSecItemNotFound || status == errSecDecode else {
            throw DesktopConnectionApprovalStore.failure(status)
        }
        let data = Self.encode(value)
        let result = status == errSecItemNotFound ? storage.add(data) : storage.replace(data)
        guard result == errSecSuccess else { throw DesktopConnectionApprovalStore.failure(result) }
    }

    /// Keeps engines that are still running, drops the oldest beyond the limit.
    static func approving(_ engine: String, after engines: [String]) -> [String] {
        let live = engines.filter { $0 != engine && DesktopProcessIdentity.isRunning(instanceID: $0) }
        return Array((live + [engine]).suffix(limit))
    }
}
