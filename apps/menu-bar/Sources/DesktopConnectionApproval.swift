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
    private var query: [String: Any] {
        let context = LAContext()
        context.interactionNotAllowed = true
        return [kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service + ".desktop-connections",
                kSecAttrAccount as String: "openwork-approval-v1",
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
        request[kSecAttrLabel as String] = "Stark OpenWork Connection"
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
    private func failure(_ status: OSStatus) -> APICredentialError {
        switch status {
        case errSecInteractionNotAllowed: return .locked
        case errSecAuthFailed, errSecUserCanceled, errSecMissingEntitlement: return .denied
        case errSecDecode: return .corrupt
        default: return .storage
        }
    }
}
