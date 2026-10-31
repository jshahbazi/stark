import Foundation
import Security

private final class MemoryCredentialStorage: APICredentialStorage {
    var readStatus: OSStatus = errSecItemNotFound
    var addStatus: OSStatus = errSecSuccess
    var replaceStatus: OSStatus = errSecSuccess
    var data: Data?
    var adds = 0
    var replaces = 0
    var racingWinner: Data?

    func read() -> (OSStatus, Data?) { (readStatus, data) }
    func add(_ data: Data) -> OSStatus {
        adds += 1
        if let racingWinner { self.data = racingWinner; readStatus = errSecSuccess; return errSecDuplicateItem }
        if addStatus == errSecSuccess { self.data = data; readStatus = errSecSuccess }
        return addStatus
    }
    func replace(_ data: Data) -> OSStatus {
        replaces += 1
        if replaceStatus == errSecSuccess { self.data = data; readStatus = errSecSuccess }
        return replaceStatus
    }
}

@main
struct APICredentialStoreTests {
    static func main() throws {
        if CommandLine.arguments.count > 1 {
            if CommandLine.arguments[1] == "isolated-keychain" { try isolatedKeychain(); return }
            if CommandLine.arguments[1] == "child" {
                let expected = FileHandle.standardInput.readDataToEndOfFile()
                let key = try APICredentialStore(service: CommandLine.arguments[2]).loadOrCreate()
                precondition(Data(key.utf8) == expected)
                return
            }
        }
        let original = String(repeating: "a", count: 64)
        let replacement = String(repeating: "b", count: 64)
        let memory = MemoryCredentialStorage()
        var generated = 0
        let store = APICredentialStore(service: "isolated.unit", storage: memory) {
            generated += 1; return generated == 1 ? original : replacement
        }
        try require(try store.loadOrCreate() == original)
        try require(try store.loadOrCreate() == original)
        try require(try APICredentialStore(service: "isolated.unit", storage: memory).loadOrCreate() == original)
        precondition(generated == 1 && memory.adds == 1)

        memory.replaceStatus = errSecNotAvailable
        expect(.storage) { _ = try store.regenerate() }
        try require(try store.loadOrCreate() == original)
        memory.replaceStatus = errSecSuccess
        try require(try store.regenerate() == replacement)
        try require(try store.loadOrCreate() == replacement)

        for (status, error): (OSStatus, APICredentialError) in [
            (errSecInteractionNotAllowed, .locked), (errSecAuthFailed, .denied),
            (errSecUserCanceled, .denied), (errSecNotAvailable, .storage)
        ] {
            let failed = MemoryCredentialStorage()
            failed.readStatus = status
            var generatedOnFailure = false
            let inaccessible = APICredentialStore(service: "isolated.unit", storage: failed) {
                generatedOnFailure = true; return replacement
            }
            expect(error) { _ = try inaccessible.loadOrCreate() }
            expect(error) { _ = try inaccessible.regenerate() }
            precondition(!generatedOnFailure && failed.adds == 0 && failed.replaces == 0)
        }
        let undecodable = MemoryCredentialStorage()
        undecodable.readStatus = errSecDecode
        undecodable.data = Data(original.utf8)
        let recovery = APICredentialStore(service: "isolated.unit", storage: undecodable) { replacement }
        expect(.corrupt) { _ = try recovery.loadOrCreate() }
        undecodable.replaceStatus = errSecNotAvailable
        expect(.storage) { _ = try recovery.regenerate() }
        precondition(undecodable.data == Data(original.utf8) && undecodable.adds == 0)
        undecodable.replaceStatus = errSecSuccess
        try require(try recovery.regenerate() == replacement)
        try require(try recovery.loadOrCreate() == replacement)
        for invalid in [Data(), Data("short".utf8), Data(String(repeating: "A", count: 64).utf8), Data([255])] {
            let corrupt = MemoryCredentialStorage()
            corrupt.readStatus = errSecSuccess; corrupt.data = invalid
            let corruptStore = APICredentialStore(service: "isolated.unit", storage: corrupt) { replacement }
            expect(.corrupt) { _ = try corruptStore.loadOrCreate() }
            precondition(corrupt.adds == 0 && corrupt.replaces == 0)
            try require(try corruptStore.regenerate() == replacement)
        }
        let failedAdd = MemoryCredentialStorage()
        failedAdd.addStatus = errSecNotAvailable
        expect(.storage) { _ = try APICredentialStore(service: "isolated.unit", storage: failedAdd) { original }.loadOrCreate() }
        precondition(failedAdd.data == nil)
        let race = MemoryCredentialStorage()
        race.racingWinner = Data(replacement.utf8)
        try require(try APICredentialStore(service: "isolated.unit", storage: race) { original }.loadOrCreate() == replacement)
        let failedRandom = MemoryCredentialStorage()
        expect(.random) { _ = try APICredentialStore(service: "isolated.unit", storage: failedRandom) { throw APICredentialError.random }.loadOrCreate() }
        precondition(failedRandom.adds == 0)
        try require(APICredentialStore.valid(try APICredentialStore.randomKey()))

        let suite = "stark.api-preferences-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let preferences = APIConnectionPreferences(service: "isolated.test", defaults: defaults)
        precondition(preferences.port == 8000)
        try preferences.save(port: 12345)
        precondition(APIConnectionPreferences(service: "isolated.test", defaults: defaults).port == 12345)
        do { try preferences.save(port: 0); preconditionFailure("invalid_port_accepted") }
        catch APIConnectionPreferenceError.invalidPort {}
        precondition(preferences.port == 12345)
        defaults.set(-1, forKey: preferences.key)
        precondition(preferences.port == -1)
        precondition(APICredentialStore.installedService != APICredentialStore.developmentService)
        precondition(APICredentialStore.namespace(bundleIdentifier: nil, signingTeam: nil) == APICredentialStore.developmentService)
        precondition(APICredentialStore.namespace(bundleIdentifier: "com.starkengine.menubar", signingTeam: nil) == APICredentialStore.developmentService)
        precondition(APICredentialStore.namespace(bundleIdentifier: "com.starkengine.menubar", signingTeam: "test-team") == APICredentialStore.installedService)
        precondition(APICredentialStore.namespace(bundleIdentifier: "com.starkengine.validation.fixture", signingTeam: "test-team") == APICredentialStore.developmentService + ".com.starkengine.validation.fixture")
        print("credential unit and port persistence checks passed")
    }

    private static func require(_ condition: @autoclosure () throws -> Bool) throws {
        let passed = try condition()
        precondition(passed)
    }

    private static func expect(_ expected: APICredentialError, _ operation: () throws -> Void) {
        do { try operation(); preconditionFailure("expected_credential_failure") }
        catch let error as APICredentialError { precondition(String(describing: error) == String(describing: expected)) }
        catch { preconditionFailure("unexpected_error_category") }
    }

    private static func isolatedKeychain() throws {
        let service = "com.starkengine.tests.local-api." + UUID().uuidString
        defer {
            SecItemDelete([kSecClass as String: kSecClassGenericPassword,
                           kSecAttrService as String: service,
                           kSecAttrAccount as String: KeychainAPICredentialStorage.account] as CFDictionary)
        }
        let store = APICredentialStore(service: service)
        let original: String
        do { original = try store.loadOrCreate() }
        catch { print("isolated_keychain_unavailable"); exit(77) }
        try require(try APICredentialStore(service: service).loadOrCreate() == original)
        try child(service: service, expected: original)
        let replacement = try store.regenerate()
        precondition(replacement != original)
        try child(service: service, expected: replacement)
        print("isolated Keychain cross-process reuse and regeneration passed")
    }

    private static func child(service: String, expected: String) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        process.arguments = ["child", service]
        let pipe = Pipe()
        process.standardInput = pipe
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        try pipe.fileHandleForWriting.write(contentsOf: Data(expected.utf8))
        try pipe.fileHandleForWriting.close()
        process.waitUntilExit()
        precondition(process.terminationStatus == 0)
    }
}
