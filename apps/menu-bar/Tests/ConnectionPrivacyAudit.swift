#if STARK_CONNECTION_VALIDATION
import Darwin
import Foundation

/// Read-only packaged-test instrumentation. No credential is exported. This is
/// compiled only by the explicit validation build, never by the product build.
enum ConnectionPrivacyAudit {
    private static var secrets: [Data] = []
    private static var secretLimitExceeded = false

    static func remember(_ value: String) {
        guard Bundle.main.bundleIdentifier?.hasPrefix("com.starkengine.menubar.validation.") == true,
              APICredentialStore.valid(value) else { return }
        let bytes = Data(value.utf8)
        guard !secrets.contains(bytes) else { return }
        guard secrets.count < 64 else { secretLimitExceeded = true; return }
        secrets.append(bytes)
    }

    static func finish() {
        guard Bundle.main.bundleIdentifier?.hasPrefix("com.starkengine.menubar.validation.") == true,
              let profile = Bundle.main.object(forInfoDictionaryKey: "StarkOpenWorkTestProfile") as? String,
              let target = Bundle.main.object(forInfoDictionaryKey: "StarkConnectionAuditReport") as? String else { return }
        defer { secrets.removeAll(); secretLimitExceeded = false }
        var files = 0, bytes = 0, matches = 0, skipped = 0
        var category = "complete"
        let deadline = ProcessInfo.processInfo.systemUptime + 30
        var rootInfo = stat()
        guard lstat(profile, &rootInfo) == 0, rootInfo.st_mode & S_IFMT == S_IFDIR,
              rootInfo.st_uid == getuid(), rootInfo.st_mode & 0o077 == 0,
              !secrets.isEmpty else { return }
        let root = URL(fileURLWithPath: profile, isDirectory: true)
        var failed = false
        let walk = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil, options: [], errorHandler: { _, _ in
            failed = true; return false
        })
        if walk == nil { failed = true }
        while let url = walk?.nextObject() as? URL {
            if ProcessInfo.processInfo.systemUptime > deadline || files >= 20_000 || bytes >= 512 * 1024 * 1024 {
                category = "scan_limit"; break
            }
            var info = stat()
            guard lstat(url.path, &info) == 0, info.st_uid == getuid() else { failed = true; break }
            if info.st_mode & S_IFMT == S_IFDIR { continue }
            guard info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1 else {
                skipped += 1; walk?.skipDescendants(); continue
            }
            guard info.st_size <= 64 * 1024 * 1024, bytes + Int(info.st_size) <= 512 * 1024 * 1024 else {
                category = "scan_limit"; break
            }
            let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { failed = true; break }
            var opened = stat()
            guard fstat(descriptor, &opened) == 0, opened.st_ino == info.st_ino, opened.st_dev == info.st_dev,
                  opened.st_size == info.st_size else { close(descriptor); failed = true; break }
            let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            var data = Data()
            do {
                while data.count < Int(opened.st_size) {
                    guard ProcessInfo.processInfo.systemUptime <= deadline else { break }
                    let chunk = try handle.read(upToCount: min(64 * 1024, Int(opened.st_size) - data.count)) ?? Data()
                    if chunk.isEmpty { break }
                    data.append(chunk)
                }
            } catch { try? handle.close(); failed = true; break }
            var after = stat()
            let stable = fstat(descriptor, &after) == 0 && after.st_size == opened.st_size
                && after.st_mtimespec.tv_sec == opened.st_mtimespec.tv_sec
                && after.st_mtimespec.tv_nsec == opened.st_mtimespec.tv_nsec
            try? handle.close()
            guard stable, data.count == info.st_size else { failed = true; break }
            files += 1; bytes += data.count
            if secrets.contains(where: { data.range(of: $0) != nil }) { matches += 1 }
        }
        if failed { category = "filesystem_changed_or_unavailable" }
        if secretLimitExceeded { category = "secret_limit" }
        let report: [String: Any] = ["complete": category == "complete", "category": category,
                                   "files": files, "bytes": bytes, "matching_files": matches,
                                   "skipped_nonregular": skipped, "secret_count": secrets.count]
        guard let encoded = try? JSONSerialization.data(withJSONObject: report, options: .sortedKeys) else { return }
        try? encoded.write(to: URL(fileURLWithPath: target), options: .atomic)
    }
}
#endif
