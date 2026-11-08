import Foundation

enum BeginnerContextPolicy {
    static func requiresRestart(expected: Int, local: Int, api: Int) -> Bool {
        expected <= 0 || local != expected || api != expected
    }
}

/// A recommendation is a short-lived backend binding, separate from the catalog.
struct BeginnerModelReport {
    struct Item: Identifiable, Equatable {
        let id: String
        let name: String
        let source: String
        let revision: String
        let quantization: String
        let downloadBytes: Int64
        let installedBytes: Int64
        let setupBytes: Int64
        let runtimeBytes: Int64
        let gpuRuntimeBytes: Int64
        let contextTokens: Int
        let installedIDs: [String]
        let license: String
        let qualification: String
        let caveats: [String]
        let recommendationReason: String
        let benchmarkDetails: [String]

        init?(_ raw: [String: Any]) {
            guard let id = BeginnerMetadata.string(raw["id"], limit: 128),
                  let name = BeginnerMetadata.string(raw["name"], limit: 200),
                  let source = BeginnerMetadata.string(raw["repo"], limit: 256),
                  let revision = BeginnerMetadata.string(raw["revision"], limit: 40),
                  revision.count == 40, revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
                  raw["artifact_mode"] as? String == "text",
                  let quantization = BeginnerMetadata.string(raw["quantization"], limit: 80),
                  let download = BeginnerMetadata.integer(raw["download_bytes"]), download > 0,
                  let installed = BeginnerMetadata.integer(raw["installed_bytes_estimate"]), installed > 0,
                  let setup = BeginnerMetadata.integer(raw["setup_disk_bytes_estimate"]), setup >= installed,
                  let runtime = BeginnerMetadata.integer(raw["runtime_memory_bytes_estimate"]), runtime > 0,
                  let context = BeginnerMetadata.integer(raw["context_tokens"]), (512...1_048_576).contains(context),
                  let ids = BeginnerMetadata.strings(raw["installed_ids"], count: 64, length: 512),
                  let license = BeginnerMetadata.string(raw["license"], limit: 160),
                  let qualification = BeginnerMetadata.string(raw["qualification"], limit: 2_000),
                  let caveats = BeginnerMetadata.strings(raw["caveats"], count: 16, length: 2_000) else { return nil }
            self.id = id; self.name = name; self.source = source; self.revision = revision
            self.quantization = quantization; downloadBytes = download; installedBytes = installed
            if let value = raw["setup_disk_bytes_remaining_estimate"] {
                guard let remaining = BeginnerMetadata.integer(value), remaining <= setup else { return nil }
                setupBytes = remaining
            } else { setupBytes = setup }
            runtimeBytes = runtime; contextTokens = Int(context)
            if let value = raw["gpu_memory_bytes_estimate"] {
                guard let gpu = BeginnerMetadata.integer(value), gpu > 0, gpu <= runtime else { return nil }
                gpuRuntimeBytes = gpu
            } else { gpuRuntimeBytes = runtime }
            installedIDs = ids; self.license = license; self.qualification = qualification; self.caveats = caveats
            if let value = raw["benchmark_details"] {
                guard let details = BeginnerMetadata.strings(value, count: 16, length: 2_000) else { return nil }
                benchmarkDetails = details
            } else { benchmarkDetails = [] }
            if raw["recommendation_reason"] != nil {
                guard let reason = BeginnerMetadata.string(raw["recommendation_reason"], limit: 2_000) else { return nil }
                recommendationReason = reason
            } else {
                recommendationReason = "A starting model checked for this setup. Review its answers and any actions in OpenWork."
            }
        }

        var memoryDetails: [String] {
            if gpuRuntimeBytes == runtimeBytes {
                return ["Estimated runtime memory: " + ByteCountFormatter.string(fromByteCount: runtimeBytes, countStyle: .memory)]
            }
            return [
                "Estimated system memory: " + ByteCountFormatter.string(fromByteCount: runtimeBytes, countStyle: .memory),
                "Estimated GPU memory: " + ByteCountFormatter.string(fromByteCount: gpuRuntimeBytes, countStyle: .memory),
                "The system estimate includes the model's disk-backed tables."
            ]
        }
    }

    let message: String
    let physicalMemoryBytes: Int64?
    let availableMemoryBytes: Int64?
    let freeDiskBytes: Int64?
    let gpuName: String?
    let gpuMemoryBudgetBytes: Int64?
    let items: [Item]

    init?(_ raw: [String: Any]) {
        guard BeginnerMetadata.integer(raw["schema_version"]) == 1,
              let status = raw["status"] as? String, ["ready", "unavailable"].contains(status),
              let message = BeginnerMetadata.string(raw["message"], limit: 2_000),
              let host = raw["host"] as? [String: Any],
              let values = raw["items"] as? [[String: Any]], values.count <= 3 else { return nil }
        let items = values.compactMap(Item.init)
        guard items.count == values.count, Set(items.map(\.id)).count == items.count,
              (status == "ready") == !items.isEmpty else { return nil }
        self.message = message; self.items = items
        physicalMemoryBytes = BeginnerMetadata.integer(host["physical_memory_bytes"])
        availableMemoryBytes = BeginnerMetadata.integer(host["available_memory_bytes"])
        freeDiskBytes = BeginnerMetadata.integer(host["free_disk_bytes"])
        gpuName = BeginnerMetadata.string(host["gpu_name"], limit: 200)
        gpuMemoryBudgetBytes = BeginnerMetadata.integer(host["gpu_memory_budget_bytes"])
    }

    var deviceSummary: String {
        var details: [String] = []
        if let gpuName { details.append(gpuName) }
        if let bytes = physicalMemoryBytes { details.append(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory) + " memory") }
        if let bytes = gpuMemoryBudgetBytes, bytes > 0 { details.append(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory) + " GPU budget") }
        if let bytes = freeDiskBytes { details.append(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) + " free for setup") }
        return details.joined(separator: " · ")
    }
}

private enum BeginnerMetadata {
    static func string(_ raw: Any?, limit: Int) -> String? {
        guard let value = raw as? String, !value.isEmpty, value.utf8.count <= limit,
              !value.unicodeScalars.contains(where: { $0.value < 32 && $0 != "\n" && $0 != "\t" }) else { return nil }
        return value
    }

    static func integer(_ raw: Any?) -> Int64? {
        guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.int64Value >= 0, number.compare(NSNumber(value: number.int64Value)) == .orderedSame else { return nil }
        return number.int64Value
    }

    static func strings(_ raw: Any?, count: Int, length: Int) -> [String]? {
        guard let values = raw as? [String], values.count <= count,
              values.allSatisfy({ string($0, limit: length) != nil }) else { return nil }
        return values
    }
}
