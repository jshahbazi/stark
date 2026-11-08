import Foundation

enum BeginnerContextPolicy {
    static func requiresRestart(expected: Int, local: Int, api: Int) -> Bool {
        expected <= 0 || local != expected || api != expected
    }
}

/// A recommendation is a short-lived backend binding, separate from the catalog.
struct BeginnerModelReport {
    struct BenchmarkReference: Equatable {
        let name: String
        let score: String
        let usedForRanking: Bool
        let sourceURL: URL?

        init?(_ raw: [String: Any]) {
            let names = ["livecodebench": "LiveCodeBench", "bfcl": "BFCL", "ifeval": "IFEval"]
            guard let key = raw["benchmark"] as? String, let name = names[key],
                  raw["evidence_kind"] as? String == "upstream_reference",
                  let text = BeginnerMetadata.string(raw["score"], limit: 32),
                  let value = Double(text), value.isFinite, (0...100).contains(value),
                  let used = raw["used_for_ranking"] as? NSNumber,
                  CFGetTypeID(used) == CFBooleanGetTypeID() else { return nil }
            self.name = name
            let formatter = NumberFormatter()
            formatter.minimumFractionDigits = 0; formatter.maximumFractionDigits = 1
            formatter.usesGroupingSeparator = false
            score = formatter.string(from: NSNumber(value: value)) ?? text
            usedForRanking = used.boolValue
            let source = raw["source"] as? [String: Any]
            if let text = BeginnerMetadata.string(source?["url"], limit: 1_000),
               let url = URL(string: text), url.scheme == "https", url.host != nil {
                sourceURL = url
            } else { sourceURL = nil }
        }

        var recommendation: String {
            switch name {
            case "LiveCodeBench": return "Recommended for coding using published benchmark results."
            case "BFCL": return "Recommended using published tool-use results."
            default: return "Recommended using published instruction-following results."
            }
        }
    }

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
        let benchmarkReference: BenchmarkReference?

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
            let references = (raw["benchmarks"] as? [[String: Any]]).flatMap { values in
                values.count <= 32 ? values.compactMap(BenchmarkReference.init) : nil
            } ?? []
            benchmarkReference = references.first(where: \.usedForRanking) ?? references.first
            if raw["recommendation_reason"] != nil {
                guard let reason = BeginnerMetadata.string(raw["recommendation_reason"], limit: 2_000) else { return nil }
                recommendationReason = reason
            } else {
                recommendationReason = "A starting model checked for this setup. Review its answers and any actions in OpenWork."
            }
        }

        func beginnerSummary(availableMemoryBytes: Int64?) -> String {
            var summary = "Selected to fit your Mac."
            if let reference = benchmarkReference, reference.usedForRanking { summary = reference.recommendation }
            if benchmarkReference == nil { summary += " No benchmark result is recorded." }
            if let availableMemoryBytes, availableMemoryBytes < runtimeBytes {
                summary += " Free up memory before starting."
            }
            return summary
        }

        var beginnerDetails: String {
            var lines = ["Estimated memory: " + ByteCountFormatter.string(fromByteCount: runtimeBytes, countStyle: .memory)]
            if let benchmarkReference {
                lines.append(benchmarkReference.name + ": " + benchmarkReference.score + "/100")
                lines.append(benchmarkReference.usedForRanking
                    ? "Published model score; local results may vary."
                    : "Reference only; it did not determine this choice.")
            }
            return lines.joined(separator: "\n")
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
