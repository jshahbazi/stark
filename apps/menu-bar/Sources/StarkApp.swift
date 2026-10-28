import AppKit
import Darwin
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

enum ImageAttachmentError: Error {
    case invalid, tooLarge, unavailable

    var message: String {
        switch self {
        case .invalid: return "Choose one PNG or JPEG image."
        case .tooLarge: return "Images must be 4 MiB or smaller."
        case .unavailable: return "The image could not be read. Choose it again."
        }
    }
}

struct ImageAttachment {
    static let maximumBytes = 4 * 1024 * 1024
    let bytes: Data
    let mediaType: String

    init(bytes: Data) throws {
        guard !bytes.isEmpty else { throw ImageAttachmentError.invalid }
        guard bytes.count <= Self.maximumBytes else { throw ImageAttachmentError.tooLarge }
        if bytes.starts(with: [137, 80, 78, 71, 13, 10, 26, 10]) { mediaType = "image/png" }
        else if bytes.starts(with: [255, 216, 255]), bytes.suffix(2) == Data([255, 217]) { mediaType = "image/jpeg" }
        else { throw ImageAttachmentError.invalid }
        self.bytes = bytes
    }

    static func read(_ url: URL) throws -> ImageAttachment {
        guard url.isFileURL else { throw ImageAttachmentError.invalid }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw ImageAttachmentError.unavailable }
        defer { Darwin.close(descriptor) }
        var before = stat()
        guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
              before.st_size > 0 else { throw ImageAttachmentError.invalid }
        guard before.st_size <= maximumBytes else { throw ImageAttachmentError.tooLarge }
        var data = Data(count: Int(before.st_size))
        let expected = data.count
        let readCount = data.withUnsafeMutableBytes { buffer -> Int in
            var offset = 0
            while offset < expected {
                let amount = Darwin.read(descriptor, buffer.baseAddress!.advanced(by: offset), expected - offset)
                if amount < 0 && errno == EINTR { continue }
                guard amount > 0 else { return -1 }
                offset += amount
            }
            return offset
        }
        var after = stat()
        guard readCount == expected, fstat(descriptor, &after) == 0,
              before.st_dev == after.st_dev, before.st_ino == after.st_ino,
              before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
              before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
            throw ImageAttachmentError.unavailable
        }
        return try ImageAttachment(bytes: data)
    }

    var thumbnail: NSImage? {
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(bytes as CFData, sourceOptions),
              CGImageSourceGetCount(source) == 1,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, width <= 16_384, height <= 16_384,
              Int64(width) * Int64(height) <= 32_000_000 else { return nil }
        let options = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                       kCGImageSourceCreateThumbnailWithTransform: true,
                       kCGImageSourceThumbnailMaxPixelSize: 256,
                       kCGImageSourceShouldCacheImmediately: false] as CFDictionary
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options) else { return nil }
        return NSImage(cgImage: image, size: .zero)
    }
}

final class AttachmentStaging {
    let root: URL
    private let descriptor: Int32
    private var entries = Set<String>()

    init() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("stark-attachments-" + UUID().uuidString.lowercased(), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        guard let path = Darwin.realpath(directory.path, nil) else {
            try? FileManager.default.removeItem(at: directory)
            throw ImageAttachmentError.unavailable
        }
        defer { free(path) }
        root = URL(fileURLWithPath: String(cString: path), isDirectory: true)
        descriptor = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0, fchmod(descriptor, 0o700) == 0 else {
            if descriptor >= 0 { Darwin.close(descriptor) }
            try? FileManager.default.removeItem(at: root)
            throw ImageAttachmentError.unavailable
        }
    }

    func stage(_ image: ImageAttachment) throws -> [String: Any] {
        let identifier = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let file = Darwin.openat(descriptor, identifier, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard file >= 0 else { throw ImageAttachmentError.unavailable }
        var complete = false
        defer { Darwin.close(file); if !complete { Darwin.unlinkat(descriptor, identifier, 0) } }
        guard fchmod(file, 0o600) == 0 else { throw ImageAttachmentError.unavailable }
        let written = image.bytes.withUnsafeBytes { buffer -> Bool in
            var offset = 0
            while offset < buffer.count {
                let amount = Darwin.write(file, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
                if amount < 0 && errno == EINTR { continue }
                guard amount > 0 else { return false }
                offset += amount
            }
            return true
        }
        guard written else { throw ImageAttachmentError.unavailable }
        complete = true
        entries.insert(identifier)
        return ["id": identifier, "media_type": image.mediaType, "byte_length": image.bytes.count]
    }

    func clear() {
        for entry in entries { Darwin.unlinkat(descriptor, entry, 0) }
        entries.removeAll()
    }

    deinit {
        clear()
        var pinned = stat(), current = stat()
        if fstat(descriptor, &pinned) == 0, lstat(root.path, &current) == 0,
           pinned.st_dev == current.st_dev, pinned.st_ino == current.st_ino {
            Darwin.rmdir(root.path)
        }
        Darwin.close(descriptor)
    }
}

struct InstalledModelDisplay: Equatable {
    let groupID: String
    let groupName: String
    let familyID: String?
    let quantization: String?
    let weightBytes: Int64?
    let sourceRevision: String?

    var repositoryURL: URL? {
        guard groupID.hasPrefix("repo:") else { return nil }
        let repository = String(groupID.dropFirst(5))
        guard repository.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$",
                               options: .regularExpression) != nil else { return nil }
        return URL(string: "https://huggingface.co/" + repository)
    }

    init?(_ value: Any?) {
        guard let value = value as? [String: Any], SupportMetadata.schemaOne(value),
              let groupID = Self.text(value["group_id"], maximum: 512),
              let groupName = Self.text(value["group_name"], maximum: 256) else { return nil }
        for key in ["family_id", "quantization", "source_revision"] {
            guard value[key] == nil || value[key] is NSNull || value[key] is String else { return nil }
        }
        let familyID = value["family_id"] as? String
        if let familyID = familyID, ModelFamily.displayName(for: familyID) == nil { return nil }
        let quantization = value["quantization"] as? String
        if let quantization = quantization, Self.text(quantization, maximum: 64) == nil { return nil }
        let revision = value["source_revision"] as? String
        if let revision = revision, revision.utf8.count != 40 || !revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) { return nil }
        var bytes: Int64?
        if let raw = value["weight_bytes"], !(raw is NSNull) {
            guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.int64Value >= 0, number.compare(NSNumber(value: number.int64Value)) == .orderedSame else { return nil }
            bytes = number.int64Value
        }
        self.groupID = groupID; self.groupName = groupName; self.familyID = familyID
        self.quantization = quantization; weightBytes = bytes; sourceRevision = revision
    }

    private static func text(_ value: Any?, maximum: Int) -> String? {
        guard let value = SupportMetadata.string(value, maximum: maximum),
              value.unicodeScalars.allSatisfy({ $0.value >= 32 && $0.value != 127 }) else { return nil }
        return value
    }
}

enum ModelArtifactMode: String, CaseIterable {
    case text
    case visionLanguage = "vision-language"
    var title: String { self == .text ? "Text" : "Text and images" }
}

struct InstalledModelRemoval: Equatable {
    enum Action: String {
        case trash, unregister, unavailable
        var title: String {
            switch self {
            case .trash: return "Move to Trash"
            case .unregister: return "Remove from Stark"
            case .unavailable: return "Removal unavailable"
            }
        }
    }
    let action: Action
    let label: String
    let message: String

    init?(_ value: Any?) {
        guard let raw = value as? [String: Any], SupportMetadata.schemaOne(raw),
              let action = (raw["action"] as? String).flatMap(Action.init(rawValue:)),
              let label = SupportMetadata.string(raw["label"], maximum: 64),
              let message = SupportMetadata.string(raw["message"]) else { return nil }
        self.action = action; self.label = label; self.message = message
    }
}

struct InstalledModel: Identifiable, Equatable {
    let id: String
    let name: String
    let path: String
    let apiID: String
    let display: InstalledModelDisplay?
    let artifactMode: ModelArtifactMode?
    let removal: InstalledModelRemoval?

    init(id: String, name: String, path: String, apiID: String, display: InstalledModelDisplay? = nil,
         artifactMode: ModelArtifactMode? = .text, removal: InstalledModelRemoval? = nil) {
        self.id = id; self.name = name; self.path = path; self.apiID = apiID; self.display = display
        self.artifactMode = artifactMode; self.removal = removal
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    var image: ImageAttachment? = nil
}

enum PlaygroundMode: String, CaseIterable, Identifiable {
    case chat, decision
    var id: String { rawValue }
    var title: String { self == .chat ? "Chat" : "Decision" }
}

struct DecisionResult {
    let answers: [String]
    let probabilities: [Double]
    let choiceIndex: Int
    var choice: String { answers[choiceIndex] }
}

struct ChatOptions {
    var useModelSystemPrompt = true
    var systemPrompt = ""
    var contextTokens = 8192
    var maxNewTokens = 512
    var thinking = false
}

struct InstallProgress {
    var state = "waiting"
    var completedBytes: Int64 = 0
    var totalBytes: Int64?

    var fraction: Double {
        if state == "complete" || state == "skipped" { return 1 }
        guard let total = totalBytes, total > 0 else { return 0 }
        return min(1, max(0, Double(completedBytes) / Double(total)))
    }

    func detail(for phase: String) -> String {
        switch state {
        case "complete": return "Complete"
        case "skipped": return "Not needed"
        case "cancelled": return "Cancelled"
        case "failed": return "Failed"
        case "waiting": return "Waiting"
        default:
            guard let total = totalBytes, total > 0 else { return "Preparing…" }
            if completedBytes >= total { return phase == "download" ? "Verifying…" : "Finalizing…" }
            return String(format: "%.0f%%", fraction * 100)
        }
    }
}

enum ModelSupportMode: String, CaseIterable {
    case text, decision
    case visionLanguage = "vision-language"

    var title: String {
        switch self {
        case .text: return "Text"
        case .decision: return "Decision"
        case .visionLanguage: return "Image"
        }
    }

}

enum ModelModeSupport: String {
    case conditional, unsupported
}

private enum SupportMetadata {
    static func schemaOne(_ value: [String: Any]) -> Bool {
        guard let number = value["schema_version"] as? NSNumber else { return false }
        return CFGetTypeID(number) != CFBooleanGetTypeID() && number.doubleValue == 1
    }

    static func string(_ value: Any?, maximum: Int = 2_000) -> String? {
        guard let value = value as? String, !value.isEmpty, value.utf8.count <= maximum,
              !value.unicodeScalars.contains(where: { $0.value < 32 && $0 != "\n" && $0 != "\t" }) else { return nil }
        return value
    }

    static func strings(_ value: Any?, maximumCount: Int = 32) -> [String]? {
        guard let values = value as? [String], values.count <= maximumCount,
              values.allSatisfy({ string($0) != nil }) else { return nil }
        return values
    }

    static func displayCheck(_ value: String) -> String {
        switch value {
        case "tensor_and_operation_validation", "tensor_headers", "tensor_operations":
            return "Tensor layout and operations must pass validation."
        case "checkpoint_specific_execution", "native_loading", "local_execution":
            return "The checkpoint must run successfully in the native engine."
        case "direct_answer_template", "decision_template":
            return "Decision requires a direct-answer chat template."
        case "unique_single_token_labels", "single_token_labels":
            return "Decision answer labels must each map to a unique token."
        case "calibration_not_inherited", "checkpoint_calibration":
            return "Decision scores need calibration for this checkpoint and workload."
        case "qwen35_dense_bf16_geometry":
            return "Images require the supported dense BF16 geometry."
        case "reviewed_processing_programs":
            return "Images require a reviewed tokenizer, processor and chat template."
        case "local_image_checks":
            return "Images must pass checks on this Mac."
        case "tokenizer_and_template":
            return "The tokenizer and chat template must pass validation."
        default:
            if value.contains("_") && value.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.contains($0) || $0 == "_" }) {
                return value.replacingOccurrences(of: "_", with: " ").capitalized + "."
            }
            return value
        }
    }

    static func modes(_ value: Any?) -> [ModelSupportMode: ModelSupportReport.Status]? {
        guard let values = value as? [String: String], Set(values.keys) == Set(ModelSupportMode.allCases.map(\.rawValue)) else { return nil }
        var modes: [ModelSupportMode: ModelSupportReport.Status] = [:]
        for mode in ModelSupportMode.allCases {
            guard let status = values[mode.rawValue].flatMap(ModelSupportReport.Status.init(rawValue:)) else { return nil }
            modes[mode] = status
        }
        return modes
    }

    static func integer(_ value: Any?) -> Int64? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.int64Value >= 0, number.compare(NSNumber(value: number.int64Value)) == .orderedSame else { return nil }
        return number.int64Value
    }

    static func boolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }

    static func revision(_ value: Any?) -> String? {
        guard let value = value as? String, value.utf8.count == 40,
              value.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { return nil }
        return value
    }
}

struct ModelFamilyMode {
    let status: ModelModeSupport
    let message: String
    let caveats: [String]
}

struct ModelFamily: Identifiable {
    let id: String
    let name: String
    let architectureAliases: [String]
    let configClasses: [String]
    let modelKind: String
    let containers: [String]
    let sourceFormats: [String]
    let outputDtypes: [String]
    let modes: [ModelSupportMode: ModelFamilyMode]

    var displayName: String { Self.displayName(for: id) ?? name }

    static func displayName(for id: String) -> String? {
        switch id {
        case "qwen3": return "Qwen3"
        case "qwen2": return "Qwen2"
        case "qwen3_5": return "Qwen3.5"
        case "llama": return "Llama"
        case "mistral": return "Mistral"
        case "gemma2": return "Gemma 2"
        case "gemma3_text": return "Gemma 3"
        case "phi3": return "Phi-3"
        case "granite": return "Granite"
        case "smollm3": return "SmolLM3"
        case "lfm2": return "LFM2"
        case "llamba": return "LLaMBA"
        case "modernbert": return "ModernBERT"
        default: return nil
        }
    }

    func hasAppMode(_ mode: ModelSupportMode) -> Bool { modes[mode]?.status == .conditional }

    init?(_ value: [String: Any]) {
        guard let id = SupportMetadata.string(value["family_id"], maximum: 128),
              let name = SupportMetadata.string(value["name"], maximum: 256),
              let aliases = SupportMetadata.strings(value["architecture_aliases"]),
              let classes = SupportMetadata.strings(value["config_classes"]),
              let kind = SupportMetadata.string(value["model_kind"], maximum: 128),
              let conversion = value["conversion"] as? [String: Any], conversion["status"] as? String == "implemented",
              let containers = SupportMetadata.strings(conversion["containers"]),
              let formats = SupportMetadata.strings(conversion["source_formats"]),
              let dtypes = SupportMetadata.strings(conversion["output_dtypes"]),
              let rawModes = value["modes"] as? [String: [String: Any]],
              Set(rawModes.keys) == Set(ModelSupportMode.allCases.map(\.rawValue)) else { return nil }
        var modes: [ModelSupportMode: ModelFamilyMode] = [:]
        for mode in ModelSupportMode.allCases {
            guard let raw = rawModes[mode.rawValue],
                  let status = (raw["status"] as? String).flatMap(ModelModeSupport.init(rawValue:)),
                  let message = SupportMetadata.string(raw["message"]),
                  let caveats = SupportMetadata.strings(raw["caveats"]) else { return nil }
            modes[mode] = ModelFamilyMode(status: status, message: message, caveats: caveats)
        }
        self.id = id; self.name = name; architectureAliases = aliases; configClasses = classes; modelKind = kind
        self.containers = containers; sourceFormats = formats; outputDtypes = dtypes; self.modes = modes
    }
}

struct ModelFamilyRegistry {
    let families: [ModelFamily]

    init?(_ value: [String: Any]) {
        guard SupportMetadata.schemaOne(value),
              let modes = value["modes"] as? [String], Set(modes) == Set(ModelSupportMode.allCases.map(\.rawValue)),
              modes.count == ModelSupportMode.allCases.count,
              let raw = value["families"] as? [[String: Any]], raw.count <= 128 else { return nil }
        let families = raw.compactMap(ModelFamily.init)
        guard families.count == raw.count, Set(families.map(\.id)).count == families.count else { return nil }
        self.families = families
    }
}

struct CatalogModelGroup: Equatable {
    let id: String
    let name: String
    let publisher: String

    init?(_ value: Any?) {
        guard let raw = value as? [String: Any],
              let id = SupportMetadata.string(raw["id"], maximum: 32), id.utf8.count == 32,
              id.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              let name = SupportMetadata.string(raw["name"], maximum: 256),
              let publisher = SupportMetadata.string(raw["publisher"], maximum: 96),
              !publisher.contains("/"),
              [name, publisher].allSatisfy({ text in text.unicodeScalars.allSatisfy({ $0.value >= 32 && $0.value != 127 }) }) else { return nil }
        self.id = id; self.name = name; self.publisher = publisher
    }
}

enum ModelBrowserSort: String, CaseIterable {
    case nameAscending, nameDescending, sizeAscending, sizeDescending

    var title: String {
        switch self {
        case .nameAscending: return "Name (A–Z)"
        case .nameDescending: return "Name (Z–A)"
        case .sizeAscending: return "Size (smallest first)"
        case .sizeDescending: return "Size (largest first)"
        }
    }

    func precedes(_ firstName: String, id firstID: String, size firstSize: Double?,
                  _ secondName: String, id secondID: String, size secondSize: Double?) -> Bool {
        if self == .sizeAscending || self == .sizeDescending {
            if firstSize != secondSize {
                guard let first = firstSize else { return false }
                guard let second = secondSize else { return true }
                return self == .sizeAscending ? first < second : first > second
            }
        }
        let comparison = firstName.localizedStandardCompare(secondName)
        if comparison != .orderedSame {
            return self == .nameDescending ? comparison == .orderedDescending : comparison == .orderedAscending
        }
        return firstID < secondID
    }
}

enum ModelBrowserScope: String, CaseIterable {
    case saved, families, catalog
    var preferenceKey: String { "modelBrowser.sort." + rawValue }
    var defaultSort: ModelBrowserSort { self == .catalog ? .sizeAscending : .nameAscending }
    var sorts: [ModelBrowserSort] {
        self == .families ? [.nameAscending, .nameDescending] : ModelBrowserSort.allCases
    }
}

enum ModelBrowserPreferences {
    static func load(_ scope: ModelBrowserScope, defaults: UserDefaults = .standard) -> ModelBrowserSort {
        guard let raw = defaults.string(forKey: scope.preferenceKey), let sort = ModelBrowserSort(rawValue: raw),
              scope.sorts.contains(sort) else { return scope.defaultSort }
        return sort
    }

    static func save(_ sort: ModelBrowserSort, for scope: ModelBrowserScope, defaults: UserDefaults = .standard) {
        guard scope.sorts.contains(sort) else { return }
        defaults.set(sort.rawValue, forKey: scope.preferenceKey)
    }
}

enum ModelBrowserSearch {
    private static func normalized(_ value: String) -> String {
        value.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(String.init).joined()
    }

    static func matches(_ query: String, fields: [String]) -> Bool {
        let query = query.replacingOccurrences(of: "(?i)\\b([48])[-_ ]*bits?\\b", with: "$1bit", options: .regularExpression)
        let terms = query.split(whereSeparator: \.isWhitespace).map { normalized(String($0)) }.filter { !$0.isEmpty }
        let text = fields.map(normalized).joined(separator: " ")
        return terms.allSatisfy { text.contains($0) }
    }
}

struct FamilyModelCatalog {
    enum Status: String { case ready, offline, error }
    struct Item: Identifiable {
        let id: String
        let repo: String
        let revision: String
        let name: String
        let sizeLabel: String?
        let parameterCount: Int64?
        let weightBytes: Int64?
        let quantization: String
        let artifactModes: [ModelArtifactMode]
        let defaultArtifactMode: ModelArtifactMode
        let downloadable: Bool
        let installedIDs: [String]
        let modelGroup: CatalogModelGroup?

        init?(_ raw: [String: Any]) {
            guard let id = SupportMetadata.string(raw["id"], maximum: 512),
                  let repo = SupportMetadata.string(raw["repo"], maximum: 256),
                  let revision = SupportMetadata.revision(raw["revision"]),
                  let name = SupportMetadata.string(raw["name"], maximum: 256),
                  let quantization = SupportMetadata.string(raw["quantization"], maximum: 64),
                  let modes = raw["artifact_modes"] as? [String], !modes.isEmpty, modes.count <= 2,
                  Set(modes).count == modes.count,
                  let defaultMode = (raw["default_artifact_mode"] as? String).flatMap(ModelArtifactMode.init(rawValue:)),
                  modes.contains(defaultMode.rawValue),
                  let downloadable = SupportMetadata.boolean(raw["downloadable"]),
                  let installedIDs = SupportMetadata.strings(raw["installed_ids"], maximumCount: 128),
                  installedIDs.allSatisfy({ $0.utf8.count <= 128 }), Set(installedIDs).count == installedIDs.count else { return nil }
            let artifactModes = modes.compactMap(ModelArtifactMode.init(rawValue:))
            guard artifactModes.count == modes.count else { return nil }
            if let size = raw["size_label"], !(size is NSNull), SupportMetadata.string(size, maximum: 64) == nil { return nil }
            for key in ["parameter_count", "weight_bytes"] {
                if let value = raw[key], !(value is NSNull), SupportMetadata.integer(value) == nil { return nil }
            }
            self.id = id; self.repo = repo; self.revision = revision; self.name = name
            sizeLabel = raw["size_label"] as? String; parameterCount = SupportMetadata.integer(raw["parameter_count"])
            weightBytes = SupportMetadata.integer(raw["weight_bytes"]); self.quantization = quantization
            self.artifactModes = artifactModes; defaultArtifactMode = defaultMode
            self.downloadable = downloadable; self.installedIDs = installedIDs
            modelGroup = CatalogModelGroup(raw["model_group"])
        }

        var detailLabels: [String] {
            [sizeLabel, quantization, weightBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) }]
                .compactMap { $0 }
        }

        var numericSize: Double? {
            if let label = sizeLabel?.uppercased(), !label.isEmpty {
                let scale: [Character: Double] = ["K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12]
                let multiplier = label.last.flatMap { scale[$0] }
                let value = multiplier == nil ? Double(label) : Double(label.dropLast())
                if let value = value, value.isFinite, value > 0 {
                    let amount = value * (multiplier ?? 1)
                    if amount.isFinite { return amount }
                }
            }
            return parameterCount.flatMap { $0 > 0 ? Double($0) : nil }
        }
    }
    let familyID: String
    let status: Status
    let message: String
    let items: [Item]
    let hasMore: Bool

    init?(_ raw: [String: Any]) {
        guard SupportMetadata.schemaOne(raw),
              let familyID = SupportMetadata.string(raw["family_id"], maximum: 128),
              let status = (raw["status"] as? String).flatMap(Status.init(rawValue:)),
              let message = raw["message"] as? String, message.isEmpty || SupportMetadata.string(message) != nil,
              let rawItems = raw["items"] as? [[String: Any]], rawItems.count <= 30,
              let hasMore = SupportMetadata.boolean(raw["has_more"]) else { return nil }
        let items = rawItems.compactMap(Item.init)
        guard items.count == rawItems.count, Set(items.map(\.id)).count == items.count else { return nil }
        self.familyID = familyID; self.status = status; self.message = message; self.hasMore = hasMore
        self.items = items
    }
}

enum CatalogQuantizationFilter: Hashable {
    case all
    case quantization(String)

    var title: String {
        switch self { case .all: return "All models"; case .quantization(let value): return value }
    }

    static func tabs(for items: [FamilyModelCatalog.Item]) -> [Self] {
        [.all] + Set(items.map(\.quantization)).sorted {
            let order = $0.localizedStandardCompare($1)
            return order == .orderedSame ? $0 < $1 : order == .orderedAscending
        }.map(Self.quantization)
    }

    func resolved(for items: [FamilyModelCatalog.Item]) -> Self {
        Self.tabs(for: items).contains(self) ? self : .all
    }

    func matches(_ item: FamilyModelCatalog.Item) -> Bool {
        switch self { case .all: return true; case .quantization(let value): return item.quantization == value }
    }
}

struct CatalogBrowserInventory {
    struct Group: Identifiable {
        let id: String
        let name: String
        let publisher: String
        let items: [FamilyModelCatalog.Item]

        func size(for sort: ModelBrowserSort) -> Double? {
            let values = items.compactMap(\.numericSize)
            return sort == .sizeDescending ? values.max() : values.min()
        }
    }
    let groups: [Group]

    init(items: [FamilyModelCatalog.Item], query: String = "", sort: ModelBrowserSort = .sizeAscending,
         quantization: CatalogQuantizationFilter = .all) {
        let quantization = quantization.resolved(for: items)
        func makeGroup(id: String, name: String, publisher: String, items: [FamilyModelCatalog.Item]) -> Group? {
            let visible = items.filter {
                quantization.matches($0) && ModelBrowserSearch.matches(query, fields: [name, publisher, $0.name, $0.repo, $0.revision,
                                                           $0.sizeLabel ?? "", $0.quantization] + $0.artifactModes.map(\.title))
            }.sorted { first, second in
                let rowSort: ModelBrowserSort = sort == .sizeDescending ? .sizeDescending : .sizeAscending
                if first.numericSize != second.numericSize {
                    return rowSort.precedes(first.name, id: first.id, size: first.numericSize,
                                            second.name, id: second.id, size: second.numericSize)
                }
                let quantization = first.quantization.localizedStandardCompare(second.quantization)
                if quantization != .orderedSame { return quantization == .orderedAscending }
                return ModelBrowserSort.nameAscending.precedes(first.repo, id: first.id, size: nil,
                                                               second.repo, id: second.id, size: nil)
            }
            return visible.isEmpty ? nil : Group(id: id, name: name, publisher: publisher, items: visible)
        }
        let candidates = Dictionary(grouping: items) { $0.modelGroup.map { "group:" + $0.id } ?? "repo:" + $0.repo }
        groups = candidates.flatMap { id, entries -> [Group] in
            let first = entries[0]
            let name = first.modelGroup?.name ?? first.name
            let publisher = first.modelGroup?.publisher ?? first.repo.components(separatedBy: "/")[0]
            if !entries.allSatisfy({ $0.modelGroup == first.modelGroup && ($0.modelGroup != nil || $0.name == first.name) }) {
                return entries.compactMap { item in
                    makeGroup(id: "item:" + item.id, name: item.name, publisher: item.repo.components(separatedBy: "/")[0], items: [item])
                }
            }
            return makeGroup(id: id, name: name, publisher: publisher, items: entries).map { [$0] } ?? []
        }.sorted { first, second in
            sort.precedes(first.name, id: first.id, size: first.size(for: sort),
                          second.name, id: second.id, size: second.size(for: sort))
        }
    }
}

struct CatalogModelInstallation {
    let requestID: String
    let catalogRequestID: String
    let entryID: String
    let name: String
    let artifactMode: ModelArtifactMode
}

struct ModelRemovalConfirmation: Identifiable {
    let id: String
    let action: InstalledModelRemoval.Action
    let message: String

    init?(model: InstalledModel, details: [String], isRunning: Bool) {
        guard let removal = model.removal, removal.action != .unavailable else { return nil }
        id = model.id; action = removal.action
        var parts = ["“" + model.name + "”"]
        if !details.isEmpty { parts.append(details.joined(separator: " · ")) }
        parts.append(removal.message)
        if isRunning { parts.append("This model is running and will be stopped.") }
        message = parts.joined(separator: "\n\n")
    }
}

enum ModelsSheetTab: String, CaseIterable {
    case onThisMac, availableFamilies
    static let initial = Self.onThisMac

    func title(modelCount: Int) -> String {
        self == .onThisMac ? "On this Mac (\(modelCount))" : "Model families"
    }

    var emptyTitle: String { self == .onThisMac ? "No models yet" : "No families available" }
    var emptyMessage: String {
        self == .onThisMac ? "Install a model to add it here." : "No model families are available in this version of Stark."
    }
}

struct ModelsInventory {
    struct Model: Identifiable {
        let id: String
        let name: String
        let isRunning: Bool
        let quantization: String?
        let weightBytes: Int64?
        let sourceRevision: String?
        let removal: InstalledModelRemoval?
        let artifactMode: ModelArtifactMode?
        let sourceName: String
        let repositoryURL: URL?

        var detailLabels: [String] {
            [quantization,
             weightBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) },
             artifactMode == .visionLanguage ? "Text and images" : nil,
             sourceRevision.map { "Version " + $0.prefix(7) }].compactMap { $0 }
        }
    }
    struct Group: Identifiable {
        let id: String
        let name: String
        let family: String?
        let models: [Model]

        var totalBytes: Double? {
            guard models.allSatisfy({ $0.weightBytes != nil }) else { return nil }
            return models.reduce(0) { $0 + Double($1.weightBytes!) }
        }
    }
    struct Family: Identifiable {
        let id: String
        let name: String
    }
    let models: [Model]
    let groups: [Group]
    let runningGroup: Group?
    let otherGroups: [Group]
    let families: [Family]?

    init(models: [InstalledModel], loadedID: String, registry: ModelFamilyRegistry?,
         query: String = "", sort: ModelBrowserSort = .nameAscending) {
        func row(_ model: InstalledModel) -> Model {
            Model(id: model.id, name: model.name, isRunning: !loadedID.isEmpty && model.id == loadedID,
                  quantization: model.display?.quantization, weightBytes: model.display?.weightBytes,
                  sourceRevision: model.display?.sourceRevision, removal: model.removal, artifactMode: model.artifactMode,
                  sourceName: model.display?.groupID ?? model.apiID, repositoryURL: model.display?.repositoryURL)
        }
        let allModels = models.sorted { Self.ordered($0.name, $1.name, $0.id, $1.id) }.map(row)
        let candidates = Dictionary(grouping: models) { model in model.display.map { "source:" + $0.groupID } ?? "model:" + model.id }
        let allGroups = candidates.flatMap { key, entries -> [Group] in
            let first = entries[0]
            let consistent = entries.allSatisfy { $0.display?.groupName == first.display?.groupName && $0.display?.familyID == first.display?.familyID }
            if !consistent {
                return entries.map { Group(id: "model:" + $0.id, name: $0.name, family: nil, models: [row($0)]) }
            }
            let rows = entries.map(row).sorted(by: Self.variantOrder)
            return [Group(id: key, name: first.display?.groupName ?? first.name,
                          family: first.display?.familyID.flatMap(ModelFamily.displayName(for:)), models: rows)]
        }
        func groupOrder(_ first: Group, _ second: Group) -> Bool {
            let firstName = first.models.count == 1 ? first.models[0].name : first.name
            let secondName = second.models.count == 1 ? second.models[0].name : second.name
            return sort.precedes(firstName, id: first.id, size: first.totalBytes,
                                 secondName, id: second.id, size: second.totalBytes)
        }
        groups = allGroups.compactMap { group in
            var visible = group.models.filter { model in
                ModelBrowserSearch.matches(query, fields: [group.name, group.family ?? "", model.name, model.sourceName,
                                                           model.sourceRevision ?? "", model.artifactMode?.title ?? "",
                                                           model.isRunning ? "Running" : ""] + model.detailLabels)
            }
            if sort == .sizeAscending || sort == .sizeDescending {
                visible.sort { first, second in
                    sort.precedes(first.name, id: first.id, size: first.weightBytes.map(Double.init),
                                  second.name, id: second.id, size: second.weightBytes.map(Double.init))
                }
            }
            return visible.isEmpty ? nil : Group(id: group.id, name: group.name, family: group.family, models: visible)
        }.sorted(by: groupOrder)
        runningGroup = groups.compactMap { group -> Group? in
            guard let active = group.models.first(where: \.isRunning) else { return nil }
            return Group(id: group.id, name: group.name, family: group.family, models: [active])
        }.first
        otherGroups = groups.compactMap { group -> Group? in
            let remaining = group.models.filter { !$0.isRunning }
            return remaining.isEmpty ? nil : Group(id: group.id, name: group.name, family: group.family, models: remaining)
        }.sorted(by: groupOrder)
        let visibleIDs = Set(groups.flatMap { $0.models.map(\.id) })
        self.models = allModels.filter { visibleIDs.contains($0.id) }
        families = registry?.families.filter { family in ModelSupportMode.allCases.contains(where: family.hasAppMode) }
            .sorted { Self.ordered($0.displayName, $1.displayName, $0.id, $1.id) }
            .map { Family(id: $0.id, name: $0.displayName) }
    }

    private static func ordered(_ first: String, _ second: String, _ firstID: String, _ secondID: String) -> Bool {
        let comparison = first.localizedStandardCompare(second)
        return comparison == .orderedSame ? firstID < secondID : comparison == .orderedAscending
    }

    private static func variantOrder(_ first: Model, _ second: Model) -> Bool {
        if first.quantization != second.quantization {
            guard let firstLabel = first.quantization else { return false }
            guard let secondLabel = second.quantization else { return true }
            let comparison = firstLabel.localizedStandardCompare(secondLabel)
            return comparison == .orderedSame ? firstLabel < secondLabel : comparison == .orderedAscending
        }
        if first.weightBytes != second.weightBytes {
            guard let firstBytes = first.weightBytes else { return false }
            guard let secondBytes = second.weightBytes else { return true }
            return firstBytes < secondBytes
        }
        if first.sourceRevision != second.sourceRevision { return (first.sourceRevision ?? "~") < (second.sourceRevision ?? "~") }
        if first.artifactMode != second.artifactMode {
            guard let firstMode = first.artifactMode else { return false }
            guard let secondMode = second.artifactMode else { return true }
            return firstMode.rawValue < secondMode.rawValue
        }
        return ordered(first.name, second.name, first.id, second.id)
    }
}

struct ModelSupportReport {
    enum Status: String { case conditional, unsupported, inconclusive }
    let status: Status
    let mode: ModelSupportMode
    let message: String
    let reasons: [String]
    let warnings: [String]
    let pendingChecks: [String]
    let modes: [ModelSupportMode: Status]

    init?(_ value: [String: Any]) {
        guard SupportMetadata.schemaOne(value),
              let status = (value["status"] as? String).flatMap(Status.init(rawValue:)),
              let mode = (value["mode"] as? String).flatMap(ModelSupportMode.init(rawValue:)),
              let message = SupportMetadata.string(value["message"]),
              let reasons = SupportMetadata.strings(value["reasons"]),
              let warnings = SupportMetadata.strings(value["warnings"]),
              let checks = SupportMetadata.strings(value["pending_checks"]),
              let modes = SupportMetadata.modes(value["modes"]) else { return nil }
        self.status = status; self.mode = mode; self.message = message; self.reasons = reasons; self.warnings = warnings
        pendingChecks = checks; self.modes = modes
    }

    var details: [String] {
        var seen = Set<String>()
        return (warnings + pendingChecks).map(SupportMetadata.displayCheck).filter { $0 != message && seen.insert($0).inserted }
    }

    var alternativeModes: [ModelSupportMode] {
        ModelSupportMode.allCases.filter { $0 != mode && modes[$0] == .conditional }
    }
}

final class EngineController: ObservableObject {
    @Published var models: [InstalledModel] = []
    @Published var selectedID = UserDefaults.standard.string(forKey: "selectedModel") ?? ""
    @Published var loadedID = ""
    @Published var state = "idle"
    @Published var message = "Starting Stark…"
    @Published var errorMessage: String? {
        didSet { installErrorMessage = nil }
    }
    @Published var apiBaseURL = ""
    @Published var apiKey = ""
    @Published var apiMessage = "Starting the local API…"
    @Published var apiContextTokens = 0
    @Published var apiMaxNewTokens = 0
    @Published var messages: [ChatMessage] = []
    @Published var draft = ""
    @Published private(set) var draftImage: ImageAttachment?
    @Published private(set) var conversationImage: ImageAttachment?
    @Published private(set) var supportsImages = false
    @Published private(set) var imageVerification = "unverified"
    @Published private(set) var imageProgress = ""
    @Published private(set) var attachmentLoading = false
    @Published private(set) var mode = PlaygroundMode(rawValue: UserDefaults.standard.string(forKey: "playgroundMode") ?? "") ?? .chat
    @Published private(set) var loadedMode: PlaygroundMode?
    @Published var decisionText = ""
    @Published var decisionQuestion = ""
    @Published var decisionAnswers = "Yes, No"
    @Published private(set) var decisionResult: DecisionResult?
    @Published var finishReason = ""
    @Published var inputSpeed: Double?
    @Published var outputSpeed: Double?
    @Published var inputTokens = 0
    @Published var outputTokens = 0
    @Published var downloadProgress = InstallProgress()
    @Published var conversionProgress = InstallProgress()
    @Published private(set) var modelFamilies: ModelFamilyRegistry?
    @Published private(set) var installSupportReport: ModelSupportReport?
    @Published private(set) var catalog: FamilyModelCatalog?
    @Published private(set) var catalogFamilyID: String?
    @Published private(set) var catalogLoading = false
    @Published private(set) var catalogError: String?
    @Published private(set) var catalogInstallation: CatalogModelInstallation?
    @Published private(set) var catalogInstallMessage: String?
    @Published private(set) var catalogInstallFailed = false
    @Published private(set) var removingModelID: String?
    @Published private(set) var removalError: String?
    @Published var options = ChatOptions()
    @Published var contextLimit = 8192
    @Published var outputLimit = 512
    var onInstalled: (() -> Void)?
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var pending = Data()
    private let reader = DispatchQueue(label: "stark.desktop.reader")
    private var shuttingDown = false
    private var pendingMessageID: UUID?
    private var pendingOptions: ChatOptions?
    private var pendingDecisionAnswers: [String]?
    private var attachmentStaging: AttachmentStaging?
    private var attachmentSelection = UUID()
    private var pendingSupportRequestID: String?
    private var installErrorMessage: String?
    private var catalogRequestID: String?
    private var activeInstallRequestID: String?
    private var removalRequestID: String?
    private var removalAction: InstalledModelRemoval.Action?

    var busy: Bool {
        ["loading", "generating", "deciding", "installing", "stopping", "configuring", "removing"].contains(state) ||
            catalogInstallation != nil || removingModelID != nil
    }
    var modelReady: Bool { state == "ready" && !busy && !loadedID.isEmpty && loadedID == selectedID && loadedMode == mode }
    var modelControlID: String {
        if loadedMode == nil && (state == "loading" || state == "stopping") { return selectedID }
        return loadedID
    }
    var activeName: String { models.first { $0.id == loadedID }?.name ?? "No model running" }
    var canAttachImage: Bool { modelReady && supportsImages && conversationImage == nil }
    var imageVerificationNote: String {
        switch imageVerification {
        case "locally_tested": return "Image support checked on this Mac. Reference parity is unverified."
        case "reference_verified": return "Image support has reference-parity verification."
        default: return "Image support available."
        }
    }
    var hasConversation: Bool { !messages.isEmpty || draftImage != nil || conversationImage != nil }
    var canSendMessage: Bool {
        modelReady && !attachmentLoading &&
            (!draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || draftImage != nil)
    }
    var statusColor: Color {
        if state == "error" { return .red }
        if state == "ready" { return .green }
        return busy ? .orange : .secondary
    }

    func start(resourceURL: URL? = Bundle.main.resourceURL, dataDirectory: URL? = nil) {
        guard process == nil, let resources = resourceURL else { return }
        clearInstallSupport()
        modelFamilies = nil
        clearBrowserRequests()
        let runtime = resources.appendingPathComponent("runtime")
        let python = runtime.appendingPathComponent("libexec/python/bin/python3.13")
        let data = dataDirectory ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Stark", isDirectory: true)
        do { attachmentStaging = try AttachmentStaging() }
        catch { state = "error"; message = "Cannot prepare private image storage. Quit and reopen Stark."; return }
        let child = Process()
        child.executableURL = python
        child.arguments = ["-I", "-B", "-m", "stark_convert.desktop", "--worker",
                           runtime.appendingPathComponent("libexec/stark-generate").path,
                           "--score-worker", runtime.appendingPathComponent("libexec/stark-score").path,
                           "--data-dir", data.path,
                           "--attachment-root", attachmentStaging!.root.path]
        let incoming = Pipe()
        let outgoing = Pipe()
        child.standardInput = incoming
        child.standardOutput = outgoing
        child.standardError = FileHandle.nullDevice
        // Keep native diagnostics and Python history out of inherited capture paths.
        child.environment = ProcessInfo.processInfo.environment.filter {
            !$0.key.hasPrefix("STARK_CAPTURE_") && !$0.key.hasPrefix("UZU_CAPTURE_") &&
            !["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"].contains($0.key)
        }
        child.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self = self, !self.shuttingDown else { return }
                if self.state == "installing" { self.endInstallProgress("failed") }
                self.state = "error"
                self.message = "Stark stopped. Quit and reopen the app to restart."
                self.loadedID = ""
                self.apiBaseURL = ""
                self.apiKey = ""
                self.apiMessage = "Stark stopped. Reopen the app to reconnect."
                self.loadedMode = nil
                self.supportsImages = false
                self.imageVerification = "unverified"
                self.pendingOptions = nil
                self.clearInstallSupport()
                self.modelFamilies = nil
                self.clearBrowserRequests()
                self.input = nil
                self.discardPendingMessage()
                self.clearConversation()
                self.clearDecision()
                self.attachmentStaging = nil
            }
        }
        output = outgoing.fileHandleForReading
        output?.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData
            guard !bytes.isEmpty else { handle.readabilityHandler = nil; return }
            self?.reader.async { [weak self] in self?.receive(bytes) }
        }
        do {
            try child.run()
            process = child
            input = incoming.fileHandleForWriting
            send(["command": "list"])
        } catch {
            output?.readabilityHandler = nil
            state = "error"
            message = "Cannot start the bundled runtime. Reinstall Stark."
            attachmentStaging = nil
        }
    }

    private func receive(_ bytes: Data) {
        pending.append(bytes)
        guard pending.count <= 20 * 1024 * 1024 else {
            pending.removeAll()
            DispatchQueue.main.async { self.stop(); self.message = "Worker reply exceeded the size limit." }
            return
        }
        while let boundary = pending.firstIndex(of: 10) {
            let line = pending.prefix(upTo: boundary)
            pending.removeSubrange(...boundary)
            guard let value = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
            DispatchQueue.main.async { [weak self] in self?.apply(value) }
        }
    }

    private func apply(_ event: [String: Any]) {
        switch event["event"] as? String {
        case "connection":
            apiBaseURL = event["base_url"] as? String ?? ""
            apiKey = event["api_key"] as? String ?? ""
            apiMessage = event["message"] as? String ?? "Local API is available on this Mac."
        case "models":
            models = (event["models"] as? [[String: Any]] ?? []).compactMap {
                guard let id = $0["id"] as? String, let name = $0["name"] as? String,
                      let path = $0["path"] as? String else { return nil }
                return InstalledModel(id: id, name: name, path: path, apiID: $0["api_id"] as? String ?? name,
                                      display: InstalledModelDisplay($0["display"]),
                                      artifactMode: $0["artifact_mode"] == nil ? .text : ModelArtifactMode(rawValue: $0["artifact_mode"] as? String ?? ""),
                                      removal: InstalledModelRemoval($0["removal"]))
            }
            if !models.contains(where: { $0.id == selectedID }) && !(catalogInstallation != nil && selectedID.isEmpty) {
                selectedID = models.first?.id ?? ""
            }
        case "model_families":
            guard let raw = event["report"] as? [String: Any], let registry = ModelFamilyRegistry(raw) else { return }
            modelFamilies = registry
        case "model_support":
            guard state == "installing", let expected = pendingSupportRequestID,
                  event["request_id"] as? String == expected,
                  let raw = event["report"] as? [String: Any], let report = ModelSupportReport(raw) else { return }
            installSupportReport = report
            message = report.message
        case "model_catalog":
            guard let expected = catalogRequestID, event["request_id"] as? String == expected,
                  event["family_id"] as? String == catalogFamilyID,
                  let raw = event["report"] as? [String: Any], let result = FamilyModelCatalog(raw),
                  result.familyID == catalogFamilyID else { return }
            catalog = result; catalogLoading = false
            catalogError = result.status == .ready ? nil : (result.message.isEmpty ? "Could not load these models. Try again." : result.message)
        case "model_removed", "model_remove_error":
            guard let expected = removalRequestID, event["request_id"] as? String == expected,
                  event["model_id"] as? String == removingModelID else { return }
            if event["event"] as? String == "model_remove_error" {
                removalError = event["message"] as? String ?? "Could not remove this model. Try again."
            } else {
                guard event["action"] as? String == removalAction?.rawValue else { return }
                removalError = nil
            }
            if let stopped = event["stopped_model_id"] as? String, stopped == loadedID {
                loadedID = ""; loadedMode = nil; supportsImages = false; imageVerification = "unverified"
                discardPendingMessage(); clearConversation(); clearDecision()
            }
            removalRequestID = nil; removingModelID = nil; removalAction = nil
            if state == "removing" { state = loadedID.isEmpty ? "idle" : "ready"; message = loadedID.isEmpty ? "Stopped" : "Ready" }
        case "status":
            state = event["state"] as? String ?? "idle"
            if state != "installing" { pendingSupportRequestID = nil }
            message = event["message"] as? String ?? ""
            if state == "idle" || state == "error" {
                loadedID = ""
                loadedMode = nil
                discardPendingMessage()
                clearConversation()
                clearDecision()
                supportsImages = false
                imageVerification = "unverified"
            }
        case "ready":
            guard let readyMode = PlaygroundMode(rawValue: event["mode"] as? String ?? "chat"), readyMode == mode,
                  event["model_id"] as? String == selectedID, state != "stopping" else { return }
            loadedID = event["model_id"] as? String ?? selectedID
            apiContextTokens = event["max_context_tokens"] as? Int ?? 0
            apiMaxNewTokens = event["max_output_tokens"] as? Int ?? 0
            loadedMode = readyMode
            applyLimits(event)
            applyImageCapabilities(event)
            state = "ready"
            message = "Ready"
        case "configured":
            guard let configured = pendingOptions else { return }
            options = configured
            pendingOptions = nil
            applyLimits(event)
            applyImageCapabilities(event)
            if state == "configuring" {
                clearConversation()
                state = "ready"
                message = "Ready"
            }
        case "installed":
            if let requestID = event["support_request_id"] as? String {
                guard requestID == activeInstallRequestID else { return }
            } else if catalogInstallation != nil { return }
            pendingSupportRequestID = nil
            activeInstallRequestID = nil
            if let completed = catalogInstallation {
                catalogInstallation = nil
                catalogInstallMessage = "Downloaded"; catalogInstallFailed = false
                if let familyID = catalogFamilyID { requestModelCatalog(familyID: familyID) }
                else { send(["command": "catalog_cancel", "request_id": completed.catalogRequestID]) }
                return
            }
            if let model = event["model"] as? [String: Any], let id = model["id"] as? String {
                selectedID = id
                UserDefaults.standard.set(id, forKey: "selectedModel")
            }
            onInstalled?()
        case "reply":
            guard mode == .chat, loadedMode == .chat, pendingMessageID != nil,
                  let text = event["text"] as? String else { return }
            messages.append(ChatMessage(text: text, isUser: false))
            if conversationImage == nil, let id = pendingMessageID {
                conversationImage = messages.first { $0.id == id }?.image
            }
            pendingMessageID = nil
            imageProgress = ""
            attachmentStaging?.clear()
            finishReason = event["finish_reason"] as? String ?? ""
            applyMetrics(event)
        case "metrics":
            applyMetrics(event)
        case "image_progress":
            guard state == "generating", pendingMessageID != nil else { return }
            if event["phase"] as? String == "preparing" { imageProgress = "Preparing image…" }
            if event["phase"] as? String == "generating" { imageProgress = "Generating reply…" }
        case "image_check":
            guard state == "installing", event["state"] as? String == "checking" else { return }
            loadedID = ""; loadedMode = nil; supportsImages = false; imageVerification = "unverified"
            clearConversation(); clearDecision()
            message = "Checking image support on this Mac…"
        case "decision":
            guard mode == .decision, loadedMode == .decision, state == "deciding",
                  let expected = pendingDecisionAnswers else { return }
            guard let answers = event["answers"] as? [String], answers == expected,
                  let probabilities = event["probabilities"] as? [Double], probabilities.count == answers.count,
                  let index = event["choice_index"] as? Int, answers.indices.contains(index),
                  probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                  abs(probabilities.reduce(0, +) - 1) <= 0.0001,
                  probabilities[index] == probabilities.max() else {
                clearDecision()
                errorMessage = "Stark returned an invalid decision. Run the model again."
                stopAfterInvalidDecision()
                return
            }
            decisionResult = DecisionResult(answers: answers, probabilities: probabilities, choiceIndex: index)
            pendingDecisionAnswers = nil
        case "error":
            let requestID = event["request_id"] as? String
            if let requestID = requestID, requestID == removalRequestID {
                removalError = event["message"] as? String ?? "Could not remove this model. Try again."
                removalRequestID = nil; removingModelID = nil; removalAction = nil
                if state == "removing" { state = loadedID.isEmpty ? "idle" : "ready"; message = loadedID.isEmpty ? "Stopped" : "Ready" }
                return
            }
            if let requestID = requestID, requestID != activeInstallRequestID { return }
            let matchingInstallError = requestID != nil && requestID == activeInstallRequestID
            let installOwned = requestID != nil || state == "installing" || (activeInstallRequestID != nil && catalogInstallation == nil)
            pendingSupportRequestID = nil
            errorMessage = event["message"] as? String ?? "Unable to complete the operation."
            if installOwned {
                installErrorMessage = errorMessage
                activeInstallRequestID = nil
                endInstallProgress("failed")
                if let failed = catalogInstallation {
                    catalogInstallMessage = errorMessage; catalogInstallFailed = true; catalogInstallation = nil
                    if catalogFamilyID == nil { send(["command": "catalog_cancel", "request_id": failed.catalogRequestID]) }
                }
            }
            discardPendingMessage()
            pendingOptions = nil
            clearDecision()
            if state == "configuring" || state == "deciding" || (matchingInstallError && state == "installing") {
                state = loadedID.isEmpty ? "idle" : "ready"
                message = loadedID.isEmpty ? "Stopped" : "Ready"
            }
        case "progress":
            guard let phase = event["phase"] as? String, ["download", "conversion"].contains(phase),
                  let progressState = event["state"] as? String,
                  ["waiting", "running", "complete", "skipped", "cancelled", "failed"].contains(progressState),
                  let completed = event["completed_bytes"] as? Int64, completed >= 0 else { return }
            let total = event["total_bytes"] as? Int64
            guard total == nil || (total! >= 0 && completed <= total!) else { return }
            let progress = InstallProgress(state: progressState, completedBytes: completed, totalBytes: total)
            if phase == "download" { downloadProgress = progress } else { conversionProgress = progress }
        default: break
        }
    }

    private func applyMetrics(_ event: [String: Any]) {
        inputTokens = event["input_tokens"] as? Int ?? 0
        outputTokens = event["output_tokens"] as? Int ?? 0
        inputSpeed = rate(event["input_tokens_per_second"])
        outputSpeed = rate(event["output_tokens_per_second"])
    }

    private func rate(_ value: Any?) -> Double? {
        guard let number = value as? Double, number.isFinite, number >= 0 else { return nil }
        return number
    }

    private func applyLimits(_ event: [String: Any]) {
        contextLimit = event["max_context_tokens"] as? Int ?? contextLimit
        outputLimit = event["max_output_tokens"] as? Int ?? outputLimit
        options.contextTokens = event["context_tokens"] as? Int ?? options.contextTokens
        options.maxNewTokens = event["max_new_tokens"] as? Int ?? options.maxNewTokens
    }

    private func applyImageCapabilities(_ event: [String: Any]) {
        supportsImages = mode == .chat && event["images"] as? Bool == true &&
            event["image_attachment_transport"] as? String == "private-file-v1" &&
            event["max_image_bytes"] as? Int == ImageAttachment.maximumBytes
        let verification = event["image_verification"] as? String ?? "unverified"
        imageVerification = supportsImages && ["locally_tested", "reference_verified"].contains(verification)
            ? verification : "unverified"
        if !supportsImages {
            draftImage = nil; conversationImage = nil; attachmentSelection = UUID(); attachmentLoading = false
        }
    }

    func selectImage() {
        guard canAttachImage else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { attachImage(url) }
    }

    func attachImage(_ url: URL) {
        guard canAttachImage else { return }
        do { try attachImage(bytes: ImageAttachment.read(url).bytes) }
        catch { imageError(error) }
    }

    func attachImage(bytes: Data) throws {
        guard canAttachImage else { throw ImageAttachmentError.unavailable }
        let image = try ImageAttachment(bytes: bytes)
        attachmentSelection = UUID()
        attachmentLoading = false
        draftImage = image
        errorMessage = nil
    }

    func removeImage() {
        guard canAttachImage else { return }
        attachmentSelection = UUID()
        attachmentLoading = false
        draftImage = nil
    }

    @discardableResult
    func acceptImageProviders(_ providers: [NSItemProvider]) -> Bool {
        guard canAttachImage else { return false }
        guard providers.count == 1, let provider = providers.first else {
            errorMessage = "Attach one image per conversation."
            return false
        }
        let selection = UUID()
        attachmentSelection = selection
        attachmentLoading = true
        let receive: (Result<ImageAttachment, Error>) -> Void = { [weak self] result in
            DispatchQueue.main.async {
                guard let self = self, self.attachmentSelection == selection, self.canAttachImage else { return }
                self.attachmentLoading = false
                do { self.draftImage = try result.get(); self.errorMessage = nil }
                catch { self.imageError(error) }
            }
        }
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier) { data, _ in
                guard let data = data, let url = URL(dataRepresentation: data, relativeTo: nil) else {
                    receive(.failure(ImageAttachmentError.unavailable)); return
                }
                receive(Result { try ImageAttachment.read(url) })
            }
            return true
        }
        for type in [UTType.png, .jpeg] where provider.hasItemConformingToTypeIdentifier(type.identifier) {
            provider.loadFileRepresentation(forTypeIdentifier: type.identifier) { url, _ in
                guard let url = url else { receive(.failure(ImageAttachmentError.unavailable)); return }
                receive(Result { try ImageAttachment.read(url) })
            }
            return true
        }
        errorMessage = "Choose one PNG or JPEG image."
        attachmentLoading = false
        return false
    }

    private func imageError(_ error: Error) {
        errorMessage = (error as? ImageAttachmentError)?.message ?? ImageAttachmentError.unavailable.message
    }

    @discardableResult
    func configure(_ updated: ChatOptions) -> Bool {
        guard modelReady, mode == .chat else { return false }
        errorMessage = nil
        let payload: [String: Any] = [
            "command": "configure", "context_tokens": updated.contextTokens,
            "max_new_tokens": updated.maxNewTokens, "thinking": updated.thinking,
            "system_prompt": updated.useModelSystemPrompt ? NSNull() : updated.systemPrompt as Any
        ]
        guard send(payload) else { return false }
        pendingOptions = updated
        state = "configuring"
        message = "Applying advanced options…"
        return true
    }

    @discardableResult
    func send(_ value: [String: Any]) -> Bool {
        guard let input = input, var bytes = try? JSONSerialization.data(withJSONObject: value) else {
            state = "error"; message = "Cannot reach the runtime. Quit and reopen Stark."
            return false
        }
        guard bytes.count < 1024 * 1024 else {
            errorMessage = "The prompt is too large. Send a shorter message."
            state = loadedID.isEmpty ? "idle" : "ready"
            message = loadedID.isEmpty ? "Stopped" : "Ready"
            return false
        }
        bytes.append(10)
        do { try input.write(contentsOf: bytes); return true }
        catch { state = "error"; message = "Cannot reach the runtime. Quit and reopen Stark."; return false }
    }

    @discardableResult
    func run() -> Bool {
        guard !selectedID.isEmpty, !busy else { return false }
        UserDefaults.standard.set(selectedID, forKey: "selectedModel")
        errorMessage = nil
        clearConversation()
        clearDecision()
        loadedMode = nil
        supportsImages = false
        imageVerification = "unverified"
        state = "loading"; message = "Loading model…"
        return send(["command": "run", "model_id": selectedID, "mode": mode.rawValue])
    }

    @discardableResult
    func runModel(id: String) -> Bool {
        guard !busy, models.contains(where: { $0.id == id }) else { return false }
        selectedID = id
        if modelReady {
            UserDefaults.standard.set(id, forKey: "selectedModel")
            return true
        }
        return run()
    }

    func canStopModel(id: String) -> Bool {
        !id.isEmpty && id == modelControlID && models.contains(where: { $0.id == id }) &&
            state != "stopping" && state != "installing" && state != "removing" &&
            catalogInstallation == nil && removingModelID == nil
    }

    func stopModel(id: String) {
        guard canStopModel(id: id) else { return }
        stop()
    }

    func changeMode(_ updated: PlaygroundMode) {
        guard !busy, updated != mode else { return }
        let reload = !loadedID.isEmpty
        mode = updated
        clearInstallSupport()
        UserDefaults.standard.set(updated.rawValue, forKey: "playgroundMode")
        errorMessage = nil
        clearConversation()
        clearDecision()
        if reload { run() }
    }

    func install(source: String, kind: String, artifactMode: String? = nil) {
        guard !busy else { return }
        errorMessage = nil
        resetInstallProgress(kind: kind)
        let requestID = UUID().uuidString.lowercased()
        pendingSupportRequestID = requestID
        activeInstallRequestID = requestID
        catalogInstallation = nil; catalogInstallMessage = nil; catalogInstallFailed = false
        let purpose = artifactMode == "vision-language" ? "vision-language" : (mode == .decision ? "decision" : "text")
        state = "installing"; message = "Checking model support…"
        var payload: [String: Any] = ["command": "install", "source": source, "kind": kind,
                                      "support_mode": purpose, "support_request_id": requestID]
        if let artifactMode = artifactMode { payload["artifact_mode"] = artifactMode }
        if !send(payload) { pendingSupportRequestID = nil; activeInstallRequestID = nil; endInstallProgress("failed") }
    }

    func requestModelCatalog(familyID: String) {
        guard SupportMetadata.string(familyID, maximum: 128) != nil else { return }
        let changedFamily = catalogFamilyID != familyID
        cancelModelCatalog()
        if changedFamily && catalogInstallation == nil {
            catalogInstallMessage = nil
            catalogInstallFailed = false
            if !busy { resetInstallProgress(kind: "huggingface") }
        }
        let requestID = UUID().uuidString.lowercased()
        catalogRequestID = requestID; catalogFamilyID = familyID; catalogLoading = true
        if !send(["command": "catalog", "family_id": familyID, "request_id": requestID]) {
            catalogLoading = false; catalogError = "Could not reach Stark. Reopen the app and try again."
        }
    }

    func cancelModelCatalog() {
        if let requestID = catalogRequestID, catalogInstallation?.catalogRequestID != requestID {
            send(["command": "catalog_cancel", "request_id": requestID])
        }
        catalogRequestID = nil; catalogFamilyID = nil; catalog = nil; catalogLoading = false; catalogError = nil
    }

    func catalogItemIsDownloaded(_ item: FamilyModelCatalog.Item, artifactMode: ModelArtifactMode) -> Bool {
        models.contains { item.installedIDs.contains($0.id) && $0.artifactMode == artifactMode }
    }

    func installCatalogItem(id: String, artifactMode: ModelArtifactMode) {
        guard !busy, let requestID = catalogRequestID, let catalog = catalog, catalog.status == .ready,
              let item = catalog.items.first(where: { $0.id == id }), item.downloadable,
              item.artifactModes.contains(artifactMode), !catalogItemIsDownloaded(item, artifactMode: artifactMode) else { return }
        errorMessage = nil; resetInstallProgress(kind: "huggingface")
        let supportID = UUID().uuidString.lowercased()
        pendingSupportRequestID = supportID; activeInstallRequestID = supportID
        catalogInstallation = CatalogModelInstallation(requestID: supportID, catalogRequestID: requestID,
                                                       entryID: id, name: item.name, artifactMode: artifactMode)
        catalogInstallMessage = nil; catalogInstallFailed = false
        state = "installing"; message = "Checking model support…"
        if !send(["command": "install_catalog", "catalog_request_id": requestID, "entry_id": id,
                  "artifact_mode": artifactMode.rawValue, "support_request_id": supportID]) {
            catalogInstallation = nil; activeInstallRequestID = nil; pendingSupportRequestID = nil
            catalogInstallMessage = "Could not start the download. Reopen the app and try again."
            catalogInstallFailed = true
            endInstallProgress("failed")
        }
    }

    func removeModel(id: String, expectedAction: InstalledModelRemoval.Action) {
        guard !busy else { return }
        guard let model = models.first(where: { $0.id == id }), let removal = model.removal,
              removal.action == expectedAction, expectedAction != .unavailable else {
            removalError = "This model changed. Reopen the list and try again."
            return
        }
        let requestID = UUID().uuidString.lowercased()
        removalRequestID = requestID; removingModelID = id; removalAction = expectedAction; removalError = nil
        state = "removing"; message = "Removing model…"
        if !send(["command": "remove_model", "model_id": id, "request_id": requestID, "action": expectedAction.rawValue]) {
            removalRequestID = nil; removingModelID = nil; removalAction = nil
            removalError = "Could not remove this model. Reopen the app and try again."
        }
    }

    private func clearBrowserRequests() {
        catalogRequestID = nil; catalogFamilyID = nil; catalog = nil; catalogLoading = false; catalogError = nil
        activeInstallRequestID = nil; catalogInstallation = nil; catalogInstallMessage = nil; catalogInstallFailed = false
        removalRequestID = nil; removingModelID = nil; removalAction = nil; removalError = nil
    }

    func generate() {
        let text = draft
        guard canSendMessage, mode == .chat else { return }
        attachmentSelection = UUID()
        let image = conversationImage ?? draftImage
        if image != nil && !supportsImages { errorMessage = "The running model does not support image attachments."; return }
        errorMessage = nil
        var payload: [String: Any] = ["command": "generate", "text": text]
        if let image = image {
            imageProgress = "Preparing image…"
            do {
                guard let storage = attachmentStaging else { throw ImageAttachmentError.unavailable }
                payload["image"] = try storage.stage(image)
            } catch { imageProgress = ""; imageError(error); return }
        }
        state = "generating"; message = "Generating…"; finishReason = ""
        if send(payload) {
            let entry = ChatMessage(text: text, isUser: true, image: conversationImage == nil ? draftImage : nil)
            messages.append(entry)
            pendingMessageID = entry.id
            draft = ""
            draftImage = nil
        } else {
            attachmentStaging?.clear()
            imageProgress = ""
        }
    }

    var parsedDecisionAnswers: [String] {
        decisionAnswers.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    var decisionValidation: String? {
        let answers = parsedDecisionAnswers
        if decisionText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter text to evaluate." }
        if decisionQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a question." }
        if !(2...16).contains(answers.count) { return "Enter between 2 and 16 answers." }
        if answers.contains(where: { $0.isEmpty }) { return "Each answer must have a label." }
        if Set(answers.map { $0.lowercased() }).count != answers.count { return "Use distinct answers." }
        return nil
    }

    func decide() {
        guard modelReady, mode == .decision else { return }
        if let validation = decisionValidation { errorMessage = validation; return }
        let answers = parsedDecisionAnswers
        errorMessage = nil
        clearDecision()
        state = "deciding"; message = "Deciding…"
        if send(["command": "decide", "text": decisionText, "question": decisionQuestion, "answers": answers]) {
            pendingDecisionAnswers = answers
        }
    }

    func clearDecision() {
        decisionResult = nil
        pendingDecisionAnswers = nil
    }

    private func stopAfterInvalidDecision() {
        state = "stopping"; message = "Stopping…"
        send(["command": "stop"])
    }

    func reset() {
        guard !busy else { return }
        errorMessage = nil
        if send(["command": "reset"]) { clearConversation(); draft = "" }
    }

    func stop() {
        errorMessage = nil
        clearInstallSupport()
        if state == "installing" { endInstallProgress("cancelled") }
        clearBrowserRequests()
        discardPendingMessage()
        clearConversation()
        clearDecision()
        supportsImages = false
        imageVerification = "unverified"
        state = "stopping"; message = "Stopping…"
        send(["command": "stop"])
    }

    func resetInstallProgress(kind: String) {
        guard state != "installing" else { return }
        clearInstallSupport()
        downloadProgress = InstallProgress(state: kind == "local" ? "skipped" : "waiting")
        conversionProgress = InstallProgress()
    }

    private func clearInstallSupport() {
        if let previous = installErrorMessage, errorMessage == previous { errorMessage = nil }
        installErrorMessage = nil
        pendingSupportRequestID = nil
        installSupportReport = nil
    }

    func requestModelFamilies() { send(["command": "support"]) }

    private func endInstallProgress(_ state: String) {
        if ["waiting", "running"].contains(downloadProgress.state) { downloadProgress.state = state }
        if ["waiting", "running"].contains(conversionProgress.state) { conversionProgress.state = state }
    }

    private func discardPendingMessage() {
        imageProgress = ""
        attachmentStaging?.clear()
        guard let id = pendingMessageID else { return }
        if let entry = messages.first(where: { $0.id == id }) { draft = entry.text; draftImage = entry.image }
        messages.removeAll { $0.id == id }
        pendingMessageID = nil
    }

    private func clearConversation() {
        messages.removeAll(); pendingMessageID = nil
        finishReason = ""; inputSpeed = nil; outputSpeed = nil
        inputTokens = 0; outputTokens = 0
        draftImage = nil; conversationImage = nil; imageProgress = ""
        attachmentLoading = false
        attachmentSelection = UUID()
        attachmentStaging?.clear()
    }

    func shutdown() {
        shuttingDown = true
        clearInstallSupport()
        clearBrowserRequests()
        modelFamilies = nil
        output?.readabilityHandler = nil
        send(["command": "quit"])
        try? input?.close()
        if let child = process, child.isRunning {
            let deadline = Date().addingTimeInterval(5)
            while child.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.02) }
            if child.isRunning { child.terminate() }
        }
        process = nil
        clearConversation()
        supportsImages = false
        imageVerification = "unverified"
        attachmentStaging = nil
    }
}

struct StatusLine: View {
    @ObservedObject var engine: EngineController
    var body: some View {
        HStack(spacing: 7) {
            Circle().fill(engine.statusColor).frame(width: 7, height: 7)
            Text(engine.message).font(.callout).lineLimit(3).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            if engine.busy { ProgressView().controlSize(.small) }
        }.accessibilityElement(children: .combine)
    }
}

struct SpeedView: View {
    @ObservedObject var engine: EngineController
    var body: some View {
        HStack(spacing: 12) {
            metric("Input", engine.inputSpeed)
            Divider().frame(height: 40)
            metric("Output", engine.outputSpeed)
        }
        .padding(12)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .help("Last request. Input uses time to first token, including setup. Output measures subsequent sampled tokens. A dash means no measurement is available.")
    }
    private func metric(_ title: String, _ value: Double?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value.map { String(format: "%.1f", $0) } ?? "—").font(.title3.monospacedDigit().weight(.medium))
                Text("tok/s").font(.caption).foregroundStyle(.secondary)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct MenuView: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var updater: AppUpdateController
    let install: () -> Void
    let manage: () -> Void
    let console: () -> Void
    let connection: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                if let art = Bundle.main.url(forResource: "Stark", withExtension: "jpg"), let image = NSImage(contentsOf: art) {
                    Image(nsImage: image).resizable().scaledToFill().frame(width: 38, height: 38).clipShape(RoundedRectangle(cornerRadius: 8))
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Stark").font(.headline)
                    Text(engine.activeName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
            }
            StatusLine(engine: engine)
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            Button("Manage Models", action: manage).buttonStyle(.borderedProminent)
            if engine.mode == .chat {
                SpeedView(engine: engine)
                Text("Last request").font(.caption2).foregroundStyle(.secondary)
                    .padding(.top, -10)
            }
            HStack {
                Button(engine.state == "installing" ? "Installation Progress…" : "Install Model…", action: install)
                    .disabled(engine.busy && engine.state != "installing")
                Spacer()
                if engine.busy || !engine.loadedID.isEmpty {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                }
            }
            Divider()
            Button("Connection Details…", action: connection)
                .frame(maxWidth: .infinity, alignment: .leading)
            AppUpdateView(updater: updater)
            HStack {
                Button("Open Playground", action: console).buttonStyle(.link)
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }.buttonStyle(.link)
            }.font(.caption)
        }.padding(18).frame(width: 350)
    }
}

struct InstallStageView: View {
    let title: String
    let phase: String
    let progress: InstallProgress
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.callout.weight(.medium))
                Spacer()
                Text(progress.detail(for: phase)).font(.caption.monospacedDigit())
                    .foregroundStyle(progress.state == "failed" ? Color.red : .secondary)
            }
            Group {
                if progress.state == "running" && progress.totalBytes == nil {
                    ProgressView()
                } else {
                    ProgressView(value: progress.fraction)
                }
            }
            .progressViewStyle(.linear)
            .tint(progress.state == "failed" ? .red : .accentColor)
            .accessibilityLabel("\(title) progress")
            .accessibilityValue(progress.detail(for: phase))
            if let total = progress.totalBytes, total > 0 {
                Text("\(bytes(progress.completedBytes)) of \(bytes(total))")
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
    }
    private func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }
}

struct ModelsView: View {
    @ObservedObject var engine: EngineController
    let onDone: () -> Void
    let onRun: (String) -> Void
    @State private var tab = ModelsSheetTab.initial
    @State private var selectedFamily: ModelsInventory.Family?
    @State private var artifactChoices: [String: ModelArtifactMode] = [:]
    @State private var removalConfirmation: ModelRemovalConfirmation?
    @State private var savedQuery = ""
    @State private var familiesQuery = ""
    @State private var catalogQuery = ""
    @State private var catalogQuantization = CatalogQuantizationFilter.all
    @State private var savedSort = ModelBrowserPreferences.load(.saved)
    @State private var familiesSort = ModelBrowserPreferences.load(.families)
    @State private var catalogSort = ModelBrowserPreferences.load(.catalog)

    private var inventory: ModelsInventory {
        ModelsInventory(models: engine.models, loadedID: engine.modelControlID, registry: engine.modelFamilies,
                        query: savedQuery, sort: savedSort)
    }

    private var browserScope: ModelBrowserScope {
        selectedFamily != nil ? .catalog : (tab == .onThisMac ? .saved : .families)
    }

    private var catalogItems: [FamilyModelCatalog.Item] {
        guard let family = selectedFamily, let catalog = engine.catalog,
              catalog.familyID == family.id, catalog.status == .ready else { return [] }
        return catalog.items
    }

    private var quantizationTabs: [CatalogQuantizationFilter] {
        CatalogQuantizationFilter.tabs(for: catalogItems)
    }

    private var quantizationBinding: Binding<CatalogQuantizationFilter> {
        Binding(get: { catalogQuantization.resolved(for: catalogItems) }, set: { catalogQuantization = $0 })
    }

    private var searchPrompt: String {
        switch browserScope {
        case .saved: return "Search downloaded models"
        case .families: return "Search families"
        case .catalog: return "Search models, sizes or quantizations"
        }
    }

    private var queryBinding: Binding<String> {
        let scope = browserScope
        return Binding(get: {
            switch scope { case .saved: return savedQuery; case .families: return familiesQuery; case .catalog: return catalogQuery }
        }, set: { query in
            switch scope { case .saved: savedQuery = query; case .families: familiesQuery = query; case .catalog: catalogQuery = query }
        })
    }

    private var sortBinding: Binding<ModelBrowserSort> {
        let scope = browserScope
        return Binding(get: {
            switch scope { case .saved: return savedSort; case .families: return familiesSort; case .catalog: return catalogSort }
        }, set: { order in
            switch scope { case .saved: savedSort = order; case .families: familiesSort = order; case .catalog: catalogSort = order }
            ModelBrowserPreferences.save(order, for: scope)
        })
    }

    private func visibleFamilies(_ families: [ModelsInventory.Family]) -> [ModelsInventory.Family] {
        families.filter { ModelBrowserSearch.matches(familiesQuery, fields: [$0.name, $0.id]) }.sorted {
            familiesSort.precedes($0.name, id: $0.id, size: nil, $1.name, id: $1.id, size: nil)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                if let family = selectedFamily {
                    Button { leaveFamily() } label: { Label("Back", systemImage: "chevron.left") }.buttonStyle(.link)
                    Text(family.name).font(.title2.weight(.semibold))
                } else { Text("Manage Models").font(.title2.weight(.semibold)) }
                Spacer(minLength: 0)
            }
            if selectedFamily == nil { Picker("Models", selection: $tab) {
                ForEach(ModelsSheetTab.allCases, id: \.self) { choice in
                    Text(choice.title(modelCount: engine.models.count)).tag(choice)
                }
            }.pickerStyle(.segmented).labelsHidden() }
            else {
                Picker("Quantization", selection: quantizationBinding) {
                    ForEach(quantizationTabs, id: \.self) { choice in Text(choice.title).tag(choice) }
                }.pickerStyle(.segmented).labelsHidden().accessibilityLabel("Model quantization")
            }
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField(searchPrompt, text: queryBinding).textFieldStyle(.roundedBorder)
                    .accessibilityLabel(searchPrompt)
                if !queryBinding.wrappedValue.isEmpty {
                    Button { queryBinding.wrappedValue = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain).foregroundStyle(.secondary).help("Clear search").accessibilityLabel("Clear search")
                }
                Picker("Sort", selection: sortBinding) {
                    ForEach(browserScope.sorts, id: \.self) { Text($0.title).tag($0) }
                }.labelsHidden().frame(width: 178).accessibilityLabel("Sort models")
            }
            ScrollView {
                if let family = selectedFamily { catalogContent(family) }
                else if tab == .onThisMac {
                    if engine.models.isEmpty { emptyState(tab) }
                    else if inventory.groups.isEmpty { noSearchResults("No matching models") }
                    else {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            if let active = inventory.runningGroup {
                                Text(engine.state == "loading" ? "Starting" : (engine.state == "stopping" ? "Stopping" : "Running"))
                                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.bottom, 4)
                                modelGroup(active)
                                if !inventory.otherGroups.isEmpty {
                                    Divider().padding(.vertical, 10)
                                    Text("Other models").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.bottom, 4)
                                }
                            }
                            ForEach(inventory.otherGroups) { group in modelGroup(group) }
                        }
                    }
                } else if let families = inventory.families {
                    if families.isEmpty { emptyState(tab) }
                    else if visibleFamilies(families).isEmpty { noSearchResults("No matching families") }
                    else { LazyVStack(spacing: 4) {
                        ForEach(visibleFamilies(families)) { family in
                            Button {
                                selectedFamily = family; artifactChoices = [:]; catalogQuery = ""; catalogQuantization = .all
                                engine.requestModelCatalog(familyID: family.id)
                            } label: {
                                HStack {
                                    Text(family.name).font(.callout.weight(.medium))
                                    Spacer(minLength: 0)
                                    Image(systemName: "chevron.right").font(.caption).foregroundStyle(.secondary)
                                }.frame(maxWidth: .infinity, minHeight: 28, alignment: .leading)
                                .padding(.horizontal, 12)
                                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 7))
                                .overlay(RoundedRectangle(cornerRadius: 7).stroke(.quaternary, lineWidth: 1))
                            }.buttonStyle(.plain).accessibilityLabel("Browse \(family.name) models")
                        }
                    } }
                } else {
                    Text("Available families will appear when Stark is ready.")
                        .foregroundStyle(.secondary).frame(maxWidth: .infinity, minHeight: 300)
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            if let installation = engine.catalogInstallation {
                VStack(alignment: .leading, spacing: 9) {
                    HStack {
                        Text(installation.name).font(.callout.weight(.medium))
                        Spacer(minLength: 0)
                        Button("Stop", action: engine.stop).controlSize(.small)
                    }
                    Text(engine.message).font(.caption).foregroundStyle(.secondary).lineLimit(3)
                    InstallStageView(title: "Download", phase: "download", progress: engine.downloadProgress)
                    InstallStageView(title: "Prepare model", phase: "conversion", progress: engine.conversionProgress)
                }.padding(12).background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 10))
            } else if let result = engine.catalogInstallMessage {
                Text(result).font(.caption).foregroundStyle(engine.catalogInstallFailed ? Color.red : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let error = engine.removalError { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            if engine.catalogInstallation == nil && (engine.busy || engine.state == "error") { StatusLine(engine: engine) }
            HStack { Spacer(); Button("Done", action: onDone).keyboardShortcut(.defaultAction) }
        }.padding(24).frame(width: 640, height: 580)
            .onAppear { tab = .initial; selectedFamily = nil; artifactChoices = [:]; catalogQuantization = .all }
            .onChange(of: tab) { leaveFamily() }
            .onChange(of: quantizationTabs) {
                guard !engine.catalogLoading, engine.catalog?.status == .ready,
                      engine.catalog?.familyID == selectedFamily?.id else { return }
                catalogQuantization = catalogQuantization.resolved(for: catalogItems)
            }
            .onDisappear { engine.cancelModelCatalog() }
            .alert(item: $removalConfirmation) { confirmation in
                Alert(title: Text(confirmation.action.title + "?"), message: Text(confirmation.message),
                      primaryButton: .destructive(Text(confirmation.action.title)) {
                          engine.removeModel(id: confirmation.id, expectedAction: confirmation.action)
                      }, secondaryButton: .cancel())
            }
    }

    private func emptyState(_ selection: ModelsSheetTab) -> some View {
        VStack(spacing: 7) {
            Text(selection.emptyTitle).font(.headline)
            Text(selection.emptyMessage).font(.callout).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, minHeight: 300)
    }

    private func noSearchResults(_ title: String) -> some View {
        VStack(spacing: 10) {
            Text(title).font(.headline)
            Button("Clear search") { queryBinding.wrappedValue = "" }
        }.frame(maxWidth: .infinity, minHeight: 220)
    }

    private func modelGroup(_ group: ModelsInventory.Group) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if group.models.count > 1 {
                HStack(spacing: 8) {
                    Text(group.name).font(.caption.weight(.semibold)).lineLimit(1)
                    if let family = group.family { Text(family).font(.caption).foregroundStyle(.secondary) }
                    Spacer(minLength: 0)
                }.padding(.vertical, 6)
            }
            ForEach(group.models) { model in
                savedModelRow(model, family: group.family)
                if model.id != group.models.last?.id { Divider() }
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 4).frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary, lineWidth: 1))
    }

    private func savedModelRow(_ model: ModelsInventory.Model, family: String?) -> some View {
        HStack(spacing: 10) {
            Text(model.name).font(.callout.weight(.medium)).lineLimit(1).truncationMode(.middle)
                .frame(minWidth: 100, maxWidth: .infinity, alignment: .leading)
            Text(model.quantization ?? "—").font(.caption).foregroundStyle(.secondary)
                .lineLimit(1).minimumScaleFactor(0.8).frame(width: 68, alignment: .leading)
            Text(model.weightBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "—")
                .font(.caption).foregroundStyle(.secondary).frame(width: 76, alignment: .leading)
            modelRepositoryLink(model).frame(width: 28, alignment: .leading)
            Group {
                if model.id == engine.modelControlID {
                    Button("Stop") { engine.stopModel(id: model.id) }
                        .disabled(!engine.canStopModel(id: model.id)).accessibilityLabel("Stop " + model.name)
                } else {
                    Button("Run") { onRun(model.id) }
                        .disabled(engine.busy).accessibilityLabel("Run " + model.name)
                }
            }.controlSize(.small).frame(width: 44)
            removalButton(model).frame(width: 22)
        }.frame(minHeight: 32)
            .help(savedModelHelp(model, family: family))
            .accessibilityElement(children: .contain).accessibilityLabel(model.name)
    }

    private func savedModelHelp(_ model: ModelsInventory.Model, family: String?) -> String {
        var details = [model.name]
        if let family = family { details.append(family) }
        details.append(contentsOf: model.detailLabels.filter { !$0.hasPrefix("Version ") })
        if let revision = model.sourceRevision { details.append("Version " + revision) }
        return details.joined(separator: "\n")
    }

    @ViewBuilder
    private func modelRepositoryLink(_ model: ModelsInventory.Model) -> some View {
        if let repositoryURL = model.repositoryURL {
            Link("link", destination: repositoryURL).font(.caption).fixedSize().help(repositoryURL.absoluteString)
        }
    }

    private func removalButton(_ model: ModelsInventory.Model) -> some View {
        Group {
            if let removal = model.removal {
                Button {
                    if let saved = engine.models.first(where: { $0.id == model.id }) {
                        removalConfirmation = ModelRemovalConfirmation(model: saved, details: model.detailLabels,
                                                                       isRunning: model.id == engine.loadedID)
                    }
                } label: { Image(systemName: "trash").font(.callout).frame(width: 20, height: 22) }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
                    .disabled(engine.busy || removal.action == .unavailable)
                    .help(removal.action == .unavailable ? removal.message : removal.label)
                    .accessibilityLabel(removal.label + " " + model.name)
            }
        }
    }

    private func leaveFamily() {
        selectedFamily = nil; artifactChoices = [:]; catalogQuantization = .all
        engine.cancelModelCatalog()
    }

    private func selectedMode(_ item: FamilyModelCatalog.Item) -> ModelArtifactMode {
        if let choice = artifactChoices[item.id], item.artifactModes.contains(choice) { return choice }
        return item.defaultArtifactMode
    }

    @ViewBuilder
    private func catalogContent(_ family: ModelsInventory.Family) -> some View {
        if engine.catalogLoading {
            VStack(spacing: 10) { ProgressView(); Text("Finding models…").foregroundStyle(.secondary) }
                .frame(maxWidth: .infinity, minHeight: 220)
        } else if let catalog = engine.catalog, catalog.familyID == family.id, catalog.status == .ready {
            let groups = CatalogBrowserInventory(items: catalog.items, query: catalogQuery, sort: catalogSort,
                                                 quantization: catalogQuantization).groups
            if catalog.items.isEmpty {
                VStack(spacing: 10) {
                    Text("No models listed yet").font(.headline)
                    Button("Retry") { engine.requestModelCatalog(familyID: family.id) }
                }.frame(maxWidth: .infinity, minHeight: 220)
            } else if groups.isEmpty { noSearchResults("No matching models") }
            else {
                LazyVStack(spacing: 12) {
                    ForEach(groups) { group in catalogGroup(group) }
                    if catalog.hasMore { Text("More models may be available on Hugging Face.").font(.caption).foregroundStyle(.secondary) }
                }
            }
        } else {
            VStack(spacing: 10) {
                Text(engine.catalog?.status == .offline ? "You’re offline" : "Couldn’t load models").font(.headline)
                if let message = engine.catalogError { Text(message).font(.callout).foregroundStyle(.secondary) }
                Button("Retry") { engine.requestModelCatalog(familyID: family.id) }
            }.frame(maxWidth: .infinity, minHeight: 220)
        }
    }

    private func catalogGroup(_ group: CatalogBrowserInventory.Group) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(group.name).font(.headline)
                Text(group.publisher).font(.caption).foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }.padding(.bottom, 10)
            ForEach(group.items) { item in
                if item.id != group.items.first?.id { Divider() }
                catalogRow(item)
            }
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary, lineWidth: 1))
    }

    private func catalogRow(_ item: FamilyModelCatalog.Item) -> some View {
        HStack(spacing: 12) {
            Text(item.sizeLabel ?? "Unknown").font(.callout.weight(.medium))
                .frame(width: 70, alignment: .leading)
            Text(item.quantization).font(.callout).frame(width: 48, alignment: .leading)
            Text(item.weightBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "—")
                .font(.caption).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
            if let repositoryURL = URL(string: "https://huggingface.co/" + item.repo) {
                Link("link", destination: repositoryURL).font(.caption).fixedSize().help(repositoryURL.absoluteString)
            }
            Spacer(minLength: 0)
            if item.artifactModes.count > 1 {
                Picker("Install", selection: Binding(get: { selectedMode(item) }, set: { artifactChoices[item.id] = $0 })) {
                    ForEach(item.artifactModes, id: \.self) { Text($0.title).tag($0) }
                }.labelsHidden().frame(width: 145).controlSize(.small).disabled(engine.busy)
                    .accessibilityLabel("Version for " + item.name)
            }
            if engine.catalogItemIsDownloaded(item, artifactMode: selectedMode(item)) {
                Text("Downloaded").font(.caption).foregroundStyle(.secondary).frame(width: 88)
            } else {
                Button("Download") { engine.installCatalogItem(id: item.id, artifactMode: selectedMode(item)) }
                    .controlSize(.small).frame(width: 88).disabled(engine.busy || !item.downloadable)
                    .accessibilityLabel("Download " + item.name + " " + item.quantization)
            }
        }.padding(.vertical, 9).help(item.repo + "\nVersion " + item.revision)
            .accessibilityElement(children: .contain)
            .accessibilityLabel(item.name + ", version " + item.revision.prefix(7))
    }
}

struct InstallSupportView: View {
    let report: ModelSupportReport
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(report.message).font(.callout.weight(.medium))
            ForEach(report.details, id: \.self) { detail in Text(detail).font(.caption) }
            if !report.alternativeModes.isEmpty {
                Text("Other modes to check: \(report.alternativeModes.map(\.title).joined(separator: ", ")).")
                    .font(.caption)
            }
        }
        .foregroundStyle(report.status == .unsupported ? Color.orange : .secondary)
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
    }
}

struct InstallView: View {
    @ObservedObject var engine: EngineController
    let manage: () -> Void
    @State private var kind = "huggingface"
    @State private var source = ""
    @State private var visionConversion = false
    var body: some View {
        ScrollView { form }
            .frame(width: 470, height: 650)
            .onAppear { engine.resetInstallProgress(kind: kind) }
            .onChange(of: source) { if !engine.busy { engine.resetInstallProgress(kind: kind) } }
            .onChange(of: visionConversion) { if !engine.busy { engine.resetInstallProgress(kind: kind) } }
    }

    private var form: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("Install a model").font(.title2.weight(.semibold))
                Spacer()
                Button("Manage Models", action: manage).controlSize(.small)
            }
            Picker("Source", selection: $kind) {
                Text("Hugging Face").tag("huggingface")
                Text("Local Folder").tag("local")
            }.pickerStyle(.segmented).onChange(of: kind) { source = ""; engine.resetInstallProgress(kind: kind) }.disabled(engine.busy)
            if kind == "local" {
                Text("Choose a converted Stark model or a supported Safetensors checkpoint folder. Converted models stay in the selected folder; keep that folder available.")
                    .font(.callout).foregroundStyle(.secondary)
                HStack {
                    Text(source.isEmpty ? "No folder selected" : source).lineLimit(2).truncationMode(.middle)
                        .font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Choose Folder…") {
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.allowsMultipleSelection = false
                        panel.prompt = "Choose Model"
                        if panel.runModal() == .OK { source = panel.url?.path ?? "" }
                    }.disabled(engine.busy)
                }
            } else {
                TextField("Repository name or full Hugging Face URL", text: $source).textFieldStyle(.roundedBorder).disabled(engine.busy)
                Text("Downloads and converts a supported model. Gated repositories use your existing Hugging Face sign-in.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 5) {
                Toggle("Enable image input when converting", isOn: $visionConversion).disabled(engine.busy)
                Text("Image conversion supports compatible dense Qwen3.5 BF16 models and merged fine-tunes. Vision installs check support on this Mac and stop the running model.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Models stay on this Mac. Conversion may take several minutes.")
                .font(.caption).foregroundStyle(.secondary)
            if let report = engine.installSupportReport { InstallSupportView(report: report) }
            VStack(spacing: 14) {
                InstallStageView(title: "Download", phase: "download", progress: engine.downloadProgress)
                InstallStageView(title: "Conversion", phase: "conversion", progress: engine.conversionProgress)
            }
            .padding(14)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
            if engine.state == "installing" || engine.state == "error" { StatusLine(engine: engine) }
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            Spacer(minLength: 0)
            HStack {
                if engine.state == "installing" { Button("Cancel Installation", action: engine.stop) }
                Spacer()
                Button("Install") {
                    engine.install(source: source.trimmingCharacters(in: .whitespacesAndNewlines), kind: kind,
                                   artifactMode: visionConversion ? "vision-language" : nil)
                }
                    .buttonStyle(.borderedProminent).disabled(source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || engine.busy)
                    .keyboardShortcut(.defaultAction)
            }
        }.padding(24)
    }
}

struct ChatBubble: View {
    let message: ChatMessage
    var body: some View {
        HStack {
            if message.isUser { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 6) {
                Text(message.isUser ? "You" : "Stark").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if let image = message.image { AttachmentPreview(image: image) }
                Text(message.text).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(message.isUser ? Color.accentColor.opacity(0.12) : Color.primary.opacity(0.045),
                        in: RoundedRectangle(cornerRadius: 12))
            if !message.isUser { Spacer(minLength: 40) }
        }.frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
    }
}

struct AttachmentPreview: View {
    let image: ImageAttachment
    var body: some View {
        HStack(spacing: 10) {
            if let thumbnail = image.thumbnail {
                Image(nsImage: thumbnail).resizable().scaledToFit().frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 6)).accessibilityLabel("Attached image preview")
            } else {
                Image(systemName: "photo").frame(width: 72, height: 72).accessibilityLabel("Attached image")
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(image.mediaType == "image/png" ? "PNG image" : "JPEG image").font(.callout)
                Text(ByteCountFormatter.string(fromByteCount: Int64(image.bytes.count), countStyle: .file))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

struct AdvancedOptionsView: View {
    @ObservedObject var engine: EngineController
    @Environment(\.dismiss) private var dismiss
    @State private var useModelSystemPrompt: Bool
    @State private var systemPrompt: String
    @State private var context: String
    @State private var output: String
    @State private var thinking: Bool

    init(engine: EngineController) {
        self.engine = engine
        _useModelSystemPrompt = State(initialValue: engine.options.useModelSystemPrompt)
        _systemPrompt = State(initialValue: engine.options.systemPrompt)
        _context = State(initialValue: String(engine.options.contextTokens))
        _output = State(initialValue: String(engine.options.maxNewTokens))
        _thinking = State(initialValue: engine.options.thinking)
    }

    private var validation: String? {
        guard let count = Int(context), (2...engine.contextLimit).contains(count) else {
            return "Context size must be between 2 and \(engine.contextLimit) tokens."
        }
        guard let maximum = Int(output), (1...engine.outputLimit).contains(maximum) else {
            return "Output limit must be between 1 and \(engine.outputLimit) tokens."
        }
        return maximum < count ? nil : "Output limit must be smaller than context size."
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Advanced Options").font(.title2.weight(.semibold))
            Text("Applying changes starts a new chat with the model still loaded.")
                .font(.callout).foregroundStyle(.secondary)
            Toggle("Use model's default system prompt", isOn: $useModelSystemPrompt)
            if !useModelSystemPrompt {
                Text("System prompt").font(.headline)
                TextEditor(text: $systemPrompt).font(.body).frame(height: 100)
                    .padding(5).overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                    .accessibilityLabel("System prompt")
                Text("Leave empty to omit the system message.").font(.caption).foregroundStyle(.secondary)
            }
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                GridRow {
                    Text("Context size (tokens)")
                    TextField("Context size", text: $context).textFieldStyle(.roundedBorder)
                        .accessibilityLabel("Context size in tokens")
                }
                GridRow {
                    Text("Output limit (tokens)")
                    TextField("Output limit", text: $output).textFieldStyle(.roundedBorder)
                        .accessibilityLabel("Output limit in tokens")
                }
            }
            Text("Context includes the conversation and room for the full reply budget.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Thinking mode", isOn: $thinking)
            Text("Enables reasoning when the model's chat template supports it.")
                .font(.caption).foregroundStyle(.secondary)
            if let validation = validation { Text(validation).font(.caption).foregroundStyle(.red) }
            if let error = engine.errorMessage { Text(error).font(.caption).foregroundStyle(.red) }
            Text("Settings stay in memory for this app session. Switching models may reduce token limits.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Restore Defaults") {
                    useModelSystemPrompt = true; systemPrompt = ""; thinking = false
                    let count = min(8192, engine.contextLimit)
                    context = String(count)
                    output = String(min(512, engine.outputLimit, count - 1))
                }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Apply and New Chat") {
                    guard let count = Int(context), let maximum = Int(output) else { return }
                    var options = ChatOptions()
                    options.useModelSystemPrompt = useModelSystemPrompt
                    options.systemPrompt = systemPrompt
                    options.contextTokens = count; options.maxNewTokens = maximum; options.thinking = thinking
                    if engine.configure(options) { dismiss() }
                }.buttonStyle(.borderedProminent).disabled(validation != nil || engine.busy || engine.state != "ready")
            }
        }.padding(24).frame(width: 510)
    }
}

struct PlaygroundView: View {
    @ObservedObject var engine: EngineController
    let connection: () -> Void
    let manage: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Stark").font(.title3.weight(.semibold))
                Spacer()
                Button("Connection Details…", action: connection)
                Picker("Mode", selection: Binding(get: { engine.mode }, set: engine.changeMode)) {
                    ForEach(PlaygroundMode.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.segmented).frame(width: 180).disabled(engine.busy)
                    .accessibilityLabel("Chat or Decision mode")
            }
            StatusLine(engine: engine)
            HStack {
                Text(engine.activeName).font(.callout.weight(.medium)).lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 0)
                Button("Manage Models", action: manage).controlSize(.small).fixedSize()
                if !engine.loadedID.isEmpty || engine.busy {
                    Button("Stop", action: engine.stop).disabled(engine.state == "stopping")
                }
            }
            if let error = engine.errorMessage {
                Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            if engine.mode == .chat { ChatContentView(engine: engine) }
            else { DecisionView(engine: engine) }
        }.padding(20).frame(minWidth: 500, minHeight: 600)
            .onChange(of: engine.selectedID) { engine.clearDecision() }
    }
}

struct DecisionView: View {
    @ObservedObject var engine: EngineController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Text to evaluate").font(.callout)
                    TextEditor(text: $engine.decisionText).font(.body).frame(minHeight: 100, maxHeight: 150)
                        .padding(5).overlay(RoundedRectangle(cornerRadius: 7).stroke(.quaternary))
                        .accessibilityLabel("Text to evaluate")
                }
                VStack(alignment: .leading, spacing: 7) {
                    Text("Question").font(.callout)
                    TextField("Which answer fits this text?", text: $engine.decisionQuestion)
                        .textFieldStyle(.roundedBorder).accessibilityLabel("Decision question")
                }
                VStack(alignment: .leading, spacing: 7) {
                    Text("Answers").font(.callout)
                    TextField("Billing, Support, Sales", text: $engine.decisionAnswers)
                        .textFieldStyle(.roundedBorder).accessibilityLabel("Decision answers")
                    Text("Separate 2–16 answers with commas.").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Text("Inputs and results stay in memory").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Run decision", action: engine.decide)
                        .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                        .disabled(!engine.modelReady || engine.decisionValidation != nil)
                }
                if let result = engine.decisionResult {
                    Divider()
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Result").font(.caption).foregroundStyle(.secondary)
                        Text(result.choice).font(.title3.weight(.medium)).textSelection(.enabled)
                        ForEach(result.answers.indices, id: \.self) { index in
                            HStack(alignment: .firstTextBaseline) {
                                Text(result.answers[index]).fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: 12)
                                Text(result.probabilities[index], format: .percent.precision(.fractionLength(1)))
                                    .monospacedDigit()
                            }.font(.callout)
                        }
                        Text("Shares across these answers; calibration is unverified.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.accessibilityElement(children: .contain)
                }
            }.padding(.vertical, 2)
        }.disabled(engine.busy)
            .onChange(of: engine.decisionText) { engine.clearDecision() }
            .onChange(of: engine.decisionQuestion) { engine.clearDecision() }
            .onChange(of: engine.decisionAnswers) { engine.clearDecision() }
    }
}

struct ChatContentView: View {
    @ObservedObject var engine: EngineController
    @FocusState private var composerFocused: Bool
    @State private var advancedOptions = false
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Button("Advanced Options…") { advancedOptions = true }
                    .disabled(!engine.modelReady)
                    .help("Run a model to adjust its chat options.")
                Spacer()
                Button("New Chat") { engine.reset(); composerFocused = true }
                    .disabled(engine.busy || !engine.hasConversation || engine.loadedID != engine.selectedID)
            }
            SpeedView(engine: engine)
            ScrollViewReader { scroll in
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        if engine.messages.isEmpty {
                            VStack(spacing: 10) {
                                Image(systemName: "bubble.left.and.bubble.right").font(.system(size: 28)).foregroundStyle(.secondary)
                                Text("Try a conversation").font(.headline)
                                Text("Run a model in Manage Models, then send a message.")
                                    .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                            }.frame(maxWidth: .infinity).padding(.vertical, 36)
                        }
                        ForEach(engine.messages) { ChatBubble(message: $0) }
                        if engine.state == "generating" {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text(engine.imageProgress.isEmpty ? "Stark is thinking…" : engine.imageProgress)
                                    .font(.callout).foregroundStyle(.secondary)
                                Button("Cancel", action: engine.stop)
                            }.padding(.horizontal, 4)
                        }
                        Color.clear.frame(height: 1).id("latest")
                    }.padding(14)
                }
                .onChange(of: engine.messages.count) { withAnimation(.easeOut(duration: 0.15)) { scroll.scrollTo("latest", anchor: .bottom) } }
                .onChange(of: engine.state) { if engine.state == "generating" { scroll.scrollTo("latest", anchor: .bottom) } }
            }.frame(maxHeight: .infinity).background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 12))
            if ["length", "context"].contains(engine.finishReason) {
                Text("Reply reached the token limit. Clear the conversation to start fresh.").font(.caption).foregroundStyle(.orange)
            }
            if engine.supportsImages {
                Text(engine.imageVerificationNote).font(.caption).foregroundStyle(.secondary)
            }
            if let image = engine.draftImage {
                HStack {
                    AttachmentPreview(image: image)
                    Spacer()
                    Button("Remove Image", action: engine.removeImage).disabled(engine.busy)
                }.padding(10).background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
            } else if engine.conversationImage != nil {
                Text("Image included with each turn. Start a new chat to change it.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if engine.attachmentLoading {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("Loading image…").font(.caption).foregroundStyle(.secondary)
                    Button("Cancel") { engine.removeImage() }
                }
            }
            ZStack(alignment: .topLeading) {
                if engine.draft.isEmpty { Text("Message the model…").font(.body).foregroundStyle(.tertiary).padding(.horizontal, 9).padding(.top, 4).allowsHitTesting(false) }
                TextEditor(text: $engine.draft).font(.body).scrollContentBackground(.hidden).frame(height: 70).padding(4)
                    .focused($composerFocused).disabled(engine.busy).accessibilityLabel("Message")
            }.overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            HStack {
                Button("Attach Image…", action: engine.selectImage)
                    .disabled(!engine.canAttachImage)
                    .help(engine.supportsImages ? "Attach one PNG or JPEG image, up to 4 MiB." : "Run a qualified vision model to attach an image.")
                Text("Conversation stays in memory").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Send") { engine.generate(); composerFocused = true }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                    .disabled(!engine.canSendMessage)
            }
        }
            .onDrop(of: [UTType.fileURL, .png, .jpeg], isTargeted: nil, perform: engine.acceptImageProviders)
            .onPasteCommand(of: [.png, .jpeg]) { _ = engine.acceptImageProviders($0) }
            .sheet(isPresented: $advancedOptions) { AdvancedOptionsView(engine: engine) }
            .onAppear { composerFocused = true }
            .onChange(of: engine.state) { if engine.state == "ready" { composerFocused = true } }
    }
}

private final class MenuPanel: NSPanel {
    var onEscape: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    override func cancelOperation(_ sender: Any?) { onEscape?() }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let engine = EngineController()
    private let updater = AppUpdateController()
    private var status: NSStatusItem!
    private let menuPanel = MenuPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
    private var localMenuMonitor: Any?
    private var globalMenuMonitor: Any?
    private var installWindow: NSWindow?
    private var modelsWindow: NSWindow?
    private var promptWindow: NSWindow?
    private var connectionWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.accessory)
        let menu = NSMenu()
        let applicationItem = NSMenuItem()
        let applicationMenu = NSMenu()
        let showMenu = NSMenuItem(title: "Show Stark Menu", action: #selector(toggle), keyEquivalent: "")
        showMenu.target = self
        applicationMenu.addItem(showMenu)
        let installMenu = NSMenuItem(title: "Install Model…", action: #selector(showInstall), keyEquivalent: "i")
        installMenu.target = self
        applicationMenu.addItem(installMenu)
        let modelsMenu = NSMenuItem(title: "Manage Models", action: #selector(showModels), keyEquivalent: "")
        modelsMenu.target = self
        applicationMenu.addItem(modelsMenu)
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(withTitle: "Quit Stark", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        applicationItem.submenu = applicationMenu
        menu.addItem(applicationItem)
        let editItem = NSMenuItem()
        let edit = NSMenu(title: "Edit")
        for (title, action, key) in [("Undo", "undo:", "z"), ("Cut", "cut:", "x"),
                                      ("Copy", "copy:", "c"), ("Paste", "paste:", "v"),
                                      ("Select All", "selectAll:", "a")] {
            edit.addItem(withTitle: title, action: Selector(action), keyEquivalent: key)
        }
        editItem.submenu = edit; menu.addItem(editItem)
        NSApplication.shared.mainMenu = menu
        status = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let url = Bundle.main.url(forResource: "MenuBarTemplate", withExtension: "png"), let icon = NSImage(contentsOf: url) {
            let ratio = icon.size.height / icon.size.width
            icon.size = NSSize(width: 24, height: 24 * ratio); icon.isTemplate = true
            status.button?.image = icon
        } else { status.button?.title = "Stark" }
        status.button?.toolTip = "Stark"
        status.button?.setAccessibilityLabel("Stark menu bar")
        status.button?.target = self; status.button?.action = #selector(toggle)
        status.button?.sendAction(on: [.leftMouseUp, .rightMouseUp])
        menuPanel.level = .popUpMenu
        menuPanel.isOpaque = false
        menuPanel.backgroundColor = .clear
        menuPanel.hasShadow = true
        menuPanel.isReleasedWhenClosed = false
        menuPanel.collectionBehavior = [.transient, .moveToActiveSpace, .fullScreenAuxiliary]
        menuPanel.onEscape = { [weak self] in self?.dismissMenu() }
        menuPanel.contentViewController = NSHostingController(rootView: MenuView(
            engine: engine, updater: updater, install: { [weak self] in self?.showInstall() },
            manage: { [weak self] in self?.showModels() },
            console: { [weak self] in self?.showPrompt() },
            connection: { [weak self] in self?.showConnection() })
            .onChange(of: updater.message) { [weak self] _, _ in
                DispatchQueue.main.async { self?.resizeMenuForUpdate() }
            }
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13))
            .clipShape(RoundedRectangle(cornerRadius: 13)))
        engine.onInstalled = { [weak self] in self?.installWindow?.orderOut(nil); self?.showPrompt() }
        engine.start()
        updater.startAutomaticChecks()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
            guard let self = self else { return }
            if self.engine.models.isEmpty { self.showInstall() } else { self.toggle() }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { if engine.models.isEmpty { showInstall() } else { showPrompt() } }
        return true
    }

    @objc private func toggle() {
        if menuPanel.isVisible { dismissMenu(); return }
        updater.checkIfNeeded()
        guard let button = status.button, let statusWindow = button.window,
              let screen = statusWindow.screen else { return }
        let buttonFrame = statusWindow.convertToScreen(button.convert(button.bounds, to: nil))
        let safe = screen.visibleFrame.insetBy(dx: 8, dy: 8)
        let size = menuPanel.contentView?.fittingSize ?? NSSize(width: 350, height: 350)
        let width = min(size.width, safe.width)
        let height = min(size.height, safe.height)
        // Match MeeCP's visible status-item highlight edge (two points outside its window).
        let x = min(max(statusWindow.frame.minX - 2, safe.minX), safe.maxX - width)
        let top = min(max(buttonFrame.minY - 5, safe.minY + height), safe.maxY)
        menuPanel.setFrame(NSRect(x: x, y: top - height, width: width, height: height), display: true)
        menuPanel.makeKeyAndOrderFront(nil)
        updateMenuHighlight()
        localMenuMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown]) { [weak self] event in
            guard let self else { return event }
            if event.type == .keyDown {
                if event.keyCode == 53 { self.dismissMenu(); return nil }
            } else if event.window !== self.menuPanel && event.window !== self.status.button?.window {
                self.dismissMenu()
            }
            return event
        }
        globalMenuMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            self?.dismissMenu()
        }
    }

    private func updateMenuHighlight() {
        status.button?.highlight(menuPanel.isVisible)
        // AppKit clears the button's tracking highlight after dispatching its action.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.status.button?.highlight(self.menuPanel.isVisible)
        }
    }

    private func resizeMenuForUpdate() {
        guard menuPanel.isVisible, let screen = menuPanel.screen,
              let size = menuPanel.contentView?.fittingSize else { return }
        let safe = screen.visibleFrame.insetBy(dx: 8, dy: 8)
        let height = min(size.height, safe.height)
        let frame = menuPanel.frame
        menuPanel.setFrame(NSRect(x: frame.minX, y: max(safe.minY, frame.maxY - height),
                                 width: frame.width, height: height), display: true)
    }

    private func dismissMenu() {
        menuPanel.orderOut(nil)
        if let monitor = localMenuMonitor { NSEvent.removeMonitor(monitor); localMenuMonitor = nil }
        if let monitor = globalMenuMonitor { NSEvent.removeMonitor(monitor); globalMenuMonitor = nil }
        updateMenuHighlight()
    }

    @objc private func showInstall() {
        dismissMenu()
        if installWindow == nil {
            installWindow = window(title: "Stark — Install Model", view: InstallView(engine: engine, manage: { [weak self] in self?.showModels() }), resizable: false)
        }
        present(installWindow)
    }

    @objc private func showModels() {
        dismissMenu()
        engine.requestModelFamilies()
        if modelsWindow?.isVisible != true {
            let view = ModelsView(engine: engine,
                                  onDone: { [weak self] in self?.modelsWindow?.close() },
                                  onRun: { [weak self] identifier in self?.runSavedModel(identifier) })
            if let modelsWindow = modelsWindow {
                modelsWindow.contentViewController = NSHostingController(rootView: view)
            } else {
                modelsWindow = window(title: "Stark — Manage Models", view: view, resizable: false)
                modelsWindow?.delegate = self
            }
        }
        present(modelsWindow)
    }

    private func runSavedModel(_ identifier: String) {
        engine.runModel(id: identifier)
    }

    func windowWillClose(_ notification: Notification) {
        guard let closing = notification.object as? NSWindow, closing === modelsWindow else { return }
        engine.cancelModelCatalog()
    }

    private func showPrompt() {
        dismissMenu()
        if promptWindow == nil {
            promptWindow = window(title: "Stark — Playground",
                                  view: PlaygroundView(engine: engine,
                                                       connection: { [weak self] in self?.showConnection() },
                                                       manage: { [weak self] in self?.showModels() }), resizable: true)
        }
        present(promptWindow)
    }

    private func showConnection() {
        dismissMenu()
        if connectionWindow == nil {
            connectionWindow = window(title: "Stark — Connection Details", view: ConnectionDetailsView(engine: engine), resizable: false)
        } else {
            // Reset the inspected model so each opening defaults to the running model.
            connectionWindow?.contentViewController = NSHostingController(rootView: ConnectionDetailsView(engine: engine))
        }
        present(connectionWindow)
    }

    private func window<V: View>(title: String, view: V, resizable: Bool) -> NSWindow {
        let result = NSWindow(contentViewController: NSHostingController(rootView: view))
        result.title = title
        result.styleMask = resizable ? [.titled, .closable, .miniaturizable, .resizable] : [.titled, .closable]
        result.isReleasedWhenClosed = false
        result.center()
        return result
    }

    private func present(_ window: NSWindow?) {
        NSApplication.shared.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationWillTerminate(_ notification: Notification) {
        updater.stopAutomaticChecks()
        engine.shutdown()
    }
}

@main
struct StarkApplication {
    @MainActor
    static func main() {
        let application = NSApplication.shared
        let delegate = AppDelegate()
        application.delegate = delegate
        application.run()
        withExtendedLifetime(delegate) {}
    }
}
