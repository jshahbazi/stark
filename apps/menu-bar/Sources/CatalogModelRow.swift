import SwiftUI

struct CatalogModelRow: View {
    let item: FamilyModelCatalog.Item
    let downloaded: Bool
    let busy: Bool
    let showsRepositoryLink: Bool
    var score: (value: Double, task: String)? = nil
    var exceedsMemory = false
    let download: () -> Void

    private var parameterLabel: String? {
        if let label = item.sizeLabel { return label }
        guard let count = item.parameterCount, count > 0 else { return nil }
        return count.formatted(.number.notation(.compactName).precision(.fractionLength(0...1)))
    }

    /// States whose model the score belongs to; an export is never presented as benchmarked itself.
    static func scoreExplanation(_ reference: CatalogReferenceScore, score: (value: Double, task: String)) -> String {
        let value = score.value.formatted(.number.precision(.fractionLength(0...1)))
        let subject = reference.relationship == .benchmarkedModel ? "this model" : reference.modelID
        var text = "whatmodel.sh \(score.task) score \(value) for \(subject), from the \(reference.captured) snapshot."
        if reference.relationship != .benchmarkedModel {
            text += "\nThis quantized version was not benchmarked itself, and quantization can change quality."
        }
        return text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    metadata
                    Spacer(minLength: 0)
                    controls
                }
                VStack(alignment: .leading, spacing: 8) {
                    ViewThatFits(in: .horizontal) {
                        metadata
                        VStack(alignment: .leading, spacing: 6) {
                            quantizationBadge
                            sizeMetadata
                        }
                    }
                    HStack { Spacer(minLength: 0); controls }
                }
            }
            if let reason = item.unavailableReason {
                Text(reason).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if exceedsMemory {
                Text("Likely too large for this Mac's memory").font(.caption).foregroundStyle(.orange)
            }
        }
        .help(item.repo + "\nVersion " + item.revision)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(item.name + ", version " + item.revision.prefix(7))
    }

    private var metadata: some View {
        HStack(spacing: 10) {
            quantizationBadge
            sizeMetadata
        }.fixedSize(horizontal: true, vertical: false)
    }

    private var quantizationBadge: some View {
        Text(item.quantization).font(.caption.weight(.medium)).lineLimit(1).truncationMode(.middle)
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 5))
            .help("Quantization: " + item.quantization)
    }

    private var sizeMetadata: some View {
        HStack(spacing: 10) {
            if let label = parameterLabel {
                Text(label).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    .help("Model parameters: " + label).accessibilityLabel(label + " parameters")
            }
            if let bytes = item.weightBytes {
                Text(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                    .help("Weight file size").accessibilityLabel("Weight files: " + ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))
            }
            if let score, let reference = item.referenceScore {
                let value = score.value.formatted(.number.precision(.fractionLength(0...1)))
                Link(destination: reference.sourceURL) {
                    Text("Score " + value).font(.caption.monospacedDigit().weight(.medium)).lineLimit(1)
                }
                .help(Self.scoreExplanation(reference, score: score))
                .accessibilityLabel("whatmodel.sh \(score.task) score \(value)")
            }
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            if showsRepositoryLink, let url = URL(string: "https://huggingface.co/" + item.repo) {
                Link(destination: url) { Image(systemName: "arrow.up.right.square") }
                    .font(.caption).help(item.repo)
                    .accessibilityLabel("View " + item.repo + " on Hugging Face")
            }
            Text(item.installationMode.title).font(.caption).foregroundStyle(.secondary)
            Group {
                if downloaded {
                    Label("Downloaded", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.secondary)
                } else if !item.downloadable {
                    Text("Unavailable").font(.caption).foregroundStyle(.secondary)
                } else {
                    Button("Download", action: download).controlSize(.small)
                        .disabled(busy)
                        .accessibilityLabel("Download " + item.name + " " + item.quantization)
                }
            }.frame(minWidth: 88, alignment: .trailing)
        }.fixedSize()
    }
}
