import SwiftUI

struct CatalogModelRow: View {
    let item: FamilyModelCatalog.Item
    @Binding var artifactMode: ModelArtifactMode
    let downloaded: Bool
    let busy: Bool
    let showsRepositoryLink: Bool
    let download: () -> Void

    private var parameterLabel: String? {
        if let label = item.sizeLabel { return label }
        guard let count = item.parameterCount, count > 0 else { return nil }
        return count.formatted(.number.notation(.compactName).precision(.fractionLength(0...1)))
    }

    var body: some View {
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
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            if showsRepositoryLink, let url = URL(string: "https://huggingface.co/" + item.repo) {
                Link(destination: url) { Image(systemName: "arrow.up.right.square") }
                    .font(.caption).help(item.repo)
                    .accessibilityLabel("View " + item.repo + " on Hugging Face")
            }
            if item.artifactModes.count > 1 {
                Picker("Install", selection: $artifactMode) {
                    ForEach(item.artifactModes, id: \.self) { Text($0.title).tag($0) }
                }.labelsHidden().frame(width: 145).controlSize(.small).disabled(busy)
                    .accessibilityLabel("Version for " + item.name)
            }
            Group {
                if downloaded {
                    Label("Downloaded", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Button("Download", action: download).controlSize(.small)
                        .disabled(busy || !item.downloadable)
                        .accessibilityLabel("Download " + item.name + " " + item.quantization)
                }
            }.frame(minWidth: 88, alignment: .trailing)
        }.fixedSize()
    }
}
