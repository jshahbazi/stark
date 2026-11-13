import Foundation
import SwiftUI

enum StarkInterfaceMode: String, CaseIterable, Identifiable {
    case beginner, advanced
    var id: String { rawValue }
    var title: String { self == .beginner ? "Beginner" : "Advanced" }
}

final class StarkInterfacePreferences: ObservableObject {
    @Published private(set) var mode: StarkInterfaceMode
    private let defaults: UserDefaults
    private var initialModeResolved: Bool
    static let key = "starkInterfaceMode"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let saved = defaults.string(forKey: Self.key).flatMap(StarkInterfaceMode.init(rawValue:))
        mode = saved ?? .advanced
        initialModeResolved = saved != nil
    }

    func select(_ mode: StarkInterfaceMode) {
        initialModeResolved = true
        self.mode = mode
        defaults.set(mode.rawValue, forKey: Self.key)
    }

    /// Wait for actual inventory before choosing a first-launch default.
    func resolveInitialMode(hasInstalledModels: Bool) {
        guard !initialModeResolved else { return }
        select(hasInstalledModels ? .advanced : .beginner)
    }
}

struct StarkInterfacePicker: View {
    @ObservedObject var preferences: StarkInterfacePreferences
    var selected: (StarkInterfaceMode) -> Void = { _ in }

    var body: some View {
        Picker("Interface", selection: Binding(get: { preferences.mode }, set: {
            preferences.select($0); selected($0)
        })) {
            ForEach(StarkInterfaceMode.allCases) { Text($0.title).tag($0) }
        }
        .pickerStyle(.segmented).labelsHidden().frame(width: 218)
        .accessibilityLabel("Beginner or Advanced interface")
    }
}
