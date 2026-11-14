import AppKit

enum ConnectionClipboard {
    private static let concealedType = NSPasteboard.PasteboardType("org.nspasteboard.ConcealedType")
    private static let transientType = NSPasteboard.PasteboardType("org.nspasteboard.TransientType")

    static func write(_ value: String, sensitive: Bool = false, to pasteboard: NSPasteboard = .general) -> Bool {
        let item = NSPasteboardItem()
        guard item.setString(value, forType: .string) else { return false }
        if sensitive {
            guard item.setData(Data(), forType: concealedType),
                  item.setData(Data(), forType: transientType) else { return false }
        }
        // Publish the string and its privacy markers together as one complete item.
        pasteboard.clearContents()
        return pasteboard.writeObjects([item])
    }
}

/// Owns only the unchanged Pi key copy, never its contents. A normal app quit
/// clears it immediately; expiry requires the process and pasteboard service.
final class ConnectionClipboardLease {
    typealias Scheduler = (TimeInterval, @escaping () -> Void) -> Void
    private let pasteboard: NSPasteboard
    private let schedule: Scheduler
    private var ownedChange: Int?

    init(pasteboard: NSPasteboard = .general,
         schedule: @escaping Scheduler = { delay, action in DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: action) }) {
        self.pasteboard = pasteboard; self.schedule = schedule
    }

    func write(_ value: String) -> Bool {
        guard ConnectionClipboard.write(value, sensitive: true, to: pasteboard) else { return false }
        let change = pasteboard.changeCount
        ownedChange = change
        schedule(120) { [weak self] in self?.clear(ifOwned: change) }
        return true
    }

    func clear() {
        if let change = ownedChange { clear(ifOwned: change) }
    }

    private func clear(ifOwned change: Int) {
        guard ownedChange == change else { return }
        ownedChange = nil
        if pasteboard.changeCount == change { pasteboard.clearContents() }
    }
}
