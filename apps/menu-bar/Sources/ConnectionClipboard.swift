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
