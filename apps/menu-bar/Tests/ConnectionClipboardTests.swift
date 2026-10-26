import AppKit

@main
struct ConnectionClipboardTests {
    static func main() {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        let info = ConnectionInfo(
            baseURL: "http://127.0.0.1:8000/v1", apiKey: "synthetic-key",
            modelID: "Local \"model\"\nname", runningModelID: "Local \"model\"\nname",
            selectedModelIsRunning: true, status: "ready", contextTokens: 8192, maxNewTokens: 512
        )
        var copies = 0
        for value in [info.apiKey, info.json!, info.curl] {
            precondition(ConnectionClipboard.write(value, sensitive: true, to: pasteboard))
            check(value, on: pasteboard, sensitive: true)
            copies += 1
        }
        // The first ordinary write also verifies that sensitive markers are cleared.
        for value in [info.baseURL, info.modelID] {
            precondition(ConnectionClipboard.write(value, to: pasteboard))
            check(value, on: pasteboard, sensitive: false)
            copies += 1
        }
        print("Clipboard checks passed: \(copies) copies")
    }

    private static func check(_ value: String, on pasteboard: NSPasteboard, sensitive: Bool) {
        precondition(pasteboard.string(forType: .string) == value)
        let items = pasteboard.pasteboardItems ?? []
        precondition(items.count == 1)
        precondition(items[0].string(forType: .string) == value)
        for markerName in ["org.nspasteboard.ConcealedType", "org.nspasteboard.TransientType"] {
            let marker = NSPasteboard.PasteboardType(markerName)
            precondition((pasteboard.types ?? []).contains(marker) == sensitive)
            precondition(items[0].types.contains(marker) == sensitive)
            precondition((items[0].data(forType: marker) != nil) == sensitive)
        }
    }
}
