import AppKit

/// Adds a status badge while AppKit renders the original cat template.
final class MenuBarStatusBadge: NSView {
    var modelRunning = false {
        didSet { needsDisplay = true }
    }

    init(iconSize: NSSize) {
        super.init(frame: NSRect(origin: .zero, size: iconSize))
        setAccessibilityElement(false)
    }

    required init?(coder: NSCoder) { nil }

    // Let the status button receive clicks anywhere in the icon.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func draw(_ dirtyRect: NSRect) {
        let color = modelRunning
            ? NSColor(srgbRed: 0.27, green: 0.62, blue: 0.40, alpha: 1)
            : NSColor(srgbRed: 0.76, green: 0.30, blue: 0.28, alpha: 1)
        color.setFill()
        let circle = NSBezierPath(ovalIn: NSRect(x: bounds.maxX - 6, y: bounds.minY + 0.5,
                                                width: 5.5, height: 5.5))
        circle.fill()
        // Separate the badge from the silhouette in either menu bar appearance.
        NSColor.windowBackgroundColor.setStroke()
        circle.lineWidth = 0.75
        circle.stroke()
    }
}
