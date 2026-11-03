import AppKit

/// Colors only the existing eye openings; AppKit still renders the cat template.
final class MenuBarStatusEyes: NSView {
    private let template: NSImage
    var modelRunning = false {
        didSet { needsDisplay = true }
    }

    init(template: NSImage) {
        self.template = template
        super.init(frame: NSRect(origin: .zero, size: template.size))
        setAccessibilityElement(false)
    }

    required init?(coder: NSCoder) { nil }

    // Let the status button receive clicks anywhere in the icon.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        context.beginTransparencyLayer(auxiliaryInfo: nil)
        let color = modelRunning
            ? NSColor(srgbRed: 0.27, green: 0.62, blue: 0.40, alpha: 1)
            : NSColor(srgbRed: 0.76, green: 0.30, blue: 0.28, alpha: 1)
        color.setFill()
        // These bounds sit inside the face around the original transparent eyes.
        // Removing the template alpha preserves the exact eyelids and pupils.
        for x in [0.25, 0.58] {
            NSRect(x: bounds.width * x, y: bounds.height * 0.36,
                   width: bounds.width * 0.16, height: bounds.height * 0.17).fill()
        }
        template.draw(in: bounds, from: .zero, operation: .destinationOut, fraction: 1)
        context.endTransparencyLayer()
    }
}
