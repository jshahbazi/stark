import AppKit

// Render the disk-image window background around Finder's own icons.
// Usage: xcrun swift RenderBackground.swift <output.tiff> <layout JSON>
// The layout comes from scripts/build_dmg.py, which also places the icons, so
// the panel, rules and drag arrow always line up with the Finder positions.

struct Point: Decodable {
    let x: CGFloat
    let y: CGFloat
}

struct Layout: Decodable {
    let width: CGFloat
    let height: CGFloat
    let iconSize: CGFloat
    let app: Point
    let applications: Point
    let installer: Point
}

func fail(_ message: String) -> Never {
    fputs("RenderBackground.swift: \(message)\n", stderr)
    exit(1)
}

guard CommandLine.arguments.count == 3 else {
    fail("usage: RenderBackground.swift <output TIFF> <layout JSON>")
}
guard let layout = try? JSONDecoder().decode(Layout.self, from: Data(CommandLine.arguments[2].utf8)) else {
    fail("invalid layout JSON")
}

func color(_ hex: UInt32) -> NSColor {
    NSColor(srgbRed: CGFloat((hex >> 16) & 255) / 255,
            green: CGFloat((hex >> 8) & 255) / 255,
            blue: CGFloat(hex & 255) / 255, alpha: 1)
}

// Brand tokens from docs/brand/README.md.
let canvas = color(0xFFFFFF)
let paper = color(0xF3F2EB)
let rule = color(0xD9DAD0)
let ink = color(0x171B19)
let muted = color(0x63675F)
let gold = color(0xA47A20)

func font(_ name: String, _ size: CGFloat, _ weight: NSFont.Weight) -> NSFont {
    NSFont(name: name, size: size) ?? NSFont.systemFont(ofSize: size, weight: weight)
}

func text(_ string: String, _ font: NSFont, _ color: NSColor, kern: CGFloat = 0) -> NSAttributedString {
    NSAttributedString(string: string, attributes: [.font: font, .foregroundColor: color, .kern: kern])
}

// Geometry, in points with a top-left origin.
let columnLeft = (layout.width - 436) / 2
let columnRight = layout.width - columnLeft
let panelTop: CGFloat = 116
let panelBottom: CGFloat = 336

func draw(scale: CGFloat) -> NSBitmapImageRep {
    let pixelsWide = Int(layout.width * scale)
    let pixelsHigh = Int(layout.height * scale)
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: pixelsWide, pixelsHigh: pixelsHigh,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    ), let base = NSGraphicsContext(bitmapImageRep: bitmap) else {
        fail("could not allocate a \(pixelsWide)x\(pixelsHigh) bitmap")
    }
    // Points and pixels differ only by scale, so Finder shows the 2x
    // representation on Retina displays at the same physical size.
    bitmap.size = NSSize(width: layout.width, height: layout.height)
    base.cgContext.translateBy(x: 0, y: CGFloat(pixelsHigh))
    base.cgContext.scaleBy(x: scale, y: -scale)
    let context = NSGraphicsContext(cgContext: base.cgContext, flipped: true)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    context.shouldAntialias = true

    canvas.setFill()
    NSRect(x: 0, y: 0, width: layout.width, height: layout.height).fill()
    paper.setFill()
    NSRect(x: columnLeft, y: panelTop, width: columnRight - columnLeft,
           height: panelBottom - panelTop).fill()

    // Hairline rules frame the drag panel and run to the window edges.
    let hairline = 1 / scale
    rule.setFill()
    for y in [panelTop, panelBottom] {
        NSRect(x: 0, y: y - hairline / 2, width: layout.width, height: hairline).fill()
    }
    for x in [columnLeft, columnRight] {
        NSRect(x: x - hairline / 2, y: 0, width: hairline, height: layout.height).fill()
    }

    // Two-line headline: the name in ink, the description in muted ink.
    let headline = font("HelveticaNeue-Medium", 27, .medium)
    let kern = -0.02 * headline.pointSize
    let first = NSMutableAttributedString(attributedString: text("Stark", headline, ink, kern: kern))
    first.append(text(", native inference", headline, muted, kern: kern))
    let second = text("built for Apple silicon", headline, muted, kern: kern)
    let lineHeight = first.size().height
    let headlineTop = (panelTop - 2 * lineHeight) / 2
    for (index, line) in [first, second].enumerated() {
        let size = line.size()
        line.draw(at: NSPoint(x: (layout.width - size.width) / 2,
                              y: headlineTop + CGFloat(index) * lineHeight))
    }

    // Drag arrow between the application and the Applications shortcut.
    let clearance = layout.iconSize / 2 + 24
    let start = NSPoint(x: layout.app.x + clearance, y: layout.app.y)
    let end = NSPoint(x: layout.applications.x - clearance, y: layout.applications.y)
    let arrow = NSBezierPath()
    arrow.move(to: start)
    arrow.line(to: end)
    let head: CGFloat = 9
    arrow.move(to: NSPoint(x: end.x - head, y: end.y - head))
    arrow.line(to: end)
    arrow.line(to: NSPoint(x: end.x - head, y: end.y + head))
    arrow.lineWidth = 2.5
    arrow.lineCapStyle = .round
    arrow.lineJoinStyle = .round
    gold.setStroke()
    arrow.stroke()

    // Upgrade guidance beside the Install Stark launcher.
    let textLeft = layout.installer.x + layout.iconSize / 2 + 26
    let paragraph = NSMutableParagraphStyle()
    paragraph.lineSpacing = 2
    let title = text("Updating Stark?", font("HelveticaNeue-Medium", 15, .medium), ink)
    let body = NSAttributedString(
        string: "Open Install Stark to replace the running app and relaunch it.",
        attributes: [.font: font("HelveticaNeue", 13, .regular), .foregroundColor: muted,
                     .paragraphStyle: paragraph])
    let bodyBounds = NSRect(x: textLeft, y: 0, width: columnRight - 18 - textLeft, height: 200)
    let bodyHeight = ceil(body.boundingRect(with: bodyBounds.size,
                                            options: [.usesLineFragmentOrigin, .usesFontLeading]).height)
    let titleHeight = title.size().height
    let blockTop = layout.installer.y - (titleHeight + 6 + bodyHeight) / 2
    title.draw(at: NSPoint(x: textLeft, y: blockTop))
    body.draw(with: NSRect(x: textLeft, y: blockTop + titleHeight + 6,
                           width: bodyBounds.width, height: bodyHeight),
              options: [.usesLineFragmentOrigin, .usesFontLeading])

    NSGraphicsContext.restoreGraphicsState()
    return bitmap
}

// A two-representation TIFF is the Finder convention for Retina backgrounds.
guard let tiff = NSBitmapImageRep.tiffRepresentationOfImageReps(
    in: [draw(scale: 1), draw(scale: 2)], using: .lzw, factor: 0
) else {
    fail("could not encode background TIFF")
}
do {
    try tiff.write(to: URL(fileURLWithPath: CommandLine.arguments[1]), options: .atomic)
} catch {
    fail("could not write background TIFF")
}
