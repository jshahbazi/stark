import AppKit

// Render the installer package around the existing Stark artwork without altering it.
// Usage: xcrun swift RenderIcon.swift <Stark.jpg> <output.png>
guard CommandLine.arguments.count == 3,
      let logo = NSImage(contentsOfFile: CommandLine.arguments[1]),
      let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: 1024, pixelsHigh: 1024,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
      ),
      let context = NSGraphicsContext(bitmapImageRep: bitmap)
else {
    fputs("Usage: RenderIcon.swift <logo image> <output PNG>\n", stderr)
    exit(1)
}

func color(_ hex: UInt32, alpha: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: CGFloat((hex >> 16) & 255) / 255,
            green: CGFloat((hex >> 8) & 255) / 255,
            blue: CGFloat(hex & 255) / 255, alpha: alpha)
}

// Artwork coordinates use a top-left origin for readable package geometry.
func point(_ x: CGFloat, _ y: CGFloat) -> NSPoint { NSPoint(x: x, y: 1024 - y) }

func polygon(_ vertices: [(CGFloat, CGFloat)]) -> NSBezierPath {
    let path = NSBezierPath()
    path.move(to: point(vertices[0].0, vertices[0].1))
    for vertex in vertices.dropFirst() { path.line(to: point(vertex.0, vertex.1)) }
    path.close()
    path.lineJoinStyle = .round
    return path
}

func fill(_ vertices: [(CGFloat, CGFloat)], _ top: UInt32, _ bottom: UInt32,
          edge: UInt32 = 0x977350) {
    let path = polygon(vertices)
    NSGradient(starting: color(bottom), ending: color(top))!.draw(in: path, angle: 90)
    color(edge, alpha: 0.55).setStroke()
    path.lineWidth = 2
    path.stroke()
}

func line(_ vertices: [(CGFloat, CGFloat)], _ hex: UInt32, width: CGFloat = 2,
          alpha: CGFloat = 1) {
    let path = NSBezierPath()
    path.move(to: point(vertices[0].0, vertices[0].1))
    for vertex in vertices.dropFirst() { path.line(to: point(vertex.0, vertex.1)) }
    path.lineCapStyle = .round
    path.lineJoinStyle = .round
    path.lineWidth = width
    color(hex, alpha: alpha).setStroke()
    path.stroke()
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.imageInterpolation = .high
context.cgContext.setAllowsAntialiasing(true)
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: 1024, height: 1024).fill(using: .copy)

// A soft, transparent shadow gives the box weight without a background tile.
NSGraphicsContext.saveGraphicsState()
let shadow = NSShadow()
shadow.shadowColor = color(0x251C15, alpha: 0.30)
shadow.shadowBlurRadius = 27
shadow.shadowOffset = NSSize(width: 0, height: -14)
shadow.set()
color(0xAF8050).setFill()
polygon([(211, 424), (407, 302), (823, 362), (823, 761),
         (635, 878), (211, 798)]).fill()
NSGraphicsContext.restoreGraphicsState()

// Rear and left flaps sit behind the dark, visible interior.
fill([(407, 302), (823, 362), (763, 190), (366, 147)], 0xE8C99F, 0xC39969)
line([(373, 155), (758, 197), (816, 354)], 0xFFF0D6, alpha: 0.70)
fill([(211, 424), (407, 302), (267, 214), (91, 332)], 0xF1D4AC, 0xCCA373)
line([(100, 333), (266, 222), (397, 303)], 0xFFF1DB, alpha: 0.75)
fill([(211, 424), (407, 302), (823, 362), (635, 480)], 0x90653F, 0x5D3E29)
fill([(407, 302), (823, 362), (635, 480), (422, 436)], 0xAB7E50, 0x755031)

// The broad front face keeps the cat legible at Finder icon sizes.
fill([(211, 424), (635, 480), (635, 878), (211, 798)], 0xD6AE7B, 0xBA8D59)
fill([(635, 480), (823, 362), (823, 761), (635, 878)], 0xB68B59, 0x92683F)
line([(638, 488), (638, 869)], 0xF0CE9B, width: 3, alpha: 0.65)
line([(220, 790), (631, 869), (815, 756)], 0x785334, alpha: 0.40)

// A paper label follows the front plane. The logo itself is drawn unchanged.
NSGraphicsContext.saveGraphicsState()
let labelTransform = NSAffineTransform()
labelTransform.transformStruct = NSAffineTransformStruct(
    m11: 1, m12: -0.169, m21: 0, m22: 1, tX: 284, tY: 230
)
labelTransform.concat()
let labelShadow = NSShadow()
labelShadow.shadowColor = color(0x4D3522, alpha: 0.25)
labelShadow.shadowBlurRadius = 5
labelShadow.shadowOffset = NSSize(width: 0, height: -3)
labelShadow.set()
let paper = NSBezierPath(roundedRect: NSRect(x: -7, y: -7, width: 304, height: 265),
                         xRadius: 5, yRadius: 5)
color(0xF8F5EF).setFill()
paper.fill()
NSShadow().set()
logo.draw(in: NSRect(x: 0, y: 0, width: 290, height: 251),
          from: .zero, operation: .sourceOver, fraction: 1)
NSGraphicsContext.restoreGraphicsState()

// The two foreground flaps complete the familiar open-package silhouette.
fill([(211, 424), (635, 480), (561, 564), (128, 500)], 0xF3D9B3, 0xD0A674)
line([(139, 496), (557, 557), (627, 481)], 0xFFF1D9, width: 3, alpha: 0.80)
line([(218, 429), (628, 483)], 0xA67C4D, alpha: 0.55)
fill([(635, 480), (823, 362), (942, 416), (806, 549)], 0xE8C498, 0xC19561)
line([(646, 480), (823, 370), (932, 419)], 0xFFF0D4, width: 3, alpha: 0.75)

NSGraphicsContext.restoreGraphicsState()
guard let data = bitmap.representation(using: .png, properties: [:]) else {
    fputs("Could not encode installer icon PNG.\n", stderr)
    exit(1)
}
do {
    try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
} catch {
    fputs("Could not write installer icon PNG.\n", stderr)
    exit(1)
}
