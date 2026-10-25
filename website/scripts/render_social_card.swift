// Renders the 1200x630 social preview card used by Open Graph and Twitter
// cards from the original Stark logo. The logo file itself is never modified.
//
//   xcrun swift website/scripts/render_social_card.swift \
//     website/public/assets/stark-engine.jpg website/public/assets/stark-social-card.jpg
//
// Colours and type mirror the site's hero: paper background, charcoal text,
// the framed logo card with its offset block shadow, Helvetica Neue headline.

import AppKit
import Foundation

func fail(_ message: String) -> Never {
  FileHandle.standardError.write("render_social_card: \(message)\n".data(using: .utf8)!)
  exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
  fail("usage: render_social_card.swift <logo.jpg> <output.jpg>")
}
let logoURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])
guard let logo = NSImage(contentsOf: logoURL) else {
  fail("could not read logo at \(logoURL.path)")
}

let width = 1200
let height = 630
guard let bitmap = NSBitmapImageRep(
  bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
  bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
  colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
) else {
  fail("could not allocate bitmap")
}
bitmap.size = NSSize(width: width, height: height)
guard let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
  fail("could not create graphics context")
}
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.imageInterpolation = .high
context.shouldAntialias = true

func rgb(_ hex: UInt32) -> NSColor {
  NSColor(
    deviceRed: CGFloat((hex >> 16) & 0xff) / 255,
    green: CGFloat((hex >> 8) & 0xff) / 255,
    blue: CGFloat(hex & 0xff) / 255,
    alpha: 1)
}

let paper = rgb(0xf3f2eb)
let ink = rgb(0x171b19)
let sage = rgb(0x747d65)
let body = rgb(0x51594d)
let muted = rgb(0x63675f)
let cardFill = rgb(0xe4e5dc)
let cardBorder = rgb(0xc4c8ba)
let cardShadow = rgb(0xdcded2)
let accent = rgb(0xa47a20)

let canvas = NSRect(x: 0, y: 0, width: width, height: height)
paper.setFill()
canvas.fill()

// Framed logo card on the right, as in the hero.
let cardSide: CGFloat = 470
let cardX = CGFloat(width) - 80 - cardSide
let cardY = (CGFloat(height) - cardSide) / 2
cardShadow.setFill()
NSRect(x: cardX + 10, y: cardY - 12, width: cardSide, height: cardSide).fill()
cardFill.setFill()
NSRect(x: cardX, y: cardY, width: cardSide, height: cardSide).fill()
cardBorder.setStroke()
let frame = NSBezierPath(rect: NSRect(x: cardX + 0.5, y: cardY + 0.5, width: cardSide - 1, height: cardSide - 1))
frame.lineWidth = 1
frame.stroke()
let inset: CGFloat = 16
logo.draw(
  in: NSRect(x: cardX + inset, y: cardY + inset, width: cardSide - 2 * inset, height: cardSide - 2 * inset),
  from: .zero, operation: .multiply, fraction: 1)

// Text column on the left. Coordinates below are measured from the top edge.
let columnX: CGFloat = 80
let columnWidth = cardX - 40 - columnX

func font(_ name: String, _ size: CGFloat, fallbackWeight: NSFont.Weight) -> NSFont {
  NSFont(name: name, size: size) ?? NSFont.systemFont(ofSize: size, weight: fallbackWeight)
}

func attributed(_ text: String, font: NSFont, color: NSColor, tracking: CGFloat) -> NSAttributedString {
  NSAttributedString(string: text, attributes: [
    .font: font, .foregroundColor: color, .kern: tracking,
  ])
}

@discardableResult
func draw(_ text: NSAttributedString, x: CGFloat, top: CGFloat) -> NSSize {
  let size = text.size()
  text.draw(at: NSPoint(x: x, y: CGFloat(height) - top - size.height))
  return size
}

// Eyebrow with the hero's status dot.
let eyebrowFont = NSFont.monospacedSystemFont(ofSize: 15, weight: .regular)
let eyebrow = attributed("RUST + METAL  /  BUILT FOR APPLE SILICON", font: eyebrowFont, color: ink, tracking: 1.4)
let eyebrowTop: CGFloat = 128
accent.setFill()
NSBezierPath(ovalIn: NSRect(x: columnX, y: CGFloat(height) - eyebrowTop - 15, width: 11, height: 11)).fill()
draw(eyebrow, x: columnX + 24, top: eyebrowTop)

// Headline: two lines, sized to fit the column, "Mac." in the hero's sage.
var headlineSize: CGFloat = 92
var line1: NSAttributedString
var line2: NSMutableAttributedString
repeat {
  let headlineFont = font("HelveticaNeue-Medium", headlineSize, fallbackWeight: .medium)
  let tracking = -0.045 * headlineSize
  line1 = attributed("Run local AI", font: headlineFont, color: ink, tracking: tracking)
  line2 = NSMutableAttributedString(attributedString: attributed("on your ", font: headlineFont, color: ink, tracking: tracking))
  line2.append(attributed("Mac.", font: headlineFont, color: sage, tracking: tracking))
  headlineSize -= 2
} while max(line1.size().width, line2.size().width) > columnWidth && headlineSize > 40
let headlineTop: CGFloat = 176
let lineHeight = line1.size().height * 0.96
draw(line1, x: columnX, top: headlineTop)
draw(line2, x: columnX, top: headlineTop + lineHeight)

// Description and address.
let descriptionFont = font("HelveticaNeue", 27, fallbackWeight: .regular)
let descriptionTop = headlineTop + 2 * lineHeight + 34
draw(attributed("Menu bar app, terminal chat and a local", font: descriptionFont, color: body, tracking: 0), x: columnX, top: descriptionTop)
draw(attributed("OpenAI-style API. Open-source Rust/Metal.", font: descriptionFont, color: body, tracking: 0), x: columnX, top: descriptionTop + 40)

let addressFont = NSFont.monospacedSystemFont(ofSize: 16, weight: .regular)
let address = attributed("starkengine.app", font: addressFont, color: muted, tracking: 0.8)
draw(address, x: columnX, top: CGFloat(height) - 80 - address.size().height)

NSGraphicsContext.restoreGraphicsState()

guard let jpeg = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.86]) else {
  fail("could not encode JPEG")
}
do {
  try jpeg.write(to: outputURL)
} catch {
  fail("could not write \(outputURL.path): \(error)")
}
print("Wrote \(width)x\(height) social card to \(outputURL.path) (\(jpeg.count) bytes)")
