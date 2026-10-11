import Foundation

#if os(iOS)
import UIKit

final class AppDelegate: UIResponder, UIApplicationDelegate {}
#elseif os(macOS)
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {}
#endif
