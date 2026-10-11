// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Stark",
    platforms: [
        .iOS("26.4"),
        .macOS("26.4"),
        .macCatalyst("26.4"),
    ],
    products: [
        .library(name: "Stark", targets: ["Stark"]),
        .executable(name: "examples", targets: ["Examples"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.6.1")
    ],
    targets: [
        .binaryTarget(
            name: "stark",
            path: "stark.xcframework"
        ),
        .target(
            name: "Stark",
            dependencies: ["stark", "StarkMetalIOSimulatorStubs"],
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedLibrary("compression"),
                .linkedFramework("SystemConfiguration", .when(platforms: [.macOS])),
                .linkedFramework("Metal"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("AVFAudio", .when(platforms: [.iOS, .macCatalyst])),
            ]
        ),
        .target(
            name: "StarkMetalIOSimulatorStubs",
            path: "Sources/StarkMetalIOSimulatorStubs",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "Examples",
            dependencies: [
                "Stark",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "StarkTests",
            dependencies: ["Stark"],
        ),
    ]
)
