From the Stark repository root, build the local Swift artifact:

```sh
cargo tools install swift
cargo tools build swift --targets macos
```

Add the checkout as a local Swift package:

```swift
dependencies: [
    .package(path: "/path/to/stark")
]
```

The package requires `crates/legacy/stark/bindings/swift/stark.xcframework`.
Published Stark Swift binaries are not configured.
