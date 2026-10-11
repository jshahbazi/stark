# Inherited Mirai playground

This sample app was inherited from Uzu and retains Mirai's name, bundle identity,
artwork, and upstream links. Its Swift engine dependency now uses the local Stark
binding. The inherited Firebase configuration,
startup collection, crash upload script, and test crash control have been removed.

Build `crates/legacy/stark/bindings/swift/stark.xcframework` from source before
generating or building this sample:

```sh
cargo tools install swift
cargo tools build swift --targets macos
```

The playground has not been built or qualified as part of the Stark scoring
acceptance. See the root README for the supported fork scope and validation.
