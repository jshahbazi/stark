# Cross-backend idle queue patch

The [candidate patch](candidate.patch) applies to
`640a6a7b6375c54e3cb87e12952bd4c7c77af464`. It records five Rust runtime files
and the shared benchmark subprocess cleanup repair. New generic harness and
regression-test files are checked in alongside the patch; their source hashes
are bound separately in the [aggregate receipt](../../../benchmarks/data/idle-wake-cross-backend-2026-11-08.json).

The [dated study](../../../benchmarks/2026-11-08-cross-backend-idle-wake.md)
records artifact identities, numerical and state controls, matched latency and
idle resource measurements. Baseline and candidate release workers use
separate build targets with Rust 1.98.1 and locked dependencies. The patch
changes scheduling and input handling; vendored shaders and precision are
unchanged. Earlier validation records retain their original source bindings.

Models, full fingerprint maps, worker binaries and native logs remain ignored.
Only source, aggregate measurements and hashes are published.
