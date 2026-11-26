# 20261126-HY4DESKTOP-01 · Hy4-flash catalog install and packaged generation

| Field | Value |
| --- | --- |
| Attempt ID / registered date | 20261126-HY4DESKTOP-01 / 2026-11-26 |
| Scope | Compatibility and install-to-generation lifecycle on Apple Silicon |
| Predecessor | [Native Q3_K record](2026-11-25-hy4-native-q3-k.md) and [flexible context verification](2026-11-25-hy4-context-ladder.md) |
| Implementation | [PR #205](https://github.com/jshahbazi/stark/pull/205) |
| Artifact | AngelSlim/Hy4-flash-GGUF `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`, two Q3_K shards and the BF16 projector; Tencent metadata `eccd3123eec73f2040f4c9ed0de1883593470510` |
| Runtime | Locked Rust 1.98.1 Metal worker with `hy4` and the packaged Python desktop backend; M5 Max with 128 GB RAM, macOS 26.7.2 |

The source-only path required a manually prepared directory and a feature-enabled
worker. The catalog path should install exactly the same artifact, retaining the
projector, NextN and load-time context sizing. It uses native download staging so
an interrupted install can resume before the model is published to the library.
The managed download is 156,415,403,072 bytes. A derived NextN expert sidecar is
built on the first load.

## Acceptance checks

- Open the exact pinned catalog row in a fresh library and install it from the
  bundled desktop backend. Reuse the two already downloaded shards through the
  Hub's completed-file resume path, fetch the projector and Tencent sidecars
  from their pinned revisions, and check all three GGUF digests before publish.
- Start the installed artifact with the ordinary desktop load settings and the
  bundled worker. Check the live context capacity, NextN and image readiness.
- Complete text requests at 2,048 and 4,096 context settings in the same loaded
  worker, then submit a synthetic image through private attachment staging.
- Keep benchmark timings and numerical evidence in the existing native records.
  This smoke test verifies the app's lifecycle and configuration.
